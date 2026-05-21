"""Unit tests for saltr/src/salt_r/evidence.py — Phase 1C EvidenceExtractor."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from salt_r.evidence import (
    CandidateEvidence,
    EvidenceExtractor,
    EvidenceFrame,
    RecoveryContext,
    TemplateContext,
)
from salt_r.feature_schema import PRODUCTION_ZERO_FEATURE_INDICES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_features(value: float = 1.0) -> np.ndarray:
    """Return a 28-dim feature vector with all values set to *value*."""
    return np.ones(28, dtype=np.float32) * value


_BBOX: tuple[float, float, float, float] = (10.0, 20.0, 50.0, 30.0)


# ---------------------------------------------------------------------------
# 1. Frame 0 produces a valid EvidenceFrame with correct frame_idx
# ---------------------------------------------------------------------------

def test_frame0_valid_evidence_frame():
    ext = EvidenceExtractor()
    ef = ext.step(_dummy_features(), _BBOX)

    assert isinstance(ef, EvidenceFrame)
    assert ef.frame_idx == 0
    assert ef.bbox == _BBOX
    assert ef.base_features.shape == (28,)


# ---------------------------------------------------------------------------
# 2. Flow indices 22-27 are zeroed in base_features
# ---------------------------------------------------------------------------

def test_flow_indices_zeroed():
    ext = EvidenceExtractor()
    ef = ext.step(np.ones(28, dtype=np.float32), _BBOX)

    for idx in range(22, 28):
        assert ef.base_features[idx] == 0.0, (
            f"Flow index {idx} should be 0.0, got {ef.base_features[idx]}"
        )


# ---------------------------------------------------------------------------
# 3. Score-map features (0-8) are NOT zeroed; rolling/dynamics are computed
# ---------------------------------------------------------------------------

def test_score_map_features_preserved():
    """Indices 0-8 (score-map features) are passed through unchanged."""
    ext = EvidenceExtractor()
    ef = ext.step(np.ones(28, dtype=np.float32), _BBOX)

    for idx in range(9):
        assert ef.base_features[idx] == 1.0, (
            f"Score-map index {idx} should remain 1.0, got {ef.base_features[idx]}"
        )


# ---------------------------------------------------------------------------
# 4. reset() clears frame counter and history
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    ext = EvidenceExtractor()
    # Advance a few frames
    for _ in range(5):
        ext.step(_dummy_features(), _BBOX)

    ext.reset()
    ef = ext.step(_dummy_features(), _BBOX)

    assert ef.frame_idx == 0, "frame_idx should be 0 after reset"
    # history length inside the extractor should reflect only this one step
    assert len(ext._feature_history) == 1
    assert len(ext._bbox_history) == 1


# ---------------------------------------------------------------------------
# 5. Top-k candidates are parsed and serializable (.to_dict() works)
# ---------------------------------------------------------------------------

def _make_candidate(score: float, offset: float = 0.0) -> dict:
    return {
        "bbox": [20.0 + offset, 25.0, 40.0, 20.0],
        "score": score,
        "source": "score_map",
        "detector_score": None,
        "teacher_score": None,
    }


def test_top_k_candidates_parsed():
    ext = EvidenceExtractor(top_k_candidates=3)
    raw_candidates = [_make_candidate(s, i * 5) for i, s in enumerate([0.9, 0.7, 0.5, 0.3])]

    ef = ext.step(_dummy_features(), _BBOX, candidates=raw_candidates)

    # top_k=3, so only 3 candidates expected
    assert len(ef.candidates) == 3

    for i, cand in enumerate(ef.candidates):
        assert isinstance(cand, CandidateEvidence)
        assert cand.rank == i

    # Serialisability
    for cand in ef.candidates:
        d = cand.to_dict()
        assert isinstance(d, dict)
        assert "bbox" in d
        assert isinstance(d["bbox"], list)
        assert "score" in d
        assert "source" in d


def test_candidate_score_ratio_to_top():
    """score_ratio_to_top for the top candidate must be 1.0."""
    ext = EvidenceExtractor()
    raw_candidates = [_make_candidate(0.8), _make_candidate(0.4)]
    ef = ext.step(_dummy_features(), _BBOX, candidates=raw_candidates)

    assert ef.candidates[0].score_ratio_to_top == pytest.approx(1.0)
    assert ef.candidates[1].score_ratio_to_top == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 6. notify_template_updated resets last_update_age to 0 then increments
# ---------------------------------------------------------------------------

def test_notify_template_updated_resets_then_increments():
    ext = EvidenceExtractor()

    # Step 0: last_update_age starts at 0
    ef0 = ext.step(_dummy_features(), _BBOX)
    assert ef0.template_context.last_update_age == 0
    assert ef0.template_context.update_count == 0

    # Step 1: age increments to 1
    ef1 = ext.step(_dummy_features(), _BBOX)
    assert ef1.template_context.last_update_age == 1

    # Notify update: age resets to 0, count increments
    ext.notify_template_updated()

    # Step 2: frame sees age == 0 (just updated)
    ef2 = ext.step(_dummy_features(), _BBOX)
    assert ef2.template_context.last_update_age == 0
    assert ef2.template_context.update_count == 1

    # Step 3: age increments again
    ef3 = ext.step(_dummy_features(), _BBOX)
    assert ef3.template_context.last_update_age == 1
    assert ef3.template_context.update_count == 1


# ---------------------------------------------------------------------------
# 7. notify_reinit increments reinit count and tracks last_reinit_age
# ---------------------------------------------------------------------------

def test_notify_reinit_count_and_age():
    ext = EvidenceExtractor()

    # Before any reinit, last_reinit_age == -1
    ef0 = ext.step(_dummy_features(), _BBOX)
    assert ef0.recovery_context.last_reinit_age == -1
    assert ef0.recovery_context.total_reinit_count == 0

    # Reinit at frame boundary (after step 0)
    ext.notify_reinit()

    # Step 1: age == 0 (just reinited)
    ef1 = ext.step(_dummy_features(), _BBOX)
    assert ef1.recovery_context.last_reinit_age == 0
    assert ef1.recovery_context.total_reinit_count == 1

    # Step 2: age increments
    ef2 = ext.step(_dummy_features(), _BBOX)
    assert ef2.recovery_context.last_reinit_age == 1

    # Second reinit
    ext.notify_reinit()
    ef3 = ext.step(_dummy_features(), _BBOX)
    assert ef3.recovery_context.last_reinit_age == 0
    assert ef3.recovery_context.total_reinit_count == 2


# ---------------------------------------------------------------------------
# 8. Wrong feature dimension raises ValueError
# ---------------------------------------------------------------------------

def test_wrong_feature_dimension_raises():
    ext = EvidenceExtractor()
    bad_features = np.ones(27, dtype=np.float32)
    with pytest.raises(ValueError):
        ext.step(bad_features, _BBOX)


def test_wrong_feature_3d_raises():
    ext = EvidenceExtractor()
    bad_features = np.ones((2, 28), dtype=np.float32)
    # 2D array with shape (2, 28) — ndim==2, last dim==28, should pass validate
    # But a 3D array should fail:
    bad_3d = np.ones((1, 2, 28), dtype=np.float32)
    with pytest.raises(ValueError):
        ext.step(bad_3d, _BBOX)


# ---------------------------------------------------------------------------
# 9. Empty candidates list works fine
# ---------------------------------------------------------------------------

def test_empty_candidates_list():
    ext = EvidenceExtractor()
    ef = ext.step(_dummy_features(), _BBOX, candidates=[])
    assert ef.candidates == []


def test_none_candidates_works():
    ext = EvidenceExtractor()
    ef = ext.step(_dummy_features(), _BBOX, candidates=None)
    assert ef.candidates == []


# ---------------------------------------------------------------------------
# 10. Module has no TSA imports (AST check)
# ---------------------------------------------------------------------------

def test_no_tsa_imports_in_evidence_module():
    evidence_path = Path(__file__).parents[2] / "saltr" / "src" / "salt_r" / "evidence.py"
    source = evidence_path.read_text()

    # AST-level check: no import statement contains 'tsa', 'targetstate', or 'trackeraction'
    tree = ast.parse(source)
    forbidden = {"tsa", "targetstate", "trackeraction"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            node_str = ast.dump(node).lower()
            for term in forbidden:
                assert term not in node_str, (
                    f"Forbidden term '{term}' found in import statement: {ast.dump(node)}"
                )


# ---------------------------------------------------------------------------
# 11. BUG-1 fix: distance_to_prev_bbox differs from distance_to_tracker
#     on frame 2 (after the tracker bbox has moved)
# ---------------------------------------------------------------------------

def test_distance_to_prev_bbox_differs_after_move():
    """On frame 2, the tracker bbox moved — distance_to_prev_bbox must differ
    from distance_to_tracker for a stationary candidate."""
    ext = EvidenceExtractor()

    bbox_frame0: BBox = (10.0, 10.0, 50.0, 30.0)
    bbox_frame1: BBox = (60.0, 60.0, 50.0, 30.0)  # large move

    # Stationary candidate sitting at the frame-0 position center
    cand_cx = bbox_frame0[0] + bbox_frame0[2] / 2   # 35.0
    cand_cy = bbox_frame0[1] + bbox_frame0[3] / 2   # 25.0
    candidate = {
        "bbox": [cand_cx - 5, cand_cy - 5, 10.0, 10.0],
        "score": 0.9,
        "source": "score_map",
    }

    ext.step(_dummy_features(), bbox_frame0)   # frame 0
    ef1 = ext.step(_dummy_features(), bbox_frame1, candidates=[candidate])  # frame 1

    assert len(ef1.candidates) == 1
    c = ef1.candidates[0]

    # distance_to_tracker measures from bbox_frame1 center
    frame1_cx = bbox_frame1[0] + bbox_frame1[2] / 2   # 85.0
    frame1_cy = bbox_frame1[1] + bbox_frame1[3] / 2   # 75.0
    expected_dist_tracker = float(np.hypot(cand_cx - frame1_cx, cand_cy - frame1_cy))

    # distance_to_prev_bbox measures from bbox_frame0 center (nearly zero)
    expected_dist_prev = float(np.hypot(cand_cx - (bbox_frame0[0] + bbox_frame0[2] / 2),
                                         cand_cy - (bbox_frame0[1] + bbox_frame0[3] / 2)))

    assert c.distance_to_tracker == pytest.approx(expected_dist_tracker, abs=1e-4)
    assert c.distance_to_prev_bbox == pytest.approx(expected_dist_prev, abs=1e-4)
    assert c.distance_to_tracker != pytest.approx(c.distance_to_prev_bbox, abs=1.0), (
        "distance_to_prev_bbox should differ from distance_to_tracker after bbox move"
    )


# ---------------------------------------------------------------------------
# 12. BUG-2 fix: rolling features (index 9 = apce_ratio_5) are non-zero
#     after 5 frames with varying APCE values
# ---------------------------------------------------------------------------

def test_apce_ratio_5_nonzero_after_five_frames():
    """After 5+ frames with varying APCE, apce_ratio_5 (index 9) must be
    non-trivially filled (not zero, and differs from 1.0 when APCE varies)."""
    ext = EvidenceExtractor()

    # Feed frames with gradually increasing APCE (50, 60, 70, 80, 90, 200)
    apce_values = [50.0, 60.0, 70.0, 80.0, 90.0, 200.0]
    last_ef = None
    for apce in apce_values:
        feats = np.zeros(28, dtype=np.float32)
        feats[0] = apce          # apce_raw
        feats[1] = apce / 256.0  # apce_norm
        last_ef = ext.step(feats, _BBOX)

    # On the 6th frame (apce=200), the prior 5 frames had apce=[60,70,80,90,...200 not yet]
    # mean_apce_last_5 ≈ mean([60,70,80,90,90... no — prev 5 before current])
    # The ratio should be > 1.0 since 200 >> prior mean (~70)
    assert last_ef is not None
    ratio = last_ef.base_features[9]
    assert ratio != 0.0, f"apce_ratio_5 must not be 0, got {ratio}"
    assert ratio > 1.0, (
        f"apce_ratio_5 should be > 1.0 when current APCE (200) >> recent mean, got {ratio}"
    )


# ---------------------------------------------------------------------------
# 13. Indices 22-27 remain zero even after rolling feature computation
# ---------------------------------------------------------------------------

def test_flow_indices_still_zero_after_rolling_features():
    """Indices 22-27 (flow features) must remain zero after rolling feature
    computation is applied."""
    ext = EvidenceExtractor()

    # Run several frames so rolling history is populated
    for apce in [50.0, 100.0, 150.0, 200.0, 75.0, 130.0]:
        feats = np.ones(28, dtype=np.float32)
        feats[0] = apce
        ef = ext.step(feats, _BBOX)

    for idx in range(22, 28):
        assert ef.base_features[idx] == 0.0, (
            f"Flow index {idx} must be 0.0 after rolling features, got {ef.base_features[idx]}"
        )

