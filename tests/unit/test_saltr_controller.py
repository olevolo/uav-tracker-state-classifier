"""Unit tests for SALTRDController (saltr/src/salt_r/controller.py)."""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import numpy as np
import pytest

from salt_r.actions import (
    ComputeAction,
    RecoveryAction,
    SearchAction,
    TemplateAction,
    TrackerAction,
)
from salt_r.controller import SALTRDController, SALTRDDecision
from salt_r.evidence import (
    CandidateEvidence,
    EvidenceFrame,
    RecoveryContext,
    TemplateContext,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONTROLLER_SRC = Path(__file__).parent.parent.parent / "saltr" / "src" / "salt_r" / "controller.py"


def _make_features(value: float = 0.5, dim: int = 28) -> np.ndarray:
    """Return a valid 28-dim feature vector."""
    return np.full(dim, value, dtype=np.float32)


def _make_evidence(features: np.ndarray | None = None, candidates=None) -> EvidenceFrame:
    """Build a minimal EvidenceFrame."""
    if features is None:
        features = _make_features()
    return EvidenceFrame(
        frame_idx=0,
        bbox=(10.0, 10.0, 50.0, 50.0),
        base_features=features,
        score_map_stats={},
        candidates=candidates or [],
        template_context=TemplateContext(),
        recovery_context=RecoveryContext(),
    )


def _make_candidate(bbox=(20.0, 20.0, 30.0, 30.0), score=0.9) -> CandidateEvidence:
    return CandidateEvidence(
        bbox=bbox,
        score=score,
        rank=0,
        score_ratio_to_top=1.0,
        distance_to_tracker=5.0,
        distance_to_prev_bbox=5.0,
        size_ratio_to_tracker=1.0,
        source="score_map",
    )


def mock_model(features):
    return {
        "risk_probs": {"false_confirmed": 0.1},
        "action_logits": {
            "compute": {"full": 0.9, "prune_light": 0.1, "prune_medium": 0.0},
            "search": {"keep": 1.0, "expand": 0.0, "freeze": 0.0, "center_on_reinit_hint": 0.0},
            "template": {"keep_current": 1.0, "update": 0.0, "block_update": 0.0},
            "recovery": {"none": 0.0, "score_candidates": 0.0, "reinit": 1.0, "reject_reinit": 0.0},
        },
        "confidence": 0.8,
    }


# ---------------------------------------------------------------------------
# Test 1 — No TSA import (AST check)
# ---------------------------------------------------------------------------

def test_no_tsa_import():
    """controller.py must not import from TSA or tracker state modules."""
    src = CONTROLLER_SRC.read_text()
    tree = ast.parse(src)
    forbidden_modules = {"tsa", "target_state", "TargetState", "TSA"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                for part in node.module.split("."):
                    assert part.lower() not in {m.lower() for m in forbidden_modules}, (
                        f"Forbidden import found: {node.module}"
                    )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for part in alias.name.split("."):
                        assert part.lower() not in {m.lower() for m in forbidden_modules}, (
                            f"Forbidden import found: {alias.name}"
                        )


# ---------------------------------------------------------------------------
# Test 2 — No model → safe NOOP
# ---------------------------------------------------------------------------

def test_no_model_returns_safe_noop():
    ctrl = SALTRDController(policy_net=None)
    decision = ctrl.step(_make_evidence())
    assert decision.safety_fallback_applied is True
    assert decision.reason == "no_model_loaded"
    assert decision.action.compute == ComputeAction.FULL
    assert decision.action.recovery == RecoveryAction.NONE


# ---------------------------------------------------------------------------
# Test 3 — NaN features → safe NOOP
# ---------------------------------------------------------------------------

def test_nan_features_returns_safe_noop():
    ctrl = SALTRDController(policy_net=mock_model)
    features = _make_features()
    features[0] = float("nan")
    decision = ctrl.step(_make_evidence(features=features))
    assert decision.safety_fallback_applied is True
    assert "features_not_finite" in decision.reason


# ---------------------------------------------------------------------------
# Test 4 — Valid model output → correct action decoded
# ---------------------------------------------------------------------------

def test_valid_model_output_decoded():
    ctrl = SALTRDController(policy_net=mock_model)
    # Provide a candidate so REINIT can resolve
    candidate = _make_candidate()
    decision = ctrl.step(_make_evidence(candidates=[candidate]))
    assert decision.safety_fallback_applied is False
    assert decision.reason == "model_output"
    assert decision.action.compute == ComputeAction.FULL
    assert decision.action.search == SearchAction.KEEP
    assert decision.action.template == TemplateAction.KEEP_CURRENT
    assert decision.model_confidence == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Test 5 — REINIT with candidates → selected_candidate is set
# ---------------------------------------------------------------------------

def test_reinit_with_candidates_selects_candidate():
    ctrl = SALTRDController(policy_net=mock_model)
    candidate = _make_candidate(bbox=(100.0, 100.0, 40.0, 40.0), score=0.95)
    decision = ctrl.step(_make_evidence(candidates=[candidate]))
    assert decision.action.recovery == RecoveryAction.REINIT
    assert decision.selected_candidate is not None
    assert decision.selected_candidate.bbox == (100.0, 100.0, 40.0, 40.0)
    assert decision.action.bbox_hint == (100.0, 100.0, 40.0, 40.0)


# ---------------------------------------------------------------------------
# Test 6 — REINIT with no candidates → falls back to SCORE_CANDIDATES
# ---------------------------------------------------------------------------

def test_reinit_no_candidates_falls_back_to_score_candidates():
    ctrl = SALTRDController(policy_net=mock_model)
    # No candidates in evidence
    decision = ctrl.step(_make_evidence(candidates=[]))
    assert decision.action.recovery == RecoveryAction.SCORE_CANDIDATES
    assert decision.selected_candidate is None


# ---------------------------------------------------------------------------
# Test 7 — reset() works without error
# ---------------------------------------------------------------------------

def test_reset_works():
    ctrl = SALTRDController(policy_net=None)
    ctrl._frame_idx = 42
    ctrl.reset()
    assert ctrl._frame_idx == 0


# ---------------------------------------------------------------------------
# Test 8 — Schema mismatch (wrong feature dim) → safe NOOP
# ---------------------------------------------------------------------------

def test_wrong_feature_dim_returns_safe_noop():
    ctrl = SALTRDController(policy_net=mock_model)
    features = _make_features(dim=10)  # wrong dimension
    decision = ctrl.step(_make_evidence(features=features))
    assert decision.safety_fallback_applied is True
    assert "feature_shape_invalid" in decision.reason


# ---------------------------------------------------------------------------
# Test 9 — Model error → safe NOOP (not exception propagated)
# ---------------------------------------------------------------------------

def test_model_error_returns_safe_noop():
    def broken_model(features):
        raise RuntimeError("GPU out of memory")

    ctrl = SALTRDController(policy_net=broken_model)
    decision = ctrl.step(_make_evidence())
    assert decision.safety_fallback_applied is True
    assert "model_error" in decision.reason
    assert "GPU out of memory" in decision.reason


# ---------------------------------------------------------------------------
# Test 10 — Default action is ComputeAction.FULL / RecoveryAction.NONE
# ---------------------------------------------------------------------------

def test_default_action_fields():
    action = TrackerAction()
    assert action.compute == ComputeAction.FULL
    assert action.recovery == RecoveryAction.NONE
    assert action.search == SearchAction.KEEP
    assert action.template == TemplateAction.KEEP_CURRENT
    assert action.bbox_hint is None
