"""Unit tests for the scene classifier: FlowFeatureExtractor + MobileNetV3TinyClassifier.

Tests use synthetic frame data only — no real UAV sequences required.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_tracker.types import BBox, FrameContext, TrackState


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_frame(h: int = 240, w: int = 360) -> np.ndarray:
    """Return a random BGR uint8 frame."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_ctx(
    frame: np.ndarray | None = None,
    frame_idx: int = 0,
    bbox: BBox | None = None,
) -> FrameContext:
    f = frame if frame is not None else _make_frame()
    prev_f = _make_frame()
    return FrameContext(
        frame=f,
        prev_frame=prev_f,
        frame_idx=frame_idx,
        bbox=bbox or BBox(x=50.0, y=40.0, w=60.0, h=40.0),
    )


def _make_state(confidence: float = 0.75) -> TrackState:
    return TrackState(
        bbox=BBox(x=50.0, y=40.0, w=60.0, h=40.0),
        confidence=confidence,
        status="locked",
    )


# --------------------------------------------------------------------------- #
# Test: registry                                                               #
# --------------------------------------------------------------------------- #


def test_classifier_registration() -> None:
    """'mobilenetv3_tiny' must be registered in SCENE_CLASSIFIERS."""
    from uav_tracker.registry import SCENE_CLASSIFIERS  # noqa: PLC0415

    # Ensure the plugin module has been imported (top-level import fires registration)
    import uav_tracker  # noqa: F401

    assert "mobilenetv3_tiny" in SCENE_CLASSIFIERS, (
        f"Expected 'mobilenetv3_tiny' in SCENE_CLASSIFIERS, got: {SCENE_CLASSIFIERS.names()}"
    )


# --------------------------------------------------------------------------- #
# Test: output shape & types                                                   #
# --------------------------------------------------------------------------- #


def test_classifier_output_shape() -> None:
    """classify() must return SceneClassification with probabilities.shape == (6,)."""
    from uav_tracker.ml.scene_classifier.cnn_classifier import MobileNetV3TinyClassifier
    from uav_tracker.types import SceneClassification

    clf = MobileNetV3TinyClassifier()
    ctx = _make_ctx(frame_idx=0)
    state = _make_state()

    result = clf.classify(ctx, state)

    assert isinstance(result, SceneClassification), (
        f"Expected SceneClassification, got {type(result)}"
    )
    assert result.probabilities.shape == (6,), (
        f"Expected (6,), got {result.probabilities.shape}"
    )
    assert result.probabilities.dtype == np.float32, (
        f"Expected float32, got {result.probabilities.dtype}"
    )


def test_confidence_in_range() -> None:
    """result.confidence must lie in [0.0, 1.0]."""
    from uav_tracker.ml.scene_classifier.cnn_classifier import MobileNetV3TinyClassifier

    clf = MobileNetV3TinyClassifier()
    ctx = _make_ctx(frame_idx=0)
    state = _make_state()

    result = clf.classify(ctx, state)
    assert 0.0 <= result.confidence <= 1.0, (
        f"confidence {result.confidence} outside [0, 1]"
    )


# --------------------------------------------------------------------------- #
# Test: uncertain gate                                                         #
# --------------------------------------------------------------------------- #


def test_uncertain_gate() -> None:
    """When all logits are equal, max_softmax ≈ 1/6 < 0.6 → scene_class == CLEAR."""
    import torch
    from unittest.mock import patch as mock_patch

    from uav_tracker.ml.scene_classifier.cnn_classifier import MobileNetV3TinyClassifier
    from uav_tracker.types import SceneClass

    clf = MobileNetV3TinyClassifier(confidence_threshold=0.60)
    ctx = _make_ctx(frame_idx=0)
    state = _make_state()

    # Uniform logits → softmax = 1/6 per class
    uniform_logits = torch.zeros(1, 6)

    # Patch the model's forward to return uniform logits
    def _fake_forward(patch, flow_feat):
        return uniform_logits

    # Load model first so _model is not None
    clf._load_model()
    assert clf._model is not None

    # Reset the counter so the first call triggers inference
    clf._frame_counter = 0
    clf._cached_result = None

    with mock_patch.object(clf._model, "forward", side_effect=_fake_forward):
        result = clf.classify(ctx, state)

    assert result.scene_class == SceneClass.CLEAR, (
        f"Expected CLEAR for uniform logits, got {result.scene_class}"
    )
    assert result.confidence == pytest.approx(1.0 / 6.0, abs=1e-4), (
        f"Expected confidence ≈ {1/6:.4f}, got {result.confidence:.4f}"
    )


# --------------------------------------------------------------------------- #
# Test: feature extractor                                                      #
# --------------------------------------------------------------------------- #


def test_feature_extractor_shape() -> None:
    """extract() must return shape (32,) float32."""
    from uav_tracker.ml.scene_classifier.feature_extractor import FlowFeatureExtractor

    extractor = FlowFeatureExtractor()
    ctx = _make_ctx(frame_idx=0)
    state = _make_state()

    feat = extractor.extract(ctx, state)

    assert feat.shape == (32,), f"Expected (32,), got {feat.shape}"
    assert feat.dtype == np.float32, f"Expected float32, got {feat.dtype}"


def test_feature_extractor_with_flow_cache() -> None:
    """extract() handles optical_flow_cache with 'flow' key correctly."""
    from uav_tracker.ml.scene_classifier.feature_extractor import FlowFeatureExtractor

    extractor = FlowFeatureExtractor()

    frame = _make_frame()
    h, w = frame.shape[:2]
    rng = np.random.default_rng(7)
    flow = rng.standard_normal((h, w, 2)).astype(np.float32)

    ctx = FrameContext(
        frame=frame,
        prev_frame=_make_frame(),
        frame_idx=1,
        bbox=BBox(x=50.0, y=40.0, w=60.0, h=40.0),
        optical_flow_cache={
            "flow": flow,
            "motion_entropy": 1.23,
            "circular_resultant": 0.85,
        },
    )
    state = _make_state()

    feat = extractor.extract(ctx, state)

    assert feat.shape == (32,)
    assert feat.dtype == np.float32
    # motion entropy should be captured at index 0
    assert feat[0] == pytest.approx(1.23, abs=1e-5)
    # circular resultant disorder at index 1 should be 1 - 0.85 = 0.15
    assert feat[1] == pytest.approx(0.15, abs=1e-5)


# --------------------------------------------------------------------------- #
# Test: reset idempotency                                                      #
# --------------------------------------------------------------------------- #


def test_reset_is_idempotent() -> None:
    """Calling reset() twice must not raise and must leave extractor in clean state."""
    from uav_tracker.ml.scene_classifier.feature_extractor import FlowFeatureExtractor

    extractor = FlowFeatureExtractor()

    # Run one extraction to accumulate state
    extractor.extract(_make_ctx(frame_idx=0), _make_state())

    # Double reset must be safe
    extractor.reset()
    extractor.reset()

    # After reset, the next extraction should return zeros for EMA/delta features
    feat = extractor.extract(_make_ctx(frame_idx=1), _make_state())
    assert feat.shape == (32,)
    # EMA features [18-21] should reflect first-frame values (no history)
    # Just ensure no crash and correct type
    assert feat.dtype == np.float32


def test_classifier_reset_is_idempotent() -> None:
    """Calling MobileNetV3TinyClassifier.reset() twice must not raise."""
    from uav_tracker.ml.scene_classifier.cnn_classifier import MobileNetV3TinyClassifier

    clf = MobileNetV3TinyClassifier()
    ctx = _make_ctx(frame_idx=0)
    state = _make_state()

    clf.classify(ctx, state)

    clf.reset()
    clf.reset()

    # Should still be usable after reset
    result = clf.classify(ctx, state)
    assert result.probabilities.shape == (6,)


# --------------------------------------------------------------------------- #
# Test: classify_interval caching                                              #
# --------------------------------------------------------------------------- #


def test_classify_interval_caching() -> None:
    """Non-classification frames should return cached result (same probabilities array)."""
    from uav_tracker.ml.scene_classifier.cnn_classifier import MobileNetV3TinyClassifier

    clf = MobileNetV3TinyClassifier(classify_interval=5)
    frame = _make_frame()
    state = _make_state()

    # Frame 0 — triggers classification
    ctx0 = _make_ctx(frame=frame, frame_idx=0)
    r0 = clf.classify(ctx0, state)

    # Frames 1–3 — should return cached probabilities
    for idx in range(1, 4):
        ctx_i = _make_ctx(frame=frame, frame_idx=idx)
        r_i = clf.classify(ctx_i, state)
        np.testing.assert_array_equal(r_i.probabilities, r0.probabilities)
        assert r_i.frame_idx == idx


# --------------------------------------------------------------------------- #
# Test: probabilities sum to 1                                                #
# --------------------------------------------------------------------------- #


def test_probabilities_sum_to_one() -> None:
    """Softmax output must sum to 1.0 within floating point tolerance."""
    from uav_tracker.ml.scene_classifier.cnn_classifier import MobileNetV3TinyClassifier

    clf = MobileNetV3TinyClassifier()
    ctx = _make_ctx(frame_idx=0)
    state = _make_state()

    result = clf.classify(ctx, state)
    assert result.probabilities.sum() == pytest.approx(1.0, abs=1e-5)
