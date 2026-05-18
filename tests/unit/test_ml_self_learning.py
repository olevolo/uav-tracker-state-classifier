"""Unit tests for Phase 13 self-learning modules: CosineAppearanceMemory and OnlineLSTMMotionPredictor.

All tests use synthetic data only — no real UAV sequences or pre-trained weights needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_tracker.types import BBox, FrameContext, TrackState


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #


def _make_frame(h: int = 120, w: int = 160, seed: int = 0) -> np.ndarray:
    """Return a random uint8 BGR frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_ctx(
    frame: np.ndarray | None = None,
    frame_idx: int = 0,
    bbox: BBox | None = None,
    seed: int = 0,
) -> FrameContext:
    f = frame if frame is not None else _make_frame(seed=seed)
    return FrameContext(
        frame=f,
        prev_frame=_make_frame(seed=seed + 100),
        frame_idx=frame_idx,
        bbox=bbox or BBox(x=20.0, y=15.0, w=30.0, h=20.0),
    )


def _make_state(confidence: float = 0.8) -> TrackState:
    return TrackState(
        bbox=BBox(x=20.0, y=15.0, w=30.0, h=20.0),
        confidence=confidence,
        status="locked",
    )


def _make_bbox(seed: int = 0) -> BBox:
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 100, 2)
    w, h = rng.uniform(10, 40, 2)
    return BBox(x=float(x), y=float(y), w=float(w), h=float(h))


# --------------------------------------------------------------------------- #
# CosineAppearanceMemory — registration                                        #
# --------------------------------------------------------------------------- #


def test_cosine_memory_registration() -> None:
    """'cosine_memory' must appear in APPEARANCE_MEMORIES after package import."""
    import uav_tracker  # noqa: F401 — triggers _register_plugins()
    from uav_tracker.registry import APPEARANCE_MEMORIES

    assert "cosine_memory" in APPEARANCE_MEMORIES, (
        f"'cosine_memory' not found; registered: {APPEARANCE_MEMORIES.names()}"
    )


# --------------------------------------------------------------------------- #
# CosineAppearanceMemory — store increments count                             #
# --------------------------------------------------------------------------- #


def test_cosine_memory_store_increments_count() -> None:
    """After a single store call with sufficient confidence, len(templates) > 0."""
    from uav_tracker.ml.appearance_memory.cosine_memory import CosineAppearanceMemory

    mem = CosineAppearanceMemory(store_interval=1, min_confidence=0.5)
    ctx = _make_ctx(frame_idx=0)
    state = _make_state(confidence=0.9)

    mem.store(ctx, state)

    assert len(mem._templates) > 0, "Expected at least one template after store()"


# --------------------------------------------------------------------------- #
# CosineAppearanceMemory — max_templates cap                                  #
# --------------------------------------------------------------------------- #


def test_cosine_memory_max_templates() -> None:
    """Stored template count must never exceed max_templates."""
    from uav_tracker.ml.appearance_memory.cosine_memory import CosineAppearanceMemory

    max_t = 5
    mem = CosineAppearanceMemory(max_templates=max_t, store_interval=1, min_confidence=0.0)

    for i in range(max_t + 5):
        ctx = _make_ctx(frame_idx=i, seed=i)
        state = _make_state(confidence=1.0)
        mem.store(ctx, state)

    assert len(mem._templates) <= max_t, (
        f"Expected <= {max_t} templates, got {len(mem._templates)}"
    )


# --------------------------------------------------------------------------- #
# CosineAppearanceMemory — retrieve_best returns AppearanceTemplate list       #
# --------------------------------------------------------------------------- #


def test_cosine_memory_retrieve_best_shape() -> None:
    """retrieve_best(query, top_k=3) must return a list of AppearanceTemplate."""
    from uav_tracker.ml.appearance_memory.cosine_memory import CosineAppearanceMemory
    from uav_tracker.types import AppearanceTemplate

    mem = CosineAppearanceMemory(store_interval=1, min_confidence=0.0, embedding_dim=16)

    # Store 5 templates
    for i in range(5):
        ctx = _make_ctx(frame_idx=i, seed=i)
        state = _make_state(confidence=1.0)
        mem.store(ctx, state)

    query = np.random.default_rng(7).standard_normal(16).astype(np.float32)
    results = mem.retrieve_best(query, top_k=3)

    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert len(results) <= 3, f"Expected <= 3 results, got {len(results)}"
    for item in results:
        assert isinstance(item, AppearanceTemplate), (
            f"Expected AppearanceTemplate, got {type(item)}"
        )


# --------------------------------------------------------------------------- #
# CosineAppearanceMemory — drift with one template                             #
# --------------------------------------------------------------------------- #


def test_cosine_memory_drift_zero_with_one_template() -> None:
    """compute_drift() must return 0.0 when fewer than 2 templates are stored."""
    from uav_tracker.ml.appearance_memory.cosine_memory import CosineAppearanceMemory

    mem = CosineAppearanceMemory(store_interval=1, min_confidence=0.0)
    ctx = _make_ctx(frame_idx=0)
    state = _make_state(confidence=1.0)
    mem.store(ctx, state)

    # Exactly one template
    assert len(mem._templates) == 1
    assert mem.compute_drift() == pytest.approx(0.0), (
        f"Expected drift=0.0 with 1 template, got {mem.compute_drift()}"
    )


# --------------------------------------------------------------------------- #
# CosineAppearanceMemory — reset clears                                       #
# --------------------------------------------------------------------------- #


def test_cosine_memory_reset_clears() -> None:
    """After reset(), retrieve_best() must return an empty list."""
    from uav_tracker.ml.appearance_memory.cosine_memory import CosineAppearanceMemory

    mem = CosineAppearanceMemory(store_interval=1, min_confidence=0.0, embedding_dim=16)

    for i in range(4):
        ctx = _make_ctx(frame_idx=i, seed=i)
        state = _make_state(confidence=1.0)
        mem.store(ctx, state)

    assert len(mem._templates) > 0, "Precondition: some templates were stored"

    mem.reset()

    query = np.random.default_rng(1).standard_normal(16).astype(np.float32)
    results = mem.retrieve_best(query)
    assert results == [], f"Expected empty list after reset, got {results}"
    assert mem._frame_count == 0, "frame_count should be reset to 0"


# --------------------------------------------------------------------------- #
# CosineAppearanceMemory — drift range sanity                                 #
# --------------------------------------------------------------------------- #


def test_cosine_memory_drift_in_range() -> None:
    """compute_drift() must lie in [0, 1] after multiple stores."""
    from uav_tracker.ml.appearance_memory.cosine_memory import CosineAppearanceMemory

    mem = CosineAppearanceMemory(store_interval=1, min_confidence=0.0, embedding_dim=16)

    for i in range(6):
        ctx = _make_ctx(frame_idx=i, seed=i * 13)
        state = _make_state(confidence=1.0)
        mem.store(ctx, state)

    drift = mem.compute_drift()
    assert 0.0 <= drift <= 1.0, f"Drift {drift} outside [0, 1]"


# --------------------------------------------------------------------------- #
# OnlineLSTMMotionPredictor — registration                                    #
# --------------------------------------------------------------------------- #


def test_lstm_registration() -> None:
    """'lstm_online' must appear in MOTION_PREDICTORS after package import."""
    import uav_tracker  # noqa: F401
    from uav_tracker.registry import MOTION_PREDICTORS

    assert "lstm_online" in MOTION_PREDICTORS, (
        f"'lstm_online' not found; registered: {MOTION_PREDICTORS.names()}"
    )


# --------------------------------------------------------------------------- #
# OnlineLSTMMotionPredictor — predict_next returns BBox                        #
# --------------------------------------------------------------------------- #


def test_lstm_predict_returns_bbox() -> None:
    """predict_next() must return a BBox instance."""
    from uav_tracker.ml.motion_predictor.lstm_predictor import OnlineLSTMMotionPredictor

    pred = OnlineLSTMMotionPredictor(hidden_size=8, seq_len=5)
    history = [_make_bbox(seed=i) for i in range(5)]
    timestamps = list(range(5))

    result = pred.predict_next(history, timestamps)

    assert isinstance(result, BBox), f"Expected BBox, got {type(result)}"
    # Width and height should match the last known box
    assert result.w == pytest.approx(history[-1].w, rel=1e-5)
    assert result.h == pytest.approx(history[-1].h, rel=1e-5)


def test_lstm_predict_with_short_history() -> None:
    """predict_next() must not raise when history is shorter than seq_len (padding)."""
    from uav_tracker.ml.motion_predictor.lstm_predictor import OnlineLSTMMotionPredictor

    pred = OnlineLSTMMotionPredictor(hidden_size=8, seq_len=10)
    history = [_make_bbox(seed=0)]  # only 1 bbox, seq_len=10 → pad 9 times
    timestamps = [0]

    result = pred.predict_next(history, timestamps)
    assert isinstance(result, BBox)


# --------------------------------------------------------------------------- #
# OnlineLSTMMotionPredictor — update does not raise                           #
# --------------------------------------------------------------------------- #


def test_lstm_update_no_crash() -> None:
    """update(actual_bbox) must not raise even before the first predict_next call."""
    from uav_tracker.ml.motion_predictor.lstm_predictor import OnlineLSTMMotionPredictor

    pred = OnlineLSTMMotionPredictor(hidden_size=8, seq_len=5)

    # update() before predict_next() — should be a safe no-op
    pred.update(_make_bbox(seed=0))

    # Now do a full round: predict → update
    history = [_make_bbox(seed=i) for i in range(5)]
    pred.predict_next(history, list(range(5)))
    pred.update(_make_bbox(seed=99))  # must not raise


def test_lstm_update_changes_weights() -> None:
    """After update(), the model parameters must have changed (gradient applied)."""
    import torch
    from uav_tracker.ml.motion_predictor.lstm_predictor import OnlineLSTMMotionPredictor

    pred = OnlineLSTMMotionPredictor(hidden_size=8, seq_len=5, lr=0.1)
    history = [_make_bbox(seed=i) for i in range(5)]

    # Capture initial weights
    before = {
        name: param.clone().detach()
        for name, param in pred._net.named_parameters()
    }

    pred.predict_next(history, list(range(5)))
    pred.update(_make_bbox(seed=77))

    changed = any(
        not torch.equal(before[name], param.detach())
        for name, param in pred._net.named_parameters()
    )
    assert changed, "Expected model weights to change after update()"


# --------------------------------------------------------------------------- #
# OnlineLSTMMotionPredictor — reset clears state                              #
# --------------------------------------------------------------------------- #


def test_lstm_reset_clears_state() -> None:
    """After reset(), _hidden, _history, and _last_pred must all be cleared."""
    from uav_tracker.ml.motion_predictor.lstm_predictor import OnlineLSTMMotionPredictor

    pred = OnlineLSTMMotionPredictor(hidden_size=8, seq_len=5)
    history = [_make_bbox(seed=i) for i in range(5)]
    pred.predict_next(history, list(range(5)))

    pred.reset()

    assert pred._hidden is None, "Expected _hidden to be None after reset"
    assert len(pred._history) == 0, "Expected _history to be empty after reset"
    assert pred._last_pred is None, "Expected _last_pred to be None after reset"


def test_lstm_reset_is_idempotent() -> None:
    """Calling reset() twice must not raise."""
    from uav_tracker.ml.motion_predictor.lstm_predictor import OnlineLSTMMotionPredictor

    pred = OnlineLSTMMotionPredictor(hidden_size=8, seq_len=5)
    pred.reset()
    pred.reset()
    # Should still be usable
    result = pred.predict_next([_make_bbox(0)], [0])
    assert isinstance(result, BBox)
