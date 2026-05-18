"""Unit tests for TrackerConfidenceSignal (Phase 3)."""

from __future__ import annotations

import math

import pytest

from uav_tracker.types import BBox, FrameContext, TrackState


def _dummy_ctx(frame_idx: int = 0):
    import numpy as np

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    return FrameContext(
        frame=frame,
        prev_frame=None,
        frame_idx=frame_idx,
        bbox=BBox(16.0, 16.0, 32.0, 32.0),
    )


def _make_state(confidence: float) -> TrackState:
    return TrackState(
        bbox=BBox(16.0, 16.0, 32.0, 32.0),
        confidence=confidence,
        status="locked",
    )


@pytest.fixture()
def signal():
    from uav_tracker.signals.tracker_confidence import TrackerConfidenceSignal

    return TrackerConfidenceSignal()


# ---------------------------------------------------------------------------


def test_value_is_one_minus_confidence(signal) -> None:
    """value == 1 - confidence when state has a valid confidence."""
    ctx = _dummy_ctx()
    state = _make_state(0.8)
    report = signal.step(ctx, state)
    assert report.reliable is True
    assert abs(report.value - 0.2) < 1e-9


def test_value_zero_confidence(signal) -> None:
    """confidence == 0 → value == 1.0."""
    ctx = _dummy_ctx()
    state = _make_state(0.0)
    report = signal.step(ctx, state)
    assert report.reliable is True
    assert abs(report.value - 1.0) < 1e-9


def test_value_full_confidence(signal) -> None:
    """confidence == 1.0 → value == 0.0."""
    ctx = _dummy_ctx()
    state = _make_state(1.0)
    report = signal.step(ctx, state)
    assert report.reliable is True
    assert abs(report.value - 0.0) < 1e-9


def test_reliable_false_when_state_none(signal) -> None:
    """Returns reliable=False when last_track_state is None (first frame)."""
    ctx = _dummy_ctx()
    report = signal.step(ctx, None)
    assert report.reliable is False
    assert report.value == 0.0


def test_reliable_false_when_confidence_nan(signal) -> None:
    """Returns reliable=False when confidence is NaN."""
    ctx = _dummy_ctx()
    state = _make_state(float("nan"))
    report = signal.step(ctx, state)
    assert report.reliable is False
    assert report.value == 0.0


def test_name_and_range(signal) -> None:
    """Signal must expose correct name and range."""
    assert signal.name == "tracker_confidence"
    assert signal.range == (0.0, 1.0)


def test_reset_is_noop(signal) -> None:
    """reset() must not raise (stateless signal)."""
    signal.reset()
    signal.reset()


def test_registered_in_signals_registry() -> None:
    """tracker_confidence must be in the SIGNALS registry after package import."""
    import uav_tracker  # triggers _register_plugins

    assert "tracker_confidence" in uav_tracker.SIGNALS.names()
