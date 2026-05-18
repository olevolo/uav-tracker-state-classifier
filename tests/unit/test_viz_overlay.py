"""Unit tests for uav_tracker.viz.overlay.draw_frame_overlay."""

from __future__ import annotations

import numpy as np
import pytest

from uav_tracker.types import BBox, SignalReport
from uav_tracker.viz.overlay import draw_frame_overlay

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(h: int = 240, w: int = 320) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _dummy_bbox() -> BBox:
    return BBox(x=50.0, y=40.0, w=80.0, h=60.0)


def _dummy_signals(n: int = 1) -> dict[str, SignalReport]:
    names = ["confidence", "entropy", "flow_div", "apce", "circ"]
    return {names[i]: SignalReport(value=float(i) / max(n - 1, 1)) for i in range(n)}


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------


def test_import() -> None:
    from uav_tracker.viz.overlay import draw_frame_overlay  # noqa: F401

    assert callable(draw_frame_overlay)


# ---------------------------------------------------------------------------
# Basic return contract
# ---------------------------------------------------------------------------


def test_returns_same_shape_and_dtype() -> None:
    frame = _make_frame()
    result = draw_frame_overlay(
        frame=frame,
        bbox=_dummy_bbox(),
        tier=0,
        signals={},
        fps=25.0,
    )
    assert result.shape == frame.shape
    assert result.dtype == frame.dtype


def test_frame_not_mutated() -> None:
    """Input frame must not be modified in-place."""
    frame = _make_frame()
    original = frame.copy()
    draw_frame_overlay(frame=frame, bbox=_dummy_bbox(), tier=0, signals={}, fps=25.0)
    np.testing.assert_array_equal(frame, original)


def test_something_was_drawn() -> None:
    """Returned frame should differ from the original (overlay was rendered)."""
    frame = _make_frame()
    result = draw_frame_overlay(
        frame=frame,
        bbox=_dummy_bbox(),
        tier=0,
        signals={},
        fps=25.0,
    )
    assert not np.array_equal(result, frame), "Overlay drew nothing on the frame."


# ---------------------------------------------------------------------------
# bbox=None still renders badge
# ---------------------------------------------------------------------------


def test_no_bbox_still_draws_badge() -> None:
    frame = _make_frame()
    result = draw_frame_overlay(
        frame=frame,
        bbox=None,
        tier=1,
        signals={},
        fps=15.0,
    )
    assert result.shape == frame.shape
    # Badge region (top-left ~100x30 px) should differ from input.
    assert not np.array_equal(result[:30, :120], frame[:30, :120])


# ---------------------------------------------------------------------------
# All three tier values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", [0, 1, 2])
def test_each_tier_does_not_crash(tier: int) -> None:
    frame = _make_frame()
    result = draw_frame_overlay(
        frame=frame,
        bbox=_dummy_bbox(),
        tier=tier,
        signals={"conf": SignalReport(value=0.5)},
        fps=30.0,
    )
    assert result.shape == frame.shape


# ---------------------------------------------------------------------------
# Unknown tier (robustness)
# ---------------------------------------------------------------------------


def test_unknown_tier_uses_fallback_colour() -> None:
    frame = _make_frame()
    result = draw_frame_overlay(
        frame=frame,
        bbox=_dummy_bbox(),
        tier=99,
        signals={},
        fps=30.0,
    )
    assert result.shape == frame.shape


# ---------------------------------------------------------------------------
# Signal gauges
# ---------------------------------------------------------------------------


def test_three_signals_rendered() -> None:
    frame = _make_frame()
    signals = _dummy_signals(n=3)
    result = draw_frame_overlay(
        frame=frame,
        bbox=_dummy_bbox(),
        tier=0,
        signals=signals,
        fps=30.0,
    )
    # Bottom region should be different from input (gauges drawn).
    assert not np.array_equal(result[-80:, :200], frame[-80:, :200])


def test_signal_value_zero_and_one() -> None:
    """Extreme signal values (0.0 and 1.0) must not crash."""
    frame = _make_frame()
    signals = {
        "zero_sig": SignalReport(value=0.0),
        "one_sig": SignalReport(value=1.0),
    }
    result = draw_frame_overlay(
        frame=frame,
        bbox=None,
        tier=2,
        signals=signals,
        fps=5.0,
    )
    assert result.shape == frame.shape


# ---------------------------------------------------------------------------
# Ground-truth bbox
# ---------------------------------------------------------------------------


def test_gt_bbox_drawn() -> None:
    frame = _make_frame()
    bbox = BBox(x=50.0, y=40.0, w=80.0, h=60.0)
    gt = BBox(x=55.0, y=45.0, w=70.0, h=50.0)
    result = draw_frame_overlay(
        frame=frame,
        bbox=bbox,
        tier=0,
        signals={},
        fps=30.0,
        gt_bbox=gt,
    )
    assert result.shape == frame.shape
