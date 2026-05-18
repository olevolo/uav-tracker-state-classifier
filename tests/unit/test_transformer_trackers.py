"""Unit tests for OSTrackTracker and STARKTracker.

Covers:
- Registry presence
- tier_hint value
- reset() clears per-sequence state without touching the model
- is_stub_mode property (True when no real weights on disk)
- flops_per_update() returns a positive number
"""
from __future__ import annotations

import numpy as np
import pytest

from uav_tracker.registry import TRACKERS
from uav_tracker.types import BBox
from uav_tracker.trackers.transformer.ostrack import OSTrackTracker
from uav_tracker.trackers.transformer.stark import STARKTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)


def _rand_frame(h: int = 180, w: int = 240) -> np.ndarray:
    """Return a random uint8 BGR frame of size (h, w, 3)."""
    return _RNG.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _center_bbox(h: int = 180, w: int = 240) -> BBox:
    """Return a small bbox near the frame centre."""
    return BBox(x=w / 2 - 20, y=h / 2 - 20, w=40, h=40)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_ostrack_in_trackers_registry() -> None:
    assert "ostrack_256" in TRACKERS


def test_stark_in_trackers_registry() -> None:
    assert "stark_s50" in TRACKERS


# ---------------------------------------------------------------------------
# tier_hint
# ---------------------------------------------------------------------------


def test_ostrack_tier_hint_is_2() -> None:
    assert OSTrackTracker.tier_hint == 2


def test_stark_tier_hint_is_2() -> None:
    assert STARKTracker.tier_hint == 2


# ---------------------------------------------------------------------------
# reset() clears per-sequence state
# ---------------------------------------------------------------------------


def test_ostrack_reset_clears_state() -> None:
    tracker = OSTrackTracker(device="cpu")
    frame = _rand_frame()
    bbox = _center_bbox()
    tracker.init(frame, bbox)
    # After init both attributes should be populated
    assert tracker._template_feat is not None
    assert tracker._last_bbox is not None
    # reset() must clear them
    tracker.reset()
    assert tracker._template_feat is None
    assert tracker._last_bbox is None
    # The model must NOT be cleared (avoids expensive reload)
    assert tracker._model is not None


def test_stark_reset_clears_state() -> None:
    tracker = STARKTracker(device="cpu")
    frame = _rand_frame()
    bbox = _center_bbox()
    tracker.init(frame, bbox)
    assert tracker._template_feat is not None
    assert tracker._last_bbox is not None
    tracker.reset()
    assert tracker._template_feat is None
    assert tracker._last_bbox is None
    assert tracker._model is not None


# ---------------------------------------------------------------------------
# is_stub_mode property
# ---------------------------------------------------------------------------


def test_ostrack_stub_mode_property() -> None:
    """A freshly constructed OSTrackTracker has is_stub == True (no weights)."""
    tracker = OSTrackTracker(device="cpu")
    # Before _load_model() is called the default is True
    assert tracker.is_stub_mode is True
    # After init() loads the stub backbone it should still be True
    tracker.init(_rand_frame(), _center_bbox())
    assert tracker.is_stub_mode is True


def test_stark_stub_mode_property() -> None:
    """A freshly constructed STARKTracker has is_stub == True (no weights)."""
    tracker = STARKTracker(device="cpu")
    assert tracker.is_stub_mode is True
    tracker.init(_rand_frame(), _center_bbox())
    assert tracker.is_stub_mode is True


# ---------------------------------------------------------------------------
# flops_per_update()
# ---------------------------------------------------------------------------


def test_ostrack_flops_positive() -> None:
    tracker = OSTrackTracker(device="cpu")
    assert tracker.flops_per_update() > 0


def test_stark_flops_positive() -> None:
    tracker = STARKTracker(device="cpu")
    assert tracker.flops_per_update() > 0
