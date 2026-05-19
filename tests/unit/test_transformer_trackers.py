"""Unit tests for OSTrackTracker and SGLATracker.

Covers:
- Registry presence
- tier_hint value
- reset() clears per-sequence state without touching the model
- is_stub_mode property
- flops_per_update() returns a positive number
"""
from __future__ import annotations

import numpy as np

from uav_tracker.registry import TRACKERS
from uav_tracker.types import BBox
from uav_tracker.trackers.transformer.ostrack import OSTrackTracker
from uav_tracker.trackers.sglatrack import SGLATracker

_RNG = np.random.default_rng(42)


def _rand_frame(h: int = 180, w: int = 240) -> np.ndarray:
    return _RNG.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _center_bbox(h: int = 180, w: int = 240) -> BBox:
    return BBox(x=w / 2 - 20, y=h / 2 - 20, w=40, h=40)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_ostrack_in_registry() -> None:
    assert "ostrack_256" in TRACKERS


def test_sglatrack_in_registry() -> None:
    assert "sglatrack" in TRACKERS


# ---------------------------------------------------------------------------
# tier_hint
# ---------------------------------------------------------------------------

def test_ostrack_tier_hint() -> None:
    assert OSTrackTracker.tier_hint == 2


def test_sglatrack_tier_hint() -> None:
    assert SGLATracker.tier_hint == 1


# ---------------------------------------------------------------------------
# reset() clears per-sequence state
# ---------------------------------------------------------------------------

def test_ostrack_reset_clears_state() -> None:
    tracker = OSTrackTracker(device="cpu")
    tracker.init(_rand_frame(), _center_bbox())
    assert tracker._template is not None
    assert tracker._last_bbox is not None
    tracker.reset()
    assert tracker._template is None
    assert tracker._last_bbox is None
    assert tracker._model is not None  # model weights retained


def test_sglatrack_reset_clears_state() -> None:
    tracker = SGLATracker(device="cpu")
    tracker.init(_rand_frame(), _center_bbox())
    assert tracker._z_tensor is not None
    assert tracker._state is not None
    tracker.reset()
    assert tracker._z_tensor is None
    assert tracker._state is None
    assert tracker._model is not None  # model weights retained


# ---------------------------------------------------------------------------
# is_stub_mode
# ---------------------------------------------------------------------------

def test_ostrack_stub_mode() -> None:
    tracker = OSTrackTracker(device="cpu")
    assert tracker.is_stub_mode is True  # before init
    tracker.init(_rand_frame(), _center_bbox())
    # After init with real weights at UAV_WEIGHTS_ROOT: False; without: True
    assert isinstance(tracker.is_stub_mode, bool)


def test_sglatrack_stub_mode_type() -> None:
    tracker = SGLATracker(device="cpu")
    assert isinstance(tracker.is_stub_mode, bool)


# ---------------------------------------------------------------------------
# flops_per_update()
# ---------------------------------------------------------------------------

def test_ostrack_flops_positive() -> None:
    assert OSTrackTracker(device="cpu").flops_per_update() > 0


def test_sglatrack_flops_positive() -> None:
    assert SGLATracker(device="cpu").flops_per_update() > 0
