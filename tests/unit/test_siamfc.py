"""Unit tests for SiamFCTracker (Phase 2).

Tests:
  - Import succeeds.
  - Registry presence.
  - Instantiation without weights.
  - weights_loaded is False when weights file is absent.
  - Forward pass: init() + update() on random frames does not raise.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_import():
    """SiamFCTracker can be imported."""
    from uav_tracker.trackers.siamese.siamfc import SiamFCTracker  # noqa: F401


def test_registry_presence():
    """'siamfc' is registered in TRACKERS after importing uav_tracker."""
    import uav_tracker  # noqa: F401 — triggers _register_plugins()
    from uav_tracker.registry import TRACKERS

    assert "siamfc" in TRACKERS.names(), f"Expected 'siamfc' in {TRACKERS.names()}"


def test_instantiation_no_crash():
    """SiamFCTracker(device='cpu') instantiates without raising."""
    from uav_tracker.trackers.siamese.siamfc import SiamFCTracker

    tracker = SiamFCTracker(device="cpu")
    assert tracker.name == "siamfc"
    assert tracker.tier_hint == 1


def test_weights_loaded_false_when_missing(tmp_path):
    """weights_loaded is False after init() when no weights file exists."""
    from uav_tracker.trackers.siamese.siamfc import SiamFCTracker
    from uav_tracker.types import BBox
    import warnings

    # Point to a non-existent path
    fake_path = str(tmp_path / "nonexistent.pth")
    tracker = SiamFCTracker(device="cpu", weights_path=fake_path)

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    bbox = BBox(64.0, 64.0, 32.0, 32.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tracker.init(frame, bbox)

    assert tracker.weights_loaded is False


def test_forward_pass_no_raise():
    """init() + update() on random BGR frames complete without raising."""
    from uav_tracker.trackers.siamese.siamfc import SiamFCTracker
    from uav_tracker.types import BBox, TrackState
    import warnings

    rng = np.random.default_rng(0)
    frame0 = rng.integers(0, 256, (255, 255, 3), dtype=np.uint8)
    frame1 = rng.integers(0, 256, (255, 255, 3), dtype=np.uint8)
    bbox = BBox(64.0, 64.0, 32.0, 32.0)

    tracker = SiamFCTracker(device="cpu")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tracker.init(frame0, bbox)
        state = tracker.update(frame1)

    assert isinstance(state, TrackState)
    # BBox fields must be finite
    assert all(
        np.isfinite(v)
        for v in (state.bbox.x, state.bbox.y, state.bbox.w, state.bbox.h)
    )
    assert 0.0 <= state.confidence <= 1.0
    assert state.status in ("locked", "uncertain", "lost")


def test_flops_per_update_returns_float():
    """flops_per_update() returns a positive float without needing weights."""
    from uav_tracker.trackers.siamese.siamfc import SiamFCTracker

    tracker = SiamFCTracker(device="cpu")
    flops = tracker.flops_per_update()
    assert isinstance(flops, float)
    assert flops > 0.0
