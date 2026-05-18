"""Unit tests for KCFHenriques2015Tracker (Phase 8 reference port).

Test coverage:
  1. Import succeeds.
  2. Registry presence.
  3. Instantiation without crash.
  4. Protocol attributes (name, tier_hint, flops_per_update).
  5. init() + update() on synthetic noise frames do not raise.
  6. update() output is a valid TrackState with finite bbox.
  7. OPE on synthetic_static: AUC >= 0.85 (stationary target).
  8. OPE on synthetic_linear: AUC >= 0.50 (constant-velocity motion).
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


# ---------------------------------------------------------------------------
# 1. Import
# ---------------------------------------------------------------------------

def test_import() -> None:
    """Module can be imported without errors."""
    from uav_tracker.trackers.kcf_henriques import KCFHenriques2015Tracker  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Registry
# ---------------------------------------------------------------------------

def test_registry_presence() -> None:
    """'kcf_henriques' is registered in TRACKERS after package import."""
    import uav_tracker  # triggers _register_plugins()

    assert "kcf_henriques" in uav_tracker.TRACKERS.names(), (
        f"Expected 'kcf_henriques' in {uav_tracker.TRACKERS.names()}"
    )


# ---------------------------------------------------------------------------
# 3-4. Instantiation + protocol attrs
# ---------------------------------------------------------------------------

def test_instantiation_no_crash() -> None:
    """KCFHenriques2015Tracker() instantiates without raising."""
    from uav_tracker.trackers.kcf_henriques import KCFHenriques2015Tracker

    t = KCFHenriques2015Tracker()
    assert t.name == "kcf_henriques"
    assert t.tier_hint == 0


def test_flops_per_update() -> None:
    """flops_per_update() returns a positive float."""
    from uav_tracker.trackers.kcf_henriques import KCFHenriques2015Tracker

    t = KCFHenriques2015Tracker()
    flops = t.flops_per_update()
    assert isinstance(flops, float)
    assert flops > 0.0


# ---------------------------------------------------------------------------
# 5-6. init + update on synthetic noise frames
# ---------------------------------------------------------------------------

def test_init_update_no_raise() -> None:
    """init() + update() on random BGR frames do not raise."""
    from uav_tracker.trackers.kcf_henriques import KCFHenriques2015Tracker
    from uav_tracker.types import BBox

    rng = np.random.default_rng(7)
    frame0 = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    frame1 = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    bbox = BBox(x=120.0, y=90.0, w=60.0, h=45.0)

    tracker = KCFHenriques2015Tracker()
    tracker.init(frame0, bbox)
    state = tracker.update(frame1)

    assert state is not None


def test_update_returns_valid_trackstate() -> None:
    """update() returns a TrackState with finite bbox and valid confidence."""
    from uav_tracker.trackers.kcf_henriques import KCFHenriques2015Tracker
    from uav_tracker.types import BBox, TrackState

    rng = np.random.default_rng(42)
    frame0 = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    frame1 = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    bbox = BBox(x=100.0, y=80.0, w=60.0, h=45.0)

    tracker = KCFHenriques2015Tracker()
    tracker.init(frame0, bbox)
    state = tracker.update(frame1)

    assert isinstance(state, TrackState)
    assert np.isfinite(state.bbox.x)
    assert np.isfinite(state.bbox.y)
    assert np.isfinite(state.bbox.w)
    assert np.isfinite(state.bbox.h)
    assert 0.0 <= state.confidence <= 1.0
    assert state.status in ("locked", "uncertain", "lost")


def test_update_called_before_init_raises() -> None:
    """update() before init() must raise RuntimeError."""
    from uav_tracker.trackers.kcf_henriques import KCFHenriques2015Tracker

    tracker = KCFHenriques2015Tracker()
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError):
        tracker.update(frame)


# ---------------------------------------------------------------------------
# 7. OPE on synthetic_static: AUC >= 0.85
# ---------------------------------------------------------------------------

def test_ope_synthetic_static_auc() -> None:
    """On a stationary target, AUC must be >= 0.85."""
    from uav_tracker.datasets.synthetic import SyntheticDataset
    from uav_tracker.evaluation.ope import OPERunner
    from uav_tracker.trackers.kcf_henriques import KCFHenriques2015Tracker

    ds = SyntheticDataset(seed=42)
    static_ds = ds.filter({"STATIC"})

    tracker = KCFHenriques2015Tracker()
    runner = OPERunner(seed=42)
    result = runner.run(tracker, static_ds)

    assert result.auc >= 0.85, (
        f"synthetic_static AUC {result.auc:.4f} < 0.85 — "
        "KCF must hold a stationary target almost perfectly"
    )


# ---------------------------------------------------------------------------
# 8. OPE on synthetic_linear: AUC >= 0.50
# ---------------------------------------------------------------------------

def test_ope_synthetic_linear_auc() -> None:
    """On a constant-velocity target, AUC must be >= 0.50."""
    from uav_tracker.datasets.synthetic import SyntheticDataset
    from uav_tracker.evaluation.ope import OPERunner
    from uav_tracker.trackers.kcf_henriques import KCFHenriques2015Tracker

    ds = SyntheticDataset(seed=42)
    linear_ds = ds.filter({"LINEAR"})

    tracker = KCFHenriques2015Tracker()
    runner = OPERunner(seed=42)
    result = runner.run(tracker, linear_ds)

    assert result.auc >= 0.50, (
        f"synthetic_linear AUC {result.auc:.4f} < 0.50 — "
        "KCF should track constant-velocity motion reliably"
    )
