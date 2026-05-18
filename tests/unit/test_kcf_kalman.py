"""Smoke tests for ``KCFKalmanTracker``.

We guard the import with ``pytest.importorskip`` so Phase 0 CI without
OpenCV-contrib still collects (and skips) cleanly.
"""

from __future__ import annotations

import pytest


def test_kcf_kalman_is_registered() -> None:
    """Registry should know about ``kcf_kalman`` after package import."""
    import uav_tracker  # triggers _register_plugins

    assert "kcf_kalman" in uav_tracker.TRACKERS.names()


def test_kcf_kalman_flops_constant() -> None:
    """``flops_per_update`` should return a positive static value."""
    pytest.importorskip("cv2")
    from uav_tracker.trackers.kcf_kalman import KCFKalmanTracker

    t = KCFKalmanTracker()
    assert t.flops_per_update() > 0
    # Metadata attrs per Protocol.
    assert t.name == "kcf_kalman"
    assert t.tier_hint == 0


@pytest.mark.skip(reason="Phase 1: KCF wiring")
def test_kcf_kalman_init_update_roundtrip() -> None:  # pragma: no cover
    """End-to-end unit test lands with the Phase 1 implementation."""
    pass
