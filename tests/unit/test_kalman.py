"""Unit tests for ``ConstantVelocityKalman``.

These tests validate the Phase 1 invariant: given straight-line motion
with no noise, the Kalman filter's predicted state matches the analytic
CV solution. Marked ``skip`` until Phase 1 implementation lands.
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_tracker import BBox
from uav_tracker.kalman.constant_velocity import ConstantVelocityKalman


@pytest.mark.skip(reason="Phase 1: Kalman implementation")
def test_kalman_tracks_straight_line() -> None:
    """Analytic CV: vx=1, vy=0 → center x grows linearly, y stays."""
    kf = ConstantVelocityKalman(process_noise=1e-6, measurement_noise=1e-6)
    kf.init(BBox(x=0.0, y=0.0, w=2.0, h=2.0))  # center=(1, 1)
    for i in range(1, 11):
        kf.predict()
        kf.update(BBox(x=float(i), y=0.0, w=2.0, h=2.0))
    x, y, vx, vy = kf.state()
    # After 10 unit-velocity steps: center should be ~ (11, 1), vx~1, vy~0.
    assert np.isclose(x, 11.0, atol=0.5)
    assert np.isclose(y, 1.0, atol=0.5)
    assert np.isclose(vx, 1.0, atol=0.1)
    assert np.isclose(vy, 0.0, atol=0.1)


def test_kalman_construction_is_cheap() -> None:
    """Construction must not raise even without any frames."""
    kf = ConstantVelocityKalman()
    assert kf.dt == 1.0
    assert kf.q == 0.01
    assert kf.r == 0.1
