"""Constant-velocity Kalman filter (4-state, dt=1).

Paper §3.1: smooths KCF measurements and short-horizon-predicts the
ROI when KCF confidence drops. State = [x, y, vx, vy]^T where (x, y) is
the bbox center.

This module is deliberately NumPy-only for portability and
testability; Phase 1 tests in ``tests/unit/test_kalman.py`` exercise
closed-form CV motion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_tracker.types import BBox


@dataclass
class _KalmanState:
    """Mutable Kalman internal state. Not part of the public API."""

    x: np.ndarray  # (4,) state vector
    P: np.ndarray  # (4, 4) covariance


class ConstantVelocityKalman:
    """Constant-velocity Kalman filter with unit time step.

    Parameters
    ----------
    process_noise:
        Diagonal entry of Q (scalar). Paper default 0.01.
    measurement_noise:
        Diagonal entry of R (scalar). Paper default 0.1.
    dt:
        Time step between frames. Default 1 (per-frame). Kept for
        completeness even though the paper assumes uniform sampling.
    """

    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        dt: float = 1.0,
    ) -> None:
        self.dt = float(dt)
        self.q = float(process_noise)
        self.r = float(measurement_noise)

        # State-transition matrix F for constant velocity
        #   x_{t+1} = x_t + vx_t * dt
        #   vx_{t+1} = vx_t
        self._F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        # Measurement matrix: we observe (x, y) only.
        self._H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self._Q = self.q * np.eye(4, dtype=np.float64)
        self._R = self.r * np.eye(2, dtype=np.float64)

        self._state: _KalmanState | None = None
        # Keep last measured w,h for state() reconstruction. Width/height
        # are not modelled in the CV state.
        self._last_wh: tuple[float, float] | None = None

    # ------------------------------------------------------------------

    def init(self, bbox: BBox) -> None:
        """Seed the filter from an initial bbox."""
        cx = bbox.x + bbox.w / 2.0
        cy = bbox.y + bbox.h / 2.0
        self._state = _KalmanState(
            x=np.array([cx, cy, 0.0, 0.0], dtype=np.float64),
            P=np.eye(4, dtype=np.float64),
        )
        self._last_wh = (bbox.w, bbox.h)

    def predict(self) -> tuple[float, float]:
        """Advance the state one step and return the predicted (cx, cy).

        Raises if ``init`` was not called.
        """
        if self._state is None:
            raise RuntimeError("ConstantVelocityKalman.predict called before init")
        # x_{t+1|t} = F @ x_{t|t}
        x_pred = self._F @ self._state.x
        # P_{t+1|t} = F @ P_{t|t} @ F^T + Q
        P_pred = self._F @ self._state.P @ self._F.T + self._Q
        self._state.x = x_pred
        self._state.P = P_pred
        return (float(x_pred[0]), float(x_pred[1]))

    def update(self, measurement: BBox) -> None:
        """Incorporate a measurement bbox (standard Kalman correction)."""
        if self._state is None:
            raise RuntimeError("ConstantVelocityKalman.update called before init")
        self._last_wh = (measurement.w, measurement.h)
        cx = measurement.x + measurement.w / 2.0
        cy = measurement.y + measurement.h / 2.0
        z = np.array([cx, cy], dtype=np.float64)

        # Innovation: y = z - H @ x_{t+1|t}
        y = z - self._H @ self._state.x
        # Innovation covariance: S = H @ P @ H^T + R
        S = self._H @ self._state.P @ self._H.T + self._R
        # Kalman gain: K = P @ H^T @ S^{-1}
        K = self._state.P @ self._H.T @ np.linalg.inv(S)
        # State update
        self._state.x = self._state.x + K @ y
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(4) - K @ self._H
        self._state.P = I_KH @ self._state.P

    def state(self) -> tuple[float, float, float, float]:
        """Return current (x, y, vx, vy) as a plain tuple."""
        if self._state is None:
            raise RuntimeError("ConstantVelocityKalman.state called before init")
        x = self._state.x
        return (float(x[0]), float(x[1]), float(x[2]), float(x[3]))
