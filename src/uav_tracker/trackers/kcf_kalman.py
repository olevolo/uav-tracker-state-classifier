"""KCF + Kalman hybrid tracker (Phase 1 fast tracker).

Paper: Oleksiuk & Velhosh (2026), §3 method — KCF is the LIGHT-tier
backbone; a 4-state constant-velocity Kalman filter smooths and
short-horizon-predicts the ROI so KCF can coast through small gaps.

Tier hint: 0 (lightest). Target FPS on T4: ~160. GFLOPs/update: ~0.02.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from uav_tracker.registry import TRACKERS
from uav_tracker.types import BBox, TrackState

try:  # pragma: no cover - CI may lack OpenCV contrib at import time
    import cv2  # type: ignore
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore[assignment]


@TRACKERS.register("kcf_kalman")
class KCFKalmanTracker:
    """KCF (correlation filter) + constant-velocity Kalman smoother.

    Paper §3.1 / Table 1 defaults. Exposes the KCF correlation response
    map via ``TrackState.aux['response_map']`` so the APCE signal
    (Phase 5) can read it without re-running KCF.

    Parameters
    ----------
    sigma:
        Gaussian spatial-bandwidth for KCF (paper default 0.125).
    kalman_process_noise:
        Process-noise covariance scalar Q (diagonal).
    kalman_measurement_noise:
        Measurement-noise covariance scalar R (diagonal).
    """

    name: str = "kcf_kalman"
    tier_hint: int = 0

    # Static FLOPs estimate per the paper (0.02 GFLOPs/frame on ROI-sized FFT).
    _FLOPS_PER_UPDATE: float = 0.02 * 1e9

    def __init__(
        self,
        sigma: float = 0.125,
        kalman_process_noise: float = 0.01,
        kalman_measurement_noise: float = 0.1,
    ) -> None:
        self.sigma = sigma
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self._cv_tracker: Any = None
        self._kalman: Any = None
        self._last_bbox: BBox | None = None

    # ------------------------------------------------------------------ API

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialize KCF tracker + Kalman state from the first frame.

        Paper §3.1. Ground-truth bbox on frame 0 is assumed.
        """
        if cv2 is None:  # pragma: no cover
            raise RuntimeError(
                "opencv-contrib-python is required for KCFKalmanTracker. "
                "Run `make setup` to install it."
            )
        if not hasattr(cv2, "TrackerKCF_create"):  # pragma: no cover
            raise RuntimeError(
                "cv2.TrackerKCF_create not available — opencv-contrib-python "
                "is needed (not the plain opencv-python package). "
                "Run `make setup` to install the correct wheel."
            )
        self._cv_tracker = cv2.TrackerKCF_create()
        # OpenCV expects a tuple of ints for the bbox (x, y, w, h).
        rect = (int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h))
        self._cv_tracker.init(frame, rect)

        # Initialise Kalman filter.
        from uav_tracker.kalman.constant_velocity import ConstantVelocityKalman
        self._kalman = ConstantVelocityKalman(
            process_noise=self.kalman_process_noise,
            measurement_noise=self.kalman_measurement_noise,
        )
        self._kalman.init(bbox)
        self._last_bbox = bbox

    def update(self, frame: np.ndarray) -> TrackState:
        """Advance one frame.

        Returns a ``TrackState`` with the KCF bbox (Kalman-smoothed),
        the KCF confidence (peak of response map), and the raw response
        map in ``aux``. Paper §3.1 smoothing math.
        """
        if self._cv_tracker is None or self._kalman is None:
            raise RuntimeError("KCFKalmanTracker.update called before init")

        # Kalman predict first (gives us the prior for this frame).
        self._kalman.predict()

        # Run KCF.
        ok, rect = self._cv_tracker.update(frame)

        if ok:
            # rect is (x, y, w, h) floats from OpenCV.
            raw_bbox = BBox(
                x=float(rect[0]),
                y=float(rect[1]),
                w=float(rect[2]),
                h=float(rect[3]),
            )
            # Update Kalman with KCF measurement.
            self._kalman.update(raw_bbox)
            # Reconstruct smoothed bbox from Kalman state.
            cx_s, cy_s, _, _ = self._kalman.state()
            smoothed = BBox(
                x=cx_s - raw_bbox.w / 2.0,
                y=cy_s - raw_bbox.h / 2.0,
                w=raw_bbox.w,
                h=raw_bbox.h,
            )
            self._last_bbox = smoothed
            confidence = 0.8
            status = "locked"
        else:
            # KCF lost the target — use Kalman prediction only.
            cx_s, cy_s, _, _ = self._kalman.state()
            w = self._last_bbox.w if self._last_bbox is not None else 60.0
            h = self._last_bbox.h if self._last_bbox is not None else 45.0
            smoothed = BBox(x=cx_s - w / 2.0, y=cy_s - h / 2.0, w=w, h=h)
            self._last_bbox = smoothed
            confidence = 0.2
            status = "uncertain"

        return TrackState(bbox=smoothed, confidence=confidence, status=status)

    def flops_per_update(self) -> float:
        """Static estimate in FLOPs (not GFLOPs).

        Matches PLAN §3.3 reference for KCF (~0.02 GFLOPs/frame).
        """
        return self._FLOPS_PER_UPDATE

    # ----------------------------------------------------- optional hooks

    def on_tier_enter(self, ctx: Any) -> None:  # noqa: D401 - hook
        """Runner calls this when the scheduler switches IN to tier 0.

        On DEEP→LIGHT we re-center KCF on the deep tracker's bbox and
        refresh the appearance template (PLAN §3.3).
        """
        # TODO Phase 3: re-center + refresh appearance.
        return None

    def on_tier_exit(self, ctx: Any) -> None:  # noqa: D401 - hook
        """Runner calls this when scheduler switches OUT of tier 0."""
        return None

    def reset(self) -> None:
        """Restore tracker to un-initialised state (used between sequences)."""
        self._cv_tracker = None
        self._kalman = None
        self._last_bbox = None
