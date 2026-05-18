"""CircularResultantSignal — directional-statistics switching signal (Phase 5).

Computes the mean resultant length R over residual-flow orientation vectors,
magnitude-weighted:

    R = |(1/W) * Σ w_j * exp(i * θ_j)|

where W = Σ w_j, θ_j is the orientation angle of residual-flow vector j, and
w_j is its magnitude (only vectors with magnitude ≥ mag_threshold are included).

Signal value: ``1 - R``  (0 = coherent/locked, 1 = disordered/entropy-like)

Advantages over Shannon entropy (see paper §2.2):
  - No histogram binning → no cyclic-boundary artefacts.
  - Stable at small N (directional mean well-defined for N ≥ 1).
  - Differentiable in closed form.

Uses the same Shi-Tomasi + LK + global-motion-subtract front-end as
MotionEntropySignal — imports helpers from optical_flow and global_motion;
does NOT reimplement them.

Registration key: ``"circular_resultant"``
Range: [0.0, 1.0]
"""

from __future__ import annotations

import numpy as np

from uav_tracker.registry import SIGNALS
from uav_tracker.types import FrameContext, SignalReport, TrackState

from .global_motion import estimate_global_flow
from .optical_flow import _to_gray, detect_corners, track_flow

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False


@SIGNALS.register("circular_resultant")
class CircularResultantSignal:
    """Circular-statistics switching signal.

    Computes ``R = |(1/N) Σ e^(iθ_j)|`` over residual-flow orientation vectors,
    weighted by flow magnitude. Emits ``1 - R`` so that high disorder (many
    directions) → high signal → scheduler may escalate.

    Parameters
    ----------
    mag_threshold:
        Minimum residual-flow magnitude in pixels; vectors below this are
        excluded from the mean-resultant calculation (default 1.0 px).
    alpha:
        EMA smoothing factor on ``1 - R`` (default 0.8, same as motion_entropy).
    max_corners:
        Shi-Tomasi ``maxCorners`` parameter (default 200).
    quality_level:
        Shi-Tomasi ``qualityLevel`` parameter (default 0.01).
    background_band:
        Pixel width of the background band for global-motion estimation
        (default 20).
    seed:
        RNG seed for RANSAC reproducibility.
    """

    name: str = "circular_resultant"
    range: tuple[float, float] = (0.0, 1.0)

    def __init__(
        self,
        mag_threshold: float = 1.0,
        alpha: float = 0.8,
        max_corners: int = 200,
        quality_level: float = 0.01,
        background_band: int = 20,
        seed: int = 42,
    ) -> None:
        self.mag_threshold = mag_threshold
        self.alpha = alpha
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.background_band = background_band
        self.seed = seed

        if _CV2_AVAILABLE:
            _cv2.setRNGSeed(seed)

        self._R_bar: float = 0.0  # EMA of (1 - R)
        self._prev_frame: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None  # (N, 1, 2) all corners
        self._prior_global: np.ndarray | None = None  # cached homography (N, 1, 2) displ
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # SwitchSignal Protocol
    # ------------------------------------------------------------------

    def step(self, ctx: FrameContext, state: TrackState | None) -> SignalReport:
        """Compute circular-resultant signal for the current frame.

        Returns
        -------
        SignalReport
            ``value`` — EMA-smoothed ``1 - R`` in [0, 1].
            ``reliable`` — ``False`` on first frame or when global-motion
            estimation falls back to a reused prior.
            ``aux`` — ``{"R": R, "n_vectors": int}``.
        """
        frame = ctx.frame
        prev_frame = ctx.prev_frame
        bbox = ctx.bbox

        _unreliable = SignalReport(
            value=self._R_bar,
            reliable=False,
            aux={"R": 1.0 - self._R_bar, "n_vectors": 0},
        )

        if prev_frame is None or not self._initialized:
            self._prev_frame = frame.copy()
            if bbox is not None:
                roi = (bbox.x, bbox.y, bbox.w, bbox.h)
                self._prev_pts = detect_corners(
                    frame, roi,
                    background_band=self.background_band,
                    max_corners=self.max_corners,
                    quality_level=self.quality_level,
                )
            else:
                self._prev_pts = None
            self._initialized = True
            return _unreliable

        if bbox is None:
            return _unreliable

        roi = (bbox.x, bbox.y, bbox.w, bbox.h)

        # Refresh corners if none.
        if self._prev_pts is None or len(self._prev_pts) == 0:
            self._prev_pts = detect_corners(
                self._prev_frame, roi,
                background_band=self.background_band,
                max_corners=self.max_corners,
                quality_level=self.quality_level,
            )

        # LK flow.
        if self._prev_pts is not None and len(self._prev_pts) > 0:
            curr_pts, status = track_flow(self._prev_frame, frame, self._prev_pts)
            good = status == 1
            prev_good = self._prev_pts[good]
            curr_good = curr_pts[good]
        else:
            prev_good = np.empty((0, 1, 2), dtype=np.float32)
            curr_good = np.empty((0, 1, 2), dtype=np.float32)

        if len(prev_good) == 0:
            self._update_state(frame, roi)
            return _unreliable

        # Split into ROI-only vectors for signal + all for global motion.
        local_flow = (curr_good - prev_good).reshape(-1, 2)

        # Global motion from all background + ROI tracked points.
        if len(prev_good) >= 4:
            disp, method = estimate_global_flow(
                prev_good, curr_good, frame.shape[:2]
            )
        else:
            disp, method = None, "failed"

        reliable_signal = True

        if disp is not None:
            residual_flow = local_flow - disp.reshape(-1, 2)
            self._prior_global = disp
        elif self._prior_global is not None and len(self._prior_global) == len(local_flow):
            residual_flow = local_flow - self._prior_global.reshape(-1, 2)
            reliable_signal = False
        else:
            residual_flow = local_flow
            reliable_signal = False

        # Magnitude-weighted circular mean resultant.
        magnitudes = np.linalg.norm(residual_flow, axis=1)
        above = magnitudes >= self.mag_threshold

        n_vectors = int(above.sum())
        if n_vectors == 0:
            R = 1.0  # No motion → perfectly coherent → disorder = 0
        else:
            mags = magnitudes[above]
            vecs = residual_flow[above]
            angles = np.arctan2(vecs[:, 1], vecs[:, 0])
            W = mags.sum()
            if W <= 0.0:
                R = 1.0
            else:
                cx = float(np.dot(mags, np.cos(angles))) / W
                cy = float(np.dot(mags, np.sin(angles))) / W
                R = float(np.sqrt(cx ** 2 + cy ** 2))
                R = float(np.clip(R, 0.0, 1.0))

        disorder = 1.0 - R
        self._R_bar = self.alpha * self._R_bar + (1.0 - self.alpha) * disorder
        self._R_bar = float(np.clip(self._R_bar, 0.0, 1.0))

        self._update_state(frame, roi)

        return SignalReport(
            value=self._R_bar,
            reliable=reliable_signal,
            aux={"R": R, "n_vectors": n_vectors},
        )

    def reset(self) -> None:
        """Restore to construction state. Idempotent."""
        self._R_bar = 0.0
        self._prev_frame = None
        self._prev_pts = None
        self._prior_global = None
        self._initialized = False

    # ------------------------------------------------------------------

    def _update_state(self, frame: np.ndarray, roi: tuple) -> None:
        self._prev_frame = frame.copy()
        self._prev_pts = detect_corners(
            frame, roi,
            background_band=self.background_band,
            max_corners=self.max_corners,
            quality_level=self.quality_level,
        )


__all__ = ["CircularResultantSignal"]
