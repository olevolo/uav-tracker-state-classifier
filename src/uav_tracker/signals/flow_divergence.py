"""FlowDivergenceSignal — residual-flow divergence switching signal (Phase 5).

Computes the divergence of the residual optical-flow field inside the tracker's
ROI using finite differences on the sparse flow grid:

    div = mean(∂u/∂x + ∂v/∂y)

where (u, v) are the (dx, dy) residual-flow components at each tracked point
inside the ROI, and the partial derivatives are estimated via finite differences
over the nearest-neighbor grid formed by those sparse points.

High divergence indicates expanding/contracting motion (object moving toward /
away from camera, or tracking drift) → high disorder signal.

Signal value: ``|div| / (|div| + 1)`` ∈ [0, 1)  (soft normalization to keep
signal bounded and differentiable at the boundary).

Uses the same Shi-Tomasi + LK + global-motion-subtract front-end as
MotionEntropySignal — imports helpers; does NOT reimplement them.

Registration key: ``"flow_divergence"``
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


def _estimate_divergence_sparse(
    positions: np.ndarray,
    flows: np.ndarray,
) -> float:
    """Estimate mean divergence from a sparse flow field.

    Uses a simple finite-difference estimate via nearest-neighbor pairs:
    for each point p_i, find its nearest neighbour p_j; estimate
    ∂u/∂x ≈ (u_j - u_i) / (x_j - x_i), ∂v/∂y analogously, then average.

    Parameters
    ----------
    positions : (N, 2) float array — (x, y) of tracked points.
    flows     : (N, 2) float array — (dx, dy) residual-flow at each point.

    Returns
    -------
    float — mean divergence estimate; 0.0 if fewer than 2 points.
    """
    N = len(positions)
    if N < 2:
        return 0.0

    divs: list[float] = []
    for i in range(N):
        xi, yi = positions[i]
        ui, vi = flows[i]

        # Find nearest neighbour (excluding self).
        diffs = positions - positions[i]  # (N, 2)
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = np.inf
        j = int(np.argmin(dists))

        xj, yj = positions[j]
        uj, vj = flows[j]

        dx = xj - xi
        dy = yj - yi

        # Avoid division by near-zero.
        du_dx = (uj - ui) / dx if abs(dx) > 0.5 else 0.0
        dv_dy = (vj - vi) / dy if abs(dy) > 0.5 else 0.0
        divs.append(du_dx + dv_dy)

    return float(np.mean(divs)) if divs else 0.0


@SIGNALS.register("flow_divergence")
class FlowDivergenceSignal:
    """Flow-divergence switching signal.

    Computes the divergence of the residual (ego-motion subtracted) optical-flow
    field inside the ROI. High |divergence| → possible occlusion, zoom, or
    tracking failure → high signal value.

    Parameters
    ----------
    mag_threshold:
        Minimum residual-flow magnitude to include a point in the divergence
        estimate (default 0.5 px — lower than motion_entropy to keep more
        points for the finite-difference grid).
    alpha:
        EMA smoothing factor on the normalized |div| signal (default 0.6).
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

    name: str = "flow_divergence"
    range: tuple[float, float] = (0.0, 1.0)

    def __init__(
        self,
        mag_threshold: float = 0.5,
        alpha: float = 0.6,
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

        self._div_bar: float = 0.0
        self._prev_frame: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._prior_global: np.ndarray | None = None
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # SwitchSignal Protocol
    # ------------------------------------------------------------------

    def step(self, ctx: FrameContext, state: TrackState | None) -> SignalReport:
        """Compute flow-divergence signal for the current frame.

        Returns
        -------
        SignalReport
            ``value`` — EMA-smoothed ``|div| / (|div| + 1)`` in [0, 1).
            ``reliable`` — ``False`` on first frame or when global-motion
            estimation falls back to a reused prior.
            ``aux`` — ``{"div_raw": div}``.
        """
        frame = ctx.frame
        prev_frame = ctx.prev_frame
        bbox = ctx.bbox

        _unreliable = SignalReport(
            value=self._div_bar,
            reliable=False,
            aux={"div_raw": 0.0},
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

        if self._prev_pts is None or len(self._prev_pts) == 0:
            self._prev_pts = detect_corners(
                self._prev_frame, roi,
                background_band=self.background_band,
                max_corners=self.max_corners,
                quality_level=self.quality_level,
            )

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

        local_flow = (curr_good - prev_good).reshape(-1, 2)
        positions = prev_good.reshape(-1, 2)

        # Global motion estimation.
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

        # Filter by magnitude threshold (keep only moving points).
        magnitudes = np.linalg.norm(residual_flow, axis=1)
        above = magnitudes >= self.mag_threshold
        if above.sum() >= 2:
            used_positions = positions[above]
            used_flows = residual_flow[above]
        else:
            used_positions = positions
            used_flows = residual_flow

        # Compute divergence.
        div_raw = _estimate_divergence_sparse(used_positions, used_flows)

        # Normalize to [0, 1): |div| / (|div| + 1)
        abs_div = abs(div_raw)
        div_norm = abs_div / (abs_div + 1.0)
        div_norm = float(np.clip(div_norm, 0.0, 1.0))

        self._div_bar = self.alpha * self._div_bar + (1.0 - self.alpha) * div_norm
        self._div_bar = float(np.clip(self._div_bar, 0.0, 1.0))

        self._update_state(frame, roi)

        return SignalReport(
            value=self._div_bar,
            reliable=reliable_signal,
            aux={"div_raw": div_raw},
        )

    def reset(self) -> None:
        """Restore to construction state. Idempotent."""
        self._div_bar = 0.0
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


__all__ = ["FlowDivergenceSignal"]
