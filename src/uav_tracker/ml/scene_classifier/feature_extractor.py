"""32-dimensional flow feature extractor for scene classifier input.

Features layout:
  [0]    motion_entropy H̄ (from ctx.optical_flow_cache or signals aux)
  [1]    circular resultant 1-R (disorder measure)
  [2]    mean flow magnitude in ROI
  [3]    std flow magnitude in ROI
  [4]    flow divergence scalar
  [5-8]  histogram of flow angles (4 quadrant counts, normalized)
  [9]    tracker confidence
  [10-13] Kalman innovation approx: bbox displacement (dx, dy) + size change (dw, dh)
  [14]   bbox aspect ratio (w/h)
  [15]   bbox area (normalized by frame area)
  [16-17] normalized bbox center (cx/W, cy/H)
  [18-21] EMA of [10-13] over last 5 frames (alpha=0.8)
  [22-25] delta of [10-13] over last 3 frames
  [26-31] zeros (reserved)
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from uav_tracker.types import FrameContext, TrackState


class FlowFeatureExtractor:
    """Extracts a 32-dimensional feature vector from per-frame tracking context.

    Parameters
    ----------
    alpha : float
        EMA smoothing factor for the innovation features at indices [18-21].
        Default: 0.8 (fast adaptation, retains ~20% of history).
    """

    def __init__(self, alpha: float = 0.8) -> None:
        self._alpha = alpha
        self._ema = np.zeros(4, dtype=np.float32)  # EMA of [dx, dy, dw, dh]
        self._recent: deque = deque(maxlen=3)       # last 3 [dx, dy, dw, dh]
        self._prev_bbox: Optional[tuple[float, float, float, float]] = None

    # ---------------------------------------------------------------------- #

    def extract(self, ctx: FrameContext, state: TrackState) -> np.ndarray:
        """Return a float32 feature vector of shape (32,).

        All components gracefully fall back to 0.0 when the underlying data is
        unavailable (no optical flow cache, no previous bbox, etc.).

        Parameters
        ----------
        ctx :
            Current-frame context assembled by the HybridRunner.
        state :
            Current tracker output for this frame.
        """
        feat = np.zeros(32, dtype=np.float32)
        frame = ctx.frame  # H × W × 3  BGR

        # ------------------------------------------------------------------ #
        # [0] motion entropy H̄                                               #
        # ------------------------------------------------------------------ #
        if ctx.optical_flow_cache is not None:
            feat[0] = float(ctx.optical_flow_cache.get("motion_entropy", 0.0))
        elif ctx.telemetry:
            feat[0] = float(ctx.telemetry.get("motion_entropy", 0.0))

        # ------------------------------------------------------------------ #
        # [1] circular resultant disorder measure (1 - R)                     #
        # ------------------------------------------------------------------ #
        if ctx.optical_flow_cache is not None:
            # 1 - R: higher value = more disordered flow directions
            r = ctx.optical_flow_cache.get("circular_resultant", None)
            if r is not None:
                feat[1] = float(np.clip(1.0 - r, 0.0, 1.0))
        elif ctx.telemetry:
            r = ctx.telemetry.get("circular_resultant", None)
            if r is not None:
                feat[1] = float(np.clip(1.0 - r, 0.0, 1.0))

        # ------------------------------------------------------------------ #
        # [2-4] flow stats (magnitude mean, std, divergence)                  #
        # ------------------------------------------------------------------ #
        flow = None
        if ctx.optical_flow_cache is not None:
            flow = ctx.optical_flow_cache.get("flow", None)  # H × W × 2 or None

        if flow is not None:
            flow_arr = np.asarray(flow, dtype=np.float32)
            # Optionally restrict to ROI if bbox is valid
            roi_flow = self._roi_flow(flow_arr, ctx, frame)
            magnitudes = np.linalg.norm(roi_flow, axis=-1)  # H' × W'
            if magnitudes.size > 0:
                feat[2] = float(np.mean(magnitudes))
                feat[3] = float(np.std(magnitudes))

            # Flow divergence: ∂u/∂x + ∂v/∂y  (mean scalar)
            feat[4] = self._flow_divergence(roi_flow)

            # [5-8] histogram of flow angles (4 quadrants), normalized
            fy = roi_flow[..., 1].ravel()
            fx = roi_flow[..., 0].ravel()
            if fx.size > 0:
                angles = np.arctan2(fy, fx)  # [-π, π]
                bins = np.array([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
                counts, _ = np.histogram(angles, bins=bins)
                total = counts.sum()
                if total > 0:
                    feat[5:9] = counts.astype(np.float32) / total
        elif ctx.optical_flow_cache is not None:
            # Cache present but no "flow" key; try divergence from aux
            div = ctx.optical_flow_cache.get("divergence", None)
            if div is not None:
                feat[4] = float(div)

        # ------------------------------------------------------------------ #
        # [9] tracker confidence                                              #
        # ------------------------------------------------------------------ #
        feat[9] = float(np.clip(state.confidence, 0.0, 1.0))

        # ------------------------------------------------------------------ #
        # [10-13] bbox innovation: (dx, dy, dw, dh)                          #
        # ------------------------------------------------------------------ #
        curr_bbox = (state.bbox.x, state.bbox.y, state.bbox.w, state.bbox.h)
        if self._prev_bbox is not None:
            px, py, pw, ph = self._prev_bbox
            cx, cy, cw, ch = curr_bbox
            innov = np.array(
                [cx - px, cy - py, cw - pw, ch - ph], dtype=np.float32
            )
        else:
            innov = np.zeros(4, dtype=np.float32)
        feat[10:14] = innov

        # ------------------------------------------------------------------ #
        # [14] aspect ratio w/h                                               #
        # ------------------------------------------------------------------ #
        w, h = curr_bbox[2], curr_bbox[3]
        if h > 0:
            feat[14] = float(w / h)

        # ------------------------------------------------------------------ #
        # [15] normalised area                                                #
        # ------------------------------------------------------------------ #
        fh, fw = frame.shape[:2]
        frame_area = float(fw * fh) if fw * fh > 0 else 1.0
        feat[15] = float(w * h) / frame_area

        # ------------------------------------------------------------------ #
        # [16-17] normalised bbox center (cx/W, cy/H)                        #
        # ------------------------------------------------------------------ #
        if fw > 0:
            feat[16] = float(curr_bbox[0] + w / 2.0) / fw
        if fh > 0:
            feat[17] = float(curr_bbox[1] + h / 2.0) / fh

        # ------------------------------------------------------------------ #
        # [18-21] EMA of innovation (alpha=self._alpha)                       #
        # ------------------------------------------------------------------ #
        self._ema = self._alpha * self._ema + (1.0 - self._alpha) * innov
        feat[18:22] = self._ema

        # ------------------------------------------------------------------ #
        # [22-25] delta of innovation over last 3 frames                      #
        # ------------------------------------------------------------------ #
        self._recent.append(innov.copy())
        if len(self._recent) >= 2:
            feat[22:26] = self._recent[-1] - self._recent[0]

        # [26-31] reserved — stay at 0.0

        # Update previous bbox for next call
        self._prev_bbox = curr_bbox

        return feat

    def reset(self) -> None:
        """Reset all internal state for a new sequence."""
        self._ema = np.zeros(4, dtype=np.float32)
        self._recent.clear()
        self._prev_bbox = None

    # ---------------------------------------------------------------------- #
    # Private helpers                                                         #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _roi_flow(
        flow: np.ndarray,
        ctx: FrameContext,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Crop flow to the tracker ROI (2× scale context) if bbox is valid."""
        if ctx.bbox is None:
            return flow

        fh, fw = frame.shape[:2]
        bx, by, bw, bh = ctx.bbox.x, ctx.bbox.y, ctx.bbox.w, ctx.bbox.h

        # 2× context crop, clamped to frame
        cx = bx + bw / 2.0
        cy = by + bh / 2.0
        half_w = bw
        half_h = bh

        x1 = int(max(0, cx - half_w))
        y1 = int(max(0, cy - half_h))
        x2 = int(min(fw, cx + half_w))
        y2 = int(min(fh, cy + half_h))

        if x2 <= x1 or y2 <= y1:
            return flow

        # flow shape may be different from frame (e.g. downscaled)
        flow_h, flow_w = flow.shape[:2]
        if flow_h == fh and flow_w == fw:
            return flow[y1:y2, x1:x2]

        # scale crop coordinates to flow resolution
        sx = flow_w / fw
        sy = flow_h / fh
        fy1 = int(y1 * sy)
        fy2 = int(y2 * sy)
        fx1 = int(x1 * sx)
        fx2 = int(x2 * sx)
        if fy2 <= fy1 or fx2 <= fx1:
            return flow
        return flow[fy1:fy2, fx1:fx2]

    @staticmethod
    def _flow_divergence(flow: np.ndarray) -> float:
        """Compute mean divergence (∂u/∂x + ∂v/∂y) of a flow patch."""
        if flow.ndim != 3 or flow.shape[2] < 2:
            return 0.0
        if flow.shape[0] < 2 or flow.shape[1] < 2:
            return 0.0
        du_dx = np.diff(flow[..., 0], axis=1)   # Δu/Δx  (H × W-1)
        dv_dy = np.diff(flow[..., 1], axis=0)   # Δv/Δy  (H-1 × W)
        # Use the overlapping sub-region
        min_h = min(du_dx.shape[0], dv_dy.shape[0])
        min_w = min(du_dx.shape[1], dv_dy.shape[1])
        divergence = du_dx[:min_h, :min_w] + dv_dy[:min_h, :min_w]
        return float(np.mean(divergence))


__all__ = ["FlowFeatureExtractor"]
