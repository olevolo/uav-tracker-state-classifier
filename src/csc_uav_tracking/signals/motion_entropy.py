"""MotionEntropySignal — paper's core switching signal (Phase 4).

Implements the motion-entropy signal from Oleksiuk & Velhosh (2026),
§3.2.  Residual optical flow after global-motion subtraction is
binned into a 16-bin magnitude-weighted orientation histogram, whose
Shannon entropy is normalized and EMA-smoothed.

Registration key: ``"motion_entropy"``
Range: [0.0, 1.0]

Module-level helpers (consumed by property tests under
tests/property/test_entropy_math.py):
  - ``shannon_entropy(p)``   — pure function over probability vector.
  - ``normalize_entropy(H, N)`` — pure normalizer.
"""

from __future__ import annotations

import math

import numpy as np

from csc_uav_tracking.registry import SIGNALS
from csc_uav_tracking.types import FrameContext, SignalReport, TrackState

from .global_motion import estimate_global_flow
from .optical_flow import _to_gray, track_flow

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Pure math helpers (exported for property tests)                             #
# --------------------------------------------------------------------------- #


def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy over a probability vector.

    Parameters
    ----------
    p:
        1-D array of non-negative values that must sum to 1 (i.e. an
        already-normalized probability vector). Zero bins are handled
        safely via ``xlogy`` convention.

    Returns
    -------
    float — ``H = -Σ p_i log2(p_i)`` in bits (range [0, log2(len(p))]).
    """
    p = np.asarray(p, dtype=np.float64)
    nonzero = p[p > 0.0]
    if len(nonzero) == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log2(nonzero)))


def normalize_entropy(H: float, N: int) -> float:
    """Normalize Shannon entropy to [0, 1].

    Parameters
    ----------
    H:
        Shannon entropy value (bits).
    N:
        Number of histogram bins (≥ 2 so log2(N) > 0).

    Returns
    -------
    float — ``H̃ = H / log2(N)`` clipped to [0, 1].
    """
    if N < 2:
        return 0.0
    h_norm = H / math.log2(N)
    return float(np.clip(h_norm, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Internal helpers                                                             #
# --------------------------------------------------------------------------- #


def _detect_roi_and_bg_corners(
    frame: np.ndarray,
    roi_bbox: tuple[float, float, float, float],
    background_band: int,
    max_corners: int,
    quality_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect Shi-Tomasi corners separately for ROI and background band.

    Returns
    -------
    (roi_pts, bg_pts) — each is (N, 1, 2) float32, may be empty.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("cv2 is required")

    gray = _to_gray(frame)
    h, w = gray.shape
    x, y, bw, bh = [int(round(v)) for v in roi_bbox]

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + bw)
    y1 = min(h, y + bh)

    bx0 = max(0, x0 - background_band)
    by0 = max(0, y0 - background_band)
    bx1 = min(w, x1 + background_band)
    by1 = min(h, y1 + background_band)

    half = max(1, max_corners // 2)
    shi_params = dict(qualityLevel=quality_level, minDistance=3, blockSize=7)

    # ROI corners
    roi_pts_list = []
    if x1 > x0 and y1 > y0:
        roi_gray = gray[y0:y1, x0:x1]
        pts = _cv2.goodFeaturesToTrack(roi_gray, maxCorners=half, **shi_params)
        if pts is not None:
            pts = pts.astype(np.float32)
            pts[:, :, 0] += x0
            pts[:, :, 1] += y0
            roi_pts_list.append(pts)

    # Background band corners (mask out ROI)
    bg_pts_list = []
    if bx1 > bx0 and by1 > by0:
        bg_gray = gray[by0:by1, bx0:bx1]
        mask = np.ones_like(bg_gray, dtype=np.uint8) * 255
        rx0 = x0 - bx0
        ry0 = y0 - by0
        rx1 = x1 - bx0
        ry1 = y1 - by0
        if rx1 > rx0 and ry1 > ry0:
            mask[max(0, ry0):min(by1 - by0, ry1),
                 max(0, rx0):min(bx1 - bx0, rx1)] = 0
        bg_pts = _cv2.goodFeaturesToTrack(bg_gray, maxCorners=half, mask=mask, **shi_params)
        if bg_pts is not None:
            bg_pts = bg_pts.astype(np.float32)
            bg_pts[:, :, 0] += bx0
            bg_pts[:, :, 1] += by0
            bg_pts_list.append(bg_pts)

    roi_out = np.concatenate(roi_pts_list, axis=0) if roi_pts_list else np.empty((0, 1, 2), dtype=np.float32)
    bg_out = np.concatenate(bg_pts_list, axis=0) if bg_pts_list else np.empty((0, 1, 2), dtype=np.float32)
    return roi_out, bg_out


def _apply_homography_displacement(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Project points through H and return the displacement (projected - original)."""
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    projected = _cv2.perspectiveTransform(pts, H)
    return (projected - pts).reshape(-1, 1, 2)


# --------------------------------------------------------------------------- #
# Signal implementation                                                        #
# --------------------------------------------------------------------------- #


@SIGNALS.register("motion_entropy")
class MotionEntropySignal:
    """Motion-entropy switching signal.

    Computes a normalized, EMA-smoothed residual-flow entropy per frame
    following the paper's pipeline (PLAN §3.2):

        Shi-Tomasi detection (ROI + bg) → LK tracking → RANSAC/LMedS
        global-motion from BACKGROUND points only → subtract from ROI
        flow → magnitude-weighted orientation histogram → Shannon entropy
        → normalization → EMA.

    Parameters
    ----------
    n_bins:
        Number of orientation histogram bins (paper default 16).
    alpha:
        EMA smoothing factor; ``H̄_t = α·H̄_{t-1} + (1-α)·H̃_t``
        (paper default 0.8).
    mag_threshold:
        Minimum residual-flow magnitude in pixels; vectors below this
        are excluded from the histogram (paper default 1.0 px).
    max_corners:
        Shi-Tomasi ``maxCorners`` (paper default 200).
    quality_level:
        Shi-Tomasi ``qualityLevel`` (paper default 0.01).
    background_band:
        Pixel width of the background band around the ROI used for
        global-motion estimation (paper default 20).
    seed:
        RNG seed passed to ``cv2.setRNGSeed`` once at construction for
        RANSAC reproducibility.
    """

    name: str = "motion_entropy"
    range: tuple[float, float] = (0.0, 1.0)

    def __init__(
        self,
        n_bins: int = 16,
        alpha: float = 0.8,
        mag_threshold: float = 1.0,
        max_corners: int = 200,
        quality_level: float = 0.01,
        background_band: int = 20,
        seed: int = 42,
    ) -> None:
        self.n_bins = n_bins
        self.alpha = alpha
        self.mag_threshold = mag_threshold
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.background_band = background_band
        self.seed = seed

        # Seed OpenCV's RNG once at construction for RANSAC reproducibility.
        if _CV2_AVAILABLE:
            _cv2.setRNGSeed(seed)

        # Internal EMA history.
        self._H_bar: float = 0.0
        # Previous-frame data for LK tracking (ROI pts and bg pts stored together
        # but we track which index range belongs to bg for global-motion fitting).
        self._prev_frame: np.ndarray | None = None
        self._prev_roi_pts: np.ndarray | None = None   # (N_roi, 1, 2)
        self._prev_bg_pts: np.ndarray | None = None    # (N_bg, 1, 2)
        # Prior global homography for ADR-0006 reuse fallback.
        self._prior_H: np.ndarray | None = None
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # SwitchSignal Protocol
    # ------------------------------------------------------------------

    def step(self, ctx: FrameContext, state: TrackState | None) -> SignalReport:
        """Compute motion-entropy signal for the current frame.

        Parameters
        ----------
        ctx:
            ``FrameContext`` with at minimum ``frame``, ``prev_frame``,
            ``frame_idx``, and ``bbox`` populated.
        state:
            Latest ``TrackState`` from the active tracker (``None`` at
            frame 0 before init).

        Returns
        -------
        SignalReport
            ``value`` — EMA-smoothed normalized entropy ``H̄ ∈ [0, 1]``.
            ``reliable`` — ``False`` when global-motion estimation falls
            back to a reused prior or when fewer than 4 tracked points are
            available.
            ``aux`` — ``{"H_raw": H, "H_norm": H̃, "residual_entropy": H̃,
            "global_flow_method": "ransac"|"lmeds"|"reused"|"failed"}``.
        """
        frame = ctx.frame
        prev_frame = ctx.prev_frame
        bbox = ctx.bbox

        _unreliable_aux = {
            "H_raw": 0.0,
            "H_norm": 0.0,
            "residual_entropy": 0.0,
            "global_flow_method": "failed",
        }

        # ----- frame 0 or no previous frame: initialize and return ------
        if prev_frame is None or not self._initialized:
            self._prev_frame = frame.copy()
            if bbox is not None:
                roi = (bbox.x, bbox.y, bbox.w, bbox.h)
                roi_pts, bg_pts = _detect_roi_and_bg_corners(
                    frame, roi,
                    background_band=self.background_band,
                    max_corners=self.max_corners,
                    quality_level=self.quality_level,
                )
                self._prev_roi_pts = roi_pts
                self._prev_bg_pts = bg_pts
            else:
                self._prev_roi_pts = None
                self._prev_bg_pts = None
            self._initialized = True
            return SignalReport(value=self._H_bar, reliable=False, aux=_unreliable_aux)

        # ----- need a bbox for ROI-based processing ----------------------
        if bbox is None:
            return SignalReport(value=self._H_bar, reliable=False, aux=_unreliable_aux)

        roi = (bbox.x, bbox.y, bbox.w, bbox.h)

        # ----- detect fresh corners if none available --------------------
        if self._prev_roi_pts is None or len(self._prev_roi_pts) == 0:
            roi_pts, bg_pts = _detect_roi_and_bg_corners(
                self._prev_frame, roi,
                background_band=self.background_band,
                max_corners=self.max_corners,
                quality_level=self.quality_level,
            )
            self._prev_roi_pts = roi_pts
            self._prev_bg_pts = bg_pts

        # ----- LK flow for ROI points ------------------------------------
        roi_curr, roi_status = track_flow(
            self._prev_frame, frame, self._prev_roi_pts
        ) if self._prev_roi_pts is not None and len(self._prev_roi_pts) > 0 else (
            np.empty((0, 1, 2), dtype=np.float32),
            np.empty((0,), dtype=np.uint8),
        )

        # ----- LK flow for background points (for global motion) ---------
        bg_curr, bg_status = track_flow(
            self._prev_frame, frame, self._prev_bg_pts
        ) if self._prev_bg_pts is not None and len(self._prev_bg_pts) > 0 else (
            np.empty((0, 1, 2), dtype=np.float32),
            np.empty((0,), dtype=np.uint8),
        )

        # Good bg tracks only
        bg_good_mask = bg_status == 1
        bg_prev_good = self._prev_bg_pts[bg_good_mask] if self._prev_bg_pts is not None else np.empty((0, 1, 2), dtype=np.float32)
        bg_curr_good = bg_curr[bg_good_mask]

        # Good ROI tracks only
        roi_good_mask = roi_status == 1
        roi_prev_good = self._prev_roi_pts[roi_good_mask] if self._prev_roi_pts is not None else np.empty((0, 1, 2), dtype=np.float32)
        roi_curr_good = roi_curr[roi_good_mask]

        if len(roi_prev_good) == 0:
            # No ROI tracks — can't compute entropy, update state
            self._prev_frame = frame.copy()
            roi_pts, bg_pts = _detect_roi_and_bg_corners(
                frame, roi,
                background_band=self.background_band,
                max_corners=self.max_corners,
                quality_level=self.quality_level,
            )
            self._prev_roi_pts = roi_pts
            self._prev_bg_pts = bg_pts
            return SignalReport(value=self._H_bar, reliable=False, aux=_unreliable_aux)

        # Local ROI flow vectors
        local_flow = (roi_curr_good - roi_prev_good).reshape(-1, 2)  # (N, 2)

        # ----- global-motion from background points only -----------------
        reliable_signal = True
        method = "failed"
        H: np.ndarray | None = None  # homography matrix

        if len(bg_prev_good) >= 4:
            # Estimate homography from bg pts
            bg_src = bg_prev_good.reshape(-1, 1, 2).astype(np.float32)
            bg_dst = bg_curr_good.reshape(-1, 1, 2).astype(np.float32)

            try:
                H_mat, mask = _cv2.findHomography(
                    bg_src, bg_dst,
                    method=_cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                )
                if H_mat is not None and mask is not None and int(mask.sum()) >= 4:
                    H = H_mat
                    method = "ransac"
            except _cv2.error:
                pass

            if H is None:
                try:
                    M, _ = _cv2.estimateAffinePartial2D(
                        bg_src.reshape(-1, 2),
                        bg_dst.reshape(-1, 2),
                        method=_cv2.LMEDS,
                    )
                    if M is not None:
                        H = np.eye(3, dtype=np.float64)
                        H[:2, :] = M
                        method = "lmeds"
                except _cv2.error:
                    pass

        if H is not None:
            # Project ROI prev points through H → get global displacement for each
            roi_prev_pts = roi_prev_good.reshape(-1, 1, 2).astype(np.float32)
            global_disp = _apply_homography_displacement(H, roi_prev_pts).reshape(-1, 2)
            residual_flow = local_flow - global_disp
            self._prior_H = H
        elif self._prior_H is not None:
            # ADR-0006 level 3: reuse prior H
            roi_prev_pts = roi_prev_good.reshape(-1, 1, 2).astype(np.float32)
            global_disp = _apply_homography_displacement(self._prior_H, roi_prev_pts).reshape(-1, 2)
            residual_flow = local_flow - global_disp
            method = "reused"
            reliable_signal = False
        else:
            # No estimate at all — use raw local flow
            residual_flow = local_flow
            method = "failed"
            reliable_signal = False

        # ----- magnitude-weighted orientation histogram ------------------
        magnitudes = np.linalg.norm(residual_flow, axis=1)  # (N,)
        above_threshold = magnitudes >= self.mag_threshold

        if not np.any(above_threshold):
            H_raw = 0.0
            H_norm = 0.0
        else:
            mags_thresh = magnitudes[above_threshold]
            vecs_thresh = residual_flow[above_threshold]

            # Orientation angles → bin into [0, 2*pi)
            angles = np.arctan2(vecs_thresh[:, 1], vecs_thresh[:, 0])
            angles = angles % (2 * np.pi)

            bin_edges = np.linspace(0, 2 * np.pi, self.n_bins + 1)
            hist, _ = np.histogram(angles, bins=bin_edges, weights=mags_thresh)

            total_weight = hist.sum()
            if total_weight <= 0.0:
                H_raw = 0.0
                H_norm = 0.0
            else:
                p = hist / total_weight
                H_raw = shannon_entropy(p)
                H_norm = normalize_entropy(H_raw, self.n_bins)

        # ----- EMA -------------------------------------------------------
        self._H_bar = self.alpha * self._H_bar + (1.0 - self.alpha) * H_norm
        self._H_bar = float(np.clip(self._H_bar, 0.0, 1.0))

        # ----- update frame state ----------------------------------------
        self._prev_frame = frame.copy()
        roi_pts, bg_pts = _detect_roi_and_bg_corners(
            frame, roi,
            background_band=self.background_band,
            max_corners=self.max_corners,
            quality_level=self.quality_level,
        )
        self._prev_roi_pts = roi_pts
        self._prev_bg_pts = bg_pts

        return SignalReport(
            value=self._H_bar,
            reliable=reliable_signal,
            aux={
                "H_raw": H_raw,
                "H_norm": H_norm,
                "residual_entropy": H_norm,
                "global_flow_method": method,
            },
        )

    def reset(self) -> None:
        """Restore signal to construction state. Idempotent."""
        self._H_bar = 0.0
        self._prev_frame = None
        self._prev_roi_pts = None
        self._prev_bg_pts = None
        self._prior_H = None
        self._initialized = False


__all__ = ["MotionEntropySignal", "shannon_entropy", "normalize_entropy"]
