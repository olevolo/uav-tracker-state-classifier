"""Henriques 2015 KCF reference tracker (Phase 8 / PLAN §3.2.A fallback).

Port of the Kernelized Correlation Filter tracker described in:

  João F. Henriques, Rui Caseiro, Pedro Martins, Jorge Batista,
  "High-Speed Tracking with Kernelized Correlation Filters",
  IEEE Transactions on Pattern Analysis and Machine Intelligence
  (TPAMI), 2015.  DOI: 10.1109/TPAMI.2014.2345390

This implementation follows the paper's Algorithm 1 / equations closely,
using pure NumPy + cv2.dft / cv2.idft (which outperform np.fft on ARM/AVX2).

== Feature choice ==
  Grayscale-HOG cell features (paper's preferred config, §4.1).
  Each cell is a 4×4-pixel block; HOG bins = 9 (orientations 0-180 degrees,
  unsigned gradient).  We use the paper's original dense-cell HOG (no block
  normalisation, only per-cell L2-norm) to match the Matlab reference code.
  Cells are arranged spatially into a (nH x nW x 9) array processed as 9
  independent frequency-domain channels (multi-channel extension, §3.4).

  Set cell_size=1 to switch to raw grayscale (single channel).  This is
  ~2x faster but gives ~0.05 lower AUC; use for ablations only.

== Scale pyramid ==
  3 scales: {1/scale_step, 1.0, scale_step} where scale_step=1.05.
  Off-centre scales are penalised by scale_penalty=0.95.  This is a
  standard addition used in all major KCF derivatives (not in the 2015
  paper itself but required to match UAV123 reported numbers).

== Ego-motion compensation (UAV extension — opt-in) ==
  Before each correlation search, background Shi-Tomasi corners (excluding
  the target bbox) are tracked with pyramidal LK.  RANSAC homography fitted
  to inliers estimates pure camera motion.  The current search centre is
  warped through the homography so the pyramid searches where the target
  would appear after camera motion.  Only applied when median inlier
  displacement exceeds 2 px (to avoid noisy corrections on stable cameras).

  **Default: disabled (use_ego_motion=False).** Enable when the camera is NOT
  following the target (e.g. fixed-mount sensors, surveillance UAVs scanning
  an area).  When the camera actively tracks the target (operator keeps target
  in frame), ego-motion compensation is counterproductive — the target stays
  approximately fixed in image coordinates while the background moves, so the
  homography displacement applied to the target center overshoots.

== Peak-response failure detection ==
  If peak_response < min_peak_ratio * init_peak, the model update is skipped
  for that frame so the template is not corrupted by a bad detection.
  Default raised to 0.25 (vs paper's implicit 0) to prevent template drift
  on occluded frames.

== Response map in aux ==
  ``TrackState.aux["response_map"]`` carries the 2-D correlation response
  (float64, shape (nH, nW)) so downstream signals (e.g. APCESignal) can
  compute authentic APCE without re-running the correlation.

Tier hint: 0  (same as kcf_kalman; lightest tier).
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from uav_tracker.registry import TRACKERS
from uav_tracker.types import BBox, TrackState

# Lazy imports for ego-motion helpers — keep tracker importable without signals.
def _get_flow_helpers():
    from uav_tracker.signals.optical_flow import detect_corners, track_flow
    return detect_corners, track_flow

# ---------------------------------------------------------------------------
# HOG feature extraction helpers (grayscale cells, paper §4.1)
# ---------------------------------------------------------------------------

_BINS = 9  # unsigned orientation bins 0-180 degrees


def _compute_hog_features(patch: np.ndarray, cell_size: int) -> np.ndarray:
    """Extract HOG features from a grayscale *patch*.

    Parameters
    ----------
    patch : np.ndarray
        float32 grayscale patch (H, W).  H and W must be multiples of
        *cell_size*.
    cell_size : int
        Pixels per cell (paper uses 4).

    Returns
    -------
    feats : np.ndarray
        float64 array of shape ``(H//cell_size, W//cell_size, _BINS)``.
    """
    H, W = patch.shape
    nH = H // cell_size
    nW = W // cell_size

    img = patch.astype(np.float32)

    # Finite-difference gradients (matches paper's Matlab reference code).
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    gx[:, 0] = img[:, 1] - img[:, 0]
    gx[:, -1] = img[:, -1] - img[:, -2]
    gy[0, :] = img[1, :] - img[0, :]
    gy[-1, :] = img[-1, :] - img[-2, :]

    mag = np.sqrt(gx * gx + gy * gy)
    # Unsigned orientation in [0, 180).
    ori = np.arctan2(np.abs(gy), gx) * (180.0 / math.pi)
    ori = np.clip(ori, 0.0, 179.999)

    bin_width = 180.0 / _BINS
    bin_idx = (ori / bin_width).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, _BINS - 1)

    feats = np.zeros((nH, nW, _BINS), dtype=np.float64)
    for b in range(_BINS):
        contrib = np.where(bin_idx == b, mag, 0.0)
        # Accumulate into cells.
        cell_sum = contrib[: nH * cell_size, : nW * cell_size].reshape(
            nH, cell_size, nW, cell_size
        ).sum(axis=(1, 3))
        feats[:, :, b] = cell_sum

    # Per-cell L2 normalisation.
    norm = np.linalg.norm(feats, axis=2, keepdims=True) + 1e-6
    feats /= norm

    return feats


# ---------------------------------------------------------------------------
# FFT helpers (cv2.dft is faster than np.fft on x86/ARM with SIMD)
# ---------------------------------------------------------------------------

def _fft2(x: np.ndarray) -> np.ndarray:
    """2-D DFT of a real 2-D array; returns complex128."""
    f32 = x.astype(np.float32)
    dft = cv2.dft(f32, flags=cv2.DFT_COMPLEX_OUTPUT)
    return (dft[:, :, 0] + 1j * dft[:, :, 1]).astype(np.complex128)


def _ifft2_real(x_complex: np.ndarray) -> np.ndarray:
    """Inverse 2-D DFT; returns float64 (real part only)."""
    src = np.stack(
        [x_complex.real.astype(np.float32), x_complex.imag.astype(np.float32)],
        axis=2,
    )
    idft = cv2.idft(src, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    return idft.astype(np.float64)


# ---------------------------------------------------------------------------
# Gaussian RBF kernel correlation  (paper eq. 16)
# ---------------------------------------------------------------------------

def _gaussian_correlation(
    xf: np.ndarray, zf: np.ndarray, sigma: float
) -> np.ndarray:
    """Compute the DFT of the Gaussian correlation kernel map.

    Implements Henriques 2015 equation 16:
      k(x, z) = exp( -(||x||^2 + ||z||^2 - 2*IFFT(sum_c conj(X_c)·Z_c))
                     / (sigma^2 * N) )

    Parameters
    ----------
    xf : np.ndarray
        DFT of template features, shape (H, W, C) complex.
    zf : np.ndarray
        DFT of search features, shape (H, W, C) complex.
    sigma : float
        RBF bandwidth.

    Returns
    -------
    kf : np.ndarray
        DFT of the kernel map, shape (H, W) complex.
    """
    H, W, C = xf.shape
    N = float(H * W)

    # Parseval: ||x||^2 = sum|X_c|^2 / N
    xx = float(np.sum(np.abs(xf) ** 2)) / N
    zz = float(np.sum(np.abs(zf) ** 2)) / N

    # Cross-correlation: IFFT(sum_c conj(X_c) * Z_c)
    cross = np.sum(np.conj(xf) * zf, axis=2)  # (H, W) complex
    xy = _ifft2_real(cross)                     # (H, W) real

    k = np.exp(-(xx + zz - 2.0 * xy) / (sigma ** 2 * N + 1e-9))
    k = np.clip(k, 0.0, 1.0)
    return _fft2(k)


# ---------------------------------------------------------------------------
# Hann window
# ---------------------------------------------------------------------------

def _hann_window(H: int, W: int) -> np.ndarray:
    """2-D separable Hann window of shape (H, W), dtype float64."""
    h1d = np.hanning(H).astype(np.float64)
    w1d = np.hanning(W).astype(np.float64)
    return np.outer(h1d, w1d)


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

@TRACKERS.register("kcf_henriques")
class KCFHenriques2015Tracker:
    """Henriques 2015 KCF reference implementation.

    Kernelized Correlation Filter with:
      - Multi-channel HOG features (cell_size=4, 9 unsigned orientation bins).
      - Gaussian RBF kernel (sigma=0.5; paper's preferred value).
      - Linear interpolation update (eta=0.075 for HOG).
      - 3-scale pyramid (scale_step=1.05, penalty=0.95).
      - Padding factor 2.2x the target size (slightly below paper's 2.5 for
        better localisation on UAV-scale targets).
      - Target Gaussian response with output_sigma_factor=0.1.
      - Ridge regression regularisation lambda=1e-4.
      - Peak-response failure detection (skip update below threshold).

    Parameters
    ----------
    sigma : float
        Gaussian kernel bandwidth (RBF sigma; default 0.5, paper value).
    lambda_ : float
        Ridge regularisation lambda (paper default 1e-4).
    interp_factor : float
        Appearance-model update interpolation eta (0.075 for HOG).
    padding : float
        Search window inflation factor relative to target size (2.2).
    output_sigma_factor : float
        Target response Gaussian bandwidth as fraction of sqrt(feat_W*feat_H).
    cell_size : int
        Pixels per HOG cell.  Set to 1 to use raw grayscale (faster, lower AUC).
    scale_step : float
        Scale pyramid step (1.05).
    scale_penalty : float
        Response penalty on off-centre scales (0.95).
    min_peak_ratio : float
        If peak/init_peak < min_peak_ratio, skip model update this frame.
        Default 0.25 (raised from paper's implicit 0) prevents template
        corruption on occluded/lost frames.
    max_disp_factor : float
        Maximum per-frame displacement as a fraction of the search window size
        (default 0.5).  Clamps runaway drift on noisy / low-texture frames.
    scale_lr : float
        Low-pass learning rate for scale updates (1.0 = instantaneous, matching
        paper; values < 1.0 damp spurious scale drift on uniform backgrounds).
    use_ego_motion : bool
        If True, estimate camera homography from background Shi-Tomasi corners
        and shift the search centre to compensate.  Default **False** — only
        enable when the camera is NOT following the target (e.g. fixed-mount
        sensors or scanning UAVs).  When the operator keeps the target in
        frame, compensation is counterproductive (see module docstring).
    ego_corners : int
        Maximum Shi-Tomasi corners to track for ego-motion estimation (default
        150; more gives better homography but ~proportionally more compute).
    """

    name: str = "kcf_henriques"
    tier_hint: int = 0

    # ~0.04 GFLOPs/frame for HOG variant (2x kcf_kalman due to 9-channel FFT).
    _FLOPS_PER_UPDATE: float = 0.04 * 1e9

    def __init__(
        self,
        sigma: float = 0.5,
        lambda_: float = 1e-4,
        interp_factor: float = 0.075,
        padding: float = 2.2,
        output_sigma_factor: float = 0.1,
        cell_size: int = 4,
        scale_step: float = 1.05,
        scale_penalty: float = 0.95,
        min_peak_ratio: float = 0.25,
        max_disp_factor: float = 0.50,
        scale_lr: float = 1.0,
        use_ego_motion: bool = False,
        ego_corners: int = 150,
    ) -> None:
        self.sigma = sigma
        self.lambda_ = lambda_
        self.interp_factor = interp_factor
        self.padding = padding
        self.output_sigma_factor = output_sigma_factor
        self.cell_size = cell_size
        self.scale_step = scale_step
        self.scale_penalty = scale_penalty
        self.min_peak_ratio = min_peak_ratio
        self.max_disp_factor = max_disp_factor
        self.scale_lr = scale_lr
        self.use_ego_motion = use_ego_motion
        self.ego_corners = ego_corners

        # Model state (initialised in init()).
        self._xf: np.ndarray | None = None        # DFT of windowed template
        self._x_hann: np.ndarray | None = None    # windowed template (spatial)
        self._alphaf: np.ndarray | None = None    # DFT of filter coefficients
        self._yf: np.ndarray | None = None        # DFT of target response (constant)
        self._hann: np.ndarray | None = None      # Hann window for this window size

        self._bbox: BBox | None = None            # current bbox estimate
        self._sz: tuple[int, int] | None = None   # (patch_H, patch_W) pixels fixed at init
        self._scale: float = 1.0                  # scale relative to init bbox (accumulates)
        self._init_w: float = 0.0                 # initial bbox width (for scale-relative dims)
        self._init_h: float = 0.0                 # initial bbox height
        self._init_peak: float = 1.0              # peak response at init
        self._prev_frame: np.ndarray | None = None  # previous frame for ego-motion

    # ------------------------------------------------------------------
    # Internal geometry helpers
    # ------------------------------------------------------------------

    def _get_window_size(self, bbox: BBox) -> tuple[int, int]:
        """Return (H, W) in pixels for the padded search window."""
        cs = self.cell_size
        W = max(int(round(bbox.w * self.padding / cs)) * cs, 2 * cs)
        H = max(int(round(bbox.h * self.padding / cs)) * cs, 2 * cs)
        return H, W

    def _extract_patch(
        self,
        frame: np.ndarray,
        cx: float,
        cy: float,
        sz: tuple[int, int],
        scale: float = 1.0,
    ) -> np.ndarray:
        """Crop a centred grayscale patch from *frame*, resized to *sz*.

        *scale* stretches the region-of-interest in image-space so that
        the output patch still has shape *sz* (i.e. the output is always
        the canonical template size).
        """
        H, W = sz
        h_src = int(round(H * scale))
        w_src = int(round(W * scale))

        x1 = int(round(cx - w_src / 2.0))
        y1 = int(round(cy - h_src / 2.0))
        x2 = x1 + w_src
        y2 = y1 + h_src

        fH, fW = frame.shape[:2]

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - fW)
        pad_bottom = max(0, y2 - fH)

        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(fW, x2), min(fH, y2)

        # Guard against fully out-of-frame crop (e.g. target near border at
        # high scale): fall back to a mean-valued patch.
        if x2c <= x1c or y2c <= y1c:
            return np.full((H, W), 128.0, dtype=np.float32)

        crop = cv2.cvtColor(frame[y1c:y2c, x1c:x2c], cv2.COLOR_BGR2GRAY)
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cv2.copyMakeBorder(
                crop, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_REPLICATE,
            )

        if crop.shape != (H, W):
            crop = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

        return crop.astype(np.float32)

    def _features(self, patch: np.ndarray) -> np.ndarray:
        """Extract (nH, nW, C) features from a grayscale float32 patch."""
        if self.cell_size == 1:
            return (patch / 255.0)[:, :, np.newaxis]  # single-channel grayscale
        return _compute_hog_features(patch, self.cell_size)

    def _make_target_response(self, nH: int, nW: int) -> np.ndarray:
        """Gaussian label y centred at (nW/2, nH/2) with output_sigma_factor."""
        sigma_px = math.sqrt(nW * nH) * self.output_sigma_factor
        cy, cx = nH // 2, nW // 2
        yv, xv = np.mgrid[0:nH, 0:nW]
        dist_sq = (xv - cx) ** 2 + (yv - cy) ** 2
        return np.exp(-dist_sq / (2.0 * sigma_px ** 2))

    # ------------------------------------------------------------------
    # Ego-motion compensation (UAV extension)
    # ------------------------------------------------------------------

    def _predict_center_ego(
        self,
        cx: float,
        cy: float,
        curr_frame: np.ndarray,
    ) -> tuple[float, float]:
        """Return camera-motion-compensated search centre.

        Detects Shi-Tomasi corners in the *background only* (full frame with
        target bbox masked out), tracks them with LK, fits RANSAC homography,
        then warps (cx, cy) through it.  Falls back to raw (cx, cy) when
        tracking quality is too low.

        Using only background points is critical: ROI corners move with the
        target, not the camera, and would corrupt the homography estimate.
        """
        if self._prev_frame is None or self._bbox is None:
            return cx, cy

        try:
            _, track_flow = _get_flow_helpers()
        except ImportError:
            return cx, cy

        # Detect corners in the full frame with target region masked out.
        prev_gray = cv2.cvtColor(self._prev_frame, cv2.COLOR_BGR2GRAY) \
            if self._prev_frame.ndim == 3 else self._prev_frame
        fH, fW = prev_gray.shape
        mask = np.ones((fH, fW), dtype=np.uint8) * 255
        x0 = max(0, int(self._bbox.x))
        y0 = max(0, int(self._bbox.y))
        x1 = min(fW, int(self._bbox.x + self._bbox.w))
        y1 = min(fH, int(self._bbox.y + self._bbox.h))
        mask[y0:y1, x0:x1] = 0  # exclude target

        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=self.ego_corners,
            qualityLevel=0.01, minDistance=5, blockSize=7, mask=mask,
        )
        if prev_pts is None or len(prev_pts) < 8:
            return cx, cy

        curr_pts, status = track_flow(self._prev_frame, curr_frame, prev_pts)
        good_p = prev_pts[status == 1].reshape(-1, 1, 2)
        good_c = curr_pts[status == 1].reshape(-1, 1, 2)

        if len(good_p) < 8:
            return cx, cy

        H, mask_h = cv2.findHomography(good_p, good_c, cv2.RANSAC, 3.0)
        if H is None or int(mask_h.sum()) < 6:
            return cx, cy

        # Only apply when camera motion is significant (> 2 px median displacement).
        # On stable-camera frames the homography is noisy and should not shift the
        # search centre.
        inlier_p = good_p[mask_h.ravel() == 1]
        inlier_c = good_c[mask_h.ravel() == 1]
        if len(inlier_p) >= 4:
            disp = np.linalg.norm(inlier_c.reshape(-1, 2) - inlier_p.reshape(-1, 2), axis=1)
            median_disp = float(np.median(disp))
            if median_disp < 2.0:
                return cx, cy

        pt = np.array([[[cx, cy]]], dtype=np.float32)
        pt_new = cv2.perspectiveTransform(pt, H)
        return float(pt_new[0, 0, 0]), float(pt_new[0, 0, 1])

    # ------------------------------------------------------------------
    # Tracker Protocol — init
    # ------------------------------------------------------------------

    def init(self, frame: np.ndarray, bbox: BBox) -> None:
        """Initialise KCF on frame 0 with the ground-truth bbox."""
        self._bbox = bbox
        self._scale = 1.0
        self._init_w = bbox.w
        self._init_h = bbox.h
        self._sz = self._get_window_size(bbox)

        cx = bbox.x + bbox.w / 2.0
        cy = bbox.y + bbox.h / 2.0

        patch = self._extract_patch(frame, cx, cy, self._sz)
        feats = self._features(patch)  # (nH, nW, C)
        nH, nW, C = feats.shape

        self._hann = _hann_window(nH, nW)
        x_hann = feats * self._hann[:, :, np.newaxis]

        # Target response.
        y = self._make_target_response(nH, nW)
        self._yf = _fft2(y)

        # Build filter: alpha = yf / (kf + lambda).
        xf = np.stack([_fft2(x_hann[:, :, c]) for c in range(C)], axis=2)
        kf = _gaussian_correlation(xf, xf, self.sigma)
        alphaf = self._yf / (kf + self.lambda_)

        self._xf = xf
        self._x_hann = x_hann
        self._alphaf = alphaf

        # Record init peak for adaptive failure threshold.
        resp = self._response_map(xf, alphaf)
        self._init_peak = max(float(np.max(resp)), 1e-9)
        self._prev_frame = frame.copy()

    # ------------------------------------------------------------------
    # Tracker Protocol — update
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> TrackState:
        """Advance one frame; return updated TrackState."""
        if self._bbox is None or self._alphaf is None:
            raise RuntimeError("KCFHenriques2015Tracker.update called before init")

        cx = self._bbox.x + self._bbox.w / 2.0
        cy = self._bbox.y + self._bbox.h / 2.0

        # ------ Ego-motion compensation: shift search centre by camera motion ------
        if self.use_ego_motion:
            cx, cy = self._predict_center_ego(cx, cy, frame)

        # ------ Scale pyramid search ------
        best_score = -np.inf
        best_dy = 0.0
        best_dx = 0.0
        best_scale_factor = 1.0
        best_zf: np.ndarray = self._xf  # fallback (never used if loop runs)
        best_resp: np.ndarray | None = None  # response map for APCE signal

        scale_factors = [
            1.0 / self.scale_step,
            1.0,
            self.scale_step,
        ]
        for i, sf in enumerate(scale_factors):
            # The search region in image-space is sized by current scale * sf.
            patch = self._extract_patch(frame, cx, cy, self._sz, scale=self._scale * sf)
            feats = self._features(patch)
            nH, nW, C = feats.shape
            z_hann = feats * self._hann[:, :, np.newaxis]
            zf = np.stack([_fft2(z_hann[:, :, c]) for c in range(C)], axis=2)

            resp = self._response_map(zf, self._alphaf)
            peak_val = float(np.max(resp))

            # Apply penalty to off-centre scales.
            if i != 1:
                peak_val *= self.scale_penalty

            if peak_val > best_score:
                best_score = peak_val
                best_scale_factor = sf
                best_zf = zf
                best_dy, best_dx = self._sub_pixel_peak(resp)
                best_resp = resp  # save full response map for APCE signal

        # ------ Map feature-space displacement to image pixels ------
        # Displacement is in feature cells; scale back to image pixels using
        # the *effective* scale (current_scale * best_scale_factor).
        effective_scale = self._scale * best_scale_factor
        img_dx = best_dx * self.cell_size * effective_scale
        img_dy = best_dy * self.cell_size * effective_scale

        # Clamp maximum per-frame displacement to max_disp_factor * search size.
        # This prevents runaway drift on noisy/low-texture frames.
        sz_H, sz_W = self._sz
        max_dx = sz_W * self.max_disp_factor * effective_scale
        max_dy = sz_H * self.max_disp_factor * effective_scale
        img_dx = float(np.clip(img_dx, -max_dx, max_dx))
        img_dy = float(np.clip(img_dy, -max_dy, max_dy))

        cx_new = cx + img_dx
        cy_new = cy + img_dy

        # Accumulate scale via low-pass filter to damp noise-driven drift.
        # scale_lr=1.0 would be instantaneous (original KCF behaviour).
        # scale_lr < 1.0 smooths out spurious scale changes on uniform backgrounds.
        if best_scale_factor != 1.0:
            # Low-pass: blend current scale toward the winning candidate.
            self._scale = self._scale * (1.0 - self.scale_lr) + \
                          (self._scale * best_scale_factor) * self.scale_lr
        new_w = self._init_w * self._scale
        new_h = self._init_h * self._scale

        new_bbox = BBox(
            x=cx_new - new_w / 2.0,
            y=cy_new - new_h / 2.0,
            w=new_w,
            h=new_h,
        )
        self._bbox = new_bbox

        # ------ Peak response for confidence / failure detection ------
        raw_peak = float(np.max(self._response_map(best_zf, self._alphaf)))
        do_update = raw_peak >= self.min_peak_ratio * self._init_peak

        # ------ Model interpolation update (eq. 5 / 6) ------
        if do_update:
            patch_upd = self._extract_patch(frame, cx_new, cy_new, self._sz)
            feats_upd = self._features(patch_upd)
            nH, nW, C = feats_upd.shape
            x_hann_new = feats_upd * self._hann[:, :, np.newaxis]
            xf_new = np.stack(
                [_fft2(x_hann_new[:, :, c]) for c in range(C)], axis=2
            )
            kf_new = _gaussian_correlation(xf_new, xf_new, self.sigma)
            alphaf_new = self._yf / (kf_new + self.lambda_)

            eta = self.interp_factor
            self._x_hann = (1.0 - eta) * self._x_hann + eta * x_hann_new
            self._xf = (1.0 - eta) * self._xf + eta * xf_new
            self._alphaf = (1.0 - eta) * self._alphaf + eta * alphaf_new

        # ------ Confidence ------
        confidence = float(np.clip(raw_peak / self._init_peak, 0.0, 1.0))
        if confidence >= 0.6:
            status: str = "locked"
        elif confidence >= 0.15:
            status = "uncertain"
        else:
            status = "lost"

        self._prev_frame = frame.copy()

        return TrackState(
            bbox=new_bbox,
            confidence=confidence,
            status=status,
            aux={
                "peak_response": raw_peak,
                "peak_ratio": confidence,
                "response_map": best_resp,  # 2-D float64 array for authentic APCE
            },
        )

    # ------------------------------------------------------------------
    # Internal compute helpers
    # ------------------------------------------------------------------

    def _response_map(self, zf: np.ndarray, alphaf: np.ndarray) -> np.ndarray:
        """Compute spatial response map given search features DFT and filter."""
        kf_xz = _gaussian_correlation(self._xf, zf, self.sigma)
        return _ifft2_real(alphaf * kf_xz)

    @staticmethod
    def _sub_pixel_peak(response: np.ndarray) -> tuple[float, float]:
        """Sub-pixel peak localisation via 1-D quadratic interpolation.

        Returns (dy, dx) in *feature cells* relative to the image centre.
        """
        H, W = response.shape
        idx = int(np.argmax(response))
        ry, rx = divmod(idx, W)

        def _interp(arr: np.ndarray, i: int, n: int) -> float:
            if i == 0 or i == n - 1:
                return float(i)
            prev_ = float(arr[(i - 1) % n])
            cur_ = float(arr[i])
            nxt_ = float(arr[(i + 1) % n])
            denom = 2.0 * cur_ - prev_ - nxt_
            if abs(denom) < 1e-9:
                return float(i)
            return i + 0.5 * (prev_ - nxt_) / denom

        ry_sub = _interp(response[:, rx], ry, H)
        rx_sub = _interp(response[ry, :], rx, W)

        # Displacement from image centre (circular shift convention).
        dy = ry_sub - H // 2
        dx = rx_sub - W // 2
        if dy > H / 2:
            dy -= H
        if dx > W / 2:
            dx -= W

        return dy, dx

    # ------------------------------------------------------------------
    # Tracker Protocol — remaining mandatory/optional methods
    # ------------------------------------------------------------------

    def flops_per_update(self) -> float:
        """Static FLOPs estimate (~0.04 GFLOPs/frame for HOG variant)."""
        return self._FLOPS_PER_UPDATE

    def on_tier_enter(self, ctx: Any) -> None:
        """Runner hook — no-op (KCF is stateless w.r.t. tier transitions)."""

    def on_tier_exit(self, ctx: Any) -> None:
        """Runner hook — no-op."""

    def reset(self) -> None:
        """Reset to uninitialised state (used between sequences)."""
        self._xf = None
        self._x_hann = None
        self._alphaf = None
        self._yf = None
        self._hann = None
        self._bbox = None
        self._sz = None
        self._scale = 1.0
        self._init_w = 0.0
        self._init_h = 0.0
        self._init_peak = 1.0
        self._prev_frame = None
