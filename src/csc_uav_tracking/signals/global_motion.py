"""Global-motion estimation utility (Phase 4).

ADR-0006 "Global-motion fallback strategy" — three levels:
    1. RANSAC homography on background keypoints.
    2. If RANSAC fails → LMedS affine.
    3. If LMedS also fails → reuse previous estimate, mark
       ``reliable=False`` so the scheduler knows to hold state.

Public API
----------
``estimate_global_flow(prev_points, curr_points, prev_frame_shape)``
    -> ``tuple[np.ndarray | None, str]``

The first element is the per-point global displacement array of the same
shape as the input points (or ``None`` when neither RANSAC nor LMedS
succeeds and there is no prior to fall back to).  The second element is
the method string: ``"ransac"``, ``"lmeds"``, ``"reused"``, or
``"failed"``.

This is a shared utility, not a ``SwitchSignal`` itself.
``MotionEntropy`` (Phase 4) calls it every frame to subtract ego-motion
from the local flow field before computing the histogram.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False

# Minimum inlier count for RANSAC homography to be accepted.
_MIN_RANSAC_INLIERS = 4


def _apply_homography_to_points(
    H: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Project ``points`` through homography ``H``, return displacements.

    Parameters
    ----------
    H:
        3×3 homography matrix.
    points:
        ``(N, 1, 2)`` float32 or ``(N, 2)`` float32 point array.

    Returns
    -------
    np.ndarray of shape ``(N, 1, 2)`` float32 — per-point displacement
    vectors (projected position minus original position).
    """
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(pts, H)
    displacement = projected - pts
    return displacement.reshape(-1, 1, 2)


def _apply_affine_to_points(
    M: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Project ``points`` through affine matrix ``M``, return displacements.

    Parameters
    ----------
    M:
        2×3 or 3×3 affine matrix.
    points:
        ``(N, 1, 2)`` or ``(N, 2)`` float32.

    Returns
    -------
    np.ndarray of shape ``(N, 1, 2)`` float32 — per-point displacement.
    """
    if M.shape[0] == 2:
        # 2×3 affine — lift to 3×3
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = M
    else:
        H = M.astype(np.float64)
    return _apply_homography_to_points(H, points)


def estimate_global_flow(
    prev_points: np.ndarray,
    curr_points: np.ndarray,
    prev_frame_shape: tuple[int, int],
    _prior_state: dict | None = None,
) -> tuple[np.ndarray | None, str]:
    """Estimate per-point global (ego-motion) displacement.

    Uses the background keypoints (supplied as ``prev_points`` /
    ``curr_points``) to fit a camera-motion model, then evaluates that
    model at every input point to get a displacement field.

    Fallback chain (ADR-0006):
      1. RANSAC homography — if ≥ ``_MIN_RANSAC_INLIERS`` inliers.
      2. LMedS affine (``estimateAffinePartial2D``) — if RANSAC fails.
      3. Return ``(None, "failed")`` — both methods failed.

    The caller (``MotionEntropySignal``) is responsible for caching a
    prior estimate and implementing the "reused" fallback, since this
    function is stateless.

    Parameters
    ----------
    prev_points:
        ``(N, 1, 2)`` float32 — keypoint positions in the previous frame.
    curr_points:
        ``(N, 1, 2)`` float32 — corresponding positions in the current
        frame (from LK tracking, with ``status == 1``).
    prev_frame_shape:
        ``(height, width)`` of the previous frame — used only for
        validation / reference; not used in computation.
    _prior_state:
        Unused here; kept for interface symmetry with callers that track
        a cached prior.

    Returns
    -------
    (displacement, method):
        ``displacement`` — ``(N, 1, 2)`` float32 of per-point ego-motion
        vectors, or ``None`` if estimation failed.
        ``method`` — ``"ransac"``, ``"lmeds"``, or ``"failed"``.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("cv2 is required for global motion estimation")

    if prev_points is None or curr_points is None:
        return None, "failed"

    src = prev_points.reshape(-1, 1, 2).astype(np.float32)
    dst = curr_points.reshape(-1, 1, 2).astype(np.float32)

    if len(src) < 4:
        return None, "failed"

    # ------------------------------------------------------------------ #
    # Level 1: RANSAC homography                                          #
    # ------------------------------------------------------------------ #
    try:
        H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is not None and mask is not None:
            n_inliers = int(mask.sum())
            if n_inliers >= _MIN_RANSAC_INLIERS:
                displacement = _apply_homography_to_points(H, src)
                return displacement, "ransac"
    except cv2.error:
        pass

    # ------------------------------------------------------------------ #
    # Level 2: LMedS partial affine                                       #
    # ------------------------------------------------------------------ #
    try:
        M, inliers = cv2.estimateAffinePartial2D(
            src.reshape(-1, 2),
            dst.reshape(-1, 2),
            method=cv2.LMEDS,
        )
        if M is not None:
            displacement = _apply_affine_to_points(M, src)
            return displacement, "lmeds"
    except cv2.error:
        pass

    return None, "failed"


__all__ = ["estimate_global_flow"]
