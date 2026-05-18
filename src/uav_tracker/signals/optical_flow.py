"""Lucas-Kanade optical-flow utility (Phase 4).

Shared between ``MotionEntropy`` and ``FlowDivergence`` signals. Paper
§3.2: Shi-Tomasi corners + pyramidal LK with maxCorners=200,
qualityLevel=0.01, winSize=(15, 15).
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CV2_AVAILABLE = False


# LK defaults matching paper §3.2
_LK_WIN_SIZE = (15, 15)
_LK_MAX_LEVEL = 3
_LK_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    30,
    0.03,
) if _CV2_AVAILABLE else None


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert BGR or grayscale frame to single-channel uint8."""
    if not _CV2_AVAILABLE:
        raise RuntimeError("cv2 is required for optical flow")
    if frame.ndim == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def detect_corners(
    frame: np.ndarray,
    roi_bbox: tuple[float, float, float, float],
    background_band: int = 20,
    max_corners: int = 200,
    quality_level: float = 0.01,
) -> np.ndarray:
    """Detect Shi-Tomasi corners inside ROI and in background band around it.

    Corners are detected separately in:
      1. The ROI region defined by *roi_bbox*.
      2. A background band of *background_band* pixels around the ROI.

    Both sets are combined and returned as a single (N, 1, 2) float32 array.

    Parameters
    ----------
    frame:
        BGR or grayscale frame, uint8.
    roi_bbox:
        ``(x, y, w, h)`` of the region-of-interest in pixel coordinates.
    background_band:
        Width (pixels) of the background band surrounding the ROI.
    max_corners:
        Maximum number of Shi-Tomasi corners (divided evenly between ROI
        and background band).
    quality_level:
        Shi-Tomasi quality level threshold (paper default 0.01).

    Returns
    -------
    np.ndarray of shape (N, 1, 2) float32. May be empty ((0, 1, 2)).
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("cv2 is required for corner detection")

    gray = _to_gray(frame)
    h, w = gray.shape
    x, y, bw, bh = [int(round(v)) for v in roi_bbox]

    # Clamp ROI to frame
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w, x + bw)
    y1 = min(h, y + bh)

    # Background band bounding box (outer rect minus inner ROI)
    bx0 = max(0, x0 - background_band)
    by0 = max(0, y0 - background_band)
    bx1 = min(w, x1 + background_band)
    by1 = min(h, y1 + background_band)

    corners_per_region = max(1, max_corners // 2)
    all_corners: list[np.ndarray] = []

    shi_params = dict(
        maxCorners=corners_per_region,
        qualityLevel=quality_level,
        minDistance=3,
        blockSize=7,
    )

    # ROI corners
    if x1 > x0 and y1 > y0:
        roi_gray = gray[y0:y1, x0:x1]
        pts = cv2.goodFeaturesToTrack(roi_gray, **shi_params)
        if pts is not None:
            pts = pts.astype(np.float32)
            pts[:, :, 0] += x0
            pts[:, :, 1] += y0
            all_corners.append(pts)

    # Background band corners — use a mask that excludes the ROI
    if bx1 > bx0 and by1 > by0:
        bg_gray = gray[by0:by1, bx0:bx1].copy()
        # Mask out the ROI interior so we only get background flow
        mask = np.ones_like(bg_gray, dtype=np.uint8) * 255
        roi_in_bg_x0 = x0 - bx0
        roi_in_bg_y0 = y0 - by0
        roi_in_bg_x1 = x1 - bx0
        roi_in_bg_y1 = y1 - by0
        if roi_in_bg_x1 > roi_in_bg_x0 and roi_in_bg_y1 > roi_in_bg_y0:
            mask[
                max(0, roi_in_bg_y0):min(by1 - by0, roi_in_bg_y1),
                max(0, roi_in_bg_x0):min(bx1 - bx0, roi_in_bg_x1),
            ] = 0
        bg_pts = cv2.goodFeaturesToTrack(bg_gray, mask=mask, **shi_params)
        if bg_pts is not None:
            bg_pts = bg_pts.astype(np.float32)
            bg_pts[:, :, 0] += bx0
            bg_pts[:, :, 1] += by0
            all_corners.append(bg_pts)

    if not all_corners:
        return np.empty((0, 1, 2), dtype=np.float32)

    return np.concatenate(all_corners, axis=0)


def track_flow(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    prev_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run pyramidal Lucas-Kanade optical flow.

    Parameters
    ----------
    prev_frame, curr_frame:
        Consecutive frames (BGR or grayscale), uint8.
    prev_pts:
        ``(N, 1, 2)`` float32 array of seed points (from Shi-Tomasi).

    Returns
    -------
    (curr_pts, status):
        ``curr_pts`` has shape ``(N, 1, 2)`` float32;
        ``status`` is ``(N,)`` uint8 — 1 for successfully tracked points.
        If ``prev_pts`` is empty, returns matching empty arrays.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("cv2 is required for optical flow")

    if prev_pts is None or len(prev_pts) == 0:
        empty = np.empty((0, 1, 2), dtype=np.float32)
        empty_status = np.empty((0,), dtype=np.uint8)
        return empty, empty_status

    prev_gray = _to_gray(prev_frame)
    curr_gray = _to_gray(curr_frame)

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_pts.astype(np.float32),
        None,
        winSize=_LK_WIN_SIZE,
        maxLevel=_LK_MAX_LEVEL,
        criteria=_LK_CRITERIA,
    )

    # status from calcOpticalFlowPyrLK is (N, 1) uint8; flatten to (N,)
    status_flat = status.flatten() if status is not None else np.zeros(len(prev_pts), dtype=np.uint8)

    return curr_pts, status_flat


__all__ = ["detect_corners", "track_flow"]
