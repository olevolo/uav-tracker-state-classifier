"""Bbox utilities — pure NumPy, xywh format throughout.

Vectorised wherever practical.  All inputs are tuples or numpy arrays
with shape (4,) or (T, 4) where the last dim is (x, y, w, h).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

BBox = Sequence[float]  # (x, y, w, h)


def _as_xyxy(b: BBox) -> tuple[float, float, float, float]:
    x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return x, y, x + w, y + h


def iou_xywh(a: BBox, b: BBox) -> float:
    """Single-pair IoU of two xywh bboxes. Returns 0 for degenerate boxes."""
    if a is None or b is None:
        return 0.0
    aw = max(0.0, float(a[2]))
    ah = max(0.0, float(a[3]))
    bw = max(0.0, float(b[2]))
    bh = max(0.0, float(b[3]))
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax1, ay1, ax2, ay2 = _as_xyxy(a)
    bx1, by1, bx2, by2 = _as_xyxy(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def iou_xywh_batch(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Vectorised IoU for two arrays of shape (T, 4) in xywh.

    Frames where either box has zero width/height return 0.
    """
    preds = np.asarray(preds, dtype=np.float64)
    gts = np.asarray(gts, dtype=np.float64)
    if preds.shape != gts.shape or preds.shape[-1] != 4:
        raise ValueError(f"shape mismatch: preds {preds.shape}, gts {gts.shape}")
    px1, py1 = preds[..., 0], preds[..., 1]
    px2, py2 = px1 + np.maximum(0, preds[..., 2]), py1 + np.maximum(0, preds[..., 3])
    gx1, gy1 = gts[..., 0], gts[..., 1]
    gx2, gy2 = gx1 + np.maximum(0, gts[..., 2]), gy1 + np.maximum(0, gts[..., 3])
    ix1 = np.maximum(px1, gx1)
    iy1 = np.maximum(py1, gy1)
    ix2 = np.minimum(px2, gx2)
    iy2 = np.minimum(py2, gy2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    pa = np.maximum(0, preds[..., 2]) * np.maximum(0, preds[..., 3])
    ga = np.maximum(0, gts[..., 2]) * np.maximum(0, gts[..., 3])
    union = pa + ga - inter
    out = np.zeros_like(inter, dtype=np.float64)
    valid = (union > 0) & (pa > 0) & (ga > 0)
    out[valid] = inter[valid] / union[valid]
    return out


def center_xy(b: BBox) -> tuple[float, float]:
    """Centre (cx, cy) of an xywh bbox."""
    return float(b[0]) + float(b[2]) / 2.0, float(b[1]) + float(b[3]) / 2.0


def center_error(a: BBox, b: BBox) -> float:
    """Pixel distance between bbox centres."""
    ax, ay = center_xy(a)
    bx, by = center_xy(b)
    return math.hypot(ax - bx, ay - by)


def normalized_center_error(a: BBox, b: BBox, image_diag: float) -> float:
    """Centre distance divided by the image diagonal."""
    if image_diag <= 0:
        return 0.0
    return center_error(a, b) / image_diag


def bbox_area(b: BBox) -> float:
    return max(0.0, float(b[2])) * max(0.0, float(b[3]))


def aspect_ratio(b: BBox) -> float:
    h = max(1e-6, float(b[3]))
    return float(b[2]) / h


def scale_ratio(a: BBox, b: BBox) -> float:
    """Area(a) / Area(b)."""
    ba = bbox_area(b)
    if ba <= 0:
        return 0.0
    return bbox_area(a) / ba


def velocity(prev: BBox, cur: BBox) -> float:
    """Pixel distance between consecutive bbox centres."""
    return center_error(prev, cur)
