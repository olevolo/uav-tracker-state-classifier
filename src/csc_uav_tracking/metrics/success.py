"""Success AUC metric (PLAN §11 Phase 1 exit demo).

Paper: integrate IoU overlap curve from 0 to 1. A bbox pair with
IoU ≥ τ counts as "successful" at threshold τ; the metric is the
integral of success rate over τ ∈ [0, 1].
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two ``(N, 4)`` bbox arrays in xywh.

    Vectorized for speed in OPE aggregation.
    """
    # Convert xywh -> xyxy
    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = ax1 + a[:, 2], ay1 + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = bx1 + b[:, 2], by1 + b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = a[:, 2] * a[:, 3]
    area_b = b[:, 2] * b[:, 3]
    union_area = area_a + area_b - inter_area

    # Guard against zero-area boxes
    result = np.where(union_area > 0.0, inter_area / union_area, 0.0)
    return result


def compute_auc(gt: Iterable[np.ndarray], pred: Iterable[np.ndarray]) -> float:
    """Success AUC: ∫ success(τ) dτ for τ ∈ [0, 1], 0.05 step.

    Parameters
    ----------
    gt, pred:
        Iterables of ``(4,)`` xywh bboxes, paired frame-by-frame.

    Returns
    -------
    float
        AUC in ``[0, 1]``.
    """
    gt_arr = np.array(list(gt), dtype=np.float64)
    pred_arr = np.array(list(pred), dtype=np.float64)

    if gt_arr.ndim == 1:
        gt_arr = gt_arr.reshape(1, -1)
    if pred_arr.ndim == 1:
        pred_arr = pred_arr.reshape(1, -1)

    if len(gt_arr) == 0:
        return 0.0

    ious = iou(gt_arr, pred_arr)   # (N,)

    # 21 thresholds: 0.0, 0.05, 0.10, …, 1.0
    thresholds = np.linspace(0.0, 1.0, 21)
    success_rates = np.array(
        [float(np.mean(ious >= t)) for t in thresholds], dtype=np.float64
    )
    # Integral via trapezoidal rule (uniform spacing → equivalent to mean)
    auc = float(np.trapz(success_rates, thresholds))
    return auc
