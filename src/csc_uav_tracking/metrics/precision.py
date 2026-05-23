"""Precision@threshold metric (center-location error).

Paper / PLAN §11 Phase 1: fraction of frames where the Euclidean
distance between the predicted bbox center and the ground-truth bbox
center is ≤ ``threshold`` pixels. Default threshold is 20 px (OTB).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def precision_at_threshold(
    gt: Iterable[np.ndarray],
    pred: Iterable[np.ndarray],
    threshold: float = 20.0,
) -> float:
    """Center-location precision at a given pixel threshold.

    Returns the fraction of frames with center-distance ≤ ``threshold``.
    """
    gt_arr = np.array(list(gt), dtype=np.float64)
    pred_arr = np.array(list(pred), dtype=np.float64)

    if len(gt_arr) == 0:
        return 0.0

    # Center: (x + w/2, y + h/2)
    gt_cx = gt_arr[:, 0] + gt_arr[:, 2] / 2.0
    gt_cy = gt_arr[:, 1] + gt_arr[:, 3] / 2.0
    pred_cx = pred_arr[:, 0] + pred_arr[:, 2] / 2.0
    pred_cy = pred_arr[:, 1] + pred_arr[:, 3] / 2.0

    dist = np.sqrt((gt_cx - pred_cx) ** 2 + (gt_cy - pred_cy) ** 2)
    return float(np.mean(dist <= threshold))
