"""Standard tracking metrics: Success AUC, Precision@τ, AO, SR_τ.

Implementations are vectorised NumPy; per-sequence variants are
available so the caller can compute macro / frame-weighted averages.
"""
from __future__ import annotations

import numpy as np

from csc_lib.eval.custom_metrics.bbox import iou_xywh_batch


def success_auc(ious: np.ndarray, n_thresholds: int = 21) -> float:
    """Success AUC from per-frame IoU values.

    Uses the standard OTB definition: integrate the success curve
    ``frac(IoU >= τ)`` over τ ∈ [0, 1].  ``ious`` may contain NaNs for
    frames without GT; they are excluded from the denominator.
    """
    ious = np.asarray(ious, dtype=np.float64)
    valid = np.isfinite(ious) & (ious >= 0)
    if not valid.any():
        return 0.0
    iou_v = ious[valid]
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    succ = (iou_v[None, :] >= thresholds[:, None]).mean(axis=1)
    return float(np.trapz(succ, thresholds))


def precision_at_threshold(center_errors: np.ndarray, threshold_px: float = 20.0) -> float:
    """Precision: fraction of frames with centre error <= threshold_px."""
    ce = np.asarray(center_errors, dtype=np.float64)
    valid = np.isfinite(ce)
    if not valid.any():
        return 0.0
    return float((ce[valid] <= threshold_px).mean())


def normalized_precision_auc(
    normalized_errors: np.ndarray, n_thresholds: int = 51, tau_max: float = 0.5
) -> float:
    """Normalised precision AUC (LaSOT/TrackingNet-style).

    ``normalized_errors`` is centre error / image diagonal (or per-frame
    target-size).  Curve = ``frac(err <= τ)`` over τ ∈ [0, tau_max].
    """
    ne = np.asarray(normalized_errors, dtype=np.float64)
    valid = np.isfinite(ne) & (ne >= 0)
    if not valid.any():
        return 0.0
    ne_v = ne[valid]
    thresholds = np.linspace(0.0, tau_max, n_thresholds)
    prec = (ne_v[None, :] <= thresholds[:, None]).mean(axis=1)
    return float(np.trapz(prec, thresholds) / tau_max)


def average_overlap(ious: np.ndarray) -> float:
    """Average IoU = AO (GOT-10k headline metric)."""
    ious = np.asarray(ious, dtype=np.float64)
    valid = np.isfinite(ious) & (ious >= 0)
    if not valid.any():
        return 0.0
    return float(ious[valid].mean())


def success_rate(ious: np.ndarray, threshold: float = 0.5) -> float:
    """SR_τ = fraction of frames with IoU >= τ."""
    ious = np.asarray(ious, dtype=np.float64)
    valid = np.isfinite(ious) & (ious >= 0)
    if not valid.any():
        return 0.0
    return float((ious[valid] >= threshold).mean())


# ---------------------------------------------------------------------------
# Per-sequence helpers
# ---------------------------------------------------------------------------


def per_sequence_metrics(
    seq_results: dict[str, dict],
) -> dict[str, dict]:
    """Compute per-sequence summary metrics from a dict of
    ``{seq_name: {ious, center_errors, n_frames, time_seconds}}``."""
    out: dict[str, dict] = {}
    for name, r in seq_results.items():
        ious = np.asarray(r.get("ious", []), dtype=np.float64)
        ce = np.asarray(r.get("center_errors", []), dtype=np.float64)
        ne = np.asarray(r.get("normalized_center_errors", []), dtype=np.float64)
        n = int(r.get("n_frames", len(ious)))
        t = float(r.get("time_seconds", 0.0))
        out[name] = {
            "n_frames": n,
            "auc": success_auc(ious),
            "precision_20": precision_at_threshold(ce, threshold_px=20.0),
            "norm_precision_auc": normalized_precision_auc(ne) if ne.size else 0.0,
            "ao": average_overlap(ious),
            "sr_50": success_rate(ious, 0.5),
            "sr_75": success_rate(ious, 0.75),
            "fps": (n / t) if t > 0 else 0.0,
        }
    return out


def macro_average(per_seq: dict[str, dict], key: str) -> float:
    """Macro-average a per-sequence metric (equal weight per sequence)."""
    vals = [v[key] for v in per_seq.values() if key in v]
    if not vals:
        return 0.0
    return float(np.mean(vals))


def frame_weighted_average(per_seq: dict[str, dict], key: str) -> float:
    """Frame-weighted average (sequences with more frames count more)."""
    num = 0.0
    den = 0
    for v in per_seq.values():
        n = v.get("n_frames", 0)
        if key in v and n > 0:
            num += float(v[key]) * n
            den += n
    if den == 0:
        return 0.0
    return num / den


def overall_iou_array(seq_results: dict[str, dict]) -> np.ndarray:
    """Concatenate per-frame IoUs across all sequences."""
    parts: list[np.ndarray] = []
    for r in seq_results.values():
        parts.append(np.asarray(r.get("ious", []), dtype=np.float64))
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


# Convenience wrapper for callers that already have raw bbox arrays.

def compute_per_frame_arrays(
    pred_bboxes: np.ndarray,
    gt_bboxes: np.ndarray,
    image_diag: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ious, center_errors_px, normalized_center_errors)."""
    ious = iou_xywh_batch(pred_bboxes, gt_bboxes)
    pcx = pred_bboxes[..., 0] + pred_bboxes[..., 2] / 2.0
    pcy = pred_bboxes[..., 1] + pred_bboxes[..., 3] / 2.0
    gcx = gt_bboxes[..., 0] + gt_bboxes[..., 2] / 2.0
    gcy = gt_bboxes[..., 1] + gt_bboxes[..., 3] / 2.0
    ce = np.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2)
    ne = ce / max(1.0, float(image_diag))
    invalid = (gt_bboxes[..., 2] <= 0) | (gt_bboxes[..., 3] <= 0)
    ious[invalid] = np.nan
    ce[invalid] = np.nan
    ne[invalid] = np.nan
    return ious, ce, ne
