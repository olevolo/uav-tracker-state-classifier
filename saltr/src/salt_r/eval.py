"""eval.py — SALT-RD GRU model evaluation.

Full per-head evaluation of a trained SALTRD checkpoint against the
val/diagnostic NPZ splits.  Reports AUROC, AUPRC, ECE, Brier, NLL,
Recall@5%FPR, NT2F, and bootstrap CIs per label head.

Usage::

    python -m salt_r.eval \\
        --npz  saltr/data/salt_rd_v0.npz \\
        --checkpoint saltr/checkpoints/salt_rd_best.pt \\
        --split val \\
        --output saltr/results/eval_val.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Module-level head names (imported lazily to avoid hard dependency at import time)
try:
    from salt_r.model import HEAD_NAMES as _HEAD_NAMES
except Exception:
    _HEAD_NAMES: list[str] = []



# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _file_md5(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _build_provenance(
    npz_path: str,
    checkpoint_path: str | None,
    split: str,
) -> dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "npz_path": str(npz_path),
        "npz_md5": _file_md5(npz_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_md5": _file_md5(checkpoint_path) if checkpoint_path else None,
        "split": split,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
    }



# ---------------------------------------------------------------------------
# Temperature scaling (post-hoc calibration)
# ---------------------------------------------------------------------------

#: Reliability heads for which calibration is well-defined (clean labels, good AUROC).
RELIABILITY_HEADS: tuple[str, ...] = ("false_confirmed", "failure_in_5", "recoverable")


def calibrate_temperature(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fit a per-head scalar temperature T by minimising NLL on the val split.

    Calibration formula: ``p_cal = sigmoid(logit(p) / T)``
    where ``logit(p) = log(p / (1-p))``.

    Uses a grid search (100 candidates over [0.1, 10]) followed by ternary
    search refinement.  No scipy dependency.

    Parameters
    ----------
    y_true:
        Binary ground-truth, shape ``(N,)``.  Must be from the **val split only**.
        Never pass train-split labels.
    y_pred:
        Predicted probabilities in ``(0, 1)``, shape ``(N,)``.

    Returns
    -------
    float
        Optimal temperature T (T=1.0 ↔ no calibration needed).
    """
    eps = 1e-6
    p_safe = np.clip(y_pred.astype(np.float64), eps, 1 - eps)
    logits = np.log(p_safe / (1 - p_safe))

    def nll_at_T(T: float) -> float:
        T = max(T, eps)
        p_cal = 1.0 / (1.0 + np.exp(-logits / T))
        p_cal = np.clip(p_cal, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(p_cal) + (1 - y_true) * np.log(1 - p_cal)))

    T_grid = np.linspace(0.1, 10.0, 100)
    best_T = float(T_grid[np.argmin([nll_at_T(T) for T in T_grid])])

    lo, hi = max(0.05, best_T - 1.0), best_T + 1.0
    for _ in range(50):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if nll_at_T(m1) < nll_at_T(m2):
            hi = m2
        else:
            lo = m1
    return float((lo + hi) / 2.0)


def apply_temperature(y_pred: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature T: ``p_cal = sigmoid(logit(p) / T)``."""
    eps = 1e-6
    p_safe = np.clip(y_pred.astype(np.float64), eps, 1 - eps)
    logits = np.log(p_safe / (1 - p_safe))
    p_cal = 1.0 / (1.0 + np.exp(-logits / T))
    return np.clip(p_cal, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Pure numpy/scipy replacements for sklearn.metrics
# (scikit-learn is not a declared project dependency)
# ---------------------------------------------------------------------------


def _roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve (FPR, TPR, thresholds) without sklearn.

    Follows the same convention as sklearn.metrics.roc_curve:
    - Sorted by decreasing threshold.
    - A sentinel threshold above the max score is prepended so the curve
      starts at (FPR=0, TPR=0).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    # Sort by descending score
    desc_idx = np.argsort(y_score)[::-1]
    sorted_scores = y_score[desc_idx]
    sorted_labels = y_true[desc_idx]

    # Unique thresholds (descending)
    thresholds = np.concatenate([[sorted_scores[0] + 1], sorted_scores])

    tp = np.concatenate([[0], np.cumsum(sorted_labels)])
    fp = np.concatenate([[0], np.cumsum(1 - sorted_labels)])

    n_pos = tp[-1]
    n_neg = fp[-1]

    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)

    return fpr, tpr, thresholds


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve via the trapezoidal rule."""
    fpr, tpr, _ = _roc_curve(y_true, y_score)
    # np.trapz expects x increasing
    return float(np.trapz(tpr, fpr))


def _average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the Precision-Recall curve (interpolation-free, sklearn convention)."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    desc_idx = np.argsort(y_score)[::-1]
    sorted_labels = y_true[desc_idx]

    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)

    precision = tp / (tp + fp)
    recall = tp / max(sorted_labels.sum(), 1)

    # Prepend (recall=0, precision=1) sentinel
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])

    # Sum of trapezoids (sklearn uses step interpolation: delta_recall * precision_right)
    return float(np.sum(np.diff(recall) * precision[1:]))

# ---------------------------------------------------------------------------
# GO/NO-GO gate thresholds (Phase 1b criteria from SALT-RD plan)
# ---------------------------------------------------------------------------

GO_THRESHOLDS: dict[str, float] = {
    # Reliability heads (all schemas)
    "auprc_false_confirmed":              0.30,   # > → GO
    "auroc_false_confirmed":              0.65,
    "auroc_failure_in_5":                 0.75,
    "ece_false_confirmed":                0.12,   # < → GO (lower is better)
    # Compute policy (all schemas)
    "auroc_needs_full_compute":           0.70,
    # v1 schema: dynamicity decomposed — replaces auroc_hard_dynamic_scene
    # Gate is skipped (NaN) when head is absent, so v0 evals are unaffected.
    "auroc_imminent_failure_dynamic":     0.75,
    "auprc_imminent_failure_dynamic":     0.15,  # ≈ 3.5x base rate (~4.2%)
}

STOP_THRESHOLDS: dict[str, float] = {
    "auprc_false_confirmed":              0.15,
    "auroc_false_confirmed":              0.55,
    "auroc_failure_in_5":                 0.65,
}

# Heads whose ECE is lower-is-better for the GO gate
_LOWER_IS_BETTER = {"ece_false_confirmed"}

# ---------------------------------------------------------------------------
# Calibration metric helpers
# ---------------------------------------------------------------------------


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins).

    Parameters
    ----------
    probs:
        Predicted probabilities in [0, 1], shape (N,).
    labels:
        Binary ground-truth labels, shape (N,).
    n_bins:
        Number of equal-width bins in [0, 1].

    Returns
    -------
    float
        Weighted mean |accuracy - confidence| across bins.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(probs))


# ---------------------------------------------------------------------------
# Per-head classification metrics
# ---------------------------------------------------------------------------


def compute_head_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    head_name: str,
) -> dict[str, float]:
    """Compute full metric suite for one binary classification head.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels, shape (N,), dtype float or int.
    y_pred:
        Predicted probabilities in [0, 1], shape (N,).
    head_name:
        Name used only for error messages / logging.

    Returns
    -------
    dict with keys: base_rate, auroc, auprc, ece, brier, nll,
    recall_at_5pct_fpr.  Values are nan when the class is
    degenerate (all-zero or all-one labels).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    base_rate = float(y_true.mean())
    if base_rate == 0.0 or base_rate == 1.0:
        return {
            "base_rate": base_rate,
            "auroc": float("nan"),
            "auprc": float("nan"),
            "ece": float("nan"),
            "brier": float("nan"),
            "nll": float("nan"),
            "recall_at_5pct_fpr": float("nan"),
        }

    auroc = _roc_auc_score(y_true, y_pred)
    auprc = _average_precision_score(y_true, y_pred)

    # ECE (10 equal-width bins)
    ece = _ece(y_pred, y_true, n_bins=10)

    # Brier score
    brier = float(np.mean((y_pred - y_true) ** 2))

    # Negative log-likelihood
    eps = 1e-8
    nll = float(
        -np.mean(
            y_true * np.log(y_pred + eps)
            + (1 - y_true) * np.log(1 - y_pred + eps)
        )
    )

    # Recall @ 5% FPR — the "false_confirmed" signature metric
    fpr, tpr, _ = _roc_curve(y_true, y_pred)
    idx = int(np.searchsorted(fpr, 0.05))
    recall_at_5fpr = float(tpr[min(idx, len(tpr) - 1)])

    return {
        "base_rate": base_rate,
        "auroc": auroc,
        "auprc": auprc,
        "ece": ece,
        "brier": brier,
        "nll": nll,
        "recall_at_5pct_fpr": recall_at_5fpr,
    }


# ---------------------------------------------------------------------------
# NT2F — Normalized Time to Failure
# ---------------------------------------------------------------------------


def compute_nt2f(
    iou_traces: dict[str, np.ndarray],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Normalized Time to Failure (from MATA 2026).

    NT2F_i = t_failure_i / seq_len_i
    where t_failure is the first frame (after init) where IoU drops below
    iou_threshold.  Sequences that never fail receive NT2F = 1.0.

    Parameters
    ----------
    iou_traces:
        Mapping from sequence key to per-frame IoU array, shape (n_frames,).
    iou_threshold:
        IoU level below which a frame is considered a tracking failure.

    Returns
    -------
    dict with keys: nt2f_mean, nt2f_std, n_sequences, n_never_failed.
    """
    scores: list[float] = []
    never_failed = 0

    for seq_key, iou in iou_traces.items():
        iou = np.asarray(iou, dtype=float)
        n = len(iou)
        if n < 2:
            continue
        # Look for the first failure *after* initialisation frame (index 0)
        failures = np.where(iou[1:] < iou_threshold)[0] + 1  # +1 for the slice offset
        if len(failures) == 0:
            scores.append(1.0)
            never_failed += 1
        else:
            t_failure = int(failures[0])
            scores.append(t_failure / max(n, 1))

    if not scores:
        return {
            "nt2f_mean": float("nan"),
            "nt2f_std": float("nan"),
            "n_sequences": 0,
            "n_never_failed": 0,
        }

    return {
        "nt2f_mean": float(np.mean(scores)),
        "nt2f_std": float(np.std(scores)),
        "n_sequences": len(scores),
        "n_never_failed": never_failed,
    }


# ---------------------------------------------------------------------------
# Event-level failure lead-time metric
# ---------------------------------------------------------------------------


def _find_failure_events(
    iou: np.ndarray,
    threshold: float = 0.30,
) -> list[tuple[int, int]]:
    """Find contiguous failure episodes (IoU < threshold) after first being above.

    Parameters
    ----------
    iou:
        Per-frame IoU array.
    threshold:
        IoU below which a frame is in failure.

    Returns
    -------
    List of (start_frame, end_frame) pairs, one per episode.
    ``start_frame`` is the first frame with IoU < threshold (after being >= threshold).
    ``end_frame`` is the last consecutive frame below threshold.
    The tracker must have had at least one above-threshold frame before each event.
    """
    events: list[tuple[int, int]] = []
    above = False
    in_failure = False
    start = 0
    n = len(iou)
    for t in range(n):
        v = float(iou[t])
        if not above:
            if v >= threshold:
                above = True
        else:
            if v < threshold and not in_failure:
                start = t
                in_failure = True
            elif v >= threshold and in_failure:
                events.append((start, t - 1))
                in_failure = False
    if in_failure:
        events.append((start, n - 1))
    return events


def compute_failure_lead_time(
    iou_dict: dict[str, np.ndarray],
    preds_dict: dict[str, np.ndarray],
    labels_dict: dict[str, np.ndarray],
    label_names: list[str],
    head_names: list[str],
    threshold: float = 0.50,
    iou_failure_threshold: float = 0.30,
    label_name: str = "imminent_failure_dynamic",
    horizon: int = 5,
) -> dict[str, Any]:
    """Event-level failure lead-time metric.

    For each failure event (first IoU-drop episode), find the FIRST alert frame
    in the window [event_start - horizon - 5, event_start).  Lead time =
    event_start - first_alert_frame.

    Parameters
    ----------
    iou_dict:
        Per-sequence IoU arrays.
    preds_dict:
        Per-sequence prediction arrays, shape (n_frames, n_heads).
    labels_dict:
        Per-sequence label arrays, shape (n_frames, n_labels).
    label_names:
        Ordered list of label column names in labels_dict arrays.
    head_names:
        Ordered list of head column names in preds_dict arrays.
    threshold:
        Probability threshold for calling an alert.
    iou_failure_threshold:
        IoU below which a frame counts as a tracking failure.
    label_name:
        Which prediction head to use for alerts.
    horizon:
        Look-back horizon in frames (the window is [event - horizon - 5, event)).

    Returns
    -------
    dict with keys:
      n_failure_events, n_detected_events, event_recall,
      per_event_lead_times, median_lead_time, mean_lead_time, p25_lead_time, p75_lead_time.
    """
    if label_name not in label_names:
        return {"note": f"{label_name} not in label schema — skipping lead-time"}
    if label_name not in head_names:
        return {"note": f"{label_name} not in model heads — skipping lead-time"}

    pred_idx = head_names.index(label_name)

    n_failure_events = 0
    n_detected_events = 0
    per_event_lead_times: list[int] = []

    for seq_key in labels_dict:
        iou = iou_dict.get(seq_key)
        pred_arr = preds_dict.get(seq_key)

        if iou is None or pred_arr is None:
            continue
        if pred_idx >= pred_arr.shape[1]:
            continue

        iou_arr = np.asarray(iou, dtype=float)
        y_pred = pred_arr[:, pred_idx]
        n = len(iou_arr)

        events = _find_failure_events(iou_arr, threshold=iou_failure_threshold)
        for event_start, _event_end in events:
            n_failure_events += 1
            # Look-back window: [event_start - horizon - 5, event_start)
            window_start = max(0, event_start - horizon - 5)
            prior_alerts = [
                t for t in range(window_start, event_start)
                if t < n and float(y_pred[t]) > threshold
            ]
            if prior_alerts:
                n_detected_events += 1
                lead_time = event_start - min(prior_alerts)
                per_event_lead_times.append(lead_time)

    event_recall = n_detected_events / max(n_failure_events, 1)

    if per_event_lead_times:
        median_lt = float(np.median(per_event_lead_times))
        mean_lt   = float(np.mean(per_event_lead_times))
        p25_lt    = float(np.percentile(per_event_lead_times, 25))
        p75_lt    = float(np.percentile(per_event_lead_times, 75))
    else:
        median_lt = mean_lt = p25_lt = p75_lt = float("nan")

    return {
        "label_name": label_name,
        "horizon": horizon,
        "threshold": threshold,
        "n_failure_events": n_failure_events,
        "n_detected_events": n_detected_events,
        "event_recall": round(event_recall, 4),
        "per_event_lead_times": per_event_lead_times,
        "median_lead_time": median_lt,
        "mean_lead_time": mean_lt,
        "p25_lead_time": p25_lt,
        "p75_lead_time": p75_lt,
    }


def run_lead_time_analysis(
    iou_dict: dict[str, np.ndarray],
    preds_dict: dict[str, np.ndarray],
    labels_dict: dict[str, np.ndarray],
    label_names: list[str],
    head_names: list[str],
    threshold: float = 0.50,
    iou_failure_threshold: float = 0.30,
) -> dict[str, Any]:
    """Run event-level lead-time analysis for all three IFD horizons.

    Calls :func:`compute_failure_lead_time` for ifd5 (horizon=5),
    ifd10 (horizon=10), and ifd20 (horizon=20).

    Returns
    -------
    dict with keys "ifd5", "ifd10", "ifd20", each containing the result
    from :func:`compute_failure_lead_time`.
    """
    configs = [
        ("ifd5",  "imminent_failure_dynamic",    5),
        ("ifd10", "imminent_failure_dynamic_10", 10),
        ("ifd20", "imminent_failure_dynamic_20", 20),
    ]
    results: dict[str, Any] = {}
    for key, lname, h in configs:
        results[key] = compute_failure_lead_time(
            iou_dict=iou_dict,
            preds_dict=preds_dict,
            labels_dict=labels_dict,
            label_names=label_names,
            head_names=head_names,
            threshold=threshold,
            iou_failure_threshold=iou_failure_threshold,
            label_name=lname,
            horizon=h,
        )
    return results


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (sequence-level resampling)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    per_sequence_scores: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """95% CI via sequence-level bootstrap resampling.

    IMPORTANT: resampling is done at the sequence level, NOT the frame level.
    Adjacent frames within a sequence are correlated; frame-level bootstrap
    would yield artificially narrow CIs.

    Parameters
    ----------
    per_sequence_scores:
        One scalar per sequence (e.g. per-sequence AUPRC).
    n_bootstrap:
        Number of bootstrap resamples.
    alpha:
        Significance level (default 0.05 → 95% CI).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    (lower, upper) confidence interval bounds.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(per_sequence_scores, dtype=float)
    boot_means = np.array(
        [
            rng.choice(scores, size=len(scores), replace=True).mean()
            for _ in range(n_bootstrap)
        ]
    )
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# GO/NO-GO gate
# ---------------------------------------------------------------------------


def check_go_nogo(results: dict[str, Any]) -> str:
    """Evaluate GO/NO-GO/BORDERLINE gate from eval results dict.

    Parameters
    ----------
    results:
        Dict as returned by :func:`evaluate`.  Must contain a
        ``"head_metrics"`` sub-dict with per-head entries.

    Returns
    -------
    "GO", "STOP", or "BORDERLINE".
    """
    head_metrics: dict[str, dict[str, float]] = results.get("head_metrics", {})

    def _get(key: str) -> float:
        """Resolve a flat metric key of the form '<metric>_<head>'."""
        # key examples: "auprc_false_confirmed", "auroc_failure_in_5",
        #               "ece_false_confirmed", "auroc_hard_dynamic_scene"
        parts = key.split("_", 1)  # split at first underscore only
        if len(parts) != 2:
            return float("nan")
        metric, head = parts[0], parts[1]
        return head_metrics.get(head, {}).get(metric, float("nan"))

    # Check STOP criteria first (any one → STOP)
    for key, threshold in STOP_THRESHOLDS.items():
        val = _get(key)
        if np.isnan(val):
            continue
        if key in _LOWER_IS_BETTER:
            if val > threshold:
                return "STOP"
        else:
            if val < threshold:
                return "STOP"

    # Check GO criteria (all must pass)
    go_count = 0
    go_total = 0
    for key, threshold in GO_THRESHOLDS.items():
        val = _get(key)
        if np.isnan(val):
            continue
        go_total += 1
        if key in _LOWER_IS_BETTER:
            if val < threshold:
                go_count += 1
        else:
            if val > threshold:
                go_count += 1

    if go_total == 0:
        return "BORDERLINE"
    if go_count == go_total:
        return "GO"
    return "BORDERLINE"


# ---------------------------------------------------------------------------
# NPZ + model loading helpers
# ---------------------------------------------------------------------------


def _load_npz_split(
    npz_path: str, split: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load features, labels, and iou_traces for the requested split.

    Parameters
    ----------
    npz_path:
        Path to the compressed NPZ produced by collect_features.py.
    split:
        One of "train", "val", "diagnostic".

    Returns
    -------
    (features_dict, labels_dict, iou_dict) each keyed by compound
    sequence key ``"dataset_name/seq_name"``.
    """
    data = np.load(npz_path, allow_pickle=True)

    compound_keys = [
        k[len("features/"):] for k in data.files if k.startswith("features/")
    ]

    features_dict: dict[str, np.ndarray] = {}
    labels_dict: dict[str, np.ndarray] = {}
    iou_dict: dict[str, np.ndarray] = {}

    for key in compound_keys:
        seq_split = str(data[f"split/{key}"])
        if seq_split != split:
            continue
        features_dict[key] = data[f"features/{key}"]
        labels_dict[key] = data[f"labels/{key}"]
        iou_dict[key] = data[f"iou_trace/{key}"]

    return features_dict, labels_dict, iou_dict


def _dataset_from_seq_key(seq_key: str) -> str:
    """Return dataset prefix from compound key ``dataset/sequence``."""
    return seq_key.split("/", 1)[0] if "/" in seq_key else "unknown"


def compute_per_dataset_head_metrics(
    labels_dict: dict[str, np.ndarray],
    preds_dict: dict[str, np.ndarray],
    label_names: list[str],
    model_head_names: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute the same per-head metrics independently for each dataset.

    Pooled val/diagnostic metrics can hide dataset-specific regressions because
    UAV123 contributes many more frames than VisDrone/DTB70.  This helper keeps
    the aggregate metrics unchanged while adding a stratified view for result
    auditing and GO/KILL decisions.
    """
    by_dataset: dict[str, list[str]] = collections.defaultdict(list)
    for seq_key in labels_dict:
        by_dataset[_dataset_from_seq_key(seq_key)].append(seq_key)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for dataset_name, seq_keys in sorted(by_dataset.items()):
        dataset_metrics: dict[str, dict[str, float]] = {}
        for hi, head in enumerate(label_names):
            y_true_parts: list[np.ndarray] = []
            y_pred_parts: list[np.ndarray] = []

            for seq_key in seq_keys:
                lab = labels_dict[seq_key].astype(float)
                if hi >= lab.shape[1]:
                    continue
                y_true_parts.append(lab[:, hi])

                if seq_key in preds_dict and head in model_head_names:
                    pred_col = model_head_names.index(head)
                    pred = preds_dict[seq_key]
                    if pred_col < pred.shape[1]:
                        y_pred_parts.append(pred[:, pred_col])
                    else:
                        y_pred_parts.append(np.full(lab.shape[0], 0.5, dtype=np.float32))
                else:
                    y_pred_parts.append(np.full(lab.shape[0], 0.5, dtype=np.float32))

            if not y_true_parts:
                continue

            y_true = np.concatenate(y_true_parts)
            y_pred = np.concatenate(y_pred_parts)
            metric = compute_head_metrics(y_true, y_pred, head)
            metric["n_frames"] = int(len(y_true))
            metric["n_sequences"] = int(len(seq_keys))
            dataset_metrics[head] = metric

        results[dataset_name] = dataset_metrics

    return results


def _load_model(checkpoint_path: str, n_features: int, n_labels: int, device: str):
    """Load SALTRD model from checkpoint.

    Reads ``head_names`` and ``memory_dim`` from checkpoint metadata so v0 (7 heads),
    v1 (9 heads), and v2.1 (37-dim input) checkpoints all load correctly.

    Returns model or None on failure.
    """
    try:
        import torch
        from salt_r.model import SALTRD, HEAD_NAMES

        state = torch.load(checkpoint_path, map_location=device)
        checkpoint_data = state if isinstance(state, dict) else {}

        # Derive head names: prefer metadata embedded by train.py, else v0 default.
        head_names = checkpoint_data.get("head_names", list(HEAD_NAMES))
        # memory_dim: default 0 for backwards compatibility with pre-v2.1 checkpoints.
        memory_dim = int(checkpoint_data.get("memory_dim", 0))
        memory_feature_names_ckpt = checkpoint_data.get("memory_feature_names", None)

        model = SALTRD(head_names=head_names, memory_dim=memory_dim)
        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
            model.load_state_dict(checkpoint_data["model_state_dict"])
        elif isinstance(checkpoint_data, dict) and "state_dict" in checkpoint_data:
            model.load_state_dict(checkpoint_data["state_dict"])
        else:
            model.load_state_dict(state)
        model.eval()
        return model
    except Exception as exc:
        warnings.warn(
            f"Could not load model from {checkpoint_path}: {exc}. "
            "Skipping model inference — only dataset-level label statistics "
            "will be reported.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _run_inference(
    model,
    features_dict: dict[str, np.ndarray],
    window_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Run model inference over all sequences.

    Each sequence is processed as one batch of overlapping windows of
    length `window_size`.  When a sequence is shorter than `window_size`
    the whole sequence is used as a single window.

    Parameters
    ----------
    model:
        Loaded SALTRD model (or None → returns empty dict).
    features_dict:
        Per-sequence feature arrays keyed by compound seq key.
    window_size:
        Temporal window length fed to the GRU.
    device:
        Torch device string.

    Returns
    -------
    Dict mapping seq_key → per-frame probability array of shape
    (n_frames, n_heads).
    """
    if model is None:
        return {}

    import torch

    model = model.to(device)
    preds: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for seq_key, feats in features_dict.items():
            n = feats.shape[0]
            x = torch.tensor(feats, dtype=torch.float32, device=device)  # (n, F)

            # Build sliding windows: (n_windows, window_size, F)
            # Each position i uses frames [i-window_size+1 .. i] (causal)
            windows = []
            for t in range(n):
                start = max(0, t - window_size + 1)
                window = feats[start : t + 1]  # (<=window_size, F)
                # Left-pad with zeros to window_size
                pad_len = window_size - len(window)
                if pad_len > 0:
                    window = np.concatenate(
                        [np.zeros((pad_len, feats.shape[1]), dtype=np.float32), window],
                        axis=0,
                    )
                windows.append(window)

            x_batch = torch.tensor(
                np.stack(windows, axis=0), dtype=torch.float32, device=device
            )  # (n, window_size, F)

            out = model(x_batch)  # dict[str, Tensor(n,)] or Tensor(n, heads)
            if isinstance(out, dict):
                # Use key order from the model's own heads — correct for v0 and v1.
                _out_heads = list(out.keys())
                prob_matrix = np.stack(
                    [out[h].detach().cpu().numpy() for h in _out_heads],
                    axis=1,
                ).astype(np.float32)
            else:
                prob_matrix = out.detach().cpu().numpy().astype(np.float32)
            prob_matrix = np.clip(prob_matrix, 0.0, 1.0)

            preds[seq_key] = prob_matrix

    return preds


# ---------------------------------------------------------------------------
# Formatted summary table
# ---------------------------------------------------------------------------


def _print_summary_table(
    head_metrics: dict[str, dict[str, float]],
    label_names: list[str],
) -> None:
    """Print a formatted table of per-head metrics to stdout."""
    col_w = 10
    header_fields = ["head", "base%", "AUROC", "AUPRC", "ECE", "Brier", "NLL", "R@5FPR"]
    header = (
        f"{'head':<22}"
        + "".join(f"{f:>{col_w}}" for f in header_fields[1:])
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for head in label_names:
        m = head_metrics.get(head, {})
        base_pct = m.get("base_rate", float("nan")) * 100

        def _fmt(v: float) -> str:
            return f"{v:>{col_w}.4f}" if not np.isnan(v) else f"{'n/a':>{col_w}}"

        row = (
            f"{head:<22}"
            f"{base_pct:>{col_w}.1f}"
            + _fmt(m.get("auroc", float("nan")))
            + _fmt(m.get("auprc", float("nan")))
            + _fmt(m.get("ece", float("nan")))
            + _fmt(m.get("brier", float("nan")))
            + _fmt(m.get("nll", float("nan")))
            + _fmt(m.get("recall_at_5pct_fpr", float("nan")))
        )
        print(row)

    print(sep + "\n")


# ---------------------------------------------------------------------------
# Predictions export helper
# ---------------------------------------------------------------------------


def _save_predictions_json(
    preds_dict: dict[str, np.ndarray],
    predictions_output: str,
    head_names: list[str] | None = None,
) -> None:
    """Save per-sequence per-frame probabilities as JSON for policy.py --probs-json."""
    if head_names is None:
        from salt_r.model import HEAD_NAMES
        head_names = list(HEAD_NAMES)
    serializable: dict[str, list[dict[str, float]]] = {}
    for seq_key, pred in preds_dict.items():
        serializable[seq_key] = [
            {h: float(pred[t, i]) for i, h in enumerate(head_names) if i < pred.shape[1]}
            for t in range(pred.shape[0])
        ]
    Path(predictions_output).parent.mkdir(parents=True, exist_ok=True)
    Path(predictions_output).write_text(json.dumps(serializable, indent=2))
    print(f"Predictions written to: {predictions_output}")


# ---------------------------------------------------------------------------
# Full evaluation entry point
# ---------------------------------------------------------------------------


def evaluate(
    npz_path: str,
    checkpoint_path: str,
    split: str = "val",
    window_size: int = 20,
    device: str = "cpu",
    output_path: str | None = None,
    predictions_output: str | None = None,
    calibrate_heads: list[str] | None = None,
    memory_sidecar_path: str | None = None,
) -> dict[str, Any]:
    """Run full evaluation and return results dict.

    Computes per-head AUROC / AUPRC / ECE / Brier / NLL / Recall@5%FPR,
    NT2F (IoU thresholds 0.5 and 0.2), bootstrap CIs on per-sequence
    AUPRC(false_confirmed), and prints a formatted summary table.

    Parameters
    ----------
    npz_path:
        Path to the NPZ produced by collect_features.py.
    checkpoint_path:
        Path to the SALTRD .pt checkpoint produced by train.py.
    split:
        Which split to evaluate: "val" | "diagnostic" | "train".
    window_size:
        Temporal window fed to the GRU during inference.
    device:
        Torch device, e.g. "cpu" or "cuda".
    output_path:
        If provided, write the full results dict as JSON to this path.
    predictions_output:
        If provided, write per-frame per-head probabilities as JSON for
        policy.py offline replay (format: {seq_key: [{head: prob}, ...]}).
    memory_sidecar_path:
        Optional path to memory sidecar NPZ with keys ``memory_features/{seq}``
        (float32, T×9).  If None or file does not exist, evaluation uses the
        base 28-dim features only.  Sequences missing from the sidecar are
        padded with zeros when the checkpoint expects a 37-dim model.

    Returns
    -------
    dict containing:
    - "split": str
    - "n_sequences": int
    - "head_metrics": {head_name: {auroc, auprc, ece, brier, nll,
                                   recall_at_5pct_fpr, base_rate}}
    - "nt2f_05": nt2f result dict at IoU threshold 0.5
    - "nt2f_02": nt2f result dict at IoU threshold 0.2
    - "bootstrap_auprc_false_confirmed": {"ci_lo", "ci_hi", "n_sequences"}
    - "go_nogo": str
    """
    # ------------------------------------------------------------------
    # 1. Load NPZ
    # ------------------------------------------------------------------
    print(f"Loading NPZ: {npz_path}  split={split}")
    features_dict, labels_dict, iou_dict = _load_npz_split(npz_path, split)

    # Read label names from NPZ (fall back to collect_features default)
    try:
        _raw = np.load(npz_path, allow_pickle=True)
        label_names: list[str] = list(_raw["label_names"].tolist())
    except Exception:
        from salt_r.collect_features import LABEL_NAMES
        label_names = list(LABEL_NAMES)

    n_seqs = len(features_dict)
    print(f"  Found {n_seqs} sequences in split '{split}'")
    if n_seqs == 0:
        warnings.warn(
            f"No sequences found for split='{split}' in {npz_path}.",
            RuntimeWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # 1b. Load and apply memory sidecar (optional — DAM-style 9-dim features)
    # ------------------------------------------------------------------
    eval_memory_features: dict[str, np.ndarray] = {}
    _ckpt_feature_names: list[str] | None = None  # filled below if sidecar + checkpoint agree
    if memory_sidecar_path and Path(memory_sidecar_path).exists():
        mem_npz = np.load(memory_sidecar_path, allow_pickle=True)
        for k in mem_npz.files:
            if k.startswith("memory_features/"):
                seq = k[len("memory_features/"):]
                eval_memory_features[seq] = mem_npz[k].astype(np.float32)
        print(f"  Loaded memory features for {len(eval_memory_features)} sequences")

        # Subset columns to match the features the checkpoint was trained on.
        # Read memory_feature_names from checkpoint; fall back to all dims if absent.
        if checkpoint_path:
            try:
                import torch as _torch
                _ckpt_state = _torch.load(checkpoint_path, map_location="cpu")
                if isinstance(_ckpt_state, dict):
                    _ckpt_feature_names = _ckpt_state.get("memory_feature_names", None)
            except Exception:
                pass
        if _ckpt_feature_names:
            _all_feat_names = list(mem_npz.get("memory_feature_names", [f"dim_{i}" for i in range(9)]))
            _selected_indices = [_all_feat_names.index(n) for n in _ckpt_feature_names]
            for seq in eval_memory_features:
                eval_memory_features[seq] = eval_memory_features[seq][:, _selected_indices]
            print(
                f"  Subsetting memory features to {len(_ckpt_feature_names)} dims "
                f"(from checkpoint): {_ckpt_feature_names}"
            )
    elif memory_sidecar_path:
        print(f"  Memory sidecar not found at {memory_sidecar_path}, using base features only")

    # ------------------------------------------------------------------
    # 2. Load model and run inference
    # ------------------------------------------------------------------
    n_features = next(iter(features_dict.values())).shape[1] if features_dict else 28
    n_labels = len(label_names)

    model = _load_model(checkpoint_path, n_features, n_labels, device)

    # Build augmented features dict for inference when model expects memory dims.
    # If model has memory_dim > 0 and sidecar was loaded, concatenate per-sequence.
    # Fail-fast if checkpoint expects memory but no sidecar was provided or too many
    # sequences are missing.  Zero-pad only for the <10% missing case.
    infer_features_dict = features_dict
    _n_sequences_with_memory = 0  # tracks sequences that had actual sidecar entries
    if model is not None and getattr(model, "memory_dim", 0) > 0:
        mem_dim = model.memory_dim
        if not eval_memory_features:
            # No sidecar loaded at all — distinguish no-arg vs file-not-found
            raise ValueError(
                f"Checkpoint was trained with memory_dim={mem_dim} but "
                + (
                    "no --memory-sidecar was provided. "
                    "Re-run with --memory-sidecar pointing to the correct sidecar NPZ."
                    if not memory_sidecar_path
                    else f"the sidecar file was not found at {memory_sidecar_path!r}. "
                    "Check the path and re-run."
                )
            )

        # Count sequences that are missing from the sidecar
        n_total = len(features_dict)
        n_missing = sum(1 for k in features_dict if k not in eval_memory_features)
        if n_missing > 0:
            missing_frac = n_missing / max(n_total, 1)
            if missing_frac > 0.10:
                raise ValueError(
                    f"{n_missing}/{n_total} sequences ({missing_frac*100:.1f}%) are missing "
                    f"from the memory sidecar at {memory_sidecar_path!r}. "
                    "More than 10% missing is not allowed — regenerate the sidecar or "
                    "check that sidecar keys match NPZ sequence keys."
                )
            else:
                print(
                    f"  WARNING: {n_missing}/{n_total} sequences missing from memory sidecar "
                    f"({missing_frac*100:.1f}%) — zero-padding those sequences."
                )

        aug: dict[str, np.ndarray] = {}
        _n_sequences_with_memory = 0
        for seq_key, feats in features_dict.items():
            if seq_key in eval_memory_features:
                mem = eval_memory_features[seq_key]
                if mem.shape[1] != mem_dim:
                    raise ValueError(
                        f"Memory sidecar width mismatch for sequence {seq_key!r}: "
                        f"checkpoint expects {mem_dim} dims but sidecar has {mem.shape[1]}. "
                        "Re-run memory_features.py with the correct --memory-feature-names selection."
                    )
                T = min(feats.shape[0], mem.shape[0])
                if T < feats.shape[0]:
                    pad = np.zeros((feats.shape[0] - T, mem.shape[1]), dtype=np.float32)
                    mem_full = np.concatenate([mem[:T], pad], axis=0)
                else:
                    mem_full = mem[:feats.shape[0]]
                aug[seq_key] = np.concatenate([feats, mem_full], axis=1)
                _n_sequences_with_memory += 1
            else:
                # Zero-pad for sequences not in sidecar (<10% missing, already validated above)
                aug[seq_key] = np.concatenate(
                    [feats, np.zeros((feats.shape[0], mem_dim), dtype=np.float32)],
                    axis=1,
                )
        infer_features_dict = aug

    preds_dict = _run_inference(model, infer_features_dict, window_size, device)
    has_preds = bool(preds_dict)

    # Derive actual head names from the loaded model (works for v0 and v1 schemas).
    # Fallback to module-level default when no model is available.
    if model is not None:
        _model_head_names: list[str] = list(model.heads.keys())
    else:
        _model_head_names = list(_HEAD_NAMES)

    # Labels not predicted by the model (e.g. "correct") get a 0.5 placeholder —
    # track them explicitly so metrics output is clearly marked as "no model prediction".
    _labels_without_model_pred: set[str] = set(label_names) - set(_model_head_names)

    # predictions_output is saved AFTER calibration (step 3.5) so that
    # exported JSON always reflects the calibrated probabilities.

    # ------------------------------------------------------------------
    # 3. Aggregate labels and predictions across sequences
    # ------------------------------------------------------------------
    # Aggregate all frames per head; also keep per-sequence AUPRC(false_confirmed)
    all_y_true: dict[str, list[np.ndarray]] = {h: [] for h in label_names}
    all_y_pred: dict[str, list[np.ndarray]] = {h: [] for h in label_names}
    per_seq_auprc_fc: list[float] = []  # for bootstrap CI

    fc_idx = label_names.index("false_confirmed") if "false_confirmed" in label_names else -1

    for seq_key in features_dict:
        lab = labels_dict[seq_key].astype(float)  # (n_frames, n_labels)
        if has_preds and seq_key in preds_dict:
            pred = preds_dict[seq_key]  # (n_frames, n_heads)
        else:
            # No model available — use label frequency as a trivial baseline
            pred = np.tile(lab.mean(axis=0), (len(lab), 1))

        for hi, head in enumerate(label_names):
            if hi >= lab.shape[1]:
                break
            all_y_true[head].append(lab[:, hi])
            # Use name-based prediction lookup (model predicts HEAD_NAMES only,
            # not "correct"). Avoids off-by-one when "correct" is label index 0.
            if has_preds and seq_key in preds_dict:
                p = preds_dict[seq_key]  # (n_frames, n_head_preds)
                try:
                    if head in _model_head_names:
                        pred_col = _model_head_names.index(head)
                        all_y_pred[head].append(p[:, pred_col] if pred_col < p.shape[1] else np.full(len(lab), 0.5, np.float32))
                    else:
                        all_y_pred[head].append(np.full(len(lab), 0.5, np.float32))
                except Exception:
                    all_y_pred[head].append(p[:, hi] if hi < p.shape[1] else np.full(len(lab), 0.5, np.float32))
            else:
                all_y_pred[head].append(np.full(len(lab), 0.5, dtype=np.float32))

        # Per-sequence AUPRC for false_confirmed (bootstrap input)
        if fc_idx >= 0:
            yt = lab[:, fc_idx]
            if has_preds and seq_key in preds_dict:
                try:
                    fc_pred_idx = _model_head_names.index("false_confirmed")
                    yp = preds_dict[seq_key][:, fc_pred_idx]
                except (ValueError, IndexError):
                    yp = preds_dict[seq_key][:, fc_idx] if fc_idx < preds_dict[seq_key].shape[1] else np.full(len(yt), yt.mean(), np.float32)
            else:
                yp = np.full(len(yt), yt.mean(), dtype=np.float32)
            br = float(yt.mean())
            if 0.0 < br < 1.0:
                try:
                    per_seq_auprc_fc.append(
                        float(_average_precision_score(yt, yp))
                    )
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # 3.5. Temperature calibration — reliability heads only (val split)
    # ------------------------------------------------------------------
    temperatures: dict[str, float] = {}
    metrics_before_cal: dict[str, dict[str, float]] = {}

    if calibrate_heads:
        print("\n--- Temperature scaling ---")
        for head in calibrate_heads:
            if head not in label_names:
                print(f"  {head}: not in label_names — skipping")
                continue
            if not all_y_true.get(head) or not all_y_pred.get(head):
                continue
            y_true_cat = np.concatenate(all_y_true[head])
            y_pred_cat = np.concatenate(all_y_pred[head])
            if len(y_true_cat) == 0:
                continue
            br = float(y_true_cat.mean())
            if br == 0.0 or br == 1.0:
                print(f"  {head}: base_rate={br:.4f} — degenerate, skipping calibration")
                continue

            metrics_before_cal[head] = compute_head_metrics(y_true_cat, y_pred_cat, head)
            T = calibrate_temperature(y_true_cat, y_pred_cat)
            temperatures[head] = T

            # Apply T in-place to per-sequence prediction lists (step 4 sees calibrated probs)
            all_y_pred[head] = [apply_temperature(arr, T) for arr in all_y_pred[head]]

            # Also calibrate the predictions JSON output (preds_dict)
            if preds_dict and head in _model_head_names:
                head_col = _model_head_names.index(head)
                for seq_key in preds_dict:
                    preds_dict[seq_key][:, head_col] = apply_temperature(
                        preds_dict[seq_key][:, head_col], T
                    )

            ece_before = metrics_before_cal[head]["ece"]
            print(f"  {head:<25}  T={T:.4f}  ECE_before={ece_before:.4f}")

    # Save predictions after calibration so the JSON reflects calibrated probs.
    if predictions_output and preds_dict:
        _save_predictions_json(preds_dict, predictions_output, head_names=_model_head_names)

    # ------------------------------------------------------------------
    # 4. Compute per-head metrics
    # ------------------------------------------------------------------
    head_metrics: dict[str, dict[str, float]] = {}
    for head in label_names:
        y_true_cat = np.concatenate(all_y_true[head]) if all_y_true[head] else np.array([])
        y_pred_cat = np.concatenate(all_y_pred[head]) if all_y_pred[head] else np.array([])
        model_predicted = head not in _labels_without_model_pred
        if len(y_true_cat) == 0:
            head_metrics[head] = {
                "base_rate": float("nan"),
                "auroc": float("nan"),
                "auprc": float("nan"),
                "ece": float("nan"),
                "brier": float("nan"),
                "nll": float("nan"),
                "recall_at_5pct_fpr": float("nan"),
                "model_predicted": model_predicted,
            }
        else:
            m = compute_head_metrics(y_true_cat, y_pred_cat, head)
            m["model_predicted"] = model_predicted
            if not model_predicted:
                m["note"] = "0.5 baseline — no model head for this label"
            head_metrics[head] = m

    per_dataset_head_metrics = compute_per_dataset_head_metrics(
        labels_dict=labels_dict,
        preds_dict=preds_dict if has_preds else {},
        label_names=label_names,
        model_head_names=_model_head_names if has_preds else [],
    )

    # ------------------------------------------------------------------
    # 5. NT2F at IoU thresholds 0.5 and 0.2
    # ------------------------------------------------------------------
    nt2f_05 = compute_nt2f(iou_dict, iou_threshold=0.5)
    nt2f_02 = compute_nt2f(iou_dict, iou_threshold=0.2)

    # ------------------------------------------------------------------
    # 6. Bootstrap CI on per-sequence AUPRC(false_confirmed)
    # ------------------------------------------------------------------
    if len(per_seq_auprc_fc) >= 2:
        ci_lo, ci_hi = bootstrap_ci(per_seq_auprc_fc)
    else:
        ci_lo, ci_hi = float("nan"), float("nan")

    bootstrap_result = {
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_sequences": len(per_seq_auprc_fc),
    }

    # ------------------------------------------------------------------
    # 7. Print formatted table
    # ------------------------------------------------------------------
    _print_summary_table(head_metrics, label_names)

    if "false_confirmed" in label_names and per_dataset_head_metrics:
        print("Per-dataset false_confirmed:")
        for dataset_name, ds_metrics in per_dataset_head_metrics.items():
            m = ds_metrics.get("false_confirmed", {})
            print(
                f"  {dataset_name:<14} "
                f"seqs={int(m.get('n_sequences', 0)):>3} "
                f"frames={int(m.get('n_frames', 0)):>6} "
                f"base={m.get('base_rate', float('nan'))*100:>5.1f}% "
                f"AUROC={m.get('auroc', float('nan')):>6.3f} "
                f"AUPRC={m.get('auprc', float('nan')):>6.3f}"
            )

    print(f"NT2F (IoU>=0.5):  mean={nt2f_05['nt2f_mean']:.4f}  "
          f"std={nt2f_05['nt2f_std']:.4f}  "
          f"n={nt2f_05['n_sequences']}  "
          f"never_failed={nt2f_05['n_never_failed']}")
    print(f"NT2F (IoU>=0.2):  mean={nt2f_02['nt2f_mean']:.4f}  "
          f"std={nt2f_02['nt2f_std']:.4f}  "
          f"n={nt2f_02['n_sequences']}  "
          f"never_failed={nt2f_02['n_never_failed']}")

    if not np.isnan(ci_lo):
        print(
            f"Bootstrap AUPRC(false_confirmed) 95%CI:  "
            f"[{ci_lo:.4f}, {ci_hi:.4f}]  "
            f"(n_seq={len(per_seq_auprc_fc)})"
        )

    # ------------------------------------------------------------------
    # 8. Assemble results dict
    # ------------------------------------------------------------------
    _ckpt_memory_dim = int(getattr(model, "memory_dim", 0)) if model is not None else 0
    _used_feature_names: list[str] = list(_ckpt_feature_names) if _ckpt_feature_names else []
    results: dict[str, Any] = {
        "split": split,
        "n_sequences": n_seqs,
        "head_metrics": head_metrics,
        "per_dataset_head_metrics": per_dataset_head_metrics,
        "nt2f_05": nt2f_05,
        "nt2f_02": nt2f_02,
        "bootstrap_auprc_false_confirmed": bootstrap_result,
        "provenance": _build_provenance(npz_path, checkpoint_path, split),
        "memory_sidecar_path": memory_sidecar_path or None,
        "memory_sidecar_md5": _file_md5(memory_sidecar_path) if memory_sidecar_path else None,
        "memory_feature_names_used": _used_feature_names,
        "memory_dim": _ckpt_memory_dim,
        "n_sequences_with_memory": _n_sequences_with_memory,
    }

    if calibrate_heads and temperatures:
        cal_summary: dict[str, Any] = {
            "heads_calibrated": list(temperatures.keys()),
            "temperatures": temperatures,
            "metrics_before": metrics_before_cal,
        }
        for head in temperatures:
            ece_after = head_metrics.get(head, {}).get("ece", float("nan"))
            ece_before = metrics_before_cal.get(head, {}).get("ece", float("nan"))
            print(f"  {head:<25}  ECE: {ece_before:.4f} → {ece_after:.4f}  (Δ={ece_after-ece_before:+.4f})")
        results["calibration"] = cal_summary

    # Lead-time analysis for all three IFD horizons (v1/v2 schema only; skipped silently for v0)
    if preds_dict:
        lead_time_result = run_lead_time_analysis(
            iou_dict=iou_dict,
            preds_dict=preds_dict,
            labels_dict=labels_dict,
            label_names=label_names,
            head_names=_model_head_names,
        )
        # Only include and print if at least one horizon was actually computed
        non_skip = {k: v for k, v in lead_time_result.items() if "note" not in v}
        if non_skip:
            results["failure_lead_time"] = lead_time_result
            for hz_key, hz_res in non_skip.items():
                print(
                    f"\nFailure lead-time ({hz_key}):  "
                    f"events={hz_res['n_failure_events']}  "
                    f"detected={hz_res['n_detected_events']}  "
                    f"recall={hz_res['event_recall']:.3f}  "
                    f"median_lead={hz_res['median_lead_time']:.1f}f"
                )

    # Determine GO/NO-GO
    verdict = check_go_nogo(results)
    results["go_nogo"] = verdict

    # ------------------------------------------------------------------
    # 9. Save JSON if requested
    # ------------------------------------------------------------------
    if output_path is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        def _json_safe(obj: Any) -> Any:
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_json_safe(v) for v in obj]
            return obj

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(results), fh, indent=2)
        print(f"\nResults written to: {output_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Command-line entry point for SALT-RD evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SALTRD GRU checkpoint against an NPZ split."
    )
    parser.add_argument("--npz", required=True, help="Path to the NPZ dataset.")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the SALTRD .pt checkpoint."
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "diagnostic"],
        help="Dataset split to evaluate (default: val).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for JSON results.",
    )
    parser.add_argument(
        "--predictions-output",
        default=None,
        help="Path to save per-frame predictions JSON for policy.py --probs-json.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string (default: cpu).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="GRU temporal window size (default: 20).",
    )
    parser.add_argument(
        "--calibrate-heads",
        nargs="*",
        metavar="HEAD",
        default=None,
        help=(
            "Apply temperature scaling to these heads (val split only). "
            "Omit to skip calibration. Use --calibrate-heads without arguments "
            "to calibrate the default reliability heads: "
            f"{list(RELIABILITY_HEADS)}."
        ),
    )
    parser.add_argument(
        "--memory-sidecar",
        default="saltr/data/salt_rd_memory_sidecar.npz",
        help=(
            "Path to optional memory sidecar NPZ with keys memory_features/{seq} "
            "(float32, T×9).  If the file does not exist, evaluation uses 28-dim "
            "features only. (default: saltr/data/salt_rd_memory_sidecar.npz)"
        ),
    )
    args = parser.parse_args()

    calibrate_heads: list[str] | None = None
    if args.calibrate_heads is not None:
        calibrate_heads = args.calibrate_heads if args.calibrate_heads else list(RELIABILITY_HEADS)

    results = evaluate(
        npz_path=args.npz,
        checkpoint_path=args.checkpoint,
        split=args.split,
        window_size=args.window_size,
        device=args.device,
        output_path=args.output,
        predictions_output=args.predictions_output,
        calibrate_heads=calibrate_heads,
        memory_sidecar_path=args.memory_sidecar,
    )
    verdict = check_go_nogo(results)
    print(f"\nGO/NO-GO: {verdict}")


if __name__ == "__main__":
    main()
