"""CSC / scene-state classifier evaluation metrics.

All inputs are 1-D arrays of integer state IDs (predicted vs target)
plus optional float ``risk_score`` arrays for failure AUROC / AUPRC.
"""
from __future__ import annotations

import numpy as np

from csc_lib.csc.labeling.label_schema import (
    DERIVED_NAMES,
    DerivedState,
    NUM_DERIVED_STATES,
)

# Back-compat aliases
NUM_STATES = NUM_DERIVED_STATES
STATE_NAMES = DERIVED_NAMES
SceneState = DerivedState


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def per_state_prf(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_states: int = NUM_STATES,
    state_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-state precision / recall / F1.  Keyed by state name."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    names = state_names if state_names is not None else STATE_NAMES
    out: dict[str, dict[str, float]] = {}
    for s in range(n_states):
        tp = int(((y_pred == s) & (y_true == s)).sum())
        fp = int(((y_pred == s) & (y_true != s)).sum())
        fn = int(((y_pred != s) & (y_true == s)).sum())
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        out[names[s] if s < len(names) else f"class_{s}"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(tp + fn),
        }
    return out


def macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_states: int = NUM_STATES,
    state_names: list[str] | None = None,
) -> float:
    """Unweighted mean of per-state F1.  Excludes states with 0 support."""
    prf = per_state_prf(y_true, y_pred, n_states=n_states, state_names=state_names)
    f1s = [v["f1"] for v in prf.values() if v["support"] > 0]
    if not f1s:
        return 0.0
    return float(np.mean(f1s))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_states: int = NUM_STATES,
) -> np.ndarray:
    """Rows = true state, cols = predicted state."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((n_states, n_states), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


# ---------------------------------------------------------------------------
# Failure-risk binary metrics — failure = LOST_AWARE or FALSE_CONFIRMED.
# ---------------------------------------------------------------------------


_FAILURE_STATES = (
    int(DerivedState.LOST_AWARE),
    int(DerivedState.FALSE_CONFIRMED),
)


def states_to_failure(y_state: np.ndarray) -> np.ndarray:
    """Binary 0/1: 1 if derived state is LOST_AWARE or FALSE_CONFIRMED."""
    y = np.asarray(y_state, dtype=np.int64)
    return np.isin(y, _FAILURE_STATES).astype(np.int8)


def failure_auroc(y_true_failure: np.ndarray, risk_score: np.ndarray) -> float:
    """Compute AUROC without sklearn (rank-sum formula)."""
    y = np.asarray(y_true_failure, dtype=np.int8)
    s = np.asarray(risk_score, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    sum_pos_ranks = float(ranks[pos].sum())
    auroc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auroc)


def failure_auprc(y_true_failure: np.ndarray, risk_score: np.ndarray) -> float:
    """Average-precision approximation of AUPRC."""
    y = np.asarray(y_true_failure, dtype=np.int8)
    s = np.asarray(risk_score, dtype=np.float64)
    if y.sum() == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / max(1, int(y.sum()))
    # AP = sum_k (R_k - R_{k-1}) * P_k
    rec_prev = np.concatenate(([0.0], recall[:-1]))
    ap = float(((recall - rec_prev) * precision).sum())
    return ap


def false_alarm_rate(
    y_true_failure: np.ndarray,
    risk_score: np.ndarray,
    threshold: float,
) -> float:
    """Per-frame FP rate at a given risk threshold."""
    y = np.asarray(y_true_failure, dtype=np.int8)
    s = np.asarray(risk_score, dtype=np.float64)
    neg = y == 0
    if not neg.any():
        return 0.0
    return float((s[neg] >= threshold).mean())


def false_alarms_per_1000(
    y_true_failure: np.ndarray,
    risk_score: np.ndarray,
    threshold: float,
) -> float:
    """False alarms scaled to 1000 frames."""
    return 1000.0 * false_alarm_rate(y_true_failure, risk_score, threshold)


# ---------------------------------------------------------------------------
# Detection-delay / early-warning
# ---------------------------------------------------------------------------


def average_detection_delay(
    y_true_failure: np.ndarray,
    risk_score: np.ndarray,
    threshold: float,
) -> float:
    """Mean number of frames between failure onset and first risk crossing.

    For each contiguous failure episode we measure the gap between
    ``episode.start_frame`` and the first frame at-or-before that where
    ``risk_score >= threshold``.  Negative delay = early warning;
    positive delay = late detection.  Returns NaN if there are no
    failure episodes.
    """
    y = np.asarray(y_true_failure, dtype=np.int8)
    s = np.asarray(risk_score, dtype=np.float64)
    above = s >= threshold

    delays: list[float] = []
    in_fail = False
    start = -1
    for t in range(len(y)):
        if y[t] == 1 and not in_fail:
            in_fail = True
            start = t
            # Search for risk crossing strictly before failure onset.
            warn_t: int | None = None
            for tt in range(t - 1, -1, -1):
                if above[tt]:
                    warn_t = tt
                    break
                if y[tt] == 1:
                    break
            if warn_t is not None:
                delays.append(float(warn_t - start))  # negative = early
            else:
                # Risk did not cross before onset — find first crossing
                # at or after onset (non-negative delay).
                hit = np.where(above[t:])[0]
                if hit.size:
                    delays.append(float(hit[0]))
                else:
                    delays.append(float(len(y) - t))  # never detected
        elif y[t] == 0:
            in_fail = False
            start = -1

    if not delays:
        return float("nan")
    return float(np.mean(delays))


def early_warning_recall(
    y_true_failure: np.ndarray,
    risk_score: np.ndarray,
    threshold: float,
    k: int = 5,
) -> float:
    """Fraction of failure episodes flagged at least ``k`` frames in advance."""
    y = np.asarray(y_true_failure, dtype=np.int8)
    s = np.asarray(risk_score, dtype=np.float64)
    above = s >= threshold

    starts: list[int] = []
    in_fail = False
    for t in range(len(y)):
        if y[t] == 1 and not in_fail:
            starts.append(t)
            in_fail = True
        elif y[t] == 0:
            in_fail = False
    if not starts:
        return 0.0
    hits = 0
    for s0 in starts:
        lo = max(0, s0 - k)
        if above[lo:s0].any():
            hits += 1
    return hits / len(starts)
