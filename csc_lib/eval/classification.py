"""Cat 1 — CSC classification metrics (thin wrappers + additions).

Re-exports core functions from scene_state_metrics and adds
balanced_accuracy / weighted_f1 / per_class_metrics for paper tables M1-M2.
"""
from __future__ import annotations

import numpy as np

from csc_lib.eval.custom_metrics.scene_state_metrics import (
    confusion_matrix,
    macro_f1,
    per_state_prf,
)

__all__ = [
    "balanced_accuracy",
    "confusion_matrix",
    "macro_f1",
    "per_class_metrics",
    "per_state_prf",
    "weighted_f1",
]


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-class recall (= sklearn balanced_accuracy_score).

    Missing classes (support == 0) are excluded from the mean.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recalls.append(float((y_pred[mask] == c).mean()))
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def weighted_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_states: int | None = None,
    state_names: list[str] | None = None,
) -> float:
    """Weighted-average F1 weighted by class support."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if n_states is None:
        n_states = int(max(y_true.max(), y_pred.max())) + 1
    prf = per_state_prf(y_true, y_pred, n_states=n_states, state_names=state_names)
    total_support = sum(v["support"] for v in prf.values())
    if total_support == 0:
        return 0.0
    weighted_sum = sum(v["f1"] * v["support"] for v in prf.values())
    return float(weighted_sum / total_support)


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    n_states: int | None = None,
) -> dict[str, dict[str, float]]:
    """Per-class precision / recall / F1 / support.

    Wraps ``per_state_prf`` with optional explicit label list.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if n_states is None:
        n_states = int(max(y_true.max(), y_pred.max())) + 1
    return per_state_prf(y_true, y_pred, n_states=n_states, state_names=labels)
