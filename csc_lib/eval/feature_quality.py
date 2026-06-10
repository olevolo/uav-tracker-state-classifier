"""Cat 3 — Feature quality metrics (public library wrappers).

Promotes private helpers from tools/diagnose_csc_features.py into importable
library functions for compute_paper_metrics.py, tests, and future callers.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np

from csc_lib.eval.custom_metrics.scene_state_metrics import (
    failure_auprc,
    failure_auroc,
)

__all__ = [
    "feature_auroc",
    "feature_auprc",
    "feature_cohens_d",
    "feature_missing_rate",
    "feature_per_state_stats",
    "feature_group_ablation",
]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    na, nb = len(a), len(b)
    if na + nb <= 2:
        return float("nan")
    pooled_var = (
        (na - 1) * (float(a.var(ddof=1)) if na > 1 else 0.0)
        + (nb - 1) * (float(b.var(ddof=1)) if nb > 1 else 0.0)
    ) / (na + nb - 2)
    if pooled_var <= 0.0:
        return float("nan")
    return (float(a.mean()) - float(b.mean())) / math.sqrt(pooled_var)


def feature_auroc(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Per-feature univariate AUROC (best polarity).

    ``features``: shape (N, F).
    ``labels``: shape (N,) binary int.
    Returns dict mapping feature name → AUROC.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    N, F = features.shape
    names = list(feature_names) if feature_names is not None else [str(i) for i in range(F)]
    out: dict[str, float] = {}
    for fi in range(F):
        col = features[:, fi].copy()
        mask = np.isfinite(col)
        col[~mask] = 0.0
        a_pos = failure_auroc(labels, col)
        a_neg = failure_auroc(labels, -col)
        out[names[fi]] = max(a_pos, a_neg)
    return out


def feature_auprc(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Per-feature univariate AUPRC (best polarity)."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    N, F = features.shape
    names = list(feature_names) if feature_names is not None else [str(i) for i in range(F)]
    out: dict[str, float] = {}
    for fi in range(F):
        col = features[:, fi].copy()
        mask = np.isfinite(col)
        col[~mask] = 0.0
        a_pos = failure_auprc(labels, col)
        a_neg = failure_auprc(labels, -col)
        out[names[fi]] = max(a_pos, a_neg)
    return out


def feature_cohens_d(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Per-feature Cohen's d (positive class vs negative class)."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    N, F = features.shape
    names = list(feature_names) if feature_names is not None else [str(i) for i in range(F)]
    pos = labels == 1
    neg = labels == 0
    out: dict[str, float] = {}
    for fi in range(F):
        col = features[:, fi]
        valid = np.isfinite(col)
        a = col[valid & pos]
        b = col[valid & neg]
        out[names[fi]] = _cohens_d(a, b)
    return out


def feature_missing_rate(
    features: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Fraction of NaN or Inf values per feature column."""
    features = np.asarray(features, dtype=np.float64)
    N, F = features.shape
    names = list(feature_names) if feature_names is not None else [str(i) for i in range(F)]
    out: dict[str, float] = {}
    for fi in range(F):
        col = features[:, fi]
        missing = int((~np.isfinite(col)).sum())
        out[names[fi]] = missing / max(1, N)
    return out


def feature_per_state_stats(
    features: np.ndarray,
    states: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> dict[tuple[str, str], dict]:
    """Median / IQR per (feature, state) combination.

    ``states``: shape (N,) int or str array.
    Returns dict keyed by (feature_name, state_name) → {count, median, iqr, p10, p90}.
    """
    features = np.asarray(features, dtype=np.float64)
    states = np.asarray(states)
    N, F = features.shape
    names = list(feature_names) if feature_names is not None else [str(i) for i in range(F)]
    unique_states = np.unique(states)
    out: dict[tuple[str, str], dict] = {}
    for fi in range(F):
        col = features[:, fi].astype(np.float64)
        for s in unique_states:
            mask = np.isfinite(col) & (states == s)
            g = col[mask]
            sname = str(s)
            if len(g) == 0:
                out[(names[fi], sname)] = {"count": 0, "median": float("nan"), "iqr": float("nan"), "p10": float("nan"), "p90": float("nan")}
                continue
            out[(names[fi], sname)] = {
                "count": len(g),
                "median": float(np.median(g)),
                "iqr": float(np.percentile(g, 75) - np.percentile(g, 25)),
                "p10": float(np.percentile(g, 10)),
                "p90": float(np.percentile(g, 90)),
            }
    return out


def feature_group_ablation(
    X_groups: dict[str, np.ndarray],
    y: np.ndarray,
    models: list[str] | None = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Macro-F1 + F1_FC for each feature group using shallow classifiers.

    Parameters
    ----------
    X_groups : dict mapping group_name → (N, F_group) feature matrix.
    y : shape (N,) int labels (derived state).
    models : list of model names; supported: "LogReg", "RF".

    Returns pandas DataFrame with columns: group, model, macro_f1, f1_fc,
    balanced_accuracy, n_features.
    """
    import pandas as pd

    if models is None:
        models = ["LogReg", "RF"]

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ImportError("scikit-learn required for feature_group_ablation") from exc

    y = np.asarray(y, dtype=np.int64)
    N = len(y)
    # Simple 80/20 sequential split (not sequence-grouped; caller can supply pre-split)
    split = int(N * 0.8)
    y_train, y_val = y[:split], y[split:]

    rows = []
    for group_name, X in X_groups.items():
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_tr, X_vl = X[:split], X[split:]
        if len(np.unique(y_train)) < 2:
            continue
        for model_name in models:
            if model_name == "LogReg":
                clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, random_state=42)
            elif model_name == "RF":
                clf = RandomForestClassifier(max_depth=5, n_estimators=50, class_weight="balanced", random_state=42, n_jobs=1)
            else:
                continue
            clf.fit(X_tr, y_train)
            y_pred = clf.predict(X_vl)

            classes = np.unique(y_val)
            f1s = []
            for c in classes:
                tp = int(((y_pred == c) & (y_val == c)).sum())
                fp = int(((y_pred == c) & (y_val != c)).sum())
                fn = int(((y_pred != c) & (y_val == c)).sum())
                if tp + fp == 0 or tp + fn == 0:
                    f1s.append(0.0)
                    continue
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
            macro_f1_val = float(np.mean(f1s)) if f1s else 0.0

            fc_class = 3
            tp_fc = int(((y_pred == fc_class) & (y_val == fc_class)).sum())
            fp_fc = int(((y_pred == fc_class) & (y_val != fc_class)).sum())
            fn_fc = int(((y_pred != fc_class) & (y_val == fc_class)).sum())
            p_fc = tp_fc / (tp_fc + fp_fc) if (tp_fc + fp_fc) > 0 else 0.0
            r_fc = tp_fc / (tp_fc + fn_fc) if (tp_fc + fn_fc) > 0 else 0.0
            f1_fc = 2 * p_fc * r_fc / (p_fc + r_fc) if (p_fc + r_fc) > 0 else 0.0

            bal_acc_parts = [float((y_pred[y_val == c] == c).mean()) for c in classes if (y_val == c).sum() > 0]
            bal_acc = float(np.mean(bal_acc_parts)) if bal_acc_parts else 0.0

            rows.append({
                "group": group_name,
                "model": model_name,
                "n_features": X.shape[1],
                "macro_f1": round(macro_f1_val, 6),
                "f1_fc": round(f1_fc, 6),
                "balanced_accuracy": round(bal_acc, 6),
            })

    return pd.DataFrame(rows)
