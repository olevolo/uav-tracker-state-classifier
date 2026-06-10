"""Cat 4 — Calibration metrics.

ECE, Brier Score, confidence entropy, and distribution percentiles.
Operates on CSC model softmax probabilities, not raw tracker confidence.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "brier_score",
    "calibrated_distribution_stats",
    "confidence_entropy",
    "expected_calibration_error",
    "pct_calibrated_high",
    "pct_calibrated_low",
]


def pct_calibrated_low(
    calibrated_conf: np.ndarray, threshold: float = 0.40
) -> float:
    """Fraction of frames with calibrated confidence < threshold."""
    c = np.asarray(calibrated_conf, dtype=np.float64)
    valid = np.isfinite(c)
    if not valid.any():
        return 0.0
    return float((c[valid] < threshold).mean())


def pct_calibrated_high(
    calibrated_conf: np.ndarray, threshold: float = 0.65
) -> float:
    """Fraction of frames with calibrated confidence >= threshold."""
    c = np.asarray(calibrated_conf, dtype=np.float64)
    valid = np.isfinite(c)
    if not valid.any():
        return 0.0
    return float((c[valid] >= threshold).mean())


def calibrated_distribution_stats(
    calibrated_conf: np.ndarray,
) -> dict[str, float]:
    """Mean / median / q25 / q75 / q95 / min / max of calibrated confidence."""
    c = np.asarray(calibrated_conf, dtype=np.float64)
    valid = c[np.isfinite(c)]
    if len(valid) == 0:
        nan = float("nan")
        return {k: nan for k in ["mean", "median", "q25", "q75", "q95", "min", "max"]}
    return {
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "q25": float(np.percentile(valid, 25)),
        "q75": float(np.percentile(valid, 75)),
        "q95": float(np.percentile(valid, 95)),
        "min": float(valid.min()),
        "max": float(valid.max()),
    }


def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Standard ECE for multi-class or binary softmax probabilities.

    ``probs``: shape (N,) for binary or (N, C) for multi-class softmax.
    ``y_true``: shape (N,) int class labels.

    For multi-class, uses the max-probability class and its confidence.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    if probs.ndim == 2:
        confidence = probs.max(axis=1)
        predicted = probs.argmax(axis=1)
        correct = (predicted == y_true).astype(np.float64)
    else:
        confidence = probs
        predicted = (probs >= 0.5).astype(np.int64)
        correct = (predicted == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(confidence)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidence >= lo) & (confidence < hi)
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        if not mask.any():
            continue
        acc = correct[mask].mean()
        conf = confidence[mask].mean()
        ece += float(mask.sum()) / N * abs(acc - conf)
    return float(ece)


def brier_score(
    probs: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Multi-class Brier score: mean squared error of probability vectors.

    ``probs``: shape (N,) for binary or (N, C) for multi-class.
    ``y_true``: shape (N,) int class labels.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    if probs.ndim == 1:
        targets = y_true.astype(np.float64)
        return float(np.mean((probs - targets) ** 2))

    N, C = probs.shape
    one_hot = np.zeros((N, C), dtype=np.float64)
    one_hot[np.arange(N), y_true] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def confidence_entropy(probs: np.ndarray) -> float:
    """Mean Shannon entropy of softmax probability distributions.

    ``probs``: shape (N, C) multi-class or (N,) binary.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        p = np.clip(probs, 1e-12, 1.0)
        q = 1.0 - p
        q = np.clip(q, 1e-12, 1.0)
        h = -(p * np.log(p) + q * np.log(q))
        return float(np.mean(h))

    p = np.clip(probs, 1e-12, 1.0)
    h = -np.sum(p * np.log(p), axis=1)
    return float(np.mean(h))
