"""Tests for csc_lib/eval/feature_quality.py"""
import numpy as np
import pytest

from csc_lib.eval.feature_quality import (
    feature_auroc,
    feature_cohens_d,
    feature_missing_rate,
)


def test_auroc_perfect_separation():
    rng = np.random.default_rng(0)
    X = np.column_stack([
        np.concatenate([rng.random(50), rng.random(50) + 2.0])
    ])
    y = np.array([0] * 50 + [1] * 50, dtype=np.int8)
    result = feature_auroc(X, y, feature_names=["feat0"])
    assert result["feat0"] > 0.95


def test_auroc_random():
    rng = np.random.default_rng(42)
    X = rng.random((100, 2))
    y = rng.integers(0, 2, size=100, dtype=np.int8)
    result = feature_auroc(X, y)
    for v in result.values():
        assert 0.0 <= v <= 1.0


def test_missing_rate_with_nan():
    X = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 4.0]])
    result = feature_missing_rate(X, feature_names=["a", "b"])
    assert result["a"] == pytest.approx(1 / 3)
    assert result["b"] == pytest.approx(1 / 3)


def test_cohens_d_sign():
    # Use more samples to ensure non-zero variance in each group
    X = np.array([[1.0], [1.1], [1.2], [3.0], [3.1], [3.2]])
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    result = feature_cohens_d(X, y, feature_names=["x"])
    # Positive class has higher mean → positive d
    assert result["x"] > 0
