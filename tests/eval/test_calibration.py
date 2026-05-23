"""Tests for csc_lib/eval/calibration.py"""
import numpy as np
import pytest

from csc_lib.eval.calibration import (
    brier_score,
    confidence_entropy,
    expected_calibration_error,
    pct_calibrated_high,
    pct_calibrated_low,
)


def test_ece_perfect_calibration():
    # Perfect multi-class: argmax matches y_true and confidence=1.0 → ECE=0
    probs = np.eye(3)[[0, 1, 2, 0, 1, 2]]
    y_true = np.array([0, 1, 2, 0, 1, 2])
    ece = expected_calibration_error(probs, y_true)
    assert ece == pytest.approx(0.0)


def test_ece_over_confident():
    probs = np.ones(100) * 0.9
    y_true = np.zeros(100, dtype=int)
    ece = expected_calibration_error(probs, y_true, n_bins=10)
    assert ece > 0.5


def test_ece_multiclass():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1, 1], size=200)
    y_true = probs.argmax(axis=1)
    ece = expected_calibration_error(probs, y_true)
    assert 0.0 <= ece <= 1.0


def test_brier_binary_perfect():
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    y_true = np.array([0, 0, 1, 1])
    assert brier_score(probs, y_true) == pytest.approx(0.0)


def test_brier_multiclass():
    probs = np.eye(3)
    y_true = np.array([0, 1, 2])
    assert brier_score(probs, y_true) == pytest.approx(0.0)


def test_pct_calibrated_low_high():
    conf = np.array([0.1, 0.2, 0.5, 0.7, 0.9])
    assert pct_calibrated_low(conf, 0.40) == pytest.approx(0.4)
    assert pct_calibrated_high(conf, 0.65) == pytest.approx(0.4)


def test_confidence_entropy_uniform():
    probs = np.full((10, 4), 0.25)
    h = confidence_entropy(probs)
    expected = float(np.log(4))
    assert h == pytest.approx(expected, rel=1e-5)
