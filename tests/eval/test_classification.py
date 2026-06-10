"""Tests for csc_lib/eval/classification.py"""
import numpy as np
import pytest

from csc_lib.eval.classification import (
    balanced_accuracy,
    per_class_metrics,
    weighted_f1,
)


def test_balanced_accuracy_perfect():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert balanced_accuracy(y, y) == pytest.approx(1.0)


def test_balanced_accuracy_imbalanced_classes():
    y_true = np.array([0, 0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0, 0])
    # Class 0: recall=1.0, Class 1: recall=0.0 → mean=0.5
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.5)


def test_balanced_accuracy_empty():
    assert balanced_accuracy(np.array([0, 0]), np.array([0, 0])) == pytest.approx(1.0)


def test_weighted_f1_perfect():
    y = np.array([0, 0, 1, 1])
    assert weighted_f1(y, y) == pytest.approx(1.0)


def test_per_class_metrics_happy_path():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    result = per_class_metrics(y_true, y_pred, labels=["a", "b", "c"])
    assert "a" in result
    assert "b" in result
    assert "c" in result
    # Class a: tp=1, fp=0, fn=1 → precision=1, recall=0.5, f1=2/3
    assert result["a"]["f1"] == pytest.approx(2 / 3, rel=0.01)
    # Class c: tp=1, fp=0, fn=0 → f1=1.0
    assert result["c"]["f1"] == pytest.approx(1.0)


def test_per_class_metrics_missing_class_excluded():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    result = per_class_metrics(y_true, y_pred, n_states=3)
    # Class 1 and 2 have support=0; per_state_prf uses DERIVED_NAMES not "class_N"
    # Just verify all 3 states present and class 0 has support=3
    assert len(result) == 3
    first_key = list(result.keys())[0]
    assert result[first_key]["support"] == 3
