"""Unit tests for saltr/src/salt_r/feature_schema.py — v3 no-flow schema."""
from __future__ import annotations

import numpy as np
import pytest

from salt_r.feature_schema import (
    BASE_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FLOW_FEATURE_INDICES,
    PRODUCTION_ZERO_FEATURE_INDICES,
    feature_names,
    schema_metadata,
    validate_feature_matrix,
    zero_production_features,
)


# ---------------------------------------------------------------------------
# 1. BASE_FEATURE_NAMES length
# ---------------------------------------------------------------------------

def test_base_feature_names_length():
    assert len(BASE_FEATURE_NAMES) == 28


# ---------------------------------------------------------------------------
# 2. Flow indices all present in FLOW_FEATURE_INDICES
# ---------------------------------------------------------------------------

def test_flow_indices_cover_22_to_27():
    for idx in range(22, 28):
        assert idx in FLOW_FEATURE_INDICES, f"Index {idx} missing from FLOW_FEATURE_INDICES"


# ---------------------------------------------------------------------------
# 3. zero_production_features zeros exactly the flow indices
# ---------------------------------------------------------------------------

def test_zero_production_features_zeroes_flow_indices():
    x = np.ones((5, 28), dtype=np.float32)
    result = zero_production_features(x)

    # Flow indices must be zero
    for idx in PRODUCTION_ZERO_FEATURE_INDICES:
        assert np.all(result[:, idx] == 0.0), f"Index {idx} should be zero"

    # Non-flow indices must remain 1
    non_flow = [i for i in range(28) if i not in PRODUCTION_ZERO_FEATURE_INDICES]
    for idx in non_flow:
        assert np.all(result[:, idx] == 1.0), f"Index {idx} should remain unchanged"


# ---------------------------------------------------------------------------
# 4. zero_production_features does NOT mutate input
# ---------------------------------------------------------------------------

def test_zero_production_features_no_mutation():
    x = np.ones((5, 28), dtype=np.float32)
    x_copy = x.copy()
    _ = zero_production_features(x)
    np.testing.assert_array_equal(x, x_copy, err_msg="Input array was mutated")


# ---------------------------------------------------------------------------
# 5. validate_feature_matrix passes for (N, 28) and (28,)
# ---------------------------------------------------------------------------

def test_validate_feature_matrix_2d_ok():
    x = np.zeros((10, 28), dtype=np.float32)
    validate_feature_matrix(x)  # should not raise


def test_validate_feature_matrix_1d_ok():
    x = np.zeros((28,), dtype=np.float32)
    validate_feature_matrix(x)  # should not raise


# ---------------------------------------------------------------------------
# 6. validate_feature_matrix raises ValueError for wrong dim
# ---------------------------------------------------------------------------

def test_validate_feature_matrix_wrong_dim_2d():
    x = np.zeros((10, 27), dtype=np.float32)
    with pytest.raises(ValueError, match="28"):
        validate_feature_matrix(x)


def test_validate_feature_matrix_wrong_dim_1d():
    x = np.zeros((30,), dtype=np.float32)
    with pytest.raises(ValueError, match="28"):
        validate_feature_matrix(x)


# ---------------------------------------------------------------------------
# 7. schema_metadata includes "feature_schema" == FEATURE_SCHEMA_VERSION
# ---------------------------------------------------------------------------

def test_schema_metadata_version_key():
    meta = schema_metadata()
    assert "feature_schema" in meta
    assert meta["feature_schema"] == FEATURE_SCHEMA_VERSION
    assert meta["feature_schema"] == "saltrd_v3_no_tsa_no_flow"


# ---------------------------------------------------------------------------
# 8. feature_names() returns list of length 28
# ---------------------------------------------------------------------------

def test_feature_names_length():
    names = feature_names()
    assert isinstance(names, list)
    assert len(names) == 28
