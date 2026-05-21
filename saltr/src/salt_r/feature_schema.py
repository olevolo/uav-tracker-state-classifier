"""SALT-RD feature schema definitions and helpers.

The 28-dim feature vector has a fixed layout defined in collect_features.FEATURE_NAMES.
Production schema v1 (saltrd_v2_online_no_flow) zeroes flow features (indices 22-27)
because online inference cannot compute Farneback optical flow at runtime without
significant latency overhead, and offline flow extraction creates train/inference
distribution mismatch.

Flow features remain available for offline analysis and teacher channels only.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np

# Production v1: no runtime flow. Indices 22-27 reserved/zero.
FLOW_FEATURE_INDICES: list[int] = [22, 23, 24, 25, 26, 27]
FLOW_FEATURE_NAMES: list[str] = [
    "global_flow_mag",
    "target_flow_mag",
    "ego_motion_residual",
    "flow_iou",
    "flow_residual",
    "flow_consistency",
]

SCHEMA_DROP_INDICES: dict[str, list[int]] = {
    "saltrd_v2_online_no_flow": FLOW_FEATURE_INDICES,
    "legacy_v2": [],                     # no zeroing (original v2_corrected/v2_retrained)
}

PRODUCTION_SCHEMA = "saltrd_v2_online_no_flow"


def get_drop_indices(schema: str) -> list[int]:
    """Return feature indices to zero for the given schema name."""
    return SCHEMA_DROP_INDICES.get(schema, [])


def apply_feature_schema(
    features: np.ndarray,
    schema_or_indices: str | Sequence[int],
) -> np.ndarray:
    """Zero out features that are not part of the given production schema.

    Args:
        features: (..., 28) float32 array - any number of leading dimensions.
        schema_or_indices: schema name (str) or explicit list of int indices.

    Returns:
        Copy of features with zeroed columns (does NOT modify in-place).
    """
    if isinstance(schema_or_indices, str):
        indices = get_drop_indices(schema_or_indices)
    else:
        indices = list(schema_or_indices)

    if not indices:
        return features

    out = features.copy()
    out[..., indices] = 0.0
    return out
