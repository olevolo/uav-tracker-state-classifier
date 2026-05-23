"""SALT-RD feature schema definitions and helpers — v3 no-flow production.

The 28-dim feature vector has a fixed layout defined below.  Production schema
v3 (saltrd_v3_no_tsa_no_flow) zeroes flow features (indices 22-27) because
online inference cannot compute Farneback optical flow at runtime without
significant latency overhead, and offline flow extraction creates train/inference
distribution mismatch.

Flow features remain reserved/zero in production; indices are kept so that v2
checkpoints trained on the same 28-dim layout remain backward-compatible.

No imports from TSA or tracker state modules.  No thresholds of any kind.

Candidate feature schema v2 (8-dim) is defined separately at the bottom of
this file.  It is used only by offline collection (build_candidate_dataset.py)
and training (train_candidate_scorer.py / train_policy.py).  It must NOT be
imported into any runtime path.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

FEATURE_SCHEMA_VERSION: str = "saltrd_v3_no_tsa_no_flow"

# ---------------------------------------------------------------------------
# Feature names — 28 total, fixed order
# ---------------------------------------------------------------------------

BASE_FEATURE_NAMES: list[str] = [
    # 0-8  numeric evidence
    "apce",                   # 0
    "apce_norm",              # 1
    "psr",                    # 2
    "response_entropy",       # 3
    "peak_margin",            # 4
    "peak_width",             # 5
    "n_secondary",            # 6
    "peak_distance",          # 7
    "heatmap_mass_topk",      # 8
    # 9-12  rolling evidence
    "apce_ratio_5",           # 9
    "apce_ratio_20",          # 10
    "entropy_delta_5",        # 11
    "peak_margin_delta_5",    # 12
    # 13-14  v2 parity only
    "high_apce_streak_legacy",  # 13
    "low_apce_streak_legacy",   # 14
    # 15-21  numeric evidence (motion / geometry)
    "bbox_vx",                # 15
    "bbox_vy",                # 16
    "bbox_speed",             # 17
    "bbox_accel",             # 18
    "bbox_scale_ratio",       # 19
    "bbox_aspect_delta",      # 20
    "dist_to_border",         # 21
    # 22-27  flow features — zero in production
    "global_flow_mag",        # 22
    "target_flow_mag",        # 23
    "ego_motion_residual",    # 24
    "flow_iou",               # 25
    "flow_residual",          # 26
    "flow_consistency",       # 27
]

assert len(BASE_FEATURE_NAMES) == 28, "BASE_FEATURE_NAMES must have exactly 28 entries"

# ---------------------------------------------------------------------------
# Flow / production-zero indices
# ---------------------------------------------------------------------------

FLOW_FEATURE_INDICES: tuple[int, ...] = (22, 23, 24, 25, 26, 27)

# Alias: these are the only indices zeroed in production v3
PRODUCTION_ZERO_FEATURE_INDICES: tuple[int, ...] = FLOW_FEATURE_INDICES

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def feature_names(schema: str = FEATURE_SCHEMA_VERSION) -> list[str]:
    """Return the ordered list of feature names for *schema*.

    Currently only one schema (``FEATURE_SCHEMA_VERSION``) is defined; the
    argument is accepted for forward-compatibility.

    Args:
        schema: Schema version string (default: production v3).

    Returns:
        List of 28 feature name strings.
    """
    # Only one schema defined; accept any string for forward-compat but always
    # return the canonical v3 names (the layout has not changed since v2).
    return list(BASE_FEATURE_NAMES)


def zero_production_features(x: np.ndarray) -> np.ndarray:
    """Return a copy of *x* with flow feature columns zeroed.

    Zeroes the columns at ``PRODUCTION_ZERO_FEATURE_INDICES`` (22-27).
    The input array is **never** mutated.

    Args:
        x: Float array with last dimension == 28.  Any number of leading
           dimensions is supported (e.g. shape ``(28,)`` or ``(N, 28)``).

    Returns:
        Copy of *x* with flow indices zeroed.
    """
    out = x.copy()
    out[..., list(PRODUCTION_ZERO_FEATURE_INDICES)] = 0.0
    return out


def validate_feature_matrix(x: np.ndarray, expected_dim: int = 28) -> None:
    """Validate that *x* has the expected feature dimension.

    Accepts both 1-D vectors of shape ``(expected_dim,)`` and 2-D matrices of
    shape ``(N, expected_dim)``.

    Args:
        x: Array to validate.
        expected_dim: Expected number of features (default: 28).

    Raises:
        ValueError: If the last dimension of *x* does not equal *expected_dim*,
                    or if *x* has more than 2 dimensions.
    """
    if x.ndim not in (1, 2):
        raise ValueError(
            f"Feature array must be 1-D or 2-D, got ndim={x.ndim}"
        )
    actual = x.shape[-1]
    if actual != expected_dim:
        raise ValueError(
            f"Expected feature dimension {expected_dim}, got {actual}"
        )


def schema_metadata() -> dict[str, Any]:
    """Return a metadata dict describing the active production schema.

    Returns:
        Dictionary with keys:
        - ``"feature_schema"``  : schema version string
        - ``"n_features"``      : total number of features (28)
        - ``"zero_indices"``    : tuple of indices zeroed in production
        - ``"names"``           : ordered list of feature names
    """
    return {
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "n_features": len(BASE_FEATURE_NAMES),
        "zero_indices": PRODUCTION_ZERO_FEATURE_INDICES,
        "names": list(BASE_FEATURE_NAMES),
    }


# ---------------------------------------------------------------------------
# Candidate feature schema v2 (8 features) — OFFLINE USE ONLY
# Used by build_candidate_dataset.py and train_candidate_scorer.py.
# Must NOT be imported into controller.py or any other runtime path.
# ---------------------------------------------------------------------------

# Feature indices — v2 (8 features)
IDX_SCORE_MAP_SCORE    = 0
IDX_BBOX_H             = 1
IDX_FRAME_AREA_RATIO   = 2
IDX_BBOX_W             = 3
IDX_DIST_FROM_LAST     = 4
IDX_CROP_SIM           = 5
IDX_ASPECT_RATIO_DELTA = 6
IDX_SIZE_DELTA_RATIO   = 7

FEATURE_DIM = 8

FEATURE_NAMES = [
    "score_map_score",
    "bbox_h",
    "frame_area_ratio",
    "bbox_w",
    "dist_from_last",
    "crop_sim",
    "aspect_ratio_delta",
    "size_delta_ratio",
]

assert len(FEATURE_NAMES) == FEATURE_DIM, "FEATURE_NAMES must have exactly 8 entries"
