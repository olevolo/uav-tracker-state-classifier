"""Tracker telemetry — feature schema and per-frame numeric extraction.

Reused from prior SALT-RD work; the 28-dim feature schema and the
``EvidenceFrame`` extractor are kept as-is so existing offline NPZ artifacts
remain readable. The CSC label-generation step layers state labels on top of
this schema using ground-truth IoU and occlusion flags.
"""

from csc_uav_tracking.telemetry.schema import (
    BASE_FEATURE_NAMES,
    FEATURE_DIM as CANDIDATE_FEATURE_DIM,
    FEATURE_NAMES as CANDIDATE_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    validate_feature_matrix,
)

PRODUCTION_FEATURE_DIM = 28

__all__ = [
    "BASE_FEATURE_NAMES",
    "CANDIDATE_FEATURE_DIM",
    "CANDIDATE_FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "PRODUCTION_FEATURE_DIM",
    "validate_feature_matrix",
]
