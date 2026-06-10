"""Offline scene-state label generation for CSC training (composite schema)."""

from csc_lib.csc.labeling.label_schema import (
    AUX_FLAGS,
    CONFIDENCE_NAMES,
    DERIVED_NAMES,
    LOCALIZATION_NAMES,
    NUM_CONFIDENCE_STATES,
    NUM_DERIVED_STATES,
    NUM_LOCALIZATION_STATES,
    STATE_NAMES,           # back-compat alias for DERIVED_NAMES
    NUM_STATES,            # back-compat alias for NUM_DERIVED_STATES
    ConfidenceState,
    DerivedState,
    FrameLabel,
    LabelSource,
    LocalizationState,
    SceneState,            # alias for DerivedState
    derive_state,
)
from csc_lib.csc.labeling.weak_labeler import LabelingThresholds, label_frame
from csc_lib.csc.labeling.sequence_labeler import label_sequence, summarize_label_distribution
from csc_lib.csc.labeling.risk_labeler import build_future_risk_labels, summarize_future_risk

__all__ = [
    "AUX_FLAGS",
    "CONFIDENCE_NAMES",
    "DERIVED_NAMES",
    "LOCALIZATION_NAMES",
    "NUM_CONFIDENCE_STATES",
    "NUM_DERIVED_STATES",
    "NUM_LOCALIZATION_STATES",
    "STATE_NAMES",
    "NUM_STATES",
    "ConfidenceState",
    "DerivedState",
    "FrameLabel",
    "LabelSource",
    "LabelingThresholds",
    "LocalizationState",
    "SceneState",
    "build_future_risk_labels",
    "derive_state",
    "label_frame",
    "label_sequence",
    "summarize_future_risk",
    "summarize_label_distribution",
]
