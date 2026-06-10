"""Scene-state taxonomy with composite FALSE_CONFIRMED.

The paper centres on the FALSE_CONFIRMED phenomenon, but the labeling
schema preserves its **composite nature**: it is a derived state that
fires when ``localization_state == LOST`` AND ``confidence_state ==
HIGH_CONFIDENCE``.  This decomposition gives the model two independent
signals to learn — a geometry/quality signal for localization and a
telemetry signal for confidence — and avoids forcing it to memorise an
implicit conjunction.

Two orthogonal axes
-------------------

``LocalizationState`` (geometric, derived from IoU offline):
    STABLE     — IoU >= tau_confirmed (default 0.5)
    UNCERTAIN  — tau_uncertain <= IoU < tau_confirmed (default 0.2..0.5)
    LOST       — IoU < tau_lost (default 0.2) for >= K frames

``ConfidenceState`` (telemetric, runtime-available):
    HIGH_CONFIDENCE — tracker score above tau_high_conf
    LOW_CONFIDENCE  — otherwise

Derived 4-class paper state
---------------------------

``DerivedState`` is computed from the two axes:

    LocalizationState  | ConfidenceState  | DerivedState
    -------------------|------------------|-----------------
    STABLE             | (any)            | CORRECT_CONFIRMED
    UNCERTAIN          | (any)            | CORRECT_UNCERTAIN
    LOST               | LOW_CONFIDENCE   | LOST_AWARE
    LOST               | HIGH_CONFIDENCE  | FALSE_CONFIRMED  ← paper novelty

Auxiliary scene flags (multi-label, NOT mutually exclusive)
-----------------------------------------------------------

These describe scene conditions, not tracking states:

    occlusion       — dataset full-occlusion flag
    out_of_view     — dataset out-of-view flag (subset of occlusion)
    fast_motion     — large GT-centre displacement
    scale_change    — large GT-area ratio between consecutive frames
    distractor_risk — heuristic: low IoU + plausible-looking pred bbox
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Axis 1 — localization state (geometric, IoU-based)
# ---------------------------------------------------------------------------


class LocalizationState(IntEnum):
    STABLE = 0
    UNCERTAIN = 1
    LOST = 2


LOCALIZATION_NAMES: list[str] = [s.name for s in LocalizationState]
NUM_LOCALIZATION_STATES: int = len(LOCALIZATION_NAMES)


# ---------------------------------------------------------------------------
# Axis 2 — confidence state (telemetry-based)
# ---------------------------------------------------------------------------


class ConfidenceState(IntEnum):
    LOW_CONFIDENCE = 0
    HIGH_CONFIDENCE = 1


CONFIDENCE_NAMES: list[str] = [s.name for s in ConfidenceState]
NUM_CONFIDENCE_STATES: int = len(CONFIDENCE_NAMES)


# ---------------------------------------------------------------------------
# Derived paper-state (CORRECT_CONFIRMED / CORRECT_UNCERTAIN /
# LOST_AWARE / FALSE_CONFIRMED).  This is what tools/evaluate_*
# compare predictions against and what the paper reports.
# ---------------------------------------------------------------------------


class DerivedState(IntEnum):
    CORRECT_CONFIRMED = 0
    CORRECT_UNCERTAIN = 1
    LOST_AWARE = 2
    FALSE_CONFIRMED = 3


DERIVED_NAMES: list[str] = [s.name for s in DerivedState]
NUM_DERIVED_STATES: int = len(DERIVED_NAMES)


def derive_state(loc: LocalizationState, conf: ConfidenceState) -> DerivedState:
    """Compose the 2 axes into the paper's 4-class derived state."""
    if loc == LocalizationState.STABLE:
        return DerivedState.CORRECT_CONFIRMED
    if loc == LocalizationState.UNCERTAIN:
        return DerivedState.CORRECT_UNCERTAIN
    # loc == LOST
    if conf == ConfidenceState.HIGH_CONFIDENCE:
        return DerivedState.FALSE_CONFIRMED
    return DerivedState.LOST_AWARE


# ---------------------------------------------------------------------------
# Auxiliary scene flags (multi-label sigmoids)
# ---------------------------------------------------------------------------


AUX_FLAGS: tuple[str, ...] = (
    "occlusion",
    "out_of_view",
    "fast_motion",
    "scale_change",
    "distractor_risk",
)


# ---------------------------------------------------------------------------
# Label-source provenance
# ---------------------------------------------------------------------------


class LabelSource(IntEnum):
    GT_RULE = 0
    DATASET_ATTRIBUTE = 1
    HEURISTIC = 2
    MIXED = 3


# ---------------------------------------------------------------------------
# Frame label
# ---------------------------------------------------------------------------


@dataclass
class FrameLabel:
    """One supervised training row.

    All optional fields default to ``None`` / ``False`` and survive
    JSONL round-trip.
    """

    dataset: str
    split: str
    sequence: str
    frame_idx: int

    # Geometry
    gt_bbox: Optional[tuple[float, float, float, float]]
    pred_bbox: Optional[tuple[float, float, float, float]]
    iou: Optional[float]
    center_error: Optional[float]
    normalized_center_error: Optional[float]
    area_ratio: Optional[float]
    velocity: Optional[float]
    acceleration: Optional[float]

    # Dataset side info
    visible_ratio: Optional[float] = None
    absent: bool = False
    dataset_attributes: dict = field(default_factory=dict)

    # Tracker telemetry
    confidence: Optional[float] = None
    apce: Optional[float] = None
    psr: Optional[float] = None

    # ---- new composite labels --------------------------------------
    localization_state: int = LocalizationState.STABLE.value
    confidence_state: int = ConfidenceState.LOW_CONFIDENCE.value
    derived_state: int = DerivedState.CORRECT_CONFIRMED.value
    false_confirmed_flag: bool = False

    # ---- proactive forecast labels (V3) ----------------------------
    # Set to 0 by default for backward compatibility with V2 labels.
    # ``ignore_forecast=1`` marks frames where the full horizon was not
    # observable; training pipelines should mask those out.
    failure_next_10: int = 0
    false_confirmed_next_10: int = 0
    lost_aware_next_10: int = 0
    ignore_forecast: int = 0

    # Auxiliary scene flags
    aux: dict = field(default_factory=dict)

    label_source: int = LabelSource.GT_RULE.value
    label_noisy: bool = False  # True for distractor/false_confirmed labels

    def to_jsonable(self) -> dict:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "sequence": self.sequence,
            "frame_idx": self.frame_idx,
            "gt_bbox": list(self.gt_bbox) if self.gt_bbox is not None else None,
            "pred_bbox": list(self.pred_bbox) if self.pred_bbox is not None else None,
            "iou": self.iou,
            "center_error": self.center_error,
            "normalized_center_error": self.normalized_center_error,
            "area_ratio": self.area_ratio,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "visible_ratio": self.visible_ratio,
            "absent": self.absent,
            "dataset_attributes": self.dataset_attributes,
            "confidence": self.confidence,
            "apce": self.apce,
            "psr": self.psr,
            "localization_state": self.localization_state,
            "localization_state_name": LocalizationState(self.localization_state).name,
            "confidence_state": self.confidence_state,
            "confidence_state_name": ConfidenceState(self.confidence_state).name,
            "derived_state": self.derived_state,
            "derived_state_name": DerivedState(self.derived_state).name,
            "false_confirmed_flag": self.false_confirmed_flag,
            "failure_next_10": self.failure_next_10,
            "false_confirmed_next_10": self.false_confirmed_next_10,
            "lost_aware_next_10": self.lost_aware_next_10,
            "ignore_forecast": self.ignore_forecast,
            "aux": self.aux,
            "label_source": self.label_source,
            "label_source_name": LabelSource(self.label_source).name,
            "label_noisy": self.label_noisy,
        }


# ---------------------------------------------------------------------------
# Backward-compat aliases — older modules still import these names.
# ``state`` field is mapped to ``derived_state`` so existing CSC training
# loops keep working until they migrate.
# ---------------------------------------------------------------------------


SceneState = DerivedState  # alias for callers that still say "SceneState"
STATE_NAMES = DERIVED_NAMES
NUM_STATES = NUM_DERIVED_STATES
