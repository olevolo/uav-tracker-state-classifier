"""Target-state taxonomy for the SALT TSA module."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class TargetState(IntEnum):
    """6-class target-state taxonomy inferred from tracking-consistency signals.

    Priority (descending) used when multiple conditions fire simultaneously:
    LOST > DISTRACTOR_RISK > OCCLUDED > DYNAMIC > LOW_RES > CONFIRMED
    """

    CONFIRMED       = 0  # stable, high confidence, predictable motion
    LOW_RES         = 1  # small target (<400 px²), need full computation
    DYNAMIC         = 2  # fast motion / high LSTM residual
    OCCLUDED        = 3  # low IoU consistency, partial occlusion
    LOST            = 4  # target missing, detector takes over
    DISTRACTOR_RISK = 5  # appearance drift detected, similar objects nearby


@dataclass
class TargetStateAssessment:
    """Output of a single ``TargetStateAssessor.assess`` call."""

    state: TargetState
    confidence: float   # calibrated score in [0, 1]
    frame_idx: int


__all__ = ["TargetState", "TargetStateAssessment"]
