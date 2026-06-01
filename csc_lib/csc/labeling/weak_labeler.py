"""Per-frame rule-based weak labeler — composite version.

Output:
    localization_state ∈ {STABLE, UNCERTAIN, LOST}
    confidence_state   ∈ {HIGH_CONFIDENCE, LOW_CONFIDENCE}
    derived_state      ∈ {CORRECT_CONFIRMED, CORRECT_UNCERTAIN, LOST_AWARE, FALSE_CONFIRMED}
    aux flags          ∈ {occlusion, out_of_view, fast_motion, scale_change, distractor_risk}

The composite design avoids forcing the model to memorise the implicit
``LOST AND HIGH_CONFIDENCE → FALSE_CONFIRMED`` conjunction; the two
axes are predicted independently and the paper-state is composed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from csc_lib.csc.labeling.label_schema import (
    AUX_FLAGS,
    ConfidenceState,
    DerivedState,
    LabelSource,
    LocalizationState,
    derive_state,
)


@dataclass
class LabelingThresholds:
    """Tunable thresholds.  Defaults match CSC.md (paper-aligned)."""

    # Localization (IoU) ---------------------------------------------
    tau_confirmed_iou: float = 0.5
    tau_uncertain_iou: float = 0.2
    tau_lost_iou: float = 0.2
    lost_min_consecutive_frames: int = 3

    # Confidence (telemetry) -----------------------------------------
    # Applies only when APCE and PSR are unavailable.
    # When calibrators are active, use 0.65 (= top 35% → HIGH_CONFIDENCE).
    tau_high_confidence: float = 0.65

    # APCE / PSR thresholds: two sets of defaults.
    # Raw (uncalibrated): SGLATrack APCE ≈ 0–300, PSR ≈ 0–15000.
    # Calibrated ([0,1] percentile): use 0.5 (above median = high).
    tau_high_apce: float = 30.0   # raw default; set to 0.5 if calibrated
    tau_high_psr: float = 8.0     # raw default; set to 0.5 if calibrated

    # Auxiliary scene flags ------------------------------------------
    tau_fast_motion_norm: float = 0.06
    tau_scale_change_log: float = 0.30
    tau_distractor_iou: float = 0.30

    # DEPRECATED — kept for backward compat with older configs / tests
    tau_distractor_iou_legacy: float = 0.3
    tau_fc_iou: float = 0.2
    tau_fc_conf: float = 0.65
    tau_low_conf: float = 0.10


def label_frame(
    iou: Optional[float],
    confidence: Optional[float],
    *,
    full_occlusion: bool = False,
    out_of_view: bool = False,
    absent: bool = False,
    consecutive_low_iou: int = 0,
    fast_motion_norm: Optional[float] = None,
    scale_change_log: Optional[float] = None,
    apce: Optional[float] = None,
    psr: Optional[float] = None,
    thresholds: Optional[LabelingThresholds] = None,
) -> tuple[LocalizationState, ConfidenceState, DerivedState, dict, LabelSource, bool]:
    """Compute the composite frame label.

    Returns
    -------
    localization_state, confidence_state, derived_state, aux_flags,
    label_source, label_noisy
    """
    th = thresholds or LabelingThresholds()
    aux = {flag: False for flag in AUX_FLAGS}

    # ----- Auxiliary scene flags ------------------------------------
    if full_occlusion or absent:
        aux["occlusion"] = True
    if out_of_view:
        aux["occlusion"] = True
        aux["out_of_view"] = True
    if fast_motion_norm is not None and fast_motion_norm > th.tau_fast_motion_norm:
        aux["fast_motion"] = True
    if scale_change_log is not None and scale_change_log > th.tau_scale_change_log:
        aux["scale_change"] = True

    # ----- Axis 1 — localization state (IoU only) -------------------
    if iou is None:
        loc = LocalizationState.STABLE  # benign default if no GT
    elif iou < th.tau_lost_iou and consecutive_low_iou >= th.lost_min_consecutive_frames:
        loc = LocalizationState.LOST
    elif iou >= th.tau_confirmed_iou:
        loc = LocalizationState.STABLE
    else:
        loc = LocalizationState.UNCERTAIN

    # ----- Axis 2 — confidence state (telemetry only) ----------------
    # Per CLAUDE.md spec: confidence is REQUIRED for HIGH_CONFIDENCE.
    # Calibration handles per-tracker scale (see feedback_csc_calibration_bugs).
    # APCE/PSR remain available as features in FrameLabel — they MUST NOT
    # override confidence at the label-decision layer (audit 2026-05-29
    # found 80% of FC labels violated this rule under the prior OR-gate).
    if confidence is not None and confidence >= th.tau_high_confidence:
        conf = ConfidenceState.HIGH_CONFIDENCE
    else:
        conf = ConfidenceState.LOW_CONFIDENCE

    # ----- Distractor-risk auxiliary flag (heuristic) ---------------
    # When localization is LOST but confidence is high, the predicted
    # bbox is "spatially plausible" — likely tracking a distractor.
    if loc == LocalizationState.LOST and conf == ConfidenceState.HIGH_CONFIDENCE:
        aux["distractor_risk"] = True

    # ----- Derived paper-state --------------------------------------
    derived = derive_state(loc, conf)

    # ----- Source / noise tagging -----------------------------------
    if aux["occlusion"]:
        source = LabelSource.DATASET_ATTRIBUTE
    elif iou is None:
        source = LabelSource.HEURISTIC
    elif derived == DerivedState.FALSE_CONFIRMED or aux["distractor_risk"]:
        source = LabelSource.MIXED
    else:
        source = LabelSource.GT_RULE

    noisy = derived == DerivedState.FALSE_CONFIRMED or aux["distractor_risk"]
    return loc, conf, derived, aux, source, noisy
