"""Runtime-safe intervention actions for SALT-RD policy.

These are the concrete actions triggered by policy decisions:
- block_template_update: prevent template corruption
- reject_recovery: prevent wrong re-initialization
- verify_before_reinit: require extra confidence before init
- expand_search: conservative search region expansion
- require_full_compute: disable token pruning
- trigger_fallback: call EfficientTAM/SAM2 (only when available)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TemplateUpdateAction(str, Enum):
    ALLOW = "allow"
    VERIFY = "verify"       # requires confirmation
    BLOCK = "block"         # hard block


class RecoveryAction(str, Enum):
    NONE = "none"
    RUN = "run"
    VERIFY = "verify"
    ABSTAIN = "abstain"


class ComputeMode(str, Enum):
    FULL = "full"
    CHEAP = "cheap"
    ADAPTIVE = "adaptive"  # let policy decide per-frame


class SearchMode(str, Enum):
    NORMAL = "normal"
    EXPAND = "expand"       # conservative expansion
    RESTRICT = "restrict"   # narrow search near KF prediction


class AlertTier(str, Enum):
    NONE = "none"
    OBSERVE = "observe"     # e-process growing but not threshold
    VERIFY = "verify"       # threshold crossed once
    INTERVENE = "intervene" # strong evidence (E >> 1/alpha)
    CRITICAL = "critical"   # e-process very high + p_fc high


@dataclass
class TrackerIntervention:
    """Complete intervention specification for one frame."""
    template_update: TemplateUpdateAction = TemplateUpdateAction.ALLOW
    recovery_action: RecoveryAction = RecoveryAction.NONE
    compute_mode: ComputeMode = ComputeMode.ADAPTIVE
    search_mode: SearchMode = SearchMode.NORMAL
    alert_tier: AlertTier = AlertTier.NONE
    triggered_by: list[str] = field(default_factory=list)
    confidence: float = 0.0
    # New v2 fields:
    kf_residual: float = 0.0        # SAMURAI-inspired spatial discontinuity
    memory_margin: float = 0.0      # DAM-inspired target-vs-distractor margin
    ifd10_triggered: bool = False   # long-horizon risk triggered expand_search
    ifd20_triggered: bool = False   # very-long-horizon risk triggered early warning

    @property
    def should_trigger_fallback(self) -> bool:
        """Only trigger heavy fallback (EfficientTAM) on critical cases."""
        return (
            self.alert_tier == AlertTier.CRITICAL
            and self.template_update == TemplateUpdateAction.BLOCK
        )


def decide_intervention(
    probs: dict[str, float],
    eprocess_value: float = 1.0,
    memory_margin: float = 0.0,
    kf_residual: float = 0.0,
    alpha: float = 0.10,
    fc_block_threshold: float = 0.65,
    fc_verify_threshold: float = 0.40,
    ifd10_expand_threshold: float = 0.60,
    ifd20_early_warn_threshold: float = 0.50,
    reinit_reject_threshold: float = 0.65,
    mem_margin_block_threshold: float = -0.05,
    kf_residual_flag_threshold: float = 0.50,
) -> TrackerIntervention:
    """V2-aware intervention decision incorporating all signals.

    Priority:
    1. false_confirmed or memory_margin < threshold → BLOCK template + ABSTAIN recovery
    2. eprocess CRITICAL (E >> 1/alpha) → INTERVENE tier
    3. ifd10 high → expand search, verify template
    4. ifd20 high → OBSERVE tier, prepare
    5. kf_residual high → flag potential identity switch
    6. recoverable + low fc → allow recovery
    """
    p_fc = float(probs.get("false_confirmed", 0.0))
    p_ifd10 = float(probs.get("imminent_failure_dynamic_10", 0.0))
    p_ifd20 = float(probs.get("imminent_failure_dynamic_20", 0.0))
    p_rec = float(probs.get("recoverable", 0.0))
    p_fi5 = float(probs.get("failure_in_5", 0.0))
    p_ifd = float(probs.get("imminent_failure_dynamic", 0.0))

    intervention = TrackerIntervention()
    intervention.kf_residual = float(kf_residual)
    intervention.memory_margin = float(memory_margin)

    eprocess_threshold = 1.0 / max(alpha, 1e-9)

    triggered: list[str] = []

    # -----------------------------------------------------------------------
    # Priority 1: false_confirmed or memory_margin → BLOCK + ABSTAIN
    # -----------------------------------------------------------------------
    fc_block = p_fc >= fc_block_threshold or memory_margin < mem_margin_block_threshold

    if fc_block:
        intervention.template_update = TemplateUpdateAction.BLOCK
        intervention.recovery_action = RecoveryAction.ABSTAIN
        intervention.compute_mode = ComputeMode.FULL
        intervention.confidence = max(p_fc, abs(min(0.0, memory_margin)))
        if p_fc >= fc_block_threshold:
            triggered.append(f"fc_block={p_fc:.2f}")
        if memory_margin < mem_margin_block_threshold:
            triggered.append(f"mem_margin={memory_margin:.3f}")

    elif p_fc >= fc_verify_threshold:
        intervention.template_update = TemplateUpdateAction.VERIFY
        triggered.append(f"fc_verify={p_fc:.2f}")

    # -----------------------------------------------------------------------
    # Priority 2: e-process CRITICAL
    # -----------------------------------------------------------------------
    if eprocess_value >= eprocess_threshold * 5:
        intervention.alert_tier = AlertTier.CRITICAL
        if intervention.template_update == TemplateUpdateAction.ALLOW:
            intervention.template_update = TemplateUpdateAction.VERIFY
        triggered.append(f"eprocess_critical={eprocess_value:.1f}")
    elif eprocess_value >= eprocess_threshold:
        if intervention.alert_tier == AlertTier.NONE:
            intervention.alert_tier = AlertTier.INTERVENE
        triggered.append(f"eprocess_intervene={eprocess_value:.1f}")

    # -----------------------------------------------------------------------
    # Priority 3: ifd10 → expand search
    # -----------------------------------------------------------------------
    if p_ifd10 >= ifd10_expand_threshold:
        intervention.search_mode = SearchMode.EXPAND
        intervention.ifd10_triggered = True
        if intervention.template_update == TemplateUpdateAction.ALLOW:
            intervention.template_update = TemplateUpdateAction.VERIFY
        triggered.append(f"ifd10={p_ifd10:.2f}")

    # -----------------------------------------------------------------------
    # Priority 4: ifd20 → early warning (OBSERVE)
    # -----------------------------------------------------------------------
    if p_ifd20 >= ifd20_early_warn_threshold:
        intervention.ifd20_triggered = True
        if intervention.alert_tier == AlertTier.NONE:
            intervention.alert_tier = AlertTier.OBSERVE
        triggered.append(f"ifd20={p_ifd20:.2f}")

    # -----------------------------------------------------------------------
    # Priority 5: kf_residual high → potential identity switch
    # -----------------------------------------------------------------------
    if kf_residual >= kf_residual_flag_threshold:
        if intervention.template_update == TemplateUpdateAction.ALLOW:
            intervention.template_update = TemplateUpdateAction.VERIFY
        triggered.append(f"kf_residual={kf_residual:.2f}")

    # -----------------------------------------------------------------------
    # Priority 6: recovery decision
    # -----------------------------------------------------------------------
    # Only explicitly run recovery when NOT in fc_block state,
    # p_rec is high, and fc is low.
    if not fc_block and p_rec >= reinit_reject_threshold and p_fc < 0.40:
        intervention.recovery_action = RecoveryAction.RUN
        triggered.append(f"recoverable={p_rec:.2f}")
    # else: stays NONE (deployment-safe default)

    intervention.triggered_by = triggered
    return intervention
