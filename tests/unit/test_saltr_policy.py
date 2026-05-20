"""Unit tests for saltr/src/salt_r/policy.py — decision logic.

No heavy dependencies: pure Python + numpy only.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test 1: false_confirmed above block threshold → block + abstain + full
# ---------------------------------------------------------------------------

def test_apply_policy_false_confirmed_blocks():
    from salt_r.policy import apply_policy, DEFAULT_THRESHOLDS

    probs = {
        "false_confirmed":    0.85,   # above block threshold 0.70
        "failure_in_5":       0.30,
        "recoverable":        0.20,
        "hard_dynamic_scene": 0.30,
        "needs_full_compute": 0.80,
    }
    action = apply_policy(probs)
    assert action.template_update == "block", f"Expected block, got {action.template_update}"
    assert action.recovery_action == "abstain", f"Expected abstain, got {action.recovery_action}"
    assert action.compute_mode == "full", f"Expected full compute, got {action.compute_mode}"
    assert "false_confirmed" in " ".join(action.triggered_by)


# ---------------------------------------------------------------------------
# Test 2: low risk → cheap compute + allow template updates
# ---------------------------------------------------------------------------

def test_apply_policy_low_risk_cheap_compute():
    from salt_r.policy import apply_policy

    probs = {
        "false_confirmed":    0.02,   # low
        "failure_in_5":       0.05,   # < 0.20
        "recoverable":        0.05,
        "hard_dynamic_scene": 0.10,   # < 0.65
        "needs_full_compute": 0.10,   # < 0.25 threshold
    }
    action = apply_policy(probs)
    assert action.compute_mode == "cheap", \
        f"Low-risk frame should use cheap compute, got {action.compute_mode}"
    assert action.template_update == "allow", \
        f"Low-risk frame should allow updates, got {action.template_update}"


# ---------------------------------------------------------------------------
# Test 3: recoverable high + false_confirmed high → no re-init
# ---------------------------------------------------------------------------

def test_apply_policy_recovery_blocked_by_false_confirmed():
    from salt_r.policy import apply_policy

    # recoverable is high BUT false_confirmed is also high -> don't re-init
    probs = {
        "false_confirmed":    0.60,   # above false_confirmed_max_for_recovery=0.40
        "failure_in_5":       0.10,
        "recoverable":        0.80,   # high
        "hard_dynamic_scene": 0.20,
        "needs_full_compute": 0.50,
    }
    action = apply_policy(probs)
    # Should not run recovery when false_confirmed is high (wrong target)
    assert action.recovery_action != "run", \
        f"Should not run recovery when false_confirmed={probs['false_confirmed']}"


# ---------------------------------------------------------------------------
# Test 4: recoverable high + false_confirmed low → run recovery
# ---------------------------------------------------------------------------

def test_apply_policy_recovery_runs_when_safe():
    from salt_r.policy import apply_policy

    probs = {
        "false_confirmed":    0.05,   # low -> safe to re-init
        "failure_in_5":       0.10,
        "recoverable":        0.75,   # above 0.60 threshold
        "hard_dynamic_scene": 0.20,
        "needs_full_compute": 0.50,
    }
    action = apply_policy(probs)
    assert action.recovery_action == "run", \
        f"Expected recovery=run, got {action.recovery_action}"


# ---------------------------------------------------------------------------
# Test 5: empty probs → safe defaults
# ---------------------------------------------------------------------------

def test_apply_policy_empty_probs_safe_defaults():
    from salt_r.policy import apply_policy, TrackerAction

    action = apply_policy({})
    # Empty probs must return safe defaults: full compute, allow update
    assert action.compute_mode == "full"
    assert action.template_update == "allow"
    assert action.recovery_action == "none"
    assert action.triggered_by == []


# ---------------------------------------------------------------------------
# Test 6: replay_policy — wrong_reinit_rate must use real IoU (not fallback)
# ---------------------------------------------------------------------------

def test_replay_policy_wrong_reinit_uses_real_iou():
    """replay_policy must use real IoU to score wrong_reinit_rate.

    If recoverable is always 1.0 and IoU is always 0.0 (wrong object),
    every recovery attempt is wrong -> wrong_reinit_rate should be ~1.0.
    This test would have failed with the old NPZ-key mismatch bug that
    silently fell back to perfect IoU.
    """
    from salt_r.policy import replay_policy

    n = 50
    # Tracker always recoverable, IoU always 0 (always wrong object)
    probs_seq = [
        {
            "false_confirmed":    0.05,
            "failure_in_5":       0.05,
            "recoverable":        0.90,   # will always trigger run
            "hard_dynamic_scene": 0.10,
            "needs_full_compute": 0.50,
        }
        for _ in range(n)
    ]
    iou_trace = np.zeros(n, dtype=np.float32)  # always IoU=0 = wrong

    result = replay_policy(probs_seq, iou_trace)
    # Every recovery is wrong -> wrong_reinit_rate must be 1.0
    assert result["wrong_reinit_rate"] == 1.0, \
        f"Expected wrong_reinit_rate=1.0 (all re-inits to wrong object), got {result['wrong_reinit_rate']}"


# ---------------------------------------------------------------------------
# Test 7: replay_policy — template_corruption_rate with allowed updates at low IoU
# ---------------------------------------------------------------------------

def test_replay_policy_template_corruption_rate():
    """When false_confirmed is low (allow updates) and IoU < 0.5, updates are corrupt."""
    from salt_r.policy import replay_policy

    n = 30
    probs_seq = [
        {
            "false_confirmed":    0.01,
            "failure_in_5":       0.01,
            "recoverable":        0.01,
            "hard_dynamic_scene": 0.01,
            "needs_full_compute": 0.50,
        }
        for _ in range(n)
    ]
    # IoU always 0.3 < 0.5 -> all allowed updates are corrupt
    iou_trace = np.full(n, 0.3, dtype=np.float32)

    result = replay_policy(probs_seq, iou_trace)
    assert result["template_blocked_rate"] < 0.1, \
        "Low false_confirmed -> should rarely block updates"
    assert result["template_corruption_rate"] > 0.5, \
        "IoU=0.3 + allowed updates -> corruption rate should be high"


# ---------------------------------------------------------------------------
# Tests for decide_intervention() — v2-aware intervention logic
# ---------------------------------------------------------------------------

def test_decide_intervention_none_by_default():
    """Empty probs with default thresholds → safe no-op defaults."""
    from salt_r.interventions import (
        decide_intervention, RecoveryAction, TemplateUpdateAction, AlertTier
    )
    iv = decide_intervention({})
    assert iv.recovery_action == RecoveryAction.NONE, \
        f"Default recovery_action should be NONE, got {iv.recovery_action}"
    assert iv.template_update == TemplateUpdateAction.ALLOW, \
        f"Default template_update should be ALLOW, got {iv.template_update}"
    assert iv.alert_tier == AlertTier.NONE, \
        f"Default alert_tier should be NONE, got {iv.alert_tier}"
    assert iv.triggered_by == [], \
        f"No triggers should fire on empty probs, got {iv.triggered_by}"


def test_decide_intervention_fc_block():
    """High p_fc (0.8 > 0.65 threshold) → BLOCK + ABSTAIN + fc_block in triggered_by."""
    from salt_r.interventions import (
        decide_intervention, RecoveryAction, TemplateUpdateAction
    )
    iv = decide_intervention({"false_confirmed": 0.8})
    assert iv.template_update == TemplateUpdateAction.BLOCK, \
        f"Expected BLOCK, got {iv.template_update}"
    assert iv.recovery_action == RecoveryAction.ABSTAIN, \
        f"Expected ABSTAIN, got {iv.recovery_action}"
    assert any("fc_block" in t for t in iv.triggered_by), \
        f"Expected fc_block entry in triggered_by, got {iv.triggered_by}"


def test_decide_intervention_recovery_run_conditions():
    """Recovery RUN fires iff p_rec >= threshold AND p_fc < 0.40; both must hold."""
    from salt_r.interventions import decide_intervention, RecoveryAction

    # p_rec=0.9, p_fc=0.1 → RUN
    iv_run = decide_intervention({"recoverable": 0.9, "false_confirmed": 0.1})
    assert iv_run.recovery_action == RecoveryAction.RUN, \
        f"Expected RUN when p_rec=0.9 and p_fc=0.1, got {iv_run.recovery_action}"

    # p_rec too low → stays NONE
    iv_low_rec = decide_intervention({"recoverable": 0.3, "false_confirmed": 0.1})
    assert iv_low_rec.recovery_action == RecoveryAction.NONE, \
        f"Expected NONE when p_rec=0.3 (below threshold), got {iv_low_rec.recovery_action}"

    # p_fc too high → stays NONE (fc not quite at block threshold but >= 0.40)
    iv_high_fc = decide_intervention({"recoverable": 0.9, "false_confirmed": 0.5})
    assert iv_high_fc.recovery_action == RecoveryAction.NONE, \
        f"Expected NONE when p_fc=0.5 (>= 0.40 guard), got {iv_high_fc.recovery_action}"


def test_decide_intervention_eprocess_critical():
    """eprocess_value >> 1/alpha → CRITICAL alert tier."""
    from salt_r.interventions import decide_intervention, AlertTier

    # alpha=0.10 → 1/alpha=10; CRITICAL requires >= 10*5=50
    iv = decide_intervention({}, eprocess_value=200.0, alpha=0.10)
    assert iv.alert_tier == AlertTier.CRITICAL, \
        f"Expected CRITICAL when eprocess_value=200 >> 50, got {iv.alert_tier}"
    assert any("eprocess_critical" in t for t in iv.triggered_by), \
        f"Expected eprocess_critical in triggered_by, got {iv.triggered_by}"


def test_decide_intervention_eprocess_intervene_tier():
    """eprocess_value >= 1/alpha but < 5/alpha → INTERVENE tier."""
    from salt_r.interventions import decide_intervention, AlertTier

    # alpha=0.10 → threshold=10; INTERVENE: 10 <= value < 50
    iv = decide_intervention({}, eprocess_value=15.0, alpha=0.10)
    assert iv.alert_tier == AlertTier.INTERVENE, \
        f"Expected INTERVENE for eprocess_value=15, got {iv.alert_tier}"


def test_decide_intervention_kf_residual_verify():
    """High kf_residual promotes template_update from ALLOW to VERIFY."""
    from salt_r.interventions import decide_intervention, TemplateUpdateAction

    iv = decide_intervention({}, kf_residual=0.8, kf_residual_flag_threshold=0.50)
    assert iv.template_update == TemplateUpdateAction.VERIFY, \
        f"High kf_residual should set VERIFY, got {iv.template_update}"
    assert any("kf_residual" in t for t in iv.triggered_by), \
        f"Expected kf_residual in triggered_by, got {iv.triggered_by}"


def test_decide_intervention_should_trigger_fallback_requires_critical_and_block():
    """should_trigger_fallback is True only when CRITICAL tier + BLOCK template."""
    from salt_r.interventions import decide_intervention

    # fc=0.8 gives BLOCK but no CRITICAL (eprocess=1)
    iv_block_only = decide_intervention({"false_confirmed": 0.8})
    assert not iv_block_only.should_trigger_fallback, \
        "BLOCK without CRITICAL should not trigger fallback"

    # CRITICAL (eprocess=200) without BLOCK (fc low)
    iv_critical_only = decide_intervention({}, eprocess_value=200.0, alpha=0.10)
    assert not iv_critical_only.should_trigger_fallback, \
        "CRITICAL without BLOCK should not trigger fallback"

    # Both CRITICAL and BLOCK: fc=0.8 + eprocess=200
    iv_both = decide_intervention({"false_confirmed": 0.8}, eprocess_value=200.0, alpha=0.10)
    assert iv_both.should_trigger_fallback, \
        "CRITICAL + BLOCK must trigger fallback"
