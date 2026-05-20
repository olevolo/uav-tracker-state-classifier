"""Unit tests for saltr/src/salt_r/policy_sweep.py and interventions.py.

Tests:
  1. SimpleBboxKalmanFilter: returns float kf_residual, zero for stationary bbox
  2. SimpleBboxKalmanFilter: high residual when bbox jumps suddenly
  3. decide_intervention: blocks template when fc > threshold
  4. decide_intervention: triggers ifd10_expand when ifd10 high
  5. decide_intervention: CRITICAL tier when e-process very high + fc high
  6. decide_intervention: allows recovery when recoverable high + fc low
  7. TrackerIntervention.should_trigger_fallback: true only on CRITICAL+BLOCK
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test 1: SimpleBboxKalmanFilter — stationary bbox gives near-zero residual
# ---------------------------------------------------------------------------

def test_kf_stationary_zero_residual():
    """Stationary bbox produces near-zero kf_residual after initialization."""
    from salt_r.policy_sweep import SimpleBboxKalmanFilter

    kf = SimpleBboxKalmanFilter()
    bbox = np.array([10.0, 20.0, 50.0, 60.0])

    # First update initializes, residual = 0
    r0 = kf.update(bbox)
    assert isinstance(r0, float), f"Expected float, got {type(r0)}"
    assert r0 == 0.0, f"First update residual should be 0.0, got {r0}"

    # Subsequent updates with same bbox should give near-zero residual
    r1 = kf.update(bbox)
    r2 = kf.update(bbox)
    r3 = kf.update(bbox)

    assert isinstance(r1, float)
    assert 0.0 <= r1 <= 1.0, f"kf_residual must be in [0,1], got {r1}"
    assert r3 < 0.3, f"Stationary bbox should have low residual after settling, got {r3}"


# ---------------------------------------------------------------------------
# Test 2: SimpleBboxKalmanFilter — high residual on sudden bbox jump
# ---------------------------------------------------------------------------

def test_kf_high_residual_on_jump():
    """Sudden bbox jump produces high kf_residual."""
    from salt_r.policy_sweep import SimpleBboxKalmanFilter

    kf = SimpleBboxKalmanFilter()
    bbox_normal = np.array([10.0, 20.0, 50.0, 60.0])  # 40x40 box at (10,20)

    # Warm up KF with consistent motion
    for _ in range(5):
        kf.update(bbox_normal)

    # Sudden jump to far away position
    bbox_jump = np.array([500.0, 500.0, 540.0, 540.0])
    r_jump = kf.update(bbox_jump)

    assert isinstance(r_jump, float)
    assert r_jump > 0.5, \
        f"Large bbox jump should produce kf_residual > 0.5, got {r_jump}"


# ---------------------------------------------------------------------------
# Test 3: decide_intervention — blocks template when fc > threshold
# ---------------------------------------------------------------------------

def test_decide_intervention_blocks_on_high_fc():
    """decide_intervention blocks template update when fc probability is high."""
    from salt_r.interventions import decide_intervention, TemplateUpdateAction, RecoveryAction

    probs = {
        "false_confirmed": 0.80,  # > default fc_block_threshold=0.65
        "imminent_failure_dynamic_10": 0.10,
        "imminent_failure_dynamic_20": 0.10,
        "recoverable": 0.30,
        "failure_in_5": 0.10,
    }

    result = decide_intervention(probs)

    assert result.template_update == TemplateUpdateAction.BLOCK, \
        f"Expected BLOCK, got {result.template_update}"
    assert result.recovery_action == RecoveryAction.ABSTAIN, \
        f"Expected ABSTAIN, got {result.recovery_action}"
    assert any("fc_block" in t for t in result.triggered_by), \
        f"Expected fc_block in triggered_by, got {result.triggered_by}"


# ---------------------------------------------------------------------------
# Test 4: decide_intervention — triggers ifd10 expand search when ifd10 high
# ---------------------------------------------------------------------------

def test_decide_intervention_ifd10_expand_search():
    """decide_intervention expands search when ifd10 probability is high."""
    from salt_r.interventions import decide_intervention, SearchMode

    probs = {
        "false_confirmed": 0.10,            # low
        "imminent_failure_dynamic_10": 0.75, # > default 0.60 threshold
        "imminent_failure_dynamic_20": 0.20,
        "recoverable": 0.10,
        "failure_in_5": 0.20,
    }

    result = decide_intervention(probs)

    assert result.search_mode == SearchMode.EXPAND, \
        f"Expected EXPAND search mode, got {result.search_mode}"
    assert result.ifd10_triggered, \
        "Expected ifd10_triggered=True"
    assert any("ifd10" in t for t in result.triggered_by), \
        f"Expected ifd10 in triggered_by, got {result.triggered_by}"


# ---------------------------------------------------------------------------
# Test 5: decide_intervention — CRITICAL tier when e-process very high + fc high
# ---------------------------------------------------------------------------

def test_decide_intervention_critical_tier():
    """CRITICAL alert tier when e-process very high."""
    from salt_r.interventions import decide_intervention, AlertTier

    probs = {
        "false_confirmed": 0.72,             # above block threshold
        "imminent_failure_dynamic_10": 0.30,
        "imminent_failure_dynamic_20": 0.30,
        "recoverable": 0.10,
        "failure_in_5": 0.40,
    }

    # e-process value = 5 * (1/alpha) = 5 * 10 = 50 → CRITICAL
    result = decide_intervention(probs, eprocess_value=50.0, alpha=0.10)

    assert result.alert_tier == AlertTier.CRITICAL, \
        f"Expected CRITICAL alert tier, got {result.alert_tier}"


# ---------------------------------------------------------------------------
# Test 6: decide_intervention — allows recovery when recoverable high + fc low
# ---------------------------------------------------------------------------

def test_decide_intervention_allows_recovery():
    """Recovery is allowed when recoverable is high and fc is low."""
    from salt_r.interventions import decide_intervention, RecoveryAction, TemplateUpdateAction

    probs = {
        "false_confirmed": 0.10,             # low → not blocking
        "imminent_failure_dynamic_10": 0.10,
        "imminent_failure_dynamic_20": 0.10,
        "recoverable": 0.80,                 # high → trigger recovery
        "failure_in_5": 0.10,
    }

    result = decide_intervention(
        probs,
        reinit_reject_threshold=0.65,  # 0.80 > 0.65 → allows recovery
        eprocess_value=1.0,
        memory_margin=0.0,
        kf_residual=0.1,
    )

    # template update should be ALLOW (fc is low)
    assert result.template_update in (
        TemplateUpdateAction.ALLOW, TemplateUpdateAction.VERIFY
    ), f"Expected ALLOW or VERIFY template update, got {result.template_update}"

    # recovery should NOT be abstain (fc is below block threshold)
    assert result.recovery_action != RecoveryAction.ABSTAIN, \
        f"Recovery should not be ABSTAIN when fc is low, got {result.recovery_action}"


# ---------------------------------------------------------------------------
# Test 7: TrackerIntervention.should_trigger_fallback — true only on CRITICAL+BLOCK
# ---------------------------------------------------------------------------

def test_should_trigger_fallback_critical_and_block_only():
    """should_trigger_fallback is True only when CRITICAL tier AND BLOCK template."""
    from salt_r.interventions import (
        TrackerIntervention,
        AlertTier,
        TemplateUpdateAction,
        RecoveryAction,
    )

    # Case 1: CRITICAL + BLOCK → should trigger
    iv_critical_block = TrackerIntervention(
        alert_tier=AlertTier.CRITICAL,
        template_update=TemplateUpdateAction.BLOCK,
    )
    assert iv_critical_block.should_trigger_fallback, \
        "CRITICAL + BLOCK should trigger fallback"

    # Case 2: CRITICAL + VERIFY → should NOT trigger
    iv_critical_verify = TrackerIntervention(
        alert_tier=AlertTier.CRITICAL,
        template_update=TemplateUpdateAction.VERIFY,
    )
    assert not iv_critical_verify.should_trigger_fallback, \
        "CRITICAL + VERIFY should NOT trigger fallback"

    # Case 3: INTERVENE + BLOCK → should NOT trigger
    iv_intervene_block = TrackerIntervention(
        alert_tier=AlertTier.INTERVENE,
        template_update=TemplateUpdateAction.BLOCK,
    )
    assert not iv_intervene_block.should_trigger_fallback, \
        "INTERVENE + BLOCK should NOT trigger fallback (only CRITICAL does)"

    # Case 4: NONE + ALLOW → should NOT trigger
    iv_none = TrackerIntervention()
    assert not iv_none.should_trigger_fallback, \
        "Default TrackerIntervention should not trigger fallback"


# ---------------------------------------------------------------------------
# Test 8: PolicySweepConfig — stores values correctly
# ---------------------------------------------------------------------------

def test_policy_sweep_config_stores_values():
    """PolicySweepConfig stores and returns values via to_dict."""
    from salt_r.policy_sweep import PolicySweepConfig

    config = PolicySweepConfig(
        fc_threshold=0.70,
        reinit_threshold=0.55,
        eprocess_alpha=0.05,
        mem_margin_threshold=0.10,
        use_ifd10=True,
        use_ifd20=False,
        kf_residual_threshold=0.35,
    )

    d = config.to_dict()
    assert abs(d["fc_threshold"] - 0.70) < 1e-9
    assert abs(d["reinit_threshold"] - 0.55) < 1e-9
    assert abs(d["eprocess_alpha"] - 0.05) < 1e-9
    assert d["use_ifd10"] is True
    assert d["use_ifd20"] is False


# ---------------------------------------------------------------------------
# Test 9: SimpleBboxKalmanFilter.predict — returns 4-element array
# ---------------------------------------------------------------------------

def test_kf_predict_shape():
    """SimpleBboxKalmanFilter.predict returns (4,) array."""
    from salt_r.policy_sweep import SimpleBboxKalmanFilter

    kf = SimpleBboxKalmanFilter()

    # Before initialization
    pred_before = kf.predict()
    assert pred_before.shape == (4,), f"Expected shape (4,), got {pred_before.shape}"

    # After some updates
    bbox = np.array([20.0, 30.0, 60.0, 70.0])
    kf.update(bbox)
    kf.update(bbox + np.array([2.0, 2.0, 2.0, 2.0]))  # slight motion

    pred_after = kf.predict()
    assert pred_after.shape == (4,), f"Expected shape (4,), got {pred_after.shape}"
    assert np.all(np.isfinite(pred_after)), "Predicted bbox should have finite values"


# ---------------------------------------------------------------------------
# Test 10: decide_intervention — memory margin triggers block
# ---------------------------------------------------------------------------

def test_decide_intervention_mem_margin_triggers_block():
    """Negative memory margin below threshold triggers BLOCK (distractor present)."""
    from salt_r.interventions import decide_intervention, TemplateUpdateAction, RecoveryAction

    probs = {
        "false_confirmed": 0.20,   # moderate — below block threshold
        "imminent_failure_dynamic_10": 0.10,
        "imminent_failure_dynamic_20": 0.10,
        "recoverable": 0.10,
    }

    result = decide_intervention(
        probs,
        memory_margin=-0.20,               # well below default threshold of -0.05
        mem_margin_block_threshold=-0.05,
    )

    assert result.template_update == TemplateUpdateAction.BLOCK, \
        f"Negative memory margin should trigger BLOCK, got {result.template_update}"
    assert result.recovery_action == RecoveryAction.ABSTAIN, \
        f"Expected ABSTAIN recovery when memory margin blocks, got {result.recovery_action}"
    assert any("mem_margin" in t for t in result.triggered_by), \
        f"Expected mem_margin in triggered_by, got {result.triggered_by}"
