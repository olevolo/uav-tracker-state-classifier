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
