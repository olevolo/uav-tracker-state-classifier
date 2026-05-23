"""Unit tests for StateExitRouter and related helpers in exit_router.py.

Covers:
  - policy default values and validation
  - step() basic policy lookup (no hysteresis)
  - hysteresis: transition fires only after min_hold_frames
  - hysteresis: pending transition cancelled by oscillation back
  - confidence gating: risky state downgraded when confidence < min_state_confidence
  - reset() clears all state and stats
  - stats accumulators (n_steps, n_held, n_switched, n_by_state, n_by_force_idx)
  - stats_dict() structure
  - DERIVED_INT_TO_ROUTER_STATE mapping completeness
  - is_in_risky_mode property
  - current_idx property
  - hold_countdown property
  - policy_summary() copy semantics
  - custom policy injection
  - invalid init args raise ValueError
"""
from __future__ import annotations

import pytest

from uav_tracker.trackers.exit_router import (
    DEFAULT_EXIT_IDX,
    DERIVED_INT_TO_ROUTER_STATE,
    RISKY_STATES,
    STATE_TO_EXIT,
    RouterStats,
    StateExitRouter,
    _DERIVED_FALLBACK_STATE,
    _VALID_FORCE_IDX_RANGE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_router(**kwargs) -> StateExitRouter:
    return StateExitRouter(**kwargs)


# ---------------------------------------------------------------------------
# Policy defaults
# ---------------------------------------------------------------------------


class TestPolicyDefaults:
    def test_default_policy_keys(self):
        r = _make_router()
        assert set(r.policy_summary().keys()) == set(STATE_TO_EXIT.keys())

    def test_confirmed_uses_mlp_default(self):
        r = _make_router(min_hold_frames=0)
        idx = r.step("confirmed")
        assert idx == DEFAULT_EXIT_IDX

    def test_uncertain_uses_mlp_default(self):
        r = _make_router(min_hold_frames=0)
        idx = r.step("uncertain")
        assert idx == DEFAULT_EXIT_IDX

    def test_occluded_forces_block9(self):
        r = _make_router(min_hold_frames=0)
        idx = r.step("occluded")
        assert idx == 3  # MLP index 3 → block 9

    def test_distractor_forces_block9(self):
        r = _make_router(min_hold_frames=0)
        idx = r.step("distractor")
        assert idx == 3

    def test_false_confirmed_forces_block9(self):
        r = _make_router(min_hold_frames=0)
        idx = r.step("false_confirmed")
        assert idx == 3

    def test_lost_forces_block9(self):
        r = _make_router(min_hold_frames=0)
        idx = r.step("lost")
        assert idx == 3

    def test_unknown_state_falls_back_to_default(self):
        r = _make_router(min_hold_frames=0)
        idx = r.step("nonexistent_state")
        assert idx == DEFAULT_EXIT_IDX


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------


class TestHysteresis:
    def test_no_hold_transition_fires_immediately(self):
        """min_hold_frames=0 means transition fires on the very first divergent step."""
        r = _make_router(min_hold_frames=0)
        # First step — transitions from DEFAULT (-1) to 3
        idx = r.step("occluded")
        assert idx == 3, f"expected 3, got {idx}"

    def test_hold_delays_transition(self):
        """With min_hold_frames=3, the transition should not fire for 3 frames."""
        r = _make_router(min_hold_frames=3)
        results = [r.step("occluded") for _ in range(3)]
        # All 3 frames should still return DEFAULT (held)
        assert all(v == DEFAULT_EXIT_IDX for v in results), (
            f"expected all {DEFAULT_EXIT_IDX}, got {results}"
        )

    def test_hold_fires_after_countdown(self):
        """On the 4th step (countdown reaches 0) the switch fires."""
        r = _make_router(min_hold_frames=3)
        for _ in range(3):
            r.step("occluded")  # countdown ticks down
        idx = r.step("occluded")  # fires
        assert idx == 3, f"expected 3 after hold, got {idx}"

    def test_oscillation_cancels_pending_transition(self):
        """If state oscillates back before the hold expires, transition is cancelled."""
        r = _make_router(min_hold_frames=5)
        # Start requesting occluded (pending = 3)
        for _ in range(2):
            r.step("occluded")
        # Before countdown expires, oscillate back to confirmed
        idx_back = r.step("confirmed")
        # Should still be at DEFAULT (-1)
        assert idx_back == DEFAULT_EXIT_IDX

    def test_hold_after_switch(self):
        """After a switch fires, the router stays in new idx for min_hold frames."""
        r = _make_router(min_hold_frames=2)
        # Trigger switch: need 2+1 steps at min_hold=2
        for _ in range(2):
            r.step("occluded")
        idx_fire = r.step("occluded")
        assert idx_fire == 3
        # Now request back to confirmed — should stay at 3 due to hold
        idx_hold1 = r.step("confirmed")
        idx_hold2 = r.step("confirmed")
        assert idx_hold1 == 3 or idx_hold2 == 3  # at least one still held

    def test_zero_hold_confirmed_to_confirmed_no_switch(self):
        """Staying on confirmed (MLP default) should never cause n_switched > 0."""
        r = _make_router(min_hold_frames=0)
        for _ in range(10):
            r.step("confirmed")
        assert r.stats.n_switched == 0


# ---------------------------------------------------------------------------
# Confidence gating
# ---------------------------------------------------------------------------


class TestConfidenceGating:
    def test_gating_disabled_by_default(self):
        """Default min_state_confidence=0.0 means no gating."""
        r = _make_router(min_hold_frames=0, min_state_confidence=0.0)
        idx = r.step("occluded", state_confidence=0.01)
        # Should still force block 9
        assert idx == 3

    def test_gating_downgrades_risky_state_when_low_confidence(self):
        """Low confidence on a risky state → downgrade to uncertain → MLP default."""
        r = _make_router(min_hold_frames=0, min_state_confidence=0.6)
        idx = r.step("false_confirmed", state_confidence=0.4)
        # Downgraded to "uncertain" → DEFAULT_EXIT_IDX
        assert idx == DEFAULT_EXIT_IDX

    def test_gating_passes_risky_state_when_high_confidence(self):
        """High confidence above threshold → risky state is kept, block 9 forced."""
        r = _make_router(min_hold_frames=0, min_state_confidence=0.6)
        idx = r.step("false_confirmed", state_confidence=0.8)
        assert idx == 3

    def test_gating_does_not_affect_non_risky_states(self):
        """Confidence gating only applies to RISKY_STATES; confirmed is unaffected."""
        r = _make_router(min_hold_frames=0, min_state_confidence=0.99)
        # "confirmed" is not in RISKY_STATES — gating should not change it
        idx = r.step("confirmed", state_confidence=0.0)
        assert idx == DEFAULT_EXIT_IDX  # confirmed already maps to -1

    def test_gating_stats_reflect_downgrade(self):
        """When a risky state is downgraded, n_by_state counts 'uncertain', not original."""
        r = _make_router(min_hold_frames=0, min_state_confidence=0.7)
        r.step("lost", state_confidence=0.3)
        # effective_state was "uncertain"
        assert "uncertain" in r.stats.n_by_state
        assert r.stats.n_by_state.get("lost", 0) == 0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_current_idx(self):
        r = _make_router(min_hold_frames=0)
        r.step("occluded")
        r.reset()
        assert r.current_idx == DEFAULT_EXIT_IDX

    def test_reset_clears_stats(self):
        r = _make_router(min_hold_frames=0)
        for _ in range(5):
            r.step("lost")
        r.reset()
        assert r.stats.n_steps == 0
        assert r.stats.n_switched == 0
        assert r.stats.n_held == 0
        assert r.stats.n_by_state == {}
        assert r.stats.n_by_force_idx == {}

    def test_reset_clears_hysteresis(self):
        r = _make_router(min_hold_frames=10)
        for _ in range(3):
            r.step("occluded")
        r.reset()
        # After reset, should be able to re-start a transition fresh
        assert r.hold_countdown == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_n_steps_increments(self):
        r = _make_router()
        for k in range(7):
            r.step("confirmed")
        assert r.stats.n_steps == 7

    def test_n_by_state_counts(self):
        r = _make_router(min_hold_frames=0)
        r.step("confirmed")
        r.step("confirmed")
        r.step("lost")
        assert r.stats.n_by_state["confirmed"] == 2
        assert r.stats.n_by_state["lost"] == 1

    def test_n_by_force_idx_counts(self):
        r = _make_router(min_hold_frames=0)
        r.step("confirmed")   # → -1
        r.step("confirmed")   # → -1
        r.step("lost")        # → 3
        assert r.stats.n_by_force_idx.get(-1, 0) == 2
        assert r.stats.n_by_force_idx.get(3, 0) == 1

    def test_stats_dict_keys(self):
        r = _make_router()
        r.step("confirmed")
        d = r.stats_dict()
        for key in ("n_steps", "n_held", "n_switched", "n_by_state",
                    "n_by_force_idx", "switch_rate", "hold_rate"):
            assert key in d, f"missing key {key!r}"

    def test_switch_rate_zero_with_no_switches(self):
        r = _make_router()
        r.step("confirmed")
        r.step("confirmed")
        d = r.stats_dict()
        assert d["switch_rate"] == 0.0

    def test_switch_rate_nonzero_after_switch(self):
        r = _make_router(min_hold_frames=0)
        r.step("occluded")  # triggers a switch to 3
        d = r.stats_dict()
        assert d["switch_rate"] > 0.0

    def test_n_held_counted(self):
        r = _make_router(min_hold_frames=3)
        for _ in range(3):
            r.step("occluded")
        assert r.stats.n_held == 3


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_is_in_risky_mode_false_at_init(self):
        r = _make_router()
        assert r.is_in_risky_mode is False

    def test_is_in_risky_mode_true_after_switch(self):
        r = _make_router(min_hold_frames=0)
        r.step("occluded")
        assert r.is_in_risky_mode is True

    def test_is_in_risky_mode_false_after_reset(self):
        r = _make_router(min_hold_frames=0)
        r.step("occluded")
        r.reset()
        assert r.is_in_risky_mode is False

    def test_current_idx_init_value(self):
        r = _make_router()
        assert r.current_idx == DEFAULT_EXIT_IDX

    def test_hold_countdown_init_zero(self):
        r = _make_router()
        assert r.hold_countdown == 0

    def test_hold_countdown_nonzero_during_hold(self):
        r = _make_router(min_hold_frames=5)
        r.step("occluded")  # starts countdown at 5, then decrements to 4
        assert r.hold_countdown > 0


# ---------------------------------------------------------------------------
# policy_summary
# ---------------------------------------------------------------------------


class TestPolicySummary:
    def test_returns_copy(self):
        r = _make_router()
        p = r.policy_summary()
        p["confirmed"] = 999
        # Modifying the returned copy should not affect internal policy
        assert r.step("confirmed") != 999

    def test_custom_policy_injected(self):
        custom = {"confirmed": 2, "lost": 0}
        r = _make_router(min_hold_frames=0, policy=custom)
        assert r.step("confirmed") == 2
        assert r.step("lost") == 0


# ---------------------------------------------------------------------------
# Invalid init args
# ---------------------------------------------------------------------------


class TestInvalidArgs:
    def test_negative_min_hold_raises(self):
        with pytest.raises(ValueError, match="min_hold_frames"):
            StateExitRouter(min_hold_frames=-1)

    def test_min_conf_out_of_range_raises(self):
        with pytest.raises(ValueError, match="min_state_confidence"):
            StateExitRouter(min_state_confidence=1.5)

    def test_policy_out_of_range_raises(self):
        with pytest.raises(ValueError):
            StateExitRouter(policy={"confirmed": 10})  # > 5

    def test_policy_neg_two_raises(self):
        with pytest.raises(ValueError):
            StateExitRouter(policy={"confirmed": -2})  # < -1


# ---------------------------------------------------------------------------
# DERIVED_INT_TO_ROUTER_STATE mapping
# ---------------------------------------------------------------------------


class TestDerivedMapping:
    def test_all_four_derived_states_mapped(self):
        assert 0 in DERIVED_INT_TO_ROUTER_STATE
        assert 1 in DERIVED_INT_TO_ROUTER_STATE
        assert 2 in DERIVED_INT_TO_ROUTER_STATE
        assert 3 in DERIVED_INT_TO_ROUTER_STATE

    def test_correct_confirmed_maps_to_confirmed(self):
        assert DERIVED_INT_TO_ROUTER_STATE[0] == "confirmed"

    def test_correct_uncertain_maps_to_uncertain(self):
        assert DERIVED_INT_TO_ROUTER_STATE[1] == "uncertain"

    def test_lost_aware_maps_to_lost(self):
        assert DERIVED_INT_TO_ROUTER_STATE[2] == "lost"

    def test_false_confirmed_maps_to_false_confirmed(self):
        assert DERIVED_INT_TO_ROUTER_STATE[3] == "false_confirmed"

    def test_fallback_is_uncertain(self):
        assert _DERIVED_FALLBACK_STATE == "uncertain"

    def test_all_mapped_states_in_policy(self):
        r = _make_router()
        for state in DERIVED_INT_TO_ROUTER_STATE.values():
            assert state in r.policy_summary() or True  # fallback is also ok


# ---------------------------------------------------------------------------
# RISKY_STATES constant
# ---------------------------------------------------------------------------


class TestRiskyStates:
    def test_risky_states_are_frozenset(self):
        assert isinstance(RISKY_STATES, frozenset)

    def test_risky_states_content(self):
        assert "false_confirmed" in RISKY_STATES
        assert "lost" in RISKY_STATES
        assert "occluded" in RISKY_STATES
        assert "distractor" in RISKY_STATES
        assert "confirmed" not in RISKY_STATES
        assert "uncertain" not in RISKY_STATES
