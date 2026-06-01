"""Unit tests for CSCAdvisor (csc_advisor.py).

Run: python -m pytest tests/unit/test_csc_advisor.py -v
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pytest
from uav_tracker.trackers.csc_advisor import (
    BLOCK_STATES,
    DERIVED_INT_TO_ADVISOR_STATE,
    NEUTRAL_STATES,
    SAFE_STATES,
    AdvisorDecision,
    AdvisorStats,
    CSCAdvisor,
    _DERIVED_FALLBACK_STATE,
)


# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------

class TestPolicyConstants:
    def test_block_states_content(self):
        assert "false_confirmed" in BLOCK_STATES
        assert "lost" in BLOCK_STATES
        assert "distractor" in BLOCK_STATES

    def test_safe_states_content(self):
        assert "confirmed" in SAFE_STATES
        assert "uncertain" in SAFE_STATES

    def test_neutral_states_content(self):
        assert "occluded" in NEUTRAL_STATES

    def test_no_overlap_between_block_and_safe(self):
        assert BLOCK_STATES.isdisjoint(SAFE_STATES)

    def test_no_overlap_between_block_and_neutral(self):
        assert BLOCK_STATES.isdisjoint(NEUTRAL_STATES)

    def test_derived_int_map_covers_four_classes(self):
        assert set(DERIVED_INT_TO_ADVISOR_STATE.keys()) == {0, 1, 2, 3}

    def test_derived_fallback_is_safe(self):
        assert _DERIVED_FALLBACK_STATE not in BLOCK_STATES

    def test_derived_false_confirmed_maps_correctly(self):
        assert DERIVED_INT_TO_ADVISOR_STATE[3] == "false_confirmed"

    def test_derived_lost_maps_correctly(self):
        assert DERIVED_INT_TO_ADVISOR_STATE[2] == "lost"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self):
        a = CSCAdvisor()
        assert a.trusted_streak >= 0
        assert a.consecutive_blocked == 0

    def test_custom_params(self):
        a = CSCAdvisor(streak_required=10, cooldown_frames=20, max_hold_frames=100)
        assert a._streak_required == 10
        assert a._cooldown == 20
        assert a._max_hold == 100

    def test_zero_streak_allowed(self):
        a = CSCAdvisor(streak_required=0)
        assert a._streak_required == 0

    def test_negative_streak_raises(self):
        with pytest.raises(ValueError, match="streak_required"):
            CSCAdvisor(streak_required=-1)

    def test_negative_cooldown_raises(self):
        with pytest.raises(ValueError, match="cooldown_frames"):
            CSCAdvisor(cooldown_frames=-1)

    def test_negative_max_hold_raises(self):
        with pytest.raises(ValueError, match="max_hold_frames"):
            CSCAdvisor(max_hold_frames=-1)

    def test_custom_block_states(self):
        a = CSCAdvisor(block_on_states=frozenset({"lost"}))
        assert a._block_states == frozenset({"lost"})

    def test_starts_warmed_up(self):
        a = CSCAdvisor(streak_required=5)
        # After construction, streak should be satisfied (starts warmed up)
        d = a.step("confirmed", frame_idx=0)
        # Streak ≥ required AND cooldown pre-satisfied → allowed
        assert not d.blocked


# ---------------------------------------------------------------------------
# Basic step behaviour
# ---------------------------------------------------------------------------

class TestStepBasic:
    def test_confirmed_allows(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0)
        d = a.step("confirmed", 0)
        assert not d.blocked
        assert d.reason == "allowed"

    def test_uncertain_allows(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0)
        d = a.step("uncertain", 0)
        assert not d.blocked

    def test_false_confirmed_blocks(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0)
        d = a.step("false_confirmed", 0)
        assert d.blocked
        assert "false_confirmed" in d.reason

    def test_lost_blocks(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0)
        d = a.step("lost", 0)
        assert d.blocked

    def test_distractor_blocks(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0)
        d = a.step("distractor", 0)
        assert d.blocked

    def test_unknown_state_allows(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0)
        d = a.step("nonexistent_xyz", 0)
        # Unknown state → treated as safe, allowed (streak=0, cooldown=0)
        assert not d.blocked

    def test_n_steps_increments(self):
        a = CSCAdvisor()
        for i in range(7):
            a.step("confirmed", i)
        assert a.stats.n_steps == 7


# ---------------------------------------------------------------------------
# Gate 1: state block
# ---------------------------------------------------------------------------

class TestGate1StateBlock:
    def test_block_resets_streak(self):
        a = CSCAdvisor(streak_required=5, cooldown_frames=0)
        # Warm up streak
        for i in range(5):
            a.step("confirmed", i)
        assert a.trusted_streak >= 5
        # Block resets streak to 0
        a.step("false_confirmed", 5)
        assert a.trusted_streak == 0

    def test_consecutive_blocked_increments(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0)
        a.step("lost", 0)
        a.step("lost", 1)
        a.step("lost", 2)
        assert a.consecutive_blocked == 3

    def test_consecutive_blocked_resets_on_safe(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0)
        a.step("lost", 0)
        a.step("lost", 1)
        a.step("confirmed", 2)
        assert a.consecutive_blocked == 0

    def test_blocked_by_state_counter(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0)
        a.step("false_confirmed", 0)
        a.step("lost", 1)
        assert a.stats.n_blocked_by_state == 2


# ---------------------------------------------------------------------------
# Gate 3: streak requirement
# ---------------------------------------------------------------------------

class TestGate3Streak:
    def test_fresh_advisor_no_streak_needed_when_warmed(self):
        a = CSCAdvisor(streak_required=5, cooldown_frames=0)
        # starts warmed — should allow immediately
        d = a.step("confirmed", 0)
        assert not d.blocked

    def test_streak_needed_after_block(self):
        a = CSCAdvisor(streak_required=3, cooldown_frames=0)
        a.step("false_confirmed", 0)  # resets streak to 0
        # Now need 3 consecutive safe frames
        d1 = a.step("confirmed", 1)
        assert d1.blocked
        assert d1.reason == "streak"
        d2 = a.step("confirmed", 2)
        assert d2.blocked
        assert d2.reason == "streak"
        d3 = a.step("confirmed", 3)  # streak satisfied
        assert not d3.blocked

    def test_streak_blocked_counter(self):
        a = CSCAdvisor(streak_required=3, cooldown_frames=0)
        a.step("lost", 0)  # reset streak
        a.step("confirmed", 1)
        a.step("confirmed", 2)
        assert a.stats.n_blocked_by_streak == 2

    def test_streak_zero_skips_gate3(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0)
        a.step("lost", 0)   # reset streak (now 0)
        d = a.step("confirmed", 1)  # streak=0 required → already satisfied
        assert not d.blocked

    def test_neutral_pauses_streak(self):
        a = CSCAdvisor(streak_required=3, cooldown_frames=0,
                       neutral_states_pause_streak=True)
        a.step("false_confirmed", 0)   # reset streak to 0
        a.step("confirmed", 1)         # streak → 1
        a.step("occluded", 2)          # neutral → pause, streak stays 1
        a.step("occluded", 3)          # neutral → pause, streak stays 1
        d = a.step("confirmed", 4)     # streak → 2 — still not 3 → blocked
        assert d.blocked
        d2 = a.step("confirmed", 5)    # streak → 3 → allowed
        assert not d2.blocked

    def test_neutral_no_pause_when_disabled(self):
        a = CSCAdvisor(streak_required=3, cooldown_frames=0,
                       neutral_states_pause_streak=False)
        a.step("false_confirmed", 0)   # reset streak
        a.step("confirmed", 1)         # streak 1
        a.step("occluded", 2)          # streak 2 (counts as safe when pause disabled)
        d = a.step("confirmed", 3)     # streak 3 → allowed
        assert not d.blocked


# ---------------------------------------------------------------------------
# Gate 5: cooldown
# ---------------------------------------------------------------------------

class TestGate5Cooldown:
    def test_cooldown_blocks_after_update(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=5)
        a.notify_template_updated(frame_idx=10)
        # Frames 11–14 should be blocked by cooldown
        for fi in range(11, 15):
            d = a.step("confirmed", fi)
            assert d.blocked, f"frame {fi} should be blocked by cooldown"
            assert d.reason == "cooldown"

    def test_cooldown_expires(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=3)
        a.notify_template_updated(frame_idx=0)
        # Frames 1, 2 blocked; frame 3 allowed
        assert a.step("confirmed", 1).blocked
        assert a.step("confirmed", 2).blocked
        assert not a.step("confirmed", 3).blocked

    def test_cooldown_counter(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=3)
        a.notify_template_updated(frame_idx=0)
        a.step("confirmed", 1)
        a.step("confirmed", 2)
        assert a.stats.n_blocked_by_cooldown == 2

    def test_zero_cooldown_never_blocks(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0)
        a.notify_template_updated(frame_idx=0)
        d = a.step("confirmed", 1)
        assert not d.blocked

    def test_cooldown_not_applied_during_risky_state(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=3)
        a.notify_template_updated(frame_idx=0)
        # Even in cooldown, a risky state should cite state, not cooldown
        d = a.step("false_confirmed", 1)
        assert d.blocked
        assert "false_confirmed" in d.reason


# ---------------------------------------------------------------------------
# Soft release
# ---------------------------------------------------------------------------

class TestSoftRelease:
    def test_soft_release_after_max_hold(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0, max_hold_frames=5)
        # Frames 0-3: blocked normally (consecutive_blocked 1→4)
        for fi in range(4):
            d = a.step("lost", fi)
            assert d.blocked, f"frame {fi} should be blocked"
        # Frame 4: 5th consecutive block → consecutive_blocked reaches 5 → soft release
        d = a.step("lost", 4)
        assert not d.blocked
        assert d.reason == "soft_release"

    def test_soft_release_counter(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0, max_hold_frames=3)
        for fi in range(4):
            a.step("lost", fi)
        assert a.stats.n_soft_released == 1

    def test_soft_release_disabled_when_max_hold_zero(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0, max_hold_frames=0)
        for fi in range(100):
            d = a.step("lost", fi)
            assert d.blocked  # never soft-released


# ---------------------------------------------------------------------------
# notify_template_updated
# ---------------------------------------------------------------------------

class TestNotifyTemplateUpdated:
    def test_notify_resets_streak(self):
        a = CSCAdvisor(streak_required=5, cooldown_frames=0)
        a.notify_template_updated(frame_idx=10)
        assert a.trusted_streak == 0

    def test_notify_sets_cooldown(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=10)
        a.notify_template_updated(frame_idx=100)
        d = a.step("confirmed", 101)
        assert d.blocked
        assert d.reason == "cooldown"

    def test_notify_counter(self):
        a = CSCAdvisor()
        a.notify_template_updated(10)
        a.notify_template_updated(50)
        assert a.stats.n_updates_notified == 2


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_streak_to_warmed(self):
        a = CSCAdvisor(streak_required=5, cooldown_frames=0)
        a.step("false_confirmed", 0)  # reset streak to 0
        a.reset()
        # After reset, starts warmed → step should be allowed
        d = a.step("confirmed", 1)
        assert not d.blocked

    def test_reset_clears_stats(self):
        a = CSCAdvisor()
        for i in range(10):
            a.step("lost", i)
        a.reset()
        assert a.stats.n_steps == 0
        assert a.stats.n_blocked == 0
        assert a.stats.n_by_state == {}

    def test_reset_clears_consecutive_blocked(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0)
        for i in range(5):
            a.step("lost", i)
        a.reset()
        assert a.consecutive_blocked == 0

    def test_reset_between_sequences(self):
        a = CSCAdvisor(streak_required=3, cooldown_frames=0)
        a.step("false_confirmed", 0)  # mid-block state
        a.reset()
        # Fresh sequence — should behave as warmed
        d = a.step("confirmed", 0)
        assert not d.blocked


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_dict_keys(self):
        a = CSCAdvisor()
        a.step("confirmed", 0)
        d = a.stats_dict()
        for key in ("n_steps", "n_blocked", "n_allowed", "n_blocked_by_state",
                    "n_blocked_by_streak", "n_blocked_by_cooldown",
                    "n_soft_released", "n_updates_notified", "n_by_state",
                    "block_rate"):
            assert key in d, f"missing key: {key}"

    def test_empty_stats_no_division_error(self):
        a = CSCAdvisor()
        d = a.stats_dict()
        assert d["block_rate"] == 0.0

    def test_block_rate_correct(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0)
        a.step("confirmed", 0)   # allowed
        a.step("lost", 1)        # blocked
        d = a.stats_dict()
        assert d["block_rate"] == pytest.approx(0.5)

    def test_n_by_state(self):
        a = CSCAdvisor(streak_required=0, cooldown_frames=0)
        a.step("confirmed", 0)
        a.step("confirmed", 1)
        a.step("lost", 2)
        assert a.stats.n_by_state["confirmed"] == 2
        assert a.stats.n_by_state["lost"] == 1


# ---------------------------------------------------------------------------
# AdvisorDecision dataclass
# ---------------------------------------------------------------------------

class TestAdvisorDecision:
    def test_frozen(self):
        d = AdvisorDecision(blocked=True, reason="streak",
                            trusted_streak=2, consecutive_blocked=0)
        with pytest.raises(Exception):
            d.blocked = False  # type: ignore[misc]

    def test_fields(self):
        d = AdvisorDecision(blocked=False, reason="allowed",
                            trusted_streak=5, consecutive_blocked=0)
        assert not d.blocked
        assert d.reason == "allowed"
        assert d.trusted_streak == 5


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_key_info(self):
        a = CSCAdvisor(streak_required=7, cooldown_frames=10)
        s = repr(a)
        assert "CSCAdvisor" in s
        assert "7" in s  # streak_required

    def test_repr_no_exception(self):
        a = CSCAdvisor()
        for i in range(5):
            a.step("confirmed" if i % 2 == 0 else "lost", i)
        repr(a)  # should not raise


# ---------------------------------------------------------------------------
# Full episode simulation
# ---------------------------------------------------------------------------

class TestEpisodeSimulation:
    def test_block_during_false_confirmed_episode(self):
        """Simulate a false_confirmed episode followed by recovery."""
        a = CSCAdvisor(streak_required=3, cooldown_frames=0)
        results = []
        for fi in range(20):
            if fi < 10:
                state = "false_confirmed"   # episode
            else:
                state = "confirmed"          # recovery
            d = a.step(state, fi)
            results.append(d.blocked)

        # All frames 0-9 should be blocked (Gate 1)
        assert all(results[:10])
        # After recovery streak of 3, frames should be allowed
        # frames 10,11: streak 1,2 → blocked by Gate 3
        # frame 12: streak 3 → allowed
        assert results[10] is True
        assert results[11] is True
        assert results[12] is False

    def test_update_then_cooldown_then_allow(self):
        """Template update triggers cooldown, then system re-allows."""
        a = CSCAdvisor(streak_required=0, cooldown_frames=5)
        # Allow first update
        d = a.step("confirmed", 0)
        assert not d.blocked
        a.notify_template_updated(frame_idx=0)
        # Frames 1-4 blocked by cooldown
        for fi in range(1, 5):
            assert a.step("confirmed", fi).blocked
        # Frame 5 allowed
        assert not a.step("confirmed", 5).blocked
