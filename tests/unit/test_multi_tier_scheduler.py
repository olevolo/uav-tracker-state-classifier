"""Unit tests for MultiTierScheduler.

Tests cover:
- 2-tier (equivalent to HysteresisBinary) and 3-tier configurations.
- Upgrade / downgrade state transitions.
- Cooldown lock-out after a switch.
- Confirm-frame accumulation and reset on direction change.
- Unreliable signal hold behaviour.
- reset() idempotency.
"""

from __future__ import annotations

import pytest

from uav_tracker.schedulers.multi_tier import MultiTierScheduler
from uav_tracker.types import SignalReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sig(value: float, reliable: bool = True) -> dict[str, SignalReport]:
    return {"motion_entropy": SignalReport(value=value, reliable=reliable)}


def _decide(sched: MultiTierScheduler, value: float, reliable: bool = True):
    return sched.decide(_sig(value, reliable), current_tier=sched._tier, frame_idx=0)


# ---------------------------------------------------------------------------
# 2-tier tests
# ---------------------------------------------------------------------------


class TestTwoTier:
    def setup_method(self):
        self.sched = MultiTierScheduler(
            tier_thresholds=[(0.65, 0.50)],
            confirm_frames=3,
            cooldown_frames=3,
            signal_name="motion_entropy",
        )

    def test_initial_state(self):
        assert self.sched._tier == 0
        assert self.sched.n_tiers == 2

    def test_upgrade_needs_confirm_frames(self):
        # 2 frames above E_hi — not yet committed.
        for _ in range(2):
            dec = _decide(self.sched, 0.80)
            assert dec.tier == 0
            assert not dec.switched

    def test_upgrade_commits_on_confirm_frames(self):
        for _ in range(2):
            _decide(self.sched, 0.80)
        dec = _decide(self.sched, 0.80)  # 3rd frame
        assert dec.tier == 1
        assert dec.switched

    def test_downgrade_after_upgrade(self):
        # Upgrade first.
        for _ in range(3):
            _decide(self.sched, 0.80)
        # Drain cooldown (3 frames).
        for _ in range(3):
            _decide(self.sched, 0.58)  # within band — no direction
        assert self.sched._tier == 1
        # Now drive below E_lo for confirm_frames.
        for _ in range(2):
            _decide(self.sched, 0.40)
        dec = _decide(self.sched, 0.40)
        assert dec.tier == 0
        assert dec.switched

    def test_cooldown_prevents_immediate_re_upgrade(self):
        # Upgrade.
        for _ in range(3):
            _decide(self.sched, 0.80)
        assert self.sched._tier == 1
        # Immediately try to drive downgrade — should be blocked by cooldown.
        for _ in range(3):
            dec = _decide(self.sched, 0.30)
            assert dec.tier == 1  # still in cooldown
            assert not dec.switched

    def test_unreliable_holds_state_no_confirm_advance(self):
        # 2 reliable frames above E_hi.
        _decide(self.sched, 0.80)
        _decide(self.sched, 0.80)
        # Unreliable frame — should not advance confirm counter.
        dec = _decide(self.sched, 0.80, reliable=False)
        assert dec.tier == 0
        assert not dec.switched
        # Still at confirm_count == 2 (unreliable did NOT advance).
        assert self.sched._confirm_count == 2

    def test_confirm_counter_resets_on_band_entry(self):
        _decide(self.sched, 0.80)
        _decide(self.sched, 0.80)
        # Signal drops back into band — counter should reset.
        _decide(self.sched, 0.60)  # between E_lo (0.50) and E_hi (0.65)
        assert self.sched._confirm_count == 0

    def test_reset_idempotent(self):
        for _ in range(3):
            _decide(self.sched, 0.80)
        self.sched.reset()
        assert self.sched._tier == 0
        assert self.sched._confirm_count == 0
        assert self.sched._cooldown_left == 0
        # Second reset doesn't change anything.
        self.sched.reset()
        assert self.sched._tier == 0


# ---------------------------------------------------------------------------
# 3-tier tests
# ---------------------------------------------------------------------------


class TestThreeTier:
    def setup_method(self):
        # Boundaries: tier 0↔1 at (0.50, 0.35), tier 1↔2 at (0.80, 0.65)
        self.sched = MultiTierScheduler(
            tier_thresholds=[(0.50, 0.35), (0.80, 0.65)],
            confirm_frames=2,
            cooldown_frames=2,
            signal_name="motion_entropy",
        )

    def test_three_tiers_configured(self):
        assert self.sched.n_tiers == 3
        assert self.sched.tiers == 3

    def test_upgrade_to_tier1(self):
        _decide(self.sched, 0.60)  # above E_hi[0]=0.50
        dec = _decide(self.sched, 0.60)
        assert dec.tier == 1
        assert dec.switched

    def test_upgrade_to_tier2_from_tier1(self):
        # Reach tier 1.
        _decide(self.sched, 0.60)
        _decide(self.sched, 0.60)
        # Drain cooldown (2 frames).
        _decide(self.sched, 0.70)  # between 0.65 and 0.80 — no direction
        _decide(self.sched, 0.70)
        # Now upgrade to tier 2.
        _decide(self.sched, 0.90)
        dec = _decide(self.sched, 0.90)
        assert dec.tier == 2
        assert dec.switched

    def test_downgrade_from_tier2_to_tier1(self):
        # Fast-forward to tier 2.
        self.sched._tier = 2  # force state for test speed
        # Drain confirm counter and cooldown reset.
        self.sched._confirm_count = 0
        self.sched._cooldown_left = 0
        self.sched._pending_direction = 0
        # Signal below E_lo[1]=0.65 for confirm_frames.
        _decide(self.sched, 0.60)
        dec = _decide(self.sched, 0.60)
        assert dec.tier == 1
        assert dec.switched

    def test_direction_change_resets_confirm(self):
        # 1 frame upgrading.
        _decide(self.sched, 0.60)
        assert self.sched._confirm_count == 1
        # Signal drops below E_hi — at tier 0 there's no downgrade target,
        # so pending_direction resets to 0 and confirm_count resets to 0.
        _decide(self.sched, 0.40)
        assert self.sched._confirm_count == 0
        assert self.sched._pending_direction == 0

    def test_accepts_list_of_lists_threshold_format(self):
        # YAML parses [[0.50, 0.35], [0.80, 0.65]] as list of lists.
        sched = MultiTierScheduler(
            tier_thresholds=[[0.50, 0.35], [0.80, 0.65]],
            confirm_frames=2,
            cooldown_frames=2,
        )
        assert sched.n_tiers == 3
        assert sched.tier_thresholds[0] == (0.50, 0.35)
        assert sched.tier_thresholds[1] == (0.80, 0.65)


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


def test_registered_in_schedulers_registry():
    from uav_tracker.registry import SCHEDULERS
    import uav_tracker  # trigger registration
    assert "multi_tier" in SCHEDULERS


def test_lost_frames_threshold_accepted_as_kwarg():
    """Config YAML includes lost_frames_threshold; must not raise."""
    sched = MultiTierScheduler(
        tier_thresholds=[(0.50, 0.35)],
        lost_frames_threshold=15,
    )
    assert sched.lost_frames_threshold == 15
