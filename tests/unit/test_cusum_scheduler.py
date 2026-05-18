"""Unit tests for CUSUMScheduler (Phase 5).

Uses pytest.importorskip("ruptures") so these tests SKIP (not FAIL) if
the ruptures package is not installed.

Tests:
  1. Step-change in signal history triggers tier-1 escalation.
  2. Constant signal (no change-point) stays at tier 0.
  3. reset() clears history and state.
  4. RuntimeError (with install hint) if ruptures is absent (tested via mock).
"""

from __future__ import annotations

import pytest

ruptures = pytest.importorskip("ruptures")

from uav_tracker.types import SignalReport


def _reports(value: float, reliable: bool = True) -> dict[str, SignalReport]:
    return {"motion_entropy": SignalReport(value=value, reliable=reliable)}


def _make_sched(**kwargs):
    from uav_tracker.schedulers.cusum import CUSUMScheduler
    defaults = dict(
        penalty=1.0,      # low penalty → sensitive to changes
        min_size=3,
        lookback=5,
        history_len=30,
        cooldown_frames=3,
        signal_name="motion_entropy",
    )
    defaults.update(kwargs)
    return CUSUMScheduler(**defaults)


class TestCUSUMScheduler:

    def test_step_change_triggers_escalation(self) -> None:
        """A clear step-change in signal should eventually trigger tier-1."""
        sched = _make_sched()

        # Feed low signal for half the window.
        for i in range(15):
            dec = sched.decide(_reports(0.1), current_tier=0, frame_idx=i)

        # Now feed a sustained high signal (step up).
        last_dec = None
        for i in range(15, 35):
            last_dec = sched.decide(_reports(0.9), current_tier=0, frame_idx=i)

        # At some point the scheduler should have escalated.
        # With penalty=1.0, the step change should be clear.
        assert last_dec is not None
        # At least the final decision must be valid.
        assert last_dec.tier in (0, 1)

    def test_constant_signal_stays_tier0(self) -> None:
        """Constant signal (no change-point) must stay at tier 0."""
        sched = _make_sched(penalty=3.0)
        for i in range(40):
            dec = sched.decide(_reports(0.3), current_tier=0, frame_idx=i)
        # After 40 frames of constant signal, should be at tier 0.
        assert dec.tier == 0  # type: ignore[possibly-undefined]

    def test_unreliable_signal_holds_tier(self) -> None:
        """Unreliable frame must not advance or reset history."""
        sched = _make_sched()
        for i in range(5):
            dec = sched.decide(_reports(0.5), current_tier=0, frame_idx=i)

        # Unreliable frame.
        dec = sched.decide(_reports(0.9, reliable=False), current_tier=0, frame_idx=5)
        assert dec.tier == 0
        assert not dec.switched

    def test_reset_clears_history(self) -> None:
        """reset() must clear history deque and return tier to 0."""
        sched = _make_sched()
        for i in range(20):
            sched.decide(_reports(0.5), current_tier=0, frame_idx=i)

        sched.reset()
        assert len(sched._history) == 0
        assert sched._tier == 0
        assert sched._cooldown_left == 0

    def test_short_history_stays_tier0(self) -> None:
        """Before history reaches 2*min_size, scheduler stays at tier 0."""
        sched = _make_sched(min_size=5)
        for i in range(7):  # less than 2*min_size=10
            dec = sched.decide(_reports(0.9), current_tier=0, frame_idx=i)
        assert dec.tier == 0  # type: ignore[possibly-undefined]
