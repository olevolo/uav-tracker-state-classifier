"""Unit tests for HysteresisBinaryScheduler state machine (Phase 3)."""

from __future__ import annotations

import pytest

from uav_tracker.types import SignalReport


def _make_report(value: float, reliable: bool = True) -> SignalReport:
    return SignalReport(value=value, reliable=reliable)


def _reports(value: float, reliable: bool = True) -> dict[str, SignalReport]:
    return {"tracker_confidence": _make_report(value, reliable)}


@pytest.fixture()
def sched():
    from uav_tracker.schedulers.hysteresis_binary import HysteresisBinaryScheduler

    return HysteresisBinaryScheduler(
        E_hi=0.65,
        E_lo=0.50,
        confirm_frames=3,
        cooldown_frames=3,
        signal_name="tracker_confidence",
    )


# ---------------------------------------------------------------------------
# Basic tier-0 behavior
# ---------------------------------------------------------------------------


def test_stays_tier0_when_signal_below_E_hi(sched) -> None:
    """Signal below E_hi should never leave tier 0."""
    for i in range(20):
        dec = sched.decide(_reports(0.3), current_tier=0, frame_idx=i)
        assert dec.tier == 0
        assert not dec.switched


def test_no_switch_before_confirm_frames(sched) -> None:
    """Signal above E_hi but fewer than confirm_frames frames: stays tier 0."""
    # confirm_frames=3 so after 2 high frames we must still be in tier 0.
    for i in range(2):
        dec = sched.decide(_reports(0.9), current_tier=0, frame_idx=i)
        assert dec.tier == 0, f"Expected tier 0 on frame {i}, got {dec.tier}"
        assert not dec.switched


def test_switches_on_confirm_frame(sched) -> None:
    """Exactly at confirm_frames-th sustained high frame the switch fires."""
    # confirm_frames=3
    for i in range(2):
        dec = sched.decide(_reports(0.9), current_tier=0, frame_idx=i)
        assert dec.tier == 0

    # 3rd frame — should switch.
    dec = sched.decide(_reports(0.9), current_tier=0, frame_idx=2)
    assert dec.tier == 1
    assert dec.switched


def test_cooldown_prevents_immediate_back_switch(sched) -> None:
    """After LIGHT→DEEP switch, cooldown prevents immediate DEEP→LIGHT."""
    # Trigger switch to tier 1.
    for i in range(3):
        sched.decide(_reports(0.9), current_tier=0, frame_idx=i)

    # Now send low signal (below E_lo) — should be blocked by cooldown.
    for i in range(3, 6):
        dec = sched.decide(_reports(0.1), current_tier=1, frame_idx=i)
        assert dec.tier == 1, f"Expected tier 1 during cooldown at frame {i}"
        assert not dec.switched


def test_unreliable_report_does_not_advance_counter(sched) -> None:
    """An unreliable report must not advance the confirm counter."""
    # 2 reliable high frames.
    sched.decide(_reports(0.9), current_tier=0, frame_idx=0)
    sched.decide(_reports(0.9), current_tier=0, frame_idx=1)
    # 1 unreliable — counter must NOT advance.
    dec = sched.decide(_reports(0.9, reliable=False), current_tier=0, frame_idx=2)
    assert dec.tier == 0
    # Even with 2 + 1 (but unreliable) high frames, we should still be tier 0.
    assert not dec.switched


def test_unreliable_does_not_reset_counter(sched) -> None:
    """An unreliable frame after some confirms keeps the counter intact."""
    # 2 confirms accumulated.
    sched.decide(_reports(0.9), current_tier=0, frame_idx=0)
    sched.decide(_reports(0.9), current_tier=0, frame_idx=1)
    # Unreliable — counter preserved.
    sched.decide(_reports(0.9, reliable=False), current_tier=0, frame_idx=2)
    # 3rd reliable high frame — switch should fire.
    dec = sched.decide(_reports(0.9), current_tier=0, frame_idx=3)
    assert dec.tier == 1
    assert dec.switched


def test_reset_zeros_counters(sched) -> None:
    """reset() must restore state so the machine behaves as if freshly built."""
    # Trigger switch to tier 1.
    for i in range(3):
        sched.decide(_reports(0.9), current_tier=0, frame_idx=i)
    assert sched._tier == 1

    sched.reset()
    assert sched._tier == 0
    assert sched._confirm_count == 0
    assert sched._cooldown_left == 0

    # After reset, machine should behave as new.
    for i in range(2):
        dec = sched.decide(_reports(0.9), current_tier=0, frame_idx=i)
        assert dec.tier == 0


def test_reset_idempotent(sched) -> None:
    """Calling reset() twice must not raise and must leave state clean."""
    sched.reset()
    sched.reset()
    assert sched._tier == 0
    assert sched._confirm_count == 0


def test_confirm_counter_resets_on_signal_drop(sched) -> None:
    """Signal dropping below E_hi before confirm_frames resets the counter."""
    sched.decide(_reports(0.9), current_tier=0, frame_idx=0)
    sched.decide(_reports(0.9), current_tier=0, frame_idx=1)
    # Drop signal — counter should reset.
    sched.decide(_reports(0.3), current_tier=0, frame_idx=2)
    # Now 2 more high frames — not enough to switch (counter was reset).
    for i in range(3, 5):
        dec = sched.decide(_reports(0.9), current_tier=0, frame_idx=i)
    # After reset + 2 high frames, should still be tier 0.
    assert dec.tier == 0  # type: ignore[possibly-undefined]
