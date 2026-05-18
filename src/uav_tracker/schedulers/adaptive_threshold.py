"""AdaptiveThresholdScheduler — per-sequence percentile thresholds (Phase 5).

Variant of HysteresisBinaryScheduler that adapts E_hi and E_lo to the
rolling distribution of the signal so far:

    E_hi = percentile(window, high_percentile)   (default 75th)
    E_lo = percentile(window, low_percentile)    (default 25th)

During warm-up (before the window fills), falls back to fixed thresholds.

Otherwise follows the exact same hysteresis state machine as
HysteresisBinaryScheduler (confirm counter, cooldown window).

Registration key: ``"adaptive_threshold"``
"""

from __future__ import annotations

from collections import deque

import numpy as np

from uav_tracker.registry import SCHEDULERS
from uav_tracker.types import SchedulerDecision, SignalReport

_LIGHT = 0
_DEEP = 1


@SCHEDULERS.register("adaptive_threshold")
class AdaptiveThresholdScheduler:
    """Percentile-adaptive hysteresis scheduler.

    Parameters
    ----------
    window_size:
        Rolling window of recent signal values used to compute percentile
        thresholds (default 30 frames).
    high_percentile:
        Percentile of the rolling window used as the escalation threshold
        (default 75 → top quartile of recent signal).
    low_percentile:
        Percentile used as the de-escalation threshold (default 25 →
        bottom quartile).
    warmup_E_hi:
        Fixed upper threshold used during warm-up (before window fills).
    warmup_E_lo:
        Fixed lower threshold used during warm-up.
    confirm_frames:
        Consecutive reliable frames above/below threshold before tier
        commits (same as HysteresisBinaryScheduler; default 5).
    cooldown_frames:
        Post-switch lockout in frames (default 5).
    signal_name:
        Which signal key to consume (default ``"motion_entropy"``).
    """

    name: str = "adaptive_threshold"
    tiers: int = 2

    def __init__(
        self,
        window_size: int = 30,
        high_percentile: float = 75.0,
        low_percentile: float = 25.0,
        warmup_E_hi: float = 0.65,
        warmup_E_lo: float = 0.50,
        confirm_frames: int = 5,
        cooldown_frames: int = 5,
        signal_name: str = "motion_entropy",
    ) -> None:
        self.window_size = window_size
        self.high_percentile = high_percentile
        self.low_percentile = low_percentile
        self.warmup_E_hi = warmup_E_hi
        self.warmup_E_lo = warmup_E_lo
        self.confirm_frames = confirm_frames
        self.cooldown_frames = cooldown_frames
        self.signal_name = signal_name

        self._window: deque[float] = deque(maxlen=window_size)
        self._tier: int = _LIGHT
        self._confirm_count: int = 0
        self._cooldown_left: int = 0

    # ------------------------------------------------------------------

    def decide(
        self,
        signals: dict[str, SignalReport],
        current_tier: int,
        frame_idx: int,
    ) -> SchedulerDecision:
        """Advance state machine and return tier decision."""
        prev_tier = self._tier
        value, reliable = self._pick_signal(signals)

        if not reliable:
            return SchedulerDecision(
                tier=self._tier,
                reason=f"unreliable signal — holding tier {self._tier}",
                switched=False,
            )

        # Update rolling window with this reliable value.
        self._window.append(value)

        # Compute adaptive thresholds (or use warmup values).
        if len(self._window) >= self.window_size:
            arr = np.array(self._window)
            E_hi = float(np.percentile(arr, self.high_percentile))
            E_lo = float(np.percentile(arr, self.low_percentile))
        else:
            E_hi = self.warmup_E_hi
            E_lo = self.warmup_E_lo

        # Cooldown ticks down on every reliable frame.
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self._confirm_count = 0
            return SchedulerDecision(
                tier=self._tier,
                reason=(
                    f"cooldown ({self._cooldown_left + 1} frames left) "
                    f"— holding tier {self._tier}"
                ),
                switched=False,
            )

        # Hysteresis state machine (identical logic to HysteresisBinaryScheduler).
        if self._tier == _LIGHT:
            if value > E_hi:
                self._confirm_count += 1
                if self._confirm_count >= self.confirm_frames:
                    self._tier = _DEEP
                    self._confirm_count = 0
                    self._cooldown_left = self.cooldown_frames
                    switched = True
                    reason = (
                        f"LIGHT→DEEP: signal {value:.3f} > E_hi {E_hi:.3f} "
                        f"sustained {self.confirm_frames} frames"
                    )
                else:
                    switched = False
                    reason = (
                        f"LIGHT: {value:.3f} > E_hi {E_hi:.3f} "
                        f"({self._confirm_count}/{self.confirm_frames} confirms)"
                    )
            else:
                self._confirm_count = 0
                switched = False
                reason = f"LIGHT: {value:.3f} ≤ E_hi {E_hi:.3f} — staying"
        else:  # _DEEP
            if value < E_lo:
                self._confirm_count += 1
                if self._confirm_count >= self.confirm_frames:
                    self._tier = _LIGHT
                    self._confirm_count = 0
                    self._cooldown_left = self.cooldown_frames
                    switched = True
                    reason = (
                        f"DEEP→LIGHT: signal {value:.3f} < E_lo {E_lo:.3f} "
                        f"sustained {self.confirm_frames} frames"
                    )
                else:
                    switched = False
                    reason = (
                        f"DEEP: {value:.3f} < E_lo {E_lo:.3f} "
                        f"({self._confirm_count}/{self.confirm_frames} confirms)"
                    )
            else:
                self._confirm_count = 0
                switched = False
                reason = f"DEEP: {value:.3f} ≥ E_lo {E_lo:.3f} — staying"

        return SchedulerDecision(tier=self._tier, reason=reason, switched=switched)

    def reset(self) -> None:
        """Restore to construction state. Idempotent."""
        self._window.clear()
        self._tier = _LIGHT
        self._confirm_count = 0
        self._cooldown_left = 0

    # ------------------------------------------------------------------

    def _pick_signal(
        self, signals: dict[str, SignalReport]
    ) -> tuple[float, bool]:
        if self.signal_name in signals:
            r = signals[self.signal_name]
            return r.value, r.reliable
        for r in signals.values():
            if r.reliable:
                return r.value, True
        if signals:
            first = next(iter(signals.values()))
            return first.value, False
        return 0.0, False


__all__ = ["AdaptiveThresholdScheduler"]
