"""TrajectoryAwareScheduler — derivative-sensitive hysteresis (Phase 5).

Variant of HysteresisBinaryScheduler that shortens the ``confirm_frames``
requirement when the signal is *rising fast*:

    effective_confirm = max(1, confirm_frames - int(signal_derivative / tau))

where ``signal_derivative`` is the finite-difference slope of the signal
over the last two reliable frames and ``tau`` is a user-supplied sensitivity
constant (default 0.1 / frame).

This makes the scheduler more responsive during sharp deterioration events
(fast signal rise) while preserving the full hysteresis window for slow drifts.

Registration key: ``"trajectory_aware"``
"""

from __future__ import annotations

from uav_tracker.registry import SCHEDULERS
from uav_tracker.types import SchedulerDecision, SignalReport

_LIGHT = 0
_DEEP = 1


@SCHEDULERS.register("trajectory_aware")
class TrajectoryAwareScheduler:
    """Hysteresis scheduler with derivative-sensitive confirm window.

    Parameters
    ----------
    E_hi:
        Upper threshold for LIGHT→DEEP escalation (default 0.65).
    E_lo:
        Lower threshold for DEEP→LIGHT de-escalation (default 0.50).
    confirm_frames:
        Base confirm window before escalation commits (default 5).
    cooldown_frames:
        Post-switch lockout in frames (default 5).
    tau:
        Derivative sensitivity constant.  When the signal derivative exceeds
        ``tau``, each extra ``tau`` reduces ``confirm_frames`` by 1, down to
        a minimum of 1 (default 0.1 / frame).
    signal_name:
        Which signal key to consume (default ``"motion_entropy"``).
    """

    name: str = "trajectory_aware"
    tiers: int = 2

    def __init__(
        self,
        E_hi: float = 0.65,
        E_lo: float = 0.50,
        confirm_frames: int = 5,
        cooldown_frames: int = 5,
        tau: float = 0.1,
        signal_name: str = "motion_entropy",
    ) -> None:
        self.E_hi = E_hi
        self.E_lo = E_lo
        self.confirm_frames = confirm_frames
        self.cooldown_frames = cooldown_frames
        self.tau = tau
        self.signal_name = signal_name

        self._tier: int = _LIGHT
        self._confirm_count: int = 0
        self._cooldown_left: int = 0
        self._prev_value: float | None = None  # last reliable signal value

    # ------------------------------------------------------------------

    def decide(
        self,
        signals: dict[str, SignalReport],
        current_tier: int,
        frame_idx: int,
    ) -> SchedulerDecision:
        """Advance state machine and return tier decision."""
        value, reliable = self._pick_signal(signals)

        if not reliable:
            return SchedulerDecision(
                tier=self._tier,
                reason=f"unreliable signal — holding tier {self._tier}",
                switched=False,
            )

        # Compute derivative from last reliable frame.
        if self._prev_value is not None:
            derivative = value - self._prev_value
        else:
            derivative = 0.0
        self._prev_value = value

        # Effective confirm window (shorter when rising fast).
        if self.tau > 0 and derivative > 0:
            shortening = int(derivative / self.tau)
        else:
            shortening = 0
        effective_confirm = max(1, self.confirm_frames - shortening)

        # Cooldown ticks down.
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

        # Hysteresis state machine.
        if self._tier == _LIGHT:
            if value > self.E_hi:
                self._confirm_count += 1
                if self._confirm_count >= effective_confirm:
                    self._tier = _DEEP
                    self._confirm_count = 0
                    self._cooldown_left = self.cooldown_frames
                    switched = True
                    reason = (
                        f"LIGHT→DEEP: signal {value:.3f} > E_hi {self.E_hi} "
                        f"sustained {effective_confirm} frames "
                        f"(effective_confirm={effective_confirm}, deriv={derivative:.3f})"
                    )
                else:
                    switched = False
                    reason = (
                        f"LIGHT: {value:.3f} > E_hi {self.E_hi} "
                        f"({self._confirm_count}/{effective_confirm} confirms, "
                        f"deriv={derivative:.3f})"
                    )
            else:
                self._confirm_count = 0
                switched = False
                reason = f"LIGHT: {value:.3f} ≤ E_hi {self.E_hi} — staying"
        else:  # _DEEP
            if value < self.E_lo:
                self._confirm_count += 1
                if self._confirm_count >= self.confirm_frames:
                    self._tier = _LIGHT
                    self._confirm_count = 0
                    self._cooldown_left = self.cooldown_frames
                    switched = True
                    reason = (
                        f"DEEP→LIGHT: signal {value:.3f} < E_lo {self.E_lo} "
                        f"sustained {self.confirm_frames} frames"
                    )
                else:
                    switched = False
                    reason = (
                        f"DEEP: {value:.3f} < E_lo {self.E_lo} "
                        f"({self._confirm_count}/{self.confirm_frames} confirms)"
                    )
            else:
                self._confirm_count = 0
                switched = False
                reason = f"DEEP: {value:.3f} ≥ E_lo {self.E_lo} — staying"

        return SchedulerDecision(tier=self._tier, reason=reason, switched=switched)

    def reset(self) -> None:
        """Restore to construction state. Idempotent."""
        self._tier = _LIGHT
        self._confirm_count = 0
        self._cooldown_left = 0
        self._prev_value = None

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


__all__ = ["TrajectoryAwareScheduler"]
