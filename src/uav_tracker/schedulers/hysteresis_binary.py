"""HysteresisBinaryScheduler — Phase 3 two-tier state machine.

Implements the binary hysteresis switching logic from PLAN §3.3:
  - LIGHT (tier 0) → DEEP (tier 1): signal stays above E_hi for
    ``confirm_frames`` consecutive reliable frames, then a cooldown
    of ``cooldown_frames`` frames must elapse before another switch.
  - DEEP  (tier 1) → LIGHT (tier 0): signal stays below E_lo for
    ``confirm_frames`` consecutive reliable frames, then same cooldown.
  - Unreliable reports (``SignalReport.reliable == False``) neither
    advance the confirm counter nor reset it — they pass through.

Registration key: ``"hysteresis_binary"``
"""

from __future__ import annotations

from uav_tracker.registry import SCHEDULERS
from uav_tracker.types import SchedulerDecision, SignalReport

# Tier constants for readability.
_LIGHT = 0
_DEEP = 1


@SCHEDULERS.register("hysteresis_binary")
class HysteresisBinaryScheduler:
    """Binary hysteresis scheduler with confirm + cooldown windows.

    Parameters
    ----------
    E_hi:
        Upper threshold.  When the selected signal crosses above this,
        the confirm counter starts counting toward a LIGHT→DEEP switch.
    E_lo:
        Lower threshold.  When the selected signal drops below this,
        the confirm counter starts counting toward a DEEP→LIGHT switch.
    confirm_frames:
        Number of consecutive reliable frames the signal must sustain
        a crossing before the tier is committed.
    cooldown_frames:
        After a switch, the scheduler is locked for this many frames
        before a new switch can begin accumulating confirms.
    signal_name:
        Which ``SignalReport`` key to drive decisions from.  If the named
        key is absent or unreliable, the scheduler falls back to the first
        reliable report; if none exist it holds the current tier.
    """

    name: str = "hysteresis_binary"
    tiers: int = 2

    def __init__(
        self,
        E_hi: float = 0.65,
        E_lo: float = 0.50,
        confirm_frames: int = 5,
        cooldown_frames: int = 5,
        signal_name: str = "tracker_confidence",
    ) -> None:
        self.E_hi = E_hi
        self.E_lo = E_lo
        self.confirm_frames = confirm_frames
        self.cooldown_frames = cooldown_frames
        self.signal_name = signal_name

        # Mutable state — reset() restores these.
        self._tier: int = _LIGHT
        self._confirm_count: int = 0   # frames sustaining the current crossing
        self._cooldown_left: int = 0   # frames until next switch is allowed

    # ------------------------------------------------------------------

    def decide(
        self,
        signals: dict[str, SignalReport],
        current_tier: int,
        frame_idx: int,
    ) -> SchedulerDecision:
        """Advance the state machine and return the tier for this frame.

        Parameters
        ----------
        signals:
            Dict mapping signal name → ``SignalReport``.
        current_tier:
            The tier that was active on the previous frame (for switched flag).
        frame_idx:
            Zero-based frame index (informational; not used for timing).

        Returns
        -------
        SchedulerDecision
            ``tier`` — the tier to use this frame.
            ``switched`` — whether tier changed vs the previous call.
            ``reason`` — human-readable explanation.
        """
        # Sync internal tier from runner-supplied current_tier on the
        # first call (frame_idx==0 or when runner resets without calling
        # our reset()).
        # We track the committed tier internally — if the runner passes
        # a different current_tier, trust our internal state (avoids
        # desyncs on per-sequence resets where runner may pass 0).
        prev_tier = self._tier

        # ----- pick the driving signal value ----------------------------
        value, reliable = self._pick_signal(signals)

        # ----- if unreliable, hold everything and return ----------------
        if not reliable:
            return SchedulerDecision(
                tier=self._tier,
                reason=f"unreliable signal — holding tier {self._tier}",
                switched=(self._tier != prev_tier),
            )

        # ----- cooldown ticks down on every reliable frame --------------
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            # While in cooldown, reset confirm counter to avoid stale
            # partial counts from before the cooldown.
            self._confirm_count = 0
            return SchedulerDecision(
                tier=self._tier,
                reason=(
                    f"cooldown ({self._cooldown_left + 1} frames left)"
                    f" — holding tier {self._tier}"
                ),
                switched=False,
            )

        # ----- state-machine logic --------------------------------------
        if self._tier == _LIGHT:
            # Watching for LIGHT → DEEP: signal > E_hi
            if value > self.E_hi:
                self._confirm_count += 1
                if self._confirm_count >= self.confirm_frames:
                    self._tier = _DEEP
                    self._confirm_count = 0
                    self._cooldown_left = self.cooldown_frames
                    switched = True
                    reason = (
                        f"LIGHT→DEEP: signal {value:.3f} > E_hi {self.E_hi} "
                        f"sustained {self.confirm_frames} frames"
                    )
                else:
                    switched = False
                    reason = (
                        f"LIGHT: signal {value:.3f} > E_hi {self.E_hi} "
                        f"({self._confirm_count}/{self.confirm_frames} confirms)"
                    )
            else:
                # Signal back below E_hi — reset confirm counter
                self._confirm_count = 0
                switched = False
                reason = (
                    f"LIGHT: signal {value:.3f} ≤ E_hi {self.E_hi} — staying"
                )

        else:  # self._tier == _DEEP
            # Watching for DEEP → LIGHT: signal < E_lo
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
                        f"DEEP: signal {value:.3f} < E_lo {self.E_lo} "
                        f"({self._confirm_count}/{self.confirm_frames} confirms)"
                    )
            else:
                # Signal above E_lo — reset confirm counter
                self._confirm_count = 0
                switched = False
                reason = (
                    f"DEEP: signal {value:.3f} ≥ E_lo {self.E_lo} — staying"
                )

        return SchedulerDecision(tier=self._tier, reason=reason, switched=switched)

    def reset(self) -> None:
        """Restore state to construction defaults. Idempotent."""
        self._tier = _LIGHT
        self._confirm_count = 0
        self._cooldown_left = 0

    # ------------------------------------------------------------------

    def _pick_signal(
        self, signals: dict[str, SignalReport]
    ) -> tuple[float, bool]:
        """Extract (value, reliable) from the reports dict.

        Priority:
        1. Named ``signal_name`` key — if present and reliable.
        2. Named ``signal_name`` key — if present but unreliable.
        3. First reliable report in dict order.
        4. Unreliable fallback (value=0.0, reliable=False).
        """
        # Try the configured signal name first.
        if self.signal_name in signals:
            report = signals[self.signal_name]
            return report.value, report.reliable

        # Fall back to first reliable report.
        for report in signals.values():
            if report.reliable:
                return report.value, True

        # All unreliable or empty dict.
        if signals:
            first = next(iter(signals.values()))
            return first.value, False

        return 0.0, False


__all__ = ["HysteresisBinaryScheduler"]
