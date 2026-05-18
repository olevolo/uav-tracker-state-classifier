"""MultiTierScheduler — Phase 6 N-tier hysteresis state machine.

Generalises ``HysteresisBinaryScheduler`` to N tiers (N-1 threshold pairs).
Each adjacent pair (i, i+1) has its own ``(E_hi_i, E_lo_i)`` hysteresis band.
Upgrade and downgrade transitions require ``confirm_frames`` consecutive
reliable frames above/below the respective threshold, plus a per-direction
cooldown of ``cooldown_frames`` frames.

ADR-0008 notes:
  - The multi-tier scheduler is tier-agnostic about what *type* of plugin
    occupies a tier (tracker vs. detector). Tier-2 re-detection semantics
    live in HybridRunner, not here.
  - Confirm counter resets when the signal crosses back within the band
    (not when an unreliable frame arrives — those are held, not reset).
  - Cooldown is shared across all tier boundaries for simplicity; per-
    boundary cooldowns are a future extension.

Registration key: ``"multi_tier"``
"""

from __future__ import annotations

from uav_tracker.registry import SCHEDULERS
from uav_tracker.types import SchedulerDecision, SignalReport


@SCHEDULERS.register("multi_tier")
class MultiTierScheduler:
    """N-tier hysteresis scheduler with confirm + cooldown windows.

    Parameters
    ----------
    tier_thresholds:
        List of ``(E_hi, E_lo)`` pairs for transitions between adjacent tiers.
        For N tiers there must be N-1 pairs.
        Example (3-tier): ``[(0.50, 0.35), (0.80, 0.65)]``:
          - tier 0→1: upgrade when signal > 0.50 for confirm_frames; downgrade
            back when < 0.35.
          - tier 1→2: upgrade when signal > 0.80; downgrade back when < 0.65.
    confirm_frames:
        Consecutive reliable frames a crossing must be sustained before the
        tier is committed.
    cooldown_frames:
        After any tier change, this many reliable frames must pass before
        another change is allowed.
    signal_name:
        Which ``SignalReport`` key drives decisions. Falls back to the first
        reliable report if the named key is absent.
    lost_frames_threshold:
        Not used by this class but accepted (and ignored) as a config knob
        so ``hybrid_with_detection.yaml`` can include it without error.
    """

    name: str = "multi_tier"

    def __init__(
        self,
        tier_thresholds: list[tuple[float, float]] | list[list[float]] | None = None,
        confirm_frames: int = 5,
        cooldown_frames: int = 5,
        signal_name: str = "motion_entropy",
        lost_frames_threshold: int = 15,  # accepted but unused (runner concern)
    ) -> None:
        if tier_thresholds is None:
            tier_thresholds = [(0.50, 0.35)]

        # Normalise to list of (float, float).
        self.tier_thresholds: list[tuple[float, float]] = [
            (float(pair[0]), float(pair[1])) for pair in tier_thresholds
        ]
        self.n_tiers: int = len(self.tier_thresholds) + 1  # N-1 boundaries → N tiers
        self.confirm_frames = int(confirm_frames)
        self.cooldown_frames = int(cooldown_frames)
        self.signal_name = signal_name
        # lost_frames_threshold stored for completeness (e.g. tests may inspect it).
        self.lost_frames_threshold = int(lost_frames_threshold)

        # Mutable state.
        self._tier: int = 0
        self._confirm_count: int = 0
        self._cooldown_left: int = 0
        # Direction: +1 = watching for upgrade, -1 = watching for downgrade, 0 = idle
        self._pending_direction: int = 0

    # ------------------------------------------------------------------

    @property
    def tiers(self) -> int:
        """Number of configured tiers."""
        return self.n_tiers

    def decide(
        self,
        signals: dict[str, SignalReport],
        current_tier: int,
        frame_idx: int,
    ) -> SchedulerDecision:
        """Advance state machine and return the tier for this frame.

        Parameters
        ----------
        signals:
            Dict mapping signal name → ``SignalReport``.
        current_tier:
            The tier active on the previous frame (not used internally —
            we track our own state; provided for API symmetry with
            HysteresisBinaryScheduler).
        frame_idx:
            Zero-based frame index (informational).
        """
        # Pick signal value.
        value, reliable = self._pick_signal(signals)

        # Unreliable frames hold everything (no counter advance, no reset).
        if not reliable:
            return SchedulerDecision(
                tier=self._tier,
                reason=f"unreliable signal — holding tier {self._tier}",
                switched=False,
            )

        # Cooldown ticks on every reliable frame.
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self._confirm_count = 0
            self._pending_direction = 0
            return SchedulerDecision(
                tier=self._tier,
                reason=(
                    f"cooldown ({self._cooldown_left + 1} frames left)"
                    f" — holding tier {self._tier}"
                ),
                switched=False,
            )

        # Determine candidate transitions.
        # Upgrade: can we go from current tier to tier+1?
        # Downgrade: can we go from current tier to tier-1?
        tier = self._tier

        # Upgrade check: boundary above current tier.
        upgrade_possible = tier < self.n_tiers - 1
        downgrade_possible = tier > 0

        if upgrade_possible:
            E_hi, _ = self.tier_thresholds[tier]
            wants_upgrade = value > E_hi
        else:
            wants_upgrade = False

        if downgrade_possible:
            _, E_lo = self.tier_thresholds[tier - 1]
            wants_downgrade = value < E_lo
        else:
            wants_downgrade = False

        # If both upgrade and downgrade could be wanted (shouldn't normally happen
        # since bands don't overlap in sane configs), upgrade wins.
        if wants_upgrade:
            new_direction = +1
        elif wants_downgrade:
            new_direction = -1
        else:
            new_direction = 0

        # Reset confirm counter if direction changed.
        if new_direction != self._pending_direction:
            self._confirm_count = 0
            self._pending_direction = new_direction

        if new_direction == 0:
            # Signal within band — hold.
            self._confirm_count = 0
            if upgrade_possible:
                E_hi, _ = self.tier_thresholds[tier]
                reason = f"tier {tier}: signal {value:.3f} within band — holding"
            elif downgrade_possible:
                _, E_lo = self.tier_thresholds[tier - 1]
                reason = f"tier {tier}: signal {value:.3f} within band — holding"
            else:
                reason = f"tier {tier}: only tier (no transition possible)"
            return SchedulerDecision(tier=self._tier, reason=reason, switched=False)

        # Accumulate confirm count.
        self._confirm_count += 1

        if self._confirm_count >= self.confirm_frames:
            # Commit the transition.
            old_tier = self._tier
            if new_direction == +1:
                self._tier = tier + 1
                E_hi, _ = self.tier_thresholds[tier]
                reason = (
                    f"tier {old_tier}→{self._tier}: signal {value:.3f} > E_hi {E_hi:.3f} "
                    f"sustained {self.confirm_frames} frames"
                )
            else:
                self._tier = tier - 1
                _, E_lo = self.tier_thresholds[tier - 1]
                reason = (
                    f"tier {old_tier}→{self._tier}: signal {value:.3f} < E_lo {E_lo:.3f} "
                    f"sustained {self.confirm_frames} frames"
                )
            self._confirm_count = 0
            self._pending_direction = 0
            self._cooldown_left = self.cooldown_frames
            return SchedulerDecision(
                tier=self._tier,
                reason=reason,
                switched=True,
            )

        # Still accumulating confirms.
        if new_direction == +1 and upgrade_possible:
            E_hi, _ = self.tier_thresholds[tier]
            reason = (
                f"tier {tier}: upgrade pending {value:.3f} > E_hi {E_hi:.3f} "
                f"({self._confirm_count}/{self.confirm_frames})"
            )
        else:
            _, E_lo = self.tier_thresholds[tier - 1]
            reason = (
                f"tier {tier}: downgrade pending {value:.3f} < E_lo {E_lo:.3f} "
                f"({self._confirm_count}/{self.confirm_frames})"
            )

        return SchedulerDecision(tier=self._tier, reason=reason, switched=False)

    def reset(self) -> None:
        """Restore state to construction defaults. Idempotent."""
        self._tier = 0
        self._confirm_count = 0
        self._cooldown_left = 0
        self._pending_direction = 0

    # ------------------------------------------------------------------

    def _pick_signal(self, signals: dict[str, SignalReport]) -> tuple[float, bool]:
        """Return (value, reliable) for the configured signal name.

        Priority:
        1. Named ``signal_name`` key (any reliability).
        2. First reliable report in dict order.
        3. Unreliable fallback (0.0, False).
        """
        if self.signal_name in signals:
            report = signals[self.signal_name]
            return report.value, report.reliable

        for report in signals.values():
            if report.reliable:
                return report.value, True

        if signals:
            first = next(iter(signals.values()))
            return first.value, False

        return 0.0, False


__all__ = ["MultiTierScheduler"]
