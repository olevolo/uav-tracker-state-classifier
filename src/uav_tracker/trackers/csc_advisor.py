"""CSC-driven template-update advisor (Variant C).

Translates per-frame CSC state predictions into template-update gate decisions,
adding hysteresis (streak requirement + cooldown) to avoid rapid oscillation.

No neural network, no checkpoint. Purely rule-based, tracker-agnostic.

Background:
    The central risk in visual tracking is updating the template while the
    tracker is in false_confirmed, lost, or distractor state — this "burns in"
    the wrong object as the new template reference, cascading into further
    failures. CSC classifies the state per frame; CSCAdvisor gates updates.

Three gates (simplified from SALT-RD 5-gate logic):
    Gate 1 (state):    block immediately when state ∈ block_on_states.
    Gate 3 (streak):   require streak_required consecutive "safe" frames before
                       unblocking after a risky period.
    Gate 5 (cooldown): enforce cooldown_frames gap after each update.

Soft release: if a block persists for max_hold_frames consecutive frames,
force-allow to prevent indefinite template starvation.

This module is tracker-agnostic. The runner calls step() → gets AdvisorDecision
→ decides whether to call the tracker's template-update method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Default policy
# ---------------------------------------------------------------------------

BLOCK_STATES: frozenset[str] = frozenset({
    "false_confirmed",
    "lost",
    "distractor",
})

# occluded: target is temporarily hidden but tracker is not wrong.
# Neutral: don't count toward streak (neither advance nor reset).
NEUTRAL_STATES: frozenset[str] = frozenset({"occluded"})

SAFE_STATES: frozenset[str] = frozenset({"confirmed", "uncertain"})

# Map CSC 4-class DerivedState int → 6-class advisor state string.
# Mirrors DERIVED_INT_TO_ROUTER_STATE in exit_router.py.
DERIVED_INT_TO_ADVISOR_STATE: dict[int, str] = {
    0: "confirmed",        # CORRECT_CONFIRMED
    1: "uncertain",        # CORRECT_UNCERTAIN
    2: "lost",             # LOST_AWARE
    3: "false_confirmed",  # FALSE_CONFIRMED
}
_DERIVED_FALLBACK_STATE = "uncertain"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class AdvisorStats:
    n_steps: int = 0
    n_blocked: int = 0
    n_allowed: int = 0
    n_blocked_by_state: int = 0     # Gate 1 blocks
    n_blocked_by_streak: int = 0    # Gate 3 blocks
    n_blocked_by_cooldown: int = 0  # Gate 5 blocks
    n_soft_released: int = 0        # forced allow after max_hold_frames
    n_updates_notified: int = 0     # actual template updates that happened
    n_by_state: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Decision return type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdvisorDecision:
    """Result of one CSCAdvisor.step() call.

    Fields:
        blocked:              True = block the template update this frame.
        reason:               Human-readable gate label.
            "allowed"         no gate triggered.
            "state_<name>"    Gate 1 — blocked by risky CSC state.
            "streak"          Gate 3 — trusted streak not yet met.
            "cooldown"        Gate 5 — too soon after last update.
            "soft_release"    held too long; forced allow to prevent starvation.
        trusted_streak:       Current streak counter at time of decision.
        consecutive_blocked:  Consecutive blocked frames at time of decision.
    """
    blocked: bool
    reason: str
    trusted_streak: int
    consecutive_blocked: int


# ---------------------------------------------------------------------------
# CSCAdvisor
# ---------------------------------------------------------------------------

class CSCAdvisor:
    """Template-update gate driven by CSC state predictions.

    Usage::

        advisor = CSCAdvisor()
        # per frame, after CSC inference:
        decision = advisor.step(csc_state_str, frame_idx)
        if not decision.blocked:
            updated = tracker.try_update_template(...)
            if updated:
                advisor.notify_template_updated(frame_idx)

    Args:
        block_on_states:          States that immediately trigger a block.
                                  Default: {false_confirmed, lost, distractor}.
        streak_required:          Consecutive non-risky frames before unblocking.
                                  Gate 3. Default 5.
        cooldown_frames:          Minimum frames between template updates.
                                  Gate 5. Default 15.
        max_hold_frames:          Consecutive blocked frames before soft release.
                                  0 = disabled. Default 50.
        neutral_states_pause_streak: If True, neutral states (occluded) pause
                                  the streak counter without resetting it.
                                  Default True.
    """

    def __init__(
        self,
        block_on_states: Optional[frozenset[str]] = None,
        streak_required: int = 5,
        cooldown_frames: int = 15,
        max_hold_frames: int = 50,
        neutral_states_pause_streak: bool = True,
    ) -> None:
        if streak_required < 0:
            raise ValueError(f"streak_required must be >= 0, got {streak_required}")
        if cooldown_frames < 0:
            raise ValueError(f"cooldown_frames must be >= 0, got {cooldown_frames}")
        if max_hold_frames < 0:
            raise ValueError(f"max_hold_frames must be >= 0, got {max_hold_frames}")

        self._block_states: frozenset[str] = (
            block_on_states if block_on_states is not None else BLOCK_STATES
        )
        self._streak_required = streak_required
        self._cooldown = cooldown_frames
        self._max_hold = max_hold_frames
        self._neutral_pause = neutral_states_pause_streak

        # Internal state — reset() mirrors these initialisations
        self._trusted_streak: int = streak_required   # start "warmed up"
        self._consecutive_blocked: int = 0
        self._last_update_frame: int = -(cooldown_frames + 1)  # cooldown pre-satisfied
        self.stats = AdvisorStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, csc_state: str, frame_idx: int) -> AdvisorDecision:
        """Process one frame's CSC state and return gate decision.

        Args:
            csc_state:  CSC-predicted state string (6-class vocabulary or
                        any string; unknown states fall back to safe behaviour).
            frame_idx:  Current frame index (used for Gate 5 cooldown check).

        Returns:
            AdvisorDecision — call .blocked to get the gate result.
        """
        self.stats.n_steps += 1
        self.stats.n_by_state[csc_state] = self.stats.n_by_state.get(csc_state, 0) + 1

        blocked = False
        reason = "allowed"

        if csc_state in self._block_states:
            # Gate 1: risky state — block immediately
            blocked = True
            reason = f"state_{csc_state}"
            self.stats.n_blocked_by_state += 1
            self._trusted_streak = 0
            self._consecutive_blocked += 1

            # Soft release: prevent indefinite template starvation
            if self._max_hold > 0 and self._consecutive_blocked >= self._max_hold:
                blocked = False
                reason = "soft_release"
                self.stats.n_soft_released += 1
                self._consecutive_blocked = 0
                self._trusted_streak = self._streak_required  # reset warm
        else:
            # Not a risky state — clear block counter
            self._consecutive_blocked = 0

            # Advance or pause streak counter
            if csc_state in NEUTRAL_STATES and self._neutral_pause:
                pass  # occluded: pause, neither advance nor reset
            else:
                self._trusted_streak = min(
                    self._trusted_streak + 1,
                    self._streak_required + 1,  # cap to avoid overflow
                )

            # Gate 3: streak not yet satisfied
            if self._trusted_streak < self._streak_required:
                blocked = True
                reason = "streak"
                self.stats.n_blocked_by_streak += 1

            # Gate 5: cooldown after last update (only if not already blocked)
            if not blocked:
                if (frame_idx - self._last_update_frame) < self._cooldown:
                    blocked = True
                    reason = "cooldown"
                    self.stats.n_blocked_by_cooldown += 1

        if blocked:
            self.stats.n_blocked += 1
        else:
            self.stats.n_allowed += 1

        return AdvisorDecision(
            blocked=blocked,
            reason=reason,
            trusted_streak=self._trusted_streak,
            consecutive_blocked=self._consecutive_blocked,
        )

    def notify_template_updated(self, frame_idx: int) -> None:
        """Call this whenever the tracker actually updates its template.

        Resets the cooldown timer and streak so the advisor requires
        a fresh run of safe frames before allowing the next update.
        """
        self._last_update_frame = frame_idx
        self._trusted_streak = 0   # require fresh streak after update
        self.stats.n_updates_notified += 1

    def reset(self) -> None:
        """Reset all state for a new sequence."""
        self._trusted_streak = self._streak_required   # start warmed-up
        self._consecutive_blocked = 0
        self._last_update_frame = -(self._cooldown + 1)
        self.stats = AdvisorStats()

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def trusted_streak(self) -> int:
        """Current consecutive safe-frame streak counter."""
        return self._trusted_streak

    @property
    def consecutive_blocked(self) -> int:
        """Frames blocked consecutively since last safe window."""
        return self._consecutive_blocked

    def stats_dict(self) -> dict:
        """Return stats as a plain dict for JSONL telemetry logging."""
        s = self.stats
        n = s.n_steps
        return {
            "n_steps": n,
            "n_blocked": s.n_blocked,
            "n_allowed": s.n_allowed,
            "n_blocked_by_state": s.n_blocked_by_state,
            "n_blocked_by_streak": s.n_blocked_by_streak,
            "n_blocked_by_cooldown": s.n_blocked_by_cooldown,
            "n_soft_released": s.n_soft_released,
            "n_updates_notified": s.n_updates_notified,
            "n_by_state": dict(s.n_by_state),
            "block_rate": s.n_blocked / n if n > 0 else 0.0,
        }

    def __repr__(self) -> str:
        return (
            f"CSCAdvisor("
            f"streak={self._trusted_streak}/{self._streak_required}, "
            f"cooldown={self._cooldown}, "
            f"block_states={sorted(self._block_states)}, "
            f"n_steps={self.stats.n_steps}"
            f")"
        )
