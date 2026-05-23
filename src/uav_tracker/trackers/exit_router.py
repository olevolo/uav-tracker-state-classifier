"""State-driven exit-block routing for SGLATrack.

CSC predicts the tracking state per frame. This router converts that state into
a force_layer_idx that the SGLATrack adapter applies to its next inference call,
overriding SGLATrack's internal MLP exit-router when needed.

Background (see project_sglatrack_mlp_collapse.md):
SGLATrack has 12 transformer blocks. Blocks 0-5 always run. After block 5, a
learned MLP picks ONE of blocks 6-11 as the exit. Empirical ablation showed the
MLP collapses to block 8 on UAV123 test data, and forcing block 9 instead under
risky states reduces False-Confirmed Rate without significant AUC loss on easy
sequences.

Ablation evidence (4 sequences, 2 UAV123 + 2 DTB70, Task #21):
  - default (block 8): Yacht4 AUC=0.37, car12 AUC=0.060, Animal1 AUC=0.731
  - block 6 (fast):    Yacht4 AUC=0.35 FCR=59%, Animal1 AUC=0.44 FCR=19%  [REJECTED]
  - block 9 (mid):     Yacht4 AUC=0.82 FCR=0%,  car12 FCR=-27% vs default  [ADOPTED]
  - block 11 (deep):   Animal1 AUC=0.638 FCR=12.2%                          [REJECTED]

This module is EXPERIMENTAL until full DTB70 validation passes (Task #24).

Architecture note: SGLATrack is a SINGLE-EXIT architecture. Force-selecting
block 9 means ONLY block 9 runs after block 5. Blocks 6, 7, 8 are skipped.
Total blocks executed is always 7 (0-5 plus 1 selected), regardless of which
exit block is selected. The "robust" mode is NOT a cascade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Policy V1 (Task #21 ablation): block 9 only, for risky states only.
# Rationale:
#   - block 6 hurts AUC catastrophically and adds FCR on easy sequences
#   - block 11 hurts AUC on easy DTB70 (Animal1)
#   - block 9 reduces FCR 27-100% on hard with delta_AUC <= 0.005 on easy
#
# force_layer_idx semantics (passed to SGLATrack backbone):
#   -1  = let MLP decide (collapsed to block 8 in practice on UAV123)
#    0  = force MLP output index 0 → block 6  (shallowest)
#    1  = force MLP output index 1 → block 7
#    2  = force MLP output index 2 → block 8  (what the collapsed MLP picks)
#    3  = force MLP output index 3 → block 9  (Policy V1 target)
#    4  = force MLP output index 4 → block 10
#    5  = force MLP output index 5 → block 11 (deepest)
#
# MLP output index k maps to block (k + 6) in 0-indexed block space.
# ---------------------------------------------------------------------------

STATE_TO_EXIT: dict[str, int] = {
    "confirmed":       -1,   # let MLP decide (which means block 8 due to collapse)
    "uncertain":       -1,   # let MLP decide
    "occluded":         3,   # force block 9
    "distractor":       3,   # force block 9
    "false_confirmed":  3,   # force block 9
    "lost":             3,   # force block 9
}

RISKY_STATES = frozenset({"occluded", "distractor", "false_confirmed", "lost"})
DEFAULT_EXIT_IDX = -1  # MLP decides
_VALID_FORCE_IDX_RANGE = (-1, 5)  # inclusive; -1 = MLP default, 0-5 = override

# ---------------------------------------------------------------------------
# DerivedState → router state string translation
#
# CSCPrediction.derived_state is an int from csc_lib DerivedState enum:
#   CORRECT_CONFIRMED = 0
#   CORRECT_UNCERTAIN = 1
#   LOST_AWARE        = 2
#   FALSE_CONFIRMED   = 3
#
# Map these to the 6-class state vocabulary used by STATE_TO_EXIT above.
# Note: the 4-class CSC derived state cannot distinguish occluded/distractor
# at runtime (those are aux_flags, not the derived state). Map them to their
# closest behavioral equivalent: CORRECT_UNCERTAIN → "uncertain" (MLP decides).
# ---------------------------------------------------------------------------

DERIVED_INT_TO_ROUTER_STATE: dict[int, str] = {
    0: "confirmed",        # CORRECT_CONFIRMED
    1: "uncertain",        # CORRECT_UNCERTAIN
    2: "lost",             # LOST_AWARE
    3: "false_confirmed",  # FALSE_CONFIRMED
}

# Fallback when derived_state int is out of range
_DERIVED_FALLBACK_STATE = "uncertain"


@dataclass
class RouterStats:
    """Accumulated routing decisions for one tracking sequence.

    Counters are updated by ``StateExitRouter.step``. Call ``reset()`` at the
    start of each new sequence to get per-sequence statistics.

    Fields:
        n_steps:       Total number of ``step()`` calls (excluding init frame).
        n_held:        Frames where hysteresis held the previous exit idx
                       (transition was requested but not yet triggered).
        n_switched:    Frames where the exit idx actually changed from the
                       previous frame's value.
        n_by_state:    Call count broken down by CSC state string.
        n_by_force_idx: Call count broken down by the *returned* force_layer_idx
                        (including hysteresis-held value, not the requested one).
    """

    n_steps: int = 0
    n_held: int = 0
    n_switched: int = 0
    n_by_state: dict[str, int] = field(default_factory=dict)
    n_by_force_idx: dict[int, int] = field(default_factory=dict)


class StateExitRouter:
    """Map CSC state → SGLATrack force_layer_idx with hysteresis.

    Per-frame contract::

        idx = router.step(state, state_confidence=None)
        tracker.set_force_layer(idx)   # (not a real method; see usage note)
        out = tracker.update(frame)
        # log: state, state_confidence, idx, out.aux["selected_block"]

    Usage note: SGLATrack's ``SGLATracker`` stores force_layer_idx as an
    init-time attribute (``self._force_layer_idx``). The runner must either:

    (a) Call ``tracker._force_layer_idx = idx`` directly before ``update()``.
        This is safe — the attribute is read on every ``update()`` call, not
        cached. Underscore access is acceptable in a research pipeline.

    (b) Use the ``SGLATracker.set_force_layer(idx)`` method if it exists.
        (Not currently present; the runner can add it as a one-liner shim.)

    Hysteresis:
        When the policy requests a *new* exit idx that differs from the current
        one, the router waits ``min_hold_frames`` frames before switching. This
        prevents rapid oscillation between states (e.g. confirmed ↔ occluded at
        an occlusion boundary) from causing per-frame exit changes. Once a
        switch fires, the new idx is held for a minimum of ``min_hold_frames``
        additional frames.

        Default ``min_hold_frames=5`` is chosen to match the typical rise-time
        of CSC state transitions (empirically ~3-7 frames in Task #21 ablation).

    Confidence gating (optional):
        If ``state_confidence`` is provided and is below ``min_state_confidence``,
        the router treats the state as ``"uncertain"`` regardless of the raw
        prediction. This avoids forcing block 9 based on a low-confidence CSC
        prediction (which may itself be wrong).

    Thread safety: NOT thread-safe. One router instance per sequence, per
    tracker, per evaluation process.

    Args:
        min_hold_frames: Minimum frames before an exit-idx change fires.
            Also the minimum hold after a switch completes. Default 5.
        min_state_confidence: Minimum CSC state confidence to trust a risky
            state prediction. Below this, the state is downgraded to
            "uncertain" (which maps to DEFAULT_EXIT_IDX = -1). Default 0.0
            (disabled — trust all predictions).
        policy: Mapping from CSC state string to force_layer_idx override.
            Defaults to ``STATE_TO_EXIT`` (Policy V1). Pass a custom dict to
            experiment with alternative policies (e.g., block 11 on lost).
    """

    def __init__(
        self,
        min_hold_frames: int = 5,
        min_state_confidence: float = 0.0,
        policy: Optional[dict[str, int]] = None,
    ) -> None:
        if min_hold_frames < 0:
            raise ValueError(
                f"min_hold_frames must be >= 0, got {min_hold_frames}"
            )
        if not (0.0 <= min_state_confidence <= 1.0):
            raise ValueError(
                f"min_state_confidence must be in [0, 1], got {min_state_confidence}"
            )
        self._min_hold = min_hold_frames
        self._min_conf = min_state_confidence
        self._policy: dict[str, int] = policy if policy is not None else dict(STATE_TO_EXIT)

        # Validate policy values
        lo, hi = _VALID_FORCE_IDX_RANGE
        for state, idx in self._policy.items():
            if not (lo <= idx <= hi):
                raise ValueError(
                    f"Policy entry {state!r} has force_layer_idx={idx}, "
                    f"which is outside valid range [{lo}, {hi}]."
                )

        # Router state
        self._current_idx: int = DEFAULT_EXIT_IDX
        self._pending_idx: Optional[int] = None  # requested but not yet fired
        self._hold_countdown: int = 0             # frames until pending fires
        self._prev_state: Optional[str] = None

        self.stats = RouterStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        state: str,
        state_confidence: Optional[float] = None,
    ) -> int:
        """Compute force_layer_idx for this frame.

        Args:
            state: CSC-predicted state string. Must be one of the policy keys
                or a fallback to DEFAULT_EXIT_IDX will be used.
            state_confidence: Optional float in [0, 1]. If provided and below
                ``min_state_confidence``, the state is downgraded to
                ``"uncertain"`` before the policy lookup.

        Returns:
            force_layer_idx: int in [-1, 5].
                -1 means "let the SGLATrack MLP decide."
                0-5 means "force this MLP output index" (→ block 6-11).
        """
        self.stats.n_steps += 1

        # Confidence gating: downgrade to uncertain if CSC is unsure
        effective_state = state
        if (
            state_confidence is not None
            and self._min_conf > 0.0
            and state_confidence < self._min_conf
            and state in RISKY_STATES
        ):
            effective_state = "uncertain"

        # Policy lookup — unknown states fall back to DEFAULT_EXIT_IDX
        requested_idx = self._policy.get(effective_state, DEFAULT_EXIT_IDX)

        # Hysteresis: only fire a change after min_hold_frames
        switched = False
        if requested_idx != self._current_idx:
            if self._pending_idx != requested_idx:
                # New transition target — start / restart countdown
                self._pending_idx = requested_idx
                self._hold_countdown = self._min_hold

            if self._hold_countdown > 0:
                self._hold_countdown -= 1
                self.stats.n_held += 1
            else:
                # Countdown expired — fire the switch
                self._current_idx = requested_idx
                self._pending_idx = None
                self._hold_countdown = self._min_hold  # hold after switch too
                switched = True
                self.stats.n_switched += 1
        else:
            # No change requested — clear any pending transition
            if self._pending_idx is not None and self._pending_idx != self._current_idx:
                # State oscillated back before switch fired; cancel transition
                self._pending_idx = None
                self._hold_countdown = 0
            if self._hold_countdown > 0:
                self._hold_countdown -= 1

        # Update per-state and per-idx counters
        self.stats.n_by_state[effective_state] = (
            self.stats.n_by_state.get(effective_state, 0) + 1
        )
        self.stats.n_by_force_idx[self._current_idx] = (
            self.stats.n_by_force_idx.get(self._current_idx, 0) + 1
        )

        self._prev_state = effective_state
        return self._current_idx

    def reset(self) -> None:
        """Reset router state for a new sequence.

        Resets the current exit idx to DEFAULT_EXIT_IDX, clears hysteresis
        state, and resets all statistics counters.
        """
        self._current_idx = DEFAULT_EXIT_IDX
        self._pending_idx = None
        self._hold_countdown = 0
        self._prev_state = None
        self.stats = RouterStats()

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    @property
    def current_idx(self) -> int:
        """The force_layer_idx that will be returned on the *next* step()
        call, assuming the state does not change and hysteresis is satisfied."""
        return self._current_idx

    @property
    def hold_countdown(self) -> int:
        """Remaining frames before a pending transition fires.

        0 means either no transition is pending or the next step() call will
        fire the switch.
        """
        return self._hold_countdown

    @property
    def is_in_risky_mode(self) -> bool:
        """True iff the current exit idx is a non-default forced value
        (i.e., CSC is actively overriding the MLP for a risky state)."""
        return self._current_idx != DEFAULT_EXIT_IDX

    def policy_summary(self) -> dict[str, int]:
        """Return a copy of the active policy mapping."""
        return dict(self._policy)

    def stats_dict(self) -> dict:
        """Return stats as a plain dict for JSONL telemetry logging."""
        return {
            "n_steps": self.stats.n_steps,
            "n_held": self.stats.n_held,
            "n_switched": self.stats.n_switched,
            "n_by_state": dict(self.stats.n_by_state),
            "n_by_force_idx": dict(self.stats.n_by_force_idx),
            "switch_rate": (
                self.stats.n_switched / self.stats.n_steps
                if self.stats.n_steps > 0
                else 0.0
            ),
            "hold_rate": (
                self.stats.n_held / self.stats.n_steps
                if self.stats.n_steps > 0
                else 0.0
            ),
        }

    def __repr__(self) -> str:
        return (
            f"StateExitRouter("
            f"current_idx={self._current_idx}, "
            f"min_hold={self._min_hold}, "
            f"min_conf={self._min_conf}, "
            f"pending={self._pending_idx}, "
            f"countdown={self._hold_countdown}, "
            f"n_steps={self.stats.n_steps}"
            f")"
        )
