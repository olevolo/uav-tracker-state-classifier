"""Scheduler Protocol.

Architect-owned. All schedulers (HysteresisBinary, CUSUM, MultiTier,
AdaptiveThreshold, TrajectoryAware, LearnedPolicy, ...) conform to this
Protocol. See ADR-0005 (baseline) and ADR-0008 (N-tier semantics).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import SchedulerDecision, SignalReport


@runtime_checkable
class Scheduler(Protocol):
    """Policy mapping (signals, current tier) -> tier index.

    Attributes
    ----------
    name : str
        Registry key.
    tiers : int
        Number of tiers this scheduler manages. The runner uses this to
        validate that the configured tier slots match what the scheduler
        expects (binary hysteresis = 2; 3-tier with detector = 3; etc.).

    Methods
    -------
    decide(signals, current_tier, frame_idx)
        Return the tier index to use this frame plus a human-readable reason
        and a ``switched`` flag indicating whether the tier changed vs. the
        previous decision. Schedulers must honor the reliability contract:
        when any input ``SignalReport.reliable`` is ``False``, the scheduler
        holds state (no transition) and the frame does not count toward
        confirm/cooldown windows (ADR-0005, ADR-0006).
    reset()
        Restore scheduler state (tier, confirm/cooldown counters) to
        construction. Must be idempotent.
    """

    name: str
    tiers: int

    def decide(
        self,
        signals: dict[str, SignalReport],
        current_tier: int,
        frame_idx: int,
    ) -> SchedulerDecision: ...

    def reset(self) -> None: ...


__all__ = ["Scheduler"]
