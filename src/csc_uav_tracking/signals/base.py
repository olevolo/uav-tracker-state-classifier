"""SwitchSignal Protocol.

Architect-owned. All switching signals (MotionEntropy, CircularResultant,
APCE, TrackerConfidence, FlowDivergence, JointMotionEntropy, LearnedGate, ...)
conform to this Protocol. See ADR-0005 for the decision record and the
reliability contract that signals and schedulers share.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import FrameContext, SignalReport, TrackState


@runtime_checkable
class SwitchSignal(Protocol):
    """Per-frame scalar signal consumed by schedulers.

    Attributes
    ----------
    name : str
        Registry key.
    range : tuple[float, float]
        Expected ``(min, max)`` of the scalar ``value`` emitted by ``step``.
        Schedulers may use this for normalization or for per-signal threshold
        rescaling. Signals must not emit values outside this range during
        ``reliable=True`` frames; the contract test enforces this.

    Methods
    -------
    step(ctx, state)
        Consume the current ``FrameContext`` (with cached flow / global-motion
        etc.) and the tracker's latest ``TrackState`` (``None`` at
        ``frame_idx=0`` before ``Tracker.init`` has run), return a
        ``SignalReport``. When the signal's underlying computation is
        untrustworthy (e.g., low-texture frame for motion entropy, or
        ``state is None`` for confidence-based signals), the
        ``SignalReport.reliable`` flag must be set to ``False`` so schedulers
        hold state (ADR-0005, ADR-0006).
    reset()
        Restore the signal to its constructed state (used by the runner at
        sequence boundaries). Must be idempotent.
    """

    name: str
    range: tuple[float, float]

    def step(self, ctx: FrameContext, state: TrackState | None) -> SignalReport: ...

    def reset(self) -> None: ...


__all__ = ["SwitchSignal"]
