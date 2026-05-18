"""Tracker Protocol.

Architect-owned. All tracker backends (KCF+Kalman, SiamFC, MobileTrack, TransT,
...) conform to this Protocol. See ADR-0003 for the decision record and
PLAN.md §4.2 for the original spec.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..types import BBox, FrameContext, TrackState


@runtime_checkable
class Tracker(Protocol):
    """Single-object tracker Protocol.

    Attributes
    ----------
    name : str
        Registry key used to construct this tracker; also the identifier used
        in per-frame telemetry and report CSVs.
    tier_hint : int
        Advisory tier index (0 = lightest). The ``HybridRunner``'s ``Scheduler``
        is authoritative; this hint is used for diagnostics and default wiring
        when a scheduler is absent (ADR-0003).

    Methods
    -------
    init(frame, bbox)
        Initialize the tracker on the first frame with the ground-truth bbox.
    update(frame)
        Run one tracking step on ``frame`` and return a ``TrackState``.
    flops_per_update()
        Return a static FLOPs estimate (thop / fvcore) for reporting.
    on_tier_enter(ctx) / on_tier_exit(ctx)
        Optional hooks invoked by the runner on tier transitions (no-op
        defaults). Typical uses: refresh Siamese template on re-entry,
        re-center KCF on exit, etc.
    """

    name: str
    tier_hint: int

    def init(self, frame: np.ndarray, bbox: BBox) -> None: ...

    def update(self, frame: np.ndarray) -> TrackState: ...

    def flops_per_update(self) -> float: ...

    def on_tier_enter(self, ctx: FrameContext) -> None:
        """No-op by default; override for appearance refresh, warm-up, etc."""

    def on_tier_exit(self, ctx: FrameContext) -> None:
        """No-op by default; override for state snapshot, re-centering, etc."""


__all__ = ["Tracker"]
