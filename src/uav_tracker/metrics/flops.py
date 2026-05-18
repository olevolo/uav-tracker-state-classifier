"""FLOPs accounting helper (PLAN §11 Phase 1 exit demo).

Aggregates per-frame tier FLOPs + signal FLOPs to the per-sequence
GFLOPs/frame reported in Table 2. Actual implementation lives with the
evaluation pipeline; here we just define the helper signature.
"""

from __future__ import annotations

from typing import Iterable, Any


def flops_per_frame(
    telemetry: Iterable[Any],
    tier_flops: dict[int, float],
    signal_flops_per_frame: float = 0.0,
) -> float:
    """Aggregate GFLOPs/frame from a telemetry stream.

    Parameters
    ----------
    telemetry:
        Iterable of ``TelemetryEntry`` — has ``tier`` per frame.
    tier_flops:
        Tier index → FLOPs per update (static estimate).
    signal_flops_per_frame:
        Sum of signal FLOPs run every frame (not tier-gated).

    Returns
    -------
    float
        GFLOPs/frame averaged over the sequence.
    """
    raise NotImplementedError("Phase 1")
