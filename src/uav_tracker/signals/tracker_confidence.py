"""TrackerConfidenceSignal — Phase 3 trivial switching signal.

Maps the active tracker's ``TrackState.confidence`` into a disorder value
so the hysteresis scheduler can promote to tier 1 when confidence drops.

Mapping:  signal_value = 1.0 - confidence
  → low confidence  (e.g. 0.2) → high signal (0.8) → may exceed E_hi
  → high confidence (e.g. 0.8) → low signal  (0.2) → stays below E_lo

Registration key: ``"tracker_confidence"``
Range: [0.0, 1.0]
"""

from __future__ import annotations

import math

from uav_tracker.registry import SIGNALS
from uav_tracker.types import FrameContext, SignalReport, TrackState


@SIGNALS.register("tracker_confidence")
class TrackerConfidenceSignal:
    """Converts tracker confidence to a switching-signal value.

    Parameters
    ----------
    None — stateless signal; no hyper-parameters.
    """

    name: str = "tracker_confidence"
    range: tuple[float, float] = (0.0, 1.0)

    # ------------------------------------------------------------------

    def step(self, ctx: FrameContext, state: TrackState | None) -> SignalReport:
        """Compute signal value from the latest track state.

        Returns
        -------
        SignalReport
            ``value = 1.0 - confidence``, ``reliable=True`` when confidence
            is a valid float in [0, 1].  Returns ``reliable=False`` (value 0.0)
            when ``state`` is ``None``, confidence is ``None``, or NaN.
        """
        if state is None:
            return SignalReport(value=0.0, reliable=False)

        conf = state.confidence
        if conf is None or (isinstance(conf, float) and math.isnan(conf)):
            return SignalReport(value=0.0, reliable=False)

        # Clamp to [0, 1] defensively — tracker bugs should not crash
        # the scheduler.
        conf_clamped = float(max(0.0, min(1.0, conf)))
        return SignalReport(value=1.0 - conf_clamped, reliable=True)

    def reset(self) -> None:
        """Stateless — no-op."""
        return None


__all__ = ["TrackerConfidenceSignal"]
