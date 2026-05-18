"""APCESignal — Peak-to-Correlation Energy switching signal (Phase 5, now authentic).

APCE (Average Peak-to-Correlation Energy) was introduced by:
    Cao et al. (2025), "Real-time UAV tracking with adaptive switching",
    formula: APCE = (F_max - F_min)^2 / mean((F - F_min)^2)

over a dense correlation response map F.  High APCE means a sharp, clean peak
→ the tracker is confident. Low APCE means a flat or multi-modal response →
the tracker is confused.

Now that KCFHenriques2015Tracker exposes ``TrackState.aux["response_map"]``
(a 2-D float64 correlation map), the authentic formula is computed directly.

Fallback behaviour:
  If ``state.aux["response_map"]`` is absent (e.g. kcf_kalman tier-0 which
  does not expose a full map), the signal falls back to the confidence proxy
  ``1 - confidence`` as before (Option A).

Signal polarity: *high value → high disorder → may escalate to heavy tier*.
  APCE high (confident) → signal close to 0.
  APCE low  (confused)  → signal close to 1.

Normalisation uses the APCE observed at initialisation (first reliable step)
so the scale adapts per-sequence rather than requiring hand-tuning.

Registration key: ``"apce"``
Range: [0.0, 1.0]
"""

from __future__ import annotations

import math

import numpy as np

from uav_tracker.registry import SIGNALS
from uav_tracker.types import FrameContext, SignalReport, TrackState


def _compute_apce(response_map: np.ndarray) -> float:
    """Authentic APCE over a 2-D correlation response map.

    APCE = (F_max - F_min)^2 / (mean((F - F_min)^2) + eps)

    Returns 0.0 for degenerate maps (all-equal values).
    """
    F = response_map.astype(np.float64)
    f_max = float(F.max())
    f_min = float(F.min())
    diff = f_max - f_min
    if diff < 1e-9:
        return 0.0
    denom = float(np.mean((F - f_min) ** 2)) + 1e-9
    return (diff ** 2) / denom


@SIGNALS.register("apce")
class APCESignal:
    """APCE-based switching signal — authentic when response_map available.

    Uses the full 2-D correlation response map from
    ``TrackState.aux["response_map"]`` (set by KCFHenriques2015Tracker).
    Falls back to ``1 - confidence`` proxy when the map is not available.

    Normalisation: signal = 1 - clip(apce / _init_apce, 0, 1), so the
    value is 0.0 when the tracker is as confident as at init, and rises
    toward 1.0 as APCE collapses.
    """

    name: str = "apce"
    range: tuple[float, float] = (0.0, 1.0)

    def __init__(self) -> None:
        self._init_apce: float | None = None

    def step(self, ctx: FrameContext, state: TrackState | None) -> SignalReport:
        if state is None:
            return SignalReport(value=0.0, reliable=False, aux={"apce_mode": "no_state"})

        # Try authentic APCE from response map.
        resp_map = None
        if state.aux:
            resp_map = state.aux.get("response_map")

        if resp_map is not None and isinstance(resp_map, np.ndarray) and resp_map.size > 1:
            apce = _compute_apce(resp_map)
            if self._init_apce is None or self._init_apce < 1e-9:
                self._init_apce = max(apce, 1e-9)
            normalized = float(np.clip(1.0 - apce / self._init_apce, 0.0, 1.0))
            return SignalReport(
                value=normalized,
                reliable=True,
                aux={"apce_mode": "authentic", "apce_raw": apce, "apce_init": self._init_apce},
            )

        # Fallback: confidence proxy (Option A) when no map available.
        conf = state.confidence
        if conf is None or (isinstance(conf, float) and math.isnan(conf)):
            return SignalReport(value=0.0, reliable=False, aux={"apce_mode": "option_a_nan"})
        conf_clamped = float(max(0.0, min(1.0, conf)))
        return SignalReport(
            value=1.0 - conf_clamped,
            reliable=True,
            aux={"apce_mode": "option_a", "confidence": conf_clamped},
        )

    def reset(self) -> None:
        self._init_apce = None


__all__ = ["APCESignal"]
