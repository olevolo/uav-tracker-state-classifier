"""Tracker-agnostic telemetry adapter.

Stage-1 CSC must be able to plug into any visual tracker (SGLATrack,
ORTrack, AVTrack, EVPTrack, OSTrack, TCTrack, KCF, ...) without
hand-coded per-tracker integration in CSC code itself.  Each tracker
gets a thin :class:`TrackerTelemetryAdapter` that maps its native
output to the shared :class:`TelemetryFrame` schema.

If a tracker does not expose confidence, the adapter must return
``confidence = None`` — never fabricate a value.  Downstream code
treats missing confidence as "false-confirmation analysis unavailable
for this tracker" rather than silently using a default.

Per-tracker confidence calibration
----------------------------------

Native tracker scores are not on the same scale.  Use the
:class:`PercentileConfidenceCalibrator` to map raw scores to a
[0, 1] range *per tracker* using a held-out calibration split (never
UAV123).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class TelemetryFrame:
    """One frame of standardised tracker telemetry."""

    pred_bbox: Optional[tuple[float, float, float, float]] = None  # xywh

    # Confidence + provenance
    confidence: Optional[float] = None      # raw native score, NOT calibrated
    confidence_source: Optional[str] = None  # e.g., "softmax_top3", "score_max"

    # Sharp-peak indicators (optional)
    response_map_peak: Optional[float] = None
    apce: Optional[float] = None
    psr: Optional[float] = None
    response_entropy: Optional[float] = None

    # Token / layer telemetry (transformer trackers only)
    token_keep_ratio: Optional[float] = None
    active_layers: Optional[int] = None

    # Timing
    latency_ms: Optional[float] = None

    # Free-form bag for tracker-specific fields
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "pred_bbox": list(self.pred_bbox) if self.pred_bbox is not None else None,
            "confidence": self.confidence,
            "confidence_source": self.confidence_source,
            "response_map_peak": self.response_map_peak,
            "apce": self.apce,
            "psr": self.psr,
            "response_entropy": self.response_entropy,
            "token_keep_ratio": self.token_keep_ratio,
            "active_layers": self.active_layers,
            "latency_ms": self.latency_ms,
        }
        d.update(self.extra)
        return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Adapter ABC
# ---------------------------------------------------------------------------


class TrackerTelemetryAdapter:
    """Subclass to wire a specific tracker's output into TelemetryFrame.

    Subclasses implement :py:meth:`from_state` to convert one tracker
    update result (whatever shape) into a :class:`TelemetryFrame`.
    """

    name: str = "abstract"
    confidence_source: Optional[str] = None  # e.g., "softmax_top3"

    def from_state(self, state: Any, *, latency_ms: Optional[float] = None) -> TelemetryFrame:
        raise NotImplementedError


class GenericAttrAdapter(TrackerTelemetryAdapter):
    """Adapter for trackers that expose direct attributes on the
    update return value (``confidence``, ``apce``, ``psr``, ...).

    Used for SGLATrack-DeiT (which returns ``TrackState`` with these
    attributes) and any tracker that follows the same convention.
    """

    name = "generic_attr"
    confidence_source = "tracker.confidence"

    def __init__(self, name: Optional[str] = None) -> None:
        if name:
            self.name = name

    def from_state(self, state: Any, *, latency_ms: Optional[float] = None) -> TelemetryFrame:
        if state is None:
            return TelemetryFrame(latency_ms=latency_ms)
        bbox_obj = getattr(state, "bbox", None)
        bbox: Optional[tuple[float, float, float, float]] = None
        if bbox_obj is not None:
            bbox = (
                float(bbox_obj.x),
                float(bbox_obj.y),
                float(bbox_obj.w),
                float(bbox_obj.h),
            )
        out = TelemetryFrame(
            pred_bbox=bbox,
            confidence=_attr_or_none(state, "confidence"),
            confidence_source=self.confidence_source,
            response_map_peak=_attr_or_none(state, "response_max"),
            apce=_attr_or_none(state, "apce"),
            psr=_attr_or_none(state, "psr"),
            response_entropy=_attr_or_none(state, "response_entropy"),
            token_keep_ratio=_attr_or_none(state, "token_keep_ratio"),
            active_layers=_attr_or_none(state, "active_layers", to_int=True),
            latency_ms=latency_ms,
        )
        return out


def _attr_or_none(state: Any, name: str, *, to_int: bool = False) -> Any:
    v = getattr(state, name, None)
    if v is None:
        return None
    try:
        if to_int:
            return int(v)
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-tracker percentile-based confidence calibration
# ---------------------------------------------------------------------------


@dataclass
class PercentileConfidenceCalibrator:
    """Maps native tracker scores to a [0, 1] range using percentiles
    measured on a calibration split.

    Recipe:
        cal = PercentileConfidenceCalibrator()
        cal.fit(raw_scores_on_calibration_split)
        x_norm = cal.transform(raw_score)

    The transform is a simple linear stretch between ``low_pct`` and
    ``high_pct`` percentiles.  Values below low → 0, above high → 1.

    The high threshold for FALSE_CONFIRMED labeling is then ``1.0`` in
    the calibrated domain or ``percentile(scores, 75)`` in raw domain.
    Either is exposed here for convenience.
    """

    low_pct: float = 5.0
    high_pct: float = 95.0
    high_confidence_pct: float = 75.0  # raw percentile used as FC threshold

    _low_value: float = 0.0
    _high_value: float = 1.0
    _high_confidence_value: float = 0.5
    _fitted: bool = False

    def fit(self, raw_scores: np.ndarray) -> "PercentileConfidenceCalibrator":
        arr = np.asarray(raw_scores, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise ValueError("no finite scores to fit calibrator on")
        self._low_value = float(np.percentile(arr, self.low_pct))
        self._high_value = float(np.percentile(arr, self.high_pct))
        if self._high_value <= self._low_value:
            self._high_value = self._low_value + 1e-6
        self._high_confidence_value = float(np.percentile(arr, self.high_confidence_pct))
        self._fitted = True
        return self

    def transform(self, raw_score: float) -> float:
        if not self._fitted:
            raise RuntimeError("calibrator not fitted")
        if raw_score is None or not np.isfinite(raw_score):
            return 0.0
        x = (float(raw_score) - self._low_value) / (self._high_value - self._low_value)
        return float(min(1.0, max(0.0, x)))

    @property
    def high_confidence_threshold(self) -> float:
        """Threshold in **raw** score units to use for FALSE_CONFIRMED."""
        return self._high_confidence_value

    def to_dict(self) -> dict:
        return {
            "low_pct": self.low_pct,
            "high_pct": self.high_pct,
            "high_confidence_pct": self.high_confidence_pct,
            "low_value": self._low_value,
            "high_value": self._high_value,
            "high_confidence_value": self._high_confidence_value,
            "fitted": self._fitted,
        }
