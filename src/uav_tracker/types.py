"""Shared type definitions for the UAV Entropy-Guided Tracker.

Architect-owned module. All plugins (trackers, detectors, signals, schedulers,
datasets) import their fundamental types from here rather than re-defining them
per-module. Keeping this module small, stable, and dependency-light is a
precondition for the plugin contract (see ADR-0003, ADR-0004, ADR-0005).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal

import numpy as np


# --------------------------------------------------------------------------- #
# Core geometry / tracker output                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in (left, top, width, height) pixel units.

    Frozen so it can be safely used as a dict key or in hashable collections.
    Values are ``float`` to accommodate sub-pixel tracker outputs (KCF peak
    interpolation, Siamese crop-to-image mapping, etc.).
    """

    x: float
    y: float
    w: float
    h: float


@dataclass
class TrackState:
    """Per-frame tracker output.

    ``confidence`` is a best-effort calibrated score in ``[0, 1]``; each tracker
    backend maps its native score (KCF peak value, SiamFC max response, etc.)
    into this range. ``status`` is a coarse trichotomy used by schedulers and
    downstream consumers. ``aux`` is a free-form bag for response maps, raw
    feature activations, or backend-specific diagnostics.
    """

    bbox: BBox
    confidence: float
    status: Literal["locked", "uncertain", "lost"]
    aux: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Detector output                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class Detection:
    """A single detector hypothesis.

    ``class_id`` is optional because open-vocabulary detectors (Grounding DINO,
    OWL-ViT) return only language-conditioned scores.
    """

    bbox: BBox
    score: float
    class_id: int | None = None


# --------------------------------------------------------------------------- #
# Signal / scheduler outputs                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class SignalReport:
    """Output of a ``SwitchSignal.step`` call.

    ``value`` is a scalar that must lie within ``SwitchSignal.range``. ``vector``
    is an optional rich payload (histograms, feature vectors) preserved so a
    future learned scheduler can consume pre-aggregation signal content.
    ``reliable=False`` instructs schedulers to hold state for this frame; see
    ADR-0005 for the reliability contract and ADR-0006 for when it fires.
    """

    value: float
    vector: "np.ndarray | None" = None
    reliable: bool = True
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerDecision:
    """Output of a ``Scheduler.decide`` call.

    ``tier`` is the tier index the runner should use this frame. ``switched``
    is ``True`` iff ``tier`` differs from the scheduler's previous decision,
    allowing the runner to fire ``Tracker.on_tier_enter`` / ``on_tier_exit``
    hooks without re-comparing.  ``aux`` is a free-form bag for
    scheduler-specific metadata (e.g. ``{"prearm_tier3": True}``).
    """

    tier: int
    reason: str
    switched: bool
    aux: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# FrameContext (passive dataclass read by signals / schedulers)               #
# --------------------------------------------------------------------------- #


@dataclass
class FrameContext:
    """Everything the per-frame signal/scheduler pipeline may need.

    Passive dataclass: the ``HybridRunner`` constructs one per frame and hands
    it to each registered component. Plugins read what they need from it; a
    plugin must not assume the presence of optional fields without checking.

    - ``frame`` / ``prev_frame`` are the current and previous raw frames.
    - ``frame_idx`` is the zero-based frame index in the sequence.
    - ``bbox`` is the tracker's current bbox estimate (``None`` at ``frame_idx=0``
      before ``Tracker.init``).
    - ``optical_flow_cache`` memoises per-frame LK / Shi-Tomasi output so signals
      consuming the same flow field don't recompute it.
    - ``global_motion`` is the accepted global-motion estimate for this frame
      (homography matrix, affine matrix, or reused prior). ``None`` if no
      estimate is available yet.
    - ``telemetry`` collects structured per-frame metrics emitted by any stage
      (tracker FPS, scheduler reason codes, signal aux payloads, etc.) for the
      JSONL log.
    """

    frame: np.ndarray
    prev_frame: np.ndarray | None
    frame_idx: int
    bbox: BBox | None
    optical_flow_cache: dict[str, Any] | None = None
    global_motion: "np.ndarray | None" = None
    telemetry: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "BBox",
    "TrackState",
    "Detection",
    "SignalReport",
    "SchedulerDecision",
    "FrameContext",
    # Phase 10 — ML extension types
    "SceneClass",
    "SceneClassification",
    "DifficultyPrediction",
    "AppearanceTemplate",
    "FrameContextV2",
]


# --------------------------------------------------------------------------- #
# Phase 10 — ML extension types                                               #
# --------------------------------------------------------------------------- #


class SceneClass(IntEnum):
    """Coarse scene-difficulty taxonomy used by the scene classifier."""

    CLEAR = 0
    MODERATE = 1
    CHALLENGING = 2
    RISK_LOSS = 3
    RECOVERY = 4
    LOW_RES = 5


@dataclass
class SceneClassification:
    """Output of a ``SceneClassifier.classify`` call."""

    scene_class: SceneClass
    probabilities: np.ndarray
    confidence: float
    frame_idx: int
    aux: dict = field(default_factory=dict)


@dataclass
class DifficultyPrediction:
    """Output of a ``DifficultyPredictor.predict`` call."""

    expected_iou_drop: float
    horizon_frames: int
    feature_vector: np.ndarray
    aux: dict = field(default_factory=dict)


@dataclass
class AppearanceTemplate:
    """A single stored appearance template in an ``AppearanceMemory``."""

    embedding: np.ndarray
    bbox: Any  # BBox — use Any to avoid circular import
    frame_idx: int
    weight: float


@dataclass
class FrameContextV2(FrameContext):
    """``FrameContext`` extended with optional Phase-10 ML predictions.

    Plugins that do not consume ML annotations can still accept this type
    because it is a strict superset of ``FrameContext``.
    """

    scene_classification: SceneClassification | None = None
    difficulty_prediction: DifficultyPrediction | None = None
