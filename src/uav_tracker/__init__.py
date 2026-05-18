"""ML-detection-focused UAV tracker package.

ML-driven scene classification and detection-based routing for UAV tracking.
Combines lightweight KCF/Kalman tracking with transformer-based heavy trackers
(OSTrack, STARK) and a YOLO detector, routing between tiers via an online
scene classifier rather than hand-crafted entropy signals.

This top-level module re-exports the core datatypes and plugin registries
so downstream code can ``from uav_tracker import BBox, TRACKERS`` without
caring about sub-package layout.

Importing this module also triggers *plugin registration*: each plugin
module's import has a side effect that decorates a class with a registry
(``@TRACKERS.register(...)`` etc.). The ``_register_plugins`` helper below
imports those modules lazily, so a user who only wants the types never
pays the cost of pulling in OpenCV/Torch/etc.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

__version__ = "0.1.0"

# Re-export core datatypes from the Architect-owned ``types`` module.
# These are the lingua franca of the plugin boundary (PLAN §4.2).
from .types import (  # noqa: F401
    BBox,
    TrackState,
    Detection,
    SignalReport,
    SchedulerDecision,
    FrameContext,
    # Phase 10 — ML extension types
    SceneClass,
    SceneClassification,
    DifficultyPrediction,
    AppearanceTemplate,
    FrameContextV2,
)

# Re-export plugin registries from the Architect-owned ``registry`` module.
# Plugins call ``@TRACKERS.register("name")`` etc. at import time; the
# runner/evaluator looks plugins up by string name via ``TRACKERS.build``.
from .registry import (  # noqa: F401
    TRACKERS,
    DETECTORS,
    SIGNALS,
    SCHEDULERS,
    DATASETS,
    # Phase 10 — ML extension registries
    SCENE_CLASSIFIERS,
    DIFFICULTY_PREDICTORS,
    APPEARANCE_MEMORIES,
    MOTION_PREDICTORS,
    ML_WARMERS,
)

_log = logging.getLogger(__name__)

# Submodule paths whose import side-effect registers a plugin.
# Each entry is documented with WHY it exists (what it registers).
_PLUGIN_MODULES: tuple[str, ...] = (
    # Registers "kcf_kalman" in TRACKERS (fast KCF+Kalman tier-1 tracker).
    "uav_tracker.trackers.kcf_kalman",
    # Registers "kcf_henriques" in TRACKERS (Henriques 2015 pure-NumPy KCF port).
    "uav_tracker.trackers.kcf_henriques",
    # Registers "uav123" in the (optional) dataset registry.
    "uav_tracker.datasets.uav123",
    # Registers "synthetic" in DATASETS (procedural eval dataset).
    "uav_tracker.datasets.synthetic",
    # Fallback signals (used by multi_tier scheduler for confidence checks).
    "uav_tracker.signals.motion_entropy",         # registers "motion_entropy"
    "uav_tracker.signals.tracker_confidence",     # registers "tracker_confidence"
    # ML-scene-driven scheduler + detector.
    "uav_tracker.schedulers.multi_tier",          # registers "multi_tier" in SCHEDULERS
    "uav_tracker.detectors.yolo",                 # registers "yolov8n" in DETECTORS
    # Heavy transformer trackers (tier-2).
    "uav_tracker.trackers.transformer.ostrack",   # registers "ostrack_256" in TRACKERS
    "uav_tracker.trackers.transformer.stark",     # registers "stark_s50" in TRACKERS
    # ML infrastructure.
    "uav_tracker.ml.warmer.model_warmer",         # registers "default" in ML_WARMERS
    "uav_tracker.schedulers.ml_scene_scheduler",  # registers "ml_scene" in SCHEDULERS
    "uav_tracker.ml.scene_classifier.cnn_classifier",  # registers "mobilenetv3_tiny" in SCENE_CLASSIFIERS
    "uav_tracker.ml.difficulty_predictor.regression_predictor",  # registers "mlp_regressor" in DIFFICULTY_PREDICTORS
    # Self-learning modules.
    "uav_tracker.ml.appearance_memory.cosine_memory",  # registers "cosine_memory" in APPEARANCE_MEMORIES
    "uav_tracker.ml.motion_predictor.lstm_predictor",  # registers "lstm_online" in MOTION_PREDICTORS
)


def _register_plugins() -> None:
    """Import every known plugin module so decorators execute.

    Failures are logged at DEBUG and swallowed — this keeps the package
    importable on minimal installs where e.g. OpenCV or Torch is missing.
    Missing deps should surface via ``uav-tracker doctor`` and/or when the
    user actually tries to construct an unavailable plugin.
    """
    for mod_path in _PLUGIN_MODULES:
        try:
            importlib.import_module(mod_path)
        except Exception as exc:  # pragma: no cover - defensive
            _log.debug("Skipping plugin module %s: %s", mod_path, exc)


# Fire registration at import time. Safe because each submodule is
# idempotent / guarded against double-registration via the registry itself.
_register_plugins()


if TYPE_CHECKING:  # pragma: no cover - type-checker only
    from . import cli as cli  # noqa: F401
