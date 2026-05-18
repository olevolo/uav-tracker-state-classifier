"""UAV Entropy-Guided Tracker package.

Implementation of Oleksiuk & Velhosh (2026),
"Entropy-Guided Tracker Switching Method for UAV Real-Time Tracking".

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
    # Registers "kcf_kalman" in TRACKERS (Phase 1 fast tracker).
    "uav_tracker.trackers.kcf_kalman",
    # Registers "siamfc" in TRACKERS (Phase 2 — comparison baseline deep tracker).
    "uav_tracker.trackers.siamese.siamfc",
    # Registers "mobiletrack" in TRACKERS (paper's primary deep tier; architecture
    # currently inherits SiamFC until the paper's MobileTrack reference is pinned —
    # see src/uav_tracker/trackers/siamese/mobiletrack.py docstring).
    "uav_tracker.trackers.siamese.mobiletrack",
    # Registers "siamban" in TRACKERS (Chen et al. CVPR 2020; SiamBAN R50 bridge
    # using weights at $UAV_WEIGHTS_ROOT/mobiletrack/siamban_r50_l234.pth).
    "uav_tracker.trackers.siamese.siamban",
    # Registers "kcf_henriques" in TRACKERS (Henriques 2015 pure-NumPy KCF port).
    "uav_tracker.trackers.kcf_henriques",
    # Registers "uav123" in the (optional) dataset registry (Phase 1 bench).
    "uav_tracker.datasets.uav123",
    # Registers "otb100" in the (optional) dataset registry (Phase 1 bench).
    "uav_tracker.datasets.otb100",
    # Registers "synthetic" in DATASETS (Phase 1 procedural eval dataset).
    "uav_tracker.datasets.synthetic",
    # Future phases append here:
    "uav_tracker.signals.motion_entropy",         # Phase 4
    "uav_tracker.signals.tracker_confidence",     # Phase 3
    "uav_tracker.schedulers.hysteresis_binary",   # Phase 3
    # Phase 5 — new signals
    "uav_tracker.signals.circular_resultant",     # Phase 5
    "uav_tracker.signals.apce",                   # Phase 5
    "uav_tracker.signals.flow_divergence",        # Phase 5
    # Phase 5 — new schedulers
    "uav_tracker.schedulers.cusum",               # Phase 5
    "uav_tracker.schedulers.adaptive_threshold",  # Phase 5
    "uav_tracker.schedulers.trajectory_aware",    # Phase 5
    # Phase 6 — new scheduler + detector
    "uav_tracker.schedulers.multi_tier",          # Phase 6 — registers "multi_tier" in SCHEDULERS
    "uav_tracker.detectors.yolo",               # Phase 6 — registers "yolov8n" in DETECTORS
    # Phase 8 — Henriques 2015 KCF reference port (PLAN §3.2.A fallback).
    "uav_tracker.trackers.kcf_henriques",         # Phase 8 — registers "kcf_henriques" in TRACKERS
    # Phase 10 — heavy transformer trackers.
    "uav_tracker.trackers.transformer.ostrack",   # Phase 10 — registers "ostrack_256" in TRACKERS
    "uav_tracker.trackers.transformer.stark",     # Phase 10 — registers "stark_s50" in TRACKERS
    # Phase 12 — ML infrastructure.
    "uav_tracker.ml.warmer.model_warmer",         # Phase 12 — registers "default" in ML_WARMERS
    "uav_tracker.schedulers.ml_scene_scheduler",  # Phase 12 — registers "ml_scene" in SCHEDULERS
    "uav_tracker.ml.scene_classifier.cnn_classifier",  # Phase 12 — registers "mobilenetv3_tiny" in SCENE_CLASSIFIERS
    "uav_tracker.ml.difficulty_predictor.regression_predictor",  # Phase 12 — registers "mlp_regressor" in DIFFICULTY_PREDICTORS
    # Phase 13 — self-learning modules.
    "uav_tracker.ml.appearance_memory.cosine_memory",  # Phase 13 — registers "cosine_memory" in APPEARANCE_MEMORIES
    "uav_tracker.ml.motion_predictor.lstm_predictor",  # Phase 13 — registers "lstm_online" in MOTION_PREDICTORS
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
