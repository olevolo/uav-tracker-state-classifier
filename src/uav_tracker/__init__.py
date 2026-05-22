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
    # -------------------------------------------------------------------------
    # ACTIVE — used by SALT v3 pipeline
    # -------------------------------------------------------------------------

    # Primary tracker
    "uav_tracker.trackers.sglatrack",              # registers "sglatrack"

    # Comparison baselines (fast_bench --mode sglatrack/kcf, paper tables)
    "uav_tracker.trackers.kcf_henriques",          # registers "kcf_henriques"
    "uav_tracker.trackers.transformer.ostrack",    # registers "ostrack_256"
    "uav_tracker.trackers.transformer.ortrack",    # registers "ortrack_deit"

    # Datasets
    "uav_tracker.datasets.uav123",
    "uav_tracker.datasets.synthetic",
    "uav_tracker.datasets.dtb70",
    "uav_tracker.datasets.visdrone_sot",

    # Recovery detectors (YOLO26m = active; RT-DETR/LEAF = ablation)
    "uav_tracker.detectors.visdrone_yolo26m",      # registers "yolo26m_visdrone" — active
    "uav_tracker.detectors.rtdetr",                # registers "rtdetrv2_s" — ablation
    "uav_tracker.detectors.leaf_yolo",             # registers "leaf_yolo", "leaf_yolo_n" — ablation

    # SALT self-learning modules
    "uav_tracker.ml.appearance_memory.cosine_memory",    # registers "cosine_memory" — active
    "uav_tracker.ml.motion_predictor.lstm_predictor",    # registers "lstm_online" — disabled in salt.yaml, code kept

    # -------------------------------------------------------------------------
    # INACTIVE — V2 pipeline (archived configs/archive/v2_full_ml.yaml).
    # Registrations kept so `DETECTORS.build("yolov8n")` etc. still work
    # if someone loads an old config, but these modules are NOT called by
    # any active pipeline or experiment config.
    # -------------------------------------------------------------------------

    # "uav_tracker.detectors.yolo",                  # registers "yolov8n" — replaced by yolo26m_visdrone
    # "uav_tracker.schedulers.ml_scene_scheduler",   # registers "ml_scene" — V2 scene routing
    # "uav_tracker.schedulers.multi_tier",           # registers "multi_tier" — V2 tier fallback
    # "uav_tracker.signals.motion_entropy",          # registers "motion_entropy" — V2 signal
    # "uav_tracker.signals.tracker_confidence",      # registers "tracker_confidence" — V2 signal
    # "uav_tracker.ml.scene_classifier.cnn_classifier",  # registers "mobilenetv3_tiny" — V2 CNN classifier
    # "uav_tracker.ml.warmer.model_warmer",          # registers "default" — V2 model warmer
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
