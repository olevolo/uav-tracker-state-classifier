"""CSC UAV Tracking — ClassStateClassifier for UAV visual object tracking.

State-aware diagnostic and optional control layer for visual trackers.
Predicts tracking state (confirmed / uncertain / occluded / lost / distractor /
false_confirmed) from tracker telemetry.

Reused from prior SALT-RD work:
    - dataset loaders (UAV123, DTB70, VisDrone-SOT, LaSOT, GOT-10k)
    - telemetry features (motion entropy, tracker confidence, optical flow)
    - feature schema (28-dim telemetry window)
    - SceneStateClassifier (GRU multi-head architecture template)
    - tracking metrics (Success AUC, Precision@20)
"""

from csc_uav_tracking.registry import (
    DATASETS,
    DETECTORS,
    SCENE_CLASSIFIERS,
    SCHEDULERS,
    SIGNALS,
    TRACKERS,
)
from csc_uav_tracking.types import BBox

# Trigger plugin registration for the data layer.
from csc_uav_tracking import datasets as _datasets  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "BBox",
    "DATASETS",
    "DETECTORS",
    "SCENE_CLASSIFIERS",
    "SCHEDULERS",
    "SIGNALS",
    "TRACKERS",
]
