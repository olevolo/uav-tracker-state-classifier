"""Protocol definition for scene classifiers."""

from typing import Protocol, runtime_checkable

import numpy as np

from uav_tracker.types import FrameContext, SceneClass, SceneClassification, TrackState


@runtime_checkable
class SceneClassifier(Protocol):
    """Protocol for all scene-classifier backends.

    Implementations are registered in ``SCENE_CLASSIFIERS`` via
    ``@SCENE_CLASSIFIERS.register("name")``.
    """

    name: str
    n_classes: int
    classify_interval: int

    def classify(self, ctx: FrameContext, state: TrackState) -> SceneClassification: ...
    def update_online(
        self, ctx: FrameContext, state: TrackState, feedback: SceneClass
    ) -> None: ...
    def reset(self) -> None: ...
