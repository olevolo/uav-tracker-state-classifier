"""Scene-state classification (CSC core).

`SceneStateClassifier` from prior SALT-RD work serves as an architectural
template (multi-head GRU on a 28-dim telemetry window). The CSC project will
adapt it to the 6-class state taxonomy: confirmed / uncertain / occluded /
lost / distractor / false_confirmed.
"""

from csc_uav_tracking.scene_state.classifier import (
    SceneState,
    SceneStateClassifier,
)

__all__ = ["SceneState", "SceneStateClassifier"]
