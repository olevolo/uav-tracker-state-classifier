"""Protocol definition for difficulty predictors."""

from typing import Protocol, runtime_checkable

from uav_tracker.types import DifficultyPrediction, FrameContext, TrackState


@runtime_checkable
class DifficultyPredictor(Protocol):
    """Protocol for all difficulty-predictor backends.

    Implementations are registered in ``DIFFICULTY_PREDICTORS`` via
    ``@DIFFICULTY_PREDICTORS.register("name")``.
    """

    name: str
    horizon_frames: int

    def predict(
        self, ctx: FrameContext, history: list[TrackState]
    ) -> DifficultyPrediction: ...
    def reset(self) -> None: ...
