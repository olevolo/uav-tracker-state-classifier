"""Protocol definition for motion predictors."""

from typing import Protocol, runtime_checkable

from uav_tracker.types import BBox


@runtime_checkable
class MotionPredictor(Protocol):
    """Protocol for all motion-predictor backends.

    Implementations are registered in ``MOTION_PREDICTORS`` via
    ``@MOTION_PREDICTORS.register("name")``.
    """

    name: str
    hidden_size: int
    seq_len: int

    def predict_next(self, history: list[BBox], timestamps: list[int]) -> BBox: ...
    def update(self, actual_bbox: BBox) -> None: ...
    def reset(self) -> None: ...
