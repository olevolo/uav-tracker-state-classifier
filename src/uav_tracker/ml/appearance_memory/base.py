"""Protocol definition for appearance memories."""

from typing import Protocol, runtime_checkable

import numpy as np

from uav_tracker.types import AppearanceTemplate, FrameContext, TrackState


@runtime_checkable
class AppearanceMemory(Protocol):
    """Protocol for all appearance-memory backends.

    Implementations are registered in ``APPEARANCE_MEMORIES`` via
    ``@APPEARANCE_MEMORIES.register("name")``.
    """

    name: str
    max_templates: int
    forgetting_factor: float

    def store(self, ctx: FrameContext, state: TrackState) -> None: ...
    def retrieve_best(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> list[AppearanceTemplate]: ...
    def compute_drift(self) -> float: ...
    def reset(self) -> None: ...
