"""Protocol definition for model warmers."""

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ModelWarmer(Protocol):
    """Protocol for all model-warmer backends.

    Implementations are registered in ``ML_WARMERS`` via
    ``@ML_WARMERS.register("name")``.
    """

    name: str

    def warmup(self, trackers: dict[int, Any]) -> None: ...
    def warmup_single(self, tracker: Any, dummy_frame: np.ndarray) -> float: ...
    def get_status(self) -> dict[str, Any]: ...
