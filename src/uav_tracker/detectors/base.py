"""Detector Protocol.

Architect-owned. All detector backends (YOLOv8-n, YOLOv10-n, DFINE, Grounding
DINO, ...) conform to this Protocol. Detection is an opt-in third tier in the
scheduler from Phase 6 onward (ADR-0008).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..types import BBox, Detection


@runtime_checkable
class Detector(Protocol):
    """Full-frame detector Protocol.

    Attributes
    ----------
    name : str
        Registry key used to construct this detector.

    Methods
    -------
    detect(frame, hint=None)
        Run detection on ``frame``. If ``hint`` is provided it is the last known
        bbox; detectors may use it to prioritise a region or to post-filter
        detections by IoU with the hint.
    flops_per_call()
        Static FLOPs estimate per call (thop / fvcore).
    """

    name: str

    def detect(
        self,
        frame: np.ndarray,
        hint: BBox | None = None,
    ) -> list[Detection]: ...

    def flops_per_call(self) -> float: ...


__all__ = ["Detector"]
