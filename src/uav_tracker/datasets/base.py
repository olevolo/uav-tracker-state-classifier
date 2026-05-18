"""Dataset and Sequence Protocols.

Architect-owned. All datasets (UAV123, OTB100, LaSOT, ...) conform to the
``Dataset`` Protocol below, which yields ``Sequence`` Protocol instances. See
ADR-0002 for the decision record.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Protocol, runtime_checkable

import numpy as np

from ..types import BBox


@runtime_checkable
class Sequence(Protocol):
    """A single tracking sequence.

    Attributes
    ----------
    name : str
        Sequence identifier (e.g., ``"group1_1"`` for UAV123, ``"Basketball"``
        for OTB100). Unique within a dataset.
    frames : Iterable[np.ndarray]
        Lazy iterable over frames. Implementations may stream from disk so RAM
        stays flat for long sequences; consumers must not random-access via
        indexing without explicit buffering.
    ground_truth : list[BBox]
        Per-frame ground-truth bboxes. ``len(ground_truth) == len(frames)``
        (though ``frames`` being an iterable hides its length; concrete
        implementations document it).
    init_bbox : BBox
        Shortcut to ``ground_truth[0]`` for OPE initialisation.
    attributes : set[str]
        Short-code attributes using OTB/UAV123 conventions (``"FM"``, ``"OCC"``,
        ``"IV"``, ``"LR"``, etc.). Consumers must tolerate unknown codes.
    """

    name: str
    frames: Iterable[np.ndarray]
    ground_truth: list[BBox]
    init_bbox: BBox
    attributes: set[str]


@runtime_checkable
class Dataset(Protocol):
    """A collection of sequences with attribute-based filtering.

    Attributes
    ----------
    name : str
        Dataset identifier (e.g., ``"uav123"``, ``"otb100"``).

    Methods
    -------
    __iter__()
        Yield ``Sequence`` instances in a deterministic order.
    filter(attributes)
        Return a new ``Dataset`` containing only sequences whose ``attributes``
        set is a superset of the argument. Enables attribute breakdowns
        (Phase 7 reports) without evaluator changes.
    """

    name: str

    def __iter__(self) -> Iterator[Sequence]: ...

    def filter(self, attributes: set[str]) -> "Dataset": ...


__all__ = ["Sequence", "Dataset"]
