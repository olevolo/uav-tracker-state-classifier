"""OTB100 dataset loader stub (Phase 1).

Paper / PLAN §3.6: Wu 2015 OTB100, 100 generic sequences, OPE protocol.
Secondary benchmark in our phase plan (UAV123 is primary).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Any

from uav_tracker.registry import DATASETS


@DATASETS.register("otb100")
class OTB100Dataset:
    """Lazily-iterable OTB100 dataset conforming to the Dataset Protocol.

    Parameters
    ----------
    root:
        Dataset root, e.g. ``$UAV_DATA_ROOT/otb100``.
    split:
        ``"test"`` covers all 100 sequences.
    attributes:
        Optional attribute filter (IV, DEF, MB, FM, ...). See Wu 2015.
    """

    name: str = "otb100"

    def __init__(
        self,
        root: Path,
        split: str = "test",
        attributes: set[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.attributes = attributes

    def __iter__(self) -> Iterator[Any]:
        """Yield one ``Sequence`` per OTB100 clip."""
        raise NotImplementedError("Phase 1: OTB100 loader")
