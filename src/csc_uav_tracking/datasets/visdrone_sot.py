"""VisDrone2019-SOT single-object tracking dataset loader.

Paper: Zhu et al., "VisDrone-SOT2019: The Vision Meets Drone Single-Object
Tracking Challenge Results", ICCV Workshop 2019.
https://github.com/VisDrone/VisDrone-Dataset

Expected on-disk layout::

    VisDrone2019-SOT-test-dev/
    ├── annotations/
    │   ├── uav0000011_00000_s.txt   ← x,y,w,h per line (comma-separated)
    │   └── ...
    └── sequences/
        ├── uav0000011_00000_s/
        │   ├── img0000001.jpg
        │   └── ...
        └── ...

Root auto-detection order:
    1. ``$VISDRONE_SOT_DATA_ROOT`` env var
    2. ``$UAV_DATA_ROOT/VisDrone-SOT/VisDrone2019-SOT-test-dev/``
    3. ``~/uav-tracker-data/VisDrone-SOT/VisDrone2019-SOT-test-dev/``

Registered as ``"visdrone_sot"`` in DATASETS.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from pathlib import Path
from typing import Generator, Iterable, Iterator

import cv2
import numpy as np

from csc_uav_tracking.registry import DATASETS
from csc_uav_tracking.types import BBox

_log = logging.getLogger(__name__)
_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        warnings.warn(msg, stacklevel=3)
        _warned.add(key)


class _BBoxAnnotated(BBox):
    __slots__ = ("valid",)

    def __new__(cls, x: float, y: float, w: float, h: float, valid: bool = True):
        obj = super().__new__(cls)
        object.__setattr__(obj, "x", x)
        object.__setattr__(obj, "y", y)
        object.__setattr__(obj, "w", w)
        object.__setattr__(obj, "h", h)
        return obj

    def __init__(self, x: float, y: float, w: float, h: float, valid: bool = True) -> None:
        object.__setattr__(self, "valid", valid)


class _VisDroneSOTSequence:
    """Single VisDrone SOT tracking sequence — lazy frame loading."""

    def __init__(
        self,
        name: str,
        frame_paths: list[Path],
        ground_truth: list[_BBoxAnnotated],
    ) -> None:
        self.name = name
        self._frame_paths = frame_paths
        self.ground_truth: list[BBox] = ground_truth  # type: ignore[assignment]

    @property
    def init_bbox(self) -> BBox:
        return self.ground_truth[0]

    @property
    def frames(self) -> Iterable[np.ndarray]:
        return self._FrameIterable(self._frame_paths)

    class _FrameIterable:
        def __init__(self, paths: list[Path]) -> None:
            self._paths = paths

        def __iter__(self) -> Generator[np.ndarray, None, None]:
            for p in self._paths:
                img = cv2.imread(str(p))
                if img is None:
                    _warn_once(
                        f"missing_frame:{p}",
                        f"VisDroneSOT: frame not found at {p} — substituting grey placeholder",
                    )
                    img = np.zeros((540, 960, 3), dtype=np.uint8)
                yield img


def _parse_gt(path: Path) -> list[_BBoxAnnotated]:
    """Parse x,y,w,h annotation file (comma-separated)."""
    bboxes: list[_BBoxAnnotated] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[,\s]+", line)
            parts = [p for p in parts if p]
            if len(parts) < 4:
                bboxes.append(_BBoxAnnotated(0.0, 0.0, 0.0, 0.0, valid=False))
                continue
            try:
                vals = [float(p) for p in parts[:4]]
                # VisDrone uses 0,0,0,0 for occluded/absent frames
                valid = vals[2] > 0 and vals[3] > 0
                bboxes.append(_BBoxAnnotated(*vals, valid=valid))
            except ValueError:
                bboxes.append(_BBoxAnnotated(0.0, 0.0, 0.0, 0.0, valid=False))
    return bboxes


def _resolve_root(root: Path) -> Path:
    """Return the directory containing annotations/ and sequences/."""
    if (root / "sequences").is_dir() and (root / "annotations").is_dir():
        return root
    for split in ("VisDrone2019-SOT-test-dev", "VisDrone2019-SOT-val",
                  "VisDrone2019-SOT-test-challenge"):
        candidate = root / split
        if (candidate / "sequences").is_dir():
            return candidate
    return root


@DATASETS.register("visdrone_sot")
class VisDroneSOTDataset:
    """Lazily-iterable VisDrone2019-SOT dataset.

    Parameters
    ----------
    root:
        Dataset root. See module docstring for accepted layouts.
    max_frames:
        Cap on frames per sequence (for fast iteration tests).
    """

    name: str = "visdrone_sot"

    def __init__(
        self,
        root: Path | str | None = None,
        max_frames: int | None = None,
    ) -> None:
        if root is None:
            env = os.environ.get("VISDRONE_SOT_DATA_ROOT", "").strip()
            if env:
                root = Path(env).expanduser().resolve()
            else:
                uav_data = os.environ.get("UAV_DATA_ROOT", "").strip()
                if uav_data:
                    base = Path(uav_data).expanduser().resolve()
                else:
                    base = Path.home() / "uav-tracker-data"
                root = base / "VisDrone-SOT"
                if not (root / "sequences").is_dir():
                    # Try with split subfolder
                    candidate = root / "VisDrone2019-SOT-test-dev"
                    if candidate.is_dir():
                        root = candidate
        self.root = Path(root)
        self.max_frames = max_frames
        self._seq_root = _resolve_root(self.root)

    def __iter__(self) -> Iterator[_VisDroneSOTSequence]:
        seq_dir = self._seq_root / "sequences"
        ann_dir = self._seq_root / "annotations"

        if not seq_dir.exists():
            raise FileNotFoundError(
                f"VisDroneSOT sequences not found at {seq_dir}.\n"
                f"Set VISDRONE_SOT_DATA_ROOT or place the dataset at {self.root}.\n"
                f"Expected layout: sequences/ and annotations/ under the root."
            )

        for seq_path in sorted(seq_dir.iterdir()):
            if not seq_path.is_dir():
                continue

            frame_paths = sorted(seq_path.glob("img*.jpg"))
            if not frame_paths:
                frame_paths = sorted(seq_path.glob("*.jpg"))
            if not frame_paths:
                _log.debug("VisDroneSOT: skipping %s — no frames", seq_path.name)
                continue

            gt_file = ann_dir / f"{seq_path.name}.txt"
            if not gt_file.exists():
                _log.debug("VisDroneSOT: skipping %s — no annotation", seq_path.name)
                continue

            try:
                bboxes = _parse_gt(gt_file)
            except Exception as exc:
                _log.warning("VisDroneSOT: skipping %s — GT parse error: %s", seq_path.name, exc)
                continue

            if not bboxes:
                _log.warning("VisDroneSOT: skipping %s — empty annotation", seq_path.name)
                continue

            n = min(len(frame_paths), len(bboxes))
            if self.max_frames is not None:
                n = min(n, self.max_frames)

            yield _VisDroneSOTSequence(
                name=seq_path.name,
                frame_paths=frame_paths[:n],
                ground_truth=bboxes[:n],
            )
