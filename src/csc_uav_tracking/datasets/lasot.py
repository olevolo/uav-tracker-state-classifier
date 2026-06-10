"""LaSOT single-object tracking dataset loader.

Paper: Fan et al., "LaSOT: A High-quality Benchmark for Large-scale Single
Object Tracking", CVPR 2019.
https://github.com/HengLan/LaSOT_Evaluation_Toolkit

Expected on-disk layout::

    LaSOT/
    ├── car/
    │   ├── car-1/
    │   │   ├── img/
    │   │   │   ├── 00000001.jpg
    │   │   │   └── ...
    │   │   ├── groundtruth.txt      ← x,y,w,h per line (comma-sep)
    │   │   ├── full_occlusion.txt   ← single line of comma-sep 0/1 flags
    │   │   └── out_of_view.txt      ← single line of comma-sep 0/1 flags
    │   └── car-2/
    │       └── ...
    ├── person/
    └── ...

Root auto-detection order:
    1. ``$LASOT_DATA_ROOT`` env var
    2. ``$UAV_DATA_ROOT/LaSOT/``
    3. ``~/uav-tracker-data/LaSOT/``

Registered as ``"lasot"`` in DATASETS.
"""

from __future__ import annotations

import configparser
import logging
import os
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


class _LaSOTSequence:
    """Single LaSOT tracking sequence — lazy frame loading."""

    def __init__(
        self,
        name: str,
        category: str,
        frame_paths: list[Path],
        ground_truth: list[_BBoxAnnotated],
        full_occlusion: np.ndarray,
        out_of_view: np.ndarray,
    ) -> None:
        self.name = name
        self.category = category
        self._frame_paths = frame_paths
        self.ground_truth: list[BBox] = ground_truth  # type: ignore[assignment]
        self.full_occlusion = full_occlusion
        self.out_of_view = out_of_view
        self.attributes: set[str] = {category}

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
                        f"LaSOT: frame not found at {p} — substituting grey placeholder",
                    )
                    img = np.zeros((540, 960, 3), dtype=np.uint8)
                yield img


def _parse_gt(path: Path) -> list[_BBoxAnnotated]:
    bboxes: list[_BBoxAnnotated] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                bboxes.append(_BBoxAnnotated(0.0, 0.0, 0.0, 0.0, valid=False))
                continue
            try:
                vals = [float(p) for p in parts[:4]]
                valid = vals[2] > 0 and vals[3] > 0
                bboxes.append(_BBoxAnnotated(*vals, valid=valid))
            except ValueError:
                bboxes.append(_BBoxAnnotated(0.0, 0.0, 0.0, 0.0, valid=False))
    return bboxes


def _parse_flags_line(path: Path, n_frames: int) -> np.ndarray:
    """Parse a single-line comma-separated 0/1 flag file."""
    if not path.exists():
        return np.zeros(n_frames, dtype=np.uint8)
    with open(path) as fh:
        content = fh.read().strip()
    parts = [p.strip() for p in content.split(",") if p.strip()]
    flags = np.zeros(n_frames, dtype=np.uint8)
    for i, p in enumerate(parts[:n_frames]):
        try:
            flags[i] = int(p)
        except ValueError:
            pass
    return flags


def _resolve_root(root: Path) -> Path:
    for child in root.iterdir():
        if child.is_dir() and (child / child.name + "-1").exists():
            return root
    return root


@DATASETS.register("lasot")
class LaSOTDataset:
    """Lazily-iterable LaSOT dataset (selected categories).

    Parameters
    ----------
    root:
        Dataset root containing category folders. See module docstring.
    categories:
        Subset of categories to load. ``None`` loads all present categories.
    max_frames:
        Cap on frames per sequence (for fast iteration tests).
    """

    name: str = "lasot"

    def __init__(
        self,
        root: Path | str | None = None,
        categories: list[str] | None = None,
        max_frames: int | None = None,
    ) -> None:
        if root is None:
            env = os.environ.get("LASOT_DATA_ROOT", "").strip()
            if env:
                root = Path(env).expanduser().resolve()
            else:
                uav_data = os.environ.get("UAV_DATA_ROOT", "").strip()
                base = Path(uav_data).expanduser().resolve() if uav_data else Path.home() / "uav-tracker-data"
                root = base / "LaSOT"
        self.root = Path(root)
        self.categories = categories
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[_LaSOTSequence]:
        if not self.root.exists():
            raise FileNotFoundError(
                f"LaSOT root not found at {self.root}.\n"
                f"Set LASOT_DATA_ROOT or place the dataset at ~/uav-tracker-data/LaSOT/."
            )

        cat_dirs = sorted(
            d for d in self.root.iterdir()
            if d.is_dir() and (self.categories is None or d.name in self.categories)
        )
        if not cat_dirs:
            raise FileNotFoundError(
                f"No category directories found in {self.root}. "
                f"Expected subdirs like car/, person/, etc."
            )

        for cat_dir in cat_dirs:
            for seq_dir in sorted(cat_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                img_dir = seq_dir / "img"
                gt_file = seq_dir / "groundtruth.txt"
                if not img_dir.is_dir() or not gt_file.exists():
                    _log.debug("LaSOT: skipping %s — missing img/ or groundtruth.txt", seq_dir.name)
                    continue

                frame_paths = sorted(img_dir.glob("*.jpg"))
                if not frame_paths:
                    _log.debug("LaSOT: skipping %s — no jpg frames", seq_dir.name)
                    continue
                if self.max_frames is not None:
                    frame_paths = frame_paths[: self.max_frames]

                gt = _parse_gt(gt_file)
                if self.max_frames is not None:
                    gt = gt[: self.max_frames]
                if not gt:
                    _log.debug("LaSOT: skipping %s — empty groundtruth", seq_dir.name)
                    continue

                n = len(gt)
                full_occ = _parse_flags_line(seq_dir / "full_occlusion.txt", n)
                out_of_view = _parse_flags_line(seq_dir / "out_of_view.txt", n)

                yield _LaSOTSequence(
                    name=seq_dir.name,
                    category=cat_dir.name,
                    frame_paths=frame_paths,
                    ground_truth=gt,
                    full_occlusion=full_occ,
                    out_of_view=out_of_view,
                )

    def filter(self, attributes: set[str]) -> "LaSOTDataset":
        cats = [c for c in (self.categories or self._all_categories()) if c in attributes]
        return LaSOTDataset(root=self.root, categories=cats or None, max_frames=self.max_frames)

    def _all_categories(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(d.name for d in self.root.iterdir() if d.is_dir())
