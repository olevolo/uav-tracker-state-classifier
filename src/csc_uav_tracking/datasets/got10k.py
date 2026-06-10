"""GOT-10k single-object tracking dataset loader.

Paper: Huang et al., "GOT-10k: A Large High-Diversity Benchmark for Generic
Object Tracking in the Wild", TPAMI 2021.
http://got-10k.aitestunion.com/

Expected on-disk layout::

    GOT_10k/
    ├── val/
    │   ├── GOT-10k_Val_000017/
    │   │   ├── 00000001.jpg
    │   │   ├── ...
    │   │   ├── groundtruth.txt   ← x,y,w,h per line (comma-sep, float)
    │   │   └── meta_info.ini
    │   └── ...
    └── test/
        ├── GOT-10k_Test_000012/
        │   ├── 00000001.jpg
        │   ├── ...
        │   └── groundtruth.txt   ← init bbox only (1 line)
        └── ...

Root auto-detection order:
    1. ``$GOT10K_DATA_ROOT`` env var
    2. ``$UAV_DATA_ROOT/GOT_10k/``
    3. ``~/uav-tracker-data/GOT_10k/``

Registered as ``"got10k"`` in DATASETS.

Notes
-----
The val split has full per-frame ground-truth and is suitable for CSC
training.  The test split exposes only the init bbox (1-line GT) and
is not suitable for label generation.  The default split is ``"val"``.
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


class _GOT10kSequence:
    """Single GOT-10k tracking sequence — lazy frame loading."""

    def __init__(
        self,
        name: str,
        frame_paths: list[Path],
        ground_truth: list[_BBoxAnnotated],
        attributes: set[str],
    ) -> None:
        self.name = name
        self._frame_paths = frame_paths
        self.ground_truth: list[BBox] = ground_truth  # type: ignore[assignment]
        self.attributes = attributes

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
                        f"GOT-10k: frame not found at {p} — substituting grey placeholder",
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


def _parse_meta(path: Path) -> set[str]:
    """Extract object_class and motion_class as attribute tags."""
    attrs: set[str] = set()
    if not path.exists():
        return attrs
    cfg = configparser.ConfigParser()
    try:
        cfg.read(str(path))
        section = "METAINFO"
        for key in ("object_class", "motion_class", "major_class"):
            if cfg.has_option(section, key):
                val = cfg.get(section, key).strip()
                if val:
                    attrs.add(val)
    except Exception:
        pass
    return attrs


@DATASETS.register("got10k")
class GOT10kDataset:
    """Lazily-iterable GOT-10k dataset.

    Parameters
    ----------
    root:
        Dataset root containing ``val/`` and/or ``test/`` subdirectories.
    split:
        ``"val"`` (default, full GT) or ``"test"`` (init bbox only).
    max_frames:
        Cap on frames per sequence (for fast iteration tests).
    """

    name: str = "got10k"

    def __init__(
        self,
        root: Path | str | None = None,
        split: str = "val",
        max_frames: int | None = None,
    ) -> None:
        if split not in ("val", "test"):
            raise ValueError(f"GOT-10k split must be 'val' or 'test', got {split!r}")
        if root is None:
            env = os.environ.get("GOT10K_DATA_ROOT", "").strip()
            if env:
                root = Path(env).expanduser().resolve()
            else:
                uav_data = os.environ.get("UAV_DATA_ROOT", "").strip()
                base = Path(uav_data).expanduser().resolve() if uav_data else Path.home() / "uav-tracker-data"
                root = base / "GOT_10k"
        self.root = Path(root)
        self.split = split
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[_GOT10kSequence]:
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"GOT-10k {self.split} split not found at {split_dir}.\n"
                f"Set GOT10K_DATA_ROOT or place the dataset at ~/uav-tracker-data/GOT_10k/."
            )

        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            gt_file = seq_dir / "groundtruth.txt"
            if not gt_file.exists():
                _log.debug("GOT-10k: skipping %s — no groundtruth.txt", seq_dir.name)
                continue

            frame_paths = sorted(seq_dir.glob("*.jpg"))
            if not frame_paths:
                _log.debug("GOT-10k: skipping %s — no jpg frames", seq_dir.name)
                continue
            if self.max_frames is not None:
                frame_paths = frame_paths[: self.max_frames]

            gt = _parse_gt(gt_file)
            if self.split == "val" and len(gt) != len(frame_paths):
                # align to shorter; warn if mismatch is large
                n = min(len(gt), len(frame_paths))
                if abs(len(gt) - len(frame_paths)) > 2:
                    _warn_once(
                        f"gt_mismatch:{seq_dir.name}",
                        f"GOT-10k {seq_dir.name}: GT has {len(gt)} lines but {len(frame_paths)} frames — trimming to {n}",
                    )
                gt = gt[:n]
                frame_paths = frame_paths[:n]
            if self.max_frames is not None:
                gt = gt[: self.max_frames]

            if not gt:
                _log.debug("GOT-10k: skipping %s — empty GT after trim", seq_dir.name)
                continue

            attrs = _parse_meta(seq_dir / "meta_info.ini")

            yield _GOT10kSequence(
                name=seq_dir.name,
                frame_paths=frame_paths,
                ground_truth=gt,
                attributes=attrs,
            )

    def filter(self, attributes: set[str]) -> "GOT10kDataset":
        # GOT-10k filtering by attribute requires iterating — return self for now
        return self
