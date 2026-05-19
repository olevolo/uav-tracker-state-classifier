"""DTB70 single-object tracking dataset loader.

Paper: Li & Yeung, "Learning to Track Drones: A New Drone Aerial Video
Dataset", ACM MM 2017. https://github.com/flyers/drone-tracking

Expected on-disk layout::

    DTB70/
    ├── Animal1/
    │   ├── img/
    │   │   ├── 00001.jpg
    │   │   ├── 00002.jpg
    │   │   └── ...
    │   └── groundtruth_rect.txt   ← x,y,w,h per line (comma-separated)
    ├── Car1/
    │   └── ...
    └── ...

Root auto-detection order:
    1. ``$DTB70_DATA_ROOT`` env var
    2. ``$UAV_DATA_ROOT/DTB70/``
    3. ``~/uav-tracker-data/DTB70/``

Registered as ``"dtb70"`` in DATASETS.
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

from uav_tracker.registry import DATASETS
from uav_tracker.types import BBox

_log = logging.getLogger(__name__)

_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        warnings.warn(msg, stacklevel=3)
        _warned.add(key)


# ---------------------------------------------------------------------------
# BBox subclass with validity flag (mirrors uav123.py pattern)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Sequence implementation
# ---------------------------------------------------------------------------


class _DTB70Sequence:
    """Single DTB70 tracking sequence — lazy frame loading."""

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
                        f"DTB70: frame not found at {p} — substituting grey placeholder",
                    )
                    img = np.zeros((360, 640, 3), dtype=np.uint8)
                yield img


# ---------------------------------------------------------------------------
# Annotation parser
# ---------------------------------------------------------------------------


def _parse_gt(path: Path) -> list[_BBoxAnnotated]:
    """Parse ``x,y,w,h`` annotation file (comma or space separated).

    DTB70 uses ``groundtruth_rect.txt`` with comma-separated integers.
    Handles both comma and whitespace delimiters for robustness.
    """
    bboxes: list[_BBoxAnnotated] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[,\s]+", line)
            parts = [p for p in parts if p]
            if len(parts) < 4:
                _warn_once(
                    f"short_line:{path}",
                    f"DTB70: short annotation line in {path}: {line!r}",
                )
                bboxes.append(_BBoxAnnotated(0.0, 0.0, 0.0, 0.0, valid=False))
                continue
            try:
                vals = [float(p) for p in parts[:4]]
                bboxes.append(_BBoxAnnotated(*vals, valid=True))
            except ValueError:
                _warn_once(
                    f"bad_value:{path}",
                    f"DTB70: unparseable value in {path}: {line!r}",
                )
                bboxes.append(_BBoxAnnotated(0.0, 0.0, 0.0, 0.0, valid=False))
    return bboxes


# ---------------------------------------------------------------------------
# Root resolution
# ---------------------------------------------------------------------------


def _resolve_root(root: Path) -> Path:
    """Return the sequence root for the given base path.

    Accepts both the outer directory (where each sequence is a subdirectory)
    and a layout where a single ``DTB70/`` subdirectory wraps the sequences.
    """
    # If the given root has sequence-like subdirs with img/ inside, use it.
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir() and (child / "img").is_dir():
            return root
    # Try a nested DTB70/ subdirectory.
    nested = root / "DTB70"
    if nested.is_dir():
        return nested
    return root


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------


@DATASETS.register("dtb70")
class DTB70Dataset:
    """Lazily-iterable DTB70 dataset conforming to the Dataset Protocol.

    Parameters
    ----------
    root:
        Dataset root. Accepted layout is described in the module docstring.
        Auto-detected from ``$DTB70_DATA_ROOT`` → ``$UAV_DATA_ROOT/DTB70/``
        → ``~/uav-tracker-data/DTB70/``.
    max_frames:
        Optional cap on frames per sequence (useful for fast integration
        tests). ``None`` means use all frames.
    """

    name: str = "dtb70"

    def __init__(
        self,
        root: Path | str | None = None,
        max_frames: int | None = None,
    ) -> None:
        if root is None:
            env_dtb = os.environ.get("DTB70_DATA_ROOT", "").strip()
            if env_dtb:
                root = Path(env_dtb).expanduser().resolve()
            else:
                from uav_tracker.paths import data_root
                uav_data = os.environ.get("UAV_DATA_ROOT", "").strip()
                base = Path(uav_data).expanduser().resolve() if uav_data else data_root()
                root = base / "DTB70"
                # Also try ~/uav-tracker-data/DTB70 as a common manual placement
                if not root.exists():
                    alt = Path.home() / "uav-tracker-data" / "DTB70"
                    if alt.exists():
                        root = alt
        self.root = Path(root)
        self.max_frames = max_frames
        self._seq_root = _resolve_root(self.root)
        # Internal flag set by filter() when no sequences can match.
        self._filter_empty: bool = False

    def __iter__(self) -> Iterator[_DTB70Sequence]:
        # Sequences filtered out via filter() with non-empty attribute set.
        if self._filter_empty:
            return

        if not self._seq_root.exists():
            raise FileNotFoundError(
                f"DTB70 sequence directory not found at {self._seq_root}.\n"
                f"Set DTB70_DATA_ROOT or place the dataset at {self.root}.\n"
                f"Download: https://github.com/flyers/drone-tracking"
            )

        for seq_dir in sorted(self._seq_root.iterdir()):
            if not seq_dir.is_dir():
                continue

            # Frames live in <seq>/img/
            img_dir = seq_dir / "img"
            if not img_dir.is_dir():
                _log.debug("DTB70: skipping %s — no img/ subdir", seq_dir.name)
                continue

            frame_paths = sorted(img_dir.glob("*.jpg"))
            if not frame_paths:
                frame_paths = sorted(img_dir.glob("*.png"))
            if not frame_paths:
                _log.debug("DTB70: skipping %s — no frames in img/", seq_dir.name)
                continue

            # GT annotation file.
            gt_file = seq_dir / "groundtruth_rect.txt"
            if not gt_file.exists():
                # Some releases use groundtruth.txt (without _rect).
                gt_file = seq_dir / "groundtruth.txt"
            if not gt_file.exists():
                _log.debug("DTB70: skipping %s — no groundtruth file", seq_dir.name)
                continue

            try:
                bboxes = _parse_gt(gt_file)
            except Exception as exc:
                _log.warning("DTB70: skipping %s — failed to parse GT: %s", seq_dir.name, exc)
                continue

            if not bboxes:
                _log.warning("DTB70: skipping %s — empty annotation", seq_dir.name)
                continue

            if not bboxes[0].valid:
                _log.warning("DTB70: skipping %s — init frame annotation invalid", seq_dir.name)
                continue

            n = min(len(frame_paths), len(bboxes))
            if self.max_frames is not None:
                n = min(n, self.max_frames)

            if n == 0:
                continue

            yield _DTB70Sequence(
                name=seq_dir.name,
                frame_paths=frame_paths[:n],
                ground_truth=bboxes[:n],
                attributes=set(),
            )

    def filter(self, attributes: set[str]) -> "DTB70Dataset":
        """Return a filtered dataset.

        DTB70 has no per-sequence attribute metadata, so any non-empty
        ``attributes`` filter results in an empty dataset.  An empty filter
        (``set()``) returns the full dataset unchanged.
        """
        if not attributes:
            return DTB70Dataset(root=self.root, max_frames=self.max_frames)
        # No attribute metadata — return a view that yields nothing.
        obj = object.__new__(DTB70Dataset)
        obj.root = self.root
        obj.max_frames = self.max_frames
        obj._seq_root = self._seq_root
        obj._filter_empty = True
        return obj


__all__ = ["DTB70Dataset"]
