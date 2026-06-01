"""UAVDT-SOT single-object tracking dataset loader.

Paper: Du et al., "The Unmanned Aerial Vehicle Benchmark: Object Detection
and Tracking", ECCV 2018.  https://sites.google.com/view/grli-uavdt/

Usage in this project: aerial domain validation / hard-negative source.
NOT used as final test (UAV123 is the reserved final benchmark).

Expected on-disk layout (two layouts supported):

Layout A — flat (SOT toolkit output):
    UAVDT/
    ├── M0101/
    │   ├── img/
    │   │   ├── 000001.jpg
    │   │   └── ...
    │   └── groundtruth.txt   ← x,y,w,h per line
    ├── M0201/
    └── ...

Layout B — split anno (original benchmark style):
    UAVDT/
    ├── data_seq/
    │   ├── M0101/
    │   │   └── img/
    │   │       ├── 000001.jpg
    │   │       └── ...
    │   └── ...
    └── anno/
        ├── M0101.txt          ← x,y,w,h per line
        └── ...

Root auto-detection order:
    1. ``$UAVDT_SOT_DATA_ROOT`` env var
    2. ``$UAV_DATA_ROOT/UAVDT/``
    3. ``~/uav-tracker-data/UAVDT/``

Registered as ``"uavdt_sot"`` in DATASETS.
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


# ---------------------------------------------------------------------------
# BBox subclass with validity flag
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
# Sequence
# ---------------------------------------------------------------------------

class _UAVDTSequence:
    """Single UAVDT-SOT tracking sequence — lazy frame loading."""

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
                        f"missing:{p}",
                        f"UAVDT-SOT: frame not found at {p}",
                    )
                    img = np.zeros((540, 1024, 3), dtype=np.uint8)
                yield img


# ---------------------------------------------------------------------------
# GT parser
# ---------------------------------------------------------------------------

def _parse_gt(path: Path) -> list[_BBoxAnnotated]:
    """Parse x,y,w,h annotation file (comma or space separated)."""
    bboxes: list[_BBoxAnnotated] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in re.split(r"[,\s]+", line) if p]
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


# ---------------------------------------------------------------------------
# Root + layout resolution
# ---------------------------------------------------------------------------

def _find_root() -> Path:
    """Resolve UAVDT-SOT root from env or default paths."""
    v = os.environ.get("UAVDT_SOT_DATA_ROOT")
    if v:
        return Path(v)
    uav_root = os.environ.get("UAV_DATA_ROOT", str(Path.home() / "uav-tracker-data"))
    base = Path(uav_root)
    # Auto-detect extracted location
    for candidate in (
        base / "UAV-benchmark-S",                    # direct extraction target
        base / "UAVDT",
        base / "UAVDT_extracted" / "UAV-benchmark-S",
        base / "UAV-benchmark-S",
    ):
        if candidate.exists():
            return candidate
    return base / "UAV-benchmark-S"


def _detect_layout(root: Path) -> str:
    """Return 'flat', 'uavdt_native', 'split', or 'unknown'."""
    if not root.exists():
        return "unknown"
    # Layout C (UAVDT native): sequence dirs contain img*.jpg directly
    # and GT is in a sibling anno/ directory
    for child in sorted(root.iterdir()):
        if child.is_dir():
            imgs = [p for p in child.iterdir()
                    if p.is_file() and p.suffix.lower() in (".jpg", ".png")
                    and not (child / "img").is_dir()]
            if imgs:
                return "uavdt_native"
            break
    # Layout A: flat — sequence dirs with img/ subdir + groundtruth.txt
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "groundtruth.txt").exists():
            return "flat"
    # Layout B: split — data_seq/ + anno/
    if (root / "data_seq").is_dir() and (root / "anno").is_dir():
        return "split"
    return "unknown"
    for child in sorted(root.iterdir()) if root.exists() else []:
        if child.is_dir() and (child / "groundtruth.txt").exists():
            return "flat"
    # Layout B: split — data_seq/ + anno/
    if (root / "data_seq").is_dir() and (root / "anno").is_dir():
        return "split"
    return "unknown"


def _load_sequence_uavdt_native(seq_dir: Path, anno_dir: Path | None) -> _UAVDTSequence | None:
    """Load one UAVDT-native sequence (images directly in seq dir, GT in anno/)."""
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    frame_paths = sorted(
        p for p in seq_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if not frame_paths:
        return None

    # Find GT: anno/<seq>_gt.txt (UAVDT v1.0 naming) or plain <seq>.txt
    gt: list[_BBoxAnnotated] = []
    if anno_dir is not None:
        gt_candidates = [
            anno_dir / f"{seq_dir.name}_gt.txt",   # UAVDT v1.0: S0101_gt.txt
            anno_dir / f"{seq_dir.name}.txt",
            anno_dir / seq_dir.name / "groundtruth.txt",
            seq_dir / "groundtruth.txt",
        ]
        for gt_path in gt_candidates:
            if gt_path.exists():
                gt = _parse_gt(gt_path)
                break

    if not gt:
        _warn_once(
            f"no_gt:{seq_dir.name}",
            f"UAVDT-SOT: no GT found for {seq_dir.name}. "
            "Download UAVDT-benchmark-SOT_v1.0.zip for annotations.",
        )
        return None

    n = min(len(frame_paths), len(gt))
    frame_paths = frame_paths[:n]
    gt = gt[:n]

    if not gt[0].valid or gt[0].w <= 0 or gt[0].h <= 0:
        return None

    # Read attributes: anno/att/<seq>_att.txt
    attrs: set[str] = set()
    if anno_dir is not None:
        att_file = anno_dir / "att" / f"{seq_dir.name}_att.txt"
        if att_file.exists():
            attrs = {line.strip() for line in att_file.read_text().splitlines() if line.strip()}

    return _UAVDTSequence(seq_dir.name, frame_paths, gt, attrs)


def _load_sequence_flat(seq_dir: Path) -> _UAVDTSequence | None:
    """Load one sequence from Layout A (flat)."""
    gt_path = seq_dir / "groundtruth.txt"
    if not gt_path.exists():
        return None

    img_dir = seq_dir / "img"
    if not img_dir.is_dir():
        # Try frames/ as fallback
        img_dir = seq_dir / "frames"
    if not img_dir.is_dir():
        return None

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    frame_paths = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in exts
    )
    if not frame_paths:
        return None

    gt = _parse_gt(gt_path)
    if not gt:
        return None

    # Trim to min length
    n = min(len(frame_paths), len(gt))
    frame_paths = frame_paths[:n]
    gt = gt[:n]

    # Skip sequences with degenerate init bbox
    if not gt[0].valid or (gt[0].w <= 0 or gt[0].h <= 0):
        return None

    # Read attributes if present
    attrs: set[str] = set()
    attr_file = seq_dir / "attributes.txt"
    if attr_file.exists():
        attrs = {line.strip() for line in attr_file.read_text().splitlines() if line.strip()}

    return _UAVDTSequence(seq_dir.name, frame_paths, gt, attrs)


def _load_sequence_split(data_seq_dir: Path, anno_dir: Path, seq_name: str) -> _UAVDTSequence | None:
    """Load one sequence from Layout B (split data_seq + anno)."""
    seq_dir = data_seq_dir / seq_name
    gt_path = anno_dir / f"{seq_name}.txt"

    if not seq_dir.is_dir() or not gt_path.exists():
        return None

    img_dir = seq_dir / "img"
    if not img_dir.is_dir():
        img_dir = seq_dir

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    frame_paths = sorted(
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if not frame_paths:
        return None

    gt = _parse_gt(gt_path)
    if not gt:
        return None

    n = min(len(frame_paths), len(gt))
    frame_paths = frame_paths[:n]
    gt = gt[:n]

    if not gt[0].valid or (gt[0].w <= 0 or gt[0].h <= 0):
        return None

    return _UAVDTSequence(seq_name, frame_paths, gt, set())


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class UAVDTSOTDataset:
    """UAVDT-SOT dataset — aerial domain validation / hard-negative source.

    Used for:
    - FC false-positive analysis (aerial perspective domain shift)
    - Threshold calibration for false_confirmed class
    - Validation domain for top-down / aerial perspective
    NOT used as final test benchmark (UAV123 is reserved for final eval).
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root) if root else _find_root()
        self._sequences: list[_UAVDTSequence] | None = None

    def _load_all(self) -> list[_UAVDTSequence]:
        if not self._root.exists():
            raise FileNotFoundError(
                f"UAVDT-SOT root not found: {self._root}\n"
                "Download from https://sites.google.com/view/grli-uavdt/ and set\n"
                "$UAVDT_SOT_DATA_ROOT or $UAV_DATA_ROOT/UAVDT/"
            )

        layout = _detect_layout(self._root)
        seqs: list[_UAVDTSequence] = []

        if layout == "uavdt_native":
            # UAVDT v1.0: images in UAV-benchmark-S/<seq>/, GT in UAV-benchmark-SOT_v1.0/anno/
            parent = self._root.parent
            anno_dir_candidates = [
                parent / "UAV-benchmark-SOT_v1.0" / "anno",
                self._root.parent / "anno",
                self._root / "anno",
            ]
            anno_dir = next((d for d in anno_dir_candidates if d.is_dir()), None)
            if anno_dir is None:
                _log.warning(
                    "UAVDT-SOT: no anno/ dir found. "
                    "Extract UAVDT-benchmark-SOT_v1.0.zip next to %s", self._root
                )
            for child in sorted(self._root.iterdir()):
                if not child.is_dir():
                    continue
                seq = _load_sequence_uavdt_native(child, anno_dir)
                if seq is not None:
                    seqs.append(seq)

        elif layout == "flat":
            for child in sorted(self._root.iterdir()):
                if not child.is_dir():
                    continue
                seq = _load_sequence_flat(child)
                if seq is not None:
                    seqs.append(seq)

        elif layout == "split":
            data_seq_dir = self._root / "data_seq"
            anno_dir = self._root / "anno"
            seq_names = sorted(
                p.stem for p in anno_dir.iterdir() if p.suffix == ".txt"
            )
            for name in seq_names:
                seq = _load_sequence_split(data_seq_dir, anno_dir, name)
                if seq is not None:
                    seqs.append(seq)

        else:
            raise RuntimeError(
                f"Cannot detect UAVDT-SOT layout under {self._root}.\n"
                "Expected either:\n"
                "  Layout A: <root>/<seq_name>/img/ + groundtruth.txt\n"
                "  Layout B: <root>/data_seq/<seq>/ + <root>/anno/<seq>.txt"
            )

        if not seqs:
            raise RuntimeError(
                f"No valid UAVDT-SOT sequences found under {self._root}"
            )

        _log.info("UAVDT-SOT: loaded %d sequences from %s (layout=%s)",
                  len(seqs), self._root, layout)
        return seqs

    def __iter__(self) -> Iterator[_UAVDTSequence]:
        if self._sequences is None:
            self._sequences = self._load_all()
        return iter(self._sequences)

    def __len__(self) -> int:
        if self._sequences is None:
            self._sequences = self._load_all()
        return len(self._sequences)


@DATASETS.register("uavdt_sot")
def _build_uavdt_sot(root: str | None = None, **_kwargs) -> UAVDTSOTDataset:
    return UAVDTSOTDataset(root=root)
