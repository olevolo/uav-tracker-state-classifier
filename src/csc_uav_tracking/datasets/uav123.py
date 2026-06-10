"""UAV123 dataset loader (Phase 6).

Paper / PLAN §3.6: Mueller 2016 UAV123, 123 sequences, ~110K frames,
attribute splits (FM, OCC, IV, LR, ...). OPE protocol — init on frame 0,
no re-init on failure.

Layout (inside the zip's nested UAV123/ dir)::

    <root>/
      data_seq/UAV123/<seq_folder>/XXXXXX.jpg
      anno/UAV123/<seq_name>.txt          # x,y,w,h per line (comma-sep)
      anno/UAV123/att/<seq_name>.txt      # 12 space-separated 0/1 flags

Auto-detect root: accept either the outer ``uav123/`` or the inner
``uav123/UAV123/`` — the code picks whichever has ``data_seq/UAV123/``.
"""

from __future__ import annotations

import logging
import math
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

# Ordered 12-column attribute names from PLAN §3.6 / UAV123 paper.
_ATTR_NAMES = ["FM", "OCC", "IV", "SV", "POC", "DEF", "MB", "CM", "BC", "SOB", "LR", "ARC"]

# Warning-once guard keys.
_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        warnings.warn(msg, stacklevel=2)
        _warned.add(key)


# ---------------------------------------------------------------------------
# Internal Sequence implementation
# ---------------------------------------------------------------------------


class _UAV123Sequence:
    """Lazy sequence object conforming to the datasets.base Sequence Protocol.

    Frames are read from disk on-the-fly so RAM stays flat for long sequences.
    ``ground_truth`` is a list of ``BBox`` with ``valid`` attribute (added via
    a lightweight subclass below).
    """

    def __init__(
        self,
        name: str,
        frame_dir: Path,
        frame_numbers: list[int],
        ground_truth: list[BBox],
        attributes: set[str],
    ) -> None:
        self.name = name
        self._frame_dir = frame_dir
        self._frame_numbers = frame_numbers
        self.ground_truth = ground_truth
        self.attributes = attributes

    @property
    def init_bbox(self) -> BBox:
        return self.ground_truth[0]

    @property
    def frames(self) -> Iterable[np.ndarray]:
        return self._FrameIterable(self._frame_dir, self._frame_numbers)

    class _FrameIterable:
        """Lazy iterable; re-reads from disk each iteration."""

        def __init__(self, frame_dir: Path, frame_numbers: list[int]) -> None:
            self._frame_dir = frame_dir
            self._frame_numbers = frame_numbers

        def __iter__(self) -> Generator[np.ndarray, None, None]:
            for n in self._frame_numbers:
                path = self._frame_dir / f"{n:06d}.jpg"
                img = cv2.imread(str(path))
                if img is None:
                    _warn_once(
                        f"missing_frame:{path}",
                        f"UAV123: frame not found at {path} — substituting grey placeholder",
                    )
                    img = np.zeros((360, 640, 3), dtype=np.uint8)
                yield img


# ---------------------------------------------------------------------------
# BBox with optional validity flag
# ---------------------------------------------------------------------------


class _BBoxAnnotated(BBox):
    """BBox subclass that carries a ``valid`` flag for NaN annotations.

    We subclass the frozen dataclass carefully using __new__ to work around
    the frozen=True constraint while still being an instance of BBox.
    """

    # We have to store ``valid`` on the instance manually since BBox is frozen.
    __slots__ = ("valid",)

    def __new__(cls, x: float, y: float, w: float, h: float, valid: bool = True):
        obj = super().__new__(cls)
        # BBox is frozen=True, we set fields via object.__setattr__.
        object.__setattr__(obj, "x", x)
        object.__setattr__(obj, "y", y)
        object.__setattr__(obj, "w", w)
        object.__setattr__(obj, "h", h)
        return obj

    def __init__(self, x: float, y: float, w: float, h: float, valid: bool = True) -> None:
        # frozen dataclass doesn't call __init__ for field assignment, so
        # we set valid via object.__setattr__ since __slots__ is defined.
        object.__setattr__(self, "valid", valid)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_anno(path: Path) -> list[_BBoxAnnotated]:
    """Parse ``x,y,w,h`` annotation file, one row per frame.

    Handles:
    - comma or whitespace separators
    - NaN entries (marked as valid=False, bbox filled with 0)
    """
    bboxes: list[_BBoxAnnotated] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Split on comma or whitespace.
            parts = re.split(r"[,\s]+", line)
            parts = [p for p in parts if p]
            if len(parts) < 4:
                _warn_once(f"short_line:{path}", f"UAV123: short annotation line in {path}")
                bboxes.append(_BBoxAnnotated(0.0, 0.0, 0.0, 0.0, valid=False))
                continue
            try:
                vals = [float(p) for p in parts[:4]]
                if any(math.isnan(v) for v in vals):
                    bboxes.append(_BBoxAnnotated(0.0, 0.0, 0.0, 0.0, valid=False))
                else:
                    bboxes.append(_BBoxAnnotated(*vals, valid=True))
            except ValueError:
                _warn_once(
                    f"bad_value:{path}",
                    f"UAV123: unparseable value in {path}: {line!r}",
                )
                bboxes.append(_BBoxAnnotated(0.0, 0.0, 0.0, 0.0, valid=False))
    return bboxes


def _parse_attributes(att_path: Path) -> set[str]:
    """Parse single-line attribute file: 12 space-separated 0/1 flags.

    Returns a set of active attribute codes. Degrades gracefully if the
    file has fewer/more columns than expected.
    """
    if not att_path.exists():
        return set()
    try:
        with open(att_path) as fh:
            line = fh.readline().strip()
        parts = [p for p in re.split(r"[,\s]+", line) if p]
        if not parts:
            return set()
        if len(parts) != len(_ATTR_NAMES):
            _warn_once(
                f"att_cols:{att_path.stem}",
                f"UAV123: attribute file {att_path.name} has {len(parts)} columns "
                f"(expected {len(_ATTR_NAMES)}); using available columns",
            )
        attrs: set[str] = set()
        for i, val in enumerate(parts):
            if i >= len(_ATTR_NAMES):
                break
            try:
                if int(float(val)) != 0:
                    attrs.add(_ATTR_NAMES[i])
            except ValueError:
                pass
        return attrs
    except Exception as exc:
        _warn_once(
            f"att_error:{att_path.stem}",
            f"UAV123: failed to parse attribute file {att_path}: {exc}",
        )
        return set()


# ---------------------------------------------------------------------------
# configSeqs.m parser to get exact start/end frames
# ---------------------------------------------------------------------------


def _parse_config_seqs(config_path: Path) -> dict[str, tuple[str, int, int]]:
    """Parse configSeqs.m and return ``{seq_name: (folder, start, end)}``.

    Handles the case where a long sequence is split into sub-segments
    (e.g. ``bird1_1``, ``bird1_2`` all map to folder ``bird1``).
    """
    result: dict[str, tuple[str, int, int]] = {}
    if not config_path.exists():
        return result
    try:
        with open(config_path) as fh:
            content = fh.read()
        # Find the seqUAV123 section.
        idx = content.find("seqUAV123=")
        if idx == -1:
            return result
        section = content[idx:]
        pattern = (
            r"struct\('name','(\w+)','path','.*?\\(\w+)\\','startFrame',(\d+),'endFrame',(\d+)"
        )
        for m in re.finditer(pattern, section):
            name, folder, start, end = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            result[name] = (folder, start, end)
    except Exception as exc:
        _warn_once("config_parse_error", f"UAV123: error parsing configSeqs.m: {exc}")
    return result


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------


@DATASETS.register("uav123")
class UAV123Dataset:
    """Lazily-iterable UAV123 dataset conforming to the Dataset Protocol.

    Parameters
    ----------
    root:
        Dataset root. Accepts either the outer ``uav123/`` dir (where
        ``UAV123/`` is a subdirectory) or the inner ``uav123/UAV123/`` dir
        (where ``data_seq/UAV123/`` lives directly). Auto-detected.
        Defaults to ``$UAV_DATA_ROOT/uav123/UAV123/`` if ``root`` is not set.
    split:
        Currently ``"test"`` (all 123 sequences) — UAV123 has no
        dedicated train/val split.
    attributes:
        If non-None, only yield sequences whose attribute set intersects
        this set (e.g. ``{"FM", "OCC"}``).
    max_frames:
        Optional cap on frames per sequence (useful for fast integration
        tests). ``None`` means use all frames.
    """

    name: str = "uav123"

    def __init__(
        self,
        root: Path | str | None = None,
        split: str = "test",
        attributes: set[str] | None = None,
        max_frames: int | None = None,
        frame_stride: int = 1,
    ) -> None:
        if root is None:
            from csc_uav_tracking.paths import data_root
            root = data_root() / "uav123" / "UAV123"
        self.root = self._resolve_root(Path(root))
        self.split = split
        self.attributes = attributes
        self.max_frames = max_frames
        self.frame_stride = frame_stride

    @staticmethod
    def _resolve_root(root: Path) -> Path:
        """Normalise root to the directory that contains ``data_seq/UAV123/``."""
        # If data_seq/UAV123/ is directly inside root, use root as-is.
        if (root / "data_seq" / "UAV123").exists():
            return root
        # If root/UAV123/data_seq/UAV123/ exists, descend once.
        nested = root / "UAV123"
        if (nested / "data_seq" / "UAV123").exists():
            return nested
        # Return as-is and let __iter__ fail with a clear message.
        return root

    def __iter__(self) -> Iterator[_UAV123Sequence]:
        """Yield one :class:`_UAV123Sequence` per UAV123 clip in deterministic order."""
        anno_dir = self.root / "anno" / "UAV123"
        att_dir = anno_dir / "att"
        data_dir = self.root / "data_seq" / "UAV123"
        config_path = self.root / "configSeqs.m"

        if not anno_dir.exists():
            raise FileNotFoundError(
                f"UAV123 anno directory not found at {anno_dir}. "
                f"Check that root={self.root} is correct."
            )

        # Load sequence config (start/end frames from configSeqs.m).
        seq_config = _parse_config_seqs(config_path)

        # Enumerate annotation files in sorted order (gives deterministic ordering).
        anno_files = sorted(anno_dir.glob("*.txt"))

        for anno_path in anno_files:
            seq_name = anno_path.stem

            # --- parse attributes ---
            att_path = att_dir / f"{seq_name}.txt"
            seq_attrs = _parse_attributes(att_path)

            # --- attribute filter ---
            if self.attributes is not None:
                if not (seq_attrs & self.attributes):
                    continue

            # --- parse annotation ---
            try:
                all_bboxes = _parse_anno(anno_path)
            except Exception as exc:
                _log.warning("UAV123: skipping %s — failed to parse anno: %s", seq_name, exc)
                continue

            if not all_bboxes:
                _log.warning("UAV123: skipping %s — empty annotation", seq_name)
                continue

            # Make sure first annotation is valid (needed for init_bbox).
            if not all_bboxes[0].valid:  # type: ignore[attr-defined]
                _log.warning("UAV123: skipping %s — init frame has NaN annotation", seq_name)
                continue

            # --- frame numbers from configSeqs.m ---
            if seq_name in seq_config:
                folder_name, start_frame, end_frame = seq_config[seq_name]
                # Frame numbers are 1-based as per the dataset.
                all_frame_numbers = list(range(start_frame, end_frame + 1))
                frame_dir = data_dir / folder_name
            else:
                # Fallback: infer from annotation length.
                folder_name = seq_name
                all_frame_numbers = list(range(1, len(all_bboxes) + 1))
                frame_dir = data_dir / folder_name
                if not frame_dir.exists():
                    _log.warning(
                        "UAV123: skipping %s — frame dir not found at %s", seq_name, frame_dir
                    )
                    continue

            # Align annotation length with frame count.
            n = min(len(all_bboxes), len(all_frame_numbers))
            bboxes = all_bboxes[:n]
            frame_numbers = all_frame_numbers[:n]

            # Apply max_frames cap.
            if self.max_frames is not None:
                bboxes = bboxes[: self.max_frames]
                frame_numbers = frame_numbers[: self.max_frames]

            # Apply frame_stride for subsampled variants (e.g. @10fps = stride 3).
            if self.frame_stride > 1:
                bboxes = bboxes[:: self.frame_stride]
                frame_numbers = frame_numbers[:: self.frame_stride]

            if not frame_numbers:
                continue

            yield _UAV123Sequence(
                name=seq_name,
                frame_dir=frame_dir,
                frame_numbers=frame_numbers,
                ground_truth=bboxes,  # type: ignore[arg-type]
                attributes=seq_attrs,
            )

    def filter(self, attributes: set[str]) -> "UAV123Dataset":
        """Return a filtered dataset keeping only sequences matching attributes."""
        merged = (self.attributes or set()) | attributes
        return UAV123Dataset(
            root=self.root,
            split=self.split,
            attributes=merged,
            max_frames=self.max_frames,
        )


@DATASETS.register("uav123_10fps")
class UAV123Dataset10fps(UAV123Dataset):
    """UAV123 subsampled to ~10 fps (every 3rd frame).

    Original UAV123 is recorded at ~30 fps.  Subsampling to every 3rd frame
    simulates a lower frame-rate UAV tracking scenario with larger
    inter-frame displacements — harder for trackers, expected to produce
    higher FCR and longer FC episodes.
    """

    def __init__(self, root=None, split: str = "test",
                 attributes=None, max_frames=None) -> None:
        super().__init__(root=root, split=split, attributes=attributes,
                         max_frames=max_frames, frame_stride=3)
