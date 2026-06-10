"""UAVTrack112 (V4RFlight112) single-object tracking dataset loader.

112 sequences from UAV viewpoint, tracked from another UAV.
Dataset also known as V4RFlight112.

Expected on-disk layout::

    V4RFlight112/          (or UAVTrack112/ symlink)
    ├── anno/
    │   ├── bike1.txt      ← x,y,w,h per line (comma-separated, 1-indexed)
    │   └── ...            (112 files: 100 day + 12 night)
    ├── attributes/
    │   ├── bike1.txt      ← 13 binary flags comma-separated
    │   └── ...
    └── data_seq/
        ├── bike1/
        │   ├── 0001.jpg
        │   └── ...
        └── ...            (42 sequences with images; 70 annotation-only)

Only sequences that have BOTH annotation AND image frames are loaded.
The night sequences (suffix -n) are excluded from the default split.

Root auto-detection order:
    1. ``$UAVTRACK112_DATA_ROOT`` env var
    2. ``~/uav-tracker-data/UAVTrack112/``  (symlink → V4RFlight112)
    3. ``~/uav-tracker-data/V4RFlight112/``

Registered as ``"uavtrack112"`` in DATASETS.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator

import numpy as np

from csc_uav_tracking.registry import DATASETS

log = logging.getLogger(__name__)

# 13 attribute names from the dataset README
ATTRIBUTES = [
    "illumination_variation", "scale_variation", "occlusion",
    "deformation", "motion_blur", "fast_motion", "in_plane_rotation",
    "out_of_plane_rotation", "out_of_view", "background_clutter",
    "low_resolution", "similar_objects", "viewpoint_change",
]


def _find_root() -> Path:
    candidates = [
        os.environ.get("UAVTRACK112_DATA_ROOT"),
        Path.home() / "uav-tracker-data" / "UAVTrack112",
        Path.home() / "uav-tracker-data" / "V4RFlight112",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(
        "UAVTrack112 (V4RFlight112) not found. Set UAVTRACK112_DATA_ROOT "
        "or place dataset at ~/uav-tracker-data/UAVTrack112/."
    )


def _read_gt(path: Path) -> np.ndarray:
    """Read xywh ground truth, return (N, 4) float array. NaN rows → [0,0,0,0]."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                vals = [float(v) for v in line.replace("\t", ",").split(",")[:4]]
                if len(vals) == 4 and not any(v != v for v in vals):  # skip NaN
                    rows.append(vals)
                else:
                    rows.append([0.0, 0.0, 0.0, 0.0])
            except ValueError:
                rows.append([0.0, 0.0, 0.0, 0.0])
    return np.array(rows, dtype=np.float32)


def _read_attributes(path: Path) -> dict:
    if not path.exists():
        return {}
    vals = path.read_text().strip().rstrip(",").split(",")
    return {ATTRIBUTES[i]: int(v) for i, v in enumerate(vals) if i < len(ATTRIBUTES)}


@DATASETS.register("uavtrack112")
class UAVTrack112Dataset:
    """UAVTrack112 / V4RFlight112 dataset."""

    name = "uavtrack112"

    def __init__(
        self,
        root: Path | None = None,
        *,
        include_night: bool = False,
        split: str = "test",
    ) -> None:
        self.root = Path(root) if root else _find_root()
        self.anno_dir  = self.root / "anno"
        self.img_dir   = self.root / "data_seq"
        self.attr_dir  = self.root / "attributes"

        # Build sequence list: only sequences with both anno + images
        sequences = []
        for anno_file in sorted(self.anno_dir.glob("*.txt")):
            seq_name = anno_file.stem
            if not include_night and seq_name.endswith("-n"):
                continue
            img_folder = self.img_dir / seq_name
            if not img_folder.exists():
                continue
            imgs = sorted(
                list(img_folder.glob("*.jpg")) +
                list(img_folder.glob("*.png"))
            )
            if not imgs:
                continue
            gt = _read_gt(anno_file)
            if len(gt) == 0:
                continue
            # Align frames and gt
            n = min(len(imgs), len(gt))
            attrs = _read_attributes(self.attr_dir / f"{seq_name}.txt")
            sequences.append({
                "name": seq_name,
                "frames": imgs[:n],
                "gt": gt[:n],
                "attributes": attrs,
            })

        self.sequences = sequences
        log.info(
            "UAVTrack112: %d sequences with images (of 112 total), "
            "%d total frames",
            len(sequences),
            sum(len(s["frames"]) for s in sequences),
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self) -> Iterator:
        for s in self.sequences:
            yield self._make_sequence(s)

    def _make_sequence(self, s: dict):
        """Return a Sequence-compatible object."""
        from csc_lib.csc.labeling.label_schema import DerivedState  # noqa

        class _Seq:
            name      = s["name"]
            dataset   = "uavtrack112"
            gt_bboxes = s["gt"]          # (N, 4) xywh
            attributes = s["attributes"]

            full_occlusion = np.array(
                [s["attributes"].get("occlusion", 0)] * len(s["frames"]),
                dtype=bool,
            )
            out_of_view = np.array(
                [s["attributes"].get("out_of_view", 0)] * len(s["frames"]),
                dtype=bool,
            )

            @property
            def init_bbox(self):
                import types
                x, y, w, h = s["gt"][0]
                return types.SimpleNamespace(x=x, y=y, w=w, h=h)

            @property
            def frames(self):
                import cv2
                for p in s["frames"]:
                    img = cv2.imread(str(p))
                    if img is not None:
                        yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            @property
            def ground_truth(self):
                import types
                bboxes = []
                for xywh in s["gt"]:
                    x, y, w, h = xywh
                    bb = types.SimpleNamespace(x=x, y=y, w=w, h=h, valid=w > 0 and h > 0)
                    bboxes.append(bb)
                return bboxes

        return _Seq()
