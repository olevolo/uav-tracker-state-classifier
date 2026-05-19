"""collect_features.py — SALT-RD offline feature collection.

Runs the frozen SGLATracker/SALTRunner on UAV123, VisDrone-SOT, and DTB70
sequences.  At each frame, 28 scalar telemetry features are extracted from
tracker internals.  Labels are derived exclusively from GT IoU and
teacher-derived signals — never from _decide_state(), TargetState, scene_class,
APCE threshold rules, or LSTM dynamic-branch outputs.

Output: a single NPZ file (see SavedDataset) suitable for downstream GRU
training with salt_r/train.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    # Score map (9 features)
    "apce_raw",
    "apce_norm",
    "psr",
    "entropy",
    "peak_margin",
    "peak_width",
    "n_secondary",
    "peak_distance",
    "heatmap_mass_topk",
    # Temporal response (6 features)
    "apce_ratio_5",
    "apce_ratio_20",
    "entropy_delta_5",
    "peak_margin_delta_5",
    "confirmed_streak",
    "low_conf_streak",
    # Target dynamics (7 features)
    "bbox_cx_velocity",
    "bbox_cy_velocity",
    "bbox_speed_norm",
    "bbox_accel_norm",
    "scale_ratio",
    "aspect_ratio_delta",
    "dist_to_search_border",
    # Camera/flow (6 features)
    "global_flow_mag",
    "target_flow_mag",
    "ego_motion_residual",
    "flow_iou",
    "flow_residual",
    "flow_consistency",
]

FEATURE_UNITS: List[str] = [
    # Score map
    "[0,∞)",
    "[0,1]",
    "[0,∞)",
    "[0,∞)",
    "[0,∞)",
    "cells",
    "count",
    "cells",
    "[0,1]",
    # Temporal response
    "[0,∞)",
    "[0,∞)",
    "[-∞,∞)",
    "[-∞,∞)",
    "frames",
    "frames",
    # Target dynamics
    "px/frame",
    "px/frame",
    "px/frame",
    "px/frame²",
    "[0,∞)",
    "rad",
    "[0,1]",
    # Camera/flow
    "px/frame",
    "px/frame",
    "px/frame",
    "[0,1]",
    "px",
    "[0,1]",
]

assert len(FEATURE_NAMES) == len(FEATURE_UNITS), (
    f"FEATURE_NAMES ({len(FEATURE_NAMES)}) and FEATURE_UNITS ({len(FEATURE_UNITS)}) must have the same length"
)

N_FEATURES: int = len(FEATURE_NAMES)  # 28

LABEL_NAMES: List[str] = [
    "correct",            # 0: tracker on right object (IoU >= 0.5)
    "false_confirmed",    # 1: tracker confident but on wrong object
    "failure_in_5",       # 2: currently OK but will fail within 5 frames
    "recoverable",        # 3: currently lost but can recover within 5-15 frames
    "target_dynamic",     # 4: high target motion relative to sequence
    "camera_dynamic",     # 5: high ego-motion relative to sequence
    "hard_dynamic_scene", # 6: dynamicity + tracking ambiguity combined
    "needs_full_compute", # 7: bootstrap — needs full SGLATrack inference (oracle v1)
]

N_LABELS: int = len(LABEL_NAMES)  # 8

# ---------------------------------------------------------------------------
# Diagnostic sequences — known edge-cases for per-sequence debugging
# ---------------------------------------------------------------------------

DIAGNOSTIC_SEQUENCES = frozenset({
    "uav0000164",         # VisDrone: 99% CONFIRMED, AUC=0.174
    "bike2",              # UAV123: identity-loss hard case
    "Gull2",              # DTB70: hard case
    "Sheep1",             # DTB70: hard case
    "StreetBasketball1",  # DTB70: hard case
})

# ---------------------------------------------------------------------------
# Label computation
# ---------------------------------------------------------------------------


def compute_labels(
    iou_trace: np.ndarray,           # (n_frames,) GT IoU per frame
    apce_norm: np.ndarray,           # (n_frames,) normalized APCE in [0,1]
    speed_norm: np.ndarray,          # (n_frames,) normalized target speed px/frame
    accel_norm: np.ndarray,          # (n_frames,) normalized target accel px/frame²
    scale_delta: np.ndarray,         # (n_frames,) scale change per frame (ratio-1)
    global_flow_mag: np.ndarray,     # (n_frames,) global optical flow magnitude px/frame
    ego_motion_residual: np.ndarray, # (n_frames,) residual motion after ego-compensation
    peak_margin: np.ndarray,         # (n_frames,) top1-top2 score gap
    flow_consistency: np.ndarray,    # (n_frames,) flow consistency score [0,1]
) -> np.ndarray:
    """Compute SALT-RD labels from GT/teacher-derived signals only.

    CRITICAL: Labels must NOT use _decide_state(), TargetState, scene_class,
    APCE thresholds as teacher, or LSTM dynamic branch outputs.

    Parameters
    ----------
    iou_trace:
        Per-frame intersection-over-union with GT bounding box.
    apce_norm:
        APCE score normalised to [0,1] range (apce_raw / 256.0 approx).
    speed_norm:
        Target bounding-box centroid speed in px/frame.
    accel_norm:
        Target bounding-box centroid acceleration in px/frame².
    scale_delta:
        Per-frame fractional change in bounding-box area (sqrt(w*h) ratio - 1).
    global_flow_mag:
        Mean optical flow magnitude across the full frame.
    ego_motion_residual:
        Magnitude of motion remaining after subtracting ego-motion model.
    peak_margin:
        Score-map top-1 minus top-2 peak value gap.
    flow_consistency:
        Agreement between optical-flow prediction and tracker bbox shift.

    Returns
    -------
    np.ndarray of shape (n_frames, 8), dtype int8.
    """
    n = len(iou_trace)
    labels = np.zeros((n, N_LABELS), dtype=np.int8)

    # 0: correct — tracker on right object
    labels[:, 0] = (iou_trace >= 0.5).astype(np.int8)

    # 1: false_confirmed — tracker confident but on wrong object
    # apce_norm > 100/256 ≈ 0.39 is the "high confidence" threshold derived from
    # SALT v3 calibration (see project_salt_calibration.md)
    labels[:, 1] = ((iou_trace < 0.2) & (apce_norm > 100.0 / 256.0)).astype(np.int8)

    # 2: failure_in_5 — currently OK but IoU will drop severely in next 5 frames
    for t in range(n):
        if iou_trace[t] >= 0.5:
            future = iou_trace[t + 1 : t + 6]
            if len(future) > 0 and future.mean() < 0.3:
                labels[t, 2] = 1

    # 3: recoverable — currently lost but target will be findable within 5-15 frames
    for t in range(n):
        if iou_trace[t] < 0.2:
            future = iou_trace[t + 5 : t + 15]
            if len(future) > 0 and future.max() >= 0.5:
                labels[t, 3] = 1

    # 4: target_dynamic — high target motion relative to sequence
    motion_score = speed_norm + 0.5 * accel_norm + 0.5 * np.abs(scale_delta)
    threshold_td = np.percentile(motion_score, 75)
    labels[:, 4] = (motion_score >= threshold_td).astype(np.int8)

    # 5: camera_dynamic — high ego-motion relative to sequence
    cam_score = global_flow_mag + ego_motion_residual
    threshold_cd = np.percentile(cam_score, 75)
    labels[:, 5] = (cam_score >= threshold_cd).astype(np.int8)

    # 6: hard_dynamic_scene — dynamicity combined with tracking ambiguity
    is_dynamic = (labels[:, 4] | labels[:, 5]).astype(bool)
    peak_margin_low = peak_margin < np.percentile(peak_margin, 25)
    flow_consistency_low = flow_consistency < 0.3
    future_risk = np.zeros(n, dtype=bool)
    for t in range(n):
        future = iou_trace[t + 1 : t + 6]
        if len(future) > 0 and future.min() < 0.3:
            future_risk[t] = True
    labels[:, 6] = (
        is_dynamic & (peak_margin_low | flow_consistency_low | future_risk)
    ).astype(np.int8)

    # 7: needs_full_compute — bootstrap approximation (oracle mode in v1)
    labels[:, 7] = (labels[:, 6] | labels[:, 2]).astype(np.int8)

    return labels


# ---------------------------------------------------------------------------
# NPZ dataset container
# ---------------------------------------------------------------------------


@dataclass
class SavedDataset:
    """In-memory container for one complete SALT-RD collection run.

    NPZ keys
    --------
    features/{seq_name}      float32  (n_frames, n_features)
    feature_names            object   list[str] as np.array
    feature_units            object   list[str] as np.array
    labels/{seq_name}        int8     (n_frames, n_labels)
    label_names              object   list[str] as np.array
    iou_trace/{seq_name}     float32  (n_frames,)
    bbox_pred/{seq_name}     float32  (n_frames, 4)  — (x, y, w, h)
    bbox_gt/{seq_name}       float32  (n_frames, 4)
    sequence_name/{seq_name} str      sequence identifier
    dataset/{seq_name}       str      "uav123" | "visdrone_sot" | "dtb70"
    split/{seq_name}         str      "train" | "val" | "diagnostic"
    tracker_version          str      e.g. "sglatrack_ep0297"
    tracker_config_hash      str      sha256 of config YAML
    created_at               str      ISO 8601 datetime (UTC)
    """

    tracker_version: str
    tracker_config_hash: str

    # Per-sequence data (keyed by sequence name)
    features: Dict[str, np.ndarray] = field(default_factory=dict)
    labels: Dict[str, np.ndarray] = field(default_factory=dict)
    iou_trace: Dict[str, np.ndarray] = field(default_factory=dict)
    bbox_pred: Dict[str, np.ndarray] = field(default_factory=dict)
    bbox_gt: Dict[str, np.ndarray] = field(default_factory=dict)
    dataset: Dict[str, str] = field(default_factory=dict)
    split: Dict[str, str] = field(default_factory=dict)

    def add_sequence(
        self,
        seq_name: str,
        features: np.ndarray,
        labels: np.ndarray,
        iou_trace: np.ndarray,
        bbox_pred: np.ndarray,
        bbox_gt: np.ndarray,
        dataset_name: str,
    ) -> None:
        """Register one sequence into this dataset.

        Parameters
        ----------
        seq_name:
            Unique identifier string, e.g. "person1_s1" or "uav0000164".
        features:
            float32 array of shape (n_frames, N_FEATURES).
        labels:
            int8 array of shape (n_frames, N_LABELS).
        iou_trace:
            float32 array of shape (n_frames,).
        bbox_pred:
            float32 array of shape (n_frames, 4) — (x, y, w, h).
        bbox_gt:
            float32 array of shape (n_frames, 4) — (x, y, w, h).
        dataset_name:
            One of "uav123", "visdrone_sot", "dtb70".
        """
        assert features.shape == (len(iou_trace), N_FEATURES), (
            f"{seq_name}: features shape {features.shape} expected ({len(iou_trace)}, {N_FEATURES})"
        )
        assert labels.shape == (len(iou_trace), N_LABELS), (
            f"{seq_name}: labels shape {labels.shape} expected ({len(iou_trace)}, {N_LABELS})"
        )
        assert dataset_name in {"uav123", "visdrone_sot", "dtb70"}, (
            f"Unknown dataset '{dataset_name}'"
        )

        split_name: str
        if seq_name in DIAGNOSTIC_SEQUENCES:
            split_name = "diagnostic"
        else:
            # Deterministic 80/20 train/val split via name hash
            h = int(hashlib.md5(seq_name.encode()).hexdigest(), 16)
            split_name = "train" if (h % 10) < 8 else "val"

        self.features[seq_name] = features.astype(np.float32)
        self.labels[seq_name] = labels.astype(np.int8)
        self.iou_trace[seq_name] = iou_trace.astype(np.float32)
        self.bbox_pred[seq_name] = bbox_pred.astype(np.float32)
        self.bbox_gt[seq_name] = bbox_gt.astype(np.float32)
        self.dataset[seq_name] = dataset_name
        self.split[seq_name] = split_name

    def save(self, output_path: str | Path) -> None:
        """Serialise the dataset to a compressed NPZ file.

        Parameters
        ----------
        output_path:
            Destination path for the .npz file.  Parent directories are
            created automatically.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict = {
            "feature_names": np.array(FEATURE_NAMES, dtype=object),
            "feature_units": np.array(FEATURE_UNITS, dtype=object),
            "label_names": np.array(LABEL_NAMES, dtype=object),
            "tracker_version": np.array(self.tracker_version),
            "tracker_config_hash": np.array(self.tracker_config_hash),
            "created_at": np.array(
                datetime.now(tz=timezone.utc).isoformat()
            ),
        }

        for seq_name in self.features:
            arrays[f"features/{seq_name}"] = self.features[seq_name]
            arrays[f"labels/{seq_name}"] = self.labels[seq_name]
            arrays[f"iou_trace/{seq_name}"] = self.iou_trace[seq_name]
            arrays[f"bbox_pred/{seq_name}"] = self.bbox_pred[seq_name]
            arrays[f"bbox_gt/{seq_name}"] = self.bbox_gt[seq_name]
            arrays[f"sequence_name/{seq_name}"] = np.array(seq_name)
            arrays[f"dataset/{seq_name}"] = np.array(self.dataset[seq_name])
            arrays[f"split/{seq_name}"] = np.array(self.split[seq_name])

        np.savez_compressed(str(output_path), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "SavedDataset":
        """Load a previously saved NPZ file back into a SavedDataset.

        Parameters
        ----------
        path:
            Path to the .npz file produced by :meth:`save`.
        """
        data = np.load(str(path), allow_pickle=True)

        tracker_version = str(data["tracker_version"])
        tracker_config_hash = str(data["tracker_config_hash"])

        ds = cls(
            tracker_version=tracker_version,
            tracker_config_hash=tracker_config_hash,
        )

        # Recover sequence names from features/ keys
        seq_names = [
            k[len("features/") :]
            for k in data.files
            if k.startswith("features/")
        ]
        for seq_name in seq_names:
            ds.features[seq_name] = data[f"features/{seq_name}"]
            ds.labels[seq_name] = data[f"labels/{seq_name}"]
            ds.iou_trace[seq_name] = data[f"iou_trace/{seq_name}"]
            ds.bbox_pred[seq_name] = data[f"bbox_pred/{seq_name}"]
            ds.bbox_gt[seq_name] = data[f"bbox_gt/{seq_name}"]
            ds.dataset[seq_name] = str(data[f"dataset/{seq_name}"])
            ds.split[seq_name] = str(data[f"split/{seq_name}"])

        return ds

    def summary(self) -> str:
        """Return a human-readable summary of the dataset."""
        total_frames = sum(v.shape[0] for v in self.features.values())
        by_dataset: Dict[str, int] = {}
        by_split: Dict[str, int] = {}
        for seq_name, feats in self.features.items():
            n = feats.shape[0]
            by_dataset[self.dataset[seq_name]] = by_dataset.get(self.dataset[seq_name], 0) + n
            by_split[self.split[seq_name]] = by_split.get(self.split[seq_name], 0) + n
        lines = [
            f"SavedDataset — tracker: {self.tracker_version}",
            f"  sequences  : {len(self.features)}",
            f"  total frames: {total_frames:,}",
            f"  by dataset : {by_dataset}",
            f"  by split   : {by_split}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config hash helper
# ---------------------------------------------------------------------------


def hash_config_file(config_path: str | Path) -> str:
    """Return the SHA-256 hex digest of a YAML config file.

    Parameters
    ----------
    config_path:
        Path to the SALT/SGLATrack YAML config file.
    """
    sha = hashlib.sha256()
    with open(config_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ---------------------------------------------------------------------------
# CLI entry point (skeleton — collection loop not yet implemented)
# ---------------------------------------------------------------------------


def main() -> None:
    """Collect SALT-RD features from frozen tracker runs.

    Usage::

        python collect_features.py --config /path/to/salt.yaml \\
                                   --datasets uav123 visdrone_sot dtb70 \\
                                   --output saltr/data/salt_rd_v0.npz
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect SALT-RD features from frozen SGLATrack/SALTRunner runs."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to SGLATrack/SALT YAML config (used to derive tracker_config_hash).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["uav123", "visdrone_sot", "dtb70"],
        choices=["uav123", "visdrone_sot", "dtb70"],
        help="Which benchmark datasets to collect from.",
    )
    parser.add_argument(
        "--output",
        default="saltr/data/salt_rd_v0.npz",
        help="Output NPZ path.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap per-sequence frame count for fast debug runs (None = no cap).",
    )
    parser.add_argument(
        "--tracker-version",
        default="sglatrack_ep0297",
        help="Human-readable tracker version tag embedded in the NPZ.",
    )

    args = parser.parse_args()

    # TODO: implement full collection loop after verifying NPZ schema
    # Steps:
    #   1. Load frozen tracker from args.config
    #   2. For each dataset in args.datasets:
    #      a. Iterate sequences
    #      b. Run tracker frame-by-frame
    #      c. Extract FEATURE_NAMES from tracker internals
    #      d. Accumulate per-frame features, bboxes, iou vs GT
    #   3. Call compute_labels() for each sequence
    #   4. Call SavedDataset.add_sequence() for each sequence
    #   5. Call SavedDataset.save(args.output)
    raise NotImplementedError(
        "Collection loop not implemented yet — implement after verifying NPZ schema "
        "and confirming feature extraction hooks in SGLATracker telemetry."
    )


if __name__ == "__main__":
    main()
