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
from typing import Any, Dict, List

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
    _ABS_MIN_MOTION = 0.05  # normalized px/frame — below this = static sequence
    motion_score = speed_norm + 0.5 * accel_norm + 0.5 * np.abs(scale_delta)
    if motion_score.max() <= _ABS_MIN_MOTION:
        labels[:, 4] = 0  # static sequence
    else:
        threshold_td = np.percentile(motion_score[motion_score > _ABS_MIN_MOTION], 75)
        labels[:, 4] = (motion_score > threshold_td).astype(np.int8)

    # 5: camera_dynamic — high ego-motion relative to sequence
    _ABS_MIN_CAM = 0.1  # px/frame for camera motion
    cam_score = global_flow_mag + ego_motion_residual
    if cam_score.max() <= _ABS_MIN_CAM:
        labels[:, 5] = 0  # static camera
    else:
        threshold_cd = np.percentile(cam_score[cam_score > _ABS_MIN_CAM], 75)
        labels[:, 5] = (cam_score > threshold_cd).astype(np.int8)

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

    All per-sequence data is keyed by a compound key
    ``"{dataset_name}/{seq_name}"`` to avoid collisions across datasets.

    NPZ keys
    --------
    features/{dataset_name}/{seq_name}      float32  (n_frames, n_features)
    feature_names                           object   list[str] as np.array
    feature_units                           object   list[str] as np.array
    labels/{dataset_name}/{seq_name}        int8     (n_frames, n_labels)
    label_names                             object   list[str] as np.array
    iou_trace/{dataset_name}/{seq_name}     float32  (n_frames,)
    bbox_pred/{dataset_name}/{seq_name}     float32  (n_frames, 4)  — (x, y, w, h)
    bbox_gt/{dataset_name}/{seq_name}       float32  (n_frames, 4)
    sequence_name/{dataset_name}/{seq_name} str      short seq identifier (no dataset prefix)
    dataset/{dataset_name}/{seq_name}       str      "uav123" | "visdrone_sot" | "dtb70"
    split/{dataset_name}/{seq_name}         str      "train" | "val" | "diagnostic"
    tracker_version                         str      e.g. "sglatrack_ep0297"
    tracker_config_hash                     str      sha256 of config YAML
    created_at                              str      ISO 8601 datetime (UTC)
    """

    tracker_version: str
    tracker_config_hash: str

    # Per-sequence data (keyed by compound key "dataset_name/seq_name")
    features: Dict[str, np.ndarray] = field(default_factory=dict)
    labels: Dict[str, np.ndarray] = field(default_factory=dict)
    iou_trace: Dict[str, np.ndarray] = field(default_factory=dict)
    bbox_pred: Dict[str, np.ndarray] = field(default_factory=dict)
    bbox_gt: Dict[str, np.ndarray] = field(default_factory=dict)
    dataset: Dict[str, str] = field(default_factory=dict)
    split: Dict[str, str] = field(default_factory=dict)
    # Short sequence name without dataset prefix (for display / diagnostics)
    seq_display_name: Dict[str, str] = field(default_factory=dict)

    def add_sequence(
        self,
        seq_name: str,
        dataset_name: str,
        split: str,
        features: np.ndarray,
        labels: np.ndarray,
        iou_trace: np.ndarray,
        bbox_pred: np.ndarray,
        bbox_gt: np.ndarray,
    ) -> None:
        """Register one sequence into this dataset.

        Parameters
        ----------
        seq_name:
            Short sequence identifier, e.g. "person1_s1" or "uav0000164".
        dataset_name:
            One of "uav123", "visdrone_sot", "dtb70".
        split:
            One of "train", "val", "diagnostic".
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
        """
        assert dataset_name in {"uav123", "visdrone_sot", "dtb70"}, (
            f"Unknown dataset '{dataset_name}'"
        )
        assert split in {"train", "val", "diagnostic"}, (
            f"Unknown split '{split}'"
        )
        assert features.shape == (len(iou_trace), N_FEATURES), (
            f"{dataset_name}/{seq_name}: features shape {features.shape} "
            f"expected ({len(iou_trace)}, {N_FEATURES})"
        )
        assert labels.shape == (len(iou_trace), N_LABELS), (
            f"{dataset_name}/{seq_name}: labels shape {labels.shape} "
            f"expected ({len(iou_trace)}, {N_LABELS})"
        )

        key = f"{dataset_name}/{seq_name}"
        self.features[key] = features.astype(np.float32)
        self.labels[key] = labels.astype(np.int8)
        self.iou_trace[key] = iou_trace.astype(np.float32)
        self.bbox_pred[key] = bbox_pred.astype(np.float32)
        self.bbox_gt[key] = bbox_gt.astype(np.float32)
        self.dataset[key] = dataset_name
        self.split[key] = split
        self.seq_display_name[key] = seq_name

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

        for key in self.features:
            arrays[f"features/{key}"] = self.features[key]
            arrays[f"labels/{key}"] = self.labels[key]
            arrays[f"iou_trace/{key}"] = self.iou_trace[key]
            arrays[f"bbox_pred/{key}"] = self.bbox_pred[key]
            arrays[f"bbox_gt/{key}"] = self.bbox_gt[key]
            arrays[f"sequence_name/{key}"] = np.array(self.seq_display_name[key])
            arrays[f"dataset/{key}"] = np.array(self.dataset[key])
            arrays[f"split/{key}"] = np.array(self.split[key])

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

        # Recover compound keys from features/ entries.
        # Keys have the form "features/{dataset_name}/{seq_name}".
        compound_keys = [
            k[len("features/"):]
            for k in data.files
            if k.startswith("features/")
        ]
        for key in compound_keys:
            ds.features[key] = data[f"features/{key}"]
            ds.labels[key] = data[f"labels/{key}"]
            ds.iou_trace[key] = data[f"iou_trace/{key}"]
            ds.bbox_pred[key] = data[f"bbox_pred/{key}"]
            ds.bbox_gt[key] = data[f"bbox_gt/{key}"]
            ds.dataset[key] = str(data[f"dataset/{key}"])
            ds.split[key] = str(data[f"split/{key}"])
            ds.seq_display_name[key] = str(data[f"sequence_name/{key}"])

        return ds

    def summary(self) -> str:
        """Return a human-readable summary of the dataset."""
        total_frames = sum(v.shape[0] for v in self.features.values())
        by_dataset: Dict[str, int] = {}
        by_split: Dict[str, int] = {}
        for key, feats in self.features.items():
            n = feats.shape[0]
            dname = self.dataset[key]
            sname = self.split[key]
            by_dataset[dname] = by_dataset.get(dname, 0) + n
            by_split[sname] = by_split.get(sname, 0) + n
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
# Per-sequence feature extraction
# ---------------------------------------------------------------------------


@dataclass
class _TruncatedSequence:
    """Minimal sequence wrapper that caps frame count before runner.run()."""
    name: str
    frames: list
    ground_truth: list

    @property
    def init_bbox(self) -> "Any":
        return self.ground_truth[0]


def _compute_bbox_motion_arrays(
    bboxes: list,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Compute speed/accel/scale_delta from GT bounding boxes.

    Used exclusively for label derivation — labels must be GT-derived,
    NOT prediction-derived, to avoid label leakage from tracker drift/jitter.

    Returns:
        speed:       (n_frames,) float32 — normalized centroid speed
        accel:       (n_frames,) float32 — normalized acceleration
        scale_delta: (n_frames,) float32 — fractional scale change (cur/prev - 1)
    """
    import numpy as np  # noqa: PLC0415
    n = len(bboxes)
    speed = np.zeros(n, dtype=np.float32)
    accel = np.zeros(n, dtype=np.float32)
    scale_delta = np.zeros(n, dtype=np.float32)

    prev_speed = 0.0
    for t in range(1, n):
        cur, prv = bboxes[t], bboxes[t - 1]
        diag = max((cur.w ** 2 + cur.h ** 2) ** 0.5, 1.0)
        cx_v = ((cur.x + cur.w / 2) - (prv.x + prv.w / 2)) / diag
        cy_v = ((cur.y + cur.h / 2) - (prv.y + prv.h / 2)) / diag
        speed[t] = (cx_v ** 2 + cy_v ** 2) ** 0.5
        accel[t] = abs(speed[t] - prev_speed)
        prev_speed = float(speed[t])

        cur_scale = (cur.w * cur.h) ** 0.5
        prv_scale = max((prv.w * prv.h) ** 0.5, 1.0)
        scale_delta[t] = cur_scale / prv_scale - 1.0

    return speed, accel, scale_delta


def collect_sequence(
    runner: "Any",
    seq: "Any",
    dataset_name: str,
    split: str,
    max_frames: "int | None" = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Run tracker on one sequence; return (features, labels, iou_trace, pred_arr, gt_arr).

    Parameters
    ----------
    runner:
        Frozen SALTRunner instance.
    seq:
        Dataset sequence object with .ground_truth, .frames, .name.
    dataset_name:
        One of "uav123", "visdrone_sot", "dtb70".
    split:
        One of "train", "val", "diagnostic".
    max_frames:
        Cap on frames for fast debug runs (None = no cap).

    Returns
    -------
    features : float32 (n_frames, N_FEATURES)
    labels   : int8    (n_frames, N_LABELS)
    iou_trace: float32 (n_frames,)
    pred_arr : float32 (n_frames, 4) — x,y,w,h
    gt_arr   : float32 (n_frames, 4) — x,y,w,h
    """
    import cv2

    iou_fn = globals()["iou_fn"]  # injected by collect_dataset

    # ------------------------------------------------------------------
    # A. Run tracker and collect raw data
    # ------------------------------------------------------------------
    gt_bboxes = list(seq.ground_truth)
    frames_list = list(seq.frames)  # re-reads from disk; needed for optical flow
    if max_frames:
        gt_bboxes = gt_bboxes[:max_frames]
        frames_list = frames_list[:max_frames]
    n = len(frames_list)

    # Wrap in _TruncatedSequence so runner.run() never processes extra frames.
    seq_for_run = _TruncatedSequence(
        name=seq.name,
        frames=frames_list,
        ground_truth=gt_bboxes,
    )
    entries = list(runner.run(seq_for_run))

    preds = [e.bbox for e in entries]
    pred_arr = np.array([[b.x, b.y, b.w, b.h] for b in preds], dtype=np.float64)
    gt_arr = np.array([[b.x, b.y, b.w, b.h] for b in gt_bboxes], dtype=np.float64)
    iou_trace = iou_fn(pred_arr, gt_arr).astype(np.float32)

    # ------------------------------------------------------------------
    # B. Feature extraction (28 features, FEATURE_NAMES order)
    # ------------------------------------------------------------------
    feature_matrix = np.zeros((n, N_FEATURES), dtype=np.float32)

    # Features 0-8: score map stats from TelemetryEntry.aux
    for t, entry in enumerate(entries):
        sms = entry.aux.get("score_map_stats", {})
        apce_raw = entry.aux.get("apce_raw", 0.0)
        feature_matrix[t, 0] = apce_raw
        feature_matrix[t, 1] = apce_raw / 256.0          # apce_norm
        feature_matrix[t, 2] = entry.aux.get("psr_raw", 0.0)
        feature_matrix[t, 3] = entry.aux.get("entropy_raw", 0.0)
        feature_matrix[t, 4] = sms.get("peak_margin", 0.0)
        feature_matrix[t, 5] = float(sms.get("peak_width", 0))
        feature_matrix[t, 6] = float(sms.get("n_secondary", 0))
        feature_matrix[t, 7] = sms.get("peak_distance", 0.0)
        feature_matrix[t, 8] = sms.get("heatmap_mass_topk", 0.0)

    # Features 9-14: temporal rolling windows (computed after score map pass)
    for t in range(n):
        apce_w5 = feature_matrix[max(0, t - 5):t, 0]
        apce_w20 = feature_matrix[max(0, t - 20):t, 0]
        feature_matrix[t, 9] = (
            feature_matrix[t, 0] / (apce_w5.mean() + 1e-8)
            if len(apce_w5) > 0 else 1.0
        )
        feature_matrix[t, 10] = (
            feature_matrix[t, 0] / (apce_w20.mean() + 1e-8)
            if len(apce_w20) > 0 else 1.0
        )
        ent_w5 = feature_matrix[max(0, t - 5):t, 3]
        feature_matrix[t, 11] = feature_matrix[t, 3] - (
            ent_w5.mean() if len(ent_w5) > 0 else feature_matrix[t, 3]
        )
        pm_w5 = feature_matrix[max(0, t - 5):t, 4]
        feature_matrix[t, 12] = feature_matrix[t, 4] - (
            pm_w5.mean() if len(pm_w5) > 0 else feature_matrix[t, 4]
        )
        # confirmed_streak: consecutive frames with APCE > 100
        streak = 0
        for k in range(t, -1, -1):
            if feature_matrix[k, 0] > 100.0:
                streak += 1
            else:
                break
        feature_matrix[t, 13] = float(streak)
        # low_conf_streak: consecutive frames with APCE < 50
        low_s = 0
        for k in range(t, -1, -1):
            if feature_matrix[k, 0] < 50.0:
                low_s += 1
            else:
                break
        feature_matrix[t, 14] = float(low_s)

    # Features 15-21: target dynamics
    for t in range(n):
        if t == 0:
            feature_matrix[t, 15:22] = [0, 0, 0, 0, 1.0, 0, 0.5]
            continue
        cur, prv = preds[t], preds[t - 1]
        diag = max((cur.w ** 2 + cur.h ** 2) ** 0.5, 1.0)
        cx_v = ((cur.x + cur.w / 2) - (prv.x + prv.w / 2)) / diag
        cy_v = ((cur.y + cur.h / 2) - (prv.y + prv.h / 2)) / diag
        speed = (cx_v ** 2 + cy_v ** 2) ** 0.5
        if t >= 2:
            pp = preds[t - 2]
            cx_v2 = ((prv.x + prv.w / 2) - (pp.x + pp.w / 2)) / diag
            cy_v2 = ((prv.y + prv.h / 2) - (pp.y + pp.h / 2)) / diag
            accel = abs(speed - (cx_v2 ** 2 + cy_v2 ** 2) ** 0.5)
        else:
            accel = 0.0
        scale_r = (cur.w * cur.h) / max(prv.w * prv.h, 1.0)
        asp_d = (cur.w / max(cur.h, 1e-3)) - (prv.w / max(prv.h, 1e-3))
        h_img, w_img = frames_list[t].shape[:2]
        cx = cur.x + cur.w / 2
        cy = cur.y + cur.h / 2
        search_sz = max(cur.w, cur.h) * 4.0
        dist_border = min(cx, cy, w_img - cx, h_img - cy) / max(search_sz, 1.0)
        feature_matrix[t, 15] = cx_v
        feature_matrix[t, 16] = cy_v
        feature_matrix[t, 17] = speed
        feature_matrix[t, 18] = accel
        feature_matrix[t, 19] = float(scale_r)
        feature_matrix[t, 20] = float(asp_d)
        feature_matrix[t, 21] = float(np.clip(dist_border, 0.0, 1.0))

    # Features 22-27: camera/flow from dense optical flow
    for t in range(n):
        if t == 0 or frames_list[t] is None or frames_list[t - 1] is None:
            continue
        gray_cur = cv2.cvtColor(frames_list[t], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_prev = cv2.cvtColor(frames_list[t - 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_cur, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.hypot(flow[..., 0], flow[..., 1])
        global_flow_mag = float(mag.mean())

        bbox = preds[t]
        h_img, w_img = frames_list[t].shape[:2]
        x1, y1 = max(0, int(bbox.x)), max(0, int(bbox.y))
        x2, y2 = min(w_img, int(bbox.x + bbox.w)), min(h_img, int(bbox.y + bbox.h))
        if x2 > x1 and y2 > y1:
            target_mag = float(mag[y1:y2, x1:x2].mean())
            tflow = flow[y1:y2, x1:x2]
        else:
            target_mag, tflow = global_flow_mag, flow

        ego_residual = abs(target_mag - global_flow_mag)

        gf_mean = flow.mean(axis=(0, 1))
        tf_mean = tflow.mean(axis=(0, 1))
        denom = np.linalg.norm(gf_mean) * np.linalg.norm(tf_mean) + 1e-8
        flow_cos = float(np.dot(gf_mean, tf_mean) / denom)
        flow_iou = float(np.clip((flow_cos + 1) / 2, 0.0, 1.0))

        if t >= 2:
            prev_gmag = feature_matrix[t - 1, 22]
            flow_consistency = 1.0 / (1.0 + abs(global_flow_mag - prev_gmag))
        else:
            flow_consistency = 0.5

        feature_matrix[t, 22] = global_flow_mag
        feature_matrix[t, 23] = target_mag
        feature_matrix[t, 24] = ego_residual
        feature_matrix[t, 25] = flow_iou
        feature_matrix[t, 26] = ego_residual       # flow_residual = ego_residual in v0
        feature_matrix[t, 27] = flow_consistency

    # ------------------------------------------------------------------
    # C. Compute GT-derived labels
    # ------------------------------------------------------------------
    # Labels use GT motion, not predicted-box motion.
    # Predicted-box motion (feature_matrix[:, 17-19]) is fine as an input feature
    # but must NOT be the label source — tracker drift/jitter would leak into labels.
    gt_speed, gt_accel, gt_scale_delta = _compute_bbox_motion_arrays(gt_bboxes)

    labels = compute_labels(
        iou_trace=iou_trace,
        apce_norm=feature_matrix[:, 1],
        speed_norm=gt_speed,
        accel_norm=gt_accel,
        scale_delta=gt_scale_delta,
        global_flow_mag=feature_matrix[:, 22],
        ego_motion_residual=feature_matrix[:, 24],
        peak_margin=feature_matrix[:, 4],
        flow_consistency=feature_matrix[:, 27],
    )
    return feature_matrix, labels, iou_trace, pred_arr.astype(np.float32), gt_arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset loaders and split assignment helpers
# ---------------------------------------------------------------------------


def _get_dataset_loaders(dataset_names: "list[str]") -> "list[tuple[str, Any]]":
    """Load requested datasets using each loader's own root autodetection.

    Passing root=None lets each dataset class read UAV_DATA_ROOT / $HOME
    paths it knows about, rather than forwarding a raw root string that may
    not match the expected sub-directory layout.
    """
    loaders = []
    for name in dataset_names:
        if name == "uav123":
            from uav_tracker.datasets.uav123 import UAV123Dataset
            loaders.append(("uav123", UAV123Dataset(root=None)))
        elif name == "visdrone_sot":
            from uav_tracker.datasets.visdrone_sot import VisDroneSOTDataset
            loaders.append(("visdrone_sot", VisDroneSOTDataset(root=None)))
        elif name == "dtb70":
            from uav_tracker.datasets.dtb70 import DTB70Dataset
            loaders.append(("dtb70", DTB70Dataset(root=None)))
        else:
            raise ValueError(f"Unknown dataset: {name}")
    return loaders


def _assign_split(seq_name: str, dataset_name: str) -> str:
    """Assign train/val/diagnostic split by sequence name.

    Diagnostic sequences get "diagnostic" regardless of dataset.
    All others use a deterministic hash-based 75/25 train/val split so
    the same sequence always lands in the same split across runs.
    """
    if seq_name in DIAGNOSTIC_SEQUENCES:
        return "diagnostic"
    h = int(hashlib.md5(f"{dataset_name}/{seq_name}".encode()).hexdigest(), 16)
    return "train" if (h % 4) != 0 else "val"


# ---------------------------------------------------------------------------
# Full collection loop
# ---------------------------------------------------------------------------


def collect_dataset(
    config_path: str,
    dataset_names: "list[str]",
    output_path: str,
    max_frames: "int | None" = None,
    dry_run: bool = False,
) -> SavedDataset:
    """Full collection loop: run frozen tracker on all sequences, save NPZ.

    Parameters
    ----------
    config_path:
        Path to the SALT/SGLATrack YAML config (frozen).
    dataset_names:
        Subset of ["uav123", "visdrone_sot", "dtb70"] to process.
    output_path:
        Destination for the compressed NPZ file.
    max_frames:
        Cap per-sequence frame count for fast debug runs (None = no cap).
    dry_run:
        If True, iterate sequences and print names without running tracker
        or writing output.

    Returns
    -------
    SavedDataset populated with all collected data (empty dict entries if
    dry_run=True).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parents[4] / "src"))

    runner = None
    config_hash = hash_config_file(config_path)
    tracker_version = "sglatrack_ep0297"  # from FROZEN.md

    if not dry_run:
        from uav_tracker.salt_runner import SALTRunner
        from uav_tracker.metrics.success import iou as iou_fn_inner
        globals()["iou_fn"] = iou_fn_inner
        runner = SALTRunner.from_config(config_path)

    ds = SavedDataset(
        tracker_version=tracker_version,
        tracker_config_hash=config_hash,
    )

    dataset_loaders = _get_dataset_loaders(dataset_names)

    for dataset_name, dataset in dataset_loaders:
        seqs = list(dataset)
        print(f"[{dataset_name}] {len(seqs)} sequences")
        for seq in seqs:
            seq_key = f"{dataset_name}/{seq.name}"
            split = _assign_split(seq.name, dataset_name)
            print(f"  {seq_key} split={split}", flush=True)
            if dry_run:
                continue
            try:
                features, labels, iou_trace, pred_arr, gt_arr = collect_sequence(
                    runner, seq, dataset_name, split, max_frames=max_frames,
                )
                ds.add_sequence(
                    seq_name=seq.name,
                    dataset_name=dataset_name,
                    split=split,
                    features=features,
                    labels=labels,
                    iou_trace=iou_trace,
                    bbox_pred=pred_arr.astype(np.float32),
                    bbox_gt=gt_arr.astype(np.float32),
                )
            except Exception as exc:
                print(f"  ERROR {seq_key}: {exc}", file=sys.stderr)
                continue

    if not dry_run:
        ds.save(output_path)
        print(f"Saved {output_path}")
        print(ds.summary())
    return ds


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Collect SALT-RD features from frozen tracker runs.

    Usage::

        python collect_features.py --config configs/prod/salt.yaml \\
                                   --datasets uav123 visdrone_sot dtb70 \\
                                   --output saltr/data/salt_rd_v0.npz
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect SALT-RD features from frozen SGLATrack/SALTRunner runs."
    )
    parser.add_argument(
        "--config",
        default="configs/prod/salt.yaml",
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
        "--dry-run",
        action="store_true",
        help="Print sequences without running tracker or writing output.",
    )

    args = parser.parse_args()
    collect_dataset(
        config_path=args.config,
        dataset_names=args.datasets,
        output_path=args.output,
        max_frames=args.max_frames,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
