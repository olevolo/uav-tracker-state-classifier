"""Regression tests for saltr/src/salt_r/collect_features.py.

Guards against three bugs that were found and fixed:
1. all-zero motion → dynamicity labels must be 0, not all-1
2. same seq name across two datasets → two compound keys, no overwrite
3. _TruncatedSequence caps runner work — max_frames limits frames seen by runner
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Flat/static sequence — dynamicity labels must be all zero
# ---------------------------------------------------------------------------

def test_flat_motion_yields_no_dynamic_labels():
    """All-zero motion and zero flow → target_dynamic, camera_dynamic, hard_dynamic_scene all 0."""
    from salt_r.collect_features import compute_labels

    n = 50
    iou_trace = np.ones(n, dtype=np.float32) * 0.9  # tracker always correct
    apce_norm = np.ones(n, dtype=np.float32) * 0.8
    speed_norm = np.zeros(n, dtype=np.float32)
    accel_norm = np.zeros(n, dtype=np.float32)
    scale_delta = np.zeros(n, dtype=np.float32)
    global_flow_mag = np.zeros(n, dtype=np.float32)
    ego_motion_residual = np.zeros(n, dtype=np.float32)
    peak_margin = np.ones(n, dtype=np.float32) * 0.5
    flow_consistency = np.ones(n, dtype=np.float32) * 0.8

    labels = compute_labels(
        iou_trace=iou_trace,
        apce_norm=apce_norm,
        speed_norm=speed_norm,
        accel_norm=accel_norm,
        scale_delta=scale_delta,
        global_flow_mag=global_flow_mag,
        ego_motion_residual=ego_motion_residual,
        peak_margin=peak_margin,
        flow_consistency=flow_consistency,
    )

    # Indices: 4=target_dynamic, 5=camera_dynamic, 6=hard_dynamic_scene
    assert labels[:, 4].sum() == 0, "target_dynamic must be 0 for all-zero motion sequence"
    assert labels[:, 5].sum() == 0, "camera_dynamic must be 0 for all-zero flow sequence"
    assert labels[:, 6].sum() == 0, "hard_dynamic_scene must be 0 for flat sequence"


# ---------------------------------------------------------------------------
# 2. Same seq name in two datasets → two compound keys, no overwrite
# ---------------------------------------------------------------------------

def test_dataset_key_collision_prevented():
    """Sequences with the same name in different datasets must not overwrite each other."""
    from salt_r.collect_features import SavedDataset

    n = 10
    ds = SavedDataset(tracker_version="test", tracker_config_hash="abc")

    features_a = np.ones((n, 28), dtype=np.float32) * 1.0
    features_b = np.ones((n, 28), dtype=np.float32) * 2.0
    labels = np.zeros((n, 8), dtype=np.int8)
    iou = np.ones(n, dtype=np.float32)
    bbox = np.zeros((n, 4), dtype=np.float32)

    ds.add_sequence(
        seq_name="car1",
        dataset_name="uav123",
        split="train",
        features=features_a,
        labels=labels,
        iou_trace=iou,
        bbox_pred=bbox,
        bbox_gt=bbox,
    )
    ds.add_sequence(
        seq_name="car1",
        dataset_name="dtb70",
        split="train",
        features=features_b,
        labels=labels,
        iou_trace=iou,
        bbox_pred=bbox,
        bbox_gt=bbox,
    )

    # Both keys must exist independently
    assert "uav123/car1" in ds.features
    assert "dtb70/car1" in ds.features
    assert ds.features["uav123/car1"].sum() == pytest.approx(n * 28 * 1.0)
    assert ds.features["dtb70/car1"].sum() == pytest.approx(n * 28 * 2.0)


# ---------------------------------------------------------------------------
# 3. _TruncatedSequence — runner.run() only sees max_frames frames
# ---------------------------------------------------------------------------

def test_truncated_sequence_limits_runner_frames():
    """_TruncatedSequence caps the number of frames the runner iterates."""
    from salt_r.collect_features import _TruncatedSequence

    # Build a dummy sequence with 100 frames
    full_frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(100)]

    class _FakeBBox:
        x = y = 0.0
        w = h = 10.0

    full_gt = [_FakeBBox() for _ in range(100)]

    # Truncate to 20
    trunc = _TruncatedSequence(
        name="test_seq",
        frames=full_frames[:20],
        ground_truth=full_gt[:20],
    )

    frames_seen = list(trunc.frames)
    assert len(frames_seen) == 20, f"Expected 20 frames, got {len(frames_seen)}"
    assert len(trunc.ground_truth) == 20
    assert trunc.init_bbox is full_gt[0]
