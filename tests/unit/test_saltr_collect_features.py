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


# ---------------------------------------------------------------------------
# 4. failure_in_10 requires a full 10-frame window (no partial-horizon positives)
# ---------------------------------------------------------------------------

def test_failure_in_10_requires_full_window():
    """Frames without a full 10-frame future window must not be labeled failure_in_10=1.

    Build a 12-frame trace:
    - Frames 0–1: IoU=0.8 (currently OK), followed by 10 frames of IoU=0.0
      (so frames 0 and 1 have mean(iou_next10)=0.0 < 0.3).
    - Frame 0 has t+10=10 <= 12, so a full window exists → must be labeled 1.
    - Frame 1 has t+10=11 <= 12, so a full window exists → must be labeled 1.
    - Frame 2 has t+10=12 > 12, so the slice iou_trace[3:13] only has 9 elements
      → partial window → must NOT be labeled 1 (this is the bug fixed by == 10).
    With the old `> 0` condition frame 2 would have been a false positive because
    its 9-element partial window also has mean < 0.3.
    """
    from salt_r.collect_features import _compute_v2_extra_labels

    # 12-frame trace: frames 0-1 good (IoU 0.8), frames 2-11 failing (IoU 0.0)
    iou_trace = np.array(
        [0.8, 0.8] + [0.0] * 10,
        dtype=np.float32,
    )
    n = len(iou_trace)  # 12

    # Minimal v1 labels — only columns 4 and 5 matter for is_dynamic,
    # set them to 0 so ifd10/20 are irrelevant here.
    labels_v1 = np.zeros((n, 10), dtype=np.int8)

    extra = _compute_v2_extra_labels(labels_v1, iou_trace)
    fi10 = extra[:, 0]  # failure_in_10 column

    # Frames 0 and 1: currently OK (IoU 0.8) and have 10 future failing frames → labeled 1
    assert fi10[0] == 1, "frame 0 has full 10-frame window with mean IoU 0 — should be 1"
    assert fi10[1] == 1, "frame 1 has full 10-frame window with mean IoU 0 — should be 1"

    # Frame 2 onwards: IoU already 0.0 so iou_trace[t] >= 0.5 fails → labeled 0 regardless
    # (also partial window at frame 2 would only contribute 9 elements)
    assert fi10[2] == 0, "frame 2 has IoU 0.0 (not >= 0.5) — should be 0"

    # Verify no false positives exist beyond frame 1
    assert fi10[2:].sum() == 0, "no partial-window or already-failed frames should be labeled 1"

    # Extra sanity: only frames 0 and 1 should be positive
    assert fi10.sum() == 2, f"expected exactly 2 positive frames, got {fi10.sum()}"
