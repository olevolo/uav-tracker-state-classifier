"""Tests for bbox convention utilities and synthetic-fallback guard in cotracker3_export."""

from __future__ import annotations

import numpy as np
import pytest

from salt_r.collect_features import xywh_to_xyxy, xyxy_to_xywh
from salt_r.teachers.cotracker3_export import process_sequence_to_sidecar, save_sidecar_npz


# ---------------------------------------------------------------------------
# xywh_to_xyxy
# ---------------------------------------------------------------------------


def test_xywh_to_xyxy_single():
    result = xywh_to_xyxy([10, 20, 5, 8])
    expected = np.array([10, 20, 15, 28], dtype=np.float64)
    np.testing.assert_array_almost_equal(result, expected)


def test_xywh_to_xyxy_batch():
    bboxes = np.array([
        [0, 0, 10, 20],
        [5, 5, 4, 6],
        [100, 200, 30, 40],
    ], dtype=np.float64)
    result = xywh_to_xyxy(bboxes)
    expected = np.array([
        [0, 0, 10, 20],
        [5, 5, 9, 11],
        [100, 200, 130, 240],
    ], dtype=np.float64)
    np.testing.assert_array_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# xyxy_to_xywh
# ---------------------------------------------------------------------------


def test_xyxy_to_xywh_single():
    result = xyxy_to_xywh([10, 20, 15, 28])
    expected = np.array([10, 20, 5, 8], dtype=np.float64)
    np.testing.assert_array_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip():
    rng = np.random.default_rng(0)
    # Generate random valid bboxes: x,y in [0,100], w,h in [1,50]
    xy = rng.uniform(0, 100, (20, 2))
    wh = rng.uniform(1, 50, (20, 2))
    bboxes = np.concatenate([xy, wh], axis=1)
    recovered = xyxy_to_xywh(xywh_to_xyxy(bboxes))
    np.testing.assert_array_almost_equal(recovered, bboxes, decimal=10)


# ---------------------------------------------------------------------------
# Synthetic fallback guard
# ---------------------------------------------------------------------------


def _minimal_gt_bboxes(T: int = 5) -> np.ndarray:
    """Return a (T, 4) array of xyxy bboxes for use in tests."""
    return np.tile(np.array([[10.0, 10.0, 50.0, 50.0]]), (T, 1))


def test_synthetic_fallback_raises_by_default():
    """process_sequence_to_sidecar must raise RuntimeError when CoTracker3 is
    unavailable and allow_synthetic is not set (defaults to False)."""
    gt = _minimal_gt_bboxes()
    T = len(gt)
    pred = gt.copy()
    iou = np.ones(T, dtype=np.float32)

    with pytest.raises(RuntimeError, match="allow_synthetic=False"):
        process_sequence_to_sidecar(
            seq_key="test/seq",
            frames_or_path=None,   # no frames → CoTracker3 cannot run
            gt_bboxes=gt,
            pred_bboxes=pred,
            iou_trace=iou,
            # allow_synthetic defaults to False
        )


def test_synthetic_fallback_ok_when_opted_in():
    """With allow_synthetic=True the function must succeed and report
    teacher_model == 'synthetic'."""
    gt = _minimal_gt_bboxes()
    T = len(gt)
    pred = gt.copy()
    iou = np.ones(T, dtype=np.float32)

    result = process_sequence_to_sidecar(
        seq_key="test/seq",
        frames_or_path=None,
        gt_bboxes=gt,
        pred_bboxes=pred,
        iou_trace=iou,
        allow_synthetic=True,
    )
    assert result["teacher_model"] == "synthetic"


def test_save_sidecar_npz_persists_teacher_model(tmp_path):
    """save_sidecar_npz must store teacher_model per sequence in the NPZ."""
    entry = {
        "seq_key": "test/seq1",
        "teacher_model": "synthetic",
        "point_tracks": np.zeros((3, 2, 2), dtype=np.float32),
        "point_visibility": np.ones((3, 2), dtype=bool),
        "point_features": np.zeros((3, 1), dtype=np.float32),
        "teacher_labels": {},
    }
    out_path = str(tmp_path / "test_sidecar_provenance.npz")
    save_sidecar_npz([entry], out_path)

    npz = np.load(out_path, allow_pickle=True)
    assert "teacher_model/test/seq1" in npz.files
    assert str(npz["teacher_model/test/seq1"]) == "synthetic"
