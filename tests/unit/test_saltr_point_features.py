"""Unit tests for saltr/src/salt_r/teachers/point_features.py and cotracker3_export.py.

All tests use synthetic numpy arrays only — no CoTracker3 import at test time.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures: synthetic point track data
# ---------------------------------------------------------------------------

def _make_tracks(T: int = 5, P: int = 9, rng_seed: int = 0) -> tuple:
    """Generate synthetic (T, P, 2) tracks with full visibility."""
    rng = np.random.default_rng(rng_seed)
    # Points at [10..90, 10..90] range
    base = rng.uniform(10, 90, (P, 2)).astype(np.float32)
    tracks = np.stack([
        base + rng.normal(0, 0.5, (P, 2)).astype(np.float32)
        for _ in range(T)
    ])  # (T, P, 2)
    visibility = np.ones((T, P), dtype=bool)
    # pred_bboxes covering the point cloud
    bbox = np.array([[5.0, 5.0, 95.0, 95.0]] * T, dtype=np.float32)
    return tracks, visibility, bbox


# ---------------------------------------------------------------------------
# Test 1: compute_point_features runs without error on single frame
# ---------------------------------------------------------------------------

def test_compute_point_features_basic():
    """compute_point_features runs on synthetic (1,9,2) tracks without error."""
    from salt_r.teachers.point_features import compute_point_features, POINT_FEATURE_NAMES

    tracks, vis, bbox = _make_tracks(T=1, P=9)
    feats = compute_point_features(tracks, vis, bbox, t=0)
    assert feats.shape == (len(POINT_FEATURE_NAMES),), \
        f"Expected ({len(POINT_FEATURE_NAMES)},) got {feats.shape}"
    assert feats.dtype == np.float32
    # At least some features should be finite
    n_finite = int(np.isfinite(feats).sum())
    assert n_finite >= 3, f"Expected at least 3 finite features, got {n_finite}"


# ---------------------------------------------------------------------------
# Test 2: pt_visible_ratio computes correct fraction
# ---------------------------------------------------------------------------

def test_pt_visible_ratio_correct_fraction():
    """pt_visible_ratio is correct when some points are NaN/invisible."""
    from salt_r.teachers.point_features import POINT_FEATURE_NAMES, compute_point_features

    P = 10
    T = 1
    tracks = np.ones((T, P, 2), dtype=np.float32) * 50.0
    vis = np.zeros((T, P), dtype=float)
    vis[0, :6] = 1.0   # 6 visible, 4 invisible
    bbox = np.array([[0.0, 0.0, 100.0, 100.0]] * T, dtype=np.float32)

    feats = compute_point_features(tracks, vis, bbox, t=0)
    idx = POINT_FEATURE_NAMES.index("pt_visible_ratio")
    assert abs(feats[idx] - 0.6) < 1e-5, \
        f"Expected pt_visible_ratio=0.6, got {feats[idx]}"


# ---------------------------------------------------------------------------
# Test 3: pt_inside_pred_ratio computes correct fraction
# ---------------------------------------------------------------------------

def test_pt_inside_pred_ratio_correct_fraction():
    """pt_inside_pred_ratio is correct when only some points inside bbox."""
    from salt_r.teachers.point_features import POINT_FEATURE_NAMES, compute_point_features

    P = 8
    T = 1
    # 4 points at (50, 50) = inside bbox [0,0,100,100]
    # 4 points at (200, 200) = outside bbox
    tracks = np.zeros((T, P, 2), dtype=np.float32)
    tracks[0, :4] = [50.0, 50.0]
    tracks[0, 4:] = [200.0, 200.0]
    vis = np.ones((T, P), dtype=float)
    bbox = np.array([[0.0, 0.0, 100.0, 100.0]] * T, dtype=np.float32)

    feats = compute_point_features(tracks, vis, bbox, t=0)
    idx = POINT_FEATURE_NAMES.index("pt_inside_pred_ratio")
    assert abs(feats[idx] - 0.5) < 1e-5, \
        f"Expected pt_inside_pred_ratio=0.5, got {feats[idx]}"


# ---------------------------------------------------------------------------
# Test 4: pt_split_score — high when one cluster, low when two clusters
# ---------------------------------------------------------------------------

def test_pt_split_score_one_vs_two_clusters():
    """pt_split_score is high for one cluster and low for two well-separated clusters."""
    from salt_r.teachers.point_features import POINT_FEATURE_NAMES, compute_point_features

    P = 12
    T = 1
    bbox = np.array([[0.0, 0.0, 200.0, 200.0]] * T, dtype=np.float32)
    vis = np.ones((T, P), dtype=float)

    # One tight cluster
    tracks_one = np.ones((T, P, 2), dtype=np.float32) * 50.0
    tracks_one[0] += np.random.default_rng(0).normal(0, 0.5, (P, 2)).astype(np.float32)
    feats_one = compute_point_features(tracks_one, vis, bbox, t=0)

    # Two well-separated clusters: 6 at (10,10), 6 at (190,190)
    tracks_two = np.zeros((T, P, 2), dtype=np.float32)
    tracks_two[0, :6] = [10.0, 10.0]
    tracks_two[0, 6:] = [190.0, 190.0]
    feats_two = compute_point_features(tracks_two, vis, bbox, t=0)

    idx = POINT_FEATURE_NAMES.index("pt_split_score")
    score_one = feats_one[idx]
    score_two = feats_two[idx]

    assert np.isfinite(score_one), f"split_score for one cluster should be finite"
    assert np.isfinite(score_two), f"split_score for two clusters should be finite"
    # One cluster → high split_score (within ≈ total variance)
    # Two clusters → low split_score (within << total variance)
    assert score_one > score_two, \
        f"One cluster score ({score_one:.3f}) should exceed two-cluster score ({score_two:.3f})"


# ---------------------------------------------------------------------------
# Test 5: pt_forward_backward_error — zero for perfect consistency
# ---------------------------------------------------------------------------

def test_pt_forward_backward_error_zero_for_perfect():
    """pt_forward_backward_error is zero (or near zero) for perfectly stationary points."""
    from salt_r.teachers.point_features import POINT_FEATURE_NAMES, compute_point_features

    P = 6
    T = 3
    # All points perfectly stationary
    tracks = np.ones((T, P, 2), dtype=np.float32) * 50.0
    vis = np.ones((T, P), dtype=float)
    bbox = np.array([[0.0, 0.0, 100.0, 100.0]] * T, dtype=np.float32)

    # At t=1 (has previous frame)
    feats = compute_point_features(tracks, vis, bbox, t=1)
    idx = POINT_FEATURE_NAMES.index("pt_forward_backward_error")
    # Stationary tracks: displacement = 0 → error = 0
    assert abs(feats[idx]) < 1e-5, \
        f"Expected near-zero fwd-bwd error for stationary tracks, got {feats[idx]}"


# ---------------------------------------------------------------------------
# Test 6: compute_point_teacher_labels — correct keys and shapes
# ---------------------------------------------------------------------------

def test_compute_point_teacher_labels_shapes():
    """compute_point_teacher_labels returns correct keys with shape (T,)."""
    from salt_r.teachers.point_features import compute_point_teacher_labels

    T = 10
    P = 9
    tracks, vis, pred_bboxes = _make_tracks(T=T, P=P)
    gt_bboxes = pred_bboxes.copy()
    iou_trace = np.ones(T, dtype=np.float32) * 0.8  # all good tracking

    result = compute_point_teacher_labels(
        tracks, vis, pred_bboxes, gt_bboxes, iou_trace
    )

    expected_keys = {"point_consistency_good", "point_identity_break", "point_recoverable"}
    assert set(result.keys()) == expected_keys, \
        f"Expected keys {expected_keys}, got {set(result.keys())}"

    for key, arr in result.items():
        assert arr.shape == (T,), f"{key}: expected shape ({T},), got {arr.shape}"
        assert arr.dtype == bool, f"{key}: expected bool dtype, got {arr.dtype}"


# ---------------------------------------------------------------------------
# Test 7: sample_query_points — correct count inside bbox
# ---------------------------------------------------------------------------

def test_sample_query_points_inside_bbox():
    """sample_query_points returns points inside the gt_bbox."""
    from salt_r.teachers.cotracker3_export import sample_query_points

    gt_bbox = np.array([10.0, 20.0, 50.0, 60.0])  # 40x40 pixels
    pts = sample_query_points(gt_bbox)

    assert pts.ndim == 2, f"Expected 2D array, got shape {pts.shape}"
    assert pts.shape[1] == 2, f"Expected (P, 2), got {pts.shape}"
    assert len(pts) >= 4, f"Expected at least 4 query points, got {len(pts)}"
    assert pts.dtype == np.float32

    # All points should be inside (or near) the bbox
    x1, y1, x2, y2 = gt_bbox
    inside_x = (pts[:, 0] >= x1) & (pts[:, 0] <= x2)
    inside_y = (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
    n_inside = int((inside_x & inside_y).sum())
    assert n_inside == len(pts), \
        f"Expected all {len(pts)} points inside bbox, got {n_inside}"


# ---------------------------------------------------------------------------
# Test 8: No CoTracker3 import at test time (gated import)
# ---------------------------------------------------------------------------

def test_no_cotracker3_at_import_time():
    """Importing point_features and cotracker3_export does not trigger CoTracker3 library import.

    The actual CoTracker3 library (cotracker, cotracker3) must NOT be imported
    as a side effect of importing our wrappers — the import is gated behind
    run_cotracker3_on_sequence which uses try/except.
    """
    import sys

    # These should import without error even if cotracker3 is not installed
    import salt_r.teachers.point_features  # noqa: F401
    import salt_r.teachers.cotracker3_export  # noqa: F401

    # Only the external CoTracker3 library should be absent — not our own wrapper modules.
    # Check that neither 'cotracker' nor 'cotracker3' top-level packages are loaded.
    external_cotracker = [
        k for k in sys.modules
        if k in ("cotracker", "cotracker3") or k.startswith(("cotracker.", "cotracker3."))
    ]
    assert len(external_cotracker) == 0, \
        f"External CoTracker3 library should not be imported at module load time, found: {external_cotracker}"


# ---------------------------------------------------------------------------
# Test 9: compute_point_features_sequence — shape (T, F)
# ---------------------------------------------------------------------------

def test_compute_point_features_sequence_shape():
    """compute_point_features_sequence returns (T, F) array."""
    from salt_r.teachers.point_features import (
        compute_point_features_sequence,
        POINT_FEATURE_NAMES,
    )

    T, P = 8, 9
    tracks, vis, bbox = _make_tracks(T=T, P=P)
    out = compute_point_features_sequence(tracks, vis, bbox)
    assert out.shape == (T, len(POINT_FEATURE_NAMES)), \
        f"Expected ({T}, {len(POINT_FEATURE_NAMES)}), got {out.shape}"
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Test 10: compute_point_teacher_labels — identity_break when IoU<0.3 and points outside
# ---------------------------------------------------------------------------

def test_point_identity_break_triggers():
    """point_identity_break is True when IoU<0.3 and points are outside pred bbox."""
    from salt_r.teachers.point_features import compute_point_teacher_labels

    T = 5
    P = 9
    # Points all at (200, 200) — far outside pred_bboxes [0,0,100,100]
    tracks = np.ones((T, P, 2), dtype=np.float32) * 200.0
    vis = np.ones((T, P), dtype=float)
    pred_bboxes = np.array([[0.0, 0.0, 100.0, 100.0]] * T, dtype=np.float32)
    gt_bboxes = pred_bboxes.copy()
    # Low IoU = tracking failed, points are outside
    iou_trace = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)

    result = compute_point_teacher_labels(
        tracks, vis, pred_bboxes, gt_bboxes, iou_trace
    )
    # identity_break should be True for frames with low IoU + points outside
    assert result["point_identity_break"].any(), \
        "Expected at least one identity_break=True when IoU<0.3 and points outside bbox"
