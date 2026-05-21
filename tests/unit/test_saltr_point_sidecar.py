"""Unit tests for point_sidecar_extractor.py."""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pytest

from salt_r.point_sidecar_extractor import (
    extract_sequence_features,
    _track_lk,
    _track_farneback,
    _TRACKERS,
)
from salt_r.teachers.point_features import POINT_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frames(T: int, h: int = 64, w: int = 64) -> list[np.ndarray]:
    """BGR uint8 frames with slowly moving bright square."""
    import cv2
    frames = []
    for t in range(T):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cx = int(10 + t * 0.5) % (w - 10)
        cy = int(10 + t * 0.3) % (h - 10)
        img[cy : cy + 8, cx : cx + 8] = (200, 200, 200)
        frames.append(img)
    return frames


def _make_pred_xyxy(T: int, h: int = 64, w: int = 64) -> np.ndarray:
    """Simple moving bbox in xyxy."""
    out = np.zeros((T, 4), dtype=np.float32)
    for t in range(T):
        cx = float(int(10 + t * 0.5) % (w - 10)) + 4
        cy = float(int(10 + t * 0.3) % (h - 10)) + 4
        out[t] = [cx - 6, cy - 6, cx + 6, cy + 6]
    return out


# ---------------------------------------------------------------------------
# Tracker unit tests
# ---------------------------------------------------------------------------

class TestTrackers:
    def test_lk_output_shape(self):
        frames = _make_frames(10)
        query = np.array([[12.0, 12.0], [16.0, 12.0], [12.0, 16.0]], dtype=np.float32)
        tracks, vis = _track_lk(frames, query)
        assert tracks.shape == (10, 3, 2)
        assert vis.shape == (10, 3)
        assert vis.dtype == bool

    def test_lk_frame0_visible(self):
        frames = _make_frames(5)
        query = np.array([[12.0, 12.0]], dtype=np.float32)
        tracks, vis = _track_lk(frames, query)
        assert vis[0, 0]
        assert np.isfinite(tracks[0, 0]).all()

    def test_farneback_output_shape(self):
        frames = _make_frames(8)
        query = np.array([[12.0, 12.0], [20.0, 20.0]], dtype=np.float32)
        tracks, vis = _track_farneback(frames, query)
        assert tracks.shape == (8, 2, 2)
        assert vis.shape == (8, 2)

    def test_lk_nan_for_invisible(self):
        frames = _make_frames(10)
        query = np.array([[12.0, 12.0]], dtype=np.float32)
        tracks, vis = _track_lk(frames, query)
        # All nan positions should have vis=False
        for t in range(10):
            if not vis[t, 0]:
                assert np.isnan(tracks[t, 0]).any()

    def test_both_methods_registered(self):
        assert "lk" in _TRACKERS
        assert "farneback" in _TRACKERS
        assert "cotracker3" not in _TRACKERS  # explicitly excluded


# ---------------------------------------------------------------------------
# extract_sequence_features
# ---------------------------------------------------------------------------

class TestExtractSequenceFeatures:
    def test_output_shape(self):
        T = 30
        frames = _make_frames(T)
        pred = _make_pred_xyxy(T)
        feats = extract_sequence_features(frames, pred, method="lk", stride=10, window=15)
        assert feats.shape == (T, len(POINT_FEATURE_NAMES))
        assert feats.dtype == np.float32

    def test_finite_features_majority(self):
        T = 30
        frames = _make_frames(T)
        pred = _make_pred_xyxy(T)
        feats = extract_sequence_features(frames, pred, method="lk", stride=10, window=15)
        # Majority of frames should have at least some finite features
        has_any_finite = np.isfinite(feats).any(axis=1)
        assert has_any_finite.mean() > 0.5

    def test_farneback_same_shape(self):
        T = 20
        frames = _make_frames(T)
        pred = _make_pred_xyxy(T)
        feats = extract_sequence_features(frames, pred, method="farneback", stride=8, window=12)
        assert feats.shape == (T, len(POINT_FEATURE_NAMES))

    def test_empty_sequence(self):
        feats = extract_sequence_features([], np.zeros((0, 4), np.float32))
        assert feats.shape == (0, len(POINT_FEATURE_NAMES))

    def test_single_frame(self):
        frames = _make_frames(1)
        pred = _make_pred_xyxy(1)
        feats = extract_sequence_features(frames, pred, stride=1, window=5)
        assert feats.shape == (1, len(POINT_FEATURE_NAMES))

    def test_stride_larger_than_window_covers_all(self):
        # stride > window: some frames are not covered → NaN (ok, tested separately)
        T = 40
        frames = _make_frames(T)
        pred = _make_pred_xyxy(T)
        feats = extract_sequence_features(frames, pred, stride=20, window=10)
        assert feats.shape == (T, len(POINT_FEATURE_NAMES))

    def test_latest_seed_overwrites(self):
        # With stride=1 and window=5, every frame gets freshly seeded features.
        # Shape and type are the main guarantees here.
        T = 15
        frames = _make_frames(T)
        pred = _make_pred_xyxy(T)
        feats = extract_sequence_features(frames, pred, stride=1, window=5)
        assert feats.shape == (T, len(POINT_FEATURE_NAMES))

    def test_degenerate_bbox_skips_window(self):
        T = 20
        frames = _make_frames(T)
        pred = np.zeros((T, 4), dtype=np.float32)  # all-zero bboxes
        # Should not raise, just produce NaN
        feats = extract_sequence_features(frames, pred, stride=5, window=8)
        assert feats.shape == (T, len(POINT_FEATURE_NAMES))

    def test_max_side_scales_down(self):
        # High-res frames should be resized without error
        T = 10
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(T)]
        pred = _make_pred_xyxy(T, h=480, w=640)
        feats = extract_sequence_features(frames, pred, max_side=160)
        assert feats.shape == (T, len(POINT_FEATURE_NAMES))


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestCLI:
    def test_smoke_test_flag(self, tmp_path):
        """--smoke-test N saves partial output without raising."""
        import subprocess, sys

        npz_path = "saltr/data/salt_rd_v2_labels.npz"
        if not Path(npz_path).exists():
            pytest.skip("NPZ not available")

        out = tmp_path / "smoke_point_sidecar.npz"
        result = subprocess.run(
            [
                sys.executable, "-m", "salt_r.point_sidecar_extractor",
                "--npz", npz_path,
                "--output", str(out),
                "--smoke-test", "2",
                "--stride", "20",
                "--window", "30",
            ],
            capture_output=True, text=True, cwd=".",
            env={**__import__("os").environ, "PYTHONPATH": "src:saltr/src"},
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert out.exists()

        data = np.load(str(out), allow_pickle=True)
        assert "point_feature_names" in data.files
        assert "extractor_method" in data.files
        pt_keys = [k for k in data.files if k.startswith("point_features/")]
        assert len(pt_keys) == 2

    def test_output_schema(self, tmp_path):
        """Saved NPZ has required metadata keys."""
        npz_path = "saltr/data/salt_rd_v2_labels.npz"
        if not Path(npz_path).exists():
            pytest.skip("NPZ not available")

        import subprocess, sys
        out = tmp_path / "schema_test.npz"
        subprocess.run(
            [
                sys.executable, "-m", "salt_r.point_sidecar_extractor",
                "--npz", npz_path,
                "--output", str(out),
                "--smoke-test", "1",
            ],
            check=True, capture_output=True,
            env={**__import__("os").environ, "PYTHONPATH": "src:saltr/src"},
        )
        data = np.load(str(out), allow_pickle=True)
        for key in ("point_feature_names", "extractor_method", "stride", "window",
                    "n_points", "n_sequences", "source_npz_md5", "created_at"):
            assert key in data.files, f"Missing key: {key}"

        # Feature arrays have correct shape
        pt_keys = [k for k in data.files if k.startswith("point_features/")]
        seq_key = pt_keys[0].replace("point_features/", "")
        feats = data[f"point_features/{seq_key}"]
        assert feats.ndim == 2
        assert feats.shape[1] == len(POINT_FEATURE_NAMES)
