"""Unit tests for salt_r.oracle_actions — Phase 5 oracle reinit label generator.

Tests:
1. Utility is negative for stable good-IoU frames (no need to reinit)
2. Utility is positive for frames followed by IoU improvement after failure
3. No TSA imports in the module
4. Output NPZ has expected arrays with matching lengths
"""
from __future__ import annotations

import ast
import inspect
import tempfile
from pathlib import Path

import numpy as np
import pytest

import salt_r.oracle_actions as oa


# ---------------------------------------------------------------------------
# Helper: build a synthetic sequence data dict
# ---------------------------------------------------------------------------

def _make_seq_data(iou_trace: np.ndarray, n_feat: int = 28) -> dict:
    """Return a minimal sequence data dict for testing."""
    n = len(iou_trace)
    rng = np.random.default_rng(42)
    return {
        "iou_trace": iou_trace.astype(np.float32),
        "bbox_pred": rng.uniform(0, 100, (n, 4)).astype(np.float32),
        "bbox_gt":   rng.uniform(0, 100, (n, 4)).astype(np.float32),
        "features":  rng.standard_normal((n, n_feat)).astype(np.float32),
        "labels":    np.zeros((n, 14), dtype=np.float32),
        "split":     "val",
        "dataset":   "uav123",
    }


# ---------------------------------------------------------------------------
# Test 1: Utility is negative for stable good-IoU frames
# ---------------------------------------------------------------------------

def test_utility_negative_for_stable_good_iou():
    """Stable tracking: IoU=0.8 for all frames → utility should be <= 0.

    When tracking is already excellent and future IoU is identical (no gain),
    the wrong_reinit_penalty fires and pushes utility below zero.
    """
    n = 120
    # Perfect tracking: IoU=0.85 everywhere
    iou = np.full(n, 0.85, dtype=np.float32)

    # Frame 50 is well inside the sequence
    t = 50
    utility, gain_20, gain_50 = oa.compute_utility(iou, t)

    # future_iou_gain_20 == 0, wrong_reinit_penalty == 1.0 (iou >= 0.5, gain < 0)
    # Actually gain_20 = 0, which is not < 0, so wrong_reinit_penalty = 0.
    # But fragmentation_penalty = 0.05 (gain_20 < 0.01).
    # utility = 0 + 0 - 0 - 0.05 = -0.05
    assert utility < 0.0, (
        f"Expected utility < 0 for stable good-IoU tracking, got {utility:.4f}"
    )

    # Label: should be reject=1, reinit=0
    label_reinit, label_reject = oa.derive_labels(float(iou[t]), utility)
    assert label_reinit == 0, "Should not reinit when tracking is stable"
    assert label_reject == 1, "Should reject reinit when IoU is high"


# ---------------------------------------------------------------------------
# Test 2: Utility is positive for recovery after failure
# ---------------------------------------------------------------------------

def test_utility_positive_for_failure_followed_by_recovery():
    """Lost target then recovery: utility should be > 0 at the failure frame.

    At frame t=50, IoU drops to 0.1 (tracking lost).  After reinit (oracle),
    future IoU would rise back to 0.8.  The utility should be positive.
    """
    n = 120
    # Simulate: good tracking, then lost, then recovery
    iou = np.full(n, 0.75, dtype=np.float32)
    # Drop at t=40 onward
    iou[40:55] = 0.1
    # After t=55 recovery (oracle would have reinitialized around t=50)
    iou[55:] = 0.8

    t = 50  # A frame during failure with recovery in future
    utility, gain_20, gain_50 = oa.compute_utility(iou, t)

    # baseline_iou_recent ≈ mean(iou[30:50]) — mostly lost frames ~0.1
    # but some good frames at start if t-20=30 includes iou[30:40]=0.75
    # gain_20 = mean(iou[51:71]) - baseline
    # The future is a mix of 0.1 (51-54) and 0.8 (55-70) ≈ 0.63
    # baseline ≈ mean(0.75*10 + 0.1*10) / 20 = 0.425
    # gain_20 ≈ 0.63 - 0.425 = +0.2 → positive
    assert utility > 0.03, (
        f"Expected utility > 0.03 for failure+recovery frame, got {utility:.4f}"
    )

    label_reinit, label_reject = oa.derive_labels(float(iou[t]), utility)
    assert label_reinit == 1, "Should reinit when tracking lost and future improves"
    assert label_reject == 0, "Should not reject reinit when IoU is low and utility > 0"


# ---------------------------------------------------------------------------
# Test 3: No TSA imports
# ---------------------------------------------------------------------------

def test_no_tsa_imports():
    """Module must not import from TSA or target_state — production constraint."""
    src = inspect.getsource(oa)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                names = [node.module or ""]
            for name in names:
                name_lower = (name or "").lower()
                assert "tsa" not in name_lower, f"TSA import found: {name}"
                assert "target_state" not in name_lower, (
                    f"TargetState import found: {name}"
                )


# ---------------------------------------------------------------------------
# Test 4: Output NPZ has expected arrays with matching lengths
# ---------------------------------------------------------------------------

def test_output_npz_shape_consistency():
    """build_oracle_dataset must produce arrays with consistent lengths."""
    # Build synthetic data for 3 short sequences
    np.random.seed(0)
    seq_len = 40  # long enough to have some valid frames after edge skip

    all_data = {
        "uav123/test1": _make_seq_data(np.random.uniform(0.3, 0.9, seq_len)),
        "dtb70/test2":  _make_seq_data(np.random.uniform(0.0, 0.5, seq_len)),
        "uav123/test3": _make_seq_data(np.random.uniform(0.6, 1.0, seq_len)),
    }

    arrays = oa.build_oracle_dataset(all_data, edge_skip=oa.EDGE_SKIP)

    # Required arrays
    expected_keys = {
        "sequence_keys",
        "frame_indices",
        "current_iou",
        "utility",
        "future_iou_gain_20",
        "future_iou_gain_50",
        "label_reinit",
        "label_reject",
        "features",
        "splits",
        "datasets",
    }
    for k in expected_keys:
        assert k in arrays, f"Missing array: {k}"

    # All 1-D arrays must have the same length M
    m = len(arrays["sequence_keys"])
    assert m > 0, "Expected at least one record"
    for k, arr in arrays.items():
        if k == "features":
            assert arr.shape == (m, 28), (
                f"features must be (M, 28), got {arr.shape}"
            )
        else:
            assert len(arr) == m, (
                f"Array '{k}' length {len(arr)} != M={m}"
            )

    # Edge skip: valid frames per sequence = seq_len - 2 * EDGE_SKIP
    expected_frames_per_seq = seq_len - 2 * oa.EDGE_SKIP
    assert m == 3 * expected_frames_per_seq, (
        f"Expected {3 * expected_frames_per_seq} records, got {m}"
    )

    # Labels are 0 or 1
    assert set(np.unique(arrays["label_reinit"])).issubset({0, 1})
    assert set(np.unique(arrays["label_reject"])).issubset({0, 1})

    # Features dtype and flow indices zeroed
    assert arrays["features"].dtype == np.float32
    flow_cols = arrays["features"][:, 22:28]
    assert np.all(flow_cols == 0.0), "Flow features (indices 22-27) must be zeroed"


# ---------------------------------------------------------------------------
# Additional: round-trip save/load test
# ---------------------------------------------------------------------------

def test_npz_roundtrip():
    """NPZ save/load round-trip preserves array shapes and values."""
    all_data = {
        "uav123/rt_seq": _make_seq_data(np.linspace(0.0, 1.0, 60).astype(np.float32))
    }
    arrays = oa.build_oracle_dataset(all_data, edge_skip=oa.EDGE_SKIP)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "test_oracle.npz"
        np.savez(str(out_path), **arrays)

        loaded = np.load(str(out_path), allow_pickle=True)
        for k in arrays:
            assert k in loaded.files, f"Key {k!r} missing from saved NPZ"
            np.testing.assert_array_equal(
                loaded[k], arrays[k],
                err_msg=f"Array {k!r} changed after round-trip"
            )
