"""Unit tests for SyntheticDataset — determinism, frame counts, bbox validity."""

from __future__ import annotations

import numpy as np
import pytest

from uav_tracker.datasets.synthetic import SyntheticDataset
from uav_tracker.types import BBox


def test_synthetic_dataset_registered() -> None:
    """SyntheticDataset must be discoverable via the DATASETS registry."""
    from uav_tracker.registry import DATASETS
    # Import triggers registration side-effect.
    import uav_tracker.datasets.synthetic  # noqa: F401
    assert "synthetic" in DATASETS


def test_synthetic_three_sequences() -> None:
    """Dataset yields exactly 3 sequences in a fixed order."""
    ds = SyntheticDataset(seed=42)
    seqs = list(ds)
    assert len(seqs) == 3
    assert seqs[0].name == "synthetic_static"
    assert seqs[1].name == "synthetic_linear"
    assert seqs[2].name == "synthetic_oscillating"


def test_synthetic_frame_count() -> None:
    """Each sequence has exactly 60 frames and 60 ground-truth bboxes."""
    ds = SyntheticDataset(seed=42)
    for seq in ds:
        frames = list(seq.frames)
        assert len(frames) == 60, f"{seq.name}: expected 60 frames, got {len(frames)}"
        assert len(seq.ground_truth) == 60, (
            f"{seq.name}: expected 60 gt bboxes, got {len(seq.ground_truth)}"
        )


def test_synthetic_frame_shape() -> None:
    """Frames are uint8 BGR arrays of shape (240, 320, 3)."""
    ds = SyntheticDataset(seed=42)
    for seq in ds:
        frame = list(seq.frames)[0]
        assert frame.dtype == np.uint8, f"{seq.name}: expected uint8, got {frame.dtype}"
        assert frame.shape == (240, 320, 3), (
            f"{seq.name}: expected (240, 320, 3), got {frame.shape}"
        )


def test_synthetic_ground_truth_bbox_type() -> None:
    """Ground-truth entries are BBox instances with positive w and h."""
    ds = SyntheticDataset(seed=42)
    for seq in ds:
        for i, bbox in enumerate(seq.ground_truth):
            assert isinstance(bbox, BBox), (
                f"{seq.name}[{i}]: expected BBox, got {type(bbox)}"
            )
            assert bbox.w > 0, f"{seq.name}[{i}]: w must be > 0"
            assert bbox.h > 0, f"{seq.name}[{i}]: h must be > 0"


def test_synthetic_init_bbox_equals_first_gt() -> None:
    """init_bbox must equal ground_truth[0] for OPE initialisation."""
    ds = SyntheticDataset(seed=42)
    for seq in ds:
        assert seq.init_bbox == seq.ground_truth[0], (
            f"{seq.name}: init_bbox != ground_truth[0]"
        )


def test_synthetic_deterministic_under_seed() -> None:
    """Two datasets with the same seed must produce identical frames."""
    ds_a = SyntheticDataset(seed=7)
    ds_b = SyntheticDataset(seed=7)
    for seq_a, seq_b in zip(ds_a, ds_b):
        frames_a = list(seq_a.frames)
        frames_b = list(seq_b.frames)
        for i, (fa, fb) in enumerate(zip(frames_a, frames_b)):
            assert np.array_equal(fa, fb), (
                f"{seq_a.name}[{i}]: frames differ between identical seeds"
            )


def test_synthetic_different_seeds_differ() -> None:
    """Two datasets with different seeds must produce different frames (noise)."""
    ds_a = SyntheticDataset(seed=1)
    ds_b = SyntheticDataset(seed=2)
    seqs_a = list(ds_a)
    seqs_b = list(ds_b)
    # At least one frame must differ across the two datasets.
    any_diff = False
    for seq_a, seq_b in zip(seqs_a, seqs_b):
        for fa, fb in zip(seq_a.frames, seq_b.frames):
            if not np.array_equal(fa, fb):
                any_diff = True
                break
        if any_diff:
            break
    assert any_diff, "Different seeds produced identical frame data — broken RNG?"


def test_synthetic_filter() -> None:
    """filter() returns only sequences whose attributes are a superset."""
    ds = SyntheticDataset(seed=42)
    static_only = ds.filter({"STATIC"})
    seqs = list(static_only)
    assert len(seqs) == 1
    assert seqs[0].name == "synthetic_static"

    # Empty filter — should return all 3.
    all_seqs = ds.filter(set())
    assert len(list(all_seqs)) == 3


def test_synthetic_static_bbox_constant() -> None:
    """static sequence: all GT bboxes must be identical (no motion)."""
    ds = SyntheticDataset(seed=42)
    static_seq = list(ds)[0]
    first = static_seq.ground_truth[0]
    for bbox in static_seq.ground_truth:
        assert bbox == first, "static sequence has non-constant GT bbox"


def test_synthetic_linear_bbox_monotone() -> None:
    """linear sequence: x-coordinate of GT bbox must be monotonically increasing."""
    ds = SyntheticDataset(seed=42)
    linear_seq = list(ds)[1]
    xs = [b.x for b in linear_seq.ground_truth]
    for prev, curr in zip(xs, xs[1:]):
        assert curr >= prev - 1e-9, (
            f"linear sequence x went backwards: {prev:.2f} -> {curr:.2f}"
        )
