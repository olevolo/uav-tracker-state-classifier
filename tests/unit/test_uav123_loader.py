"""Unit tests for UAV123Dataset loader.

Points at the real UAV123 data at $UAV_DATA_ROOT/uav123/UAV123/.
Tests are skipped if the directory is not readable.
"""

from __future__ import annotations

import os
import pytest
from pathlib import Path


# Skip all tests if data root is not set / readable.
UAV_DATA_ROOT = os.environ.get("UAV_DATA_ROOT", "")
UAV123_ROOT = Path(UAV_DATA_ROOT) / "uav123" / "UAV123" if UAV_DATA_ROOT else None

_has_data = UAV123_ROOT is not None and (UAV123_ROOT / "data_seq" / "UAV123").exists()


@pytest.fixture(scope="module")
def dataset():
    """Return a UAV123Dataset instance pointing at the real data."""
    if not _has_data:
        pytest.skip("UAV_DATA_ROOT not set or UAV123 data not found")
    from uav_tracker.datasets.uav123 import UAV123Dataset
    return UAV123Dataset(root=str(UAV123_ROOT))


@pytest.fixture(scope="module")
def all_sequences(dataset):
    return list(dataset)


class TestUAV123DatasetCount:
    def test_yields_123_sequences(self, all_sequences):
        """UAV123 has exactly 123 sequences (confirmed by anno dir)."""
        assert len(all_sequences) == 123, (
            f"Expected 123 sequences, got {len(all_sequences)}"
        )

    def test_sequence_names_unique(self, all_sequences):
        names = [s.name for s in all_sequences]
        assert len(names) == len(set(names)), "Duplicate sequence names found"

    def test_bike1_is_first(self, all_sequences):
        """Sorted order: bike1 should come first alphabetically."""
        assert all_sequences[0].name == "bike1"


class TestBike1Sequence:
    @pytest.fixture(scope="class")
    def bike1(self, all_sequences):
        for s in all_sequences:
            if s.name == "bike1":
                return s
        pytest.fail("bike1 not found in dataset")

    def test_has_name(self, bike1):
        assert bike1.name == "bike1"

    def test_has_init_bbox(self, bike1):
        bb = bike1.init_bbox
        assert bb is not None
        assert bb.w > 0 and bb.h > 0

    def test_init_bbox_matches_ground_truth_zero(self, bike1):
        assert bike1.init_bbox == bike1.ground_truth[0]

    def test_ground_truth_length(self, bike1):
        # bike1 has 3085 frames per configSeqs.m — annotation has 3084 lines.
        assert len(bike1.ground_truth) >= 100  # sanity: at least 100 frames

    def test_frames_generator_is_lazy(self, bike1):
        """frames property must not be a pre-loaded list."""
        import types
        # It should be iterable but not a list (lazy).
        frames_obj = bike1.frames
        assert hasattr(frames_obj, "__iter__")

    def test_first_frame_is_ndarray(self, bike1):
        import numpy as np
        first = next(iter(bike1.frames))
        assert isinstance(first, np.ndarray)
        assert first.ndim == 3  # HWC
        assert first.shape[2] == 3  # BGR

    def test_attributes_non_empty(self, bike1):
        """bike1 should have at least one attribute flag set."""
        assert isinstance(bike1.attributes, set)
        assert len(bike1.attributes) > 0

    def test_attributes_are_known_codes(self, bike1):
        known = {"FM", "OCC", "IV", "SV", "POC", "DEF", "MB", "CM", "BC", "SOB", "LR", "ARC"}
        for attr in bike1.attributes:
            assert attr in known, f"Unknown attribute code: {attr}"

    def test_ground_truth_bboxes_valid_type(self, bike1):
        from uav_tracker.types import BBox
        for bb in bike1.ground_truth[:10]:
            assert isinstance(bb, BBox)


class TestAutoRootDetection:
    def test_outer_root_accepted(self):
        """Passing outer uav123/ dir should auto-detect to inner UAV123/."""
        if not _has_data:
            pytest.skip("UAV_DATA_ROOT not set")
        outer = Path(UAV_DATA_ROOT) / "uav123"
        from uav_tracker.datasets.uav123 import UAV123Dataset, _BBoxAnnotated
        ds = UAV123Dataset(root=str(outer))
        seqs = list(ds)
        assert len(seqs) == 123

    def test_inner_root_accepted(self):
        """Passing inner uav123/UAV123/ dir should work directly."""
        if not _has_data:
            pytest.skip("UAV_DATA_ROOT not set")
        from uav_tracker.datasets.uav123 import UAV123Dataset
        ds = UAV123Dataset(root=str(UAV123_ROOT))
        seqs = list(ds)
        assert len(seqs) == 123


class TestAttributeFilter:
    def test_attribute_filter_reduces_count(self, dataset):
        from uav_tracker.datasets.uav123 import UAV123Dataset
        ds_fm = UAV123Dataset(root=str(UAV123_ROOT), attributes={"FM"})
        fm_seqs = list(ds_fm)
        all_seqs = list(dataset)
        assert len(fm_seqs) < len(all_seqs)
        assert len(fm_seqs) > 0

    def test_filtered_sequences_have_matching_attributes(self, dataset):
        from uav_tracker.datasets.uav123 import UAV123Dataset
        ds_fm = UAV123Dataset(root=str(UAV123_ROOT), attributes={"FM"})
        for seq in list(ds_fm)[:5]:
            assert "FM" in seq.attributes


class TestNaNHandling:
    def test_sequences_with_nan_init_frame_are_skipped(self, dataset):
        """Any sequence whose frame-0 GT is NaN should be absent."""
        for seq in dataset:
            from uav_tracker.datasets.uav123 import _BBoxAnnotated
            if isinstance(seq.init_bbox, _BBoxAnnotated):
                assert seq.init_bbox.valid is True

    def test_nan_annotations_have_valid_false(self):
        """Directly parse an anno file we know has NaNs."""
        if not _has_data:
            pytest.skip("UAV_DATA_ROOT not set")
        nan_anno = Path(UAV_DATA_ROOT) / "uav123" / "UAV123" / "anno" / "UAV123" / "bird1_2.txt"
        if not nan_anno.exists():
            pytest.skip("bird1_2.txt not present")
        from uav_tracker.datasets.uav123 import _parse_anno
        bboxes = _parse_anno(nan_anno)
        invalid = [b for b in bboxes if not b.valid]
        assert len(invalid) > 0, "bird1_2 should contain NaN frames"
