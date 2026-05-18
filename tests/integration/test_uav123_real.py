"""Integration test: OPE on UAV123 bike1 with kcf_kalman.

Requires UAV_DATA_ROOT env var pointing at readable UAV123 data.
Skipped if absent.

Asserts AUC > 0.05 — a modest but non-zero threshold for KCF on real UAV123.
"""

from __future__ import annotations

import os
import pytest
from pathlib import Path

UAV_DATA_ROOT = os.environ.get("UAV_DATA_ROOT", "")
_has_data = bool(UAV_DATA_ROOT) and (
    (Path(UAV_DATA_ROOT) / "uav123" / "UAV123" / "data_seq" / "UAV123").exists()
)


@pytest.fixture(scope="module")
def bike1_sequence():
    """Yield the bike1 sequence from the real UAV123 dataset."""
    if not _has_data:
        pytest.skip("UAV_DATA_ROOT not set or UAV123 data not accessible")
    from uav_tracker.datasets.uav123 import UAV123Dataset
    ds = UAV123Dataset(root=UAV_DATA_ROOT + "/uav123")
    for seq in ds:
        if seq.name == "bike1":
            return seq
    pytest.fail("bike1 not found in dataset")


def test_kcf_kalman_ope_on_bike1(bike1_sequence):
    """Run OPE on bike1 with kcf_kalman; assert AUC > 0.05."""
    import uav_tracker  # noqa: trigger plugin registration
    from uav_tracker.registry import TRACKERS
    from uav_tracker.evaluation.ope import OPERunner

    tracker = TRACKERS.build("kcf_kalman")
    ope = OPERunner(seed=42)
    result = ope.run(tracker=tracker, dataset=[bike1_sequence], limit=1)

    assert len(result.per_sequence) == 1, "Expected exactly one sequence result"
    sr = result.per_sequence[0]

    print(f"\n  UAV123 bike1 — AUC={sr.auc:.4f}  Pr@20={sr.precision_at_20:.4f}  FPS={sr.fps:.1f}")

    assert sr.auc > 0.05, (
        f"KCF on bike1 AUC={sr.auc:.4f} is below 0.05 — tracker may have broken"
    )


def test_dataset_yields_123_sequences_no_crash():
    """Iterate the full dataset without crashing — no frame loading, just metadata."""
    if not _has_data:
        pytest.skip("UAV_DATA_ROOT not set or UAV123 data not accessible")
    from uav_tracker.datasets.uav123 import UAV123Dataset
    ds = UAV123Dataset(root=UAV_DATA_ROOT + "/uav123")
    count = 0
    for seq in ds:
        # Access metadata but do not load frames.
        assert seq.name
        assert seq.init_bbox is not None
        assert isinstance(seq.attributes, set)
        count += 1
    assert count == 123, f"Expected 123 sequences, got {count}"
