"""Integration tests for HybridRunner end-to-end (Phase 3)."""

from __future__ import annotations

import pytest

pytest.importorskip("cv2")

import numpy as np

from uav_tracker.types import BBox


@pytest.fixture()
def synthetic_dataset():
    from uav_tracker.datasets.synthetic import SyntheticDataset

    return SyntheticDataset(seed=42)


@pytest.fixture()
def hybrid_runner():
    from uav_tracker.trackers.kcf_kalman import KCFKalmanTracker
    from uav_tracker.trackers.siamese.siamfc import SiamFCTracker
    from uav_tracker.signals.tracker_confidence import TrackerConfidenceSignal
    from uav_tracker.schedulers.hysteresis_binary import HysteresisBinaryScheduler
    from uav_tracker.runner import HybridRunner

    return HybridRunner(
        trackers={
            0: KCFKalmanTracker(),
            1: SiamFCTracker(device="cpu", dtype="float32"),
        },
        signals=[TrackerConfidenceSignal()],
        scheduler=HysteresisBinaryScheduler(
            E_hi=0.65,
            E_lo=0.50,
            confirm_frames=5,
            cooldown_frames=5,
        ),
        seed=42,
    )


def test_trajectory_length_equals_num_frames(hybrid_runner, synthetic_dataset) -> None:
    """Trajectory must have exactly (n_frames - 1) entries (frames 1..N-1)."""
    seq = next(iter(synthetic_dataset))
    frames = list(seq.frames)
    entries = list(hybrid_runner.run(seq))
    # HybridRunner.run() yields one entry per frame[1:].
    assert len(entries) == len(frames) - 1


def test_time_in_tier_sums_to_num_tracked_frames(hybrid_runner, synthetic_dataset) -> None:
    """time_in_tier[0] + time_in_tier[1] must equal the number of update frames."""
    seq = next(iter(synthetic_dataset))
    frames = list(seq.frames)
    entries = list(hybrid_runner.run(seq))
    n_updates = len(frames) - 1

    tit = hybrid_runner.time_in_tier
    assert sum(tit.values()) == n_updates, (
        f"time_in_tier sums to {sum(tit.values())}, expected {n_updates}"
    )


def test_mostly_tier0_with_high_kcf_confidence(hybrid_runner, synthetic_dataset) -> None:
    """With KCF returning confidence=0.8 (locked), signal=0.2 < E_hi=0.65.

    Tier 0 should dominate — we don't assert a specific split but do
    assert tier 0 accounts for at least 50% of frames.
    """
    seq = next(iter(synthetic_dataset))
    list(hybrid_runner.run(seq))
    tit = hybrid_runner.time_in_tier
    t0 = tit.get(0, 0)
    total = sum(tit.values())
    assert total > 0
    assert t0 / total >= 0.5, (
        f"Expected tier0 to dominate. time_in_tier={tit}"
    )


def test_tier_sequence_consistent_with_entries(hybrid_runner, synthetic_dataset) -> None:
    """TelemetryEntry.tier must match hybrid_runner.tier_sequence entries."""
    seq = next(iter(synthetic_dataset))
    entries = list(hybrid_runner.run(seq))
    tier_seq = hybrid_runner.tier_sequence
    assert len(tier_seq) == len(entries)
    for i, (entry, tier) in enumerate(zip(entries, tier_seq)):
        assert entry.tier == tier, (
            f"Mismatch at frame {i}: entry.tier={entry.tier}, tier_seq={tier}"
        )


def test_reset_clears_state(hybrid_runner, synthetic_dataset) -> None:
    """After reset(), runner state is clean for re-use."""
    seq = next(iter(synthetic_dataset))
    list(hybrid_runner.run(seq))
    hybrid_runner.reset()
    assert hybrid_runner.trajectory == []
    assert hybrid_runner.tier_sequence == []
    assert all(v == 0 for v in hybrid_runner.time_in_tier.values())
