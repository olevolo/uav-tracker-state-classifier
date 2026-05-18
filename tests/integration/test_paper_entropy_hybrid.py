"""Integration test: paper_entropy_hybrid config with MotionEntropySignal (Phase 4).

Runs the full hybrid pipeline (KCFKalman + MobileTrack tier, MotionEntropySignal,
HysteresisBinaryScheduler) over the SyntheticDataset.

Assertions (per PLAN §11 exit criteria):
  - Trajectory length == n_frames - 1 (one entry per tracked frame).
  - time_in_tier values sum to n_frames - 1.
  - No exceptions / crashes.
  - At least one sequence produces a non-NaN motion_entropy signal value.

AUC gate is intentionally omitted — synthetic sequences don't reproduce
paper Table 2 numbers. The unit tests in test_motion_entropy.py validate
that entropy values are in the correct ranges.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cv2")

import numpy as np

from uav_tracker.datasets.synthetic import SyntheticDataset
from uav_tracker.runner import HybridRunner
from uav_tracker.schedulers.hysteresis_binary import HysteresisBinaryScheduler
from uav_tracker.signals.motion_entropy import MotionEntropySignal
from uav_tracker.trackers.kcf_kalman import KCFKalmanTracker
from uav_tracker.trackers.siamese.mobiletrack import MobileTrackTracker


@pytest.fixture()
def entropy_hybrid_runner():
    """Hybrid runner matching paper_entropy_hybrid.yaml Phase 4 config."""
    try:
        tier0 = KCFKalmanTracker()
    except Exception as exc:
        pytest.skip(f"Cannot build KCFKalmanTracker: {exc}")

    try:
        tier1 = MobileTrackTracker(device="cpu", dtype="float32")
    except Exception as exc:
        pytest.skip(f"Cannot build MobileTrackTracker: {exc}")

    signal = MotionEntropySignal(
        n_bins=16,
        alpha=0.8,
        mag_threshold=1.0,
        seed=42,
    )

    scheduler = HysteresisBinaryScheduler(
        E_hi=0.65,
        E_lo=0.50,
        confirm_frames=5,
        cooldown_frames=5,
        signal_name="motion_entropy",
    )

    return HybridRunner(
        trackers={0: tier0, 1: tier1},
        signals=[signal],
        scheduler=scheduler,
        seed=42,
    )


@pytest.fixture()
def synthetic_dataset():
    return SyntheticDataset(seed=42)


def test_trajectory_length_matches_frame_count(
    entropy_hybrid_runner, synthetic_dataset
) -> None:
    """Trajectory must have exactly (n_frames - 1) entries."""
    runner = entropy_hybrid_runner
    seq = next(iter(synthetic_dataset))
    frames = list(seq.frames)
    entries = list(runner.run(seq))
    assert len(entries) == len(frames) - 1, (
        f"Expected {len(frames) - 1} telemetry entries, got {len(entries)}"
    )


def test_time_in_tier_sums_correctly(
    entropy_hybrid_runner, synthetic_dataset
) -> None:
    """time_in_tier values must sum to the number of tracked frames."""
    runner = entropy_hybrid_runner
    seq = next(iter(synthetic_dataset))
    frames = list(seq.frames)
    list(runner.run(seq))
    n_updates = len(frames) - 1
    tit = runner.time_in_tier
    assert sum(tit.values()) == n_updates, (
        f"time_in_tier sums to {sum(tit.values())}, expected {n_updates}. "
        f"time_in_tier={tit}"
    )


def test_motion_entropy_signal_value_not_nan(
    entropy_hybrid_runner, synthetic_dataset
) -> None:
    """At least one frame must emit a non-NaN motion_entropy signal value."""
    runner = entropy_hybrid_runner
    seq = next(iter(synthetic_dataset))
    entries = list(runner.run(seq))

    entropy_values = [e.signals.get("motion_entropy", float("nan")) for e in entries]
    non_nan = [v for v in entropy_values if not np.isnan(v)]
    assert len(non_nan) > 0, (
        "All motion_entropy signal values were NaN — signal is broken."
    )


def test_no_crash_on_all_sequences(
    entropy_hybrid_runner, synthetic_dataset
) -> None:
    """Runner must complete all synthetic sequences without raising."""
    runner = entropy_hybrid_runner
    for seq in synthetic_dataset:
        runner.reset()
        entries = list(runner.run(seq))
        assert len(entries) > 0, f"Sequence {seq.name} produced zero entries"


def test_tier_sequence_consistent_with_entries(
    entropy_hybrid_runner, synthetic_dataset
) -> None:
    """TelemetryEntry.tier must match runner.tier_sequence entries."""
    runner = entropy_hybrid_runner
    seq = next(iter(synthetic_dataset))
    entries = list(runner.run(seq))
    tier_seq = runner.tier_sequence
    assert len(tier_seq) == len(entries)
    for i, (entry, tier) in enumerate(zip(entries, tier_seq)):
        assert entry.tier == tier, (
            f"Mismatch at frame {i}: entry.tier={entry.tier}, tier_seq={tier}"
        )
