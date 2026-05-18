"""Integration test: hybrid runner with detection tier on UAV123 bike1.

Requires:
- UAV_DATA_ROOT env var pointing at a readable UAV123 root.
- ultralytics package (Engineer A's YOLOv8 plugin).

Both are skipped gracefully if absent.
"""

from __future__ import annotations

import os
import pytest

ultralytics = pytest.importorskip("ultralytics", reason="ultralytics not installed — skipping")

UAV_DATA_ROOT = os.environ.get("UAV_DATA_ROOT", "")
_has_data = bool(UAV_DATA_ROOT) and (
    (__import__("pathlib").Path(UAV_DATA_ROOT) / "uav123" / "UAV123" / "data_seq" / "UAV123").exists()
)


@pytest.fixture(scope="module")
def bike1_100frames():
    """Return bike1 sequence capped at 100 frames."""
    if not _has_data:
        pytest.skip("UAV_DATA_ROOT not set or UAV123 data not accessible")
    from uav_tracker.datasets.uav123 import UAV123Dataset
    ds = UAV123Dataset(root=UAV_DATA_ROOT + "/uav123", max_frames=100)
    for seq in ds:
        if seq.name == "bike1":
            return seq
    pytest.fail("bike1 not found")


@pytest.fixture(scope="module")
def yolo_detector():
    """Build YOLOv8n detector — skip if Engineer A's plugin isn't importable."""
    try:
        from uav_tracker.registry import DETECTORS
        import uav_tracker  # noqa: trigger registration
        return DETECTORS.build("yolov8n", device="cpu", conf_threshold=0.25)
    except Exception as exc:
        pytest.skip(f"yolov8n detector not available: {exc}")


def test_hybrid_run_with_detector_tier(bike1_100frames, yolo_detector):
    """Run HybridRunner with a 3-tier config (kcf_kalman + detector) on 100 frames."""
    from uav_tracker.registry import TRACKERS, SIGNALS, SCHEDULERS
    import uav_tracker  # noqa: trigger registration
    from uav_tracker.runner import HybridRunner

    tracker0 = TRACKERS.build("kcf_kalman")
    sched = SCHEDULERS.build(
        "multi_tier",
        tier_thresholds=[[0.50, 0.35], [0.80, 0.65]],
        confirm_frames=3,
        cooldown_frames=3,
        signal_name="motion_entropy",
    )
    sig = SIGNALS.build("motion_entropy", n_bins=16, alpha=0.8)

    runner = HybridRunner(
        trackers={0: tracker0},
        signals=[sig],
        scheduler=sched,
        detectors={2: yolo_detector},
        seed=42,
    )

    entries = list(runner.run(bike1_100frames))
    assert len(entries) > 0, "Expected at least one telemetry entry"
    assert len(entries) == len(list(bike1_100frames.frames)) - 1

    # All tier indices should be valid (0, 1, or 2).
    for e in entries:
        assert e.tier in {0, 1, 2}

    # recoveries counter exists and is non-negative.
    assert runner.recoveries >= 0

    # Signals must be present in telemetry.
    for e in entries:
        assert "motion_entropy" in e.signals


def test_hybrid_run_auc_positive(bike1_100frames):
    """Simple sanity: AUC on 100 frames of bike1 with kcf_kalman should be > 0."""
    from uav_tracker.registry import TRACKERS, SIGNALS, SCHEDULERS
    import uav_tracker  # noqa
    from uav_tracker.runner import HybridRunner
    from uav_tracker.evaluation.ope import OPERunner

    tracker0 = TRACKERS.build("kcf_kalman")
    sched = SCHEDULERS.build(
        "multi_tier",
        tier_thresholds=[[0.50, 0.35]],
        confirm_frames=3,
        cooldown_frames=3,
        signal_name="motion_entropy",
    )
    sig = SIGNALS.build("motion_entropy", n_bins=16, alpha=0.8)

    runner = HybridRunner(
        trackers={0: tracker0},
        signals=[sig],
        scheduler=sched,
        seed=42,
    )

    ope = OPERunner(seed=42)
    # Wrap single sequence in a list-like for OPERunner.
    result = ope.run(tracker=runner, dataset=[bike1_100frames], limit=1)
    assert result.auc > 0.0, f"Expected positive AUC, got {result.auc}"
