"""tests/unit/test_saltr_flops.py — Unit tests for uav_tracker.metrics.flops."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Test 1: flops_per_frame returns expected keys
# ---------------------------------------------------------------------------

def test_flops_per_frame_returns_expected_keys():
    from uav_tracker.metrics.flops import flops_per_frame

    telemetry = [
        {"tier": "full", "used_detector": False, "saltrd_action_compute": "FULL"},
        {"tier": "full", "used_detector": True,  "saltrd_action_compute": "FULL"},
        {"tier": "light", "used_detector": False, "saltrd_action_compute": "SKIP"},
    ]
    tier_flops = {"full": 0.9, "light": 0.6}
    result = flops_per_frame(telemetry, tier_flops, signal_flops_per_frame=0.001)

    assert isinstance(result, dict), "Result must be a dict"
    for key in ("mean_gflops", "tracker_gflops", "detector_gflops", "saltrd_gflops"):
        assert key in result, f"Missing key: {key}"
        assert isinstance(result[key], float), f"{key} must be a float"

    # Sanity: mean should be sum of components
    assert abs(
        result["mean_gflops"]
        - (result["tracker_gflops"] + result["detector_gflops"] + result["saltrd_gflops"])
    ) < 1e-9, "mean_gflops must equal sum of tracker + detector + saltrd"

    # Detector fired once out of 3 frames → detector_gflops > 0
    assert result["detector_gflops"] > 0.0


# ---------------------------------------------------------------------------
# Test 2: flops_per_frame with empty telemetry returns zeros
# ---------------------------------------------------------------------------

def test_flops_per_frame_empty_telemetry_returns_zeros():
    from uav_tracker.metrics.flops import flops_per_frame

    result = flops_per_frame([], {"full": 0.9, "light": 0.6}, signal_flops_per_frame=0.001)

    assert result["mean_gflops"] == 0.0
    assert result["tracker_gflops"] == 0.0
    assert result["detector_gflops"] == 0.0
    assert result["saltrd_gflops"] == 0.0


# ---------------------------------------------------------------------------
# Test 3: measure_tracker_gflops returns a positive float
# ---------------------------------------------------------------------------

def test_measure_tracker_gflops_returns_positive_float():
    from uav_tracker.metrics.flops import measure_tracker_gflops

    # Use a minimal stub class that exposes flops_per_update
    class _StubTracker:
        def __init__(self):
            pass

        def flops_per_update(self) -> float:
            return 0.9e9  # SGLATrack-equivalent

    result = measure_tracker_gflops(_StubTracker)
    assert isinstance(result, float), "measure_tracker_gflops must return float"
    assert result > 0.0, "GFLOPs must be positive"
