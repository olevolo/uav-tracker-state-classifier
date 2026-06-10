"""Regression test: SGLATrack must not grow RSS linearly with frame count.

Root cause of the original OOM crash (bicycle-18, 1355 frames): when the
tracker is lost the predicted bbox w/h grew without bound each frame, causing
``_sample_target``'s crop_sz (∝ √(w·h)) to double every frame until the
padded numpy crop required gigabytes of RAM.  The fix clamps w/h to frame
dimensions in ``update()``.

This test:
- Runs ~200 dummy 720×1280 frames through ``update()``.
- Asserts RSS growth < 200 MB between frame 50 and frame 200.
- Skips automatically if SGLATrack weights are absent (CI-safe).
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the source packages importable without installation.
_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT / "src"))
sys.path.insert(0, str(_PROJECT / "salrtd" / "src"))


def _weights_present() -> bool:
    try:
        from uav_tracker.paths import weights_root  # type: ignore[import]
        p = weights_root() / "sglatrack" / "sglatrack_ep0297.pth.tar"
        return p.exists()
    except Exception:
        return False


@pytest.mark.skipif(not _weights_present(), reason="SGLATrack weights not present")
def test_sglatrack_rss_stable_over_200_frames() -> None:
    """RSS at frame 200 must be within 200 MB of RSS at frame 50."""
    import psutil

    try:
        from uav_tracker.trackers.sglatrack import SGLATracker  # type: ignore[import]
        from uav_tracker.types import BBox  # type: ignore[import]
    except ImportError as exc:
        pytest.skip(f"SGLATrack adapter not importable: {exc}")

    proc = psutil.Process(os.getpid())

    tracker = SGLATracker(device="cpu")
    # 720×1280 matches real LaSOT frame size; the crash originally triggered on this size.
    H, W = 720, 1280
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    bbox = BBox(x=600.0, y=300.0, w=40.0, h=40.0)
    tracker.init(frame, bbox)

    rss_at_50: float = 0.0
    rss_at_200: float = 0.0

    for i in range(1, 201):
        state = tracker.update(frame)
        del state
        if i & 0xFF == 0:
            gc.collect()
        if i == 50:
            gc.collect()
            rss_at_50 = proc.memory_info().rss / 1024 / 1024
        if i == 200:
            gc.collect()
            rss_at_200 = proc.memory_info().rss / 1024 / 1024

    growth_mb = rss_at_200 - rss_at_50
    assert growth_mb < 200.0, (
        f"SGLATrack RSS grew {growth_mb:.0f} MB between frame 50 and frame 200 "
        f"(rss_50={rss_at_50:.0f}MB rss_200={rss_at_200:.0f}MB). "
        "Likely cause: unbounded bbox/crop growth when tracker is lost."
    )


@pytest.mark.skipif(not _weights_present(), reason="SGLATrack weights not present")
def test_sglatrack_bbox_clipped_to_frame() -> None:
    """Predicted bbox must never exceed the frame dimensions."""
    try:
        from uav_tracker.trackers.sglatrack import SGLATracker  # type: ignore[import]
        from uav_tracker.types import BBox  # type: ignore[import]
    except ImportError as exc:
        pytest.skip(f"SGLATrack adapter not importable: {exc}")

    tracker = SGLATracker(device="cpu")
    H, W = 720, 1280
    # All-black frame forces low confidence / lost state every frame — exercises
    # the worst-case unconstrained growth path.
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    bbox = BBox(x=600.0, y=300.0, w=40.0, h=40.0)
    tracker.init(frame, bbox)

    for i in range(1, 101):
        state = tracker.update(frame)
        assert state.bbox.w <= W, (
            f"frame {i}: predicted w={state.bbox.w:.0f} exceeds frame width {W}"
        )
        assert state.bbox.h <= H, (
            f"frame {i}: predicted h={state.bbox.h:.0f} exceeds frame height {H}"
        )
        del state
