"""Integration test: end-to-end demo rendering (Phase 8).

Marked ``@pytest.mark.slow`` — excluded from the default ``-m not slow``
CI run; run explicitly with ``pytest -v -m slow``.

Verifies the full pipeline:
  1. Build a SyntheticDataset and iterate the first sequence.
  2. Render per-frame overlays via ``draw_frame_overlay``.
  3. Write an MP4 via ``write_mp4``.
  4. Assert the file is non-empty and re-openable with ``cv2.VideoCapture``
     with a positive frame count.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pytest

from uav_tracker.datasets.synthetic import SyntheticDataset
from uav_tracker.types import BBox, SignalReport
from uav_tracker.viz.overlay import draw_frame_overlay
from uav_tracker.viz.video import write_mp4


@pytest.mark.slow
def test_render_synthetic_sequence_to_mp4() -> None:
    ds = SyntheticDataset(seed=42)

    # Take only the first sequence for speed.
    first_seq = next(iter(ds))

    frames = first_seq.frames
    gt_list = first_seq.ground_truth
    assert len(frames) > 0, "SyntheticDataset returned an empty sequence."

    # Build dummy signals representative of a real run.
    def _make_signals(i: int) -> dict[str, SignalReport]:
        t = i / max(len(frames) - 1, 1)
        return {
            "confidence": SignalReport(value=0.8 - 0.3 * t),
            "entropy": SignalReport(value=0.2 + 0.5 * t),
        }

    annotated = []
    for i, frame in enumerate(frames):
        gt = gt_list[i] if i < len(gt_list) else None
        # Cycle through all tiers to exercise each colour path.
        tier = i % 3
        ann = draw_frame_overlay(
            frame=frame,
            bbox=gt,
            tier=tier,
            signals=_make_signals(i),
            fps=30.0,
            gt_bbox=None,
        )
        annotated.append(ann)

    assert len(annotated) == len(frames)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "demo_test.mp4"
        result_path = write_mp4(annotated, out, fps=30)

        # File existence and non-empty.
        assert result_path.exists(), "MP4 file was not created."
        file_size = result_path.stat().st_size
        assert file_size > 0, "MP4 file is empty (0 bytes)."

        # Re-open and count frames to confirm integrity.
        cap = cv2.VideoCapture(str(result_path))
        try:
            assert cap.isOpened(), (
                f"cv2.VideoCapture could not open {result_path} "
                f"(size={file_size} bytes)."
            )
            n_read = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                n_read += 1
        finally:
            cap.release()

        assert n_read > 0, (
            f"VideoCapture read 0 frames from {result_path} "
            f"(file size={file_size} bytes). "
            "Codec or container may be unsupported on this platform."
        )
