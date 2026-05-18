"""Unit tests for uav_tracker.viz.video.write_mp4."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from uav_tracker.viz.video import write_mp4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid_frames(n: int, h: int = 240, w: int = 320, value: int = 100) -> list[np.ndarray]:
    """Return *n* identical solid-colour BGR frames."""
    return [np.full((h, w, 3), value, dtype=np.uint8) for _ in range(n)]


# ---------------------------------------------------------------------------
# Happy-path: write and verify
# ---------------------------------------------------------------------------


def test_write_5_frames_creates_nonempty_file() -> None:
    frames = _solid_frames(5)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.mp4"
        result_path = write_mp4(frames, out, fps=30)
        assert result_path.exists(), "Output file does not exist."
        assert result_path.stat().st_size > 0, "Output file is empty."


def test_write_returns_resolved_path() -> None:
    frames = _solid_frames(3)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out.mp4"
        result = write_mp4(frames, out)
        assert result.is_absolute()
        assert result == out.resolve()


def test_write_creates_parent_directory() -> None:
    frames = _solid_frames(3)
    with tempfile.TemporaryDirectory() as tmp:
        nested = Path(tmp) / "a" / "b" / "c" / "test.mp4"
        assert not nested.parent.exists()
        write_mp4(frames, nested)
        assert nested.exists()


def test_accepts_str_path() -> None:
    frames = _solid_frames(3)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out.mp4")
        result = write_mp4(frames, out)
        assert result.exists()


def test_accepts_generator() -> None:
    """write_mp4 must accept a lazy generator, not just a list."""
    def _gen():
        for _ in range(4):
            yield np.zeros((240, 320, 3), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "gen.mp4"
        result = write_mp4(_gen(), out)
        assert result.stat().st_size > 0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_empty_iterable_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "empty.mp4"
        with pytest.raises(ValueError, match="empty"):
            write_mp4([], out)


def test_mismatched_shape_raises() -> None:
    frames: list[np.ndarray] = [
        np.zeros((240, 320, 3), dtype=np.uint8),
        np.zeros((480, 640, 3), dtype=np.uint8),  # wrong shape
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "bad.mp4"
        with pytest.raises(ValueError, match="shape"):
            write_mp4(frames, out)


def test_non_3channel_raises() -> None:
    frames: list[np.ndarray] = [
        np.zeros((240, 320), dtype=np.uint8),  # grayscale, wrong ndim
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "gray.mp4"
        with pytest.raises(ValueError):
            write_mp4(frames, out)


# ---------------------------------------------------------------------------
# Integrity: mp4 can be re-opened with VideoCapture
# ---------------------------------------------------------------------------


def test_written_mp4_is_readable_by_opencv() -> None:
    n_frames = 5
    frames = _solid_frames(n_frames, h=240, w=320)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "check.mp4"
        write_mp4(frames, out, fps=30)
        cap = cv2.VideoCapture(str(out))
        try:
            assert cap.isOpened(), "VideoCapture could not open the written file."
            read_count = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                read_count += 1
            assert read_count > 0, "VideoCapture read 0 frames from written file."
        finally:
            cap.release()
