"""MP4 writer using cv2.VideoWriter (Phase 8).

Public API
----------
write_mp4(frames, out_path, fps=30) -> Path
    Write an iterable of BGR uint8 frames to an MP4 file using the
    ``mp4v`` fourcc codec.  Auto-detects frame size from the first frame.
    Validates shape consistency; raises ``ValueError`` on mismatch or on
    an empty iterable.  The parent directory is created if it does not
    exist.  Returns the resolved ``Path`` to the written file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import cv2
import numpy as np


def write_mp4(
    frames: Union[Iterable[np.ndarray], list[np.ndarray]],
    out_path: Union[Path, str],
    fps: int = 30,
) -> Path:
    """Write *frames* to an MP4 file at *out_path*.

    Parameters
    ----------
    frames:
        Iterable of BGR uint8 numpy arrays, all with identical shape
        ``(H, W, 3)``.
    out_path:
        Destination file path.  The ``.mp4`` extension is recommended but
        not enforced.  Parent directory is created automatically.
    fps:
        Playback frame rate for the encoded video.

    Returns
    -------
    Path
        Resolved absolute path to the written file.

    Raises
    ------
    ValueError
        If *frames* is empty, or if any frame has a shape different from
        the first frame.
    """
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_iter = iter(frames)

    # Pull the first frame to determine size.
    try:
        first = next(frame_iter)
    except StopIteration:
        raise ValueError(
            "write_mp4: frames iterable is empty — nothing to write."
        )

    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(
            f"write_mp4: frames must be BGR (H, W, 3) arrays; "
            f"got shape {first.shape!r}."
        )

    expected_shape = first.shape  # (H, W, 3)
    h, w = expected_shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))

    try:
        _write_frame(writer, first, expected_shape, 0)
        for idx, frame in enumerate(frame_iter, start=1):
            _write_frame(writer, frame, expected_shape, idx)
    finally:
        writer.release()

    return out_path


def _write_frame(
    writer: cv2.VideoWriter,
    frame: np.ndarray,
    expected_shape: tuple[int, ...],
    idx: int,
) -> None:
    """Validate *frame* shape and write it via *writer*."""
    if frame.shape != expected_shape:
        raise ValueError(
            f"write_mp4: frame {idx} shape {frame.shape!r} does not match "
            f"first frame shape {expected_shape!r}. All frames must be the "
            f"same size."
        )
    # Ensure uint8 BGR before writing.
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    writer.write(frame)


__all__ = ["write_mp4"]
