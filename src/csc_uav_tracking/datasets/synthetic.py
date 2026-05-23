"""Synthetic dataset for Phase 1 OPE harness validation.

Generates three short sequences (60 frames, 320x240 uint8 BGR) without
any filesystem dependency, making end-to-end CI tests viable on a bare
machine.  All sequences are deterministic under ``seed``.

Registered as ``"synthetic"`` in DATASETS.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np

from csc_uav_tracking.registry import DATASETS
from csc_uav_tracking.types import BBox


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_W, _H = 320, 240
_BW, _BH = 60, 45   # target rectangle size (pixels)
_N_FRAMES = 60


def _make_frame(bg_value: int = 40) -> np.ndarray:
    """Return a solid-gray uint8 BGR frame."""
    return np.full((_H, _W, 3), bg_value, dtype=np.uint8)


def _paint_rect(frame: np.ndarray, bbox: BBox) -> None:
    """In-place paint a white rectangle defined by *bbox* onto *frame*."""
    x0 = int(round(max(bbox.x, 0.0)))
    y0 = int(round(max(bbox.y, 0.0)))
    x1 = int(round(min(bbox.x + bbox.w, _W)))
    y1 = int(round(min(bbox.y + bbox.h, _H)))
    frame[y0:y1, x0:x1] = 220


def _add_noise(frame: np.ndarray, rng: np.random.Generator, sigma: float) -> None:
    """Add zero-mean Gaussian noise to *frame* in-place (clipped to uint8)."""
    if sigma <= 0.0:
        return
    noise = rng.normal(0.0, sigma, frame.shape)
    frame[:] = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sequence implementation
# ---------------------------------------------------------------------------


class _SyntheticSequence:
    """Concrete implementation of the Sequence Protocol for synthetic data."""

    def __init__(
        self,
        name: str,
        frames: list[np.ndarray],
        ground_truth: list[BBox],
        attributes: set[str],
    ) -> None:
        assert len(frames) == len(ground_truth), "frame/gt length mismatch"
        self.name = name
        self.frames: list[np.ndarray] = frames
        self.ground_truth: list[BBox] = ground_truth
        self.init_bbox: BBox = ground_truth[0]
        self.attributes: set[str] = attributes


def _build_static(rng: np.random.Generator) -> _SyntheticSequence:
    """Stationary rectangle with low-amplitude noise.

    Baseline floor: any working tracker should achieve AUC > 0.85.
    """
    cx, cy = _W / 2, _H / 2
    x0, y0 = cx - _BW / 2, cy - _BH / 2
    gt_bbox = BBox(x=x0, y=y0, w=float(_BW), h=float(_BH))

    frames: list[np.ndarray] = []
    gts: list[BBox] = []
    for _ in range(_N_FRAMES):
        frame = _make_frame()
        _paint_rect(frame, gt_bbox)
        _add_noise(frame, rng, sigma=4.0)
        frames.append(frame)
        gts.append(gt_bbox)

    return _SyntheticSequence(
        name="synthetic_static",
        frames=frames,
        ground_truth=gts,
        attributes={"STATIC"},
    )


def _build_linear(rng: np.random.Generator) -> _SyntheticSequence:
    """Rectangle translating left-to-right at constant velocity.

    Velocity chosen so the target stays within frame for all 60 frames.
    """
    vx = (_W - _BW - 10 - 10) / (_N_FRAMES - 1)   # spans almost full width
    vy = 0.0
    x_start = 10.0
    y_start = (_H - _BH) / 2

    frames: list[np.ndarray] = []
    gts: list[BBox] = []
    for i in range(_N_FRAMES):
        x = x_start + vx * i
        y = y_start + vy * i
        bbox = BBox(x=x, y=y, w=float(_BW), h=float(_BH))
        frame = _make_frame()
        _paint_rect(frame, bbox)
        _add_noise(frame, rng, sigma=4.0)
        frames.append(frame)
        gts.append(bbox)

    return _SyntheticSequence(
        name="synthetic_linear",
        frames=frames,
        ground_truth=gts,
        attributes={"LINEAR"},
    )


def _build_oscillating(rng: np.random.Generator) -> _SyntheticSequence:
    """Rectangle following a sine-wave horizontal path.

    Amplitude is kept moderate so the target never leaves the frame.
    Stresses the Kalman predictor more than pure constant-velocity motion.
    """
    cx_mid = _W / 2.0
    amplitude = 80.0          # pixels
    period = _N_FRAMES / 2.0  # one full cycle in 30 frames
    y_center = (_H - _BH) / 2.0

    frames: list[np.ndarray] = []
    gts: list[BBox] = []
    for i in range(_N_FRAMES):
        cx = cx_mid + amplitude * math.sin(2.0 * math.pi * i / period)
        x = cx - _BW / 2.0
        bbox = BBox(x=x, y=y_center, w=float(_BW), h=float(_BH))
        frame = _make_frame()
        _paint_rect(frame, bbox)
        _add_noise(frame, rng, sigma=4.0)
        frames.append(frame)
        gts.append(bbox)

    return _SyntheticSequence(
        name="synthetic_oscillating",
        frames=frames,
        ground_truth=gts,
        attributes={"OSCILLATING"},
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@DATASETS.register("synthetic")
class SyntheticDataset:
    """Three short procedurally-generated sequences (no filesystem required).

    Parameters
    ----------
    seed:
        RNG seed for deterministic noise generation.

    Sequences emitted (in order):
      1. ``synthetic_static``      — stationary target, low noise.
      2. ``synthetic_linear``      — constant-velocity horizontal translation.
      3. ``synthetic_oscillating`` — sine-wave path (challenges Kalman predictor).
    """

    name: str = "synthetic"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        # Build sequences eagerly so the dataset is fully deterministic
        # and can be iterated multiple times without re-generating.
        rng = np.random.default_rng(seed)
        self._sequences: list[_SyntheticSequence] = [
            _build_static(rng),
            _build_linear(rng),
            _build_oscillating(rng),
        ]

    def __iter__(self) -> Iterator[_SyntheticSequence]:
        return iter(self._sequences)

    def filter(self, attributes: set[str]) -> "SyntheticDataset":
        """Return a view containing only sequences whose attributes are a
        superset of *attributes*.  Returns a lightweight wrapper so the
        Dataset Protocol is satisfied.
        """
        filtered = [s for s in self._sequences if s.attributes >= attributes]
        obj = object.__new__(SyntheticDataset)
        obj.seed = self.seed
        obj._sequences = filtered
        return obj


__all__ = ["SyntheticDataset"]
