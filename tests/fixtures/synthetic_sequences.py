"""Synthetic sequence generators for tests.

Not application code — purely fixture utilities. Kept small so tests
stay fast; larger fixtures live under ``tests/fixtures/data/`` (not
shipped here).

All generators accept a numpy ``Generator`` so callers can pin seeds
for bit-stable tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class SyntheticSequence:
    """Lightweight sequence: frames + ground-truth bboxes + init bbox."""

    name: str
    frames: list[np.ndarray]
    ground_truth: list[tuple[float, float, float, float]]  # xywh per frame
    init_bbox: tuple[float, float, float, float]
    attributes: frozenset[str] = frozenset()


def _blank(h: int, w: int, value: int = 32) -> np.ndarray:
    """Solid-gray uint8 RGB frame as background."""
    frame = np.full((h, w, 3), value, dtype=np.uint8)
    return frame


def _paint_rect(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> None:
    """In-place paint a white rectangle on ``frame``."""
    x, y, w, h = bbox
    x0, y0 = int(round(x)), int(round(y))
    x1, y1 = int(round(x + w)), int(round(y + h))
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(frame.shape[1], x1)
    y1 = min(frame.shape[0], y1)
    frame[y0:y1, x0:x1] = 220


def translating_rectangle(
    n_frames: int = 20,
    frame_size: tuple[int, int] = (120, 160),
    bbox_size: tuple[int, int] = (24, 24),
    velocity: tuple[float, float] = (2.0, 1.0),
    start: tuple[float, float] = (10.0, 10.0),
) -> SyntheticSequence:
    """Smooth linear translation — integration sanity test baseline.

    KCF+Kalman should track this near-perfectly (IoU > 0.8 steady).
    """
    h, w = frame_size
    bw, bh = bbox_size
    frames: list[np.ndarray] = []
    gts: list[tuple[float, float, float, float]] = []
    for i in range(n_frames):
        x = start[0] + velocity[0] * i
        y = start[1] + velocity[1] * i
        frame = _blank(h, w)
        _paint_rect(frame, (x, y, bw, bh))
        frames.append(frame)
        gts.append((x, y, float(bw), float(bh)))
    return SyntheticSequence(
        name="smooth_translation",
        frames=frames,
        ground_truth=gts,
        init_bbox=gts[0],
    )


def erratic_motion(
    n_frames: int = 20,
    frame_size: tuple[int, int] = (120, 160),
    bbox_size: tuple[int, int] = (24, 24),
    start: tuple[float, float] = (60.0, 50.0),
    step: float = 6.0,
    rng: np.random.Generator | None = None,
) -> SyntheticSequence:
    """Random-walk translation — stresses the KCF→SiamFC switch.

    Intended for Phase 3/4 integration tests where the scheduler should
    escalate to tier 1.
    """
    rng = rng or np.random.default_rng(42)
    h, w = frame_size
    bw, bh = bbox_size
    frames: list[np.ndarray] = []
    gts: list[tuple[float, float, float, float]] = []
    x, y = start
    for _ in range(n_frames):
        frame = _blank(h, w)
        _paint_rect(frame, (x, y, bw, bh))
        frames.append(frame)
        gts.append((x, y, float(bw), float(bh)))
        # Random walk with reflecting boundaries.
        dx, dy = rng.uniform(-step, step, size=2)
        x = float(np.clip(x + dx, 0.0, w - bw))
        y = float(np.clip(y + dy, 0.0, h - bh))
    return SyntheticSequence(
        name="erratic",
        frames=frames,
        ground_truth=gts,
        init_bbox=gts[0],
    )


def disappear_reappear(
    n_frames: int = 30,
    frame_size: tuple[int, int] = (120, 160),
    bbox_size: tuple[int, int] = (24, 24),
    start: tuple[float, float] = (20.0, 20.0),
    gone_range: tuple[int, int] = (10, 20),
) -> SyntheticSequence:
    """Rectangle that vanishes mid-sequence then reappears elsewhere.

    Used by Phase 6 detection-tier integration tests.
    """
    h, w = frame_size
    bw, bh = bbox_size
    frames: list[np.ndarray] = []
    gts: list[tuple[float, float, float, float]] = []
    x, y = start
    for i in range(n_frames):
        frame = _blank(h, w)
        visible = not (gone_range[0] <= i < gone_range[1])
        if visible:
            _paint_rect(frame, (x, y, bw, bh))
            if i >= gone_range[1]:
                # Reappear somewhere else after the gap.
                x = w - bw - 20
        frames.append(frame)
        gts.append((x, y, float(bw), float(bh)))
    return SyntheticSequence(
        name="disappear_reappear",
        frames=frames,
        ground_truth=gts,
        init_bbox=gts[0],
    )


def as_frame_iterable(seq: SyntheticSequence) -> Iterable[np.ndarray]:
    """Convenience: plain iterable over frames (matches Dataset Protocol)."""
    return iter(seq.frames)


# --------------------------------------------------------------------------- #
# Phase 4: entropy-property fixture generators                                #
# --------------------------------------------------------------------------- #


def translating_rectangle_entropy(
    n_frames: int = 30,
    frame_size: tuple[int, int] = (240, 320),
    bbox_size: tuple[int, int] = (60, 45),
    velocity: tuple[float, float] = (3.0, 0.0),
    start: tuple[float, float] = (20.0, 98.0),
    noise_sigma: float = 15.0,
    rng: np.random.Generator | None = None,
) -> SyntheticSequence:
    """Rectangle translates linearly on a STABLE textured noise background.

    Both the background and the rectangle have rich Gaussian texture so that
    Shi-Tomasi finds many corners everywhere.  The background texture is
    generated once and held fixed across all frames.  A different constant
    texture is painted inside the rectangle.  Only the rectangle moves at
    constant velocity — the background is entirely static.

    Since the camera does not move, RANSAC estimates zero global (camera)
    motion, and the residual flow after subtraction equals the raw local flow
    (all pointing right).  Coherent unidirectional flow → low orientation
    entropy.  Target: final ``H̄ < 0.20`` after EMA warm-up.

    Parameters
    ----------
    noise_sigma:
        Std-dev of Gaussian noise added to both background and target
        textures (high value ensures Shi-Tomasi finds many corners).
    rng:
        Optional seeded Generator for bit-stable tests; defaults to
        Generator(42).
    """
    rng = rng or np.random.default_rng(42)
    h, w = frame_size
    bw, bh = bbox_size
    frames: list[np.ndarray] = []
    gts: list[tuple[float, float, float, float]] = []

    # Generate the background texture ONCE and keep it fixed.
    bg_base = np.full((h, w, 3), 40, dtype=np.uint8)
    bg_noise = rng.normal(0, noise_sigma, (h, w, 3))
    bg_texture = np.clip(bg_base.astype(np.float32) + bg_noise, 0, 255).astype(np.uint8)

    # Generate the target texture ONCE (different brightness level).
    tgt_base = np.full((bh, bw, 3), 160, dtype=np.uint8)
    tgt_noise = rng.normal(0, noise_sigma, (bh, bw, 3))
    tgt_texture = np.clip(tgt_base.astype(np.float32) + tgt_noise, 0, 255).astype(np.uint8)

    for i in range(n_frames):
        x = start[0] + velocity[0] * i
        y = start[1] + velocity[1] * i
        x = float(np.clip(x, 0, w - bw))
        y = float(np.clip(y, 0, h - bh))

        # Compose: stable background + moving textured target.
        frame = bg_texture.copy()
        x0 = int(round(x))
        y0 = int(round(y))
        x1 = min(w, x0 + bw)
        y1 = min(h, y0 + bh)
        frame[y0:y1, x0:x1] = tgt_texture[: y1 - y0, : x1 - x0]

        frames.append(frame)
        gts.append((x, y, float(bw), float(bh)))

    return SyntheticSequence(
        name="translating_rectangle_entropy",
        frames=frames,
        ground_truth=gts,
        init_bbox=gts[0],
        attributes=frozenset({"LINEAR", "LOW_ENTROPY"}),
    )


def noisy_rectangle_entropy(
    n_frames: int = 30,
    frame_size: tuple[int, int] = (240, 320),
    bbox_size: tuple[int, int] = (60, 45),
    jitter_std: float = 8.0,
    start: tuple[float, float] = (130.0, 98.0),
    noise_sigma: float = 15.0,
    rng: np.random.Generator | None = None,
) -> SyntheticSequence:
    """Rectangle moves with random-direction jitter on a stable textured background.

    Both the background and the rectangle have rich Gaussian texture (same
    approach as ``translating_rectangle_entropy``).  The background is static
    (global motion ≈ zero); the rectangle jumps in a different random direction
    each frame (drawn from N(0, jitter_std)).

    After background global-motion subtraction the residual ROI flow vectors
    point in random directions each frame → incoherent orientation histogram →
    high entropy.  Target: final ``H̄ > 0.75`` after EMA warm-up.

    Parameters
    ----------
    jitter_std:
        Std-dev of the per-frame random displacement in pixels.  Higher
        values produce more incoherent flow.
    noise_sigma:
        Background and target texture noise std-dev.
    rng:
        Optional seeded Generator; defaults to Generator(42).
    """
    rng = rng or np.random.default_rng(42)
    h, w = frame_size
    bw, bh = bbox_size
    frames: list[np.ndarray] = []
    gts: list[tuple[float, float, float, float]] = []

    # Stable background texture.
    bg_base = np.full((h, w, 3), 40, dtype=np.uint8)
    bg_noise = rng.normal(0, noise_sigma, (h, w, 3))
    bg_texture = np.clip(bg_base.astype(np.float32) + bg_noise, 0, 255).astype(np.uint8)

    # Stable target texture (different brightness from background).
    tgt_base = np.full((bh, bw, 3), 160, dtype=np.uint8)
    tgt_noise = rng.normal(0, noise_sigma, (bh, bw, 3))
    tgt_texture = np.clip(tgt_base.astype(np.float32) + tgt_noise, 0, 255).astype(np.uint8)

    x, y = start
    for _ in range(n_frames):
        x = float(np.clip(x, 0, w - bw))
        y = float(np.clip(y, 0, h - bh))

        frame = bg_texture.copy()
        x0 = int(round(x))
        y0 = int(round(y))
        x1 = min(w, x0 + bw)
        y1 = min(h, y0 + bh)
        frame[y0:y1, x0:x1] = tgt_texture[: y1 - y0, : x1 - x0]

        frames.append(frame)
        gts.append((x, y, float(bw), float(bh)))

        # Random-direction jitter for next frame.
        dx, dy = rng.normal(0, jitter_std, size=2)
        x = float(np.clip(x + dx, 0, w - bw))
        y = float(np.clip(y + dy, 0, h - bh))

    return SyntheticSequence(
        name="noisy_rectangle_entropy",
        frames=frames,
        ground_truth=gts,
        init_bbox=gts[0],
        attributes=frozenset({"ERRATIC", "HIGH_ENTROPY"}),
    )
