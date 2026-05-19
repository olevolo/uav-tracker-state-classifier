"""Cosine-similarity appearance memory with exponential forgetting.

Self-learning: no offline training needed. Stores target appearance embeddings
computed from tracker ROI crops and updates them online during tracking.

Registration: APPEARANCE_MEMORIES["cosine_memory"]
"""

from __future__ import annotations

import numpy as np

from uav_tracker.registry import APPEARANCE_MEMORIES
from uav_tracker.types import AppearanceTemplate, BBox, FrameContext, TrackState

# Fixed random projection matrix — seeded at 42, built once at module load.
# Shape: (32 * 32 * 3, embedding_dim_max) where embedding_dim_max = 512.
# Crop is resized to 32×32 (was 64×64) — 4× cheaper matmul, same 64-d output.
_PROJ_SEED = 42
_PROJ_IN = 32 * 32 * 3  # 3072 — was 12288 (64×64×3), ~12× fewer FLOPs/call
_PROJ_DIM_MAX = 512

_rng_proj = np.random.default_rng(_PROJ_SEED)
_PROJECTION_MATRIX: np.ndarray = _rng_proj.standard_normal(
    (_PROJ_IN, _PROJ_DIM_MAX)
).astype(np.float32)


def _build_projection(embedding_dim: int) -> np.ndarray:
    """Return a (3072, embedding_dim) projection matrix (float32, fixed seed)."""
    if embedding_dim > _PROJ_DIM_MAX:
        raise ValueError(
            f"embedding_dim {embedding_dim} exceeds maximum {_PROJ_DIM_MAX}"
        )
    return _PROJECTION_MATRIX[:, :embedding_dim]


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D vector in place; return the result."""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return v
    return v / norm


def _crop_and_resize(frame: np.ndarray, bbox: BBox, target_size: int = 64) -> np.ndarray:
    """Crop a 2x context region around *bbox* and resize to target_size × target_size.

    Returns a float32 array of shape (target_size, target_size, 3) with values
    in [0, 1].  Does NOT require OpenCV; uses only NumPy + stdlib.
    """
    import cv2  # lightweight — already a hard dep of this project

    h, w = frame.shape[:2]
    cx = bbox.x + bbox.w / 2.0
    cy = bbox.y + bbox.h / 2.0
    half_w = bbox.w
    half_h = bbox.h

    x1 = int(max(0, cx - half_w))
    y1 = int(max(0, cy - half_h))
    x2 = int(min(w, cx + half_w))
    y2 = int(min(h, cy + half_h))

    if x2 <= x1 or y2 <= y1:
        # Degenerate bbox — return zeros
        return np.zeros((target_size, target_size, 3), dtype=np.float32)

    crop = frame[y1:y2, x1:x2]
    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32) / 255.0


@APPEARANCE_MEMORIES.register("cosine_memory")
class CosineAppearanceMemory:
    """Cosine-similarity appearance memory with exponential forgetting.

    Stores up to *max_templates* L2-normalised embeddings computed from tracker
    ROI crops.  Embeddings are produced via a deterministic random projection
    (no neural net required) so the module works with only NumPy.

    Protocol compliance: implements ``AppearanceMemory`` from
    ``uav_tracker.ml.appearance_memory.base``.
    """

    name: str = "cosine_memory"
    max_templates: int = 50
    forgetting_factor: float = 0.95

    def __init__(
        self,
        max_templates: int = 50,
        forgetting_factor: float = 0.95,
        store_interval: int = 10,
        min_confidence: float = 0.6,
        embedding_dim: int = 64,
    ) -> None:
        self.max_templates = max_templates
        self.forgetting_factor = forgetting_factor
        self.store_interval = store_interval
        self.min_confidence = min_confidence
        self.embedding_dim = embedding_dim

        self._proj: np.ndarray = _build_projection(embedding_dim)
        self._templates: list[AppearanceTemplate] = []
        self._frame_count: int = 0

    # ---------------------------------------------------------------------- #
    # Protocol methods                                                         #
    # ---------------------------------------------------------------------- #

    def store(self, ctx: FrameContext, state: TrackState) -> None:
        """Extract embedding from ROI crop and store if conditions are met.

        Steps:
        1. Honour ``store_interval`` — skip frames between stores.
        2. Skip if ``state.confidence < min_confidence``.
        3. Crop ROI (2× context), resize to 32×32, flatten, project, L2-norm.
        4. Decay existing weights by ``forgetting_factor``.
        5. Append new template (weight=1.0).
        6. Evict lowest-weight template if over capacity.
        """
        self._frame_count += 1

        # Gate 1: interval
        if (self._frame_count - 1) % self.store_interval != 0:
            return

        # Gate 2: confidence
        if state.confidence < self.min_confidence:
            return

        embedding = self._extract_embedding(ctx.frame, state.bbox)

        # Decay existing weights
        for t in self._templates:
            t.weight *= self.forgetting_factor

        # Store new template
        self._templates.append(
            AppearanceTemplate(
                embedding=embedding,
                bbox=state.bbox,
                frame_idx=ctx.frame_idx,
                weight=1.0,
            )
        )

        # Evict if over capacity
        if len(self._templates) > self.max_templates:
            min_idx = int(np.argmin([t.weight for t in self._templates]))
            self._templates.pop(min_idx)

    def retrieve_best(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> list[AppearanceTemplate]:
        """Return up to *top_k* templates sorted by cosine similarity (desc)."""
        if not self._templates:
            return []

        q = _l2_normalize(query_embedding.astype(np.float32))
        sims = np.array(
            [float(np.dot(q, t.embedding)) for t in self._templates], dtype=np.float32
        )
        k = min(top_k, len(self._templates))
        top_indices = np.argsort(sims)[::-1][:k]
        return [self._templates[i] for i in top_indices]

    def compute_drift(self) -> float:
        """Return cosine *distance* between oldest and newest template.

        Returns 0.0 if fewer than 2 templates are stored.  The result lies in
        [0, 1] because both vectors are L2-normalised (cosine similarity ∈ [-1, 1]
        → distance = (1 - sim) / 2 ∈ [0, 1]).
        """
        if len(self._templates) < 2:
            return 0.0

        oldest = self._templates[0].embedding
        newest = self._templates[-1].embedding
        cos_sim = float(np.dot(oldest, newest))
        # Clamp for floating-point safety
        cos_sim = max(-1.0, min(1.0, cos_sim))
        return (1.0 - cos_sim) / 2.0

    def get_best_template(self) -> AppearanceTemplate | None:
        """Return the template with the highest weight, or *None* if empty."""
        if not self._templates:
            return None
        return max(self._templates, key=lambda t: t.weight)

    def reset(self) -> None:
        """Clear all stored templates and reset the frame counter."""
        self._templates.clear()
        self._frame_count = 0

    # ---------------------------------------------------------------------- #
    # Internal helpers                                                         #
    # ---------------------------------------------------------------------- #

    def _extract_embedding(self, frame: np.ndarray, bbox: BBox) -> np.ndarray:
        """Crop, resize, flatten, project, and L2-normalise into an embedding."""
        patch = _crop_and_resize(frame, bbox, target_size=32)  # was 64; 4× fewer pixels
        flat = patch.flatten()
        projected = flat @ self._proj
        return _l2_normalize(projected)
