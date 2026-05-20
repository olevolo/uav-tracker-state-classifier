"""Lightweight distractor-aware memory for SALT-RD.

Offline only in Phase 2B — no runtime dependency on full DINO or SAM2.
Follows DAM4SAM design (CVPR2025): RAM + DRM split.
"""

from __future__ import annotations
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class MemoryEntry:
    embedding: np.ndarray    # (D,) normalized embedding vector
    frame_idx: int
    iou: float               # GT IoU at this frame (offline only)
    apce_norm: float
    p_fc: float              # false_confirmed probability
    source: str              # "target_confident" | "secondary_peak" | "false_confirmed_teacher"


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / denom)


class PositiveMemory:
    """Recent target appearance memory (RAM analog).

    FIFO buffer. Updated when tracking is reliable (low p_fc, good IoU).
    Entries are recency-weighted.
    """

    def __init__(self, max_slots: int = 6, update_interval: int = 5):
        self.max_slots = max_slots
        self.update_interval = update_interval
        self._entries: deque[MemoryEntry] = deque()
        self._last_update_frame: int = -999

    def should_update(self, p_fc: float, p_ifd: float, apce_norm: float,
                      current_frame: int = 0, iou: Optional[float] = None) -> bool:
        """Update when: p_fc < 0.20 AND p_ifd < 0.30 AND apce_norm > 0.4
        AND at least update_interval frames since last update."""
        if p_fc >= 0.20:
            return False
        if p_ifd >= 0.30:
            return False
        if apce_norm <= 0.4:
            return False
        if (current_frame - self._last_update_frame) < self.update_interval:
            return False
        return True

    def add(self, entry: MemoryEntry) -> None:
        """FIFO: remove oldest if full."""
        if len(self._entries) >= self.max_slots:
            self._entries.popleft()
        self._entries.append(entry)
        self._last_update_frame = entry.frame_idx

    def _all_embeddings(self) -> Optional[np.ndarray]:
        if not self._entries:
            return None
        return np.stack([e.embedding for e in self._entries])  # (n, D)

    def mean_similarity(self, query_emb: np.ndarray) -> float:
        """Average cosine similarity to all entries."""
        embs = self._all_embeddings()
        if embs is None:
            return 0.0
        sims = np.array([_cosine_sim(query_emb, e) for e in embs])
        return float(sims.mean())

    def max_similarity(self, query_emb: np.ndarray) -> float:
        embs = self._all_embeddings()
        if embs is None:
            return 0.0
        sims = np.array([_cosine_sim(query_emb, e) for e in embs])
        return float(sims.max())

    def recency_weighted_similarity(self, query_emb: np.ndarray) -> float:
        """Weight by 0.9^age (more recent = higher weight).

        age=0 is the most recent entry (last appended).
        """
        if not self._entries:
            return 0.0
        entries = list(self._entries)  # oldest first
        n = len(entries)
        total_weight = 0.0
        weighted_sum = 0.0
        for age, entry in enumerate(reversed(entries)):  # age=0 → most recent
            w = 0.9 ** age
            sim = _cosine_sim(query_emb, entry.embedding)
            weighted_sum += w * sim
            total_weight += w
        if total_weight < 1e-8:
            return 0.0
        return float(weighted_sum / total_weight)

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def frames_since_update(self) -> int:
        """Frames elapsed since the last positive memory update.

        Returns a large sentinel (9999) when memory is empty.
        """
        if self._last_update_frame < 0:
            return 9999
        return 9999  # caller provides current_frame via compute_features

    def reset(self) -> None:
        self._entries.clear()
        self._last_update_frame = -999


class NegativeMemory:
    """Distractor-resolving memory (DRM analog).

    NOT pure FIFO — entries are NOT evicted by age alone (timeless prior).
    Updated only when distractor detected AND tracking reliable.
    Eviction: least-similar-to-current when full.
    """

    def __init__(self, max_slots: int = 6):
        self.max_slots = max_slots
        self._entries: List[MemoryEntry] = []

    def should_update(self, secondary_peak_ratio: float, p_fc: float,
                      apce_norm: float, fc_proxy: bool = False) -> bool:
        """Distractor detected: secondary_peak_ratio > 0.65
        Tracking reliable: p_fc < 0.25 AND apce_norm > 0.4
        Both conditions must hold (DAM4SAM DRM trigger).

        fc_proxy=True: use p_fc directly as distractor signal (bypasses
        the secondary_peak gate; avoids the contradiction where
        secondary_peak_ratio=p_fc/0.4 requires p_fc>0.26 but
        tracking_reliable requires p_fc<0.25).
        """
        if fc_proxy:
            # p_fc is the distractor signal — tracker is likely on wrong object
            return p_fc > 0.40
        distractor_detected = secondary_peak_ratio > 0.65
        tracking_reliable = (p_fc < 0.25) and (apce_norm > 0.4)
        return distractor_detected and tracking_reliable

    def add(self, entry: MemoryEntry) -> None:
        """Evict by least-similar-to-current if full, not oldest."""
        if len(self._entries) < self.max_slots:
            self._entries.append(entry)
            return

        # Evict the entry least similar to the new entry (most dissimilar distractor)
        sims = np.array([_cosine_sim(entry.embedding, e.embedding) for e in self._entries])
        evict_idx = int(np.argmin(sims))
        self._entries[evict_idx] = entry

    def max_similarity(self, query_emb: np.ndarray) -> float:
        if not self._entries:
            return 0.0
        sims = np.array([_cosine_sim(query_emb, e.embedding) for e in self._entries])
        return float(sims.max())

    def mean_similarity(self, query_emb: np.ndarray) -> float:
        if not self._entries:
            return 0.0
        sims = np.array([_cosine_sim(query_emb, e.embedding) for e in self._entries])
        return float(sims.mean())

    def count_nearby(self, query_bbox: np.ndarray, distance_threshold: float = 0.3) -> int:
        """Count entries with spatial proximity to query_bbox.

        Proximity = normalized bbox center distance (by image diagonal approximation).
        query_bbox: (4,) array of [x, y, w, h] in pixel coords.
        Entries without bbox info (bbox=None) are skipped.
        """
        count = 0
        if query_bbox is None or len(query_bbox) < 4:
            return 0
        qcx = query_bbox[0] + query_bbox[2] / 2.0
        qcy = query_bbox[1] + query_bbox[3] / 2.0
        qdiag = max((query_bbox[2] ** 2 + query_bbox[3] ** 2) ** 0.5, 1.0)
        for e in self._entries:
            if not hasattr(e, '_bbox') or e._bbox is None:
                continue
            bbox = e._bbox
            ecx = bbox[0] + bbox[2] / 2.0
            ecy = bbox[1] + bbox[3] / 2.0
            dist = ((qcx - ecx) ** 2 + (qcy - ecy) ** 2) ** 0.5 / qdiag
            if dist < distance_threshold:
                count += 1
        return count

    @property
    def size(self) -> int:
        return len(self._entries)

    def reset(self) -> None:
        self._entries.clear()


class DistractorAwareMemory:
    """Combined RAM+DRM memory state for one tracking episode."""

    FEATURE_NAMES = [
        "mem_pos_max_sim",
        "mem_pos_mean_sim",
        "mem_pos_recency_sim",
        "mem_neg_max_sim",
        "mem_neg_mean_sim",
        "mem_neg_count_nearby",
        "mem_target_minus_distractor_margin",
        "mem_update_age",
        "mem_neg_size",
    ]

    def __init__(self, pos_slots: int = 6, neg_slots: int = 6,
                 update_interval: int = 5):
        self.positive = PositiveMemory(pos_slots, update_interval)
        self.negative = NegativeMemory(neg_slots)
        self._current_frame: int = 0

    def reset(self) -> None:
        """Reset at start of new sequence or after re-init."""
        self.positive.reset()
        self.negative.reset()
        self._current_frame = 0

    def step(
        self,
        frame_idx: int,
        embedding: np.ndarray,
        p_fc: float,
        p_ifd: float,
        apce_norm: float,
        secondary_peak_ratio: float = 0.0,
        iou: Optional[float] = None,
        bbox: Optional[np.ndarray] = None,
        fc_proxy_distractor: bool = False,
    ) -> None:
        """Update memory for one frame.

        fc_proxy_distractor=True: use p_fc>0.40 as distractor signal for
        negative memory instead of secondary_peak_ratio (avoids gate conflict
        when n_secondary=0 in tracker).
        """
        self._current_frame = frame_idx

        # Positive memory update
        if self.positive.should_update(
            p_fc=p_fc,
            p_ifd=p_ifd,
            apce_norm=apce_norm,
            current_frame=frame_idx,
        ):
            entry = MemoryEntry(
                embedding=embedding.copy(),
                frame_idx=frame_idx,
                iou=iou if iou is not None else float("nan"),
                apce_norm=apce_norm,
                p_fc=p_fc,
                source="target_confident",
            )
            self.positive.add(entry)

        # Negative memory update
        if self.negative.should_update(
            secondary_peak_ratio=secondary_peak_ratio,
            p_fc=p_fc,
            apce_norm=apce_norm,
            fc_proxy=fc_proxy_distractor,
        ):
            entry = MemoryEntry(
                embedding=embedding.copy(),
                frame_idx=frame_idx,
                iou=iou if iou is not None else float("nan"),
                apce_norm=apce_norm,
                p_fc=p_fc,
                source="secondary_peak",
            )
            entry._bbox = bbox.copy() if bbox is not None else None  # type: ignore[attr-defined]
            self.negative.add(entry)

    def compute_features(
        self,
        query_emb: np.ndarray,
        query_bbox: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Compute scalar memory features for current frame."""
        pos_max = self.positive.max_similarity(query_emb)
        pos_mean = self.positive.mean_similarity(query_emb)
        pos_rec = self.positive.recency_weighted_similarity(query_emb)
        neg_max = self.negative.max_similarity(query_emb)
        neg_mean = self.negative.mean_similarity(query_emb)
        neg_nearby = float(self.negative.count_nearby(
            query_bbox if query_bbox is not None else np.zeros(4),
        ))
        margin = pos_mean - neg_max

        # Frames since last positive memory update
        if self.positive._last_update_frame < 0:
            update_age = float(9999)
        else:
            update_age = float(self._current_frame - self.positive._last_update_frame)

        neg_size = float(self.negative.size)

        return {
            "mem_pos_max_sim": pos_max,
            "mem_pos_mean_sim": pos_mean,
            "mem_pos_recency_sim": pos_rec,
            "mem_neg_max_sim": neg_max,
            "mem_neg_mean_sim": neg_mean,
            "mem_neg_count_nearby": neg_nearby,
            "mem_target_minus_distractor_margin": margin,
            "mem_update_age": update_age,
            "mem_neg_size": neg_size,
        }
