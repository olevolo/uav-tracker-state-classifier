"""Candidate-generation helpers for the FC recover controller.

Phase 0 oracle on UAV123 (top-10 hardest sequences) showed that SGLATrack's
internal multi-crop redetect alone has **0% recall@IoU.5** on true-FC frames
— the candidate set rarely contains the actual target. Phase 2's answer is to
add an *event-triggered* external detector (RT-DETRv2-S) that fires only
during a challenge episode (1-15 frames per FC streak, not per-frame), so
average FPS overhead stays small while the candidate pool gains class-
agnostic boxes that the SGLATrack backbone alone misses.

The generator merges two sources into a single ``list[dict]`` for the
verifier:

  1. SGLATrack ``tracker.redetect(top_k=K)`` — internal multi-crop pyramid
     (legacy V3 path).
  2. RT-DETRv2-S detection ``boxes -> SGLATrack-template-scoring`` to get
     per-box appearance embedding + sim_to_init in SGLATrack's space (so the
     verifier compares apples to apples regardless of source).

Each detector box becomes a candidate by passing it as ``anchor_bbox`` to a
narrow SGLATrack redetect (``factors=(4.0,)``, no grid). That forward gives
the box a score-map peak embedding + sim_to_init in 192-D SGLATrack token
space without an extra encoder.

Source defaults are conservative: when RT-DETR cannot be loaded (no weights
or repo missing), the generator silently degrades to SGLA-only. The
controller never blocks on detector init.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from uav_tracker.types import BBox

logger = logging.getLogger(__name__)


@dataclass
class CandidateGeneratorConfig:
    """Knobs for the merged-source generator."""

    sgla_top_k: int = 5
    """Top-K spatially-distinct candidates from SGLATrack's internal redetect."""

    sgla_factors: tuple[float, ...] = (8.0, 12.0, 16.0)
    sgla_grid_size: int = 3
    sgla_max_candidates_per_crop: int = 5

    use_external_detector: bool = True
    """Whether to query RT-DETRv2-S during a CHALLENGE step (event-triggered)."""

    detector_top_k: int = 5
    """Cap on the number of RT-DETR boxes per call (sorted by proximity to
    the last-CC hint, so distant detections are dropped before scoring)."""

    detector_conf_threshold: float = 0.30
    detector_anchor_factor: float = 4.0
    """Search factor used when scoring a detector box through SGLATrack's
    backbone — the standard tracker search factor (no widening) so the
    embedding is comparable to the tracker's native template/search space."""


def _detection_to_bbox(det: Any) -> Optional[BBox]:
    """Adapt a detector return (object with .bbox or dict) to BBox."""
    if det is None:
        return None
    box = getattr(det, "bbox", None)
    if box is None and isinstance(det, dict):
        box = det.get("bbox")
    if box is None:
        return None
    if hasattr(box, "x") and hasattr(box, "w"):
        return BBox(x=float(box.x), y=float(box.y),
                     w=float(box.w), h=float(box.h))
    if isinstance(box, dict):
        return BBox(x=float(box["x"]), y=float(box["y"]),
                     w=float(box["w"]), h=float(box["h"]))
    seq = list(box)
    if len(seq) >= 4:
        return BBox(x=float(seq[0]), y=float(seq[1]),
                     w=float(seq[2]), h=float(seq[3]))
    return None


def _score_detector_box_with_tracker(tracker, frame, det_bbox: BBox,
                                       factor: float) -> Optional[dict]:
    """Run a single SGLATrack ``redetect`` forward at the detector box.

    Returns a candidate dict in the tracker.redetect format (with embedding
    and sim_to_init populated when the tracker exposes them), or None on
    failure. Side-effect-light: never mutates tracker state.
    """
    redetect = getattr(tracker, "redetect", None)
    if redetect is None:
        return None
    try:
        cand = redetect(
            frame,
            factors=(float(factor),),
            anchor_bboxes=[det_bbox],
            include_current=False,
            grid_size=0,
            max_candidates=1,
            top_k=1,
            rank_by="quality",
        )
    except Exception as exc:
        logger.debug("detector-box redetect failed: %s", exc)
        return None
    if cand is None:
        return None
    if isinstance(cand, list):
        return cand[0] if cand else None
    return cand


class MultiSourceCandidateGenerator:
    """Generator that merges SGLA-internal + (optional) RT-DETR candidates.

    Usage in run_with_csc::

        gen = MultiSourceCandidateGenerator(detector=rt_detr_or_none)
        # During a challenge step:
        cands, ms = gen(tracker, frame, last_cc_bbox=last_cc_bbox)

    Returns ``(list[dict], elapsed_ms)``. Each dict is the SGLATrack.redetect
    candidate format augmented with a ``"source"`` field ('sgla' or
    'rtdetr') so downstream metrics can attribute switches.
    """

    def __init__(
        self,
        detector: Optional[Any] = None,
        config: Optional[CandidateGeneratorConfig] = None,
    ) -> None:
        self.detector = detector
        self.config = config or CandidateGeneratorConfig()

    def __call__(
        self,
        tracker,
        frame: np.ndarray,
        *,
        last_cc_bbox: Optional[BBox] = None,
        frame_idx: int = -1,
    ) -> tuple[list[dict], float]:
        cfg = self.config
        t0 = time.perf_counter()
        cands: list[dict] = []

        # ---- Source 1: SGLATrack internal multi-crop redetect ----
        sgla_redetect = getattr(tracker, "redetect", None)
        if sgla_redetect is not None:
            try:
                sgla_cands = sgla_redetect(
                    frame,
                    factors=tuple(cfg.sgla_factors),
                    grid_size=int(cfg.sgla_grid_size),
                    max_candidates=int(cfg.sgla_max_candidates_per_crop),
                    top_k=int(cfg.sgla_top_k),
                    rank_by="quality",
                    frame_idx=int(frame_idx),
                )
            except TypeError:
                # Older redetect signatures may not accept frame_idx.
                sgla_cands = sgla_redetect(
                    frame,
                    factors=tuple(cfg.sgla_factors),
                    grid_size=int(cfg.sgla_grid_size),
                    max_candidates=int(cfg.sgla_max_candidates_per_crop),
                    top_k=int(cfg.sgla_top_k),
                    rank_by="quality",
                )
            except Exception as exc:
                logger.debug("SGLA redetect failed: %s", exc)
                sgla_cands = None

            if sgla_cands:
                if isinstance(sgla_cands, dict):
                    sgla_cands = [sgla_cands]
                for c in sgla_cands:
                    c.setdefault("source", "sgla")
                    cands.append(c)

        # ---- Source 2: RT-DETR detector boxes -> tracker-scoring ----
        if cfg.use_external_detector and self.detector is not None:
            try:
                detections = self.detector.detect(
                    frame,
                    hint_bbox=last_cc_bbox if last_cc_bbox is not None else None,
                )
            except Exception as exc:
                logger.debug("external detector failed: %s", exc)
                detections = []

            for det in detections[: int(cfg.detector_top_k)]:
                det_bbox = _detection_to_bbox(det)
                if det_bbox is None or det_bbox.w <= 0 or det_bbox.h <= 0:
                    continue
                cand = _score_detector_box_with_tracker(
                    tracker, frame, det_bbox, factor=cfg.detector_anchor_factor)
                if cand is None:
                    continue
                cand["source"] = "rtdetr"
                cand["detector_score"] = float(getattr(det, "confidence", 0.0))
                cand["detector_class"] = int(getattr(det, "class_id", -1))
                cands.append(cand)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return cands, elapsed_ms


__all__ = [
    "CandidateGeneratorConfig",
    "MultiSourceCandidateGenerator",
]
