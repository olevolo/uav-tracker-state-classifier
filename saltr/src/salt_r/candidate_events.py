"""candidate_events.py — per-reinit candidate event logger.

Collects every proposed reinit candidate during live SALTRunner inference so the
events can be labeled offline (GT IoU + future utility) and used to train the
candidate scorer head in SALTRDPolicyNet.

Design goals
------------
- Zero runtime cost when disabled (logger.enabled = False by default).
- No dependency on tracking ground truth at collection time; GT IoU is added
  offline by a separate labeling pass (label_candidate_events.py).
- Schema matches the fields required by OracleReinitDataset candidate extension
  (BUG-26 items b/c) once that dataset is built.

Usage (in SALTRunner._step, around recovery execution)
-------------------------------------------------------
    runner.candidate_logger.record(
        frame_idx=self._frame_idx,
        seq_id=current_seq_id,
        candidate_bbox=candidate_bbox,   # (x, y, w, h)
        source=candidate.source,         # "score_map" | "detector"
        score_map_score=candidate.score if source=="score_map" else None,
        frame_area_ratio=_area_ratio,
        accepted=True,                   # geometry guard passed
        tracker_bbox=track_state.bbox,
        dist_from_last=_dist_from_last,
        crop_sim=_crop_sim,
        aspect_ratio_delta=_aspect_ratio_delta,
        size_delta_ratio=_size_delta_ratio,
    )

Offline labeling
----------------
    python -m salt_r.label_candidate_events \\
        --events saltr/results/candidate_events.jsonl \\
        --npz    saltr/data/salt_rd_v2_labels.npz \\
        --output saltr/data/candidate_events_labeled.npz

The labeling pass adds:
    - candidate_iou: IoU(candidate_bbox, gt_bbox[frame_idx])
    - future_iou_gain: mean IoU[frame+1..frame+20] - IoU[frame]
    - label_good_candidate: (candidate_iou > 0.3) AND (future_iou_gain > 0)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class CandidateEvent:
    """One proposed reinit candidate event during live inference."""

    frame_idx: int
    seq_id: str
    timestamp_s: float

    # Candidate geometry
    candidate_bbox: List[float]     # [x, y, w, h] — proposed reinit bbox
    tracker_bbox: List[float]       # [x, y, w, h] — current tracker estimate

    # Candidate source and scores
    source: str                     # "score_map" | "detector"
    score_map_score: Optional[float]   # score-map peak value, None if detector-only

    # Geometry guard features (from BUG-19 fix)
    frame_area_ratio: float         # candidate_area / frame_area

    # Execution outcome
    accepted: bool                  # True if geometry guard and cosine guard passed
    reject_reason: Optional[str]    # "size_ratio" | "frame_area" | "in_frame" | "cosine" | None

    # Frame dimensions for bbox normalization (BUG-27 fix)
    frame_h: int = 0                # frame height at event time; 0 = not recorded
    frame_w: int = 0                # frame width at event time; 0 = not recorded
    # Relative position feature (Track A pivot — dist to last known target)
    dist_from_last: float = 0.0     # distance(candidate_center, last_good_bbox_center) / frame_diagonal
    # Identity signal features (v2 candidate schema — crop_sim via MobileNetV3)
    crop_sim: float = 0.0           # cosine sim between MobileNetV3 embeddings of candidate and template crops
    aspect_ratio_delta: float = 0.0 # |cand_w/cand_h - tmpl_w/tmpl_h|
    size_delta_ratio: float = 0.0   # |cand_area - tmpl_area| / tmpl_area, clipped to [0, 1]

    # Offline GT labels (filled by labeling pass, None at collection time)
    candidate_iou: Optional[float] = None           # IoU(candidate_bbox, gt_bbox[frame_idx])
    future_iou_gain: Optional[float] = None         # mean IoU[+1..+20] - IoU[frame]; may be 0 when oracle absent
    label_good_candidate: Optional[int] = None      # legacy: 1 if IoU>0.3 AND gain>0 — do NOT use as training gate
    candidate_correct_iou03: Optional[int] = None   # 1 if candidate_iou >= 0.30 (primary training label)
    candidate_correct_iou05: Optional[int] = None   # 1 if candidate_iou >= 0.50 (stricter report)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_feature_vector(self) -> "np.ndarray":
        """Return v2 candidate feature vector (8-dim) matching FEATURE_NAMES order.

        Index layout (matches feature_schema.FEATURE_NAMES):
            0: score_map_score
            1: bbox_h
            2: frame_area_ratio
            3: bbox_w
            4: dist_from_last
            5: crop_sim
            6: aspect_ratio_delta
            7: size_delta_ratio
        """
        import numpy as np
        bb = self.candidate_bbox  # [x, y, w, h]
        return np.array([
            float(self.score_map_score or 0.0),  # 0: score_map_score
            float(bb[3]) if len(bb) > 3 else 0.0,  # 1: bbox_h
            float(self.frame_area_ratio),            # 2: frame_area_ratio
            float(bb[2]) if len(bb) > 2 else 0.0,  # 3: bbox_w
            float(self.dist_from_last),              # 4: dist_from_last
            float(self.crop_sim),                    # 5: crop_sim
            float(self.aspect_ratio_delta),          # 6: aspect_ratio_delta
            float(self.size_delta_ratio),            # 7: size_delta_ratio
        ], dtype=np.float32)

    def as_array(self) -> "np.ndarray":
        """Alias for to_feature_vector()."""
        return self.to_feature_vector()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CandidateEvent":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class CandidateEventLogger:
    """Accumulates CandidateEvent records during a tracking run.

    Disabled by default; enable with logger.enabled = True before calling runner.run().
    Thread-unsafe — single-sequence use only.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._events: List[CandidateEvent] = []
        self._current_seq_id: str = ""

    def reset(self, seq_id: str = "") -> None:
        """Call at the start of each sequence."""
        self._current_seq_id = seq_id
        if self.enabled:
            self._events.clear()

    def record(
        self,
        frame_idx: int,
        candidate_bbox: tuple[float, float, float, float],
        tracker_bbox: tuple[float, float, float, float],
        source: str,
        score_map_score: Optional[float],
        frame_area_ratio: float,
        accepted: bool,
        reject_reason: Optional[str] = None,
        seq_id: Optional[str] = None,
        frame_h: int = 0,
        frame_w: int = 0,
        dist_from_last: float = 0.0,
        crop_sim: float = 0.0,
        aspect_ratio_delta: float = 0.0,
        size_delta_ratio: float = 0.0,
        # Legacy parameters kept for backward compatibility (ignored)
        detector_score: Optional[float] = None,
        geometry_area_ratio: Optional[float] = None,
        cosine_sim: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return
        self._events.append(CandidateEvent(
            frame_idx=frame_idx,
            seq_id=seq_id or self._current_seq_id,
            timestamp_s=time.time(),
            candidate_bbox=list(candidate_bbox),
            tracker_bbox=list(tracker_bbox),
            source=source,
            score_map_score=score_map_score,
            frame_area_ratio=frame_area_ratio,
            accepted=accepted,
            reject_reason=reject_reason,
            frame_h=frame_h,
            frame_w=frame_w,
            dist_from_last=dist_from_last,
            crop_sim=crop_sim,
            aspect_ratio_delta=aspect_ratio_delta,
            size_delta_ratio=size_delta_ratio,
        ))

    def events(self) -> List[CandidateEvent]:
        return list(self._events)

    def save_jsonl(self, path: str | Path) -> int:
        """Append all recorded events to a JSONL file. Returns number written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with path.open("a") as f:
            for ev in self._events:
                f.write(json.dumps(ev.to_dict()) + "\n")
                n += 1
        return n

    @staticmethod
    def load_jsonl(path: str | Path) -> List[CandidateEvent]:
        events = []
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(CandidateEvent.from_dict(json.loads(line)))
        return events
