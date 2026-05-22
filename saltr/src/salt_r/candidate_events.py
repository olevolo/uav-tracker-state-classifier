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
        detector_score=candidate.score,
        score_map_score=candidate.score if source=="score_map" else None,
        geometry_ratio=_area_ratio,
        cosine_sim=best_sim,
        accepted=True,                   # geometry guard passed
        tracker_bbox=track_state.bbox,
        frame_idx_post_reinit=self._frame_idx,
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
    detector_score: Optional[float]    # YOLO/RT-DETR confidence, None if score_map
    score_map_score: Optional[float]   # score-map peak value, None if detector-only

    # Geometry guard features (from BUG-19 fix)
    geometry_area_ratio: float      # candidate_area / last_good_bbox_area
    frame_area_ratio: float         # candidate_area / frame_area

    # Appearance guard (cosine sim from _best_detection)
    cosine_sim: float               # cosine sim between candidate embed and ref embed

    # Execution outcome
    accepted: bool                  # True if geometry guard and cosine guard passed
    reject_reason: Optional[str]    # "size_ratio" | "frame_area" | "in_frame" | "cosine" | None

    # Frame dimensions for bbox normalization (BUG-27 fix)
    frame_h: int = 0                # frame height at event time; 0 = not recorded
    frame_w: int = 0                # frame width at event time; 0 = not recorded
    # Relative position feature (Track A pivot — dist to last known target)
    dist_from_last: float = 0.0     # distance(candidate_center, last_good_bbox_center) / frame_diagonal

    # Offline GT labels (filled by labeling pass, None at collection time)
    candidate_iou: Optional[float] = None        # IoU(candidate_bbox, gt_bbox[frame_idx])
    future_iou_gain: Optional[float] = None      # mean IoU[+1..+20] - IoU[frame]
    label_good_candidate: Optional[int] = None  # 1 if IoU>0.3 AND gain>0, else 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

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
        detector_score: Optional[float],
        score_map_score: Optional[float],
        geometry_area_ratio: float,
        frame_area_ratio: float,
        cosine_sim: float,
        accepted: bool,
        reject_reason: Optional[str] = None,
        seq_id: Optional[str] = None,
        frame_h: int = 0,
        frame_w: int = 0,
        dist_from_last: float = 0.0,
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
            detector_score=detector_score,
            score_map_score=score_map_score,
            geometry_area_ratio=geometry_area_ratio,
            frame_area_ratio=frame_area_ratio,
            cosine_sim=cosine_sim,
            accepted=accepted,
            reject_reason=reject_reason,
            frame_h=frame_h,
            frame_w=frame_w,
            dist_from_last=dist_from_last,
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
