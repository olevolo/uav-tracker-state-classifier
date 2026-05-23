"""evidence.py — SALT-RD typed evidence frames for Phase 1C.

Converts raw tracker outputs into EvidenceFrame objects.  This module is
deliberately isolated from tracking decisions:

- No imports from TSA or tracker state modules
- No TrackerAction / TargetState references
- No thresholds or decisions
- No detector calls
- Pure numeric extraction
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from csc_uav_tracking.telemetry.schema import (
    BASE_FEATURE_NAMES,
    PRODUCTION_ZERO_FEATURE_INDICES,
    validate_feature_matrix,
    zero_production_features,
)

BBox = tuple[float, float, float, float]  # x, y, w, h


@dataclass
class CandidateEvidence:
    """Single score-map or detector candidate."""

    bbox: BBox
    score: float
    rank: int
    score_ratio_to_top: float          # score / top_candidate_score
    distance_to_tracker: float         # pixel distance from candidate center to tracker bbox center
    distance_to_prev_bbox: float       # pixel distance from candidate center to prev bbox center
    size_ratio_to_tracker: float       # candidate area / tracker bbox area
    source: str                        # "score_map" | "detector"
    detector_score: float | None = None
    teacher_score: float | None = None  # offline teacher identity score, None at runtime

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "score": self.score,
            "rank": self.rank,
            "score_ratio_to_top": self.score_ratio_to_top,
            "distance_to_tracker": self.distance_to_tracker,
            "distance_to_prev_bbox": self.distance_to_prev_bbox,
            "size_ratio_to_tracker": self.size_ratio_to_tracker,
            "source": self.source,
            "detector_score": self.detector_score,
            "teacher_score": self.teacher_score,
        }


@dataclass
class TemplateContext:
    last_update_age: int = 0       # frames since last template update
    update_count: int = 0          # total template updates so far


@dataclass
class RecoveryContext:
    last_reinit_age: int = -1      # frames since last reinit, -1 if never
    total_reinit_count: int = 0


@dataclass
class EvidenceFrame:
    frame_idx: int
    bbox: BBox
    base_features: np.ndarray               # shape (28,), production flow zeros applied
    score_map_stats: dict[str, Any]         # raw score map stats from tracker
    candidates: list[CandidateEvidence]     # top-k candidates
    template_context: TemplateContext
    recovery_context: RecoveryContext
    image_shape: tuple[int, int] | None = None  # (height, width) for bbox normalization


class EvidenceExtractor:
    """
    Converts raw tracker outputs into EvidenceFrame.
    Pure numeric extraction — no decisions, no TSA, no actions.
    """

    def __init__(self, history_len: int = 20, top_k_candidates: int = 5):
        self._history_len = history_len
        self._top_k = top_k_candidates
        self._feature_history: deque[np.ndarray] = deque(maxlen=history_len)
        self._bbox_history: deque[BBox] = deque(maxlen=history_len)
        self._template_ctx = TemplateContext()
        self._recovery_ctx = RecoveryContext()
        self._frame_idx = 0

    def reset(self) -> None:
        self._feature_history.clear()
        self._bbox_history.clear()
        self._template_ctx = TemplateContext()
        self._recovery_ctx = RecoveryContext()
        self._frame_idx = 0

    def notify_template_updated(self) -> None:
        self._template_ctx.last_update_age = 0
        self._template_ctx.update_count += 1

    def notify_reinit(self) -> None:
        self._recovery_ctx.last_reinit_age = 0
        self._recovery_ctx.total_reinit_count += 1

    def step(
        self,
        base_features: np.ndarray,
        bbox: BBox,
        score_map_stats: dict[str, Any] | None = None,
        candidates: list[dict[str, Any]] | None = None,
        image_shape: tuple[int, int] | None = None,
    ) -> EvidenceFrame:
        """
        Build one EvidenceFrame from raw tracker outputs.

        Args:
            base_features: raw feature vector (28,) — flow features will be zeroed
            bbox: current tracker bbox (x, y, w, h)
            score_map_stats: dict of raw response map stats (pass-through)
            candidates: list of dicts with keys matching CandidateEvidence fields
            image_shape: (height, width) of the frame, used for dist_to_border
                         computation (index 21).  If None, dist_to_border is 0.
        """
        validate_feature_matrix(base_features, expected_dim=28)
        prod_features = zero_production_features(base_features)

        # BUG-1 fix: capture prev_bbox BEFORE appending current bbox so that
        # _parse_candidates can measure candidate distance to the *previous*
        # tracker position rather than the current one.
        prev_bbox = self._bbox_history[-1] if len(self._bbox_history) >= 1 else bbox

        self._feature_history.append(prod_features)
        self._bbox_history.append(bbox)

        # BUG-2 fix: compute rolling features (indices 9-21) from history.
        enriched_features = self._compute_rolling_features(prod_features, bbox, image_shape)

        parsed_candidates = self._parse_candidates(candidates or [], bbox, prev_bbox=prev_bbox)

        ef = EvidenceFrame(
            frame_idx=self._frame_idx,
            bbox=bbox,
            base_features=enriched_features,
            score_map_stats=score_map_stats or {},
            candidates=parsed_candidates,
            template_context=TemplateContext(
                last_update_age=self._template_ctx.last_update_age,
                update_count=self._template_ctx.update_count,
            ),
            recovery_context=RecoveryContext(
                last_reinit_age=self._recovery_ctx.last_reinit_age,
                total_reinit_count=self._recovery_ctx.total_reinit_count,
            ),
            image_shape=image_shape,
        )

        self._frame_idx += 1
        self._template_ctx.last_update_age += 1
        if self._recovery_ctx.last_reinit_age >= 0:
            self._recovery_ctx.last_reinit_age += 1

        return ef

    def _parse_candidates(
        self, raw: list[dict[str, Any]], tracker_bbox: BBox, prev_bbox: BBox | None = None
    ) -> list[CandidateEvidence]:
        if not raw:
            return []
        cx = tracker_bbox[0] + tracker_bbox[2] / 2
        cy = tracker_bbox[1] + tracker_bbox[3] / 2
        tracker_area = max(tracker_bbox[2] * tracker_bbox[3], 1e-6)
        _prev = prev_bbox if prev_bbox is not None else tracker_bbox
        prev_cx = _prev[0] + _prev[2] / 2
        prev_cy = _prev[1] + _prev[3] / 2

        top_score = max((c.get("score", 0.0) for c in raw), default=1.0)
        if top_score <= 0:
            top_score = 1.0

        result = []
        for i, c in enumerate(raw[: self._top_k]):
            b: BBox = tuple(c["bbox"])  # type: ignore
            ccx = b[0] + b[2] / 2
            ccy = b[1] + b[3] / 2
            candidate_area = max(b[2] * b[3], 1e-6)
            result.append(
                CandidateEvidence(
                    bbox=b,
                    score=float(c.get("score", 0.0)),
                    rank=i,
                    score_ratio_to_top=float(c.get("score", 0.0)) / top_score,
                    distance_to_tracker=float(np.hypot(ccx - cx, ccy - cy)),
                    distance_to_prev_bbox=float(np.hypot(ccx - prev_cx, ccy - prev_cy)),
                    size_ratio_to_tracker=candidate_area / tracker_area,
                    source=str(c.get("source", "score_map")),
                    detector_score=c.get("detector_score"),
                    teacher_score=c.get("teacher_score"),
                )
            )
        return result

    def _compute_rolling_features(
        self,
        prod_features: np.ndarray,
        bbox: BBox,
        image_shape: tuple[int, int] | None,
    ) -> np.ndarray:
        """Return a 28-dim array with rolling features (indices 9-21) filled in.

        Uses the already-appended ``_feature_history`` and ``_bbox_history``
        (current frame is the last element of each deque).

        Indices 0-8 are copied unchanged from *prod_features*.
        Indices 22-27 are already zero (flow, set by zero_production_features).
        """
        out = prod_features.copy()

        # History as a list for indexing (most recent is last).
        hist = list(self._feature_history)   # length >= 1 (current already appended)
        bboxes = list(self._bbox_history)    # same length

        n = len(hist)

        # Current frame scalars (index into the CURRENT feature vector)
        apce_now = float(prod_features[0])    # idx 0: apce_raw
        ent_now = float(prod_features[3])     # idx 3: response_entropy
        pm_now = float(prod_features[4])      # idx 4: peak_margin

        # -- Indices 9-12: rolling APCE/entropy/peak_margin ratios --------------

        # apce_ratio_5: apce_now / mean(apce over last 5 PREVIOUS frames)
        apce_w5 = np.array([hist[i][0] for i in range(max(0, n - 6), n - 1)], dtype=np.float32)
        if len(apce_w5) > 0:
            out[9] = apce_now / (float(apce_w5.mean()) + 1e-8)
        else:
            out[9] = 1.0

        # apce_ratio_20: apce_now / mean(apce over last 20 PREVIOUS frames)
        apce_w20 = np.array([hist[i][0] for i in range(max(0, n - 21), n - 1)], dtype=np.float32)
        if len(apce_w20) > 0:
            out[10] = apce_now / (float(apce_w20.mean()) + 1e-8)
        else:
            out[10] = 1.0

        # entropy_delta_5: entropy_now - mean(entropy over last 5 PREVIOUS frames)
        ent_w5 = np.array([hist[i][3] for i in range(max(0, n - 6), n - 1)], dtype=np.float32)
        if len(ent_w5) > 0:
            out[11] = ent_now - float(ent_w5.mean())
        else:
            out[11] = 0.0

        # peak_margin_delta_5: pm_now - mean(pm over last 5 PREVIOUS frames)
        pm_w5 = np.array([hist[i][4] for i in range(max(0, n - 6), n - 1)], dtype=np.float32)
        if len(pm_w5) > 0:
            out[12] = pm_now - float(pm_w5.mean())
        else:
            out[12] = 0.0

        # -- Indices 13-14: streak counters (v2 parity) -----------------------

        # high_apce_streak: consecutive frames (including current) with apce > 100
        streak_high = 0
        for i in range(n - 1, -1, -1):
            if float(hist[i][0]) > 100.0:
                streak_high += 1
            else:
                break
        out[13] = float(streak_high)

        # low_apce_streak: consecutive frames (including current) with apce < 50
        streak_low = 0
        for i in range(n - 1, -1, -1):
            if float(hist[i][0]) < 50.0:
                streak_low += 1
            else:
                break
        out[14] = float(streak_low)

        # -- Indices 15-21: bbox dynamics -------------------------------------

        if n >= 2:
            cur_b = bboxes[-1]
            prv_b = bboxes[-2]
            diag = max((cur_b[2] ** 2 + cur_b[3] ** 2) ** 0.5, 1.0)
            cur_cx = cur_b[0] + cur_b[2] / 2
            cur_cy = cur_b[1] + cur_b[3] / 2
            prv_cx = prv_b[0] + prv_b[2] / 2
            prv_cy = prv_b[1] + prv_b[3] / 2
            vx = (cur_cx - prv_cx) / diag
            vy = (cur_cy - prv_cy) / diag
            speed = (vx ** 2 + vy ** 2) ** 0.5

            if n >= 3:
                pp_b = bboxes[-3]
                pp_cx = pp_b[0] + pp_b[2] / 2
                pp_cy = pp_b[1] + pp_b[3] / 2
                vx2 = (prv_cx - pp_cx) / diag
                vy2 = (prv_cy - pp_cy) / diag
                accel = abs(speed - (vx2 ** 2 + vy2 ** 2) ** 0.5)
            else:
                accel = 0.0

            scale_r = (cur_b[2] * cur_b[3]) / max(prv_b[2] * prv_b[3], 1.0)
            asp_d = (cur_b[2] / max(cur_b[3], 1e-3)) - (prv_b[2] / max(prv_b[3], 1e-3))

            if image_shape is not None:
                h_img, w_img = image_shape
                search_sz = max(cur_b[2], cur_b[3]) * 4.0
                dist_border = min(cur_cx, cur_cy, w_img - cur_cx, h_img - cur_cy) / max(search_sz, 1.0)
                dist_border = float(np.clip(dist_border, 0.0, 1.0))
            else:
                dist_border = 0.0

            out[15] = float(vx)
            out[16] = float(vy)
            out[17] = float(speed)
            out[18] = float(accel)
            out[19] = float(np.clip(scale_r, 0.0, 10.0))
            out[20] = float(asp_d)
            out[21] = dist_border
        else:
            # First frame: no previous bbox available
            out[15] = 0.0
            out[16] = 0.0
            out[17] = 0.0
            out[18] = 0.0
            out[19] = 1.0
            out[20] = 0.0
            out[21] = 0.0

        return out

