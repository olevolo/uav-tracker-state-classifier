"""SALTRunner — unified SALT inference pipeline.

Connects:
  SGLATracker        — primary tracker
  SALTRDController   — SALT-RD decision controller (optional)
  EvidenceExtractor  — raw feature → EvidenceFrame
  CosineAppearanceMemory — appearance drift signal
  OnlineLSTMMotionPredictor — LSTM motion residual signal
  YOLOv8Detector     — recovery detector (REINIT/SCORE_CANDIDATES action only)

Usage:
    runner = SALTRunner.from_config("configs/prod/salt.yaml")
    for entry in runner.run(sequence):
        print(entry.aux["saltrd_action_compute"], entry.bbox)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from uav_tracker.runner import TelemetryEntry
from uav_tracker.types import BBox, FrameContext, TrackState

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _iou_bbox(a: BBox, b: BBox) -> float:
    """Intersection-over-union for two BBox objects."""
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / (a.w * a.h + b.w * b.h - inter + 1e-6)


def _get_embed_helper():
    """Return the module-level CosineAppearanceMemory used for all embedding extractions.

    Single instance with a fixed random projection matrix — reused across all
    Guard-3 calls, template snapshot checks, and recovery cosine guards.
    Using fresh instances per call (as in the original code) regenerates the
    random projection each time, making cosine similarities incomparable across
    calls (BUG-02).
    """
    if _get_embed_helper._instance is None:
        from uav_tracker.ml.appearance_memory.cosine_memory import CosineAppearanceMemory
        _get_embed_helper._instance = CosineAppearanceMemory(max_templates=1)
    return _get_embed_helper._instance


_get_embed_helper._instance = None  # type: ignore[attr-defined]


def _size_ok(det_bbox: BBox, ref_bbox: BBox) -> bool:
    """Return True if det_bbox size is within ±70% of ref_bbox in both w and h.

    Loosened from ±50% to ±70% (0.3–3.0 range) to handle scale-changing
    targets (e.g. UAV approaching camera at 2× scale change).
    """
    if ref_bbox.w <= 0 or ref_bbox.h <= 0:
        return True
    w_ratio = det_bbox.w / ref_bbox.w
    h_ratio = det_bbox.h / ref_bbox.h
    return 0.3 <= w_ratio <= 3.0 and 0.3 <= h_ratio <= 3.0


def _cosine_sim(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Cosine similarity between two 1-D float32 arrays."""
    n1 = np.linalg.norm(emb1) + 1e-8
    n2 = np.linalg.norm(emb2) + 1e-8
    return float(np.dot(emb1 / n1, emb2 / n2))


# ---------------------------------------------------------------------------
# SALTRunner
# ---------------------------------------------------------------------------

@dataclass
class SALTRunner:
    """SALT: Self-supervised Adaptive Learning Tracker pipeline.

    All components except `tracker` are optional — omitting them degrades
    gracefully to pure SGLATrack inference.
    """

    tracker: Any                          # SGLATracker
    detector: Any | None = None           # YOLOv8Detector
    appearance_memory: Any | None = None  # CosineAppearanceMemory
    motion_predictor: Any | None = None   # OnlineLSTMMotionPredictor
    saltrd_controller: Any | None = None  # SALTRDController (optional)
    evidence_extractor: Any | None = None # EvidenceExtractor (optional)
    seed: int = 42

    # Center-freeze disabled by default — oracle showed 0.000 AUC gain and it
    # causes regression on standard sequences (car13 -0.396, truck1 -0.156).
    # Enable with runner._enable_center_freeze = True for experimental evaluation.
    _enable_center_freeze: bool = field(default=False, init=False, repr=False)

    # per-run state
    _trajectory: list = field(default_factory=list, init=False, repr=False)
    _prev_frame: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _frame_idx: int = field(default=0, init=False, repr=False)
    # Previous SALT-RD action — used to route tracker on the next frame.
    # Initialized to TrackerAction() (full compute, no recovery) at run() start.
    _prev_action: Any = field(default=None, init=False, repr=False)
    # Frames remaining before RT-DETR recovery can fire again (prevents loop)
    _lost_cooldown: int = field(default=0, init=False, repr=False)
    # Consecutive frames where recovery action was requested — fires after N
    _consecutive_recovery_frames: int = field(default=0, init=False, repr=False)
    _RECOVERY_MIN_FRAMES: int = field(default=5, init=False, repr=False)
    # Startup warmup: recovery blocked for first N frames.
    _RECOVERY_WARMUP_FRAMES: int = field(default=10, init=False, repr=False)
    # Last confirmed bbox — used to reject size-inconsistent recovery detections
    _last_good_bbox: Optional[BBox] = field(default=None, init=False, repr=False)
    # Temporal voting accumulator for recovery (prevents single-frame false re-inits)
    _recovery_votes: list = field(default_factory=list, init=False, repr=False)
    _RECOVERY_VOTE_FRAMES: int = field(default=2, init=False, repr=False)
    _RECOVERY_VOTE_THRESHOLD: int = field(default=2, init=False, repr=False)
    # Guard 3: reference appearance embedding from frame 0 for cosine similarity check
    # at recovery time. Set at init, updated slowly during confirmed tracking via EMA.
    _ref_embedding: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    # Guard 4: class id of the initial target (from detector, if available) — filters
    # recovery candidates to the same semantic class as the original target.
    _target_class: Optional[int] = field(default=None, init=False, repr=False)
    # Cosine similarity of the last accepted recovery candidate — set by
    # _best_detection() and read in _step() for the displacement+cosine check.
    _last_recovery_sim: float = field(default=0.0, init=False, repr=False)

    # ---------------------------------------------------------------------------
    # Factory
    # ---------------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path) -> "SALTRunner":
        """Build a SALTRunner from a YAML config (configs/prod/salt.yaml)."""
        import yaml
        cfg = yaml.safe_load(Path(config_path).read_text())

        from uav_tracker.registry import TRACKERS, DETECTORS
        from uav_tracker.ml.appearance_memory.cosine_memory import CosineAppearanceMemory
        from uav_tracker.ml.motion_predictor.lstm_predictor import OnlineLSTMMotionPredictor

        tracker_name = cfg.get("tracker", {}).get("name", "sglatrack")
        tracker = TRACKERS.build(tracker_name, enable_ce=cfg.get("enable_ce", True))

        det_cfg = cfg.get("detector", {})
        det_name = det_cfg.get("name")
        if det_name:
            det_kwargs = {k: v for k, v in det_cfg.items() if k != "name"}
            detector = DETECTORS.build(det_name, **det_kwargs)
        else:
            detector = None

        mem_cfg = cfg.get("appearance_memory", {})
        if mem_cfg.get("enabled", True):
            memory = CosineAppearanceMemory(
                max_templates=mem_cfg.get("max_templates", 50),
                forgetting_factor=mem_cfg.get("forgetting_factor", 0.95),
                store_interval=mem_cfg.get("store_interval", 10),
                min_confidence=mem_cfg.get("min_confidence", 0.60),
            )
        else:
            memory = None

        mp_cfg = cfg.get("motion_predictor", {})
        mp_enabled = mp_cfg.get("enabled", True) and cfg.get("enable_dynamic", True)
        if mp_enabled:
            motion_pred = OnlineLSTMMotionPredictor(
                hidden_size=mp_cfg.get("hidden_size", 32),
                seq_len=mp_cfg.get("seq_len", 10),
            )
        else:
            motion_pred = None

        # Optional SALT-RD controller — constructed only when config provides a path
        saltrd_controller = None
        evidence_extractor = None
        saltrd_cfg = cfg.get("saltrd", {})
        if saltrd_cfg.get("enabled", False):
            try:
                from salt_r.controller import SALTRDController
                from salt_r.evidence import EvidenceExtractor
                saltrd_controller = SALTRDController()
                evidence_extractor = EvidenceExtractor()
            except ImportError:
                _logger.warning("SALT-RD controller not available — running without it")

        runner = cls(
            tracker=tracker,
            detector=detector,
            appearance_memory=memory,
            motion_predictor=motion_pred,
            saltrd_controller=saltrd_controller,
            evidence_extractor=evidence_extractor,
            seed=cfg.get("seed", 42),
        )

        # Eagerly warm up components so the first sequence bears no cold-start cost.
        # prepare() loads tracker weights; detector warmup() pays JIT/CUDA-graph init.
        runner.prepare()
        if runner.detector is not None and hasattr(runner.detector, "warmup"):
            try:
                runner.detector.warmup()
            except Exception:
                pass  # warmup failure is non-fatal (e.g. weights not present)

        return runner

    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------

    def run(self, sequence: Any) -> Iterator[TelemetryEntry]:
        """Run SALT on a sequence. Yields one TelemetryEntry per frame."""
        self._reset()

        frames = list(sequence.frames)
        gt = sequence.ground_truth

        # Frame 0: initialise tracker
        self.tracker.init(frames[0], gt[0])
        self._trajectory.append(gt[0])
        self._prev_frame = frames[0]
        self._frame_idx = 0

        # Log warning for degenerate GT bbox at frame 0 (informational only).
        if gt[0].w <= 0 or gt[0].h <= 0:
            _logger.warning(
                "SALTRunner: degenerate GT bbox at frame 0 "
                "(w=%.1f h=%.1f) — tracker init may be unreliable",
                gt[0].w, gt[0].h,
            )

        # Guard 3: store reference appearance embedding from the ground-truth crop
        # on frame 0. Used at recovery time to reject appearance-mismatched detections.
        try:
            self._ref_embedding = _get_embed_helper()._extract_embedding(frames[0], gt[0])
        except Exception:
            self._ref_embedding = None

        yield TelemetryEntry(
            frame_idx=0,
            bbox=gt[0],
            confidence=1.0,
            tier=1,
            switched=False,
            aux={
                # SALT-RD telemetry for init frame
                "saltrd_action_compute": "full",
                "saltrd_action_search": "keep",
                "saltrd_action_template": "keep_current",
                "saltrd_action_recovery": "none",
                "saltrd_action_confidence": 1.0,
                "saltrd_changed_bbox": False,
                "saltrd_safety_fallback": False,
                "saltrd_reason": "init_frame",
                "score_map_stats": {},
                "apce_raw": 0.0,
                "psr_raw": 0.0,
                "entropy_raw": 0.0,
            },
        )

        for idx, frame in enumerate(frames[1:], start=1):
            self._frame_idx = idx
            t0 = time.perf_counter()

            entry = self._step(frame)
            entry = TelemetryEntry(
                frame_idx=idx,
                bbox=entry.bbox,
                confidence=entry.confidence,
                tier=entry.tier,
                switched=entry.switched,
                timings_ms={"total": (time.perf_counter() - t0) * 1000},
                aux=entry.aux,
            )
            self._prev_frame = frame
            yield entry

    def _step(self, frame: np.ndarray) -> TelemetryEntry:
        """Process one frame. Returns a TelemetryEntry (frame_idx not set).

        SALT-RD controller loop
        -----------------------
        1. Run tracker using previous frame's action for compute routing.
        2. Feed tracker telemetry into EvidenceExtractor -> EvidenceFrame.
        3. Run SALTRDController.step(evidence) -> SALTRDDecision.
        4. Cache action for next frame routing.
        5. Execute recovery if decision.action.recovery in (REINIT, SCORE_CANDIDATES)
           and detector is available.
        6. Execute template update if decision.action.template == UPDATE.

        If saltrd_controller is None: use TrackerAction() (full compute, no recovery)
        as the default, and call tracker.update_with_action(frame, TrackerAction())
        which falls back to tracker.update(frame).
        """
        from salt_r.actions import TrackerAction, TemplateAction, RecoveryAction

        # ---- Motion predictor: LSTM hint ----
        lstm_pred: BBox | None = None
        if self.motion_predictor and len(self._trajectory) >= 2:
            try:
                lstm_pred = self.motion_predictor.predict_next(
                    history=list(self._trajectory),
                    timestamps=list(range(len(self._trajectory))),
                )
            except Exception:
                lstm_pred = None

        # ---- Appearance drift ----
        drift = 0.0
        if self.appearance_memory:
            try:
                drift = float(self.appearance_memory.compute_drift())
            except Exception:
                drift = 0.0

        # ---- Step 1: run tracker using previous-frame action ----
        prev_action = self._prev_action if self._prev_action is not None else TrackerAction()

        t_tracker = time.perf_counter()
        track_state: TrackState = self.tracker.update_with_action(frame, prev_action)
        tracker_ms = (time.perf_counter() - t_tracker) * 1000

        # Extract score-map quality metrics (populated by SGLATracker; 0.0 for others)
        apce = getattr(track_state, 'apce', 0.0)
        psr = getattr(track_state, 'psr', 0.0)
        response_entropy = getattr(track_state, 'response_entropy', 0.0)

        # SALT-RD advisory p_fc (0.0 if no advisor attached) -- kept for risk logging
        _advisor = getattr(self.tracker, '_salt_rd_advisor', None)
        _advisory_p_fc: float = _advisor.last_p_fc if _advisor is not None else 0.0

        # ---- Step 2+3: SALT-RD controller ----
        # Build EvidenceFrame and run controller if available; else use safe NOOP action.
        decision = None
        if self.saltrd_controller is not None and self.evidence_extractor is not None:
            # Build 28-dim feature vector from tracker telemetry
            current_bbox = track_state.bbox
            bbox_tuple = (current_bbox.x, current_bbox.y, current_bbox.w, current_bbox.h)
            score_map_stats = getattr(track_state, 'score_map_stats', {}) or {}

            # Populate available base features (flow features 22-27 stay zero)
            import numpy as _np
            features = _np.zeros(28, dtype=_np.float32)
            features[0] = apce
            features[1] = apce / (apce + 1.0)   # apce_norm: bounded [0,1)
            features[2] = psr
            features[3] = response_entropy
            features[4] = float(score_map_stats.get('peak_margin', 0.0))
            features[5] = float(score_map_stats.get('peak_width', 0.0))
            features[6] = float(score_map_stats.get('n_secondary', 0.0))
            features[7] = float(score_map_stats.get('peak_distance', 0.0))
            features[8] = float(score_map_stats.get('heatmap_mass_topk', 0.0))

            candidates_raw = score_map_stats.get('candidates', []) or []
            try:
                evidence_frame = self.evidence_extractor.step(
                    base_features=features,
                    bbox=bbox_tuple,
                    score_map_stats=score_map_stats,
                    candidates=candidates_raw,
                )
                decision = self.saltrd_controller.step(evidence_frame)
            except Exception as exc:
                _logger.debug("SALT-RD controller step failed: %s", exc)
                decision = None

        # Use safe NOOP if controller unavailable or failed
        if decision is None:
            from salt_r.controller import SALTRDDecision
            decision = SALTRDDecision(
                action=TrackerAction(),
                safety_fallback_applied=True,
                reason="no_controller",
            )

        # ---- Step 4: cache action for next frame ----
        self._prev_action = decision.action

        # ---- Per-frame telemetry flags ----
        _template_attempted: bool = False
        _template_updated: bool = False
        _reinit_vetoed: bool = False
        _saltrd_changed_bbox: bool = False

        # ---- Guard 1: remember last good bbox for size-consistency filtering ----
        if track_state.confidence >= 0.14:
            self._last_good_bbox = track_state.bbox

        # ---- Guard 3: EMA update of reference embedding on stable frames every 50 ----
        if (track_state.confidence >= 0.014
                and self._frame_idx % 50 == 0
                and self._ref_embedding is not None):
            try:
                new_emb = _get_embed_helper()._extract_embedding(frame, track_state.bbox)
                self._ref_embedding = 0.80 * self._ref_embedding + 0.20 * new_emb
                norm = np.linalg.norm(self._ref_embedding) + 1e-8
                self._ref_embedding = self._ref_embedding / norm
            except Exception:
                pass

        if self._frame_idx % 50 == 0:
            _logger.warning(
                "Frame %d: apce=%.1f psr=%.1f entropy=%.3f conf=%.3f tracker=%.1fms",
                self._frame_idx, apce, psr, response_entropy,
                track_state.confidence, tracker_ms,
            )

        # ---- Track consecutive recovery-requested frames ----
        _recovery_action = decision.action.recovery
        if _recovery_action in (RecoveryAction.REINIT, RecoveryAction.SCORE_CANDIDATES):
            self._consecutive_recovery_frames += 1
        else:
            self._consecutive_recovery_frames = 0
        if self._lost_cooldown > 0:
            self._lost_cooldown -= 1

        # prev_bbox for recovery hint
        prev_bbox: BBox = self._trajectory[-1] if self._trajectory else track_state.bbox

        # ---- Step 5: Recovery -- only when controller requests it ----
        _genuine_recovery_request = (
            self._consecutive_recovery_frames >= self._RECOVERY_MIN_FRAMES
        )
        _past_warmup = (self._frame_idx >= self._RECOVERY_WARMUP_FRAMES)
        if (_genuine_recovery_request
                and _past_warmup
                and self.detector is not None
                and self._lost_cooldown == 0):
            try:
                _frozen_hint = getattr(self.tracker, '_state', None) or prev_bbox
                _logger.warning(
                    "SALT recovery: frame=%d hint=(%s) consecutive_recovery=%d",
                    self._frame_idx,
                    f"{_frozen_hint.x:.0f},{_frozen_hint.y:.0f} {_frozen_hint.w:.0f}x{_frozen_hint.h:.0f}"
                    if _frozen_hint else "None",
                    self._consecutive_recovery_frames,
                )

                # Constrained recovery: use advisor crop if available
                _use_constrained_recovery: bool = False
                _recovery_crop: "tuple[float, float, float, float] | None" = None

                # Use controller's spatial hint if available (from learned reinit candidate)
                _recovery_hint = decision.action.detector_hint or decision.action.bbox_hint
                if _recovery_hint is not None:
                    _recovery_crop = _recovery_hint
                    _use_constrained_recovery = True

                if _use_constrained_recovery and _recovery_crop is not None:
                    _rx, _ry, _rw, _rh = _recovery_crop
                    _rx_i, _ry_i = int(_rx), int(_ry)
                    _rw_i, _rh_i = max(1, int(_rw)), max(1, int(_rh))
                    cropped_frame = frame[_ry_i:_ry_i + _rh_i, _rx_i:_rx_i + _rw_i]
                    if cropped_frame.size > 0:
                        crop_detections = self.detector.detect(cropped_frame, hint_bbox=None)

                        class _OffsetDet:
                            __slots__ = ("bbox", "confidence", "class_id")
                            def __init__(self, d: Any, dx: int, dy: int) -> None:
                                self.bbox = BBox(x=d.bbox.x + dx, y=d.bbox.y + dy,
                                                 w=d.bbox.w, h=d.bbox.h)
                                self.confidence = float(
                                    getattr(d, "confidence", None) or getattr(d, "score", 0.0)
                                )
                                self.class_id = getattr(d, "class_id", None)

                        full_frame_detections = [
                            _OffsetDet(_det, _rx_i, _ry_i) for _det in crop_detections
                        ]
                        if full_frame_detections:
                            detections = full_frame_detections
                        else:
                            detections = self.detector.detect(frame, hint_bbox=_frozen_hint)
                            _use_constrained_recovery = False
                    else:
                        detections = self.detector.detect(frame, hint_bbox=_frozen_hint)
                        _use_constrained_recovery = False
                else:
                    detections = self.detector.detect(frame, hint_bbox=_frozen_hint)

                if detections:
                    if self._target_class is not None:
                        class_filtered = [
                            d for d in detections
                            if getattr(d, 'class_id', self._target_class) == self._target_class
                        ]
                        if class_filtered:
                            detections = class_filtered

                    n_lost = self._consecutive_recovery_frames
                    best = self._best_detection(detections, frame, self._last_good_bbox, n_lost=n_lost)
                    _logger.warning(
                        "SALT recovery: %d detections after class-filter, best=%s accepted=%s",
                        len(detections),
                        f"({best.bbox.x:.0f},{best.bbox.y:.0f} {best.bbox.w:.0f}x{best.bbox.h:.0f})"
                        if best else "None",
                        best is not None,
                    )
                    if best is not None:
                        self._recovery_votes.append((best.bbox, self._last_recovery_sim))
                        if self._target_class is None:
                            self._target_class = getattr(best, 'class_id', None)

                    if len(self._recovery_votes) >= self._RECOVERY_VOTE_FRAMES:
                        winner, winner_sim = self._pick_voted_detection(self._recovery_votes)
                        self._recovery_votes = []
                        if winner is not None:
                            _w_cx = winner.x + winner.w / 2
                            _w_cy = winner.y + winner.h / 2

                            _traj_confirmed = [b for b in self._trajectory[-10:] if b is not None]
                            if len(_traj_confirmed) >= 5:
                                _pts = _traj_confirmed[-5:]
                                _disps_x = [
                                    (_pts[i + 1].x + _pts[i + 1].w / 2) - (_pts[i].x + _pts[i].w / 2)
                                    for i in range(4)
                                ]
                                _disps_y = [
                                    (_pts[i + 1].y + _pts[i + 1].h / 2) - (_pts[i].y + _pts[i].h / 2)
                                    for i in range(4)
                                ]
                                _vx = float(np.mean(_disps_x))
                                _vy = float(np.mean(_disps_y))
                                _n_lost = max(1, self._consecutive_recovery_frames)
                                _last_pt = _traj_confirmed[-1]
                                _pred_cx = _last_pt.x + _last_pt.w / 2 + _vx * _n_lost
                                _pred_cy = _last_pt.y + _last_pt.h / 2 + _vy * _n_lost
                            else:
                                _ref = self._last_good_bbox if self._last_good_bbox else (
                                    self._trajectory[-1] if self._trajectory else winner
                                )
                                _pred_cx = _ref.x + _ref.w / 2
                                _pred_cy = _ref.y + _ref.h / 2

                            _dist_to_pred = ((_w_cx - _pred_cx) ** 2 + (_w_cy - _pred_cy) ** 2) ** 0.5
                            _lgb = self._last_good_bbox if self._last_good_bbox else (
                                self._trajectory[-1] if self._trajectory else winner
                            )
                            _lgb_cx = _lgb.x + _lgb.w / 2
                            _lgb_cy = _lgb.y + _lgb.h / 2
                            _dist_to_lgb = ((_w_cx - _lgb_cx) ** 2 + (_w_cy - _lgb_cy) ** 2) ** 0.5
                            _closer_to_pred = _dist_to_pred < _dist_to_lgb * 0.7
                            _high_sim = winner_sim >= 0.70

                            _logger.warning(
                                "SALT Guard5: winner=(%s) pred=(%.0f,%.0f) "
                                "dist_pred=%.1f dist_lgb=%.1f closer=%s cosine_sim=%.3f high_sim=%s",
                                f"{winner.x:.0f},{winner.y:.0f} {winner.w:.0f}x{winner.h:.0f}",
                                _pred_cx, _pred_cy, _dist_to_pred, _dist_to_lgb,
                                _closer_to_pred, winner_sim, _high_sim,
                            )

                            if not _closer_to_pred and not _high_sim:
                                _logger.warning(
                                    "SALT Guard5: REJECTED recovery "
                                    "(dist_pred=%.1f >= 0.7xdist_lgb=%.1f and cosine_sim=%.3f < 0.70)",
                                    _dist_to_pred, _dist_to_lgb * 0.7, winner_sim,
                                )
                                self._lost_cooldown = 40
                                _reinit_vetoed = True
                            else:
                                self.tracker.init(frame, winner)
                                track_state = TrackState(
                                    bbox=winner,
                                    confidence=1.0,
                                    status="locked",
                                )
                                if self.motion_predictor:
                                    self.motion_predictor.reset()
                                self._lost_cooldown = 0
                                self._consecutive_recovery_frames = 0
                                _saltrd_changed_bbox = True
                                if self.evidence_extractor is not None:
                                    self.evidence_extractor.notify_reinit()
                                self._trajectory.append(track_state.bbox)
                                if len(self._trajectory) > 200:
                                    self._trajectory = self._trajectory[-200:]
                                return TelemetryEntry(
                                    frame_idx=self._frame_idx,
                                    bbox=track_state.bbox,
                                    confidence=track_state.confidence,
                                    tier=3,
                                    switched=True,
                                    aux={
                                        "saltrd_action_compute": decision.action.compute.value,
                                        "saltrd_action_search": decision.action.search.value,
                                        "saltrd_action_template": decision.action.template.value,
                                        "saltrd_action_recovery": decision.action.recovery.value,
                                        "saltrd_action_confidence": decision.model_confidence,
                                        "saltrd_changed_bbox": True,
                                        "saltrd_safety_fallback": decision.safety_fallback_applied,
                                        "saltrd_reason": decision.reason,
                                        "recovered": True,
                                        "score_map_stats": {},
                                        "apce_raw": 0.0,
                                        "psr_raw": 0.0,
                                        "entropy_raw": 0.0,
                                        "lstm_pred": (
                                            [lstm_pred.x, lstm_pred.y, lstm_pred.w, lstm_pred.h]
                                            if lstm_pred else None
                                        ),
                                        "appearance_drift": round(drift, 4),
                                    },
                                )
                        else:
                            self._lost_cooldown = 25
                    elif best is None:
                        self._lost_cooldown = 25
                else:
                    self._lost_cooldown = 10
            except Exception as exc:
                _logger.debug("SALT recovery failed: %s", exc)
                self._lost_cooldown = 10

        self._trajectory.append(track_state.bbox)
        if len(self._trajectory) > 200:
            self._trajectory = self._trajectory[-200:]

        # ---- Appearance memory: store on confident frames ----
        if self.appearance_memory and track_state.confidence >= 0.6:
            try:
                ctx = FrameContext(frame=frame, frame_idx=self._frame_idx)
                self.appearance_memory.store(ctx, track_state)
            except Exception:
                pass

        # ---- Motion predictor: online update on confident frames ----
        if self.motion_predictor and track_state.confidence >= 0.14:
            try:
                self.motion_predictor.update(track_state.bbox)
            except Exception:
                pass

        # ---- Step 6: Template update -- gated by controller action ----
        if decision.action.template == TemplateAction.UPDATE:
            _sw = getattr(self.tracker, '_last_search_score_weighted', None)
            _te = getattr(self.tracker, '_last_template_embedding', None)
            if _sw is not None and _te is not None:
                _cosine = float(_cosine_sim(_sw.cpu().numpy(), _te.cpu().numpy()))
            else:
                _cosine = 1.0
            _template_attempted = True
            _template_updated = self.tracker.try_update_template(
                frame, track_state.bbox, apce, psr, self._frame_idx, _cosine,
                apce_threshold=150.0, psr_threshold=500.0,
                min_interval=100, cosine_threshold=0.80, max_updates=3,
            )
            if _template_updated and self.evidence_extractor is not None:
                self.evidence_extractor.notify_template_updated()

        # ---- Build telemetry aux dict ----
        _aux: dict[str, Any] = {
            # SALT-RD controller fields
            "saltrd_action_compute": decision.action.compute.value,
            "saltrd_action_search": decision.action.search.value,
            "saltrd_action_template": decision.action.template.value,
            "saltrd_action_recovery": decision.action.recovery.value,
            "saltrd_action_confidence": decision.model_confidence,
            "saltrd_changed_bbox": _saltrd_changed_bbox,
            "saltrd_safety_fallback": decision.safety_fallback_applied,
            "saltrd_reason": decision.reason,
            # Raw tracker metrics
            "apce_raw": apce,
            "psr_raw": psr,
            "entropy_raw": response_entropy,
            "score_map_stats": getattr(track_state, "score_map_stats", {}),
            # Auxiliary signals
            "lstm_pred": (
                [lstm_pred.x, lstm_pred.y, lstm_pred.w, lstm_pred.h]
                if lstm_pred else None
            ),
            "appearance_drift": round(drift, 4),
            # Template update
            "template_update_attempted": _template_attempted,
            "template_updated": _template_updated,
            "reinit_vetoed": _reinit_vetoed,
        }
        # Advisory p_fc for risk logging (read-only, not used for control)
        if _advisor is not None:
            _aux["salt_rd_p_fc"] = _advisory_p_fc

        return TelemetryEntry(
            frame_idx=self._frame_idx,
            bbox=track_state.bbox,
            confidence=track_state.confidence,
            tier=1,
            switched=False,
            aux=_aux,
        )

    def _best_detection(self, detections: list, frame: np.ndarray,
                        ref_bbox: Optional[BBox] = None,
                        n_lost: int = 0) -> "tuple[Any, float] | tuple[None, float]":
        """Return best detection using spatial proximity + confidence.

        Picks the detection closest to the last known trajectory position,
        weighted by detector confidence. Falls back to pure confidence if no
        trajectory. Returns None if best candidate looks like a distractor
        (too far from last known position after tracking failure).

        Spatial gate (Bug 3 fix): scales with n_lost. After N bad frames
        the target may have moved far, so we widen the search radius.
        Base gate is max(3 * diag, 50px) * (1 + min(n_lost, 20) * 0.1).
        Example: after 10 bad frames → 3×diag × 2.0 = 6×diag.

        Guard 1: if ref_bbox is given, reject detections whose w or h differ
        by more than 70% from ref_bbox (size-consistency filter, loosened for
        scale-changing targets — 0.3–3.0 range; was 0.5–2.0).

        Guard 3 (Bug 1 fix): use threshold-based cosine reject instead of
        multiplicative zero-forcing. sim < _SIM_THRESHOLD rejects cleanly wrong
        appearance; for valid detections (sim ≥ threshold) score is weighted by
        (0.5 + 0.5*sim). This prevents valid detections being zeroed when
        appearance has drifted (negative cosine after long occlusion is normal
        for the correct target). Also falls through to proximity-only scoring
        when _ref_embedding is None.
        """
        _SIM_THRESHOLD = 0.25  # reject only clearly-wrong appearance

        if not detections:
            return None

        # Guard 1: size-consistency filter
        if ref_bbox is not None:
            detections = [d for d in detections if _size_ok(d.bbox, ref_bbox)]
            if not detections:
                _logger.warning(
                    "SALT: all detections rejected by size-consistency filter "
                    "(ref w=%.0f h=%.0f)", ref_bbox.w, ref_bbox.h)
                return None

        if not self._trajectory:
            return max(detections, key=lambda d: d.confidence)

        # Last known position before LOST state
        last = self._trajectory[-1]
        lcx = last.x + last.w / 2
        lcy = last.y + last.h / 2
        diag = (last.w ** 2 + last.h ** 2) ** 0.5 + 1e-6

        # Bug 3 fix: widen gate proportionally to how long we've been lost
        _lost_scale = 1.0 + min(n_lost, 20) * 0.1  # up to ×3 after 20 frames
        max_dist = max(3.0 * diag, 50.0) * _lost_scale

        def _score(det) -> float:
            dcx = det.bbox.x + det.bbox.w / 2
            dcy = det.bbox.y + det.bbox.h / 2
            dist = ((dcx - lcx) ** 2 + (dcy - lcy) ** 2) ** 0.5
            # Normalise distance by last bbox diagonal
            dist_penalty = max(0.0, 1.0 - dist / (diag * 3))
            return det.confidence * (0.4 + 0.6 * dist_penalty)

        # Pre-filter to candidates within spatial gate before scoring
        candidates = [
            d for d in detections
            if ((d.bbox.x + d.bbox.w / 2 - lcx) ** 2
                + (d.bbox.y + d.bbox.h / 2 - lcy) ** 2) ** 0.5 <= max_dist
        ]

        if not candidates:
            _logger.warning(
                "SALT: all %d detections rejected by spatial gate "
                "(max_dist=%.1f n_lost=%d diag=%.1f)",
                len(detections), max_dist, n_lost, diag)
            return None

        # Tighter gate from _last_good_bbox (ref_bbox) when not lost for long.
        # The trajectory tail may have drifted (tracker was near the target but
        # not confirmed), so ref_bbox is a more reliable anchor for where the
        # target was before escalation. Use a slower-widening gate (x0.05/frame
        # vs x0.10/frame above) to reject distractors in crowded scenes.
        if ref_bbox is not None and n_lost < 30:
            gb_cx = ref_bbox.x + ref_bbox.w / 2
            gb_cy = ref_bbox.y + ref_bbox.h / 2
            gb_diag = (ref_bbox.w ** 2 + ref_bbox.h ** 2) ** 0.5 + 1e-6
            gb_scale = 1.0 + n_lost * 0.05  # tighter: x0.05 per lost frame
            gb_max_dist = max(3.0 * gb_diag, 50.0) * gb_scale
            filtered_candidates = [
                d for d in candidates
                if ((d.bbox.x + d.bbox.w / 2 - gb_cx) ** 2
                    + (d.bbox.y + d.bbox.h / 2 - gb_cy) ** 2) ** 0.5 <= gb_max_dist
            ]
            if filtered_candidates:
                candidates = filtered_candidates
            else:
                _logger.warning(
                    "SALT last_good_bbox gate: all %d candidates rejected "
                    "(gb_max_dist=%.1f n_lost=%d); keeping original candidates",
                    len(candidates), gb_max_dist, n_lost)
                # Don't discard -- fall back to trajectory-gate candidates

        # Raise cosine threshold in crowded scenes (>=2 candidates, n_lost < 30).
        # Multiple nearby candidates means there are distractors (e.g. two cyclists);
        # require stronger appearance match to avoid locking onto the wrong one.
        if n_lost < 30 and len(candidates) >= 2:
            _SIM_THRESHOLD = 0.50

        # Guard 3 (Bug 1 fix): threshold-based cosine filtering.
        # Old code: combined = _score(det) * max(0.0, sim) → zeros valid detections
        #   when sim < 0 (appearance drift after long occlusion).
        # New code: reject only if sim < _SIM_THRESHOLD (clearly wrong object);
        #   otherwise weight by (0.5 + 0.5*sim) so positive sim is rewarded but
        #   negative sim in [-1, threshold) is not penalised to zero.
        # Fall through to proximity-only if _ref_embedding is None or extraction fails.
        if self._ref_embedding is not None:
            try:
                _helper = _get_embed_helper()

                scored_with_sim = []
                for det in candidates:
                    emb = _helper._extract_embedding(frame, det.bbox)
                    sim = _cosine_sim(self._ref_embedding, emb)
                    if sim >= _SIM_THRESHOLD:
                        combined = _score(det) * (0.5 + 0.5 * sim)
                    else:
                        combined = 0.0  # reject clearly-wrong appearance only

                    scored_with_sim.append((det, combined, sim))

                scored_with_sim.sort(key=lambda x: x[1], reverse=True)
                best_det, best_combined, best_sim = scored_with_sim[0]

                _logger.warning(
                    "SALT cosine guard: best_sim=%.3f combined=%.3f accepted=%s "
                    "threshold=%.2f n_candidates=%d",
                    best_sim, best_combined, best_combined > 0.0,
                    _SIM_THRESHOLD, len(candidates),
                )

                if best_combined > 0.0:
                    self._last_recovery_sim = best_sim
                    return best_det
                else:
                    _logger.warning(
                        "SALT cosine guard: all %d candidates rejected "
                        "(best_sim=%.3f < threshold=%.2f)",
                        len(candidates), best_sim, _SIM_THRESHOLD,
                    )
                    return None  # all detections rejected by appearance guard
            except Exception:
                pass  # fall back to proximity-only on any error

        best = max(candidates, key=_score)
        self._last_recovery_sim = 0.0  # no cosine info available (proximity-only)
        return best

    def _pick_voted_detection(
        self, votes: list
    ) -> "tuple[Optional[BBox], float]":
        """Return (bbox, avg_cosine_sim) if 2+ votes cluster spatially, else (None, -1.0).

        votes is a list of (BBox, cosine_sim) tuples.
        Uses center-distance instead of IoU because small UAV targets (10-20px)
        have IoU < 0.3 between consecutive detections even when correct.
        """
        if len(votes) < self._RECOVERY_VOTE_THRESHOLD:
            return None, -1.0
        candidate_bbox, _ = votes[-1]
        diag = ((candidate_bbox.w ** 2 + candidate_bbox.h ** 2) ** 0.5) + 1e-6
        cx, cy = candidate_bbox.x + candidate_bbox.w / 2, candidate_bbox.y + candidate_bbox.h / 2

        def _close(v_bbox: BBox) -> bool:
            vx, vy = v_bbox.x + v_bbox.w / 2, v_bbox.y + v_bbox.h / 2
            return ((vx - cx) ** 2 + (vy - cy) ** 2) ** 0.5 < 3.0 * diag

        supporting = [(v_bbox, v_sim) for v_bbox, v_sim in votes if _close(v_bbox)]
        if len(supporting) >= self._RECOVERY_VOTE_THRESHOLD:
            valid_sims = [s for _, s in supporting if s >= 0.0]
            avg_sim = float(np.mean(valid_sims)) if valid_sims else -1.0
            return BBox(
                x=sum(v.x for v, _ in supporting) / len(supporting),
                y=sum(v.y for v, _ in supporting) / len(supporting),
                w=sum(v.w for v, _ in supporting) / len(supporting),
                h=sum(v.h for v, _ in supporting) / len(supporting),
            ), avg_sim
        return None, -1.0

    def _reset(self) -> None:
        t0 = time.perf_counter()
        self._trajectory = []
        self._prev_frame = None
        self._frame_idx = 0
        self._prev_action = None
        self._lost_cooldown = 0
        self._consecutive_recovery_frames = 0
        self._last_good_bbox = None
        self._recovery_votes = []
        self._ref_embedding = None
        self._target_class = None
        self._last_recovery_sim = 0.0
        if self.appearance_memory:
            self.appearance_memory.reset()
        if self.motion_predictor:
            self.motion_predictor.reset()
        if self.saltrd_controller is not None:
            self.saltrd_controller.reset()
        if self.evidence_extractor is not None:
            self.evidence_extractor.reset()
        # Reset tracker template counters and advisor rolling buffers so they
        # do not leak between sequences.  tracker.reset() calls advisor.reset()
        # internally, so this is the single authoritative reset point.
        if hasattr(self.tracker, 'reset'):
            self.tracker.reset()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > 50:
            _logger.warning(
                "_reset() took %.1fms (> 50ms threshold) -- slow component in reset",
                elapsed_ms,
            )

    def prepare(self) -> None:
        """Eagerly load the tracker model weights so the first sequence pays no cold-start.

        Safe to call multiple times — the tracker's _load() is guarded by
        ``if self._model is None``.  Called by from_config() so users who
        instantiate via the factory get a warm tracker for free.
        """
        if hasattr(self.tracker, "_load") and getattr(self.tracker, "_model", None) is None:
            self.tracker._load()
