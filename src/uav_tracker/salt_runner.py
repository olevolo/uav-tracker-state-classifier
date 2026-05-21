"""SALTRunner — unified SALT inference pipeline.

Connects:
  SGLATracker        — primary tracker with state-conditioned compute budget
  TargetStateAssessor — self-supervised semantic state estimation (Track A)
  HeadAdaptor        — drift-protected TTT (Track C, optional)
  CosineAppearanceMemory — appearance drift signal
  OnlineLSTMMotionPredictor — LSTM motion residual signal
  YOLOv8Detector     — recovery detector (LOST state only)

Usage:
    runner = SALTRunner.from_config("configs/prod/salt.yaml")
    for entry in runner.run(sequence):
        print(entry.aux["target_state"], entry.bbox)
"""
from __future__ import annotations

import logging
import time
from collections import deque
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
    tsa: Any | None = None                # TargetStateAssessor
    detector: Any | None = None           # YOLOv8Detector
    appearance_memory: Any | None = None  # CosineAppearanceMemory
    motion_predictor: Any | None = None   # OnlineLSTMMotionPredictor
    seed: int = 42

    # per-run state
    _trajectory: list = field(default_factory=list, init=False, repr=False)
    _prev_frame: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _frame_idx: int = field(default=0, init=False, repr=False)
    # TSA state from the previous frame — used to route tracker compute budget
    # on the current frame before TSA has run on the current frame's output.
    # Initialized to CONFIRMED (0) for normal sequences. At run() start, frame-0
    # GT bbox validity is checked: degenerate bbox (w<=0 or h<=0) sets this to
    # OCCLUDED so the first real frame uses full compute instead of CE-pruned.
    _prev_tsa_state_int: int = field(default=0, init=False, repr=False)
    # Frames remaining before RT-DETR recovery can fire again (prevents loop)
    _lost_cooldown: int = field(default=0, init=False, repr=False)
    # Consecutive LOST frames counter — RT-DETR only fires after N consecutive LOST
    _consecutive_lost: int = field(default=0, init=False, repr=False)
    # Threshold of 5 consecutive LOST frames before recovery fires.  Both genuine
    # loss (car7 frame 356: IoU=0.951 after recovery) and false-alarm loss
    # (bike2 frame 107: detector finds wrong cyclist at IoU=0.000) generate
    # exactly 5-consecutive-LOST streaks — no threshold distinguishes them.
    # The bike2 regression (−0.023) is a detector quality issue: YOLO26m finds a
    # different cyclist with cosine_sim=0.921 but wrong position. Fix requires
    # a better appearance model or post-recovery IoU validation, not a higher count.
    _RECOVERY_MIN_LOST: int = field(default=5, init=False, repr=False)
    # Startup warmup: TSA LOST → recovery blocked for first N frames.
    # Rationale: car13 exhibited 5 false-LOST frames (frames 1–5) due to
    # optical-flow IoU startup transients before the tracker settled. With APCE
    # as the primary signal (not flow-IoU), Farneback is valid from frame 1,
    # but the 10-frame warmup is kept as a conservative safety margin to prevent
    # spurious recovery on sequences where the tracker needs a few frames to
    # lock onto a small or fast-moving target.
    _RECOVERY_WARMUP_FRAMES: int = field(default=10, init=False, repr=False)
    # OCCLUDED → LOST escalation: if the tracker stays in OCCLUDED for more than
    # N consecutive frames, treat it as LOST so the recovery pipeline fires.
    # uav2 is 47% OCCLUDED (APCE 20-80) and the detector is never called without
    # this because the LOST threshold (apce < 20) is too tight to fire.
    _consecutive_occluded: int = field(default=0, init=False, repr=False)
    _OCCLUDED_ESCALATION_FRAMES: int = field(default=25, init=False, repr=False)
    # APCE from the previous frame — used as proxy for TSA temporal gating
    _prev_apce: float = field(default=0.0, init=False, repr=False)
    # Rolling buffer of last 25 APCE values — used by staged OCCLUDED escalation
    # to require sustained low APCE (not just frame count) before firing recovery.
    _apce_buffer: deque = field(default_factory=lambda: deque(maxlen=25), init=False, repr=False)
    # APCE of the most recent frame that was escalated from OCCLUDED→LOST.
    # Used for APCE-trend gating: if the current frame's APCE has risen ≥15%
    # above this value, the tracker is self-recovering and escalation is skipped.
    _prev_escalated_apce: float = field(default=0.0, init=False, repr=False)
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
        from uav_tracker.ml.tsa.target_state_assessor import TargetStateAssessor

        tracker_name = cfg.get("tracker", {}).get("name", "sglatrack")
        tracker = TRACKERS.build(tracker_name, enable_ce=cfg.get("enable_ce", True))

        tsa = TargetStateAssessor(
            device=cfg.get("tsa", {}).get("device", "auto"),
            adapt_interval=cfg.get("tsa", {}).get("adapt_interval", 20),
            buffer_size=cfg.get("tsa", {}).get("buffer_size", 100),
            motion_threshold=cfg.get("tsa", {}).get("motion_residual_threshold", 0.5),
            drift_threshold=cfg.get("tsa", {}).get("drift_threshold", 0.35),
            adapt_enabled=cfg.get("tsa", {}).get("adapt_enabled", True),
            weights_path=cfg.get("tsa", {}).get("weights_path"),
            enable_velocity_drift=cfg.get("enable_velocity_drift", True),
        )

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

        runner = cls(
            tracker=tracker,
            tsa=tsa,
            detector=detector,
            appearance_memory=memory,
            motion_predictor=motion_pred,
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

        # BUG-14 fix: validate frame-0 GT bbox quality.
        # If the initial GT bbox is degenerate (w<=0 or h<=0), override the
        # default CONFIRMED compute routing for frame 1 to OCCLUDED so that
        # CE token pruning is not applied when initialization quality is poor.
        if gt[0].w <= 0 or gt[0].h <= 0:
            from uav_tracker.ml.tsa.target_state import TargetState
            self._prev_tsa_state_int = TargetState.OCCLUDED.value
            _logger.warning(
                "SALTRunner: degenerate GT bbox at frame 0 "
                "(w=%.1f h=%.1f) — routing frame 1 as OCCLUDED",
                gt[0].w, gt[0].h,
            )

        # Guard 3: store reference appearance embedding from the ground-truth crop
        # on frame 0. Used at recovery time to reject appearance-mismatched detections.
        try:
            self._ref_embedding = _get_embed_helper()._extract_embedding(frames[0], gt[0])
        except Exception:
            self._ref_embedding = None

        _state_counts: dict = {}

        yield TelemetryEntry(
            frame_idx=0,
            bbox=gt[0],
            confidence=1.0,
            tier=1,
            switched=False,
            aux={
                "target_state": "CONFIRMED",
                "tsa_confidence": 1.0,
                # SALT-RD telemetry — empty for init frame
                "score_map_stats": {},
                "apce_raw": 0.0,
                "psr_raw": 0.0,
                "entropy_raw": 0.0,
            },
        )
        _state_counts["CONFIRMED"] = _state_counts.get("CONFIRMED", 0) + 1

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
            sname = entry.aux.get("target_state", "UNKNOWN")
            _state_counts[sname] = _state_counts.get(sname, 0) + 1
            self._prev_frame = frame
            yield entry

        total = sum(_state_counts.values())
        _logger.warning(
            "TSA state distribution (%d frames): %s",
            total,
            {k: f"{v}({100 * v // total}%)" for k, v in sorted(_state_counts.items())},
        )

    def _step(self, frame: np.ndarray) -> TelemetryEntry:
        """Process one frame. Returns a TelemetryEntry (frame_idx not set).

        Two-step deferred consistency model
        ------------------------------------
        1. Run tracker using the *previous* frame's TSA state for compute routing.
        2. Run TSA on the *current* tracker output to validate it:
               consistency = IoU(tracker_pred_current, flow_warp(prev_bbox))
           This correctly detects drift/loss: a slow UAV that the tracker has
           abandoned will show low IoU here, whereas the old model (IoU of prev
           vs warp(prev)) just measured motion magnitude and nearly always ≥ 0.9.
        3. Cache TSA state for next frame's routing decision.
        4. If LOST: attempt detector-based recovery and re-init in this frame.
        """
        from uav_tracker.ml.tsa.target_state import TargetState

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

        # ---- Step 1: run tracker (uses previous-frame TSA state for budget) ----
        t_tracker = time.perf_counter()
        try:
            track_state: TrackState = self.tracker.update_with_state(
                frame, self._prev_tsa_state_int,
                consecutive_occluded=self._consecutive_occluded,
                prev_apce=self._prev_apce,
            )
        except AttributeError:
            track_state = self.tracker.update(frame)
        tracker_ms = (time.perf_counter() - t_tracker) * 1000

        # SALT-RD advisory p_fc (0.0 if no advisor attached)
        _advisor = getattr(self.tracker, '_salt_rd_advisor', None)
        _advisory_p_fc: float = _advisor.last_p_fc if _advisor is not None else 0.0

        # SALT-RD per-frame telemetry flags (updated below as events occur)
        _template_attempted: bool = False
        _template_updated: bool = False
        _reinit_vetoed: bool = False

        # Stage 3: Get SALT-RD primary state when advisor attached
        _saltrd_policy: dict | None = None
        _saltrd_state = None
        if _advisor is not None:
            _tsa_int = self._prev_tsa_state_int
            _saltrd_policy = _advisor.stage3_policy(_tsa_int)
            _saltrd_state = _saltrd_policy['state']

        # Extract score-map quality metrics (populated by SGLATracker; 0.0 for others)
        apce = getattr(track_state, 'apce', 0.0)
        psr = getattr(track_state, 'psr', 0.0)
        response_entropy = getattr(track_state, 'response_entropy', 0.0)

        # prev_bbox for flow warp: last confirmed position before this frame
        prev_bbox: BBox = self._trajectory[-1] if self._trajectory else track_state.bbox

        # ---- Step 2: TSA — assess current tracker output for consistency ----
        state_int = TargetState.CONFIRMED.value
        tsa_confidence = 1.0
        state_name = "CONFIRMED"

        # TSA temporal gating: on consecutive stable frames, skip full TSA assessment.
        # Use previous-frame APCE as proxy — if high, current frame is almost certainly
        # CONFIRMED too. Also guard with current-frame APCE to avoid gating transitions
        # to OCCLUDED (low current APCE even when previous was high).
        # Reduces optical flow computation by ~40% on easy sequences.
        _skip_tsa = (
            self._prev_tsa_state_int == TargetState.CONFIRMED.value
            and self._prev_apce > 120.0   # strong previous peak → likely stable
            and apce > 120.0              # current peak also high → not transitioning
            and self._frame_idx > 5       # always run full TSA for first few frames
        )

        t_tsa = time.perf_counter()
        if self.tsa and not _skip_tsa:
            try:
                assessment = self.tsa.assess(
                    frame=frame,
                    prev_frame=self._prev_frame,
                    tracker_pred_bbox=track_state.bbox,
                    prev_bbox=prev_bbox,
                    tracker_confidence=track_state.confidence,
                    lstm_pred_bbox=lstm_pred,
                    appearance_drift=drift,
                    apce=apce,
                    psr=psr,
                    response_entropy=response_entropy,
                )
                state_int = int(assessment.state)
                tsa_confidence = float(assessment.confidence)
                state_name = assessment.state.name
            except Exception as exc:
                _logger.debug("TSA assess failed: %s", exc)
        elif _skip_tsa:
            # Reuse previous CONFIRMED state — no Farneback needed
            state_int = TargetState.CONFIRMED.value
            tsa_confidence = 1.0
            state_name = "CONFIRMED"
        tsa_ms = (time.perf_counter() - t_tsa) * 1000

        # Log per-component timing every 100 frames
        if self._frame_idx % 100 == 0:
            _logger.warning(
                "Timing frame %d: tracker=%.1fms tsa=%.1fms",
                self._frame_idx, tracker_ms, tsa_ms,
            )
        self._prev_tsa_state_int = state_int
        self._prev_apce = apce  # store for next frame's TSA temporal gating

        # Guard 1: remember last good bbox for size-consistency filtering
        if state_int in (TargetState.CONFIRMED.value, TargetState.DYNAMIC.value):
            self._last_good_bbox = track_state.bbox

        # Guard 3: EMA update of reference embedding on stable CONFIRMED frames every 50.
        # SALT-RD gate: skip EMA when p_fc >= 0.30 — ref_embedding stays frozen if
        # model suspects false-confirmed risk (prevents cosine guard from drifting to
        # match the wrong target, which was the root cause of the car7 regression).
        if (state_int == TargetState.CONFIRMED.value
                and track_state.confidence >= 0.014   # top-3 softmax scale (~0.016 typical)
                and self._frame_idx % 50 == 0
                and self._ref_embedding is not None
                and _advisory_p_fc < 0.30):
            try:
                new_emb = _get_embed_helper()._extract_embedding(frame, track_state.bbox)
                # EMA: 80% old + 20% new — slow drift to handle legitimate appearance change
                self._ref_embedding = 0.80 * self._ref_embedding + 0.20 * new_emb
                # Re-normalise
                norm = np.linalg.norm(self._ref_embedding) + 1e-8
                self._ref_embedding = self._ref_embedding / norm
            except Exception:
                pass

        if self._frame_idx % 50 == 0:
            _logger.warning(
                "Frame %d: apce=%.1f psr=%.1f entropy=%.3f state=%s conf=%.3f",
                self._frame_idx, apce, psr, response_entropy, state_name, track_state.confidence,
            )

        # OCCLUDED escalation: if OCCLUDED persists > N frames, treat as LOST so
        # the recovery pipeline fires. Without this, uav2's 47% OCCLUDED frames
        # run indefinitely with no detector call (LOST threshold apce < 20 is too
        # tight — uav2 APCE sits at 20-80, always OCCLUDED, never LOST).
        #
        # Staged criterion: require both sustained frame count AND sustained low APCE
        # (mean of the last 25 APCE values < 0.75 * occluded_threshold).  This
        # prevents false escalation on sequences like car7 where APCE dips for 25
        # frames but the mean remains above threshold (tracker not genuinely lost).
        if state_int == TargetState.OCCLUDED.value:
            self._apce_buffer.append(apce)
            self._consecutive_occluded += 1
            _count_met = self._consecutive_occluded >= self._OCCLUDED_ESCALATION_FRAMES
            _apce_occluded_thr = getattr(self.tsa, '_apce_calibrator', None)
            _occ_thr = _apce_occluded_thr.thresholds()[1] if _apce_occluded_thr else 80.0
            _mean_low = (
                len(self._apce_buffer) >= 25
                and float(np.mean(list(self._apce_buffer))) < _occ_thr * 0.75
            )
            if _count_met and _mean_low:
                # APCE-trend gating: if APCE is rising ≥15% above the previous
                # escalated frame's APCE, the tracker is self-recovering — skip
                # escalation this frame so _consecutive_lost does not increment and
                # the recovery pipeline is not triggered prematurely (bike2 fix).
                _apce_trend_rising = (
                    self._prev_escalated_apce > 0.0
                    and apce >= self._prev_escalated_apce * 1.15
                )
                if _apce_trend_rising:
                    _logger.debug(
                        "SALT: APCE-trend gate blocked OCCLUDED→LOST at frame %d "
                        "(apce=%.1f prev_esc=%.1f)",
                        self._frame_idx, apce, self._prev_escalated_apce,
                    )
                else:
                    _logger.debug(
                        "SALT: escalating OCCLUDED→LOST after %d frames (mean_apce=%.1f thr=%.1f)",
                        self._consecutive_occluded,
                        float(np.mean(list(self._apce_buffer))),
                        _occ_thr * 0.75,
                    )
                    state_int = TargetState.LOST.value
                    state_name = "LOST"
                    self._prev_escalated_apce = apce
                    # Clear any cooldown on the first escalation frame so the recovery
                    # pipeline can fire immediately after _RECOVERY_MIN_LOST LOST frames.
                    # (Cooldown may have been set by a prior LOST→detector-fail event;
                    # re-opening the window here is intentional for the OCCLUDED path.)
                    if self._consecutive_occluded == self._OCCLUDED_ESCALATION_FRAMES:
                        self._lost_cooldown = 0
                    # Do NOT reset _consecutive_occluded: keep it elevated so every
                    # subsequent OCCLUDED frame also escalates to LOST until recovery
                    # succeeds. This lets _consecutive_lost accumulate to _RECOVERY_MIN_LOST.
        else:
            self._consecutive_occluded = 0  # reset on any non-OCCLUDED frame
            self._apce_buffer.clear()
            self._prev_escalated_apce = 0.0

        # Track consecutive LOST frames; reset on any non-LOST state
        if state_int == TargetState.LOST.value:
            self._consecutive_lost += 1
        else:
            self._consecutive_lost = 0
        if self._lost_cooldown > 0:
            self._lost_cooldown -= 1

        # ---- Step 4: Recovery — only after N consecutive LOST frames ----
        _genuine_loss = (self._consecutive_lost >= self._RECOVERY_MIN_LOST)
        _past_warmup = (self._frame_idx >= self._RECOVERY_WARMUP_FRAMES)
        if (_genuine_loss
                and _past_warmup
                and self.detector is not None
                and self._lost_cooldown == 0):
            try:
                # Bug 2 fix: use the tracker's frozen _state (last CONFIRMED position)
                # as the detector hint, not the drifted trajectory tail (prev_bbox).
                # After OCCLUDED→LOST escalation, prev_bbox keeps receiving drifted
                # track_state.bbox values, so it no longer points at the frozen centre.
                _frozen_hint = getattr(self.tracker, '_state', None) or prev_bbox
                _logger.warning(
                    "SALT recovery: frame=%d hint=(%s) consecutive_lost=%d consecutive_occluded=%d",
                    self._frame_idx,
                    f"{_frozen_hint.x:.0f},{_frozen_hint.y:.0f} {_frozen_hint.w:.0f}×{_frozen_hint.h:.0f}"
                    if _frozen_hint else "None",
                    self._consecutive_lost,
                    self._consecutive_occluded,
                )
                detections = self.detector.detect(frame, hint_bbox=_frozen_hint)
                if detections:
                    # Guard 4: class filter — keep only detections matching the target class
                    if self._target_class is not None:
                        class_filtered = [
                            d for d in detections
                            if getattr(d, 'class_id', self._target_class) == self._target_class
                        ]
                        if class_filtered:
                            detections = class_filtered
                        # else: fall back to all detections (no class match found)

                    n_lost = self._consecutive_lost + self._consecutive_occluded
                    best = self._best_detection(detections, frame, self._last_good_bbox,
                                                n_lost=n_lost)
                    _logger.warning(
                        "SALT recovery: %d detections after class-filter, best=%s accepted=%s",
                        len(detections),
                        f"({best.bbox.x:.0f},{best.bbox.y:.0f} {best.bbox.w:.0f}×{best.bbox.h:.0f})"
                        if best else "None",
                        best is not None,
                    )
                    # Guard 2: accumulate votes; only re-init after spatial consensus
                    if best is not None:
                        self._recovery_votes.append((best.bbox, self._last_recovery_sim))
                        # Guard 4: remember class of the best candidate for future filtering
                        if self._target_class is None:
                            self._target_class = getattr(best, 'class_id', None)

                    if len(self._recovery_votes) >= self._RECOVERY_VOTE_FRAMES:
                        winner, winner_sim = self._pick_voted_detection(self._recovery_votes)
                        self._recovery_votes = []
                        if winner is not None:
                            # Guard 5: velocity-based trajectory prediction check.
                            # Use last 5 non-LOST trajectory entries to estimate where
                            # the target SHOULD be. If the winner is significantly closer
                            # to that prediction than to _last_good_bbox, accept as usual.
                            # Otherwise require cosine_sim >= 0.70 to accept.
                            _w_cx = winner.x + winner.w / 2
                            _w_cy = winner.y + winner.h / 2

                            # Compute predicted center from last 5 non-LOST entries
                            _traj_confirmed = [
                                b for b in self._trajectory[-10:]
                                if b is not None
                            ]
                            if len(_traj_confirmed) >= 5:
                                # Fit linear velocity: mean of last 4 displacements
                                _pts = _traj_confirmed[-5:]
                                _disps_x = [
                                    (_pts[i + 1].x + _pts[i + 1].w / 2)
                                    - (_pts[i].x + _pts[i].w / 2)
                                    for i in range(4)
                                ]
                                _disps_y = [
                                    (_pts[i + 1].y + _pts[i + 1].h / 2)
                                    - (_pts[i].y + _pts[i].h / 2)
                                    for i in range(4)
                                ]
                                _vx = float(np.mean(_disps_x))
                                _vy = float(np.mean(_disps_y))
                                _n_lost = max(1, self._consecutive_lost)
                                _last_pt = _traj_confirmed[-1]
                                _pred_cx = _last_pt.x + _last_pt.w / 2 + _vx * _n_lost
                                _pred_cy = _last_pt.y + _last_pt.h / 2 + _vy * _n_lost
                            else:
                                # Not enough history — use _last_good_bbox center
                                _ref = self._last_good_bbox if self._last_good_bbox else (
                                    self._trajectory[-1] if self._trajectory else winner
                                )
                                _pred_cx = _ref.x + _ref.w / 2
                                _pred_cy = _ref.y + _ref.h / 2

                            _dist_to_pred = (
                                (_w_cx - _pred_cx) ** 2 + (_w_cy - _pred_cy) ** 2
                            ) ** 0.5

                            # Distance from winner to last good bbox center
                            _lgb = self._last_good_bbox if self._last_good_bbox else (
                                self._trajectory[-1] if self._trajectory else winner
                            )
                            _lgb_cx = _lgb.x + _lgb.w / 2
                            _lgb_cy = _lgb.y + _lgb.h / 2
                            _dist_to_lgb = (
                                (_w_cx - _lgb_cx) ** 2 + (_w_cy - _lgb_cy) ** 2
                            ) ** 0.5

                            _closer_to_pred = _dist_to_pred < _dist_to_lgb * 0.7
                            _high_sim = winner_sim >= 0.70

                            _logger.warning(
                                "SALT Guard5: winner=(%s) pred=(%.0f,%.0f) "
                                "dist_pred=%.1f dist_lgb=%.1f closer=%s "
                                "cosine_sim=%.3f high_sim=%s",
                                f"{winner.x:.0f},{winner.y:.0f} {winner.w:.0f}×{winner.h:.0f}",
                                _pred_cx, _pred_cy,
                                _dist_to_pred, _dist_to_lgb, _closer_to_pred,
                                winner_sim, _high_sim,
                            )

                            if not _closer_to_pred and not _high_sim:
                                _logger.warning(
                                    "SALT Guard5: REJECTED recovery "
                                    "(dist_pred=%.1f >= 0.7×dist_lgb=%.1f "
                                    "and cosine_sim=%.3f < 0.70)",
                                    _dist_to_pred, _dist_to_lgb * 0.7, winner_sim,
                                )
                                self._lost_cooldown = 40
                                self._prev_tsa_state_int = TargetState.OCCLUDED.value
                            elif _advisor is not None and _advisor.should_block_reinit():
                                # SALT-RD: p_fc >= fc_block_reinit — tracker likely false-confirmed, skip reinit
                                _logger.debug(
                                    "SALT-RD advisory: blocking reinit at frame %d (p_fc=%.3f)",
                                    self._frame_idx, _advisory_p_fc,
                                )
                                _reinit_vetoed = True
                                self._lost_cooldown = 25
                                self._prev_tsa_state_int = TargetState.OCCLUDED.value
                            else:
                                self.tracker.init(frame, winner)
                                track_state = TrackState(
                                    bbox=winner,
                                    confidence=1.0,
                                    status="locked",
                                )
                                if self.motion_predictor:
                                    self.motion_predictor.reset()
                                self._prev_tsa_state_int = TargetState.CONFIRMED.value
                                self._lost_cooldown = 0
                                self._consecutive_lost = 0  # reset after successful recovery
                                self._consecutive_occluded = 0  # reset so escalation restarts cleanly
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
                                        "target_state": state_name,
                                        "tsa_confidence": tsa_confidence,
                                        "recovered": True,
                                        # SALT-RD telemetry — empty for init frame
                                        "score_map_stats": {},
                                        "apce_raw": 0.0,
                                        "psr_raw": 0.0,
                                        "entropy_raw": 0.0,
                                    },
                                )
                        else:
                            # Votes accumulated but no spatial consensus — back off
                            self._lost_cooldown = 25
                            self._prev_tsa_state_int = TargetState.OCCLUDED.value
                    # else: still accumulating votes — continue as LOST this frame
                    elif best is None:
                        # No good candidate — back off for 25 frames, route as OCCLUDED
                        self._lost_cooldown = 25
                        self._prev_tsa_state_int = TargetState.OCCLUDED.value
                else:
                    # Detector found nothing — back off for 10 frames
                    self._lost_cooldown = 10
                    self._prev_tsa_state_int = TargetState.OCCLUDED.value
            except Exception as exc:
                _logger.debug("SALT recovery failed: %s", exc)
                self._lost_cooldown = 10

        self._trajectory.append(track_state.bbox)
        if len(self._trajectory) > 200:
            self._trajectory = self._trajectory[-200:]

        # ---- Appearance memory: store if confirmed (or distractor-risk at lower threshold) ----
        # DISTRACTOR_RISK must also store so the memory tracks appearance drift
        # and doesn't get stuck in a one-way trap where drift stays high forever.
        _should_store = (
            self.appearance_memory and (
                (state_int == TargetState.CONFIRMED.value and track_state.confidence >= 0.6)
                or (state_int == TargetState.DISTRACTOR_RISK.value and track_state.confidence >= 0.4)
            )
        )
        if _should_store:
            try:
                ctx = FrameContext(frame=frame, frame_idx=self._frame_idx)
                self.appearance_memory.store(ctx, track_state)
            except Exception:
                pass

        # ---- Motion predictor: online update ----
        if self.motion_predictor and state_int in (TargetState.CONFIRMED.value, TargetState.DYNAMIC.value):
            try:
                self.motion_predictor.update(track_state.bbox)
            except Exception:
                pass

        # Dynamic template update — gated by SALTRDAdvisor 5-gate temporal guard.
        # Previous p_fc<0.60 gate caused car7 0.570→0.321 regression: distractor appeared
        # before GRU accumulated evidence. New 5-gate guard catches rising edge and
        # score-map competition at t=1–2, before damage occurs.
        if (_advisor is not None
                and not _advisor.should_block_template_update()
                and state_int == TargetState.CONFIRMED.value):
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


        # Build aux dict; attach extended SALT-RD telemetry when an advisor is present
        _aux: dict[str, Any] = {
            "target_state": state_name,
            "tsa_confidence": tsa_confidence,
            "lstm_pred": (
                [lstm_pred.x, lstm_pred.y, lstm_pred.w, lstm_pred.h]
                if lstm_pred else None
            ),
            "appearance_drift": round(drift, 4),
            # SALT-RD telemetry — zero/empty-defaulted, does not change tracker behaviour
            "score_map_stats": getattr(track_state, "score_map_stats", {}),
            "apce_raw": getattr(track_state, "apce", 0.0),
            "psr_raw": getattr(track_state, "psr", 0.0),
            "entropy_raw": getattr(track_state, "response_entropy", 0.0),
        }
        if _advisor is not None:
            _aux["salt_rd_p_fc"] = _advisory_p_fc
            _aux["salt_rd_template_attempted"] = _template_attempted
            _aux["salt_rd_template_updated"] = _template_updated
            _aux["salt_rd_reinit_vetoed"] = _reinit_vetoed
            # Stage 3: SALT-RD primary state telemetry
            if _saltrd_policy is not None:
                _aux["saltrd_state"] = _saltrd_policy['state'].value
                _aux["saltrd_allow_ce"] = _saltrd_policy['allow_ce_pruning']
                _aux["saltrd_force_full"] = _saltrd_policy['force_full_compute']

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

        Spatial gate (Bug 3 fix): scales with n_lost. After N LOST/OCCLUDED
        frames the target may have moved far, so we widen the search radius.
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
        self._prev_tsa_state_int = 0  # CONFIRMED
        self._lost_cooldown = 0
        self._consecutive_lost = 0
        self._consecutive_occluded = 0
        self._prev_apce = 0.0
        self._apce_buffer.clear()
        self._prev_escalated_apce = 0.0
        self._last_good_bbox = None
        self._recovery_votes = []
        self._ref_embedding = None
        self._target_class = None
        self._last_recovery_sim = 0.0
        if self.tsa:
            self.tsa.reset()
        if self.appearance_memory:
            self.appearance_memory.reset()
        if self.motion_predictor:
            self.motion_predictor.reset()
        # Reset tracker template counters and advisor rolling buffers so they
        # do not leak between sequences.  tracker.reset() calls advisor.reset()
        # internally, so this is the single authoritative reset point.
        if hasattr(self.tracker, 'reset'):
            self.tracker.reset()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > 50:
            _logger.warning(
                "_reset() took %.1fms (> 50ms threshold) — slow component in TSA/memory/predictor reset",
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
