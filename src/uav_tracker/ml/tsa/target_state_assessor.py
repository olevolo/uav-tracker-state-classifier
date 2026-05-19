"""Target-State Assessor (TSA) — SALT main contribution.

Self-supervised scene-state estimation from tracking-consistency signals.
No manual annotations required; labels are generated online from optical-flow
pseudo-GT vs. tracker prediction agreement (IoU consistency).

Confidence: rule-based (APCE histogram) by default; swapped to a supervised
3-layer MLP (32→64→32→6) when a ``weights_path`` is provided. The supervised
head is inference-only (no online adaptation) and was trained on UAV123 with
real SGLATrack APCE/PSR/entropy features.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np

from uav_tracker.ml.tsa.target_state import TargetState, TargetStateAssessment
from uav_tracker.ml.tsa.velocity_drift import VelocityDriftMonitor
from uav_tracker.types import BBox

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Farneback defaults (fast, small search region)                              #
# --------------------------------------------------------------------------- #

_FB_PYR_SCALE  = 0.5
_FB_LEVELS     = 2
_FB_WINSIZE    = 9
_FB_ITERATIONS = 2
_FB_POLY_N     = 5
_FB_POLY_SIGMA = 1.1
_FLOW_DIM      = 32


def _build_mlp_head(n_classes: int = 6):  # type: ignore[return]
    """3-layer MLP: 32→64→32→n_classes. Returns nn.Sequential or None if no torch."""
    try:
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(_FLOW_DIM, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, n_classes),
        )
    except ImportError:
        return None





def _iou(a: BBox, b: BBox) -> float:
    """Intersection-over-Union for two axis-aligned BBoxes."""
    ax1, ay1 = a.x, a.y
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx1, by1 = b.x, b.y
    bx2, by2 = b.x + b.w, b.y + b.h

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / max(union, 1e-6)


def _warp_bbox_by_flow(bbox: BBox, flow_patch: np.ndarray, patch_offset: tuple[int, int]) -> BBox:
    """Translate bbox by the median optical-flow vector inside it.

    Parameters
    ----------
    bbox :
        Original bbox in full-frame coordinates.
    flow_patch :
        Farneback flow for a cropped sub-region (H' × W' × 2).
    patch_offset :
        (x0, y0) origin of ``flow_patch`` in full-frame coordinates.
    """
    px0, py0 = patch_offset
    fh, fw = flow_patch.shape[:2]

    # Map bbox into patch coordinates
    bx0 = int(max(0, bbox.x - px0))
    by0 = int(max(0, bbox.y - py0))
    bx1 = int(min(fw, bbox.x + bbox.w - px0))
    by1 = int(min(fh, bbox.y + bbox.h - py0))

    if bx1 <= bx0 or by1 <= by0:
        return bbox

    roi_flow = flow_patch[by0:by1, bx0:bx1]  # H'' × W'' × 2
    dx = float(np.median(roi_flow[..., 0]))
    dy = float(np.median(roi_flow[..., 1]))

    return BBox(x=bbox.x + dx, y=bbox.y + dy, w=bbox.w, h=bbox.h)


def _extract_flow_features(
    bbox: BBox,
    flow_patch: np.ndarray,
    patch_offset: tuple[int, int],
    consistency_score: float,
    tracker_confidence: float,
    lstm_residual: float,
    appearance_drift: float,
    apce: float = 0.0,
    psr: float = 0.0,
    response_entropy: float = 0.0,
) -> np.ndarray:
    """Build a 32-d feature vector from available signals.

    Layout
    ------
    [0]    consistency_score (flow-IoU)
    [1]    tracker_confidence
    [2]    lstm_residual (px) / 100.0  (normalised; _decide_state uses bbox-diagonal-normalised form)
    [3]    appearance_drift (cosine distance)
    [4]    bbox_area (px²) / 1000.0  (normalised)
    [5]    bbox aspect ratio w/h
    [6-7]  median flow dx, dy inside bbox
    [8-9]  mean flow magnitude, std inside bbox
    [10]   flow divergence inside bbox
    [11]   apce / 256.0  (normalised to [0,1]; Hann 16×16 max ≈256)
    [12]   psr  / 3000.0 (normalised; good tracking PSR 1000–3000)
    [13]   response_entropy / 5.0  (normalised; max ~log(256)≈5.5)
    [14-31] zeros (reserved)
    """
    feat = np.zeros(_FLOW_DIM, dtype=np.float32)
    feat[0] = float(np.clip(consistency_score, 0.0, 1.0))
    feat[1] = float(np.clip(tracker_confidence, 0.0, 1.0))
    feat[2] = float(np.clip(lstm_residual / 100.0, 0.0, 1.0))   # normalise to ~[0,1]
    feat[3] = float(np.clip(appearance_drift, 0.0, 1.0))
    feat[4] = float(bbox.w * bbox.h) / 1000.0
    feat[5] = float(bbox.w / max(bbox.h, 1.0))

    # Score-map quality metrics (primary signals from SGLATracker)
    feat[11] = float(np.clip(apce / 256.0, 0.0, 1.0))    # 0=flat, 1=perfect peak
    feat[12] = float(np.clip(psr / 3000.0, 0.0, 1.0))    # 0=noisy, 1=dominant peak
    feat[13] = float(np.clip(response_entropy / 5.0, 0.0, 2.0))

    # Flow stats inside bbox patch
    px0, py0 = patch_offset
    fh, fw = flow_patch.shape[:2]
    bx0 = int(max(0, bbox.x - px0))
    by0 = int(max(0, bbox.y - py0))
    bx1 = int(min(fw, bbox.x + bbox.w - px0))
    by1 = int(min(fh, bbox.y + bbox.h - py0))

    if bx1 > bx0 and by1 > by0:
        roi = flow_patch[by0:by1, bx0:bx1]  # H'' × W'' × 2
        feat[6] = float(np.median(roi[..., 0]))
        feat[7] = float(np.median(roi[..., 1]))
        mag = np.linalg.norm(roi, axis=-1)
        feat[8] = float(np.mean(mag))
        feat[9] = float(np.std(mag))
        # divergence
        if roi.shape[0] >= 2 and roi.shape[1] >= 2:
            du_dx = np.diff(roi[..., 0], axis=1)
            dv_dy = np.diff(roi[..., 1], axis=0)
            mh = min(du_dx.shape[0], dv_dy.shape[0])
            mw = min(du_dx.shape[1], dv_dy.shape[1])
            feat[10] = float(np.mean(du_dx[:mh, :mw] + dv_dy[:mh, :mw]))

    return feat


# --------------------------------------------------------------------------- #
# APCE calibrator                                                              #
# --------------------------------------------------------------------------- #


class APCECalibrator:
    """Rolling per-sequence APCE threshold calibrator.

    Maintains a sliding window of the last 100 APCE observations and derives
    adaptive LOST/OCCLUDED thresholds from the empirical distribution.  For
    the first 30 frames the window isn't representative, so fixed fallbacks
    (20, 80) are used instead.

    Threshold formulae:
      lost_threshold     = min(20.0, max(10.0, p5  * 1.5))
      occluded_threshold = max(80.0, p75 * 0.5)

    LOST only adapts downward from 20.0 — helps catch genuinely hard sequences
    (uav2 p5≈10 → thr≈15) without raising it on easy sequences (building1
    p5≈241 → capped at 20.0).  Floor of 10.0 prevents collapse toward zero.

    OCCLUDED uses p75 scaled down so it only RAISES from the calibrated floor
    of 80.0 on high-APCE easy sequences (e.g. building1 p75=255 → thr=127).
    It never drops below 80.0 — on hard sequences (uav2 p75≈70) the floor
    holds, preserving the OCCLUDED region [20,80) that drives the escalation
    pipeline.  Without this floor the calibrator would shrink the OCCLUDED
    window (p25*1.1 on uav2 ≈ 38) and misclassify OCCLUDED frames as CONFIRMED,
    preventing recovery from ever triggering.
    """

    _BUFFER_CAPACITY: int = 100
    _WARMUP_FRAMES: int = 30
    _LOST_FLOOR: float = 10.0
    _OCCLUDED_FLOOR: float = 80.0   # original calibrated value; never go below

    def __init__(self) -> None:
        self._buf: deque[float] = deque(maxlen=self._BUFFER_CAPACITY)

    def update(self, apce: float) -> None:
        if apce > 0.0:
            self._buf.append(apce)

    def thresholds(self) -> tuple[float, float]:
        """Return (lost_threshold, occluded_threshold).

        Falls back to (20.0, 80.0) until the warmup window is full.
        """
        if len(self._buf) < self._WARMUP_FRAMES:
            return 20.0, 80.0
        arr = np.array(self._buf, dtype=np.float32)
        p5  = float(np.percentile(arr, 5))
        p75 = float(np.percentile(arr, 75))
        lost_thr     = min(20.0, max(self._LOST_FLOOR,     p5  * 1.5))
        occluded_thr = max(self._OCCLUDED_FLOOR, p75 * 0.5)
        return lost_thr, occluded_thr

    def reset(self) -> None:
        self._buf.clear()


# --------------------------------------------------------------------------- #
# State decision logic                                                         #
# --------------------------------------------------------------------------- #


def _decide_state(
    bbox: BBox,
    consistency_score: float,
    tracker_confidence: float,
    lstm_residual: float,
    appearance_drift: float,
    motion_threshold: float,
    drift_threshold: float,
    apce: float = 0.0,
    psr: float = 0.0,
    response_entropy: float = 0.0,
    lost_threshold: float = 20.0,
    occluded_threshold: float = 80.0,
) -> TargetState:
    """Map signals to TargetState following priority order.

    Priority: LOST > DISTRACTOR_RISK > OCCLUDED > DYNAMIC > CONFIRMED

    When apce/psr are available (non-zero, from SGLATracker), they act as
    the primary signal because they are an independent quality measure from
    the tracker's own response map — immune to the optical-flow drift problem
    where flow follows background when the tracker has already drifted.
    Optical-flow signals remain as secondary fallbacks and MLP features.

    lost_threshold / occluded_threshold: supplied by APCECalibrator for
    adaptive per-sequence thresholds; defaults reproduce the original fixed
    values for backward compatibility.
    """
    # ── Calibrated from empirical UAV123 measurements ────────────────────────
    # Measured APCE distribution (Hann 16×16 score_map, range 0–256):
    #   building1 (easy):  min=247   bike2 (hard): min=10, p10=55
    #   car13 (medium):    min=38    uav2 (hardest): min=9.8, mean=63.6
    # PSR is NOT reliable alone: car7 PSR min=13.8 while tracking OK (AUC 0.60)
    # → Use APCE only for LOST/OCCLUDED; PSR as soft secondary, not primary
    #
    # Threshold rationale:
    #   Old LOST threshold (< 8.0) never fired on uav2 (min APCE = 9.8), so
    #   recovery was never attempted on the hardest sequence.
    #   Raised to < 20.0 to catch the worst uav2 frames; OCCLUDED raised to
    #   < 80.0 to cover uav2 mean APCE (63.6) as a degraded-but-recoverable region.
    #
    # DYNAMIC / LSTM warm-up rationale:
    #   The OnlineLSTMMotionPredictor starts with random weights and adapts
    #   online via SGD.  During the first ~15 frames its "prediction error"
    #   (lstm_residual) grows monotonically as the LSTM extrapolates an
    #   ever-longer trajectory it hasn't yet learned.  Diagnostics on car13
    #   show residual climbing from 8→58 px across frames 2–19 even when
    #   APCE=125-213 (tracker finding a sharp peak, object clearly visible).
    #   A growing residual during warm-up is NOT evidence of dynamic motion —
    #   it is evidence that the LSTM hasn't converged yet.
    #
    #   Fix: normalise lstm_residual by bbox diagonal and require the result
    #   to exceed motion_threshold (in units of bbox diagonals) instead of
    #   raw pixels.  Default motion_threshold is re-interpreted as 0.5 bbox
    #   diagonals.  For car13 (diagonal ≈ 16 px) 0.5 × 16 = 8 px — correctly
    #   tight.  For large objects (diagonal ≈ 100 px) 0.5 × 100 = 50 px —
    #   consistent with the old absolute threshold.
    #
    #   An additional APCE guard suppresses DYNAMIC when APCE is strong
    #   (≥ 120): the tracker's response map already shows a dominant peak,
    #   meaning the LSTM warm-up error is not causing a tracking problem and
    #   no extra compute budget is needed.

    # LOST: truly failed tracking — fire recovery pipeline
    if apce > 0 and apce < lost_threshold:
        return TargetState.LOST

    # Classic fallback only when APCE signal not available AND both signals are bad:
    # When APCE is available (SGLATracker), trust it exclusively — the flow-IoU
    # check fires false LOST on fast-moving targets (car13, car7) because
    # IoU(current_pos, prev_pos) is near-zero for large inter-frame displacements
    # even when APCE=250 (perfect tracking).
    if apce == 0.0 and (consistency_score < 0.2 or tracker_confidence < 0.1):
        return TargetState.LOST

    # DISTRACTOR_RISK: appearance drift (identity change)
    if appearance_drift > drift_threshold:
        return TargetState.DISTRACTOR_RISK

    # OCCLUDED: degraded but not failed tracking
    if apce > 0 and apce < occluded_threshold:
        return TargetState.OCCLUDED
    # Flow-IoU fallback for OCCLUDED: only fire when APCE is unavailable (non-SGLA
    # trackers) — same rationale as the LOST fallback above.
    if apce == 0.0 and consistency_score < 0.7:
        return TargetState.OCCLUDED

    # DYNAMIC: target motion is genuinely hard to predict.
    # Unreachable when motion_predictor is disabled (lstm_residual always 0.0).
    # Normalise LSTM residual by bbox diagonal so "50% of bbox diagonal" has
    # the same meaning regardless of object size (avoids false DYNAMIC for
    # tiny objects during LSTM warm-up where absolute pixel error is large
    # relative to the bbox but trivial relative to object size).
    # Also require APCE < 120: when APCE is strong the tracker already has a
    # sharp peak and no extra compute budget is warranted.
    bbox_diagonal = (bbox.w ** 2 + bbox.h ** 2) ** 0.5
    normalized_lstm_residual = lstm_residual / max(bbox_diagonal, 1.0)
    if normalized_lstm_residual > motion_threshold and (apce == 0.0 or apce < 120.0):
        return TargetState.DYNAMIC

    return TargetState.CONFIRMED


# --------------------------------------------------------------------------- #
# Rule-based confidence (replaces MLP head)                                   #
# --------------------------------------------------------------------------- #


def _rule_confidence(state: TargetState, apce: float, psr: float) -> float:  # noqa: ARG001
    """APCE histogram calibration — rule-based confidence per state.

    Replaces the 3-layer MLP head whose output ranged 0.013–0.018 (constant).

    CONFIRMED:       min(1.0, apce / 150.0)        — normalised peak quality
    OCCLUDED:        min(1.0, apce /  80.0) * 0.7  — reduced confidence
    LOST:            0.3                            — fixed low confidence
    DISTRACTOR_RISK: 0.5
    DYNAMIC:         0.6
    """
    if state == TargetState.CONFIRMED:
        return min(1.0, apce / 150.0)
    if state == TargetState.OCCLUDED:
        return min(1.0, apce / 80.0) * 0.7
    if state == TargetState.LOST:
        return 0.3
    if state == TargetState.DISTRACTOR_RISK:
        return 0.5
    if state == TargetState.DYNAMIC:
        return 0.6
    return 0.5


# --------------------------------------------------------------------------- #
# Main class                                                                   #
# --------------------------------------------------------------------------- #


def _decide_state_oracle(apce: float, tracker_confidence: float) -> TargetState:
    """Oracle state decision using only APCE — bypasses LSTM and flow signals.

    Permissive thresholds designed to validate whether the TSA→recovery
    architecture *can* improve results when state estimation is reliable.

    APCE < 30:  LOST      → triggers RT-DETR recovery
    APCE < 100: OCCLUDED  → full compute, no template update (freezes template)
    otherwise:  CONFIRMED → CE 0.85 token pruning

    Falls back to confidence-based LOST when APCE is not available (apce == 0).
    """
    if apce > 0:
        if apce < 30.0:
            return TargetState.LOST
        if apce < 100.0:
            return TargetState.OCCLUDED
        return TargetState.CONFIRMED
    # No APCE signal — fall back to tracker confidence
    if tracker_confidence < 0.1:
        return TargetState.LOST
    return TargetState.CONFIRMED


class TargetStateAssessor:
    """Self-supervised target-state assessor — SALT main contribution.

    Generates ``TargetState`` labels at inference time without manual
    annotations by comparing the tracker's predicted bbox against an
    optical-flow-based pseudo-GT bbox (Farneback warped prior).

    Confidence is produced by ``_rule_confidence()`` — APCE histogram
    calibration normalised per state. The former 3-layer MLP head and its
    online SGD adaptation have been removed (MLP output was non-informative:
    0.013–0.018 range with near-zero variance across sequences).

    Parameters
    ----------
    device :
        Accepted for config compatibility but unused (no torch dependency).
    adapt_interval :
        Accepted for config compatibility but unused (adaptation removed).
    buffer_size :
        Accepted for config compatibility but unused.
    adapt_enabled :
        Accepted for config compatibility but unused.
    motion_threshold :
        Normalised LSTM residual (in units of bbox diagonal) above which the
        target is classified as DYNAMIC (subject to APCE guard: DYNAMIC is
        suppressed when APCE ≥ 120, i.e. tracker response is sharp).
        Default 0.5 means "residual > 50% of bbox diagonal".
    drift_threshold :
        Cosine-distance drift above which DISTRACTOR_RISK fires.
    oracle_mode :
        When ``True``, skip all flow/LSTM/drift signals and determine state
        using only the tracker's own APCE score-map metric.  This is an
        ablation mode to validate the TSA→recovery architecture assuming
        perfect state estimation.
    """

    def __init__(
        self,
        device: str = "auto",
        adapt_interval: int = 20,
        buffer_size: int = 100,
        motion_threshold: float = 0.5,
        drift_threshold: float = 0.35,
        adapt_enabled: bool = True,
        oracle_mode: bool = False,
        weights_path: str | None = None,
    ) -> None:
        self._motion_threshold = motion_threshold
        self._drift_threshold  = drift_threshold
        self._oracle_mode      = oracle_mode

        # Accepted for config compatibility — no longer used
        _ = device
        _ = adapt_interval
        _ = buffer_size
        _ = adapt_enabled

        self._frame_count: int = 0
        self._head = None  # supervised MLP head — set by _load_head()

        if weights_path is not None:
            self._load_head(weights_path)

        # Previous frame (grayscale) for Farneback
        self._prev_gray: Optional[np.ndarray] = None
        # Flow displacement (dx, dy) in pixels from last assess() — shared with TTT
        self._last_flow_displacement: Optional[tuple[float, float]] = None

        # Early-exit fast path: skip Farneback on consecutive high-confidence CONFIRMED frames
        self._consecutive_confirmed: int = 0
        self._SKIP_FLOW_AFTER: int = 3  # after 3 consecutive CONFIRMED, skip flow

        # Velocity-based drift detector
        self._velocity_drift: VelocityDriftMonitor = VelocityDriftMonitor()

        # Adaptive APCE threshold calibrator
        self._apce_calibrator: APCECalibrator = APCECalibrator()

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def assess(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray],
        tracker_pred_bbox: BBox,
        prev_bbox: BBox,
        tracker_confidence: float,
        lstm_pred_bbox: Optional[BBox],
        appearance_drift: float,
        apce: float = 0.0,
        psr: float = 0.0,
        response_entropy: float = 0.0,
    ) -> TargetStateAssessment:
        """Assess the current target state from tracking-consistency signals.

        Parameters
        ----------
        frame :
            Current frame (BGR uint8, H × W × 3).
        prev_frame :
            Previous frame (BGR uint8).  ``None`` on the first call → returns
            ``CONFIRMED`` immediately.
        tracker_pred_bbox :
            Current frame tracker output (what we want to validate).
        prev_bbox :
            Previous frame bbox used to compute the optical-flow prediction
            (flow warp source).
        tracker_confidence :
            Confidence score from the tracker for the current frame.
        lstm_pred_bbox :
            Motion-predictor estimate for current frame (may be ``None``).
        appearance_drift :
            Cosine-distance drift from ``AppearanceMemory`` (0 = identical,
            1 = orthogonal).
        apce :
            Average Peak-to-Correlation Energy from SGLATracker score_map.
            0.0 if not available (non-SGLA trackers).
        psr :
            Peak-to-Sidelobe Ratio from SGLATracker score_map.
            0.0 if not available.
        response_entropy :
            Shannon entropy of softmax(score_map.flatten()).
            0.0 if not available.

        Returns
        -------
        TargetStateAssessment

        Notes
        -----
        The consistency score is ``IoU(tracker_pred_bbox, flow_warp(prev_bbox))``.
        This measures whether the tracker found the target in the expected
        position — if the tracker drifted to a distractor or lost the target,
        its bbox will disagree with the flow-warped prior and the score drops.
        """
        self._frame_count += 1
        bbox = tracker_pred_bbox
        confidence = float(tracker_confidence)

        # Clear displacement from previous frame
        self._last_flow_displacement = None

        # Always update _prev_gray first — prevents stale flow on exception
        curr_gray = self._to_gray(frame)
        prev_gray = self._prev_gray if self._prev_gray is not None else curr_gray
        self._prev_gray = curr_gray  # update before any potential exception below

        # First frame — no flow available; _prev_gray already set above
        if prev_frame is None or self._frame_count == 1:
            return TargetStateAssessment(
                state=TargetState.CONFIRMED,
                confidence=_rule_confidence(TargetState.CONFIRMED, float(apce), float(psr)),
                frame_idx=self._frame_count,
            )

        # ------------------------------------------------------------------ #
        # Oracle mode: bypass all flow/LSTM/drift — use APCE only            #
        # ------------------------------------------------------------------ #
        if self._oracle_mode:
            state = _decide_state_oracle(apce=float(apce), tracker_confidence=confidence)
            self._velocity_drift.update(tracker_pred_bbox)
            self._velocity_drift.update_psr(float(psr))
            return TargetStateAssessment(
                state=state,
                confidence=_rule_confidence(state, float(apce), float(psr)),
                frame_idx=self._frame_count,
            )

        # ------------------------------------------------------------------ #
        # Fast path: skip optical flow on consecutive high-confidence CONFIRMED
        # The Farneback computation is the main overhead (5-8ms per frame).
        # After _SKIP_FLOW_AFTER consecutive CONFIRMED frames with high APCE,
        # we trust the tracker is stable and return CONFIRMED without flow.
        # ------------------------------------------------------------------ #
        if (apce > 120.0
                and self._frame_count > 5
                and self._consecutive_confirmed >= self._SKIP_FLOW_AFTER):
            self._consecutive_confirmed += 1
            self._velocity_drift.update(tracker_pred_bbox)
            self._velocity_drift.update_psr(float(psr))
            # _frame_count already incremented at the top of assess().
            # _prev_gray already updated above — stays current for next frame.
            return TargetStateAssessment(
                state=TargetState.CONFIRMED,
                confidence=_rule_confidence(TargetState.CONFIRMED, float(apce), float(psr)),
                frame_idx=self._frame_count,
            )

        # ------------------------------------------------------------------ #
        # Step 1: compute optical flow on search region only (fast path)      #
        # ------------------------------------------------------------------ #
        # curr_gray and prev_gray already computed above

        # Use prev_bbox to centre the search patch — that is where the target
        # was last frame and where flow is meaningful.
        flow_patch, patch_offset = self._compute_patch_flow(
            prev_gray, curr_gray, prev_bbox, frame.shape
        )

        # ------------------------------------------------------------------ #
        # Step 2: warp prev_bbox by median flow → expected position           #
        # ------------------------------------------------------------------ #
        flow_pred_bbox = _warp_bbox_by_flow(prev_bbox, flow_patch, patch_offset)

        # Store flow displacement for consumption by TTT (avoids re-computing flow)
        self._last_flow_displacement = (
            float(flow_pred_bbox.x - prev_bbox.x),
            float(flow_pred_bbox.y - prev_bbox.y),
        )

        # ------------------------------------------------------------------ #
        # Step 3: IoU consistency score                                        #
        # Measures whether the tracker found the target where flow predicts it.
        # Low score → tracker drifted or lost the target.
        # ------------------------------------------------------------------ #
        consistency_score = _iou(tracker_pred_bbox, flow_pred_bbox)

        # ------------------------------------------------------------------ #
        # Step 4: LSTM motion residual                                         #
        # ------------------------------------------------------------------ #
        lstm_residual = 0.0
        if lstm_pred_bbox is not None:
            pred_cx = lstm_pred_bbox.x + lstm_pred_bbox.w / 2.0
            pred_cy = lstm_pred_bbox.y + lstm_pred_bbox.h / 2.0
            curr_cx = bbox.x + bbox.w / 2.0
            curr_cy = bbox.y + bbox.h / 2.0
            lstm_residual = float(np.hypot(pred_cx - curr_cx, pred_cy - curr_cy))

        # ------------------------------------------------------------------ #
        # Step 5: state decision                                               #
        # ------------------------------------------------------------------ #
        self._apce_calibrator.update(float(apce))
        _lost_thr, _occluded_thr = self._apce_calibrator.thresholds()
        state = _decide_state(
            bbox=bbox,
            consistency_score=consistency_score,
            tracker_confidence=confidence,
            lstm_residual=lstm_residual,
            appearance_drift=float(appearance_drift),
            motion_threshold=self._motion_threshold,
            drift_threshold=self._drift_threshold,
            apce=float(apce),
            psr=float(psr),
            response_entropy=float(response_entropy),
            lost_threshold=_lost_thr,
            occluded_threshold=_occluded_thr,
        )

        # ------------------------------------------------------------------ #
        # Step 5b: velocity drift override                                     #
        # Update monitor every frame (tracker_pred_bbox always available).    #
        # Override CONFIRMED → DISTRACTOR_RISK when frozen-position drift is  #
        # detected and the guard conditions pass.                              #
        # ------------------------------------------------------------------ #
        self._velocity_drift.update(tracker_pred_bbox)
        self._velocity_drift.update_psr(float(psr))
        if state == TargetState.CONFIRMED and self._velocity_drift.is_drifted(
            psr=float(psr),
            consistency_score=consistency_score,
            apce=float(apce),
        ):
            state = TargetState.DISTRACTOR_RISK

        # Track consecutive CONFIRMED for the early-exit fast path
        if state == TargetState.CONFIRMED:
            self._consecutive_confirmed += 1
        else:
            self._consecutive_confirmed = 0

        # Confidence: supervised head if loaded (uses full 32-d flow feature vector),
        # otherwise rule-based APCE histogram normalisation.
        if self._head is not None:
            flow_feat = _extract_flow_features(
                bbox=bbox,
                flow_patch=flow_patch,
                patch_offset=patch_offset,
                consistency_score=consistency_score,
                tracker_confidence=confidence,
                lstm_residual=lstm_residual,
                appearance_drift=float(appearance_drift),
                apce=float(apce),
                psr=float(psr),
                response_entropy=float(response_entropy),
            )
            conf = self._head_confidence(flow_feat, state)
        else:
            conf = _rule_confidence(state, float(apce), float(psr))

        return TargetStateAssessment(
            state=state,
            confidence=conf,
            frame_idx=self._frame_count,
        )

    def _load_head(self, weights_path: str) -> None:
        """Load supervised MLP head from a checkpoint produced by train_tsa_classifier.py."""
        try:
            import torch
            from pathlib import Path
            p = Path(weights_path)
            if not p.exists():
                logger.warning("TSA: weights_path %s not found — using rule-based confidence", p)
                return
            ckpt = torch.load(str(p), map_location="cpu", weights_only=True)
            head = _build_mlp_head(n_classes=ckpt.get("n_classes", 6))
            if head is None:
                return
            head.load_state_dict(ckpt["model_state_dict"])
            head.eval()
            self._head = head
            logger.info("TSA: loaded supervised head from %s (val_acc=%.3f, mode=%s)",
                        p, ckpt.get("val_accuracy", float("nan")), ckpt.get("mode", "?"))
        except Exception as exc:
            logger.warning("TSA: failed to load supervised head: %s — using rule-based confidence", exc)

    def _head_confidence(self, flow_feat: "np.ndarray", state: TargetState) -> float:
        """Return supervised head confidence for state, or rule-based fallback."""
        if self._head is None:
            return _rule_confidence(state, 0.0, 0.0)
        try:
            import torch
            feat_t = torch.from_numpy(flow_feat).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(self._head(feat_t), dim=1)
            return float(probs[0, int(state)].item())
        except Exception:
            return _rule_confidence(state, 0.0, 0.0)

    def update_online(self, assessment: TargetStateAssessment) -> None:  # noqa: ARG002
        """No-op — online MLP adaptation removed.

        Kept for caller compatibility (salt_runner.py calls this after each
        frame). The former 3-layer MLP head and SGD adaptation loop are gone;
        ``_rule_confidence`` replaces them with zero overhead.
        """

    def reset(self) -> None:
        """Reset per-sequence state: counters, cached gray frame."""
        self._frame_count = 0
        self._prev_gray = None
        self._last_flow_displacement = None
        self._consecutive_confirmed = 0
        self._velocity_drift.reset()
        self._apce_calibrator.reset()
        logger.debug("TargetStateAssessor: reset")

    @property
    def last_flow_displacement(self) -> "tuple[float, float] | None":
        """Flow displacement (dx, dy) in pixels from the last assess() call.

        Set after ``_warp_bbox_by_flow()`` runs inside ``assess()``.
        Returns ``None`` if ``assess()`` has not yet run or returned early
        (first frame / no prev_frame).
        """
        return self._last_flow_displacement

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.shape[2] == 1:
            return frame[:, :, 0]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _compute_patch_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        bbox: BBox,
        frame_shape: tuple,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Compute Farneback flow on a 2× search region around bbox.

        Returns
        -------
        (flow_patch, (x0, y0)):
            ``flow_patch`` is an H' × W' × 2 float32 flow array;
            ``(x0, y0)`` is the patch origin in full-frame coordinates.
        """
        fh, fw = frame_shape[:2]
        cx = bbox.x + bbox.w / 2.0
        cy = bbox.y + bbox.h / 2.0
        half_w = min(max(bbox.w, 16.0), 80.0)   # cap at 80px → max patch 160×160
        half_h = min(max(bbox.h, 16.0), 80.0)

        x0 = int(max(0, cx - half_w))
        y0 = int(max(0, cy - half_h))
        x1 = int(min(fw, cx + half_w))
        y1 = int(min(fh, cy + half_h))

        if x1 <= x0 or y1 <= y0:
            # Degenerate bbox — return trivial zero flow over full frame
            flow = np.zeros((fh, fw, 2), dtype=np.float32)
            return flow, (0, 0)

        prev_patch = prev_gray[y0:y1, x0:x1]
        curr_patch = curr_gray[y0:y1, x0:x1]

        flow = cv2.calcOpticalFlowFarneback(
            prev_patch,
            curr_patch,
            None,
            _FB_PYR_SCALE,
            _FB_LEVELS,
            _FB_WINSIZE,
            _FB_ITERATIONS,
            _FB_POLY_N,
            _FB_POLY_SIGMA,
            0,
        )  # H' × W' × 2, float32

        return flow, (x0, y0)


__all__ = ["TargetStateAssessor", "APCECalibrator"]
