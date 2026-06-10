"""Causal per-frame feature builder for CSC.

Inputs are :class:`FrameLabel`-style dicts (already loaded from JSONL)
plus an image-size tuple per sequence.  Output is a fixed-width float
matrix of shape ``(T, F)`` where row ``t`` uses information from frames
``[0, t]`` only — no future-frame leakage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from csc_lib.csc.config import CSCFeatureConfig

# Feature vector v3fix (2026-05-29):
#   Slots 10-15 redesigned based on Cohen's d analysis on UAV123:
#   10  edge_contact_score      d=3.905 — bbox touching frame borders
#   11  log_w_ratio_to_init     log(w_t/w_0) — width explosion relative to start
#   12  log_area_ratio_to_init  d=2.576 — area explosion (log scale, clipped)
#   13  motion_angle_change     d=0.997 — direction change (kept, strongest motion signal)
#   14  log_h_ratio_to_init     log(h_t/h_0) — height explosion, complements slot 11
#   15  conf_ema_trend          EMA8-EMA32 of confidence — helps FC vs LA separation
FEATURE_NAMES: tuple[str, ...] = (
    "confidence",            # 0
    "apce",                  # 1
    "psr",                   # 2
    "cx_norm",               # 3
    "cy_norm",               # 4
    "w_norm",                # 5  d=3.5
    "h_norm",                # 6  d=3.3
    "area_norm",             # 7  d=2.9
    "aspect_ratio",          # 8  d=1.7
    "velocity_norm",         # 9  d=1.1
    "edge_contact_score",    # 10 d=3.9
    "log_w_ratio_to_init",   # 11 log(w_t/w_0) clipped [-3,6]
    "log_area_ratio_to_init",# 12 d=2.6 log(area_t/area_0) clipped [-3,6]
    "motion_angle_change",   # 13 d=1.0
    "log_h_ratio_to_init",   # 14 log(h_t/h_0) clipped [-3,6]
    "conf_ema_trend",        # 15 EMA8(conf) - EMA32(conf)
)
FEATURE_DIM = len(FEATURE_NAMES)

# Feature vector V2 (Run 2 — break v3fix shortcut):
#   Replaces 4 weak slots (8/11/14/15) with scale-context features that
#   distinguish "natural object approach" from "FC bbox explosion".
FEATURE_NAMES_V2: tuple[str, ...] = (
    "confidence",            # 0
    "apce",                  # 1
    "psr",                   # 2
    "cx_norm",               # 3
    "cy_norm",               # 4
    "w_norm",                # 5
    "h_norm",                # 6
    "area_norm",             # 7
    "log_aspect_ratio",      # 8  NEW: log(w/h) clipped [-3,3]
    "velocity_norm",         # 9
    "edge_contact_score",    # 10
    "edge_pressure_score",   # 11 NEW: continuous edge proximity, [0,1]
    "log_area_ratio_to_init",# 12
    "motion_angle_change",   # 13
    "scale_smoothness_8",    # 14 NEW: std of Δlog_area over 8 frames
    "aspect_instability_8",  # 15 NEW: std of log_aspect over 8 frames
)
FEATURE_DIM_V2 = len(FEATURE_NAMES_V2)

_EDGE_MARGIN = 0.02
_LOG_EPS     = 1e-8
_EMA8_ALPHA  = 2.0 / 9.0
_EMA32_ALPHA = 2.0 / 33.0
_EDGE_PRESSURE_MARGIN = 0.1
_HISTORY_LEN = 8


@dataclass
class _State:
    prev_cx: float | None = None
    prev_cy: float | None = None
    prev_vel: float | None = None
    prev_dx: float | None = None
    prev_dy: float | None = None
    # Init-anchored
    init_w: float | None = None
    init_h: float | None = None
    init_area: float | None = None
    # Confidence EMA
    conf_ema_short: float | None = None
    conf_ema_long: float | None = None
    # V2 history (unused by V1 builder)
    log_area_history: list = field(default_factory=list)
    log_aspect_history: list = field(default_factory=list)

    def __post_init__(self):
        pass


def _safe(x: float | None, default: float = 0.0) -> float:
    if x is None or not np.isfinite(x):
        return default
    return float(x)


def _compute_frame(
    pred_bbox: tuple[float, float, float, float] | None,
    confidence: float | None,
    image_size: tuple[int, int],
    state: _State,
) -> tuple:
    img_w, img_h = image_size
    img_w = max(1.0, float(img_w))
    img_h = max(1.0, float(img_h))
    img_diag = math.hypot(img_w, img_h)

    cx_n = cy_n = w_n = h_n = area_n = ar = vel_n = 0.0
    angle_change = edge_contact = 0.0
    log_w_ratio = log_area_ratio = log_h_ratio = 0.0

    if pred_bbox is not None:
        x, y, w, h = pred_bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        cx_n = cx / img_w
        cy_n = cy / img_h
        w_n = max(0.0, w) / img_w
        h_n = max(0.0, h) / img_h
        area_n = w_n * h_n
        ar = w / max(1e-6, h)

        # Edge contact score
        x1_n = x / img_w
        y1_n = y / img_h
        x2_n = (x + w) / img_w
        y2_n = (y + h) / img_h
        edge_contact = (
            float(x1_n <= _EDGE_MARGIN) +
            float(y1_n <= _EDGE_MARGIN) +
            float(x2_n >= 1.0 - _EDGE_MARGIN) +
            float(y2_n >= 1.0 - _EDGE_MARGIN)
        ) / 4.0

        # Init-anchored ratios
        if state.init_w is None:
            state.init_w = max(w_n, _LOG_EPS)
            state.init_h = max(h_n, _LOG_EPS)
            state.init_area = max(area_n, _LOG_EPS)

        log_w_ratio = float(np.clip(
            math.log((w_n + _LOG_EPS) / state.init_w), -3.0, 6.0))
        log_h_ratio = float(np.clip(
            math.log((h_n + _LOG_EPS) / state.init_h), -3.0, 6.0))
        log_area_ratio = float(np.clip(
            math.log((area_n + _LOG_EPS) / state.init_area), -3.0, 6.0))

        # Motion
        if state.prev_cx is not None and state.prev_cy is not None:
            dx = cx - state.prev_cx
            dy = cy - state.prev_cy
            vel = math.hypot(dx, dy)
            vel_n = vel / img_diag

            if state.prev_dx is not None and state.prev_dy is not None:
                prev_mag = math.hypot(state.prev_dx, state.prev_dy)
                if prev_mag > 1e-6 and vel > 1e-6:
                    cos_a = (dx * state.prev_dx + dy * state.prev_dy) / (vel * prev_mag)
                    angle_change = math.acos(max(-1.0, min(1.0, cos_a))) / math.pi

            state.prev_vel = vel
            state.prev_dx = dx
            state.prev_dy = dy

        state.prev_cx = cx
        state.prev_cy = cy

    # Confidence EMA trend
    conf_val = _safe(confidence)
    if state.conf_ema_short is None:
        state.conf_ema_short = conf_val
        state.conf_ema_long = conf_val
    else:
        state.conf_ema_short = _EMA8_ALPHA * conf_val + (1.0 - _EMA8_ALPHA) * state.conf_ema_short
        state.conf_ema_long = _EMA32_ALPHA * conf_val + (1.0 - _EMA32_ALPHA) * state.conf_ema_long
    conf_trend = state.conf_ema_short - state.conf_ema_long

    return (cx_n, cy_n, w_n, h_n, area_n, ar, vel_n,
            edge_contact, log_w_ratio, log_area_ratio,
            angle_change, log_h_ratio, conf_trend)


def build_runtime_feature(
    *,
    confidence: float | None,
    apce: float | None,
    psr: float | None,
    pred_bbox: tuple[float, float, float, float] | None,
    image_size: tuple[int, int],
    state: _State,
) -> np.ndarray:
    (cx_n, cy_n, w_n, h_n, area_n, ar, vel_n,
     edge_contact, log_w_ratio, log_area_ratio,
     angle_change, log_h_ratio, conf_trend) = _compute_frame(
        pred_bbox, confidence, image_size, state)
    return np.array([
        _safe(confidence), _safe(apce), _safe(psr),
        cx_n, cy_n, w_n, h_n, area_n, ar, vel_n,
        edge_contact, log_w_ratio, log_area_ratio,
        angle_change, log_h_ratio, conf_trend,
    ], dtype=np.float32)


def build_runtime_feature_into(
    buf: np.ndarray,
    *,
    confidence: float | None,
    apce: float | None,
    psr: float | None,
    pred_bbox: tuple[float, float, float, float] | None,
    image_size: tuple[int, int],
    state: _State,
) -> None:
    (cx_n, cy_n, w_n, h_n, area_n, ar, vel_n,
     edge_contact, log_w_ratio, log_area_ratio,
     angle_change, log_h_ratio, conf_trend) = _compute_frame(
        pred_bbox, confidence, image_size, state)
    buf[0]  = _safe(confidence)
    buf[1]  = _safe(apce)
    buf[2]  = _safe(psr)
    buf[3]  = cx_n
    buf[4]  = cy_n
    buf[5]  = w_n
    buf[6]  = h_n
    buf[7]  = area_n
    buf[8]  = ar
    buf[9]  = vel_n
    buf[10] = edge_contact
    buf[11] = log_w_ratio
    buf[12] = log_area_ratio
    buf[13] = angle_change
    buf[14] = log_h_ratio
    buf[15] = conf_trend


def build_offline_feature(
    *,
    iou: float | None,
    confidence: float | None,
    apce: float | None,
    psr: float | None,
    pred_bbox: tuple[float, float, float, float] | None,
    image_size: tuple[int, int],
    state: _State,
) -> np.ndarray:
    return build_runtime_feature(
        confidence=confidence, apce=apce, psr=psr,
        pred_bbox=pred_bbox, image_size=image_size, state=state,
    )


def build_sequence_features(
    rows: list[dict],
    image_size: tuple[int, int],
    *,
    cfg: CSCFeatureConfig | None = None,
    use_iou: bool = True,
) -> np.ndarray:
    cfg = cfg or CSCFeatureConfig()
    state = _State()
    out = np.zeros((len(rows), FEATURE_DIM), dtype=np.float32)
    for t, r in enumerate(rows):
        pred = tuple(r["pred_bbox"]) if r.get("pred_bbox") else None
        out[t] = build_offline_feature(
            iou=r.get("iou") if use_iou else None,
            confidence=r.get("confidence"),
            apce=r.get("apce"),
            psr=r.get("psr"),
            pred_bbox=pred,
            image_size=image_size,
            state=state,
        )
    np.clip(out, -cfg.clip_value, cfg.clip_value, out=out)
    return out


# ----------------------------------------------------------------------
# V2 feature builder (Run 2)
# Replaces 4 weak slots (8/11/14/15) of v3fix with scale-context features
# that distinguish "natural object approach" from "FC bbox explosion".
# Does NOT modify any V1 function above.
# ----------------------------------------------------------------------


def _compute_frame_v2(
    pred_bbox: tuple[float, float, float, float] | None,
    confidence: float | None,
    image_size: tuple[int, int],
    state: _State,
) -> tuple:
    """V2 frame builder. Returns 13-tuple of dynamic values used by V2.

    Returns: (cx_n, cy_n, w_n, h_n, area_n, log_aspect, vel_n,
              edge_contact, edge_pressure, log_area_ratio,
              angle_change, scale_smoothness, aspect_instability)
    """
    img_w, img_h = image_size
    img_w = max(1.0, float(img_w))
    img_h = max(1.0, float(img_h))
    img_diag = math.hypot(img_w, img_h)

    cx_n = cy_n = w_n = h_n = area_n = log_aspect = vel_n = 0.0
    angle_change = edge_contact = edge_pressure = 0.0
    log_area_ratio = scale_smoothness = aspect_instability = 0.0

    if pred_bbox is not None:
        x, y, w, h = pred_bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        cx_n = cx / img_w
        cy_n = cy / img_h
        w_n = max(0.0, w) / img_w
        h_n = max(0.0, h) / img_h
        area_n = w_n * h_n
        # log_aspect (replaces aspect_ratio at slot 8)
        log_aspect = float(np.clip(
            math.log(max(_LOG_EPS, w_n) / max(_LOG_EPS, h_n)),
            -3.0, 3.0
        ))

        # edge metrics
        x1_n = x / img_w
        y1_n = y / img_h
        x2_n = (x + w) / img_w
        y2_n = (y + h) / img_h
        edge_contact = (
            float(x1_n <= _EDGE_MARGIN) +
            float(y1_n <= _EDGE_MARGIN) +
            float(x2_n >= 1.0 - _EDGE_MARGIN) +
            float(y2_n >= 1.0 - _EDGE_MARGIN)
        ) / 4.0
        # edge_pressure_score (continuous)
        min_dist = min(x1_n, y1_n, 1.0 - x2_n, 1.0 - y2_n)
        edge_pressure = float(np.clip(
            1.0 - min_dist / _EDGE_PRESSURE_MARGIN,
            0.0, 1.0
        ))

        # init-anchored ratios (same as V1)
        if state.init_w is None:
            state.init_w = max(w_n, _LOG_EPS)
            state.init_h = max(h_n, _LOG_EPS)
            state.init_area = max(area_n, _LOG_EPS)
        log_area_ratio = float(np.clip(
            math.log((area_n + _LOG_EPS) / state.init_area), -3.0, 6.0))

        # motion (same as V1)
        if state.prev_cx is not None and state.prev_cy is not None:
            dx = cx - state.prev_cx
            dy = cy - state.prev_cy
            vel = math.hypot(dx, dy)
            vel_n = vel / img_diag

            if state.prev_dx is not None and state.prev_dy is not None:
                prev_mag = math.hypot(state.prev_dx, state.prev_dy)
                if prev_mag > 1e-6 and vel > 1e-6:
                    cos_a = (dx * state.prev_dx + dy * state.prev_dy) / (vel * prev_mag)
                    angle_change = math.acos(max(-1.0, min(1.0, cos_a))) / math.pi

            state.prev_vel = vel
            state.prev_dx = dx
            state.prev_dy = dy

        state.prev_cx = cx
        state.prev_cy = cy

        # Update histories
        state.log_area_history.append(log_area_ratio)
        state.log_aspect_history.append(log_aspect)
        if len(state.log_area_history) > _HISTORY_LEN + 1:
            state.log_area_history.pop(0)
        if len(state.log_aspect_history) > _HISTORY_LEN:
            state.log_aspect_history.pop(0)

        # scale_smoothness_8 (slot 14): std of deltas of log_area over last 8 frames
        if len(state.log_area_history) >= 2:
            arr = np.asarray(state.log_area_history, dtype=np.float32)
            deltas = np.diff(arr)
            scale_smoothness = float(np.std(deltas)) if len(deltas) >= 1 else 0.0

        # aspect_instability_8 (slot 15): std of log_aspect over last 8 frames
        if len(state.log_aspect_history) >= 2:
            aspect_instability = float(np.std(np.asarray(state.log_aspect_history, dtype=np.float32)))

    return (cx_n, cy_n, w_n, h_n, area_n, log_aspect, vel_n,
            edge_contact, edge_pressure, log_area_ratio,
            angle_change, scale_smoothness, aspect_instability)


def build_runtime_feature_v2(
    *,
    confidence: float | None,
    apce: float | None,
    psr: float | None,
    pred_bbox: tuple[float, float, float, float] | None,
    image_size: tuple[int, int],
    state: _State,
) -> np.ndarray:
    (cx_n, cy_n, w_n, h_n, area_n, log_aspect, vel_n,
     edge_contact, edge_pressure, log_area_ratio,
     angle_change, scale_smoothness, aspect_instability) = _compute_frame_v2(
        pred_bbox, confidence, image_size, state)
    return np.array([
        _safe(confidence), _safe(apce), _safe(psr),
        cx_n, cy_n, w_n, h_n, area_n,
        log_aspect, vel_n,
        edge_contact, edge_pressure,
        log_area_ratio, angle_change,
        scale_smoothness, aspect_instability,
    ], dtype=np.float32)


def build_runtime_feature_into_v2(
    buf: np.ndarray,
    *,
    confidence: float | None,
    apce: float | None,
    psr: float | None,
    pred_bbox: tuple[float, float, float, float] | None,
    image_size: tuple[int, int],
    state: _State,
) -> None:
    (cx_n, cy_n, w_n, h_n, area_n, log_aspect, vel_n,
     edge_contact, edge_pressure, log_area_ratio,
     angle_change, scale_smoothness, aspect_instability) = _compute_frame_v2(
        pred_bbox, confidence, image_size, state)
    buf[0]  = _safe(confidence)
    buf[1]  = _safe(apce)
    buf[2]  = _safe(psr)
    buf[3]  = cx_n
    buf[4]  = cy_n
    buf[5]  = w_n
    buf[6]  = h_n
    buf[7]  = area_n
    buf[8]  = log_aspect
    buf[9]  = vel_n
    buf[10] = edge_contact
    buf[11] = edge_pressure
    buf[12] = log_area_ratio
    buf[13] = angle_change
    buf[14] = scale_smoothness
    buf[15] = aspect_instability


def build_offline_feature_v2(
    *,
    iou: float | None,
    confidence: float | None,
    apce: float | None,
    psr: float | None,
    pred_bbox: tuple[float, float, float, float] | None,
    image_size: tuple[int, int],
    state: _State,
) -> np.ndarray:
    return build_runtime_feature_v2(
        confidence=confidence, apce=apce, psr=psr,
        pred_bbox=pred_bbox, image_size=image_size, state=state,
    )


def build_sequence_features_v2(
    rows: list[dict],
    image_size: tuple[int, int],
    *,
    cfg: CSCFeatureConfig | None = None,
    use_iou: bool = True,
) -> np.ndarray:
    cfg = cfg or CSCFeatureConfig()
    state = _State()
    out = np.zeros((len(rows), FEATURE_DIM_V2), dtype=np.float32)
    for t, r in enumerate(rows):
        pred = tuple(r["pred_bbox"]) if r.get("pred_bbox") else None
        out[t] = build_offline_feature_v2(
            iou=r.get("iou") if use_iou else None,
            confidence=r.get("confidence"),
            apce=r.get("apce"),
            psr=r.get("psr"),
            pred_bbox=pred,
            image_size=image_size,
            state=state,
        )
    np.clip(out, -cfg.clip_value, cfg.clip_value, out=out)
    return out


# ----------------------------------------------------------------------
# V3 feature builder (Run 4 — 2026-05-31)
# Adds response-map STRUCTURE passthroughs that out-discriminate failures
# (Cohen's d up to 1.56 on 291K LaSOT frames, vs the model's current best
# 1.28 for confidence) and are ALREADY populated 100% in BOTH training
# (baselines) and runtime (eval_v3fix) telemetry -> zero re-extraction.
# They target the "peaky-but-wrong" / distractor-competition blind spot that
# the coarse confidence/apce/psr summary misses (failures have ~4x more
# secondary response peaks). Does NOT modify any V1/V2 function.
# Missing values (trackers that don't emit sm_*) degrade safely to 0.0.
# NOTE: these passthroughs go in RAW (clipped); per-tracker percentile
# normalization of sm_* is a separate calibration concern (Phase 1b).
# ----------------------------------------------------------------------

# Response-structure fields appended after the 16 V2 slots, ordered by
# measured |Cohen's d| failure-discrimination (success IoU>=0.5 vs fail <0.2).
# Weak slots (sm_top2 0.66, sm_peak_width 0.48, sm_peak_distance 0.27) excluded.
V3_EXTRA_FIELDS: tuple[str, ...] = (
    "response_entropy",      # 16  d=1.45  diffuse map on failure
    "sm_local_peak_margin",  # 17  d=1.56  local peak dominance
    "sm_n_secondary",        # 18  d=1.47  # competing peaks (distractor)
    "sm_top1",               # 19  d=1.47  normalized top peak
    "sm_heatmap_mass_topk",  # 20  d=1.33  mass concentration
    "sm_peak_margin",        # 21  d=1.30  global peak dominance
    "sm_local_top2_ratio",   # 22  d=1.25  2nd/1st peak ratio (ambiguity)
)
FEATURE_NAMES_V3: tuple[str, ...] = FEATURE_NAMES_V2 + V3_EXTRA_FIELDS
FEATURE_DIM_V3 = len(FEATURE_NAMES_V3)


def build_runtime_feature_v3(
    *,
    confidence: float | None,
    apce: float | None,
    psr: float | None,
    pred_bbox: tuple[float, float, float, float] | None,
    image_size: tuple[int, int],
    state: _State,
    extra: dict | None = None,
) -> np.ndarray:
    """V3 runtime feature: V2 (16) + response-structure passthroughs (7).

    ``extra`` is the raw telemetry row (any mapping) providing V3_EXTRA_FIELDS.
    Missing fields degrade to 0.0 (tracker-agnostic graceful degradation).
    """
    base = build_runtime_feature_v2(
        confidence=confidence, apce=apce, psr=psr,
        pred_bbox=pred_bbox, image_size=image_size, state=state,
    )
    extra = extra or {}
    tail = np.array([_safe(extra.get(k)) for k in V3_EXTRA_FIELDS], dtype=np.float32)
    return np.concatenate([base, tail])


def build_runtime_feature_into_v3(
    buf: np.ndarray,
    *,
    confidence: float | None,
    apce: float | None,
    psr: float | None,
    pred_bbox: tuple[float, float, float, float] | None,
    image_size: tuple[int, int],
    state: _State,
    extra: dict | None = None,
) -> None:
    build_runtime_feature_into_v2(
        buf, confidence=confidence, apce=apce, psr=psr,
        pred_bbox=pred_bbox, image_size=image_size, state=state,
    )
    extra = extra or {}
    for i, k in enumerate(V3_EXTRA_FIELDS):
        buf[FEATURE_DIM_V2 + i] = _safe(extra.get(k))


def build_offline_feature_v3(
    *,
    iou: float | None,
    confidence: float | None,
    apce: float | None,
    psr: float | None,
    pred_bbox: tuple[float, float, float, float] | None,
    image_size: tuple[int, int],
    state: _State,
    extra: dict | None = None,
) -> np.ndarray:
    return build_runtime_feature_v3(
        confidence=confidence, apce=apce, psr=psr,
        pred_bbox=pred_bbox, image_size=image_size, state=state, extra=extra,
    )


def build_sequence_features_v3(
    rows: list[dict],
    image_size: tuple[int, int],
    *,
    cfg: CSCFeatureConfig | None = None,
    use_iou: bool = True,
) -> np.ndarray:
    cfg = cfg or CSCFeatureConfig()
    state = _State()
    out = np.zeros((len(rows), FEATURE_DIM_V3), dtype=np.float32)
    for t, r in enumerate(rows):
        pred = tuple(r["pred_bbox"]) if r.get("pred_bbox") else None
        out[t] = build_offline_feature_v3(
            iou=r.get("iou") if use_iou else None,
            confidence=r.get("confidence"),
            apce=r.get("apce"),
            psr=r.get("psr"),
            pred_bbox=pred,
            image_size=image_size,
            state=state,
            extra=r,
        )
    np.clip(out, -cfg.clip_value, cfg.clip_value, out=out)
    return out
