"""Causal per-frame feature builder for CSC.

Inputs are :class:`FrameLabel`-style dicts (already loaded from JSONL)
plus an image-size tuple per sequence.  Output is a fixed-width float
matrix of shape ``(T, F)`` where row ``t`` uses information from frames
``[0, t]`` only — no future-frame leakage.

Why not reuse the 28-dim SALT-RD schema?  The SALT-RD schema includes
fields that need a full SGLATrack response map (response peak, token
keep ratio).  Stage-1 keeps the feature set narrow and tracker-agnostic
so we can run it on KCF + SGLATrack alike.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from csc_lib.csc.config import CSCFeatureConfig

# Feature names — keep stable: training and inference must agree.
# IMPORTANT: ``iou`` is intentionally NOT in this list.  Including it
# would make CSC trivially memorise the offline IoU-thresholding rule
# used to generate the labels, and the model would collapse to ~3%
# macro-F1 the moment it runs without GT (verified empirically — see
# CSC.md "Anti-patterns").
FEATURE_NAMES: tuple[str, ...] = (
    "confidence",
    "apce",
    "psr",
    "cx_norm",
    "cy_norm",
    "w_norm",
    "h_norm",
    "area_norm",
    "aspect_ratio",
    "velocity_norm",
    "accel_norm",
)
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass
class _State:
    prev_cx: float | None = None
    prev_cy: float | None = None
    prev_vel: float | None = None


def _safe(x: float | None, default: float = 0.0) -> float:
    if x is None or not np.isfinite(x):
        return default
    return float(x)


def build_runtime_feature(
    *,
    confidence: float | None,
    apce: float | None,
    psr: float | None,
    pred_bbox: tuple[float, float, float, float] | None,
    image_size: tuple[int, int],
    state: _State,
) -> np.ndarray:
    """One-frame causal feature.  Used both for offline training (when
    we have ground truth + tracker prediction) and at runtime (when we
    only have tracker prediction).

    Note: the ``iou`` slot is intentionally zero at runtime — runtime
    CSC does not see GT.  Training fills it from the offline label
    JSONL row.  Models that overfit on the IoU slot will behave badly
    online; we rely on regularisation + the clip_value to soften this.
    """
    img_w, img_h = image_size
    img_w = max(1.0, float(img_w))
    img_h = max(1.0, float(img_h))
    img_diag = math.hypot(img_w, img_h)

    cx_n = cy_n = w_n = h_n = area_n = ar = vel_n = acc_n = 0.0
    if pred_bbox is not None:
        x, y, w, h = pred_bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        cx_n = cx / img_w
        cy_n = cy / img_h
        w_n = max(0.0, w) / img_w
        h_n = max(0.0, h) / img_h
        area_n = w_n * h_n
        ar = (w / max(1e-6, h)) if h > 0 else 0.0

        if state.prev_cx is not None and state.prev_cy is not None:
            dx = cx - state.prev_cx
            dy = cy - state.prev_cy
            vel = math.hypot(dx, dy)
            vel_n = vel / img_diag
            if state.prev_vel is not None:
                acc_n = (vel - state.prev_vel) / img_diag
            state.prev_vel = vel
        state.prev_cx = cx
        state.prev_cy = cy

    feats = np.array(
        [
            _safe(confidence),
            _safe(apce),
            _safe(psr),
            cx_n,
            cy_n,
            w_n,
            h_n,
            area_n,
            ar,
            vel_n,
            acc_n,
        ],
        dtype=np.float32,
    )
    return feats


def build_offline_feature(
    *,
    iou: float | None,        # accepted but ignored (kept for API stability)
    confidence: float | None,
    apce: float | None,
    psr: float | None,
    pred_bbox: tuple[float, float, float, float] | None,
    image_size: tuple[int, int],
    state: _State,
) -> np.ndarray:
    return build_runtime_feature(
        confidence=confidence,
        apce=apce,
        psr=psr,
        pred_bbox=pred_bbox,
        image_size=image_size,
        state=state,
    )


def build_sequence_features(
    rows: list[dict],
    image_size: tuple[int, int],
    *,
    cfg: CSCFeatureConfig | None = None,
    use_iou: bool = True,
) -> np.ndarray:
    """Build a (T, F) feature matrix from FrameLabel JSONL rows.

    ``use_iou=False`` zeroes the IoU column to simulate runtime
    (no-GT) operation — useful for sanity checks.
    """
    cfg = cfg or CSCFeatureConfig()
    state = _State()
    out = np.zeros((len(rows), FEATURE_DIM), dtype=np.float32)
    for t, r in enumerate(rows):
        pred = tuple(r["pred_bbox"]) if r.get("pred_bbox") else None
        feat = build_offline_feature(
            iou=r.get("iou") if use_iou else None,
            confidence=r.get("confidence"),
            apce=r.get("apce"),
            psr=r.get("psr"),
            pred_bbox=pred,
            image_size=image_size,
            state=state,
        )
        out[t] = feat
    np.clip(out, -cfg.clip_value, cfg.clip_value, out=out)
    return out
