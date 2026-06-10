"""Sequence-level scene-state labeler — composite version."""
from __future__ import annotations

import math
from typing import Iterable, Optional

from csc_lib.csc.labeling.label_schema import (
    DerivedState,
    FrameLabel,
    LocalizationState,
    ConfidenceState,
    LabelSource,
    derive_state,
)
from csc_lib.csc.labeling.risk_labeler import build_future_risk_labels
from csc_lib.csc.labeling.weak_labeler import LabelingThresholds, label_frame
from csc_lib.eval.custom_metrics.bbox import iou_xywh, center_xy, normalized_center_error


def _safe_log_ratio(num: float, den: float) -> float:
    if num <= 0 or den <= 0:
        return 0.0
    return abs(math.log(num / den))


def label_sequence(
    *,
    dataset: str,
    split: str,
    sequence: str,
    gt_bboxes: list[Optional[tuple[float, float, float, float]]],
    pred_bboxes: list[Optional[tuple[float, float, float, float]]],
    image_size: tuple[int, int],
    full_occlusion: Optional[list[bool]] = None,
    out_of_view: Optional[list[bool]] = None,
    absent: Optional[list[bool]] = None,
    confidences: Optional[list[Optional[float]]] = None,
    apces: Optional[list[Optional[float]]] = None,
    psrs: Optional[list[Optional[float]]] = None,
    dataset_attributes: Optional[dict] = None,
    thresholds: Optional[LabelingThresholds] = None,
    forecast_horizon: int = 10,
) -> list[FrameLabel]:
    n = len(gt_bboxes)
    if len(pred_bboxes) != n:
        raise ValueError(
            f"pred_bboxes length {len(pred_bboxes)} != gt {n} for {dataset}/{sequence}"
        )

    th = thresholds or LabelingThresholds()
    img_w, img_h = image_size
    image_diag = max(1.0, math.hypot(float(img_w), float(img_h)))

    full_occlusion = full_occlusion or [False] * n
    out_of_view = out_of_view or [False] * n
    absent = absent or [False] * n
    confidences = confidences or [None] * n
    apces = apces or [None] * n
    psrs = psrs or [None] * n
    dataset_attributes = dataset_attributes or {}

    labels: list[FrameLabel] = []
    consecutive_low_iou = 0
    prev_centre: Optional[tuple[float, float]] = None
    prev_centre_pp: Optional[tuple[float, float]] = None
    prev_area: Optional[float] = None

    for t in range(n):
        gt = gt_bboxes[t]
        pred = pred_bboxes[t]
        conf = confidences[t]
        full_occ = bool(full_occlusion[t])
        oov = bool(out_of_view[t])
        absent_t = bool(absent[t]) or gt is None or (gt is not None and (gt[2] <= 0 or gt[3] <= 0))

        if gt is not None and pred is not None and not absent_t:
            cur_iou = iou_xywh(pred, gt)
        else:
            cur_iou = None

        if gt is not None and pred is not None and not absent_t:
            ce = math.hypot(*[a - b for a, b in zip(center_xy(pred), center_xy(gt))])
            n_ce = normalized_center_error(pred, gt, image_diag)
        else:
            ce = None
            n_ce = None

        if gt is not None and not absent_t:
            cur_centre = center_xy(gt)
            cur_area = max(0.0, gt[2]) * max(0.0, gt[3])
        else:
            cur_centre = None
            cur_area = None

        if cur_centre is not None and prev_centre is not None:
            dx = cur_centre[0] - prev_centre[0]
            dy = cur_centre[1] - prev_centre[1]
            velocity = math.hypot(dx, dy)
            fast_motion_norm = velocity / image_diag
        else:
            velocity = None
            fast_motion_norm = None

        if velocity is not None and prev_centre_pp is not None and prev_centre is not None:
            prev_dx = prev_centre[0] - prev_centre_pp[0]
            prev_dy = prev_centre[1] - prev_centre_pp[1]
            prev_velocity = math.hypot(prev_dx, prev_dy)
            acceleration = velocity - prev_velocity
        else:
            acceleration = None

        if cur_area is not None and prev_area is not None:
            scale_change_log = _safe_log_ratio(cur_area, prev_area)
            area_ratio = cur_area / max(1e-6, prev_area)
        else:
            scale_change_log = None
            area_ratio = None

        if cur_iou is not None and cur_iou < th.tau_lost_iou:
            consecutive_low_iou += 1
        else:
            consecutive_low_iou = 0

        loc, conf_state, derived, aux, source, noisy = label_frame(
            iou=cur_iou,
            confidence=conf,
            apce=apces[t],
            psr=psrs[t],
            full_occlusion=full_occ,
            out_of_view=oov,
            absent=absent_t,
            consecutive_low_iou=consecutive_low_iou,
            fast_motion_norm=fast_motion_norm,
            scale_change_log=scale_change_log,
            thresholds=th,
        )

        labels.append(
            FrameLabel(
                dataset=dataset,
                split=split,
                sequence=sequence,
                frame_idx=t,
                gt_bbox=tuple(gt) if gt is not None else None,
                pred_bbox=tuple(pred) if pred is not None else None,
                iou=cur_iou,
                center_error=ce,
                normalized_center_error=n_ce,
                area_ratio=area_ratio,
                velocity=velocity,
                acceleration=acceleration,
                visible_ratio=None,
                absent=absent_t,
                dataset_attributes=dataset_attributes,
                confidence=conf,
                apce=apces[t],
                psr=psrs[t],
                localization_state=int(loc),
                confidence_state=int(conf_state),
                derived_state=int(derived),
                false_confirmed_flag=(derived == DerivedState.FALSE_CONFIRMED),
                aux=aux,
                label_source=int(source),
                label_noisy=noisy,
            )
        )

        prev_centre_pp = prev_centre
        prev_centre = cur_centre
        prev_area = cur_area

    # Attach proactive forecast labels (V3) — strict no-leakage, derived-state-only.
    derived_seq = [int(lab.derived_state) for lab in labels]
    risk_labels = build_future_risk_labels(derived_seq, horizon=forecast_horizon)
    for lab, rl in zip(labels, risk_labels):
        lab.failure_next_10 = int(rl["failure_next_10"])
        lab.false_confirmed_next_10 = int(rl["false_confirmed_next_10"])
        lab.lost_aware_next_10 = int(rl["lost_aware_next_10"])
        lab.ignore_forecast = int(rl["ignore_forecast"])

    return labels


def summarize_label_distribution(labels: Iterable[FrameLabel]) -> dict:
    """Per-axis and per-derived counts."""
    loc_counts = {s.name: 0 for s in LocalizationState}
    conf_counts = {s.name: 0 for s in ConfidenceState}
    derived_counts = {s.name: 0 for s in DerivedState}
    source_counts = {s.name: 0 for s in LabelSource}
    aux_counts: dict[str, int] = {}
    n = 0
    n_noisy = 0
    n_fc = 0
    for lab in labels:
        n += 1
        loc_counts[LocalizationState(lab.localization_state).name] += 1
        conf_counts[ConfidenceState(lab.confidence_state).name] += 1
        derived_counts[DerivedState(lab.derived_state).name] += 1
        source_counts[LabelSource(lab.label_source).name] += 1
        if lab.label_noisy:
            n_noisy += 1
        if lab.false_confirmed_flag:
            n_fc += 1
        for k, v in lab.aux.items():
            if v:
                aux_counts[k] = aux_counts.get(k, 0) + 1
    return {
        "n_frames": n,
        "n_noisy": n_noisy,
        "n_false_confirmed": n_fc,
        "localization_counts": loc_counts,
        "confidence_counts": conf_counts,
        "derived_counts": derived_counts,
        "source_counts": source_counts,
        "aux_counts": aux_counts,
        # Backward-compat alias for older readers
        "state_counts": derived_counts,
    }


# Back-compat: older code imported ``state_counts`` directly.
__all__ = [
    "FrameLabel",
    "LabelingThresholds",
    "label_frame",
    "label_sequence",
    "summarize_label_distribution",
]
