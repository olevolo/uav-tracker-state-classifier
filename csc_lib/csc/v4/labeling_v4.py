"""CSC-v4 redesigned FC / LA weak labels with training-time subtypes (module A4).

WHY THIS EXISTS
---------------
V3's label rule was ``FALSE_CONFIRMED = LOST && HIGH_CONFIDENCE``.  On the live
SGLATrack telemetry the ``confidence`` channel is *degenerate* — it sits near
zero almost everywhere (a UAV123 sample row shows ``confidence=0.0149`` on a
correctly-tracked frame), so "HIGH_CONFIDENCE" is meaningless and FC labels were
either empty or noise.  See MEMORY: ``project_csc_ortrack_fcr_zero_calibration_bug``
and ``project_la_gate_is_feasible`` (score-map structure + appearance separate
true-loss from false-LA at AUROC ~0.85).

V4 therefore re-derives the *false_confirmed* signal from the response-map
**targetness/peakiness** plus **appearance identity**, never from confidence:

    FC := the tracker is wrong (iou < tau_fail) AND its score map is sharply
          peaked (low response_entropy, high sm_local_top2_ratio /
          sm_local_peak_margin) AND identity is off (sim_to_target low OR
          sim_to_distractor high) AND it is NOT a pure occlusion.

This module is a WEAK LABELER (offline, GT available).  It is *not* the runtime
predictor — the V4 model (A6) learns these labels from telemetry only.

OUTPUT (per frame)
------------------
``{'derived': DerivedStateV4, 'fc_subtype': FCSubtype, 'la_subtype': LASubtype}``

* ``derived``    collapses to the runtime 4-class space (CC / CU / LA / FC).
* ``fc_subtype`` is NONE unless ``derived == FC``  (DISTRACTOR vs BACKGROUND).
* ``la_subtype`` is NONE unless ``derived == LA``  (FALSE/SMOOTH/ABRUPT/OCCLUDED/CANDIDATE).

PRIORITY: **FC outranks LA.**  A wrong-but-peaky-and-mis-identified frame is FC
even if it would also qualify as "lost".

This file is additive (V3 labeling/* untouched) and imports shared enums from
``csc_lib.csc.v4.v4types``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from csc_lib.csc.v4.v4types import DerivedStateV4, FCSubtype, LASubtype

# Reuse the audited V3 bbox kernels (pure-numpy; not a V3 *edit*).
from csc_lib.eval.custom_metrics.bbox import center_xy, iou_xywh

BBox = tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Thresholds — every knob the A4 spec asks for, documented, with sane defaults.
# ---------------------------------------------------------------------------
@dataclass
class LabelingThresholdsV4:
    """All V4 labeling knobs.  Defaults are deliberately conservative.

    Targetness / peakiness gates are tuned to SGLATrack's normalized-ish
    response stats (a UAV123 sample row: response_entropy ~3.9 on a hard frame,
    sm_local_top2_ratio ~0.33, sm_local_peak_margin ~0.21).  They are exposed so
    A11 (v4_diagnose) can recalibrate per-tracker without editing this file.
    """

    # --- localization geometry (IoU) ---------------------------------------
    tau_confirmed_iou: float = 0.50   # >= => CC
    tau_uncertain_iou: float = 0.20   # [uncertain, confirmed) => CU
    tau_fail_iou: float = 0.20        # iou < this => candidate for LA/FC ("wrong")
    tau_lafalse_iou: float = 0.50     # LA_FALSE only when iou actually >= this
    lost_min_consecutive: int = 3     # consecutive low-IoU frames before "lost"

    # --- FC targetness / peakiness (response-map structure) -----------------
    # FC requires a *sharp, confident-looking* peak. Lower entropy = peakier.
    tau_fc_entropy_max: float = 3.6          # response_entropy must be <= this
    tau_fc_top2_ratio_max: float = 0.55      # sm_local_top2_ratio <= this (1st >> 2nd)
    tau_fc_peak_margin_min: float = 0.12     # sm_local_peak_margin >= this
    tau_fc_apce_min: float = 20.0            # APCE >= this (peak sharpness, optional)
    fc_min_targetness_votes: int = 2         # how many of the 4 targetness gates must pass

    # --- FC identity (appearance) -------------------------------------------
    # FC also requires identity to be OFF: either we no longer look like the
    # target, OR we look like a known distractor.
    tau_fc_sim_target_low: float = 0.60      # sim_to_target < this => identity lost
    tau_fc_sim_distractor_high: float = 0.55  # sim_to_distractor >= this => on a distractor
    # Fallback when no prototype/candidate sims are supplied: use appearance drift.
    tau_fc_appearance_drift: float = 0.30    # appearance_drift >= this => identity off

    # --- FC subtype split (DISTRACTOR vs BACKGROUND) ------------------------
    tau_fcd_sim_distractor: float = 0.50     # sim_to_distractor >= this => FC_D
    # Secondary cue when distractor sim is unavailable: a strong secondary peak
    # nearby implies a competing object (distractor) rather than blank bg.
    tau_fcd_top2_ratio: float = 0.45         # sm_local_top2_ratio >= this => FC_D-ish
    tau_fcd_n_secondary: float = 1.0         # sm_n_secondary >= this => competitor present

    # --- LA occlusion -------------------------------------------------------
    # (occ / oov flags drive LA_OCCLUDED directly; no numeric knob needed)

    # --- LA motion smoothness (uses a centred GT window) --------------------
    # "smooth" = current step velocity close to the recent average velocity.
    # "abrupt" = a large deviation (stop / turn / jump).
    tau_la_smooth_accel_ratio: float = 0.60  # |v_t - v_prev| / (v_ref+eps) <= this => SMOOTH
    tau_la_abrupt_accel_ratio: float = 1.50  # ... >= this => ABRUPT
    tau_la_static_vel_norm: float = 0.004    # normalized speed below this => "stopped" (ABRUPT)
    motion_window: int = 5                   # half-not — total frames sampled around t for v_ref

    # --- LA candidate availability ------------------------------------------
    # A relocatable candidate exists (secondary peak or distractor-store hit).
    tau_la_cand_sim_min: float = 0.45        # a candidate that looks like the target
    tau_la_cand_n_secondary: float = 1.0     # >= this secondary peaks => candidate present
    tau_la_cand_top2_ratio: float = 0.40     # strong-enough secondary peak

    # --- misc ---------------------------------------------------------------
    eps: float = 1e-6


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _get(d: Optional[dict], *keys: str, default: Optional[float] = None) -> Optional[float]:
    """First present, finite value among ``keys`` in dict ``d`` (else ``default``)."""
    if not d:
        return default
    for k in keys:
        if k in d and d[k] is not None:
            try:
                v = float(d[k])
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                return v
    return default


def _sim_to_target(cand_sims: Optional[dict]) -> Optional[float]:
    """Identity-to-target similarity.

    Accepts either the A4-spec name ``sim_to_target`` or the PrototypeMemory
    (A2) names ``sim_to_init`` / ``sim_to_recent`` (we take the max — looking
    like *either* the anchor or a recent CC counts as "still on target").
    """
    if not cand_sims:
        return None
    direct = _get(cand_sims, "sim_to_target")
    if direct is not None:
        return direct
    init = _get(cand_sims, "sim_to_init")
    recent = _get(cand_sims, "sim_to_recent")
    vals = [v for v in (init, recent) if v is not None]
    return max(vals) if vals else None


def _sim_to_distractor(cand_sims: Optional[dict]) -> Optional[float]:
    return _get(cand_sims, "sim_to_distractor", "distractor_sim", "sim_distractor")


# ---------------------------------------------------------------------------
# Per-frame label
# ---------------------------------------------------------------------------
def label_frame_v4(
    iou: Optional[float],
    occ: bool,
    oov: bool,
    tel: dict,
    cand_sims: Optional[dict] = None,
    motion: Optional[dict] = None,
    *,
    consecutive_low_iou: int = 0,
    thresholds: Optional[LabelingThresholdsV4] = None,
) -> dict:
    """Label one frame in the V4 scheme.

    Parameters
    ----------
    iou:
        IoU(pred, gt) for this frame, or ``None`` if GT is absent. ``None`` is
        treated as "geometry unknown" -> defaults to CC (benign) unless an
        occlusion flag is set.
    occ, oov:
        Dataset full-occlusion / out-of-view flags for this frame.
    tel:
        V3 telemetry row (response_entropy, sm_local_top2_ratio,
        sm_local_peak_margin, sm_n_secondary, apce, appearance_drift,
        last_cosine_sim, initial_template_sim, ...). Missing keys are tolerated.
    cand_sims:
        Optional prototype/candidate identity sims. Either ``{'sim_to_target',
        'sim_to_distractor'}`` (A4 spec) or the A2 ``PrototypeMemory.sims``
        output ``{'sim_to_init','sim_to_recent','sim_to_distractor'}``. If
        ``None``, identity falls back to telemetry (appearance_drift /
        last_cosine_sim).
    motion:
        Optional precomputed motion smoothness for this frame::

            {'v_t': float, 'v_ref': float, 'v_norm': float}

        where ``v_t`` = current centre speed (px), ``v_ref`` = recent average
        centre speed over a window, ``v_norm`` = ``v_t`` normalized by image
        diagonal. ``build_v4_labels`` fills this in; callers of the single-frame
        API may pass ``None`` (LA then falls back to OCCLUDED/CANDIDATE/abrupt
        heuristics without the smooth/abrupt split).

    Returns
    -------
    dict with keys ``derived`` (DerivedStateV4), ``fc_subtype`` (FCSubtype),
    ``la_subtype`` (LASubtype).
    """
    th = thresholds or LabelingThresholdsV4()

    occluded = bool(occ) or bool(oov)

    # ---- geometry ---------------------------------------------------------
    if iou is None:
        # No GT this frame. If flagged occluded -> LA_OCCLUDED, else benign CC.
        if occluded:
            return {
                "derived": DerivedStateV4.LA,
                "fc_subtype": FCSubtype.NONE,
                "la_subtype": LASubtype.OCCLUDED,
            }
        return {
            "derived": DerivedStateV4.CC,
            "fc_subtype": FCSubtype.NONE,
            "la_subtype": LASubtype.NONE,
        }

    iou = float(iou)
    tracker_wrong = iou < th.tau_fail_iou
    # Sustained loss requires N consecutive low-IoU frames; a single-frame dip is
    # transient, not a hard loss. Consumed below to gate persistent-loss subtypes.
    lost_now = tracker_wrong and consecutive_low_iou >= th.lost_min_consecutive

    # ---- correct localization (CC / CU) -----------------------------------
    if iou >= th.tau_confirmed_iou:
        # rationale (P1): high-IoU frame whose telemetry RESEMBLES a loss (diffuse
        # / multi-peak score map) is the canonical "CSC would over-fire" case. GT
        # is correct so derived stays CC, but the la_subtype channel carries FALSE
        # so the la_subtype head can learn the DO-NOT-ACT safety case (otherwise
        # LA_FALSE is unreachable — the LA branch only runs when iou < tau_fail).
        return {
            "derived": DerivedStateV4.CC,
            "fc_subtype": FCSubtype.NONE,
            "la_subtype": _cc_lafalse_subtype(iou, tel, th),
        }
    if iou >= th.tau_uncertain_iou:
        # Partial overlap -> uncertain (not a failure, not yet lost).
        return {
            "derived": DerivedStateV4.CU,
            "fc_subtype": FCSubtype.NONE,
            "la_subtype": LASubtype.NONE,
        }

    # ====================== FAILURE REGION (iou < tau_fail) =================
    # Decide FC vs LA. FC has strict priority.

    # ---- FC targetness / peakiness votes ----------------------------------
    entropy = _get(tel, "response_entropy")
    top2_ratio = _get(tel, "sm_local_top2_ratio", "local_top2_ratio")
    peak_margin = _get(tel, "sm_local_peak_margin", "local_peak_margin")
    apce = _get(tel, "apce")

    votes = 0
    if entropy is not None and entropy <= th.tau_fc_entropy_max:
        votes += 1
    if top2_ratio is not None and top2_ratio <= th.tau_fc_top2_ratio_max:
        votes += 1
    if peak_margin is not None and peak_margin >= th.tau_fc_peak_margin_min:
        votes += 1
    if apce is not None and apce >= th.tau_fc_apce_min:
        votes += 1
    peaky = votes >= th.fc_min_targetness_votes

    # ---- FC identity (off-target?) ----------------------------------------
    sim_tgt = _sim_to_target(cand_sims)
    sim_dis = _sim_to_distractor(cand_sims)
    drift = _get(tel, "appearance_drift")
    last_cos = _get(tel, "last_cosine_sim", "initial_template_sim")

    identity_off = False
    if sim_tgt is not None and sim_tgt < th.tau_fc_sim_target_low:
        identity_off = True
    if sim_dis is not None and sim_dis >= th.tau_fc_sim_distractor_high:
        identity_off = True
    if not identity_off and sim_tgt is None and sim_dis is None:
        # No prototype sims supplied -> fall back to telemetry appearance.
        if drift is not None and drift >= th.tau_fc_appearance_drift:
            identity_off = True
        elif last_cos is not None and last_cos < th.tau_fc_sim_target_low:
            identity_off = True

    # FC = wrong AND peaky AND identity-off AND not pure occlusion.
    is_fc = tracker_wrong and peaky and identity_off and not occluded

    if is_fc:
        fc_subtype = _fc_subtype(sim_dis, top2_ratio, tel, th)
        return {
            "derived": DerivedStateV4.FC,
            "fc_subtype": fc_subtype,
            "la_subtype": LASubtype.NONE,
        }

    # ====================== LA region (wrong but not FC) ===================
    la_subtype = _la_subtype(
        iou=iou,
        occluded=occluded,
        cand_sims=cand_sims,
        sim_dis=sim_dis,
        top2_ratio=top2_ratio,
        tel=tel,
        motion=motion,
        lost_now=lost_now,
        th=th,
    )
    # P2: a hard loss (true-loss SMOOTH/ABRUPT) needs sustained low IoU. If we are
    # wrong but not yet lost_now, _la_subtype returns NONE for that path -> treat
    # as a transient dip (CU), not a committed loss. Persistent-loss subtypes
    # (OCCLUDED) and relocatable CANDIDATE/over-fire FALSE are unaffected.
    if la_subtype == LASubtype.NONE:
        return {
            "derived": DerivedStateV4.CU,
            "fc_subtype": FCSubtype.NONE,
            "la_subtype": LASubtype.NONE,
        }
    return {
        "derived": DerivedStateV4.LA,
        "fc_subtype": FCSubtype.NONE,
        "la_subtype": la_subtype,
    }


def _fc_subtype(
    sim_dis: Optional[float],
    top2_ratio: Optional[float],
    tel: dict,
    th: LabelingThresholdsV4,
) -> FCSubtype:
    """FC_D (locked onto a similar wrong object) vs FC_B (confident on background)."""
    # Primary cue: appearance similarity to a known distractor.
    if sim_dis is not None:
        return FCSubtype.DISTRACTOR if sim_dis >= th.tau_fcd_sim_distractor else FCSubtype.BACKGROUND
    # Fallback: a strong competing secondary peak implies a real object.
    n_sec = _get(tel, "sm_n_secondary", "n_secondary")
    competitor = (top2_ratio is not None and top2_ratio >= th.tau_fcd_top2_ratio) or (
        n_sec is not None and n_sec >= th.tau_fcd_n_secondary
    )
    return FCSubtype.DISTRACTOR if competitor else FCSubtype.BACKGROUND


def _la_subtype(
    *,
    iou: float,
    occluded: bool,
    cand_sims: Optional[dict],
    sim_dis: Optional[float],
    top2_ratio: Optional[float],
    tel: dict,
    motion: Optional[dict],
    lost_now: bool = True,
    th: LabelingThresholdsV4,
) -> LASubtype:
    """Resolve the LA training subtype.

    Order of precedence:
      1. LA_FALSE     — iou actually >= tau_lafalse_iou (CSC would over-fire).
      2. LA_OCCLUDED  — occlusion / out-of-view flag set.
      3. LA_CANDIDATE — a relocatable candidate exists (sim or secondary peak).
      4. LA_SMOOTH / LA_ABRUPT — from motion smoothness (needs ``motion``).
      5. LA_ABRUPT    — safe default when motion is unavailable.

    ``lost_now`` (P2): the true-loss tail (steps 4-5) is a *committed/sustained*
    loss and is only emitted when ``lost_now`` is True (N consecutive low-IoU
    frames). When False, returns :data:`LASubtype.NONE` so the caller can treat a
    transient dip as CU. Steps 1-3 (over-fire FALSE, genuine OCCLUDED, relocatable
    CANDIDATE) are independent of run length and are unaffected.
    """
    # 1) The tracker is actually fine -> CSC over-fired. DO NOT ACT downstream.
    if iou >= th.tau_lafalse_iou:
        return LASubtype.FALSE

    # 2) Genuine occlusion / out-of-view (a real per-frame flag, not run-length).
    if occluded:
        return LASubtype.OCCLUDED

    # 3) A relocatable candidate exists.
    sim_cand = _get(cand_sims, "sim_to_recent", "sim_to_init", "sim_to_target")
    n_sec = _get(tel, "sm_n_secondary", "n_secondary")
    has_candidate = False
    if sim_cand is not None and sim_cand >= th.tau_la_cand_sim_min:
        has_candidate = True
    elif sim_dis is not None and sim_dis >= th.tau_la_cand_sim_min:
        has_candidate = True
    elif (n_sec is not None and n_sec >= th.tau_la_cand_n_secondary) and (
        top2_ratio is not None and top2_ratio >= th.tau_la_cand_top2_ratio
    ):
        has_candidate = True
    if has_candidate:
        return LASubtype.CANDIDATE

    # --- below here is the committed/sustained true-loss tail (P2) ----------
    # A single-frame dip without N consecutive low-IoU frames is transient, not a
    # hard loss; signal NONE so label_frame_v4 falls back to CU.
    if not lost_now:
        return LASubtype.NONE

    # 4) Motion smoothness split (requires a centred-window motion summary).
    if motion is not None:
        v_t = _get(motion, "v_t")
        v_ref = _get(motion, "v_ref")
        v_norm = _get(motion, "v_norm")
        # "Stopped/turned" -> abrupt: target was moving, now (near) static.
        if (
            v_norm is not None
            and v_norm < th.tau_la_static_vel_norm
            and v_ref is not None
            and v_ref > th.eps
        ):
            return LASubtype.ABRUPT
        if v_t is not None and v_ref is not None:
            accel_ratio = abs(v_t - v_ref) / (v_ref + th.eps)
            if accel_ratio <= th.tau_la_smooth_accel_ratio:
                return LASubtype.SMOOTH
            if accel_ratio >= th.tau_la_abrupt_accel_ratio:
                return LASubtype.ABRUPT
            # middle band -> lean smooth (motion bridge is the cheap safe action).
            return LASubtype.SMOOTH

    # 5) No motion info -> conservative default (bridge is risky without it).
    return LASubtype.ABRUPT


def _cc_lafalse_subtype(
    iou: float,
    tel: dict,
    th: LabelingThresholdsV4,
) -> LASubtype:
    """LA_FALSE training tag for a HIGH-IoU (GT-correct) frame whose telemetry
    RESEMBLES a loss — i.e. the diffuse / multi-peak score map a true loss shows.

    rationale (P1): these are the canonical "CSC over-fired -> DO-NOT-ACT" training
    examples. The frame's geometry is correct (iou >= tau_lafalse_iou) so ``derived``
    stays CC, but the la_subtype channel must carry FALSE so the la_subtype head can
    learn to distinguish a genuine loss from a false alarm. Without this, LA_FALSE is
    unreachable: the LA branch only runs when iou < tau_fail (< tau_lafalse), disjoint
    from the iou>=tau_lafalse guard in ``_la_subtype``.

    Loss-like is detected by inverting the FC peakiness gates: a DIFFUSE map
    (``response_entropy >= tau_fc_entropy_max``) OR a competing secondary peak
    (``sm_local_top2_ratio >= tau_fcd_top2_ratio``). Returns NONE for clean
    high-IoU frames (no over-fire signal).
    """
    if iou < th.tau_lafalse_iou:
        return LASubtype.NONE
    entropy = _get(tel, "response_entropy")
    top2_ratio = _get(tel, "sm_local_top2_ratio", "local_top2_ratio")
    loss_like = (entropy is not None and entropy >= th.tau_fc_entropy_max) or (
        top2_ratio is not None and top2_ratio >= th.tau_fcd_top2_ratio
    )
    return LASubtype.FALSE if loss_like else LASubtype.NONE


# ---------------------------------------------------------------------------
# Sequence-level labeling
# ---------------------------------------------------------------------------
def _centre_window_motion(
    centres: list[Optional[tuple[float, float]]],
    t: int,
    window: int,
    image_diag: float,
    eps: float,
) -> Optional[dict]:
    """Centred-window motion summary for frame ``t`` from GT centres.

    Returns ``{'v_t','v_ref','v_norm'}`` (px, px, normalized) or ``None`` if the
    current step velocity cannot be computed (e.g. missing GT centre).
    """
    cur = centres[t]
    prev = centres[t - 1] if t - 1 >= 0 else None
    if cur is None or prev is None:
        return None
    v_t = math.hypot(cur[0] - prev[0], cur[1] - prev[1])

    # Reference speed = mean step speed over a centred window [t-window, t+window].
    lo = max(1, t - window)
    hi = min(len(centres) - 1, t + window)
    steps: list[float] = []
    for k in range(lo, hi + 1):
        a, b = centres[k], centres[k - 1]
        if a is not None and b is not None:
            steps.append(math.hypot(a[0] - b[0], a[1] - b[1]))
    v_ref = (sum(steps) / len(steps)) if steps else v_t
    v_norm = v_t / image_diag if image_diag > eps else 0.0
    return {"v_t": v_t, "v_ref": v_ref, "v_norm": v_norm}


def build_v4_labels(
    telemetry_rows: list[dict],
    gt_bboxes: list[Optional[BBox]],
    occ: Optional[list[bool]] = None,
    oov: Optional[list[bool]] = None,
    *,
    pred_bboxes: Optional[list[Optional[BBox]]] = None,
    cand_sims_seq: Optional[list[Optional[dict]]] = None,
    image_size: Optional[tuple[float, float]] = None,
    thresholds: Optional[LabelingThresholdsV4] = None,
    # --- threshold passthrough (documented, all of LabelingThresholdsV4) ----
    tau_confirmed_iou: Optional[float] = None,
    tau_uncertain_iou: Optional[float] = None,
    tau_fail_iou: Optional[float] = None,
    tau_lafalse_iou: Optional[float] = None,
    lost_min_consecutive: Optional[int] = None,
    tau_fc_entropy_max: Optional[float] = None,
    tau_fc_top2_ratio_max: Optional[float] = None,
    tau_fc_peak_margin_min: Optional[float] = None,
    tau_fc_apce_min: Optional[float] = None,
    fc_min_targetness_votes: Optional[int] = None,
    tau_fc_sim_target_low: Optional[float] = None,
    tau_fc_sim_distractor_high: Optional[float] = None,
    tau_fc_appearance_drift: Optional[float] = None,
    tau_fcd_sim_distractor: Optional[float] = None,
    tau_fcd_top2_ratio: Optional[float] = None,
    tau_fcd_n_secondary: Optional[float] = None,
    tau_la_smooth_accel_ratio: Optional[float] = None,
    tau_la_abrupt_accel_ratio: Optional[float] = None,
    tau_la_static_vel_norm: Optional[float] = None,
    motion_window: Optional[int] = None,
    tau_la_cand_sim_min: Optional[float] = None,
    tau_la_cand_n_secondary: Optional[float] = None,
    tau_la_cand_top2_ratio: Optional[float] = None,
    eps: Optional[float] = None,
) -> list[dict]:
    """Build V4 labels for a whole sequence.

    Computes per-frame IoU (pred vs GT), a centred-window motion summary, and
    calls :func:`label_frame_v4`.  ``pred_bboxes`` may be omitted, in which case
    each telemetry row is expected to carry a ``bbox`` field (xywh); if neither
    is present IoU is ``None`` for that frame.

    All thresholds are exposed two ways: pass a ``LabelingThresholdsV4`` via
    ``thresholds=...`` and/or override individual fields by keyword (the keyword
    wins). This mirrors the V3 ``label_sequence(..., thresholds=...)`` API while
    making every knob CLI-friendly for A11.

    Returns one dict per frame with keys ``derived`` / ``fc_subtype`` /
    ``la_subtype`` (plus ``iou`` for convenience/auditing).
    """
    n = len(telemetry_rows)
    if len(gt_bboxes) != n:
        raise ValueError(
            f"telemetry_rows ({n}) and gt_bboxes ({len(gt_bboxes)}) length mismatch"
        )

    th = _resolve_thresholds(
        thresholds,
        tau_confirmed_iou=tau_confirmed_iou,
        tau_uncertain_iou=tau_uncertain_iou,
        tau_fail_iou=tau_fail_iou,
        tau_lafalse_iou=tau_lafalse_iou,
        lost_min_consecutive=lost_min_consecutive,
        tau_fc_entropy_max=tau_fc_entropy_max,
        tau_fc_top2_ratio_max=tau_fc_top2_ratio_max,
        tau_fc_peak_margin_min=tau_fc_peak_margin_min,
        tau_fc_apce_min=tau_fc_apce_min,
        fc_min_targetness_votes=fc_min_targetness_votes,
        tau_fc_sim_target_low=tau_fc_sim_target_low,
        tau_fc_sim_distractor_high=tau_fc_sim_distractor_high,
        tau_fc_appearance_drift=tau_fc_appearance_drift,
        tau_fcd_sim_distractor=tau_fcd_sim_distractor,
        tau_fcd_top2_ratio=tau_fcd_top2_ratio,
        tau_fcd_n_secondary=tau_fcd_n_secondary,
        tau_la_smooth_accel_ratio=tau_la_smooth_accel_ratio,
        tau_la_abrupt_accel_ratio=tau_la_abrupt_accel_ratio,
        tau_la_static_vel_norm=tau_la_static_vel_norm,
        motion_window=motion_window,
        tau_la_cand_sim_min=tau_la_cand_sim_min,
        tau_la_cand_n_secondary=tau_la_cand_n_secondary,
        tau_la_cand_top2_ratio=tau_la_cand_top2_ratio,
        eps=eps,
    )

    occ = occ or [False] * n
    oov = oov or [False] * n
    cand_sims_seq = cand_sims_seq or [None] * n

    if image_size is not None:
        image_diag = max(1.0, math.hypot(float(image_size[0]), float(image_size[1])))
    else:
        image_diag = _infer_image_diag(gt_bboxes, telemetry_rows)

    # GT centres for the motion window (None where GT is absent/degenerate).
    centres: list[Optional[tuple[float, float]]] = []
    for gt in gt_bboxes:
        if gt is not None and float(gt[2]) > 0 and float(gt[3]) > 0:
            centres.append(center_xy(gt))
        else:
            centres.append(None)

    labels: list[dict] = []
    consecutive_low_iou = 0
    for t in range(n):
        tel = telemetry_rows[t] or {}
        gt = gt_bboxes[t]

        # predicted bbox: explicit arg first, else from the telemetry row.
        pred: Optional[BBox] = None
        if pred_bboxes is not None and t < len(pred_bboxes):
            pred = pred_bboxes[t]
        if pred is None:
            pred = _row_bbox(tel)

        gt_ok = gt is not None and float(gt[2]) > 0 and float(gt[3]) > 0
        if gt_ok and pred is not None:
            cur_iou: Optional[float] = iou_xywh(pred, gt)
        else:
            cur_iou = None

        if cur_iou is not None and cur_iou < th.tau_fail_iou:
            consecutive_low_iou += 1
        else:
            consecutive_low_iou = 0

        motion = _centre_window_motion(centres, t, th.motion_window, image_diag, th.eps)

        lab = label_frame_v4(
            iou=cur_iou,
            occ=bool(occ[t]),
            oov=bool(oov[t]),
            tel=tel,
            cand_sims=cand_sims_seq[t],
            motion=motion,
            consecutive_low_iou=consecutive_low_iou,
            thresholds=th,
        )
        lab["iou"] = cur_iou
        labels.append(lab)

    return labels


# ---------------------------------------------------------------------------
# Internal: threshold resolution + bbox/diag inference
# ---------------------------------------------------------------------------
def _resolve_thresholds(
    base: Optional[LabelingThresholdsV4], **overrides: Optional[float]
) -> LabelingThresholdsV4:
    th = base or LabelingThresholdsV4()
    # Copy so we never mutate a caller-shared dataclass instance.
    th = LabelingThresholdsV4(**{f: getattr(th, f) for f in th.__dataclass_fields__})
    # Int-typed fields may arrive as floats from CLI/JSON (e.g. motion_window=5.0);
    # coerce so range()/comparisons stay correct (range() rejects floats).
    _int_fields = {"motion_window", "lost_min_consecutive", "fc_min_targetness_votes"}
    for name, val in overrides.items():
        if val is not None:
            if name in _int_fields:
                val = int(val)
            setattr(th, name, val)
    return th


def _row_bbox(row: dict) -> Optional[BBox]:
    """Extract an xywh bbox from a telemetry row under common key spellings."""
    for key in ("bbox", "pred_bbox", "box"):
        b = row.get(key)
        if b is not None:
            try:
                if len(b) >= 4:
                    return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            except (TypeError, ValueError):
                continue
    return None


def _infer_image_diag(
    gt_bboxes: list[Optional[BBox]], telemetry_rows: list[dict]
) -> float:
    """Best-effort image diagonal when ``image_size`` is not supplied.

    Falls back to a coarse estimate from the GT extent so that ``v_norm`` stays
    on a sensible scale (only the *ratio*-based LA split uses v_ref, so this is
    forgiving). Returns at least 1.0.
    """
    max_x = 1.0
    max_y = 1.0
    for gt in gt_bboxes:
        if gt is not None:
            max_x = max(max_x, float(gt[0]) + float(gt[2]))
            max_y = max(max_y, float(gt[1]) + float(gt[3]))
    return max(1.0, math.hypot(max_x, max_y))


__all__ = [
    "LabelingThresholdsV4",
    "label_frame_v4",
    "build_v4_labels",
]


# ---------------------------------------------------------------------------
# Standalone smoke test (no datasets, CPU-only, fast).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    random.seed(0)
    th = LabelingThresholdsV4()

    # ---- 1) Single-frame rule checks -------------------------------------
    # (a) FC: wrong + peaky + identity-off + not occluded.
    fc = label_frame_v4(
        iou=0.05,
        occ=False,
        oov=False,
        tel={
            "response_entropy": 1.2,        # peaky (low entropy)
            "sm_local_top2_ratio": 0.20,    # 1st >> 2nd
            "sm_local_peak_margin": 0.40,
            "apce": 120.0,
        },
        cand_sims={"sim_to_target": 0.20, "sim_to_distractor": 0.80},  # identity off, distractor
    )
    assert fc["derived"] == DerivedStateV4.FC, fc
    assert fc["fc_subtype"] == FCSubtype.DISTRACTOR, fc
    assert fc["la_subtype"] == LASubtype.NONE, fc

    # (b) FC on background: identity off (drift) but no distractor match.
    fc_b = label_frame_v4(
        iou=0.0,
        occ=False,
        oov=False,
        tel={
            "response_entropy": 1.0,
            "sm_local_top2_ratio": 0.10,
            "sm_local_peak_margin": 0.50,
            "appearance_drift": 0.6,        # identity off via telemetry fallback
            "sm_n_secondary": 0.0,          # no competitor -> background
        },
        cand_sims=None,
    )
    assert fc_b["derived"] == DerivedStateV4.FC, fc_b
    assert fc_b["fc_subtype"] == FCSubtype.BACKGROUND, fc_b

    # (c) FC PRIORITY over LA: same peaky+off-identity frame, also "lost"
    #     (long low-IoU run). Must still be FC, not LA.
    fc_prio = label_frame_v4(
        iou=0.0,
        occ=False,
        oov=False,
        tel={
            "response_entropy": 1.0,
            "sm_local_top2_ratio": 0.15,
            "sm_local_peak_margin": 0.45,
        },
        cand_sims={"sim_to_target": 0.1, "sim_to_distractor": 0.9},
        consecutive_low_iou=50,
    )
    assert fc_prio["derived"] == DerivedStateV4.FC, fc_prio

    # (d) LA_OCCLUDED: wrong + occlusion flag (occlusion suppresses FC).
    la_occ = label_frame_v4(
        iou=0.0,
        occ=True,
        oov=False,
        tel={"response_entropy": 1.0, "sm_local_top2_ratio": 0.1, "sm_local_peak_margin": 0.5},
        cand_sims={"sim_to_target": 0.1, "sim_to_distractor": 0.9},
    )
    assert la_occ["derived"] == DerivedStateV4.LA, la_occ
    assert la_occ["fc_subtype"] == FCSubtype.NONE, la_occ
    assert la_occ["la_subtype"] == LASubtype.OCCLUDED, la_occ

    # (e) LA_FALSE: telemetry looks bad but IoU is actually fine -> over-fire.
    #     Reached via the failure path only if iou<tau_fail, so LA_FALSE is
    #     exercised inside _la_subtype when iou>=tau_lafalse but a caller forces
    #     the LA branch. Verify the geometry guard: high IoU -> CC (never FC/LA).
    cc = label_frame_v4(
        iou=0.9,
        occ=False,
        oov=False,
        tel={"response_entropy": 1.0, "sm_local_top2_ratio": 0.1, "sm_local_peak_margin": 0.5},
        cand_sims={"sim_to_target": 0.1, "sim_to_distractor": 0.9},
    )
    assert cc["derived"] == DerivedStateV4.CC, cc

    # (f) LA_SMOOTH vs LA_ABRUPT via motion summary (diffuse map -> not FC).
    #     These are committed true-loss frames, so flag a sustained low-IoU run
    #     (consecutive_low_iou >= lost_min_consecutive) to clear the P2 lost_now gate.
    diffuse = {"response_entropy": 6.0, "sm_local_top2_ratio": 0.95, "sm_local_peak_margin": 0.01}
    la_smooth = label_frame_v4(
        0.0, False, False, diffuse, cand_sims=None,
        motion={"v_t": 10.0, "v_ref": 10.0, "v_norm": 0.02},
        consecutive_low_iou=10,
    )
    assert la_smooth["derived"] == DerivedStateV4.LA and la_smooth["la_subtype"] == LASubtype.SMOOTH, la_smooth
    la_abrupt = label_frame_v4(
        0.0, False, False, diffuse, cand_sims=None,
        motion={"v_t": 80.0, "v_ref": 10.0, "v_norm": 0.16},
        consecutive_low_iou=10,
    )
    assert la_abrupt["la_subtype"] == LASubtype.ABRUPT, la_abrupt

    # (g) P2 transient gate: same diffuse-map wrong frame but only a single-frame
    #     dip (consecutive_low_iou < lost_min_consecutive) -> transient CU, not a
    #     committed loss.
    cu_transient = label_frame_v4(
        0.0, False, False, diffuse, cand_sims=None,
        motion={"v_t": 10.0, "v_ref": 10.0, "v_norm": 0.02},
        consecutive_low_iou=1,
    )
    assert cu_transient["derived"] == DerivedStateV4.CU, cu_transient
    assert cu_transient["la_subtype"] == LASubtype.NONE, cu_transient

    # ---- 2) Sequence-level smoke (synthetic list-of-dicts) ----------------
    N = 40
    img = (640.0, 360.0)
    gts: list[Optional[BBox]] = []
    tels: list[dict] = []
    occ_seq: list[bool] = []
    oov_seq: list[bool] = []
    cands: list[Optional[dict]] = []

    for t in range(N):
        # GT marches smoothly across the frame.
        gx = 50.0 + 5.0 * t
        gt = (gx, 150.0, 40.0, 40.0)
        gts.append(gt)
        occ_seq.append(False)
        oov_seq.append(False)

        if t < 10:
            # Tracking well: pred ~= GT, diffuse-ish map, on-identity.
            pred = (gx + 2.0, 151.0, 40.0, 40.0)
            tel = {
                "bbox": list(pred),
                "response_entropy": 4.5,
                "sm_local_top2_ratio": 0.6,
                "sm_local_peak_margin": 0.05,
                "apce": 40.0,
                "appearance_drift": 0.05,
            }
            cs = {"sim_to_init": 0.9, "sim_to_recent": 0.9, "sim_to_distractor": 0.2}
        elif t < 22:
            # FALSE_CONFIRMED: jumped onto a distractor, peaky + off-identity.
            pred = (gx + 200.0, 151.0, 40.0, 40.0)  # far from GT -> low IoU
            tel = {
                "bbox": list(pred),
                "response_entropy": 1.0 + 0.05 * random.random(),
                "sm_local_top2_ratio": 0.18,
                "sm_local_peak_margin": 0.42,
                "apce": 110.0,
                "appearance_drift": 0.5,
            }
            cs = {"sim_to_init": 0.2, "sim_to_recent": 0.25, "sim_to_distractor": 0.82}
        else:
            # LOST_AWARE (occluded): wrong + occlusion flag.
            pred = (gx + 200.0, 151.0, 40.0, 40.0)
            tel = {
                "bbox": list(pred),
                "response_entropy": 6.2,
                "sm_local_top2_ratio": 0.97,
                "sm_local_peak_margin": 0.005,
            }
            cs = None
            occ_seq[-1] = True
        tels.append(tel)
        cands.append(cs)

    labels = build_v4_labels(
        tels, gts, occ_seq, oov_seq, cand_sims_seq=cands, image_size=img
    )

    # Every label has the 3 required keys.
    assert len(labels) == N
    for lab in labels:
        assert "derived" in lab and "fc_subtype" in lab and "la_subtype" in lab, lab
        assert isinstance(lab["derived"], DerivedStateV4)
        assert isinstance(lab["fc_subtype"], FCSubtype)
        assert isinstance(lab["la_subtype"], LASubtype)

    derived_seq = [int(l["derived"]) for l in labels]
    # Early good frames -> CC.
    assert all(d == int(DerivedStateV4.CC) for d in derived_seq[:10]), derived_seq[:10]
    # Distractor window -> at least some FC, and those carry an FC subtype.
    fc_idx = [i for i, l in enumerate(labels) if l["derived"] == DerivedStateV4.FC]
    assert fc_idx, "expected FC frames in the distractor window"
    for i in fc_idx:
        assert labels[i]["fc_subtype"] != FCSubtype.NONE
        assert labels[i]["la_subtype"] == LASubtype.NONE
    # Occluded tail -> LA (occlusion suppresses FC even though map is wrong).
    assert all(labels[i]["derived"] == DerivedStateV4.LA for i in range(22, N)), derived_seq[22:]
    assert all(labels[i]["la_subtype"] == LASubtype.OCCLUDED for i in range(22, N))

    # FC-over-LA priority holds globally: no frame is simultaneously tagged FC
    # in derived but given an LA subtype (or vice-versa).
    for lab in labels:
        if lab["derived"] == DerivedStateV4.FC:
            assert lab["la_subtype"] == LASubtype.NONE
        if lab["derived"] == DerivedStateV4.LA:
            assert lab["fc_subtype"] == FCSubtype.NONE

    # ---- 3) LA_FALSE coverage --------------------------------------------
    # (3a) Direct unit check of _la_subtype: iou>=tau_lafalse -> LA_FALSE.
    assert _la_subtype(
        iou=0.7, occluded=False, cand_sims=None, sim_dis=None,
        top2_ratio=None, tel={}, motion=None, th=th,
    ) == LASubtype.FALSE

    # (3b) P1: LA_FALSE now emitted through the OUTER label_frame_v4 on a HIGH-IoU
    #      frame whose telemetry RESEMBLES a loss (diffuse map). derived stays CC.
    la_false_cc = label_frame_v4(
        iou=0.85,                       # GT-correct -> derived CC
        occ=False,
        oov=False,
        tel={"response_entropy": 5.5, "sm_local_top2_ratio": 0.9},  # loss-like map
        cand_sims=None,
    )
    assert la_false_cc["derived"] == DerivedStateV4.CC, la_false_cc
    assert la_false_cc["la_subtype"] == LASubtype.FALSE, la_false_cc
    assert la_false_cc["fc_subtype"] == FCSubtype.NONE, la_false_cc
    # And a clean high-IoU frame (peaky map) is plain CC with no over-fire tag.
    cc_clean = label_frame_v4(
        iou=0.85, occ=False, oov=False,
        tel={"response_entropy": 1.0, "sm_local_top2_ratio": 0.1}, cand_sims=None,
    )
    assert cc_clean["la_subtype"] == LASubtype.NONE, cc_clean
    # LA_FALSE also appears somewhere in the sequence labels (early CC frames have
    # diffuse-ish maps), confirming the channel is populated end-to-end.
    assert any(l["la_subtype"] == LASubtype.FALSE for l in labels), \
        "expected at least one LA_FALSE tag in the sequence"

    # ---- 4) P3: int-typed thresholds passed as floats must not crash range() --
    labels_float_win = build_v4_labels(
        tels, gts, occ_seq, oov_seq, cand_sims_seq=cands, image_size=img,
        motion_window=5.0, lost_min_consecutive=3.0, fc_min_targetness_votes=2.0,
    )
    assert len(labels_float_win) == N

    # ---- 5) P4: newly exposed threshold kwargs are accepted and forwarded -----
    labels_extra_kwargs = build_v4_labels(
        tels, gts, occ_seq, oov_seq, cand_sims_seq=cands, image_size=img,
        tau_fcd_top2_ratio=0.45, tau_fcd_n_secondary=1.0, tau_la_cand_sim_min=0.45,
        tau_la_cand_n_secondary=1.0, tau_la_cand_top2_ratio=0.40, eps=1e-6,
    )
    assert len(labels_extra_kwargs) == N

    from collections import Counter
    dist = Counter(int(d) for d in derived_seq)
    n_la_false = sum(1 for l in labels if l["la_subtype"] == LASubtype.FALSE)
    print("V4 labeling smoke OK")
    print("  derived distribution (CC/CU/LA/FC):",
          {DerivedStateV4(k).name: v for k, v in sorted(dist.items())})
    print(f"  FC frames: {len(fc_idx)}  (subtypes: "
          f"{Counter(labels[i]['fc_subtype'].name for i in fc_idx)})")
    print(f"  LA_FALSE tags in sequence (high-IoU over-fire): {n_la_false}")
    print("  asserts passed: FC rule, FC_D/FC_B split, FC>LA priority, "
          "LA_OCCLUDED, LA_SMOOTH/ABRUPT, P2 transient-CU gate, "
          "P1 LA_FALSE@high-IoU (outer), P3 float motion_window, "
          "P4 extra kwargs, 3-key contract")
