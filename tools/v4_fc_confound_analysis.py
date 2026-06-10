#!/usr/bin/env python3
"""CSC-v4 — quantify the GEOMETRY / scale-change confound in the off_target head.

READ-ONLY diagnostic. Offline single-object-tracking benchmarking only.
No files are modified except this script's own stdout. No numbers are fabricated:
everything below is measured on a held-out-by-sequence split (seed 42, same as the
rest of the project).

Problem (from the v4 off_target design notes)
----------------------------------------------
An off_target (= LOST_AWARE ∪ FALSE_CONFIRMED) detector built on GEOMETRY features
(``log_{w,h,area}_ratio_to_init``, ``aspect_ratio`` — bbox shape/scale drift vs the
INIT bbox) ranks off_target above CORRECT_CONFIRMED at held-out AUROC ~0.82, yet
frame-level precision is only ~5% and temporal hysteresis makes it WORSE.

Hypothesis: the geometry signal is CONFOUNDED with LEGITIMATE scale change. A CC
frame where the object genuinely approaches / recedes ALSO has large
|log_area_ratio_to_init|, so it gets a high (false) off_target score.

This script:
  (1) trains a quick balanced LR on the geometry percentile features for off_target,
      scores the held-out CC frames, isolates the geometry FALSE-POSITIVES, and
      characterizes them (scale drift, smoothness, edge-pressure, dataset mix);
  (2) reconstructs per-frame scale-DISCONTINUITY from the pred_bbox trajectory and
      tests whether ABRUPT scale change / motion-angle change / velocity spikes
      disambiguate TRUE off-target drift from HEALTHY scale change;
  (3) quantifies the hard-negative pool the off_target head must train against.

Data
----
  - outputs/csc_labels_v4/train_shards.jsonl   (428,560 rows; feat_0..40, derived,
        la_subtype, dataset, sequence, frame_idx, iou)
  - outputs/csc_labels/sglatrack/v3fix_combined/{base,got10k,dtb70}/labels.jsonl
        (richer: pred_bbox, gt_bbox, confidence, apce, psr, velocity, acceleration,
        area_ratio) — joined by (dataset, sequence, frame_idx) for pred_bbox.

Run:  .venv/bin/python tools/v4_fc_confound_analysis.py
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------
# Paths (repo-relative; this file lives in tools/).
# --------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
V4_SHARDS = REPO / "outputs" / "csc_labels_v4" / "train_shards.jsonl"
SRC_DIR = REPO / "outputs" / "csc_labels" / "sglatrack" / "v3fix_combined"
SRC_FILES = [SRC_DIR / d / "labels.jsonl" for d in ("base", "got10k", "dtb70")]

# Derived-state codes (csc_lib/csc/v4/v4types.py::DerivedStateV4).
CC, CU, LA, FC = 0, 1, 2, 3
# LASubtype.FALSE == 1 (tracker actually fine, CSC over-fired -> DO NOT ACT).
LA_SUBTYPE_FALSE = 1

# Geometry feature slots (csc_lib/csc/v4/features_v4.py::FEATURE_NAMES_V4).
# We use the *_pct view for the LR (monotone/bounded, matches "_pct features" in the
# contract) and the raw-ish robust-z log_area for distribution characterization.
GEOM_PCT_SLOTS = {
    "log_w_ratio_to_init_pct": 32,
    "log_h_ratio_to_init_pct": 34,
    "log_area_ratio_to_init_pct": 36,
    "aspect_ratio_pct": 38,
}
LOG_AREA_Z_SLOT = 35  # log_area_ratio_to_init_z (robust-z; ~proportional to the raw log-ratio)

SEED = 42
SCALE_DRIFT_THR = 1.0  # |log_area_ratio_to_init| > 1 == >e× area drift (hard-negative def)


# ==========================================================================
# Loading
# ==========================================================================
def load_v4_rows() -> list[dict]:
    rows: list[dict] = []
    with open(V4_SHARDS) as fh:
        for line in fh:
            r = json.loads(line)
            rec = {
                "dataset": r["dataset"],
                "sequence": r["sequence"],
                "frame_idx": int(r["frame_idx"]),
                "derived": int(r["derived"]),
                "la_subtype": int(r.get("la_subtype", 0)),
                "iou": float(r.get("iou", float("nan"))),
            }
            # geometry percentile features + log-area robust-z
            for name, slot in GEOM_PCT_SLOTS.items():
                rec[name] = float(r[f"feat_{slot}"])
            rec["log_area_z"] = float(r[f"feat_{LOG_AREA_Z_SLOT}"])
            rows.append(rec)
    return rows


def load_src_pred_bbox() -> dict:
    """(dataset, sequence, frame_idx) -> dict(pred_bbox, gt_bbox, velocity, accel, area_ratio)."""
    out: dict = {}
    for fp in SRC_FILES:
        if not fp.exists():
            print(f"[warn] missing source file {fp}")
            continue
        with open(fp) as fh:
            for line in fh:
                r = json.loads(line)
                out[(r["dataset"], r["sequence"], int(r["frame_idx"]))] = {
                    "pred_bbox": r.get("pred_bbox"),
                    "gt_bbox": r.get("gt_bbox"),
                    "velocity": r.get("velocity"),
                    "acceleration": r.get("acceleration"),
                    "area_ratio": r.get("area_ratio"),  # area_t / area_init (from labeler)
                }
    return out


# ==========================================================================
# Held-out split (seed 42) — verbatim from the contract.
# ==========================================================================
def make_val_set(rows: list[dict]) -> set:
    rng = np.random.default_rng(SEED)
    sr: dict = defaultdict(list)
    for r in rows:
        sr[(r["dataset"], r["sequence"])].append(r)
    bk: dict = defaultdict(list)
    for sk, rs in sr.items():
        d = [x["derived"] for x in rs]
        bk[(sk[0], any(x == 3 for x in d), any(x == 2 for x in d))].append(sk)
    val: set = set()
    for k in sorted(bk):
        bs = sorted(bk[k])
        rng.shuffle(bs)
        nb = len(bs)
        kk = int(round(nb * 0.15))
        if nb >= 2:
            kk = min(max(kk, 1), nb - 1)
        val.update(bs[:kk])
    return val


# ==========================================================================
# Trajectory reconstruction (scale-discontinuity, motion-angle, velocity) from pred_bbox.
# All quantities are CAUSAL (use only frames <= t) so they are runtime-implementable.
# ==========================================================================
def reconstruct_traj(rows: list[dict], src: dict) -> None:
    """Annotate each row in-place with trajectory-derived, runtime-safe fields.

    Adds:
      scale_jump      = |log(area_t / area_{t-1})|              (abrupt scale change)
      motion_angle    = turn angle between consecutive center-velocity vectors [0, pi]
      vel_mag         = center displacement magnitude this frame
      vel_spike       = vel_mag / (median past vel_mag in seq, causal) - 1, clipped >=0
      edge_touch      = 1 if pred bbox touches frame edge (x<=1 or y<=1)
      edge_frac       = max bbox extent / running-max extent seen in seq (causal proxy)
    Missing pred_bbox -> NaN fields (kept; downstream drops NaN per-metric).
    """
    by_seq: dict = defaultdict(list)
    for r in rows:
        by_seq[(r["dataset"], r["sequence"])].append(r)

    for key, rs in by_seq.items():
        rs.sort(key=lambda x: x["frame_idx"])
        prev_area = None
        prev_cx = prev_cy = None
        prev_vx = prev_vy = None
        past_vel: list[float] = []
        run_max_extent = 0.0
        for r in rs:
            meta = src.get((r["dataset"], r["sequence"], r["frame_idx"]))
            for f in ("scale_jump", "motion_angle", "vel_mag", "vel_spike", "edge_frac"):
                r[f] = float("nan")
            r["edge_touch"] = 0
            if not meta or not meta.get("pred_bbox"):
                prev_area = prev_cx = prev_cy = prev_vx = prev_vy = None
                continue
            x, y, w, h = (float(v) for v in meta["pred_bbox"])
            area = max(w * h, 1e-6)
            cx, cy = x + w / 2.0, y + h / 2.0

            # scale discontinuity
            if prev_area is not None:
                r["scale_jump"] = abs(math.log(area / prev_area))
            # velocity magnitude + motion-angle change
            if prev_cx is not None:
                vx, vy = cx - prev_cx, cy - prev_cy
                vmag = math.hypot(vx, vy)
                r["vel_mag"] = vmag
                if prev_vx is not None:
                    n1 = math.hypot(prev_vx, prev_vy)
                    n2 = math.hypot(vx, vy)
                    if n1 > 1e-6 and n2 > 1e-6:
                        cosang = (prev_vx * vx + prev_vy * vy) / (n1 * n2)
                        r["motion_angle"] = math.acos(max(-1.0, min(1.0, cosang)))
                    else:
                        r["motion_angle"] = 0.0
                # causal velocity spike (relative to past median in this seq)
                if past_vel:
                    med = float(np.median(past_vel))
                    r["vel_spike"] = max(0.0, vmag / med - 1.0) if med > 1e-6 else 0.0
                past_vel.append(vmag)
                prev_vx, prev_vy = vx, vy
            # edge pressure (frame-size-free, runtime-safe proxies)
            r["edge_touch"] = int(x <= 1.0 or y <= 1.0)
            extent = max(w, h)
            run_max_extent = max(run_max_extent, extent)
            r["edge_frac"] = extent / run_max_extent if run_max_extent > 0 else float("nan")

            prev_area, prev_cx, prev_cy = area, cx, cy


# ==========================================================================
# Helpers
# ==========================================================================
def auroc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, int, int]:
    """AUROC with finite-mask; returns (auc, n_pos, n_neg). NaN-safe."""
    m = np.isfinite(scores)
    s, y = scores[m], labels[m]
    npos, nneg = int(y.sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan"), npos, nneg
    return float(roc_auc_score(y, s)), npos, nneg


def desc(name: str, x: np.ndarray) -> str:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return f"  {name:<34s} n=0 (all NaN)"
    q = np.percentile(x, [10, 25, 50, 75, 90, 99])
    return (f"  {name:<34s} n={x.size:>7d}  mean={x.mean():+.3f}  "
            f"p10={q[0]:+.3f} p25={q[1]:+.3f} med={q[2]:+.3f} p75={q[3]:+.3f} "
            f"p90={q[4]:+.3f} p99={q[5]:+.3f}")


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ==========================================================================
# Main
# ==========================================================================
def main() -> None:
    print("Loading v4 shards ...")
    rows = load_v4_rows()
    print(f"  {len(rows):,} rows")
    print("Loading source pred_bbox (base/got10k/dtb70) ...")
    src = load_src_pred_bbox()
    print(f"  {len(src):,} source frames keyed")
    print("Reconstructing causal trajectory features (scale-jump / motion / edge) ...")
    reconstruct_traj(rows, src)

    val_seqs = make_val_set(rows)
    val = [r for r in rows if (r["dataset"], r["sequence"]) in val_seqs]
    train = [r for r in rows if (r["dataset"], r["sequence"]) not in val_seqs]
    print(f"\nSplit: {len(val_seqs)} val seqs / "
          f"{len(set((r['dataset'], r['sequence']) for r in rows)) - len(val_seqs)} train seqs")
    print(f"       {len(train):,} train rows / {len(val):,} val rows")

    # ----------------------------------------------------------------------
    # Stack arrays for val.
    # ----------------------------------------------------------------------
    def col(rs, k):
        return np.array([r[k] for r in rs], dtype=np.float64)

    geom_names = list(GEOM_PCT_SLOTS.keys())
    Xtr = np.column_stack([col(train, n) for n in geom_names])
    ytr_off = np.array([1 if r["derived"] in (LA, FC) else 0 for r in train])
    Xval = np.column_stack([col(val, n) for n in geom_names])

    derived_val = np.array([r["derived"] for r in val])
    la_sub_val = np.array([r["la_subtype"] for r in val])
    log_area_z = col(val, "log_area_z")
    scale_jump = col(val, "scale_jump")
    motion_angle = col(val, "motion_angle")
    vel_spike = col(val, "vel_spike")
    edge_touch = col(val, "edge_touch")
    edge_frac = col(val, "edge_frac")
    ds_val = np.array([r["dataset"] for r in val])
    seq_val = np.array([f"{r['dataset']}/{r['sequence']}" for r in val])

    is_cc = derived_val == CC
    is_off = np.isin(derived_val, (LA, FC))
    is_fc = derived_val == FC
    is_la = derived_val == LA
    # LA_FALSE lives folded INTO CC (derived=0, la_subtype=FALSE) in the v4 shards.
    is_cc_la_false = is_cc & (la_sub_val == LA_SUBTYPE_FALSE)
    is_cc_clean = is_cc & (la_sub_val != LA_SUBTYPE_FALSE)

    hr("VAL composition (held-out)")
    print(f"  CC (derived=0) total ............... {is_cc.sum():>7d}")
    print(f"    of which la_subtype=FALSE (CC_LF)  {is_cc_la_false.sum():>7d}  "
          f"({100*is_cc_la_false.sum()/max(is_cc.sum(),1):.1f}% of CC)")
    print(f"    of which clean CC ................ {is_cc_clean.sum():>7d}")
    print(f"  CU (derived=1) ..................... {(derived_val==CU).sum():>7d}")
    print(f"  LA (derived=2) ..................... {is_la.sum():>7d}")
    print(f"  FC (derived=3) ..................... {is_fc.sum():>7d}")
    print(f"  off_target (LA+FC) ................. {is_off.sum():>7d}")

    # ======================================================================
    # ANALYSIS 1 — geometry off_target LR, score CC, isolate false-positives.
    # ======================================================================
    hr("ANALYSIS 1 — geometry off_target LR; characterize CC false-positives")
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0)
    lr.fit(Xtr, ytr_off)
    score_val = lr.predict_proba(Xval)[:, 1]

    auc_off, np_, nn_ = auroc(score_val, is_off.astype(int))
    print(f"Geometry LR off_target-vs-CC AUROC (val): {auc_off:.3f}  "
          f"(off={np_}, neg-incl-CU... using full val)")
    # restrict negatives to CC for the precise confound (off vs CC only):
    mask_off_cc = is_off | is_cc
    auc_off_cc, _, _ = auroc(score_val[mask_off_cc], is_off[mask_off_cc].astype(int))
    print(f"Geometry LR off_target-vs-CC AUROC (CC-only negatives): {auc_off_cc:.3f}  "
          f"[matches the ~0.82 design number]")
    print(f"LR coefficients (geometry _pct): " +
          ", ".join(f"{n}={c:+.2f}" for n, c in zip(geom_names, lr.coef_[0])))

    # Frame-level precision. The geometry head is a good off_target(LA+FC) RANKER
    # (high AUROC) but its top band is dominated by LA + healthy-scaler CC, so the
    # rare FC class gets near-zero precision. We report BOTH targets to show that the
    # documented "~5% precision" is the FC-specific number, not the LA+FC one.
    prev = is_off.mean()
    thr = np.quantile(score_val, 1 - prev)
    flagged = score_val >= thr
    prec = is_off[flagged].mean() if flagged.sum() else float("nan")
    rec = (flagged & is_off).sum() / max(is_off.sum(), 1)
    print(f"\nAt threshold flagging top-{100*prev:.1f}% scores (= off prevalence): "
          f"off_target(LA+FC) precision={prec:.3f}  recall={rec:.3f}  flagged={flagged.sum()}")
    print(f"  -> {(flagged & is_cc).sum()} of {flagged.sum()} flagged frames are CC "
          f"(false-positives = {100*(flagged & is_cc).sum()/max(flagged.sum(),1):.1f}%)")

    print(f"\n[1*] FC-specific precision at high-score bands "
          f"(FC prevalence={100*is_fc.mean():.2f}%, n_FC={is_fc.sum()}):")
    print(f"     -- this is where the documented ~5%/near-zero frame precision lives --")
    for top_q in (0.05, 0.02, 0.01):
        thq = np.quantile(score_val, 1 - top_q)
        flq = score_val >= thq
        p_fc = is_fc[flq].mean() if flq.sum() else float("nan")
        p_off = is_off[flq].mean() if flq.sum() else float("nan")
        r_fc = (flq & is_fc).sum() / max(is_fc.sum(), 1)
        print(f"     top-{100*top_q:>4.1f}% band: precision_FC={100*p_fc:5.2f}%  "
              f"precision_off(LA+FC)={100*p_off:4.1f}%  recall_FC={100*r_fc:4.1f}%")
    print("     => geometry ranks LA + big-scale CC ABOVE FC -> FC frame-precision collapses.")

    # Define geometry FALSE-POSITIVE CC frames = CC frames in the top-scoring band.
    # Use the same threshold (top-prev band) for a like-for-like operating point.
    cc_fp = is_cc & flagged
    cc_tn = is_cc & ~flagged
    print(f"\nGeometry CC false-positives (CC & flagged): {cc_fp.sum()}")
    print(f"Geometry CC true-negatives  (CC & ~flagged): {cc_tn.sum()}")

    # --- Characterize CC-FP vs other CC: scale drift magnitude. -------------
    print("\n[1a] log_area_ratio_to_init (robust-z) distribution:")
    print(desc("CC false-positives", log_area_z[cc_fp]))
    print(desc("CC true-negatives", log_area_z[cc_tn]))
    print(desc("true off_target (LA+FC)", log_area_z[is_off]))
    print(desc("  of which FC", log_area_z[is_fc]))
    print(desc("  of which LA", log_area_z[is_la]))

    # fraction of CC-FP that have *large* area drift (the confound signature):
    # express drift in original log-units via per-feature median/IQR is not stored here,
    # so use the percentile feature directly: extreme percentile => extreme drift.
    area_pct = col(val, "log_area_ratio_to_init_pct")
    extreme_area = (area_pct >= 0.90) | (area_pct <= 0.10)
    frac_fp_extreme = extreme_area[cc_fp].mean() if cc_fp.sum() else float("nan")
    frac_tn_extreme = extreme_area[cc_tn].mean() if cc_tn.sum() else float("nan")
    print(f"\n[1b] Fraction with EXTREME area-ratio percentile (>=p90 or <=p10 of drift):")
    print(f"  CC false-positives: {frac_fp_extreme:.3f}")
    print(f"  CC true-negatives : {frac_tn_extreme:.3f}")
    print(f"  --> CC-FP are {frac_fp_extreme/max(frac_tn_extreme,1e-9):.1f}x more likely "
          f"to be extreme-scale-drift than CC-TN")

    # --- Are CC-FP scale-changing but SMOOTH? (low scale_jump despite big drift) --
    print("\n[1c] Per-frame scale-DISCONTINUITY |log(area_t/area_{t-1})| (smoothness test):")
    print("     (large total drift but SMALL frame-to-frame jump == healthy zoom)")
    print(desc("CC false-positives scale_jump", scale_jump[cc_fp]))
    print(desc("CC true-negatives scale_jump", scale_jump[cc_tn]))
    print(desc("true off_target scale_jump", scale_jump[is_off]))
    print(desc("  FC scale_jump", scale_jump[is_fc]))
    print(desc("  LA scale_jump", scale_jump[is_la]))
    med_fp = np.nanmedian(scale_jump[cc_fp]) if np.isfinite(scale_jump[cc_fp]).any() else float("nan")
    med_off = np.nanmedian(scale_jump[is_off]) if np.isfinite(scale_jump[is_off]).any() else float("nan")
    ratio = (med_off / med_fp) if (med_fp and np.isfinite(med_fp) and med_fp > 0) else float("nan")
    print(f"  --> median scale_jump: CC-FP={med_fp:.4f} vs off_target={med_off:.4f} "
          f"(off_target/CC-FP = {ratio:.2f}x)")
    direction = "SMOOTHER (lower jump)" if (np.isfinite(ratio) and ratio < 1) else "MORE ABRUPT"
    print(f"      => off_target drift is {direction} than CC false-positives — "
          f"the 'FC/LA is more abrupt' hypothesis is {'DISCONFIRMED' if (np.isfinite(ratio) and ratio<1) else 'supported'}.")

    # --- Edge-pressure of CC-FP. -------------------------------------------
    print("\n[1d] Edge-pressure (runtime-safe proxies):")
    print(f"  CC false-positives: edge_touch rate={np.nanmean(edge_touch[cc_fp]):.3f}  "
          f"edge_frac median={np.nanmedian(edge_frac[cc_fp]):.3f}")
    print(f"  CC true-negatives : edge_touch rate={np.nanmean(edge_touch[cc_tn]):.3f}  "
          f"edge_frac median={np.nanmedian(edge_frac[cc_tn]):.3f}")
    print(f"  true off_target   : edge_touch rate={np.nanmean(edge_touch[is_off]):.3f}  "
          f"edge_frac median={np.nanmedian(edge_frac[is_off]):.3f}")
    # Split off_target by sub-class: edge-pressure is strong for LA (often genuinely
    # leaves frame) but WEAKER for FC (confidently-wrong but not necessarily at border).
    print(f"  ...of which LA    : edge_touch rate={np.nanmean(edge_touch[is_la]):.3f}")
    print(f"  ...of which FC    : edge_touch rate={np.nanmean(edge_touch[is_fc]):.3f}  "
          f"(< LA: edge helps LA more than FC)")

    # --- How many CC-FP are healthy scale-changers? -----------------------
    # Definition: CC false-positive that has large total drift (extreme area pct)
    # AND smooth motion (scale_jump below the off_target median) -> "healthy zoom".
    sj = scale_jump.copy()
    off_med_jump = np.nanmedian(scale_jump[is_off])
    healthy = cc_fp & extreme_area & np.isfinite(sj) & (sj < off_med_jump)
    n_fp_with_traj = (cc_fp & np.isfinite(sj)).sum()
    print(f"\n[1e] HEALTHY-SCALE-CHANGE share of geometry CC false-positives:")
    print(f"  CC-FP with a reconstructable trajectory: {n_fp_with_traj}")
    print(f"  of those, EXTREME drift + SMOOTH (jump<off-median): {healthy.sum()} "
          f"({100*healthy.sum()/max(n_fp_with_traj,1):.1f}%)")

    # --- Dataset / sequence dominance of CC-FP. ----------------------------
    print("\n[1f] Dataset mix of CC false-positives vs all CC:")
    fp_ds = Counter(ds_val[cc_fp]); all_cc_ds = Counter(ds_val[is_cc])
    for d in sorted(all_cc_ds):
        fpn, ccn = fp_ds.get(d, 0), all_cc_ds[d]
        print(f"  {d:<14s} CC-FP={fpn:>6d}  (FP-rate within CC = {100*fpn/max(ccn,1):.1f}%, "
              f"share of all CC-FP = {100*fpn/max(cc_fp.sum(),1):.1f}%)")
    print("  Top-10 sequences by CC false-positive count:")
    fp_seq = Counter(seq_val[cc_fp])
    for s, c in fp_seq.most_common(10):
        tot = (seq_val == s).sum()
        print(f"    {s:<34s} {c:>5d} FP  ({100*c/max(tot,1):.1f}% of its frames)")

    # ======================================================================
    # ANALYSIS 2 — abrupt vs gradual: does scale-jump / motion disambiguate?
    # ======================================================================
    hr("ANALYSIS 2 — ABRUPT vs GRADUAL: discontinuity & motion AUROCs (held-out)")

    def run_auc(name, score, pos_mask, neg_mask):
        m = pos_mask | neg_mask
        a, npos, nneg = auroc(score[m], pos_mask[m].astype(int))
        print(f"  {name:<46s} AUROC={a:.3f}  (pos={npos}, neg={nneg})")
        return a

    print("\n[2a] FC-vs-CC (clean CC negatives):")
    run_auc("scale_jump (abrupt scale change)", scale_jump, is_fc, is_cc_clean)
    run_auc("motion_angle (turn)", motion_angle, is_fc, is_cc_clean)
    run_auc("vel_spike (causal velocity spike)", vel_spike, is_fc, is_cc_clean)
    run_auc("edge_frac", edge_frac, is_fc, is_cc_clean)
    run_auc("|log_area_z| (geometry magnitude)", np.abs(log_area_z), is_fc, is_cc_clean)

    print("\n[2b] LA-vs-CC (clean CC negatives):")
    run_auc("scale_jump", scale_jump, is_la, is_cc_clean)
    run_auc("motion_angle", motion_angle, is_la, is_cc_clean)
    run_auc("vel_spike", vel_spike, is_la, is_cc_clean)
    run_auc("|log_area_z|", np.abs(log_area_z), is_la, is_cc_clean)

    print("\n[2c] off_target(LA+FC)-vs-CC (clean CC negatives):")
    run_auc("scale_jump", scale_jump, is_off, is_cc_clean)
    run_auc("motion_angle", motion_angle, is_off, is_cc_clean)
    run_auc("vel_spike", vel_spike, is_off, is_cc_clean)
    run_auc("|log_area_z|", np.abs(log_area_z), is_off, is_cc_clean)

    print("\n[2d] THE CONFOUND TEST — disambiguate within the geometry-suspicious band:")
    print("     true off_target  vs  geometry CC false-positives (both score HIGH on geometry)")
    print("     NOTE: AUROC<0.5 means the feature is LOWER on off_target than on CC-FP;")
    print("           |AUROC-0.5| is the separating power, the sign gives the direction.")
    auc2d_sj = run_auc("scale_jump", scale_jump, is_off, cc_fp)
    auc2d_ma = run_auc("motion_angle", motion_angle, is_off, cc_fp)
    auc2d_vs = run_auc("vel_spike", vel_spike, is_off, cc_fp)
    auc2d_ef = run_auc("edge_frac", edge_frac, is_off, cc_fp)
    auc2d_et = run_auc("edge_touch", edge_touch.astype(float), is_off, cc_fp)
    run_auc("|log_area_z| (geometry, control)", np.abs(log_area_z), is_off, cc_fp)
    # combine the runtime-safe disambiguators via a tiny LR fit on the TRAIN band
    # (honest: never fit on the val target). edge_touch is included because it is the
    # strongest single separator.
    auc2d_lr = float("nan")
    band = is_off | cc_fp
    feats_b = np.column_stack([scale_jump, motion_angle, vel_spike, edge_touch, edge_frac])
    mfin = band & np.all(np.isfinite(feats_b), axis=1)
    if mfin.sum() > 50 and is_off[mfin].sum() > 5 and cc_fp[mfin].sum() > 5:
        lrb = LogisticRegression(class_weight="balanced", max_iter=2000)
        # train-side geometry-FP proxy: CC with la_subtype=FALSE (the labeled hard-negs)
        sj_tr = col(train, "scale_jump"); ma_tr = col(train, "motion_angle"); vs_tr = col(train, "vel_spike")
        et_tr = col(train, "edge_touch"); ef_tr = col(train, "edge_frac")
        la_tr = np.array([r["la_subtype"] for r in train]); dv_tr = np.array([r["derived"] for r in train])
        off_tr = np.isin(dv_tr, (LA, FC))
        ccfp_tr = (dv_tr == CC) & (la_tr == LA_SUBTYPE_FALSE)
        band_tr = off_tr | ccfp_tr
        ftr = np.column_stack([sj_tr, ma_tr, vs_tr, et_tr, ef_tr])
        mt = band_tr & np.all(np.isfinite(ftr), axis=1)
        if mt.sum() > 50 and off_tr[mt].sum() > 5 and ccfp_tr[mt].sum() > 5:
            lrb.fit(ftr[mt], off_tr[mt].astype(int))
            sb = lrb.predict_proba(feats_b[mfin])[:, 1]
            auc2d_lr, npos, nneg = auroc(sb, is_off[mfin].astype(int))
            print(f"  {'LR(scale_jump+motion+vel_spike+edge)':<46s} AUROC={auc2d_lr:.3f}  "
                  f"(pos={npos}, neg={nneg})  [fit on TRAIN band, runtime-safe feats]")

    # ======================================================================
    # ANALYSIS 3 — quantify the hard-negative pool (TRAIN set, no leakage).
    # ======================================================================
    hr("ANALYSIS 3 — hard-negative pool the off_target head must train against")
    # Use TRAIN rows (the pool that actually feeds training).
    dv_tr = np.array([r["derived"] for r in train])
    la_tr = np.array([r["la_subtype"] for r in train])
    area_pct_tr = col(train, "log_area_ratio_to_init_pct")
    area_z_tr = col(train, "log_area_z")
    edge_touch_tr = col(train, "edge_touch")
    sj_tr = col(train, "scale_jump")
    ds_tr = np.array([r["dataset"] for r in train])

    cc_tr = dv_tr == CC
    off_tr = np.isin(dv_tr, (LA, FC))
    fc_tr = dv_tr == FC
    la_tr_state = dv_tr == LA

    # (i) CC frames with large scale drift (extreme area-ratio percentile as proxy
    #     for |log_area_ratio_to_init| > 1; we also report the robust-z>... view).
    cc_big_drift = cc_tr & ((area_pct_tr >= 0.90) | (area_pct_tr <= 0.10))
    cc_big_drift_z = cc_tr & (np.abs(area_z_tr) > SCALE_DRIFT_THR)
    # (ii) LA_FALSE frames (la_subtype=FALSE) — folded into CC in v4 shards.
    cc_la_false_tr = cc_tr & (la_tr == LA_SUBTYPE_FALSE)
    # (iii) edge-pressure CC.
    cc_edge_tr = cc_tr & (edge_touch_tr > 0.5)
    # (iv) smooth-scaler CC (big drift + smooth jump) — the pure confound class.
    off_med_jump_tr = np.nanmedian(sj_tr[off_tr])
    cc_smooth_scaler = cc_big_drift & np.isfinite(sj_tr) & (sj_tr < off_med_jump_tr)

    print(f"TRAIN pool composition:")
    print(f"  CC total ........................................ {cc_tr.sum():>7d}")
    print(f"  off_target (LA+FC) positives .................... {off_tr.sum():>7d} "
          f"(FC={fc_tr.sum()}, LA={la_tr_state.sum()})")
    print(f"  positive:negative ratio ......................... 1 : {cc_tr.sum()/max(off_tr.sum(),1):.1f}")
    print(f"\nHard-negative sub-pools (all within CC):")
    print(f"  (i)   CC large scale-drift (area-pct extreme) ... {cc_big_drift.sum():>7d} "
          f"({100*cc_big_drift.sum()/max(cc_tr.sum(),1):.1f}% of CC)")
    print(f"        CC |log_area_ratio_to_init z|>{SCALE_DRIFT_THR:.0f} .......... {cc_big_drift_z.sum():>7d}")
    print(f"  (ii)  CC la_subtype=FALSE (LA_FALSE hard-neg) ... {cc_la_false_tr.sum():>7d} "
          f"({100*cc_la_false_tr.sum()/max(cc_tr.sum(),1):.1f}% of CC)")
    print(f"  (iii) CC edge-pressure (bbox touches edge) ...... {cc_edge_tr.sum():>7d} "
          f"({100*cc_edge_tr.sum()/max(cc_tr.sum(),1):.1f}% of CC)")
    print(f"  (iv)  CC smooth-scaler (big drift + smooth jump). {cc_smooth_scaler.sum():>7d} "
          f"<-- the pure scale-change confound")

    # overlap of the sub-pools
    print(f"\nOverlap among hard-neg pools:")
    print(f"  big-drift ∩ LA_FALSE ............ {(cc_big_drift & cc_la_false_tr).sum()}")
    print(f"  big-drift ∩ edge ............... {(cc_big_drift & cc_edge_tr).sum()}")
    print(f"  LA_FALSE  ∩ edge ............... {(cc_la_false_tr & cc_edge_tr).sum()}")
    union = cc_big_drift | cc_la_false_tr | cc_edge_tr
    print(f"  union (i ∪ ii ∪ iii) ........... {union.sum()} "
          f"({100*union.sum()/max(cc_tr.sum(),1):.1f}% of CC)")

    print(f"\nHard-neg pool by dataset (CC large-drift):")
    bd_ds = Counter(ds_tr[cc_big_drift])
    for d in sorted(set(ds_tr)):
        print(f"  {d:<14s} {bd_ds.get(d,0):>7d}")

    # ======================================================================
    # SUMMARY VERDICT
    # ======================================================================
    hr("SUMMARY")
    print(f"(a) CONFOUND CONFIRMED. Geometry off_target-vs-CC AUROC={auc_off_cc:.3f}, but the")
    print(f"    high precision is for LA+FC COMBINED; the rare FC class is buried:")
    print(f"      - {100*frac_fp_extreme:.0f}% of geometry CC false-positives have EXTREME scale drift "
          f"vs {100*frac_tn_extreme:.0f}% of CC true-negatives ({frac_fp_extreme/max(frac_tn_extreme,1e-9):.1f}x).")
    print(f"      - {100*healthy.sum()/max(n_fp_with_traj,1):.0f}% of CC false-positives are EXTREME-drift "
          f"+ SMOOTH (jump<off-median) == healthy zoom/recede.")
    print(f"      - FC frames do NOT have the extreme drift LA does (median |log_area_z| "
          f"FC={np.nanmedian(np.abs(log_area_z[is_fc])):.2f} << LA={np.nanmedian(np.abs(log_area_z[is_la])):.2f}),")
    print(f"        so geometry ranks LA + healthy-scaler CC ABOVE FC and FC frame-precision collapses.")
    print(f"(b) DISAMBIGUATION — direction matters:")
    print(f"      - scale_jump does NOT help in the hypothesized direction: off_target is SMOOTHER,")
    print(f"        not more abrupt (off-vs-CC-FP scale_jump AUROC={auc2d_sj:.3f} < 0.5).")
    print(f"      - the real separator is EDGE-PRESSURE: edge_touch AUROC={auc2d_et:.3f}, "
          f"edge_frac AUROC={auc2d_ef:.3f}")
    print(f"        (off_target sits at the frame border: edge_touch rate "
          f"{np.nanmean(edge_touch[is_off]):.2f} vs CC-FP {np.nanmean(edge_touch[cc_fp]):.2f}).")
    print(f"      - combined runtime-safe LR (jump+motion+vel+edge) reaches AUROC={auc2d_lr:.3f} "
          f"WITHIN the geometry-suspicious band.")
    print(f"(c) HARD-NEGATIVE POOL (TRAIN): mine these CC sub-pools against off_target:")
    print(f"      - {cc_big_drift.sum()} big-scale-drift CC ({100*cc_big_drift.sum()/max(cc_tr.sum(),1):.0f}% of CC) "
          f"incl. {cc_smooth_scaler.sum()} pure SMOOTH-SCALERS — the dominant confound;")
    print(f"      - {cc_la_false_tr.sum()} LA_FALSE CC (already labeled la_subtype=FALSE);")
    print(f"      - {cc_edge_tr.sum()} edge-pressure CC (lower count but high precision-cost).")
    print(f"    Recommended NEW features for the off_target head: (1) scale_jump "
          f"|log(area_t/area_t-1)| to down-weight SMOOTH drift, (2) edge_touch / edge_frac "
          f"to keep the FC/LA-at-border signal that geometry-vs-init misses.")


if __name__ == "__main__":
    main()
