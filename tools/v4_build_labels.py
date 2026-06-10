#!/usr/bin/env python
"""V4 training-shard builder (keystone of the V4 retrain).

Joins the freshly re-extracted FULL telemetry (response + appearance, from
outputs/baselines_v4/) into the merged GT label file (train2_v3_combined labels.jsonl,
which carries gt_bbox / pred_bbox / iou / occlusion proxies), then for every frame emits
the V4 training targets:
  - features  : features_v4.build_v4_features (FV.FEATURE_DIM_V4-dim normalized features:
                response + extras + appearance)
  - diagnosis : labeling_v4.build_v4_labels  -> derived(4) / fc_subtype(3) / la_subtype(6)
  - hazard    : failure (IoU<tau_fail) within the next {1,3,10} frames  (causal future label)
  - action    : offline counterfactual ΔIoU per Action (7) + do_not_act + template_update_safe.
                Only motion_bridge (velocity-extrapolation ΔIoU) and the position-neutral
                hold/freeze/template_update actions carry honest labels; relocate / widen /
                global_search are UNTRUSTED (utility 0, trust mask 0) because no honest
                offline label exists for them (a real label would require tracker-in-the-loop
                candidate sweeps, which we do not run here). A per-action trust mask
                (act_trust_<name>) lets the trainer ignore the untrusted action labels.
Output: JSONL shards of per-frame rows (grouped by dataset/sequence, ordered by frame_idx;
pyarrow not required) + the fitted V4 feature calibrators (JSON) for inference. Train set
only (NEVER UAV123).

Run AFTER outputs/baselines_v4/ telemetry exists. CPU; offline; no tracker rerun.
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    sys.path.insert(0, str(_p))

from csc_lib.csc.v4 import features_v4 as FV
from csc_lib.csc.v4.labeling_v4 import build_v4_labels, LabelingThresholdsV4
from csc_lib.csc.v4.v4types import Action, ACTION_NAMES, DerivedStateV4, FCSubtype, LASubtype

TRACKER = "sglatrack"
# Actions for which we have an honest offline ΔIoU label: motion_bridge is a real
# velocity-extrapolation counterfactual; hold/freeze/template_update are position-neutral
# (0.0). relocate / widen / global_search are UNTRUSTED — a real label would need
# tracker-in-the-loop candidate sweeps (often jumping onto a distractor), which we do not
# run offline. The per-action trust mask (act_trust_<name>) lets the trainer ignore them.
TRUSTED_ACTIONS = {"hold", "motion_bridge", "freeze", "template_update"}
# telemetry fields to merge in from baselines_v4 (response + appearance)
TELE_FIELDS = [
    "response_entropy", "sm_local_top2_ratio", "sm_local_peak_margin", "sm_peak_distance",
    "sm_heatmap_mass_topk", "sm_n_secondary", "sm_peak_width", "sm_top1", "sm_top2",
    "sm_local_top1", "sm_local_top2", "sm_peak_margin",
    "last_cosine_sim", "initial_template_sim", "appearance_drift", "apce", "psr", "confidence",
]


def _fc_subtype_from_telemetry(row: dict) -> FCSubtype:
    """Fallback FC subtype when preserving audited V3 FC rows.

    V4's strict identity-off rule can erase all FC when SGLATrack appearance
    features remain high-sim even on false-confirmed frames. The source V3 labels
    still contain audited FC supervision, so keep those frames trainable and split
    subtype from response-map competitor cues.
    """
    try:
        n_secondary = float(row.get("sm_n_secondary", 0.0) or 0.0)
    except (TypeError, ValueError):
        n_secondary = 0.0
    try:
        top2 = float(row.get("sm_local_top2_ratio", 0.0) or 0.0)
    except (TypeError, ValueError):
        top2 = 0.0
    return (
        FCSubtype.DISTRACTOR
        if n_secondary >= 1.0 or top2 >= 0.45
        else FCSubtype.BACKGROUND
    )


def _iou_xywh(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _telemetry_index(tele_root: Path, dataset: str, split: str, seq: str) -> dict[int, dict]:
    p = tele_root / TRACKER / dataset / split / "telemetry" / f"{seq}.jsonl"
    out: dict[int, dict] = {}
    if not p.exists():
        return out
    for ln in open(p):
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        out[int(d.get("frame_idx", -1))] = d
    return out


def _inject_geometry(rows: list[dict]) -> None:
    """Add runtime-safe geometry/shape-vs-init features to each row IN PLACE.

    Computed from the pred_bbox trajectory + the init (first valid) pred_bbox + a
    confidence EMA — NO ground truth, so runtime-available. These are the
    FALSE_CONFIRMED-vs-CORRECT_CONFIRMED discriminators the V4 feature redesign had
    dropped (see csc_lib/csc/v4/features_v4.GEOM_FEATURES). MEASURED leakage-free:
    they lift held-out FC-vs-CC AUROC 0.65 -> 0.81. Field names match GEOM_FEATURES.
    """
    w0 = h0 = a0 = None
    for r in rows:
        pb = r.get("pred_bbox")
        try:
            if pb and len(pb) >= 4 and float(pb[2]) > 0 and float(pb[3]) > 0:
                w0 = float(pb[2]); h0 = float(pb[3]); a0 = w0 * h0
                break
        except (TypeError, ValueError):
            continue
    if w0 is None:
        w0 = h0 = a0 = 1.0
    ema = None
    for r in rows:
        pb = r.get("pred_bbox") or [0.0, 0.0, 1.0, 1.0]
        try:
            w = max(float(pb[2]), 1e-3); h = max(float(pb[3]), 1e-3)
        except (TypeError, ValueError, IndexError):
            w = h = 1.0
        a = w * h
        try:
            conf = float(r.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        ema = conf if ema is None else 0.7 * ema + 0.3 * conf
        r["log_w_ratio_to_init"] = math.log(w / max(w0, 1e-3))
        r["log_h_ratio_to_init"] = math.log(h / max(h0, 1e-3))
        r["log_area_ratio_to_init"] = math.log(a / max(a0, 1e-3))
        r["aspect_ratio"] = w / h
        r["conf_ema_trend"] = conf - ema


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_jsonl", nargs="+", default=[
        "outputs/csc_labels/sglatrack/v3fix_combined/base/labels.jsonl",     # lasot+uavdt+visdrone
        "outputs/csc_labels/sglatrack/v3fix_combined/got10k/labels.jsonl",
        "outputs/csc_labels/sglatrack/v3fix_combined/dtb70/labels.jsonl",
    ], help="one or more merged GT label shards (v3fix_combined base/got10k/dtb70)")
    ap.add_argument("--telemetry_root", default="outputs/baselines_v4")
    ap.add_argument("--out", default="outputs/csc_labels_v4/train_shards.jsonl")
    ap.add_argument("--calib_out", default="outputs/csc_labels_v4/v4_feature_calibrators.json")
    ap.add_argument("--datasets", nargs="*", default=None, help="filter to these dataset names")
    ap.add_argument("--tau_fail", type=float, default=0.20)
    ap.add_argument("--risk_iou", type=float, default=0.50, help="action-window: frames with iou<this")
    ap.add_argument("--pre_window", type=int, default=3)
    ap.add_argument("--max_missing_frac", type=float, default=0.02,
                    help="abort if fraction of rows missing telemetry exceeds this (default 2%%)")
    ap.add_argument("--limit_seqs", type=int, default=0, help="dev: cap sequences (0=all)")
    args = ap.parse_args()

    tele_root = Path(args.telemetry_root)
    print(f"loading {len(args.labels_jsonl)} label shard(s) ...", file=sys.stderr)

    # group label rows by (dataset, split, sequence) across all shards
    seqs: dict[tuple, list[dict]] = defaultdict(list)
    for lp in args.labels_jsonl:
        lp = Path(lp)
        if not lp.exists():
            print(f"  WARN missing shard {lp}", file=sys.stderr)
            continue
        with open(lp) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                ds = r.get("dataset"); sp = r.get("split", "train"); sq = r.get("sequence")
                if args.datasets and ds not in args.datasets:
                    continue
                seqs[(ds, sp, sq)].append(r)
    keys = sorted(seqs.keys())
    if args.limit_seqs:
        keys = keys[: args.limit_seqs]
    print(f"sequences: {len(keys)}  datasets: {sorted({k[0] for k in keys})}", file=sys.stderr)

    # ---- PASS 1: merge telemetry, collect response values, fit calibrators ----
    merged: dict[tuple, list[dict]] = {}
    miss_tele = 0
    miss_datasets: set[str] = set()
    for k in keys:
        ds, sp, sq = k
        tele = _telemetry_index(tele_root, ds, sp, sq)
        rows = sorted(seqs[k], key=lambda r: int(r.get("frame_idx", 0)))
        for r in rows:
            t = tele.get(int(r.get("frame_idx", -1)), {})
            if not t:
                miss_tele += 1
                miss_datasets.add(ds)
            for f in TELE_FIELDS:
                if f in t and t[f] is not None:
                    r[f] = t[f]
        _inject_geometry(rows)   # runtime-safe geometry/shape-vs-init (FC-vs-CC); before calib fit
        merged[k] = rows
    flat = [r for k in keys for r in merged[k]]
    miss_frac = miss_tele / max(1, len(flat))
    cals = FV.fit_v4_calibrators(flat)
    print(f"fitted {len(cals)} calibrators; rows={len(flat)}; rows-missing-telemetry={miss_tele} "
          f"({miss_frac:.4f})", file=sys.stderr)
    # HARD FAIL: never build labels on incomplete telemetry (silent partial-telemetry bug).
    if miss_frac > args.max_missing_frac:
        print(
            f"ABORT: missing-telemetry fraction {miss_frac:.4f} exceeds tolerance "
            f"{args.max_missing_frac:.4f} ({miss_tele}/{len(flat)} rows). "
            f"datasets with missing telemetry: {sorted(miss_datasets)}. "
            f"Re-extract telemetry or relax with --max_missing_frac.",
            file=sys.stderr,
        )
        return 1

    # ---- PASS 2: per-seq features + labels + hazard + action-gain ----
    out_rows: list[dict] = []
    fc_preserved_from_source = 0
    th = LabelingThresholdsV4()
    for k in keys:
        ds, sp, sq = k
        rows = merged[k]
        n = len(rows)
        iou = np.array([float(r.get("iou", 0.0) or 0.0) for r in rows], float)
        gt = [r.get("gt_bbox") for r in rows]
        pred = [r.get("pred_bbox") for r in rows]
        occ = [bool(r.get("absent", 0)) for r in rows]
        oov = [(float(r.get("visible_ratio", 1.0) or 1.0) < 0.05) for r in rows]

        # diagnosis labels (derived / fc_subtype / la_subtype) — uses merged telemetry + GT.
        # NOTE: per-candidate cand_sims (prototype/distractor memory similarities) are NOT
        # available offline, so FC subtypes (FC_D / FC_B) fall back to telemetry structure
        # (sm_* fields) rather than memory similarity. This is a known, documented limitation
        # — we do NOT fabricate cand_sims.
        try:
            v4labels = build_v4_labels(rows, gt, occ=occ, oov=oov, pred_bboxes=pred)
        except Exception as exc:
            print(f"  WARN build_v4_labels failed for {k}: {exc}", file=sys.stderr)
            continue
        # Preserve audited source FC frames when V4's stricter appearance rule
        # would erase them. Without this fallback the full shard had 0 FC rows,
        # making the FC head impossible to train.
        for t, r in enumerate(rows):
            if int(r.get("derived_state", -1)) == int(DerivedStateV4.FC) and v4labels[t]["derived"] != DerivedStateV4.FC:
                v4labels[t] = {
                    "derived": DerivedStateV4.FC,
                    "fc_subtype": _fc_subtype_from_telemetry(r),
                    "la_subtype": LASubtype.NONE,
                }
                fc_preserved_from_source += 1

        # features (causal: prev row)
        prev = None
        feats = []
        for r in rows:
            fv = FV.build_v4_features(r, cals, prev=prev); prev = r
            feats.append(fv)

        # hazard: failure (iou<tau_fail) within next {1,3,10} frames
        fail = (iou < args.tau_fail).astype(np.int8)
        def hz(t, h):
            j2 = min(n, t + 1 + h)
            return int(fail[t + 1:j2].any()) if t + 1 < n else 0

        # velocity for motion_bridge approx (centers)
        def center(b):
            return None if b is None else (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)
        cen = [center(b) for b in pred]

        for t in range(n):
            r = rows[t]
            in_win = bool(iou[t] < args.risk_iou) or (t + 1 < n and (iou[t + 1:min(n, t + 1 + args.pre_window)] < args.risk_iou).any())
            # ---- offline action-gain (ΔIoU vs passive hold), only in loss/risk window ----
            # Only motion_bridge is a real (velocity-extrapolation) counterfactual; hold/
            # freeze/template_update are honestly position-neutral (0.0). relocate / widen /
            # global_search are UNTRUSTED (utility 0, trust mask 0) because no honest offline
            # label exists — a real label needs tracker-in-the-loop candidate sweeps that we
            # do not run here (the re-detector frequently jumps onto a distractor).
            ag = {a: 0.0 for a in ACTION_NAMES}
            dna = 1; tus = 1
            if in_win and gt[t] is not None:
                gtb = gt[t]
                base = iou[t]                                   # hold == passive
                # motion_bridge: extrapolate last-good center velocity (simple: prev velocity)
                mb_iou = base
                if t >= 2 and cen[t - 1] and cen[t - 2] and pred[t] is not None:
                    vx = cen[t - 1][0] - cen[t - 2][0]; vy = cen[t - 1][1] - cen[t - 2][1]
                    cx = cen[t - 1][0] + vx; cy = cen[t - 1][1] + vy
                    pw, ph = pred[t][2], pred[t][3]
                    mb_iou = _iou_xywh([cx - pw / 2, cy - ph / 2, pw, ph], gtb)
                ag["hold"] = 0.0
                ag["motion_bridge"] = mb_iou - base
                ag["relocate"] = 0.0                            # UNTRUSTED: no honest offline label
                ag["widen"] = 0.0                               # UNTRUSTED: no honest offline label
                ag["global_search"] = 0.0                       # UNTRUSTED: no honest offline label
                ag["template_update"] = 0.0                     # position-neutral this frame
                ag["freeze"] = 0.0
                best = max(ag.values())                         # only motion_bridge can be >0 now
                dna = int(best <= 1e-6)                         # do_not_act when even motion_bridge doesn't help
                tus = int(base >= args.risk_iou)                # safe to update only when on-target
            out_rows.append({
                **{f"feat_{i}": float(feats[t][i]) for i in range(len(feats[t]))},
                "derived": int(v4labels[t]["derived"]),
                "fc_subtype": int(v4labels[t]["fc_subtype"]),
                "la_subtype": int(v4labels[t]["la_subtype"]),
                "hazard_1": hz(t, 1), "hazard_3": hz(t, 3), "hazard_10": hz(t, 10),
                **{f"act_{a}": float(ag[a]) for a in ACTION_NAMES},
                **{f"act_trust_{a}": (1.0 if a in TRUSTED_ACTIONS else 0.0) for a in ACTION_NAMES},
                "do_not_act": int(dna), "template_update_safe": int(tus),
                "in_action_window": int(in_win),
                "dataset": ds, "split": sp, "sequence": sq,
                "frame_idx": int(r.get("frame_idx", t)), "iou": float(iou[t]),
            })

    # ---- write (jsonl; pyarrow not required) ----
    from collections import Counter
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    # save calibrators (per-feature JSON) + a manifest for inference
    cal_path = Path(args.calib_out); cal_path.parent.mkdir(parents=True, exist_ok=True)
    cal_path.write_text(json.dumps({"feature_dim": FV.FEATURE_DIM_V4, "feature_names": FV.FEATURE_NAMES_V4,
                                    "calibrators": list(cals.keys())}, indent=1))
    for name, c in cals.items():
        try:
            c.save(cal_path.parent / f"cal_{name}.json")
        except Exception as exc:
            print(f"  WARN could not save calibrator {name}: {exc}", file=sys.stderr)
    # report
    der = Counter(r["derived"] for r in out_rows)
    las = Counter(r["la_subtype"] for r in out_rows)
    fcs = Counter(r["fc_subtype"] for r in out_rows)
    print(f"\nWROTE {len(out_rows)} rows -> {out_path}")
    print(f"derived dist: {{{', '.join(f'{DerivedStateV4(int(kk)).name}:{vv}' for kk,vv in sorted(der.items()))}}}")
    print(f"la_subtype dist: {dict(sorted(las.items()))}")
    print(f"fc_subtype dist: {dict(sorted(fcs.items()))}")
    print(f"source FC rows preserved by fallback: {fc_preserved_from_source}")
    print(f"action-window frames: {sum(r['in_action_window'] for r in out_rows)}/{len(out_rows)}; "
          f"do_not_act=1: {sum(r['do_not_act'] for r in out_rows)}; hazard_10 pos: {sum(r['hazard_10'] for r in out_rows)}")
    print(f"calibrators -> {cal_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
