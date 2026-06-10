#!/usr/bin/env python3
"""Generate paper §4 control table across trackers × datasets.

Reads eval10_* runs + eval_control_matrix baselines and prints
HARD-bucket AUC/ΔAUC + FCR/FCD for all combos.

Usage: python tools/paper_table.py
"""
from __future__ import annotations
import sys, pathlib, json, argparse
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT, ROOT / "tools"):
    sys.path.insert(0, str(_p))
sys.argv = [sys.argv[0]]
import csc_uav_tracking  # noqa
import la_smoke as ls

HARD_THRESH = 0.30

# ── directory map ─────────────────────────────────────────────────────────
# (tracker, dataset) → {phase: predictions_dir, passive_ref: predictions_dir}
PASSIVE_REFS = {
    ("sglatrack", "uav123"):      ROOT / "outputs/eval5_clamp/csc/sglatrack/uav123/test/passive",
    ("avtrack",   "uav123"):      ROOT / "outputs/baselines/avtrack/uav123/test",
    ("ortrack",   "uav123"):      ROOT / "outputs/baselines/ortrack/uav123/test",
    ("sglatrack", "uav123_10fps"): ROOT / "outputs/eval10_sgla_uav123_10fps/csc/sglatrack/uav123_10fps/test/passive",
    ("avtrack",   "uav123_10fps"): ROOT / "outputs/eval10_avtrack_uav123_10fps/csc/avtrack/uav123_10fps/test/passive",
    ("ortrack",   "uav123_10fps"): ROOT / "outputs/eval10_ortrack_uav123_10fps/csc/ortrack/uav123_10fps/test/passive",
}

CTRL_DIRS = {
    # sgla_redetect + min_sim configs (eval11)
    ("sglatrack", "uav123"): ROOT / "outputs/eval11_sgla/csc/sglatrack/uav123/test",
    ("avtrack",   "uav123"): ROOT / "outputs/eval12_avtrack/csc/avtrack/uav123/test",
    ("ortrack",   "uav123"): ROOT / "outputs/eval12_ortrack/csc/ortrack/uav123/test",
    # motion_bridge + risk_gate configs (eval7/eval13)
    ("sglatrack", "uav123", "mb"): ROOT / "outputs/eval7_gated/csc/sglatrack/uav123/test",
    ("avtrack",   "uav123", "mb"): ROOT / "outputs/eval13_avtrack/csc/avtrack/uav123/test",
    ("ortrack",   "uav123", "mb"): ROOT / "outputs/eval13_ortrack/csc/ortrack/uav123/test",
}

PHASES = ["passive", "la_only", "fc_only", "combo"]
TRACKERS = ["sglatrack", "avtrack", "ortrack"]
DATASETS = ["uav123", "uav123_10fps"]


def gtfail_map(idx: dict, passive_dir: pathlib.Path) -> dict[str, float]:
    out = {}
    for name, seq in idx.items():
        ious, _ = ls.seq_iou(seq, passive_dir / "predictions")
        if ious is not None:
            fin = np.isfinite(ious)
            out[name] = float((ious[fin] < 0.2).mean())
    return out


def mean_auc(idx: dict, preds_dir: pathlib.Path, seqs: set[str] | None = None) -> float | None:
    vals = []
    for name, seq in idx.items():
        if seqs and name not in seqs:
            continue
        ious, _ = ls.seq_iou(seq, preds_dir / "predictions")
        if ious is not None:
            fin = np.isfinite(ious)
            if fin.any():
                vals.append(float(ious[fin].mean()))
    return float(np.mean(vals)) if vals else None


def fcr_from_states(states_dir: pathlib.Path, seqs: set[str] | None = None) -> tuple[float, float] | None:
    """Returns (FCR%, FCD) or None."""
    if not states_dir.exists():
        return None
    total = fc = 0; fcd_v = []
    for f in states_dir.glob("*.jsonl"):
        if seqs and f.stem not in seqs:
            continue
        try:
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            rows = [r for r in rows if not r.get("init")]
            if not rows:
                continue
            total += len(rows)
            fc_frames = [i for i, r in enumerate(rows)
                         if r.get("false_confirmed_flag") or r.get("derived_state") == 3]
            fc += len(fc_frames)
            # FCD
            states_arr = [1 if (r.get("false_confirmed_flag") or r.get("derived_state") == 3) else 0
                          for r in rows]
            i = 0
            while i < len(states_arr):
                if states_arr[i]:
                    j = i
                    while j < len(states_arr) and states_arr[j]:
                        j += 1
                    fcd_v.append(j - i)
                    i = j
                else:
                    i += 1
        except Exception:
            pass
    if not total:
        return None
    return fc / total * 100, float(np.mean(fcd_v)) if fcd_v else 0.0


def fmt(v, ref=None, is_delta=False):
    if v is None:
        return "  ---  "
    if is_delta:
        return f"{v:>+7.4f}"
    return f"{v:>7.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="Filter to one dataset")
    args = ap.parse_args(sys.argv[1:])

    datasets = [args.dataset] if args.dataset else DATASETS

    # Use SGLATrack passive as difficulty reference
    sgla_idx = ls.build_index("uav123")
    sgla_passive_ref = PASSIVE_REFS[("sglatrack", "uav123")]
    gf = gtfail_map(sgla_idx, sgla_passive_ref)
    hard_seqs_uav123 = {n for n, g in gf.items() if g >= HARD_THRESH}

    print(f"\n{'='*80}")
    print("  CSC CONTROL TABLE — HARD BUCKET  (gtfail ≥ 0.30 by SGLA passive reference)")
    print(f"{'='*80}")

    for ds in datasets:
        idx = ls.build_index(ds)
        # For @10fps: same seq names, hard seqs inherited from UAV123
        hard_seqs = hard_seqs_uav123 & set(idx.keys())
        n_hard = len(hard_seqs)

        print(f"\n  Dataset: {ds}  |  HARD seqs: {n_hard}")
        print(f"  {'tracker':<12} {'phase':<9} {'AUC':>7} {'ΔAUC':>8} {'FCR%':>6} {'FCD':>5}  {'n':>4}")
        print("  " + "-" * 60)

        for tracker in TRACKERS:
            ctrl_root = CTRL_DIRS.get((tracker, ds))
            passive_ref = PASSIVE_REFS.get((tracker, ds))

            # Determine passive reference for AUC baseline
            if passive_ref and passive_ref.exists():
                base_auc = mean_auc(idx, passive_ref, hard_seqs)
            elif ctrl_root and (ctrl_root / "passive").exists():
                base_auc = mean_auc(idx, ctrl_root / "passive", hard_seqs)
                passive_ref = ctrl_root / "passive"
            else:
                base_auc = None

            for phase in PHASES:
                if phase == "passive":
                    # Use passive reference
                    auc = base_auc
                    delta = None
                    preds_dir = passive_ref
                else:
                    if ctrl_root is None:
                        auc = delta = None
                        preds_dir = None
                    else:
                        preds_dir = ctrl_root / phase
                        auc = mean_auc(idx, preds_dir, hard_seqs) if preds_dir and preds_dir.exists() else None
                        delta = (auc - base_auc) if (auc is not None and base_auc is not None) else None

                # FCR
                fcr_str = " ---"
                if preds_dir and (preds_dir / "states").exists():
                    fcr_fcd = fcr_from_states(preds_dir / "states", hard_seqs)
                    if fcr_fcd:
                        fcr_str = f"{fcr_fcd[0]:>5.2f}"

                n_seqs = 0
                if preds_dir and (preds_dir / "predictions").exists():
                    n_seqs = sum(1 for f in (preds_dir / "predictions").glob("*")
                                 if f.stem in hard_seqs)

                auc_s = f"{auc:>7.4f}" if auc is not None else "   --- "
                d_s = f"{delta:>+8.4f}" if delta is not None else "    --- "
                print(f"  {tracker:<12} {phase:<9} {auc_s} {d_s} {fcr_str:>6}  ---  {n_seqs:>4}")

            print()


if __name__ == "__main__":
    main()
