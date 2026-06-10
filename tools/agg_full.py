#!/usr/bin/env python
"""Aggregate a full-UAV123 control run vs the passive baseline into a paper-ready
difficulty-tercile summary: per-tercile (EASY/MID/HARD) mean AUC passive vs control
+ ΔAUC, win/loss counts, top winners/losers, and an EASY-scene guard check
(how many easy seqs regressed — should be ~0 for a 'do nothing on easy' gate).

AUC == mean finite per-frame IoU (OPE success area). Reuses la_smoke's loaders.
Read-only; offline SOT benchmark.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "salrtd" / "src", PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "tools"):
    sys.path.insert(0, str(_p))
_argv = sys.argv; sys.argv = [sys.argv[0]]
import csc_uav_tracking  # noqa
import la_smoke as ls
sys.argv = _argv


def auc_gtfail(seq, preds_dir):
    ious, n = ls.seq_iou(seq, preds_dir)
    if ious is None:
        return None, None, None
    fin = np.isfinite(ious)
    if not fin.any():
        return None, None, None
    return float(ious[fin].mean()), float((ious[fin] < 0.2).mean()), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--baseline", default=str(ls.PASSIVE))
    ap.add_argument("--dataset", default="uav123")
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    base = Path(args.baseline); run = Path(args.run_dir)
    print(f"loading {args.dataset} + scoring {args.tag} ...", file=sys.stderr)
    idx = ls.build_index(args.dataset)

    rows = []  # (name, gtfail, base_auc, run_auc, d)
    for name, seq in idx.items():
        b_auc, gtfail, bn = auc_gtfail(seq, base / "predictions")
        r_auc, _, rn = auc_gtfail(seq, run / "predictions")
        if b_auc is None or r_auc is None:
            continue
        rows.append((name, gtfail, b_auc, r_auc, r_auc - b_auc))
    if not rows:
        print("no rows"); return

    def bucket(gt):
        return "EASY" if gt < 0.10 else ("MID" if gt < 0.30 else "HARD")

    print(f"\n================ {args.tag}  ({len(rows)} seqs) vs {base.name} ================")
    print(f"{'tercile':<6} {'n':>4} {'base AUC':>9} {'ctrl AUC':>9} {'ΔAUC':>9} {'wins':>5} {'losses':>7}")
    overall_d = []
    for tname in ("EASY", "MID", "HARD"):
        bk = [r for r in rows if bucket(r[1]) == tname]
        if not bk:
            continue
        bauc = np.mean([r[2] for r in bk]); rauc = np.mean([r[3] for r in bk])
        d = np.mean([r[4] for r in bk])
        wins = sum(1 for r in bk if r[4] > 0.02); losses = sum(1 for r in bk if r[4] < -0.02)
        overall_d += [r[4] for r in bk]
        print(f"{tname:<6} {len(bk):>4} {bauc:>9.3f} {rauc:>9.3f} {d:>+9.4f} {wins:>5} {losses:>7}")
    allb = np.mean([r[2] for r in rows]); allr = np.mean([r[3] for r in rows])
    print(f"{'ALL':<6} {len(rows):>4} {allb:>9.3f} {allr:>9.3f} {np.mean(overall_d):>+9.4f} "
          f"{sum(1 for r in rows if r[4]>0.02):>5} {sum(1 for r in rows if r[4]<-0.02):>7}")

    # EASY-scene guard: easy seqs that regressed (violates 'do nothing on easy')
    easy_reg = sorted([r for r in rows if r[1] < 0.10 and r[4] < -0.02], key=lambda r: r[4])
    print(f"\nEASY-scene regressions (gtfail<0.10, ΔAUC<-0.02): {len(easy_reg)}")
    for name, gt, ba, ra, d in easy_reg[:12]:
        print(f"  {name:<14} gtfail={gt:.3f}  {ba:.3f} -> {ra:.3f}  Δ{d:+.4f}")

    print("\nTOP WINNERS:")
    for name, gt, ba, ra, d in sorted(rows, key=lambda r: -r[4])[:10]:
        print(f"  {name:<14} gtfail={gt:.3f}  {ba:.3f} -> {ra:.3f}  Δ{d:+.4f}")
    print("TOP LOSERS:")
    for name, gt, ba, ra, d in sorted(rows, key=lambda r: r[4])[:10]:
        print(f"  {name:<14} gtfail={gt:.3f}  {ba:.3f} -> {ra:.3f}  Δ{d:+.4f}")


if __name__ == "__main__":
    main()
