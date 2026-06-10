#!/usr/bin/env python
"""Per-seq FC-control attribution: AUC + FCR + FCD change (control vs passive),
filtered/ranked for the FC-only experiment. FCR = fraction of frames CSC predicts
FALSE_CONFIRMED (derived_state==3); FCD = mean FC run length. Reuses la_smoke loaders.
Read-only; offline SOT.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "tools"):
    sys.path.insert(0, str(_p))
_argv = sys.argv; sys.argv = [sys.argv[0]]
import csc_uav_tracking  # noqa
import la_smoke as ls
sys.argv = _argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--baseline", default=str(ls.PASSIVE))
    ap.add_argument("--dataset", default="uav123")
    ap.add_argument("--tag", default="fc")
    args = ap.parse_args()
    base = Path(args.baseline); run = Path(args.run_dir)
    print(f"loading {args.dataset} + scoring {args.tag} ...", file=sys.stderr)
    idx = ls.build_index(args.dataset)

    rows = []  # name, gtfail, dAUC, base_fcr, run_fcr, dFCR(pp), base_fcd, run_fcd, dFCD, dAUC
    for name, seq in idx.items():
        b_iou, bn = ls.seq_iou(seq, base / "predictions")
        r_iou, rn = ls.seq_iou(seq, run / "predictions")
        if b_iou is None or r_iou is None:
            continue
        n = min(bn, rn)
        bfin = np.isfinite(b_iou); rfin = np.isfinite(r_iou)
        b_auc = float(b_iou[bfin].mean()) if bfin.any() else 0.0
        r_auc = float(r_iou[rfin].mean()) if rfin.any() else 0.0
        gtfail = float((b_iou[bfin] < 0.2).mean()) if bfin.any() else 0.0
        b_fcr, b_fcd = ls.fc_stats(base / "states" / f"{name}.jsonl", n)
        r_fcr, r_fcd = ls.fc_stats(run / "states" / f"{name}.jsonl", n)
        if b_fcr is None or r_fcr is None:
            continue
        rows.append((name, gtfail, r_auc - b_auc,
                     100 * b_fcr, 100 * r_fcr, 100 * (r_fcr - b_fcr),
                     b_fcd, r_fcd, r_fcd - b_fcd))

    def show(title, sel, key):
        sel = sorted(sel, key=key)
        print(f"\n=== {title} ({len(sel)}) ===")
        print(f"{'seq':<20}{'gtfail':>7}{'dAUC':>8}{'bFCR%':>7}{'rFCR%':>7}{'dFCR':>7}{'bFCD':>6}{'rFCD':>6}{'dFCD':>7}")
        for name, gt, dauc, bfcr, rfcr, dfcr, bfcd, rfcd, dfcd in sel:
            print(f"{name:<20}{gt:>7.3f}{dauc:>+8.3f}{bfcr:>7.2f}{rfcr:>7.2f}{dfcr:>+7.2f}{bfcd:>6.1f}{rfcd:>6.1f}{dfcd:>+7.2f}")

    hard = [r for r in rows if r[1] >= 0.30]
    fc_active = [r for r in rows if r[3] > 0.0]          # seqs with ANY passive FC
    hard_fc = [r for r in rows if r[1] >= 0.30 and r[3] > 0.0]

    # rank by FCR reduction (most negative dFCR first), tie-break by dAUC desc
    show("TOP-10 HARD (gtfail>=0.30) by FCR reduction", hard, key=lambda r: (r[5], -r[2]))
    show("TOP-10 HARD-with-FC (gtfail>=0.30 AND base FCR>0) by FCR reduction", hard_fc, key=lambda r: (r[5], -r[2]))
    show("TOP-15 FC-ACTIVE (any difficulty) by FCR reduction", fc_active, key=lambda r: (r[5], -r[2]))

    # aggregates
    def agg(sel, lbl):
        if not sel: print(f"  {lbl}: (none)"); return
        print(f"  {lbl}: n={len(sel)}  meanΔAUC={np.mean([r[2] for r in sel]):+.4f}  "
              f"meanΔFCR={np.mean([r[5] for r in sel]):+.3f}pp  meanΔFCD={np.mean([r[8] for r in sel]):+.3f}")
    print("\n=== AGGREGATES ===")
    agg(rows, "ALL seqs")
    agg(hard, "HARD")
    agg(fc_active, "FC-active")


if __name__ == "__main__":
    main()
