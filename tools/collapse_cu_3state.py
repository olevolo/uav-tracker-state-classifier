#!/usr/bin/env python3
"""Collapse the broken CU state into CC and report a 3-state OPERATIONAL
confusion / macro-F1 over the safety-relevant states {CC', LA, FC}.

Why: CU (=CORRECT_UNCERTAIN) is defined purely as localization IoU in [0.2,0.5)
(see csc_lib/csc/labeling/label_schema.py:derive_state — the confidence axis is
ignored unless loc is LOST). That GT-IoU middle band has no distinctive runtime
telemetry signature, so it is structurally unlearnable (CU recall 0.13-0.16) and
drags macro-F1 down. It is NOT safety-critical: CU is "still tracking, just lower
overlap", so the principled operational merge is CU -> CC ("not lost"). Merging
leaves LA and FC precision/recall/F1 EXACTLY unchanged (pure arithmetic check)
and isolates the safety contribution the paper actually makes.

NO RETRAIN, NO TRACKER RE-RUN — this only re-reads the already-saved 4x4
confusion_matrix_uav123.csv files and re-aggregates. Reproducible.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STATES4 = ["CC", "CU", "LA", "FC"]
STATES3 = ["CC*", "LA", "FC"]  # CC* = CC u CU


def read_cm4(csv_path: Path) -> np.ndarray:
    """Parse the 4x4 integer confusion from confusion_matrix_uav123.csv."""
    cm = np.zeros((4, 4), dtype=np.int64)
    for ln in csv_path.read_text().splitlines():
        parts = ln.split(",")
        if parts and parts[0] in STATES4:
            i = STATES4.index(parts[0])
            cm[i] = [int(float(parts[1 + j])) for j in range(4)]
    return cm


def collapse_cu_to_cc(cm4: np.ndarray) -> np.ndarray:
    """Merge index 1 (CU) into index 0 (CC), rows and columns -> 3x3."""
    # merge column CU into column CC
    m = cm4.copy()
    m[:, 0] += m[:, 1]
    m = np.delete(m, 1, axis=1)
    # merge row CU into row CC
    m[0, :] += m[1, :]
    m = np.delete(m, 1, axis=0)
    return m  # order: CC*, LA, FC


def prf(cm: np.ndarray, labels: list[str]):
    rows = []
    f1s = []
    for k in range(len(labels)):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s.append(f1)
        rows.append((labels[k], int(cm[k].sum()), p, r, f1))
    acc = np.trace(cm) / cm.sum() if cm.sum() else 0.0
    return rows, float(np.mean(f1s)), float(acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackers", nargs="+", default=["sglatrack", "avtrack", "ortrack"])
    ap.add_argument("--run_tag", default="{t}_r3_passive")
    ap.add_argument("--dataset", default="uav123",
                    help="Final-test dataset (e.g. uav123 or uav123_10fps). Default uav123.")
    ap.add_argument("--eval_root", default="outputs/eval")
    ap.add_argument("--save", action="store_true",
                    help="write confusion_3state_<dataset>.csv next to each 4x4 csv")
    args = ap.parse_args()

    ds = args.dataset
    root = ROOT / args.eval_root
    md = ["| tracker | 4-state macro-F1 | **3-state macro-F1** | CC* F1 | LA F1 | FC F1 | FC recall | FC prec |",
          "|---|---|---|---|---|---|---|---|"]
    for t in args.trackers:
        csv = root / t / ds / "test" / args.run_tag.format(t=t) / "paper_metrics" / f"confusion_matrix_{ds}.csv"
        if not csv.is_file():
            print(f"[skip] {t}: missing {csv}")
            continue
        cm4 = read_cm4(csv)
        rows4, mf4, acc4 = prf(cm4, STATES4)
        cm3 = collapse_cu_to_cc(cm4)
        rows3, mf3, acc3 = prf(cm3, STATES3)

        print(f"\n===== {t}  (n={int(cm4.sum())}) =====")
        print(f"  4-state macro-F1={mf4:.4f} acc={acc4:.4f}  ->  "
              f"3-state(CU->CC) macro-F1={mf3:.4f} acc={acc3:.4f}  (+{mf3-mf4:.4f})")
        print(f"  {'state':>5} {'support':>9} {'prec':>7} {'recall':>7} {'F1':>7}")
        for name, sup, p, r, f1 in rows3:
            print(f"  {name:>5} {sup:>9} {p:>7.3f} {r:>7.3f} {f1:>7.3f}")
        # invariance check: LA & FC F1 must be identical before/after collapse
        la4 = next(x for x in rows4 if x[0] == "LA")
        fc4 = next(x for x in rows4 if x[0] == "FC")
        la3 = next(x for x in rows3 if x[0] == "LA")
        fc3 = next(x for x in rows3 if x[0] == "FC")
        inv = abs(la4[4] - la3[4]) < 1e-9 and abs(fc4[4] - fc3[4]) < 1e-9
        print(f"  invariance(LA,FC F1 unchanged by CU collapse): {'PASS' if inv else 'FAIL'}")

        md.append(f"| {t} | {mf4:.3f} | **{mf3:.3f}** | {rows3[0][4]:.3f} | "
                  f"{la3[4]:.3f} | {fc3[4]:.3f} | {fc3[3]:.3f} | {fc3[2]:.3f} |")

        if args.save:
            out = csv.with_name(f"confusion_3state_{ds}.csv")
            with open(out, "w") as f:
                f.write("gt_state," + ",".join(f"pred_{s}" for s in STATES3) + ",recall,support\n")
                for i, s in enumerate(STATES3):
                    sup = int(cm3[i].sum())
                    rec = cm3[i, i] / sup if sup else 0.0
                    f.write(f"{s}," + ",".join(str(int(c)) for c in cm3[i]) + f",{rec:.4f},{sup}\n")
                f.write(f"# 3state_macro_f1={mf3:.4f} acc={acc3:.4f} (4state_macro_f1={mf4:.4f}) merge=CU->CC\n")
            print(f"  -> {out}")

    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
