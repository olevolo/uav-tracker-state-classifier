#!/usr/bin/env python3
"""Confusion matrix of CSC predicted state vs GT-derived weak label on UAV123.

Diagnosis-only, offline. For each tracker's passive run we align the model's
per-frame `derived_state` (states/<seq>.jsonl) with the GT-rule weak label
`derived_state` (labels_v3/.../labels.jsonl, computed from that tracker's own
IoU+confidence). Rows = GT state, cols = predicted state. Row-normalised values
are per-state recall; the diagonal is "how often the model got that state right".
No fabricated numbers — pure count of measured predictions.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

STATES = ["CC", "CU", "LA", "FC"]  # 0,1,2,3 = CORRECT_CONFIRMED / _UNCERTAIN / LOST_AWARE / FALSE_CONFIRMED


def load_gt(labels_jsonl: Path) -> dict[tuple[str, int], int]:
    gt: dict[tuple[str, int], int] = {}
    with open(labels_jsonl) as f:
        for line in f:
            d = json.loads(line)
            s = d.get("derived_state")
            if s is None:
                continue
            gt[(d["sequence"], int(d["frame_idx"]))] = int(s)
    return gt


def load_preds(states_dir: Path) -> dict[tuple[str, int], int]:
    pred: dict[tuple[str, int], int] = {}
    for fp in sorted(states_dir.glob("*.jsonl")):
        seq = fp.stem
        with open(fp) as f:
            for line in f:
                d = json.loads(line)
                if d.get("init"):
                    continue
                s = d.get("derived_state")
                if s is None:
                    continue
                pred[(seq, int(d["frame_idx"]))] = int(s)
    return pred


def confusion(gt: dict, pred: dict) -> tuple[np.ndarray, int]:
    cm = np.zeros((4, 4), dtype=np.int64)
    n = 0
    for key, p in pred.items():
        g = gt.get(key)
        if g is None:
            continue
        cm[g, p] += 1
        n += 1
    return cm, n


def macro_f1(cm: np.ndarray) -> tuple[float, list[float]]:
    f1s = []
    for k in range(4):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), f1s


def report(name: str, cm: np.ndarray, n: int, out_csv: Path | None):
    total = cm.sum()
    acc = np.trace(cm) / total if total else 0.0
    mf1, f1s = macro_f1(cm)
    print(f"\n===== {name}  (n_aligned={n}) =====")
    hdr = "GT\\pred"
    print(f"{hdr:>8} " + " ".join(f"{s:>9}" for s in STATES) + f" {'recall':>8} {'support':>9}")
    for i, s in enumerate(STATES):
        row = cm[i]
        supp = row.sum()
        rec = row[i] / supp if supp else 0.0
        print(f"{s:>8} " + " ".join(f"{int(c):>9}" for c in row) + f" {rec:>8.3f} {int(supp):>9}")
    # precision row
    prec = [cm[k, k] / cm[:, k].sum() if cm[:, k].sum() else 0.0 for k in range(4)]
    print(f"{'prec':>8} " + " ".join(f"{p:>9.3f}" for p in prec))
    print(f"  accuracy={acc:.4f}  macro-F1={mf1:.4f}  per-state F1: " +
          ", ".join(f"{s}={f:.3f}" for s, f in zip(STATES, f1s)))
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w") as f:
            f.write("gt_state," + ",".join(f"pred_{s}" for s in STATES) + ",recall,support\n")
            for i, s in enumerate(STATES):
                supp = int(cm[i].sum())
                rec = cm[i, i] / supp if supp else 0.0
                f.write(f"{s}," + ",".join(str(int(c)) for c in cm[i]) + f",{rec:.4f},{supp}\n")
            f.write("precision," + ",".join(f"{p:.4f}" for p in prec) + f",,\n")
            f.write(f"# accuracy={acc:.4f} macro_f1={mf1:.4f} n_aligned={n}\n")
        print(f"  -> {out_csv}")
    return acc, mf1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", default="outputs/eval")
    ap.add_argument("--trackers", nargs="+", default=["sglatrack", "avtrack", "ortrack"])
    ap.add_argument("--run_tag", default="{t}_r3_passive")
    ap.add_argument("--dataset", default="uav123",
                    help="Final-test dataset (e.g. uav123 or uav123_10fps). Default uav123.")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    ds = args.dataset
    root = Path(args.eval_root)
    overall = np.zeros((4, 4), dtype=np.int64)
    for t in args.trackers:
        run = root / t / ds / "test" / args.run_tag.format(t=t)
        states_dir = run / "states"
        labels = run / "labels_v3" / ds / "test" / "labels.jsonl"
        if not labels.is_file():  # matrix puts labels under run_tag; fall back to legacy path
            labels = root / t / ds / "test" / "labels_v3" / ds / "test" / "labels.jsonl"
        if not states_dir.is_dir() or not labels.is_file():
            print(f"[skip] {t}: missing states ({states_dir.is_dir()}) or labels ({labels.is_file()})", file=sys.stderr)
            continue
        gt = load_gt(labels)
        pred = load_preds(states_dir)
        cm, n = confusion(gt, pred)
        overall += cm
        out = (run / "paper_metrics" / f"confusion_matrix_{ds}.csv") if args.save else None
        report(t, cm, n, out)
    report(f"OVERALL ({len(args.trackers)} trackers, {ds})", overall, int(overall.sum()), None)


if __name__ == "__main__":
    main()
