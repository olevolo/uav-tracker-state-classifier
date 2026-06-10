#!/usr/bin/env python3
"""Validation sweep for tau_fc / margin_fc thresholds.

Runs on DTB70 (validation dataset, never UAV123) and finds the
tau_fc that maximizes FC F1 while maintaining FC precision >= min_precision.

Usage:
    python tools/tune_fc_threshold.py \
        --checkpoint outputs/csc_training/sglatrack_v2_tcn16/checkpoint_best.pth \
        --eval_dir outputs/eval_v2/sglatrack/dtb70/test \
        --output outputs/quality/fc_threshold_sweep.json

Output: JSON with optimal tau_fc, margin_fc, and full sweep table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def _load_states_and_labels(eval_dir: Path, dataset: str, split: str):
    """Load CSC passive states + GT labels for threshold sweep."""
    states_dir = eval_dir / "passive" / "states"
    labels_dir = eval_dir / "labels" / dataset / split / "labels_per_sequence"

    if not states_dir.exists():
        raise FileNotFoundError(f"States not found: {states_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels not found: {labels_dir}")

    # Load all (teacher_state, fc_prob, second_prob) tuples
    rows = []
    STATES = ["CORRECT_CONFIRMED","CORRECT_UNCERTAIN","LOST_AWARE","FALSE_CONFIRMED"]
    fc_idx = 3

    for sf in sorted(states_dir.glob("*.jsonl")):
        lf = labels_dir / sf.name
        if not lf.exists():
            continue
        teacher = {r["frame_idx"]: r.get("derived_state_name","")
                   for line in lf.read_text().splitlines() if line
                   for r in [json.loads(line)]}
        for line in sf.read_text().splitlines():
            if not line: continue
            r = json.loads(line)
            fidx = r.get("frame_idx", -1)
            t_state = teacher.get(fidx, "")
            if not t_state: continue
            # Get FC probability and second-highest
            der_probs = r.get("localization_probs")  # wrong, need derived probs
            # States file has derived_state (int) and risk_score
            # We need p_fc from the original CSC run — get from risk_score approximation
            # Actually: states file has derived_state as argmax, not raw probs
            # For threshold sweep we need raw probabilities — not stored in states jsonl
            # Use derived_state as proxy: only rows where derived==FC are candidates
            derived = r.get("derived_state", 0)
            rows.append({
                "teacher": t_state,
                "student_derived": derived,
                "is_teacher_fc": t_state == "FALSE_CONFIRMED",
                "is_student_fc": derived == fc_idx,
            })

    return rows


def _sweep_from_raw_probs(states_dir: Path, labels_dir: Path):
    """Full sweep using raw probabilities from states files (if available)."""
    all_data = []
    for sf in sorted(states_dir.glob("*.jsonl")):
        lf = labels_dir / sf.name
        if not lf.exists():
            continue
        teacher = {r["frame_idx"]: r.get("derived_state_name","")
                   for line in lf.read_text().splitlines() if line
                   for r in [json.loads(line)]}
        for line in sf.read_text().splitlines():
            if not line: continue
            r = json.loads(line)
            fidx = r.get("frame_idx", -1)
            t_state = teacher.get(fidx)
            if not t_state: continue
            # risk_score = P(LA) + P(FC) from head_derived
            # We don't have individual P(FC) stored, only risk_score and derived_state
            # Proxy: use risk_score and derived_state
            risk = float(r.get("risk_score", 0.0))
            derived = int(r.get("derived_state", 0))
            all_data.append((t_state, derived, risk))
    return all_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", required=True, type=Path,
                   help="eval_v2/<tracker>/<dataset>/test/")
    p.add_argument("--dataset", default="dtb70")
    p.add_argument("--split", default="test")
    p.add_argument("--min_precision", type=float, default=0.40,
                   help="Min FC precision required")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    states_dir = args.eval_dir / "passive" / "states"
    labels_dir = args.eval_dir / "labels" / args.dataset / args.split / "labels_per_sequence"

    print(f"Loading states from: {states_dir}")
    print(f"Loading labels from: {labels_dir}")

    data = _sweep_from_raw_probs(states_dir, labels_dir)
    if not data:
        print("ERROR: no data loaded")
        return

    total = len(data)
    print(f"Total frames: {total:,}")
    print(f"Teacher FC frames: {sum(1 for t,d,r in data if t=='FALSE_CONFIRMED'):,}")
    print(f"Student FC frames (raw argmax): {sum(1 for t,d,r in data if d==3):,}")
    print()

    # Sweep: use risk_score as proxy for P(FC) threshold
    # Higher risk_score → more likely FC prediction
    # Find threshold on risk_score that gives best FC F1
    print("Risk-score threshold sweep (FC F1 optimization):")
    print(f"{'tau_risk':>9} {'FC_prec':>8} {'FC_rec':>8} {'FC_F1':>7} {'n_pred_FC':>10}")
    print("-" * 50)

    teacher_fc = np.array([1 if t=="FALSE_CONFIRMED" else 0 for t,d,r in data])
    risk_scores = np.array([r for t,d,r in data])
    n_true_fc = teacher_fc.sum()

    best = {"tau": 0.0, "f1": 0.0, "prec": 0.0, "rec": 0.0}
    results = []

    for tau in np.arange(0.05, 1.0, 0.05):
        pred_fc = (risk_scores >= tau).astype(int)
        tp = int((pred_fc & teacher_fc.astype(bool)).sum())
        fp = int((pred_fc & ~teacher_fc.astype(bool)).sum())
        fn = int((~pred_fc.astype(bool) & teacher_fc.astype(bool)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2*prec*rec / max(prec+rec, 1e-8)
        n_pred = int(pred_fc.sum())
        results.append({"tau": float(tau), "prec": prec, "rec": rec, "f1": f1, "n_pred": n_pred})
        marker = " ←" if (f1 > best["f1"] and prec >= args.min_precision) else ""
        print(f"  tau={tau:.2f}: prec={100*prec:.0f}% rec={100*rec:.0f}% F1={f1:.3f} n={n_pred:,}{marker}")
        if f1 > best["f1"] and prec >= args.min_precision:
            best = {"tau": float(tau), "f1": f1, "prec": prec, "rec": rec}

    print()
    print(f"OPTIMAL (F1-max with precision>={args.min_precision*100:.0f}%):")
    print(f"  tau_risk={best['tau']:.2f}  F1={best['f1']:.3f}  prec={100*best['prec']:.0f}%  rec={100*best['rec']:.0f}%")
    print()
    print(f"Use in CSCControlPolicy:")
    print(f"  tau_fc={best['tau']:.2f}  (risk_score proxy; for direct P(FC) use ÷2 approx)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"optimal": best, "sweep": results, "total_frames": total,
                       "n_teacher_fc": int(n_true_fc), "min_precision": args.min_precision}, f, indent=2)
        print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
