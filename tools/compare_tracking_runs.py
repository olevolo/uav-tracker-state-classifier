#!/usr/bin/env python
"""Compare two tracking-result prediction sets per-sequence + macro.

Reads metrics_summary.json + metrics_per_sequence.csv from two
``evaluate_tracking_results.py`` output dirs, prints a side-by-side delta
table, and writes ``compare_summary.json`` to the output dir.

Usage::

    python tools/compare_tracking_runs.py \
        --baseline outputs/fc_recover_v1/baseline_eval5_clamp \
        --candidate outputs/fc_recover_v1/full_uav123_with_detector_eval \
        --output outputs/fc_recover_v1/compare_with_detector \
        --label_baseline "V3 passive (csc_prod)" \
        --label_candidate "fc_recover + RT-DETR"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _read_per_seq(eval_dir: Path) -> dict[str, dict]:
    csv_path = eval_dir / "metrics_per_sequence.csv"
    if not csv_path.exists():
        return {}
    rows: dict[str, dict] = {}
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            name = r.get("sequence") or r.get("name") or r.get("seq")
            if not name:
                continue
            try:
                rows[name] = {
                    "auc": float(r.get("auc", "nan")),
                    "precision_20": float(r.get("precision_20", "nan")),
                    "n_frames": int(float(r.get("n_frames", 0))),
                }
            except (TypeError, ValueError):
                continue
    return rows


def _read_summary(eval_dir: Path) -> dict:
    sp = eval_dir / "metrics_summary.json"
    if not sp.exists():
        return {}
    return json.loads(sp.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Baseline eval dir.")
    ap.add_argument("--candidate", required=True, help="Candidate eval dir.")
    ap.add_argument("--output", required=True, help="Compare output dir.")
    ap.add_argument("--label_baseline", default="baseline")
    ap.add_argument("--label_candidate", default="candidate")
    ap.add_argument("--top_k_winners", type=int, default=10)
    ap.add_argument("--top_k_losers", type=int, default=10)
    args = ap.parse_args()

    base_dir = Path(args.baseline)
    cand_dir = Path(args.candidate)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_sum = _read_summary(base_dir)
    cand_sum = _read_summary(cand_dir)
    base_seq = _read_per_seq(base_dir)
    cand_seq = _read_per_seq(cand_dir)

    common = sorted(set(base_seq) & set(cand_seq))
    deltas: list[dict] = []
    for n in common:
        b, c = base_seq[n], cand_seq[n]
        d_auc = c["auc"] - b["auc"]
        d_prec = c["precision_20"] - b["precision_20"]
        deltas.append({
            "sequence": n,
            "baseline_auc": b["auc"], "candidate_auc": c["auc"],
            "delta_auc": d_auc,
            "baseline_precision_20": b["precision_20"],
            "candidate_precision_20": c["precision_20"],
            "delta_precision_20": d_prec,
            "n_frames": b["n_frames"],
        })
    deltas.sort(key=lambda x: x["delta_auc"], reverse=True)

    base_macro = base_sum.get("macro", {})
    cand_macro = cand_sum.get("macro", {})
    macro_delta = {
        "baseline_auc": float(base_macro.get("auc", float("nan"))),
        "candidate_auc": float(cand_macro.get("auc", float("nan"))),
        "delta_auc": float(cand_macro.get("auc", float("nan"))) - float(base_macro.get("auc", float("nan"))),
        "baseline_precision_20": float(base_macro.get("precision_20", float("nan"))),
        "candidate_precision_20": float(cand_macro.get("precision_20", float("nan"))),
        "delta_precision_20": float(cand_macro.get("precision_20", float("nan"))) - float(base_macro.get("precision_20", float("nan"))),
        "baseline_failures": int(base_sum.get("n_failures_total", 0)),
        "candidate_failures": int(cand_sum.get("n_failures_total", 0)),
        "delta_failures": int(cand_sum.get("n_failures_total", 0)) - int(base_sum.get("n_failures_total", 0)),
    }

    n = max(1, len(deltas))
    aggregate = {
        "n_sequences": len(deltas),
        "mean_delta_auc": sum(d["delta_auc"] for d in deltas) / n,
        "n_wins": sum(1 for d in deltas if d["delta_auc"] > 0.005),
        "n_losses": sum(1 for d in deltas if d["delta_auc"] < -0.005),
        "n_neutral": sum(1 for d in deltas if abs(d["delta_auc"]) <= 0.005),
        "max_gain": max((d["delta_auc"] for d in deltas), default=0.0),
        "max_loss": min((d["delta_auc"] for d in deltas), default=0.0),
        "label_baseline": args.label_baseline,
        "label_candidate": args.label_candidate,
    }

    summary = {
        "macro": macro_delta,
        "aggregate": aggregate,
        "top_winners": deltas[: args.top_k_winners],
        "top_losers": deltas[-args.top_k_losers :][::-1],
        "all_deltas": deltas,
    }
    (out_dir / "compare_summary.json").write_text(json.dumps(summary, indent=2))

    # Pretty print.
    label_b = args.label_baseline
    label_c = args.label_candidate
    print(f"\n=== {label_c} vs {label_b}  (UAV123/test) ===")
    print(f"macro AUC         : {macro_delta['baseline_auc']:.4f} -> "
          f"{macro_delta['candidate_auc']:.4f}  Δ={macro_delta['delta_auc']:+.4f}")
    print(f"macro Precision@20: {macro_delta['baseline_precision_20']:.4f} -> "
          f"{macro_delta['candidate_precision_20']:.4f}  Δ={macro_delta['delta_precision_20']:+.4f}")
    print(f"failures          : {macro_delta['baseline_failures']} -> "
          f"{macro_delta['candidate_failures']}  Δ={macro_delta['delta_failures']:+d}")
    print(f"sequences         : {aggregate['n_sequences']} "
          f"(wins>0.005={aggregate['n_wins']}, losses<-0.005={aggregate['n_losses']}, "
          f"neutral={aggregate['n_neutral']})")
    print(f"per-seq mean ΔAUC : {aggregate['mean_delta_auc']:+.4f}  "
          f"max+={aggregate['max_gain']:+.4f}  max-={aggregate['max_loss']:+.4f}")
    print()
    print("Top winners:")
    print(f"  {'sequence':<18}{'base':>8}{'cand':>8}{'Δ':>10}")
    for d in summary["top_winners"]:
        print(f"  {d['sequence']:<18}{d['baseline_auc']:>8.3f}"
              f"{d['candidate_auc']:>8.3f}{d['delta_auc']:>+10.4f}")
    print("Top losers:")
    for d in summary["top_losers"]:
        print(f"  {d['sequence']:<18}{d['baseline_auc']:>8.3f}"
              f"{d['candidate_auc']:>8.3f}{d['delta_auc']:>+10.4f}")
    print(f"\nreport: {out_dir / 'compare_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
