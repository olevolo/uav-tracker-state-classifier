"""Per-sequence breakdown: where does FC have higher/lower scale_smoothness vs CC?

Aim: understand if the negative Cohen's d (-0.65) is uniform across sequences
or driven by specific cases. Output top sequences in each direction.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from csc_lib.csc.config import CSCFeatureConfig
from csc_lib.csc.features import build_sequence_features_v2

FC_LABEL = 3
CC_LABEL = 0

# V2 slot indices
SLOT_SCALE_SMOOTH = 14
SLOT_ASPECT_INST  = 15
SLOT_LOG_AREA     = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default="outputs/csc_labels/sglatrack/v3fix_combined")
    ap.add_argument("--out", default="outputs/v3fix_diag/per_sequence_breakdown.json")
    ap.add_argument("--top-k", type=int, default=15)
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        print(f"FATAL: labels dir not found: {labels_dir}")
        sys.exit(2)

    # group rows by (dataset, sequence)
    seq_rows = defaultdict(list)
    for sub in sorted(labels_dir.iterdir()):
        jsonl = sub / "labels.jsonl"
        if not jsonl.exists():
            continue
        with jsonl.open() as f:
            for line in f:
                r = json.loads(line)
                key = (r.get("dataset", sub.name), r.get("sequence", "?"))
                seq_rows[key].append(r)

    print(f"loaded {len(seq_rows)} sequences")

    feat_cfg = CSCFeatureConfig()
    feat_cfg.feature_version = "v2"

    per_seq_stats = []
    for (ds, seq), rows in tqdm(sorted(seq_rows.items()), desc="seq"):
        rows.sort(key=lambda r: r.get("frame_idx", 0))
        feats = build_sequence_features_v2(rows, image_size=(1280, 720), cfg=feat_cfg)
        labels = np.array([r.get("derived_state", -1) for r in rows], dtype=np.int32)

        fc_mask = labels == FC_LABEL
        cc_mask = labels == CC_LABEL
        n_fc = int(fc_mask.sum())
        n_cc = int(cc_mask.sum())

        if n_fc < 10 or n_cc < 10:
            continue  # skip sequences with too few of either

        ss_fc = float(feats[fc_mask, SLOT_SCALE_SMOOTH].mean())
        ss_cc = float(feats[cc_mask, SLOT_SCALE_SMOOTH].mean())
        ai_fc = float(feats[fc_mask, SLOT_ASPECT_INST].mean())
        ai_cc = float(feats[cc_mask, SLOT_ASPECT_INST].mean())
        la_fc = float(feats[fc_mask, SLOT_LOG_AREA].mean())
        la_cc = float(feats[cc_mask, SLOT_LOG_AREA].mean())

        per_seq_stats.append({
            "dataset": ds, "sequence": seq,
            "n_fc": n_fc, "n_cc": n_cc,
            "ss_fc": ss_fc, "ss_cc": ss_cc, "ss_diff": ss_fc - ss_cc,
            "ai_fc": ai_fc, "ai_cc": ai_cc, "ai_diff": ai_fc - ai_cc,
            "la_fc": la_fc, "la_cc": la_cc, "la_diff": la_fc - la_cc,
        })

    print(f"\nsequences with both FC and CC: {len(per_seq_stats)}")
    AERIAL = {"dtb70", "uavdt_sot", "visdrone_sot", "uavtrack112"}
    aer = [s for s in per_seq_stats if s["dataset"] in AERIAL]
    lasot = [s for s in per_seq_stats if s["dataset"] == "lasot"]
    other = [s for s in per_seq_stats if s["dataset"] not in AERIAL and s["dataset"] != "lasot"]
    print(f"  aerial: {len(aer)}  lasot: {len(lasot)}  other: {len(other)}")

    # Sort by ss_diff
    sorted_by_ss = sorted(per_seq_stats, key=lambda s: s["ss_diff"])

    print("\n" + "=" * 110)
    print(f"TOP {args.top_k} sequences where FC scale_smoothness is LOWER than CC (FC = stuck):")
    print(f"{'dataset':<14}{'sequence':<28}{'n_fc':>6}{'n_cc':>6}{'ss_fc':>10}{'ss_cc':>10}{'ss_diff':>10}{'la_diff':>10}")
    print("-" * 110)
    for s in sorted_by_ss[:args.top_k]:
        print(f"{s['dataset']:<14}{s['sequence']:<28}{s['n_fc']:>6}{s['n_cc']:>6}{s['ss_fc']:>10.4f}{s['ss_cc']:>10.4f}{s['ss_diff']:>10.4f}{s['la_diff']:>10.3f}")

    print("\n" + "=" * 110)
    print(f"TOP {args.top_k} sequences where FC scale_smoothness is HIGHER than CC (FC = chaotic):")
    print(f"{'dataset':<14}{'sequence':<28}{'n_fc':>6}{'n_cc':>6}{'ss_fc':>10}{'ss_cc':>10}{'ss_diff':>10}{'la_diff':>10}")
    print("-" * 110)
    for s in reversed(sorted_by_ss[-args.top_k:]):
        print(f"{s['dataset']:<14}{s['sequence']:<28}{s['n_fc']:>6}{s['n_cc']:>6}{s['ss_fc']:>10.4f}{s['ss_cc']:>10.4f}{s['ss_diff']:>10.4f}{s['la_diff']:>10.3f}")

    # Aggregate stats
    diffs = [s["ss_diff"] for s in per_seq_stats]
    n_neg = sum(1 for d in diffs if d < 0)
    n_pos = sum(1 for d in diffs if d > 0)
    print(f"\nss_diff per sequence:  negative={n_neg}/{len(diffs)} ({100*n_neg/len(diffs):.1f}%)  positive={n_pos}")

    diffs_a = [s["ai_diff"] for s in per_seq_stats]
    n_neg_a = sum(1 for d in diffs_a if d < 0)
    print(f"ai_diff per sequence:  negative={n_neg_a}/{len(diffs_a)} ({100*n_neg_a/len(diffs_a):.1f}%)")

    # Per-dataset breakdown
    print("\nPer-dataset breakdown of ss_diff sign:")
    for label, group in [("aerial", aer), ("lasot", lasot), ("other", other)]:
        if not group:
            continue
        gd = [s["ss_diff"] for s in group]
        gn = sum(1 for d in gd if d < 0)
        gmean = float(np.mean(gd))
        print(f"  {label:<8}: n={len(group):>4}  ss_diff_negative={gn:>4} ({100*gn/len(group):.1f}%)  mean_diff={gmean:+.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(per_seq_stats, indent=2))
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
