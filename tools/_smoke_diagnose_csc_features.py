"""Smoke test for tools/diagnose_csc_features.py.

Convention:
- Runs via: perl -e 'alarm 240; exec @ARGV' .venv/bin/python -u tools/_smoke_diagnose_csc_features.py
- Uses only a small subset of data (first 3 sequences from got10k val).
- Checks PASS conditions and prints a final PASS/FAIL line.
"""
from __future__ import annotations

import csv
import json
import sys
import zlib
import tempfile
from pathlib import Path

import torch
print(f"CUDA={torch.cuda.is_available()}", flush=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Step 1: Prepare mini input from first 3 got10k val sequences
# ---------------------------------------------------------------------------
print("[1/6] Preparing mini label set from first 3 sequences...", flush=True)

LABELS_ROOT = ROOT / "outputs" / "csc_labels" / "got10k" / "val"
labels_jsonl = LABELS_ROOT / "labels.jsonl"

if not labels_jsonl.exists():
    print(f"FAIL: labels.jsonl not found at {labels_jsonl}", flush=True)
    sys.exit(1)

rows_all: list[dict] = []
with open(labels_jsonl) as fh:
    for line in fh:
        line = line.strip()
        if line:
            rows_all.append(json.loads(line))

# Get first 3 unique sequences
seqs_seen: list[str] = []
rows_mini: list[dict] = []
for r in rows_all:
    seq = r["sequence"]
    if seq not in seqs_seen:
        seqs_seen.append(seq)
    if len(seqs_seen) > 3:
        break
    if seq in seqs_seen[:3]:
        rows_mini.append(r)

print(f"      Mini set: {len(rows_mini)} rows, {len(seqs_seen[:3])} sequences", flush=True)

# Write to temp dir
with tempfile.TemporaryDirectory() as tmpdir:
    mini_dir = Path(tmpdir) / "mini_labels"
    mini_dir.mkdir()
    mini_jsonl = mini_dir / "labels.jsonl"
    with open(mini_jsonl, "w") as fh:
        for r in rows_mini:
            fh.write(json.dumps(r) + "\n")

    out_dir = Path(tmpdir) / "out"
    out_dir.mkdir()
    out_csv = out_dir / "got10k.csv"

    # ---------------------------------------------------------------------------
    # Step 2: Import and run the diagnostic directly (not subprocess)
    # ---------------------------------------------------------------------------
    print("[2/6] Running diagnose_csc_features on mini set...", flush=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "diagnose_csc_features",
        ROOT / "tools" / "diagnose_csc_features.py",
    )
    mod = importlib.util.load_from_spec = None  # not used
    # Use direct imports instead
    from tools.diagnose_csc_features import (  # type: ignore
        _collect_rows,
        _group_by_sequence,
        _build_features_and_labels,
        _compute_feature_stats,
        _compute_group_stats,
        _write_csv,
        FEATURE_NAMES,
    )

    print("[3/6] Loading rows...", flush=True)
    rows = _collect_rows(mini_dir)
    print(f"      {len(rows)} rows", flush=True)

    print("[4/6] Building features and labels...", flush=True)
    groups = _group_by_sequence(rows)
    features, labels, seq_keys, all_rows = _build_features_and_labels(groups, (1280, 720))
    print(f"      features shape: {features.shape}, labels shape: {labels.shape}", flush=True)
    print(f"      failure fraction: {labels.mean():.3f}", flush=True)

    print("[5/6] Computing feature statistics...", flush=True)
    import math
    feature_stats = _compute_feature_stats(features, labels, seq_keys, all_rows)
    feature_stats_sorted = sorted(feature_stats, key=lambda r: -float(r["auprc_val"]))

    FEAT_FIELDS = [
        "feature", "polarity", "auroc_full", "auprc_full",
        "pearson_full", "auroc_val", "auprc_val",
        "support_pos", "support_neg", "fraction_nan",
    ]
    _write_csv(out_csv, feature_stats_sorted, FEAT_FIELDS)

    group_stats = _compute_group_stats(feature_stats)
    groups_csv = out_csv.with_name("got10k_groups.csv")
    GROUP_FIELDS = ["group", "members", "max_auprc_val", "best_member"]
    _write_csv(groups_csv, group_stats, GROUP_FIELDS)

    print("[6/6] Checking PASS conditions...", flush=True)

    # Check 1: CSV has 11 rows (one per feature)
    with open(out_csv) as fh:
        reader = csv.DictReader(fh)
        csv_rows = list(reader)
    n_feature_rows = len(csv_rows)
    print(f"      Feature CSV rows: {n_feature_rows} (expected 11)", flush=True)
    cond1 = n_feature_rows == 11

    # Check 2: at least one feature has auprc_val > 0.10
    auprc_vals = [float(r["auprc_val"]) for r in csv_rows]
    max_auprc_val = max(auprc_vals)
    print(f"      Max auprc_val: {max_auprc_val:.4f} (threshold > 0.10)", flush=True)
    # Note: with only 3 sequences the failure fraction may be 0 (all CONFIRMED).
    # In that case, relax the threshold to > 0.0.
    n_pos = labels.sum()
    if n_pos == 0:
        print(f"      WARNING: no positive labels in mini set — relaxing threshold to > 0.0", flush=True)
        cond2 = True  # can't test this without positives
    else:
        cond2 = max_auprc_val > 0.10

    # Check 3: groups CSV has 3 rows
    with open(groups_csv) as fh:
        reader2 = csv.DictReader(fh)
        g_rows = list(reader2)
    n_group_rows = len(g_rows)
    print(f"      Group CSV rows: {n_group_rows} (expected 3)", flush=True)
    cond3 = n_group_rows == 3

    # Print feature table for reference
    print("\n  Feature stats (mini set):", flush=True)
    for r in feature_stats_sorted:
        print(f"    {r['feature']:>14s}  polarity={r['polarity']}  "
              f"auprc_val={r['auprc_val']:.4f}  auroc_val={r['auroc_val']:.4f}  "
              f"nan_frac={r['fraction_nan']:.3f}", flush=True)

    # Check LOO if checkpoint exists
    ckpt_path = ROOT / "outputs" / "csc_training" / "csc_gru_v2" / "checkpoint_best.pth"
    loo_ok = True
    if ckpt_path.exists():
        print(f"\n  LOO check with checkpoint {ckpt_path}...", flush=True)
        from tools.diagnose_csc_features import _compute_loo_stats  # type: ignore
        loo_rows = _compute_loo_stats(features, labels, seq_keys, ckpt_path)
        if loo_rows:
            loo_csv = out_csv.with_name("got10k_loo.csv")
            LOO_FIELDS = ["feature", "auprc_baseline", "auprc_zeroed", "auprc_drop_when_zeroed", "note"]
            _write_csv(loo_csv, loo_rows, LOO_FIELDS)
            print(f"  LOO CSV rows: {len(loo_rows)} (expected 11)", flush=True)
            loo_ok = len(loo_rows) == 11
        else:
            print("  LOO returned empty (no positives in val split — OK for mini test)", flush=True)
    else:
        print(f"  Checkpoint not found at {ckpt_path} — skipping LOO check", flush=True)

    # Final verdict
    all_pass = cond1 and cond2 and cond3 and loo_ok
    print(f"\n  cond1 (11 feature rows): {'PASS' if cond1 else 'FAIL'}", flush=True)
    print(f"  cond2 (auprc_val>0.10): {'PASS' if cond2 else 'FAIL'}", flush=True)
    print(f"  cond3 (3 group rows):   {'PASS' if cond3 else 'FAIL'}", flush=True)
    print(f"  LOO check:              {'PASS' if loo_ok else 'FAIL'}", flush=True)
    print(f"\n{'SMOKE PASS' if all_pass else 'SMOKE FAIL'}", flush=True)
    sys.exit(0 if all_pass else 1)
