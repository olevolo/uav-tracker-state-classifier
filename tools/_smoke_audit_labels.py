"""Smoke test for audit_label_distribution.py.

Run via:
    perl -e 'alarm 240; exec @ARGV' .venv/bin/python -u tools/_smoke_audit_labels.py
"""
import sys
import time
from pathlib import Path

import torch
print(f"CUDA={torch.cuda.is_available()}", flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

N_STEPS = 3

print(f"[1/{N_STEPS}] Importing audit_label_distribution...", flush=True)
from tools.audit_label_distribution import audit

print(f"[2/{N_STEPS}] Running audit on got10k labels...", flush=True)
labels_dir = PROJECT_ROOT / "outputs" / "csc_labels" / "got10k"
out_dir = PROJECT_ROOT / "outputs" / "_smoke_audit"
out_path = out_dir / "got10k_dist.csv"

exit_code = audit(labels_dir, out_path)
print(f"[2/{N_STEPS}] audit() returned exit_code={exit_code}", flush=True)
# exit_code == 1 is acceptable (imbalanced labels) — just no crash
assert out_path.exists(), f"CSV not written: {out_path}"

summary_path = out_path.with_stem(out_path.stem + "_summary").with_suffix(".json")
assert summary_path.exists(), f"Summary JSON not written: {summary_path}"

print(f"[3/{N_STEPS}] Checking output files...", flush=True)
import json
with open(summary_path) as f:
    s = json.load(f)
print(f"  total_frames={s['total_frames']}, n_sequences={s['n_sequences']}", flush=True)

print(f"SMOKE PASS (exit_code={exit_code} — 0=OK, 1=imbalanced but not a crash)", flush=True)
