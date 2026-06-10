"""Smoke test for check_stage1_gate.py.

Run via:
    perl -e 'alarm 240; exec @ARGV' .venv/bin/python -u tools/_smoke_stage1_gate.py
"""
import sys
import time
from pathlib import Path

import torch
print(f"CUDA={torch.cuda.is_available()}", flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

N_STEPS = 3

print(f"[1/{N_STEPS}] Importing gate checker...", flush=True)
from tools.check_stage1_gate import evaluate_gate, print_report

print(f"[2/{N_STEPS}] Evaluating gate on csc_gru_v2 metrics...", flush=True)
metrics_path = PROJECT_ROOT / "outputs" / "csc_training" / "csc_gru_v2" / "val_metrics.json"
if not metrics_path.exists():
    print(f"  SKIP: metrics not found at {metrics_path}", flush=True)
    sys.exit(0)

results, n_failed = evaluate_gate(metrics_path)

print(f"[3/{N_STEPS}] Printing gate report...", flush=True)
print_report(results, metrics_path)

print(f"SMOKE PASS (n_auto_failed={n_failed}; FAIL results are expected — model collapsed)", flush=True)
