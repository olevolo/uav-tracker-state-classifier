"""Smoke test for audit_visualizer.py.

Run via:
    perl -e 'alarm 240; exec @ARGV' .venv/bin/python -u tools/_smoke_audit_viz.py
"""
import sys
import time
from pathlib import Path

import torch
print(f"CUDA={torch.cuda.is_available()}", flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

N_STEPS = 3

print(f"[1/{N_STEPS}] Importing audit_visualizer...", flush=True)
from tools.audit_visualizer import visualize

print(f"[2/{N_STEPS}] Running visualizer (5 seqs × 4 frames)...", flush=True)
pred_dir = PROJECT_ROOT / "outputs" / "baselines" / "sglatrack" / "got10k" / "val" / "predictions"
labels_dir = PROJECT_ROOT / "outputs" / "csc_labels" / "got10k"
out_path = PROJECT_ROOT / "outputs" / "_smoke_audit" / "got10k_audit.png"

import os
data_root = Path(os.environ.get("UAV_DATA_ROOT", str(Path.home() / "uav-tracker-data")))

n_panels = visualize(
    pred_dir=pred_dir,
    labels_dir=labels_dir,
    dataset="got10k",
    split="val",
    n_seqs=5,
    frames_per_seq=4,
    hardest=True,
    out_path=out_path,
    data_root=data_root,
)

print(f"[3/{N_STEPS}] Checking output...", flush=True)
assert out_path.exists(), f"PNG not written: {out_path}"
assert n_panels > 0, f"0 panels rendered — check labels/predictions dirs"

print(f"SMOKE PASS — {n_panels} panels at {out_path}", flush=True)
