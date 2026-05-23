"""Smoke test for profile_pipeline.py tooling.

Runs 1 init + 1 track step + 1 CSC forward + flop counting.
Does NOT run the 200-frame benchmark.

Pass criteria (checked at the end of this script):
- outputs/profile/sglatrack_smoke.json exists and parses
- tracker.gflops_per_frame > 0
- csc.gflops_per_frame > 0
- total.mean_fps < tracker_stage.mean_fps  (CSC adds latency)
- count_csc_flops(CSCGRU, window=16, feature_dim=11) is ~4x larger
  than count_csc_flops(CSCGRU, window=4, feature_dim=11)  (monotonic in window)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Make project importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "salrtd" / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import torch

print(f"CUDA={torch.cuda.is_available()}", flush=True)
print("[1/7] imports OK", flush=True)

# ---------------------------------------------------------------------------
# 1. Import runtime_metrics new helpers
# ---------------------------------------------------------------------------
from csc_lib.eval.custom_metrics.runtime_metrics import (
    count_csc_flops,
    summarise_latencies,
    combined_pipeline_stats,
)
print("[2/7] runtime_metrics helpers imported", flush=True)

# ---------------------------------------------------------------------------
# 2. Build a tiny CSCGRU (untrained, default config) for flop counting
# ---------------------------------------------------------------------------
from csc_lib.csc.config import CSCModelConfig
from csc_lib.csc.model import CSCGRU

cfg = CSCModelConfig(feature_dim=11, hidden_dim=64, num_layers=1, dropout=0.0)
model_gru = CSCGRU(cfg)
model_gru.eval()

flops_w4 = count_csc_flops(model_gru, window=4, feature_dim=11)
flops_w16 = count_csc_flops(model_gru, window=16, feature_dim=11)
print(f"[3/7] CSCGRU FLOPs: window=4 → {flops_w4:,d}  window=16 → {flops_w16:,d}", flush=True)

assert flops_w16 > flops_w4, (
    f"FAIL: FLOPs should be monotonic in window, but w=16 ({flops_w16}) <= w=4 ({flops_w4})"
)
ratio = flops_w16 / flops_w4
print(f"       ratio w16/w4 = {ratio:.2f}  (expect ~4x)", flush=True)

# ---------------------------------------------------------------------------
# 3. Load SGLATrack and run 1 init + 1 update step
# ---------------------------------------------------------------------------
from uav_tracker.trackers import sglatrack  # noqa: F401
from uav_tracker.registry import TRACKERS

print("[4/7] loading SGLATrack...", flush=True)
tracker = TRACKERS.build("sglatrack", device="cpu")

# Build a synthetic 360×240 BGR frame (3-channel uint8)
import numpy as np

H, W = 240, 360
dummy_frame = np.zeros((H, W, 3), dtype=np.uint8)
# Small bbox in the centre (x, y, w, h)
from uav_tracker.types import BBox
init_bbox = BBox(x=160, y=100, w=40, h=30)

t0 = time.perf_counter()
tracker.init(dummy_frame, init_bbox)
init_ms = (time.perf_counter() - t0) * 1000.0

t0 = time.perf_counter()
state = tracker.update(dummy_frame)
update_ms = (time.perf_counter() - t0) * 1000.0

print(f"       SGLATrack init={init_ms:.1f} ms  update={update_ms:.1f} ms", flush=True)
print("[5/7] tracker step done", flush=True)

# SGLATrack _FLOPS_PER_UPDATE
from uav_tracker.trackers import sglatrack as _sgla_mod
tracker_flops_raw = getattr(_sgla_mod, "_FLOPS_PER_UPDATE", None)
assert tracker_flops_raw is not None, "FAIL: sglatrack does not declare _FLOPS_PER_UPDATE"
tracker_gflops = tracker_flops_raw / 1e9
print(f"       tracker_gflops_per_frame = {tracker_gflops:.4f}", flush=True)

# ---------------------------------------------------------------------------
# 4. Run a minimal CSC forward (untrained model, synthetic telemetry)
# ---------------------------------------------------------------------------
from csc_lib.csc.features import FEATURE_DIM
WINDOW = 16  # matching default cfg window_size

# Build a single synthetic window (B=1, T=WINDOW, F=FEATURE_DIM)
x = torch.zeros(1, WINDOW, FEATURE_DIM)
with torch.no_grad():
    t0 = time.perf_counter()
    _ = model_gru(x)
    csc_forward_ms = (time.perf_counter() - t0) * 1000.0

csc_gflops = count_csc_flops(model_gru, window=WINDOW, feature_dim=FEATURE_DIM) / 1e9
print(f"[6/7] CSC forward: {csc_forward_ms:.3f} ms  csc_gflops = {csc_gflops:.6f}", flush=True)

# ---------------------------------------------------------------------------
# 5. summarise_latencies + combined_pipeline_stats smoke
# ---------------------------------------------------------------------------
tracker_ms_list = [update_ms]
csc_ms_list = [csc_forward_ms]

stats = combined_pipeline_stats(tracker_ms_list, csc_ms_list)
assert "tracker" in stats and "csc" in stats and "total" in stats, "FAIL: combined_pipeline_stats keys"

total_mean_fps = stats["total"]["mean_fps"]
tracker_mean_fps = stats["tracker"]["mean_fps"]

print(f"       total mean_fps={total_mean_fps:.1f}  tracker_only mean_fps={tracker_mean_fps:.1f}", flush=True)

# ---------------------------------------------------------------------------
# 6. Write smoke output JSON
# ---------------------------------------------------------------------------
output_path = PROJECT_ROOT / "outputs" / "profile" / "sglatrack_smoke.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

result = {
    "tracker": "sglatrack",
    "csc_checkpoint": None,
    "device": "cpu",
    "n_warmup": 0,
    "n_frames_timed": 1,
    "tracker_stage": {
        **summarise_latencies(tracker_ms_list),
        "gflops_per_frame": tracker_gflops,
        "note": None,
    },
    "csc": {
        **summarise_latencies(csc_ms_list),
        "gflops_per_frame": csc_gflops,
        "params": sum(p.numel() for p in model_gru.parameters()),
    },
    "total": stats["total"],
    "csc_overhead_pct_runtime": 100.0 * csc_forward_ms / (update_ms + csc_forward_ms),
    "csc_overhead_pct_gflops": 100.0 * csc_gflops / (tracker_gflops + csc_gflops),
    "flop_monotonicity_check": {
        "flops_w4": flops_w4,
        "flops_w16": flops_w16,
        "ratio_w16_w4": ratio,
    },
}

with open(output_path, "w") as fh:
    json.dump(result, fh, indent=2)

print(f"       wrote {output_path}", flush=True)

# ---------------------------------------------------------------------------
# 7. Assert pass criteria
# ---------------------------------------------------------------------------
with open(output_path) as fh:
    parsed = json.load(fh)

assert parsed["tracker_stage"]["gflops_per_frame"] > 0, \
    "FAIL: tracker_stage.gflops_per_frame must be > 0"
assert parsed["csc"]["gflops_per_frame"] > 0, \
    "FAIL: csc.gflops_per_frame must be > 0"
assert parsed["total"]["mean_fps"] < parsed["tracker_stage"]["mean_fps"], \
    (f"FAIL: total.mean_fps ({parsed['total']['mean_fps']:.1f}) must be < "
     f"tracker_stage.mean_fps ({parsed['tracker_stage']['mean_fps']:.1f})")
assert flops_w16 > flops_w4, "FAIL: flop monotonicity"

print("[7/7] ALL PASS", flush=True)
print(f"\n=== SMOKE RESULTS ===", flush=True)
print(f"tracker_gflops_per_frame : {tracker_gflops:.4f}", flush=True)
print(f"csc_gflops_per_frame     : {csc_gflops:.8f}", flush=True)
print(f"csc_overhead_pct_runtime : {result['csc_overhead_pct_runtime']:.2f}%", flush=True)
print(f"csc_overhead_pct_gflops  : {result['csc_overhead_pct_gflops']:.4f}%", flush=True)
print(f"total_mean_fps           : {total_mean_fps:.1f}", flush=True)
print(f"flops_w4={flops_w4:,d}  flops_w16={flops_w16:,d}  ratio={ratio:.2f}", flush=True)
print(f"output: {output_path}", flush=True)
