#!/bin/bash
# MobileTrack baselines on UAV123 + UAVTrack112 — runs AFTER paper_v2_fill finishes.
# Polls outputs/_logs/paper_v2_fill.log for "ALL DONE" before starting.
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
LOG=outputs/_logs/mobiletrack_baseline.log
: > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Wait for paper_v2_fill to complete (up to 6h)
say "waiting for paper_v2_fill to finish..."
for i in $(seq 1 360); do  # 360 * 60s = 6h max
  if grep -q "=== ALL DONE ===" outputs/_logs/paper_v2_fill.log 2>/dev/null; then
    say "paper_v2_fill complete; starting mobiletrack"
    break
  fi
  sleep 60
done

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4

say "=== 1/2 MobileTrack UAV123 baseline ==="
$PY -u tools/run_baseline.py \
  --tracker mobiletrack --dataset uav123 --split test \
  --device cpu --output_dir outputs/baselines_v4/mobiletrack \
  --skip_existing >> "$LOG" 2>&1 || say "FAIL mobiletrack uav123"

say "=== 2/2 MobileTrack UAVTrack112 baseline ==="
$PY -u tools/run_baseline.py \
  --tracker mobiletrack --dataset uavtrack112 --split test \
  --device cpu --output_dir outputs/baselines_v4/mobiletrack \
  --skip_existing >> "$LOG" 2>&1 || say "FAIL mobiletrack uavtrack112"

say "=== ALL DONE ==="
