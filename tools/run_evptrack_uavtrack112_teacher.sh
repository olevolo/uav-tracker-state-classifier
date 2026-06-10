#!/bin/bash
# Run EVPTrack baseline on UAVTrack112, then build labels + run teacher eval.
# Chained AFTER paper_v2_fill (CPU contention).
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
LOG=outputs/_logs/evptrack_uavtrack112_teacher.log
: > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Wait for paper_v2_fill (~3-4h)
say "waiting for paper_v2_fill..."
for i in $(seq 1 360); do
  if grep -q "=== ALL DONE ===" outputs/_logs/paper_v2_fill.log 2>/dev/null; then
    say "paper_v2_fill complete"
    break
  fi
  sleep 60
done

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4

say "=== EVPTrack UAVTrack112 baseline ==="
$PY -u tools/run_baseline.py \
  --tracker evptrack --dataset uavtrack112 --split test \
  --device cpu --output_dir outputs/baselines \
  --skip_existing >> "$LOG" 2>&1 || say "FAIL evptrack uavtrack112 baseline"

say "=== EVPTrack UAVTrack112 teacher labels ==="
$PY -u tools/build_scene_state_labels.py --tracker evptrack --dataset uavtrack112 \
  --split test --baseline_dir outputs/baselines \
  --output_dir outputs/eval/evptrack/uavtrack112/test/evptrack_r3_passive/labels_v3 \
  >> "$LOG" 2>&1 || say "FAIL evptrack uavtrack112 labels"

cp -r outputs/baselines/evptrack/uavtrack112/test/tracking_metrics \
   outputs/eval/evptrack/uavtrack112/test/evptrack_r3_passive/tracking_metrics 2>>"$LOG"

say "=== Re-run teacher eval ==="
$PY -u tools/offline_teacher_eval.py uav123 uavtrack112 >> "$LOG" 2>&1

say "=== ALL DONE ==="
