#!/bin/bash
# Auto-finalize MobileTrack: tracking_metrics + Teacher labels + rerun teacher eval
# Polls for UT112 baseline completion, then finalizes.
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
LOG=outputs/_logs/mobiletrack_finalize.log
: > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "waiting for MobileTrack UAVTrack112 baseline..."
for i in $(seq 1 240); do
  if grep -q "=== ALL DONE ===" outputs/_logs/mobiletrack_baseline.log 2>/dev/null; then
    say "mobiletrack baselines complete"
    break
  fi
  sleep 30
done

say "=== MobileTrack UAV123 tracking_metrics + labels ==="
$PY -u tools/evaluate_tracking_results.py --dataset uav123 --split test \
  --pred_dir outputs/baselines_v4/mobiletrack/mobiletrack/uav123/test/predictions \
  --telemetry_dir outputs/baselines_v4/mobiletrack/mobiletrack/uav123/test/telemetry \
  --output_dir outputs/baselines_v4/mobiletrack/mobiletrack/uav123/test/tracking_metrics \
  >> "$LOG" 2>&1

$PY -u tools/build_scene_state_labels.py --tracker mobiletrack --dataset uav123 --split test \
  --baseline_dir outputs/baselines_v4/mobiletrack \
  --output_dir outputs/eval/mobiletrack/uav123/test/mobiletrack_r3_passive/labels_v3 \
  >> "$LOG" 2>&1

cp -r outputs/baselines_v4/mobiletrack/mobiletrack/uav123/test/tracking_metrics \
   outputs/eval/mobiletrack/uav123/test/mobiletrack_r3_passive/tracking_metrics 2>>"$LOG" || true

say "=== MobileTrack UAVTrack112 tracking_metrics + labels ==="
$PY -u tools/evaluate_tracking_results.py --dataset uavtrack112 --split test \
  --pred_dir outputs/baselines_v4/mobiletrack/mobiletrack/uavtrack112/test/predictions \
  --telemetry_dir outputs/baselines_v4/mobiletrack/mobiletrack/uavtrack112/test/telemetry \
  --output_dir outputs/baselines_v4/mobiletrack/mobiletrack/uavtrack112/test/tracking_metrics \
  >> "$LOG" 2>&1

$PY -u tools/build_scene_state_labels.py --tracker mobiletrack --dataset uavtrack112 --split test \
  --baseline_dir outputs/baselines_v4/mobiletrack \
  --output_dir outputs/eval/mobiletrack/uavtrack112/test/mobiletrack_r3_passive/labels_v3 \
  >> "$LOG" 2>&1

cp -r outputs/baselines_v4/mobiletrack/mobiletrack/uavtrack112/test/tracking_metrics \
   outputs/eval/mobiletrack/uavtrack112/test/mobiletrack_r3_passive/tracking_metrics 2>>"$LOG" || true

say "=== Re-run Teacher eval ==="
$PY -u tools/offline_teacher_eval.py uav123 uavtrack112 >> "$LOG" 2>&1

say "=== ALL DONE ==="
echo ""
say "Updated Teacher tables:"
cat outputs/eval/_offline_teacher_eval_uav123.md | head -20 | tee -a "$LOG"
echo ""
cat outputs/eval/_offline_teacher_eval_uavtrack112.md | head -20 | tee -a "$LOG"
