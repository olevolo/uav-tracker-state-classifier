#!/bin/bash
# Corrective rerun: AVTrack & ORTrack with csc_head gate (not scoremap2).
# AV/OR don't expose sm_* score-map features so scoremap2 gate never fires.
# Use csc_head gate (the forecast-head-based gate) per memory project_multitracker_control_enabled.md.
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/csc_prod/checkpoint_best.pth
LOG=outputs/_logs/paper_v2_fill_avor_fix.log
: > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Wait for paper_v2_fill to finish
say "waiting for paper_v2_fill to finish..."
for i in $(seq 1 120); do
  if grep -q "=== ALL DONE ===" outputs/_logs/paper_v2_fill.log 2>/dev/null; then
    say "paper_v2_fill done; starting AV/OR fix runs"
    break
  fi
  sleep 60
done

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4

# csc_head gate (forecast-head probability >= threshold) — works without sm_* features
GATE_FLAGS="--gate_preset csc_head --gate_lostaware 0.90 \
  --redetect_action sgla_redetect --sgla_redetect_factors 8,12,16 \
  --sgla_redetect_min_apce 0 --gated_freeze --no_runner_template_update"
FC_FLAGS="--fc_action hold_lastgood --fc_streak_frames 2"

run_combo () {
  local trk=$1 ds=$2 calib=$3
  local out=outputs/eval_paperv2/${trk}_${ds}_combo_csc_head
  say "[$trk/$ds] combo_csc_head → $out"
  $PY -u tools/run_with_csc.py \
    --tracker $trk --dataset $ds --split test \
    --csc_mode control --csc_checkpoint $CKPT --calibration_prefix $calib \
    --device cpu --output_dir $out --run_tag combo_csc_head \
    --policy_gated_redetect $GATE_FLAGS \
    --policy_fc_control $FC_FLAGS >> "$LOG" 2>&1 || say "FAIL $trk $ds"
}

say "=== 1/4 AVTrack UAV123 csc_head ==="
run_combo avtrack uav123 avtrack_aerial_v2
say "=== 2/4 AVTrack UAVTrack112 csc_head ==="
run_combo avtrack uavtrack112 avtrack_aerial_v2
say "=== 3/4 ORTrack UAV123 csc_head ==="
run_combo ortrack uav123 ortrack_aerial_v2
say "=== 4/4 ORTrack UAVTrack112 csc_head ==="
run_combo ortrack uavtrack112 ortrack_aerial_v2

say "=== ALL DONE ==="
