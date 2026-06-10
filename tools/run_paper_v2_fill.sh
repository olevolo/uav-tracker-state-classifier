#!/bin/bash
# Fill Table 3 + new MobileTrack row for paper v2.
# Runs sequentially to avoid CPU contention. ~3-4h total on Apple Silicon.
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/csc_prod/checkpoint_best.pth
LOG=outputs/_logs/paper_v2_fill.log
: > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4

GATE_FLAGS="--gate_preset scoremap2 --gate_top2ratio 0.45 --gate_entropy 4.3 \
  --redetect_action sgla_redetect --sgla_redetect_factors 8,12,16 \
  --sgla_redetect_min_apce 0 --gated_freeze --no_runner_template_update"
FC_FLAGS="--fc_action hold_lastgood --fc_streak_frames 2 --fc_gate_vote_k 2"

run_passive_frozen () {
  local trk=$1 ds=$2 calib=$3
  local out=outputs/eval_paperv2/${trk}_${ds}_passive_frozen
  say "[$trk/$ds] passive_frozen → $out"
  $PY -u tools/run_with_csc.py \
    --tracker $trk --dataset $ds --split test \
    --csc_mode passive --csc_checkpoint $CKPT --calibration_prefix $calib \
    --device cpu --output_dir $out --run_tag passive_frozen \
    --no_runner_template_update >> "$LOG" 2>&1 || say "FAIL $trk $ds passive"
}

run_combo () {
  local trk=$1 ds=$2 calib=$3
  local out=outputs/eval_paperv2/${trk}_${ds}_combo
  say "[$trk/$ds] combo → $out"
  $PY -u tools/run_with_csc.py \
    --tracker $trk --dataset $ds --split test \
    --csc_mode control --csc_checkpoint $CKPT --calibration_prefix $calib \
    --device cpu --output_dir $out --run_tag combo \
    --policy_gated_redetect $GATE_FLAGS \
    --policy_fc_control $FC_FLAGS >> "$LOG" 2>&1 || say "FAIL $trk $ds combo"
}

# 1. AVTrack UAV123 (paper Table 3 row 2, UAV123 columns)
say "=== 1/8 AVTrack UAV123 passive_frozen ==="
run_passive_frozen avtrack uav123 avtrack_aerial_v2
say "=== 2/8 AVTrack UAV123 combo ==="
run_combo avtrack uav123 avtrack_aerial_v2

# 2. ORTrack UAV123
say "=== 3/8 ORTrack UAV123 passive_frozen ==="
run_passive_frozen ortrack uav123 ortrack_aerial_v2
say "=== 4/8 ORTrack UAV123 combo ==="
run_combo ortrack uav123 ortrack_aerial_v2

# 3. AVTrack UAVTrack112
say "=== 5/8 AVTrack UAVTrack112 passive_frozen ==="
run_passive_frozen avtrack uavtrack112 avtrack_aerial_v2
say "=== 6/8 AVTrack UAVTrack112 combo ==="
run_combo avtrack uavtrack112 avtrack_aerial_v2

# 4. ORTrack UAVTrack112
say "=== 7/8 ORTrack UAVTrack112 passive_frozen ==="
run_passive_frozen ortrack uavtrack112 ortrack_aerial_v2
say "=== 8/8 ORTrack UAVTrack112 combo ==="
run_combo ortrack uavtrack112 ortrack_aerial_v2

say "=== ALL DONE ==="
