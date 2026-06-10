#!/bin/bash
# motion_bridge displacement-cap test on UAVTrack112 catastrophic-easy + winner seqs.
# Does --bridge_max_disp kill the car6_2 (-0.58) / tricycle1 (-0.55) runaway while
# keeping hard recoveries (hiker1 +0.44, couple +0.37, group4_1 +0.22)? Compare vs
# val_passive; val_frozen (no cap) had car6_2 -0.58.
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval8_cosine/csc/sglatrack/uavtrack112/test
CALIB=sglatrack_all_v2
SEQS="car6_2 tricycle1_2 tricycle1_1 hiker1 couple group4_1 truck car4"
COMBO="--csc_mode control --policy_gated_redetect --gate_preset scoremap2 --gate_top2ratio 0.30 --gate_entropy 4.0 \
 --redetect_action motion_bridge --redetect_arm_frames 3 --bridge_max_frames 30 --bridge_vel_ema 0.5 \
 --policy_fc_control --fc_action hold_lastgood --fc_streak_frames 2 --fc_gate_vote_k 2 \
 --gated_freeze --no_runner_template_update"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
for DISP in 6 10; do
  TAG="val_disp${DISP}"; RUN="$ROOT/$TAG"; rm -rf "$RUN"
  $PY -u tools/run_with_csc.py --tracker sglatrack --dataset uavtrack112 --split test \
    --csc_checkpoint "$CKPT" --calibration_prefix "$CALIB" --device cpu \
    --output_dir "$ROOT" --run_tag "$TAG" --include_sequences $SEQS $COMBO --bridge_max_disp $DISP \
    > outputs/_logs/run_${TAG}.log 2>&1 || { echo "FAIL $TAG"; continue; }
  echo "============ bridge_max_disp=$DISP ============"
  $PY tools/agg_full.py --dataset uavtrack112 --baseline "$ROOT/val_passive" --run_dir "$RUN" --tag "$TAG" 2>/dev/null \
    | sed -n '/====/,/TOP LOSERS/p'
done
echo "DISP_TEST_DONE"
