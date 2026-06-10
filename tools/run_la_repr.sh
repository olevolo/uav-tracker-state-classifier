#!/bin/bash
# HONEST representative LA validation — motion_bridge on a DIVERSE hard subset
# (vehicles + persons + small/erratic uav + birds + boat), incl the known erratic
# loser bird1_1, so the aggregate is not car-cherry-picked. clean (no FC scaffold)
# vs passive. UAV123 EXPLORATORY.
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval7_gated/csc/sglatrack/uav123/test
SEQS="bird1_2 uav8 uav4 uav2 group3_2 uav7 uav1_3 bird1_3 car9 person19_3 car1_s person21 bird1_1 car14 car1_3 boat8 uav6 car5"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
LOG=outputs/_logs/run_la_repr.log; : > "$LOG"
RUN="$ROOT/mb_repr"; rm -rf "$RUN"
$PY -u tools/run_with_csc.py --tracker sglatrack --dataset uav123 --split test \
  --csc_checkpoint "$CKPT" --calibration_prefix sglatrack_aerial_v2 --device cpu \
  --output_dir "$ROOT" --run_tag mb_repr --include_sequences $SEQS \
  --csc_mode control --policy_gated_redetect --redetect_action motion_bridge \
  --gate_preset combined --gate_vote_k 5 --redetect_arm_frames 3 --bridge_max_frames 30 >>"$LOG" 2>&1
$PY -u tools/la_smoke.py --run_dir "$RUN" --seqs $SEQS --tag mb_repr 2>/dev/null | tee -a "$LOG"
echo "ALLDONE"
