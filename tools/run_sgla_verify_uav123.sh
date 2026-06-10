#!/bin/bash
# sgla_redetect + V4 VERIFIER (sim_to_init gate) on UAV123 hard set. Does the identity
# floor reject the distractor jumps (bird1_1/car11) while keeping the wins (uav2 +0.46,
# group3_2, person14_1)? Two thresholds. Compare to mb/sgla(unverified) already in eval9_sgla.
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval9_sgla/csc/sglatrack/uav123/test
SEQS="group3_2 car1_s person14_1 car11 bird1_1 uav2 uav6 car6_2"
GATE="--csc_mode control --policy_gated_redetect --gate_preset scoremap2 --gate_top2ratio 0.45 --gate_entropy 4.3 --redetect_arm_frames 3 --gated_freeze --no_runner_template_update --redetect_action sgla_redetect --sgla_redetect_factors 8,12,16"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
LOG=outputs/_logs/run_sgla_verify.log; : > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
run_cfg(){ local TAG="$1"; shift; local RUN="$ROOT/$TAG"; rm -rf "$RUN"
  say ">>> [$TAG] $*"
  $PY -u tools/run_with_csc.py --tracker sglatrack --dataset uav123 --split test \
    --csc_checkpoint "$CKPT" --calibration_prefix sglatrack_all_v2 --device cpu \
    --output_dir "$ROOT" --run_tag "$TAG" --include_sequences $SEQS $GATE "$@" >>"$LOG" 2>&1 || { say "FAIL $TAG"; return 1; }
  say "--- [$TAG] ΔAUC ---"
  $PY -u tools/la_smoke.py --run_dir "$RUN" --seqs $SEQS --tag "$TAG" 2>/dev/null | tee -a "$LOG"; }
say "===== sgla VERIFIER sweep ($SEQS) ====="
run_cfg sgla_v50 --sgla_redetect_min_sim 0.5
run_cfg sgla_v60 --sgla_redetect_min_sim 0.6
say "===== DONE ====="; echo "SGLA_VERIFY_DONE" >> "$LOG"
