#!/bin/bash
# FC challenge-and-switch: passive-vs-challenge + streak sweep on the FC-heavy
# UAV123 subset (FC>=15 frames under the prod model, measured from eval5_clamp
# passive states). EXPLORATORY offline SOT — NOT a final UAV123 result.
#
# Fully controlled: passive and every challenge run use the IDENTICAL harness +
# flags (same prod ckpt, same calibration, frozen template via
# --recovery_update_window 0). Only --policy_fc_challenge / --fc_challenge_streak
# differ. So any AUC/FCR/FCD delta is attributable to the controller alone.
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval_fc_challenge
SEQS="car6_4 wakeboard10 boat8 car17 car7 car2_s wakeboard6 car1_s bike2 truck2"
COMMON="--tracker sglatrack --dataset uav123 --split test \
  --csc_checkpoint $CKPT --calibration_prefix sglatrack_all_v2 --device cpu \
  --include_sequences $SEQS --recovery_update_window 0 --output_dir $ROOT"
# Challenge knobs held fixed across the sweep; only --fc_challenge_streak varies.
CHAL="--csc_mode control --policy_fc_challenge \
  --fc_challenge_confirm 3 --fc_challenge_abort_window 5 --fc_challenge_max_frames 10 \
  --fc_challenge_sim_margin 0.05 --fc_challenge_apce_keep 0.6 \
  --sgla_redetect_factors 8,12,16 --sgla_redetect_grid 0"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
mkdir -p "$ROOT" outputs/_logs
LOG=outputs/_logs/fc_challenge_sweep.log; : > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

run(){ local TAG="$1"; shift; rm -rf "$ROOT/$TAG"
  say ">>> [$TAG] $*"
  $PY -u tools/run_with_csc.py $COMMON --run_tag "$TAG" "$@" >>"$LOG" 2>&1 \
    || { say "  FAIL $TAG"; return 1; }
  say "  done $TAG"; }

say "===== FC challenge sweep ($SEQS) ====="
run passive   --csc_mode passive
run chal_s2   $CHAL --fc_challenge_streak 2
run chal_s3   $CHAL --fc_challenge_streak 3
run chal_s5   $CHAL --fc_challenge_streak 5
say "===== sweep DONE ====="
echo "FC_CHALLENGE_SWEEP_DONE" >> "$LOG"