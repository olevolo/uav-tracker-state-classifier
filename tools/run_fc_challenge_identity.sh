#!/bin/bash
# FC challenge — identity-ranked + global-grid redetect variant. Tests whether
# global search (grid) + ranking candidates by IDENTITY (sim_to_init, not the
# sharp-peak quality) lets the controller actually SWITCH on true-FC, and
# whether those switches help or hurt. EXPLORATORY offline SOT.
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval_fc_challenge
SEQS="car6_4 wakeboard10 boat8 car17 car7 car2_s wakeboard6 car1_s bike2 truck2"
COMMON="--tracker sglatrack --dataset uav123 --split test --csc_checkpoint $CKPT \
  --calibration_prefix sglatrack_all_v2 --device cpu --include_sequences $SEQS \
  --recovery_update_window 0 --output_dir $ROOT --csc_mode control --policy_fc_challenge \
  --fc_challenge_streak 3 --fc_challenge_abort_window 5 --fc_challenge_max_frames 10 \
  --fc_challenge_sim_margin 0.05 --sgla_redetect_factors 8,12,16 \
  --sgla_redetect_rank_by identity --sgla_redetect_grid 2"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
LOG=outputs/_logs/fc_challenge_identity.log; : > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
run(){ local TAG="$1"; shift; rm -rf "$ROOT/$TAG"
  say ">>> [$TAG] $*"
  $PY -u tools/run_with_csc.py $COMMON --run_tag "$TAG" "$@" >>"$LOG" 2>&1 \
    || { say "  FAIL $TAG"; return 1; }
  say "  done $TAG"; }

say "===== FC challenge identity+grid variants ====="
run chal_id     --fc_challenge_apce_keep 0.3 --fc_challenge_confirm 3
run chal_id_c2  --fc_challenge_apce_keep 0.3 --fc_challenge_confirm 2
say "===== DONE ====="
echo "IDENTITY_DONE" >> "$LOG"