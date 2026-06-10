#!/bin/bash
# ============================================================================
# Isolate the control-mode TEMPLATE-FREEZE easy-scene regression.
# car6_5 regressed -0.1086 in ALL FC configs incl freeze_only, despite FC%=0 →
# the culprit is freezing the template on CSC-LA frames (should_skip_template_update).
#   mb_sm2_b   : scoremap2 + motion_bridge, DEFAULT freeze (freeze on LA+FC)
#   mb_sm2_ffo : + --policy_freeze_fc_only (freeze ONLY on FC, never on LA)
# If car6_5/car6_2 recover under _ffo while car9/group3_2/car1_s wins survive,
# freeze_fc_only is the config for the full-UAV123 run. EXPLORATORY offline SOT.
# ============================================================================
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval8_cosine/csc/sglatrack/uav123/test
SEQS="group3_2 car9 car1_s uav6 truck2 bird1_1 car6_2 car6_5"
CALIB=sglatrack_all_v2
PRO="--csc_mode control --policy_gated_redetect --gate_preset scoremap2 --gate_top2ratio 0.30 --gate_entropy 4.0 --redetect_action motion_bridge --redetect_arm_frames 3 --bridge_max_frames 30 --bridge_vel_ema 0.5"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
mkdir -p outputs/_logs "$ROOT"
LOG=outputs/_logs/run_la_ffo_smoke.log; : > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
run_cfg(){ local TAG="$1"; shift; local RUN="$ROOT/$TAG"; rm -rf "$RUN"
  say ">>> [$TAG] $*"
  $PY -u tools/run_with_csc.py --tracker sglatrack --dataset uav123 --split test \
    --csc_checkpoint "$CKPT" --calibration_prefix "$CALIB" --device cpu \
    --output_dir "$ROOT" --run_tag "$TAG" --include_sequences $SEQS \
    $PRO "$@" >>"$LOG" 2>&1 || { say "  FAIL $TAG"; return 1; }
  say "--- [$TAG] ΔAUC ---"
  $PY -u tools/la_smoke.py --run_dir "$RUN" --seqs $SEQS --tag "$TAG" 2>/dev/null | tee -a "$LOG"; }
say "===== freeze_fc_only isolation ($SEQS) ====="
run_cfg mb_sm2_b
run_cfg mb_sm2_ffo  --policy_freeze_fc_only
say "===== DONE ====="
echo "FFO_DONE" >> "$LOG"
