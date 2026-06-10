#!/bin/bash
# ============================================================================
# REFINE smoke — kill the two residual mb_sm2 drags:
#   bird1_1 -0.155 (erratic motion: tighten --bridge_max_resid_ratio so the
#                   motion-bridge refuses to extrapolate on irregular velocity)
#   car6_2  -0.012 (easy: conservative gate top2>=0.40 entropy>=4.1)
# Wins to PRESERVE: car9/group3_2/car1_s (+0.38..+0.46). Guard: uav6 ΔAUC>=-0.01.
# Baseline = la_smoke default (eval5_clamp passive). EXPLORATORY offline SOT.
# ============================================================================
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval8_cosine/csc/sglatrack/uav123/test
SEQS="group3_2 car9 uav2 uav8 car1_s uav6 truck2 bird1_1 car6_2"
CALIB=sglatrack_all_v2
PRO="--csc_mode control --policy_gated_redetect --redetect_action motion_bridge --redetect_arm_frames 3 --bridge_max_frames 30 --bridge_vel_ema 0.5"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
mkdir -p outputs/_logs "$ROOT"
LOG=outputs/_logs/run_cosine_smoke2.log
: > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

run_cfg(){
  local TAG="$1"; shift
  local RUN="$ROOT/$TAG"
  say ">>> [$TAG] $*"
  rm -rf "$RUN"
  $PY -u tools/run_with_csc.py --tracker sglatrack --dataset uav123 --split test \
    --csc_checkpoint "$CKPT" --calibration_prefix "$CALIB" --device cpu \
    --output_dir "$ROOT" --run_tag "$TAG" --include_sequences $SEQS \
    $PRO "$@" >>"$LOG" 2>&1 || { say "  FAIL $TAG"; return 1; }
  say "--- [$TAG] ΔAUC vs eval5_clamp passive ---"
  $PY -u tools/la_smoke.py --run_dir "$RUN" --seqs $SEQS --tag "$TAG" 2>/dev/null | tee -a "$LOG"
}

say "===== refine smoke START ($SEQS) ====="
run_cfg mb_mg10       --gate_preset scoremap2 --gate_top2ratio 0.30 --gate_entropy 4.0 --bridge_max_resid_ratio 1.0
run_cfg mb_mg05       --gate_preset scoremap2 --gate_top2ratio 0.30 --gate_entropy 4.0 --bridge_max_resid_ratio 0.5
run_cfg mb_cons_mg10  --gate_preset scoremap2 --gate_top2ratio 0.40 --gate_entropy 4.1 --bridge_max_resid_ratio 1.0
say "===== refine smoke DONE ====="
echo "REFINE_SMOKE_DONE" >> "$LOG"
