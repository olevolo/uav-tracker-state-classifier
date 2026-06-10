#!/bin/bash
# ============================================================================
# CLEAN gated LA re-detect — NO FC scaffold (no exit_router/proactive/risk_gate).
# ctrl_hold revealed the scaffold alone costs HARD -0.030 / uav6 -0.016 vs passive,
# swamping the LA action. policy_gated_redetect is router-independent, so we can
# run the LA action ALONE and measure it directly against passive.
# action variants: widen / motion_bridge / relocate. UAV123 EXPLORATORY.
# ============================================================================
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval7_gated/csc/sglatrack/uav123/test
SEQS="bird1_1 car1_3 bike3 car11 uav6 car7"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
mkdir -p outputs/_logs
LOG=outputs/_logs/run_clean_smoke.log; : > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

run_cfg(){
  local TAG="$1"; shift
  local RUN="$ROOT/$TAG"
  if [ ! -f "$RUN/metrics.json" ]; then
    say ">>> [$TAG] $*"; rm -rf "$RUN"
    $PY -u tools/run_with_csc.py --tracker sglatrack --dataset uav123 --split test \
      --csc_checkpoint "$CKPT" --calibration_prefix sglatrack_aerial_v2 --device cpu \
      --output_dir "$ROOT" --run_tag "$TAG" --include_sequences $SEQS \
      --csc_mode control --policy_gated_redetect "$@" >>"$LOG" 2>&1 \
      || { say "  FAIL $TAG"; return 1; }
  else say ">>> [$TAG] skip"; fi
  say "--- [$TAG] ΔAUC vs PASSIVE ---"
  $PY -u tools/la_smoke.py --run_dir "$RUN" --seqs $SEQS --tag "$TAG" 2>/dev/null | tee -a "$LOG"
}

say "============ CLEAN gated LA re-detect (no FC scaffold) ============"
run_cfg clean_widen     --redetect_action widen         --gate_preset combined --gate_vote_k 5 --redetect_arm_frames 3 --lost_widen_max 1.5
run_cfg clean_mb_K4     --redetect_action motion_bridge --gate_preset combined --gate_vote_k 4 --redetect_arm_frames 3 --bridge_max_frames 30 --bridge_vel_ema 0.5
run_cfg clean_mb_K5     --redetect_action motion_bridge --gate_preset combined --gate_vote_k 5 --redetect_arm_frames 3 --bridge_max_frames 30 --bridge_vel_ema 0.5
run_cfg clean_reloc     --redetect_action relocate      --gate_preset combined --gate_vote_k 4 --relocate_min_ratio 0.5 --redetect_arm_frames 3
say "============ DONE ============"
say "==== SUMMARY ===="
grep -E 'ΔAUC: |mean ΔAUC|GUARD' "$LOG" | tee -a "$LOG"