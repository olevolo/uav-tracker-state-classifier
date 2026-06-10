#!/bin/bash
# ============================================================================
# Tight-gate WIDEN-ONLY operating-point sweep — zero the uav6 guard while
# keeping the HARD +ΔAUC (widen_K5 gave HARD +0.0227 / bird1_1 +0.048 but uav6
# -0.016). Lever to fix uav6/easy regressions = higher --redetect_arm_frames
# (uav6 LA is scattered → a long arm never sustains → uav6 → ~0) + gentler widen.
# WIDEN-ONLY only (relocate/widen_relocate proven net-negative on this smoke).
# UAV123 EXPLORATORY (final calibration belongs on VisDrone/UAVDT, never DTB70).
# bash (NOT zsh) word-splitting. Sequential for clean timing.
# ============================================================================
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval7_gated/csc/sglatrack/uav123/test
SEQS="bird1_1 car1_3 bike3 car11 uav6 car7"
PRO="--csc_mode control --exit_router --proactive_v3 --proactive_threshold 0.7 --control_risk_gate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
mkdir -p outputs/_logs "$ROOT"
LOG=outputs/_logs/run_gated_sweep.log
: > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

run_cfg(){
  local TAG="$1"; shift
  local RUN="$ROOT/$TAG"
  if [ ! -f "$RUN/metrics.json" ]; then
    say ">>> [$TAG] $*"
    rm -rf "$RUN"
    $PY -u tools/run_with_csc.py --tracker sglatrack --dataset uav123 --split test \
      --csc_checkpoint "$CKPT" --calibration_prefix sglatrack_aerial_v2 --device cpu \
      --output_dir "$ROOT" --run_tag "$TAG" --include_sequences $SEQS \
      $PRO --policy_gated_redetect --redetect_action widen "$@" >>"$LOG" 2>&1 \
      || { say "  FAIL $TAG"; return 1; }
  else say ">>> [$TAG] skip (exists)"; fi
  say "--- [$TAG] ΔAUC ---"
  $PY -u tools/la_smoke.py --run_dir "$RUN" --seqs $SEQS --tag "$TAG" 2>/dev/null | tee -a "$LOG"
}

say "================ gated_sweep START (widen-only) ================"
# combined K=5 gate, sweep arm × widen_max
run_cfg w_K5_arm5_w15  --gate_preset combined --gate_vote_k 5 --redetect_arm_frames 5  --lost_widen_max 1.5
run_cfg w_K5_arm8_w15  --gate_preset combined --gate_vote_k 5 --redetect_arm_frames 8  --lost_widen_max 1.5
run_cfg w_K5_arm12_w15 --gate_preset combined --gate_vote_k 5 --redetect_arm_frames 12 --lost_widen_max 1.5
run_cfg w_K5_arm8_w13  --gate_preset combined --gate_vote_k 5 --redetect_arm_frames 8  --lost_widen_max 1.3
# tightest appearance gate variant
run_cfg w_app_arm8     --gate_preset appearance --gate_cosine 0.82 --gate_apce 90 --redetect_arm_frames 8 --lost_widen_max 1.5
say "================ gated_sweep DONE ================"
echo "" | tee -a "$LOG"
say "==== SUMMARY (HARD-mean / uav6-guard per config) ===="
grep -E 'ΔAUC: |mean ΔAUC \(HARD|GUARD' "$LOG" | tee -a "$LOG"