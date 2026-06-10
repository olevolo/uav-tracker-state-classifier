#!/bin/bash
# ============================================================================
# GATED RE-DETECT smoke sweep — does ANY (gate, action) give +ΔAUC?
# UAV123 (final test; EXPLORATORY threshold search, NOT a final-tuned result).
#
# Gate operating point chosen OFFLINE (tools/la_gate_tune.py): a precise gate
# fires on ~0.67 of true-loss frames but only ~0.02 of false-LA. The OPEN
# question is the ACTION: does widen / relocate actually recover IoU on a
# RECOVERABLE loss? Smoke set = recoverable losses + false-LA guards.
#   recoverable (gain possible): bird1_1 car1_3 bike3 car11
#   false-LA / easy GUARDS (must NOT regress): uav6 car7
# Baseline for ΔAUC = the clamped PASSIVE run (tools/la_smoke.py).
#
# SAFETY: offline SOT benchmark, control-only, INFERENCE. No training.
# bash word-splitting (NOT zsh) so $FLAGS expands. Sequential (clean timing).
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
LOG=outputs/_logs/run_gated_smoke.log
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
      $PRO --policy_gated_redetect "$@" >>"$LOG" 2>&1 \
      || { say "  FAIL $TAG"; return 1; }
  else say ">>> [$TAG] skip (exists)"; fi
  say "--- [$TAG] ΔAUC ---"
  $PY -u tools/la_smoke.py --run_dir "$RUN" --seqs $SEQS --tag "$TAG" 2>/dev/null | tee -a "$LOG"
}

say "================ gated_smoke START (seqs: $SEQS) ================"
run_cfg widen_appear      --gate_preset appearance --gate_cosine 0.85 --gate_apce 110 \
                          --redetect_action widen --redetect_arm_frames 3 --lost_widen_max 1.5 --lost_widen_step 0.15
run_cfg reloc_K4          --gate_preset combined --gate_vote_k 4 \
                          --redetect_action relocate --relocate_min_ratio 0.5 --redetect_arm_frames 3
run_cfg widenreloc_K4     --gate_preset combined --gate_vote_k 4 \
                          --redetect_action widen_relocate --relocate_min_ratio 0.5 --redetect_arm_frames 3 --lost_widen_max 1.5
run_cfg widen_K5          --gate_preset combined --gate_vote_k 5 \
                          --redetect_action widen --redetect_arm_frames 3 --lost_widen_max 1.5
say "================ gated_smoke DONE ================"