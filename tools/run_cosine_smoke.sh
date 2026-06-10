#!/bin/bash
# ============================================================================
# COSINE / SCOREMAP2 gate smoke — does the 123-seq-validated gate + motion_bridge
# give +ΔAUC on HARD scenes while leaving guards/easy untouched?
# Gate = scoremap2 (top2_ratio>=0.30 AND entropy>=4.0): uav6 fires 2.8% (guard).
# Action = motion_bridge (verified LA win on smooth motion). mgate variant adds a
# motion-smoothness cap (--bridge_max_resid_ratio) to exclude erratic bird1_1.
# Baseline for ΔAUC = la_smoke default (eval5_clamp clamped PASSIVE, all 123 seqs).
# SAFETY: offline SOT benchmark, control-only INFERENCE, no training. EXPLORATORY.
# bash word-splitting (run via `bash`), NOT zsh. Sequential for clean timing.
# ============================================================================
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval8_cosine/csc/sglatrack/uav123/test
# hard-recoverable (expect +) | guards (expect ~0) | erratic-anomalous | easy (expect 0)
SEQS="group3_2 car9 uav2 uav8 car1_s uav6 truck2 bird1_1 car6_2"
CALIB=sglatrack_all_v2
PRO="--csc_mode control --policy_gated_redetect --redetect_action motion_bridge --redetect_arm_frames 3 --bridge_max_frames 30 --bridge_vel_ema 0.5"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
mkdir -p outputs/_logs "$ROOT"
LOG=outputs/_logs/run_cosine_smoke.log
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

say "===== cosine_smoke START ($SEQS) ====="
run_cfg mb_sm2        --gate_preset scoremap2 --gate_top2ratio 0.30 --gate_entropy 4.0
run_cfg mb_sm2_mgate  --gate_preset scoremap2 --gate_top2ratio 0.30 --gate_entropy 4.0 --bridge_max_resid_ratio 2.0
run_cfg mb_comb_k3    --gate_preset combined  --gate_vote_k 3
say "===== cosine_smoke DONE ====="
echo "COSINE_SMOKE_DONE" >> "$LOG"
