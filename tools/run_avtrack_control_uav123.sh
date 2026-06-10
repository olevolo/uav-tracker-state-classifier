#!/bin/bash
# AVTrack x CSC — passive + LA-only + FC-only + combo on UAV123 (full 123 seqs).
# Uses csc_prod (R3-fcw3, cross-tracker, avtrack_aerial_v2 calibration).
# Estimated ~20 min/mode at ~97 FPS x 113k frames.
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/csc_prod/checkpoint_best.pth
ROOT=outputs/eval10_avtrack/csc/avtrack/uav123/test
CALIB=avtrack_aerial_v2
LOG=outputs/_logs/run_avtrack_control.log
: > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Common flags
BASE_FLAGS="--tracker avtrack --dataset uav123 --split test \
  --csc_checkpoint $CKPT --calibration_prefix $CALIB --device cpu \
  --output_dir $ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4

# Shared control flags (scoremap2 gate, same as sglatrack scoremap2 tune)
GATE_FLAGS="--gate_preset scoremap2 --gate_top2ratio 0.45 --gate_entropy 4.3 \
  --redetect_action sgla_redetect --sgla_redetect_factors 8,12,16 \
  --sgla_redetect_min_apce 0 --gated_freeze --no_runner_template_update"
FC_FLAGS="--fc_action hold_lastgood --fc_streak_frames 2 --fc_gate_vote_k 2"

# ── 1. PASSIVE ──────────────────────────────────────────────────────────────
say "=== PASSIVE (1/4) ==="
$PY -u tools/run_with_csc.py $BASE_FLAGS --run_tag passive --csc_mode passive \
  >> "$LOG" 2>&1 || { say "FAIL passive"; exit 1; }
say "passive done"

# ── 2. LA-ONLY ──────────────────────────────────────────────────────────────
say "=== LA-only (2/4) ==="
$PY -u tools/run_with_csc.py $BASE_FLAGS --run_tag la_only --csc_mode control \
  --policy_gated_redetect $GATE_FLAGS >> "$LOG" 2>&1 || { say "FAIL la_only"; exit 1; }
say "la_only done"

# ── 3. FC-ONLY ──────────────────────────────────────────────────────────────
say "=== FC-only (3/4) ==="
$PY -u tools/run_with_csc.py $BASE_FLAGS --run_tag fc_only --csc_mode control \
  --policy_fc_control $FC_FLAGS --no_runner_template_update >> "$LOG" 2>&1 || { say "FAIL fc_only"; exit 1; }
say "fc_only done"

# ── 4. COMBO ────────────────────────────────────────────────────────────────
say "=== COMBO (4/4) ==="
$PY -u tools/run_with_csc.py $BASE_FLAGS --run_tag combo --csc_mode control \
  --policy_gated_redetect $GATE_FLAGS --policy_fc_control $FC_FLAGS \
  >> "$LOG" 2>&1 || { say "FAIL combo"; exit 1; }
say "combo done"

say "=== ALL DONE ==="
say "Results in: $ROOT"
