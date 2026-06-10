#!/bin/bash
# ============================================================================
# FC-GATE smoke — does the high-precision true-FC gate (--fc_gate_vote_k) stop
# hold_lastgood from corrupting good tracks? UAV123: ~86% of FC predictions are
# false-FC (tracker fine, IoU>=0.5); vote>=2 fires on only ~0.4% of those.
#   fc_freeze   : do-no-harm baseline (freeze template only)
#   fc_hold     : hold_lastgood, NO gate (legacy — expected easy-scene regression)
#   fc_hold_g2  : hold_lastgood + fc_gate_vote_k=2 (should keep true-FC fixes,
#                 drop the false-FC corruption)
# Metrics: AUC + FCR + FCD vs eval5_clamp passive. EXPLORATORY offline SOT.
# ============================================================================
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval8_cosine/csc/sglatrack/uav123/test
SEQS="car6_4 wakeboard10 car17 boat8 car7 car2_s car5 car6_5"
CALIB=sglatrack_all_v2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
mkdir -p outputs/_logs "$ROOT"
LOG=outputs/_logs/run_fc_gate_smoke.log; : > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
run_fc(){ local TAG="$1"; shift; local RUN="$ROOT/$TAG"; rm -rf "$RUN"
  say ">>> [$TAG] $*"
  $PY -u tools/run_with_csc.py --tracker sglatrack --dataset uav123 --split test \
    --csc_checkpoint "$CKPT" --calibration_prefix "$CALIB" --device cpu \
    --output_dir "$ROOT" --run_tag "$TAG" --include_sequences $SEQS \
    --csc_mode control --policy_fc_control "$@" >>"$LOG" 2>&1 || { say "  FAIL $TAG"; return 1; }
  say "--- [$TAG] AUC+FC vs passive ---"
  $PY -u tools/la_smoke.py --run_dir "$RUN" --seqs $SEQS --tag "$TAG" --fc_metrics 2>/dev/null | tee -a "$LOG"; }
say "============ FC-gate smoke ($SEQS) ============"
run_fc fc_freeze    --fc_action freeze_only  --fc_streak_frames 2
run_fc fc_hold      --fc_action hold_lastgood --fc_streak_frames 2
run_fc fc_hold_g2   --fc_action hold_lastgood --fc_streak_frames 2 --fc_gate_vote_k 2
say "============ DONE ============"
echo "FC_GATE_DONE" >> "$LOG"
