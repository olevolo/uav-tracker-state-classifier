#!/bin/bash
# ============================================================================
# FULL UAV123 (or subset via SEQS) — combined LA+FC control. Template policy is
# chosen via TPL env (the ONE remaining design axis):
#   TPL="--recovery_update_window -1"  (default) ADAPTIVE: runner refreshes the
#        template every unfrozen frame. Bigger hard wins (car9 +0.46, car1_s +0.38)
#        but these are an ADAPTIVE-TEMPLATE effect (drifts a few healthy easy tracks,
#        e.g. car6_5 -0.11) — NOT a pure CSC-lever contribution.
#   TPL="--no_runner_template_update"  FROZEN: frame-0 template == PASSIVE baseline,
#        so ΔAUC isolates the CSC state-aware LEVERS (motion_bridge recovery + gated
#        FC). Easy scenes stay at ΔAUC~0 ("do nothing on easy"); wins = group3_2 etc.
# Levers (both modes): scoremap2 gate (top2>=0.30 AND entropy>=4.0) + motion_bridge,
#   gated FC hold_lastgood (--fc_gate_vote_k 2), --gated_freeze. NO --control_risk_gate.
# Metrics vs eval5_clamp passive: ΔAUC overall+HARD, LA-rate, FCR, FCD.
# SAFETY: offline SOT benchmark, control-only INFERENCE, no training. EXPLORATORY
# (gate thresholds chosen on UAV123 — final paper numbers re-confirm on a val set).
# ============================================================================
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval8_cosine/csc/sglatrack/uav123/test
CALIB=sglatrack_all_v2
TAG="${1:-full_combo}"
TPL="${TPL:---recovery_update_window -1}"
COMBO="--csc_mode control \
 --policy_gated_redetect --gate_preset scoremap2 --gate_top2ratio ${T2R:-0.45} --gate_entropy ${ENT:-4.3} \
 --redetect_action ${ACT:-motion_bridge} --relocate_min_ratio ${RMR:-0.30} --redetect_arm_frames 3 --bridge_max_frames ${BMF:-30} --bridge_vel_ema 0.5 --bridge_max_resid_ratio ${BRC:-1000000000} \
 --policy_fc_control --fc_action hold_lastgood --fc_streak_frames 2 --fc_gate_vote_k 2 \
 --gated_freeze $TPL"
INC=""; SMOKE_SEQS="${SEQS:-}"
[ -n "$SMOKE_SEQS" ] && INC="--include_sequences $SMOKE_SEQS"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
mkdir -p outputs/_logs "$ROOT"
LOG=outputs/_logs/run_${TAG}.log; : > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
RUN="$ROOT/$TAG"; rm -rf "$RUN"
say ">>> [$TAG] TPL=$TPL  $COMBO $INC"
$PY -u tools/run_with_csc.py --tracker sglatrack --dataset uav123 --split test \
  --csc_checkpoint "$CKPT" --calibration_prefix "$CALIB" --device cpu \
  --output_dir "$ROOT" --run_tag "$TAG" $COMBO $INC >>"$LOG" 2>&1 || { say "FAIL"; exit 1; }
say "--- [$TAG] ΔAUC + FC metrics vs eval5_clamp passive ---"
if [ -n "$SMOKE_SEQS" ]; then
  $PY -u tools/la_smoke.py --run_dir "$RUN" --seqs $SMOKE_SEQS --tag "$TAG" --fc_metrics 2>/dev/null | tee -a "$LOG"
else
  $PY -u tools/la_smoke.py --run_dir "$RUN" --tag "$TAG" --fc_metrics 2>/dev/null | tee -a "$LOG"
fi
say "===== [$TAG] DONE ====="; echo "FULL_${TAG}_DONE" >> "$LOG"
