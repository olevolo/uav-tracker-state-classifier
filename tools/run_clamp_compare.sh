#!/bin/bash
# ============================================================================
# CLAMP + LOST-ACTION comparison on UAV123 (final test), fresh from scratch.
# User 2026-06-02: "implement proper LOST wider-search + do nothing on EASY".
#
# Two tracker-level fixes landed in src/uav_tracker/trackers/sglatrack.py:
#   (1) bbox CLAMP to frame dims (project rule; was missing) — prevents out-of-
#       frame drift boxes scoring 0 AND stops the wider-search runaway/OOM.
#       This is an UNCONDITIONAL baseline correctness fix (no CSC dependency).
#   (2) instance _search_factor + set_search_factor() + capabilities.can_widen
#       = the real wider-search lever (the proper LOST action).
#
# 4 conditions, all on the CLAMPED tracker so every row is comparable:
#   1. bare              (no CSC)                         -> Teacher(GT) states
#   2. passive           (CSC observe)                    -> CSC runtime states
#   3. ctrl_hold         (PRO + risk_gate + HOLD on LA)   -> recommended SAFE
#        LA = do nothing (hold); FC = block-9. "Do nothing on easy" by never
#        intervening on the (noisy) LA state.
#   4. ctrl_widen        (PRO + risk_gate + WIDEN on LA)  -> HONEST ablation
#        LA = progressive wider search (arm=5 consec, gentle max 1.5x); FC = block-9.
#        Measured-negative: the CSC over-predicts LA on ambiguous-but-OK seqs
#        (uav6: 98% LA, 72-frame run, yet GT-fine) and no causal runtime signal
#        (confidence is degenerate ~0.017; displacement does not separate)
#        distinguishes false-LA from true loss, so widen damages them.
#
# SAFETY: offline SOT benchmark, diagnosis/control-only, INFERENCE (no training).
# UAV123 = final test; thresholds NOT tuned here (defaults / policy-aligned).
# bash-3.2 safe, resumable (skip on metrics.json/manifest.json). 1 wave of 4.
# ============================================================================
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
DS=uav123
ROOT=outputs/eval5_clamp
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
CALIB=sglatrack_aerial_v2
CALIB_JSON=outputs/calibration/${CALIB}_confidence.json
mkdir -p outputs/_logs "$ROOT"
LOG=outputs/_logs/run_clamp_compare.log
exec >>"$LOG" 2>&1
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

say "================ run_clamp_compare START ================"
[ -f "$CKPT" ]       || { say "FATAL ckpt missing $CKPT"; exit 1; }
[ -f "$CALIB_JSON" ] || { say "FATAL calib missing $CALIB_JSON"; exit 1; }
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3

# ---------------------------------------------------------------- downstream
# $1=run_dir  $2=tag  $3=has_states(0/1)
downstream(){
  local RUN="$1" TAG="$2" HAS_STATES="$3"
  local SHADOW=outputs/_live_matrix_base/$TAG/sglatrack/$DS/test
  mkdir -p "$SHADOW"
  ln -sfn "$(pwd)/$RUN/predictions" "$SHADOW/predictions"
  ln -sfn "$(pwd)/$RUN/telemetry"   "$SHADOW/telemetry"
  if [ ! -f "$RUN/labels_v3/$DS/test/labels.jsonl" ] && [ ! -d "$RUN/labels_v3/$DS/test/labels_per_sequence" ]; then
    $PY -u tools/build_scene_state_labels.py --tracker sglatrack --dataset "$DS" --split test \
      --baseline_dir outputs/_live_matrix_base/"$TAG"/sglatrack \
      --calibration_dir outputs/calibration --calibrator_tag "$CALIB" \
      --output_dir "$RUN/labels_v3" || say "    FAIL labels $TAG"
  fi
  if [ ! -f "$RUN/tracking_metrics/metrics_summary.json" ]; then
    $PY -u tools/evaluate_tracking_results.py --dataset "$DS" --split test \
      --pred_dir "$RUN/predictions" --telemetry_dir "$RUN/telemetry" \
      --output_dir "$RUN/tracking_metrics" || say "    FAIL trackmetrics $TAG"
  fi
  if [ "$HAS_STATES" = "1" ] && [ ! -f "$RUN/paper_metrics/paper_metrics.csv" ]; then
    $PY -u tools/compute_paper_metrics.py --tracker sglatrack --dataset "$DS" --split test \
      --predictions_dir "$RUN/predictions" --states_dir "$RUN/states" --telemetry_dir "$RUN/telemetry" \
      --labels_dir "$RUN/labels_v3/$DS/test" --tracking_metrics_dir "$RUN/tracking_metrics" \
      --confidence_calib "$CALIB_JSON" --output_dir "$RUN/paper_metrics" --recovery_k 30 \
      || say "    FAIL papermetrics $TAG"
  fi
  say "    downstream done: $TAG"
}

run_bare(){
  local RUN=$ROOT/bare/sglatrack/$DS/test
  if [ ! -f "$RUN/manifest.json" ]; then
    say ">>> [bare] run_baseline sglatrack/$DS (clamped tracker, no CSC)"
    $PY -u tools/run_baseline.py --tracker sglatrack --dataset "$DS" --split test \
      --device cpu --output_dir "$ROOT/bare" || { say "  FAIL bare"; return 1; }
  else say ">>> [bare] skip (manifest exists)"; fi
  downstream "$RUN" bare5 0
}
# $1=tag  $2..=extra flags
run_csc(){
  local TAG="$1"; shift
  local RUN=$ROOT/csc/sglatrack/$DS/test/$TAG
  if [ ! -f "$RUN/metrics.json" ]; then
    say ">>> [$TAG] run_with_csc  flags: $*"
    rm -rf "$RUN/states" "$RUN/paper_metrics" "$RUN/tracking_metrics"
    $PY -u tools/run_with_csc.py --tracker sglatrack --dataset "$DS" --split test \
      --csc_checkpoint "$CKPT" --calibration_prefix "$CALIB" --device cpu \
      --output_dir "$ROOT/csc/sglatrack/$DS/test" --run_tag "$TAG" "$@" \
      || { say "  FAIL $TAG"; return 1; }
  else say ">>> [$TAG] skip (metrics.json exists)"; fi
  [ -d "$RUN/states" ] && downstream "$RUN" "$TAG" 1 || say "  no states for $TAG"
}

PRO="--csc_mode control --exit_router --proactive_v3 --proactive_threshold 0.7 --control_risk_gate"
HOLD="--policy_hold_on_la"
WIDEN="--policy_lost_widen_search --lost_arm_frames 5 --lost_widen_max 1.5 --lost_widen_step 0.15"

say "=== WAVE (bare, passive, ctrl_hold, ctrl_widen) — 4 parallel ==="
run_bare &
run_csc passive --csc_mode passive &
run_csc ctrl_hold  $PRO $HOLD &
run_csc ctrl_widen $PRO $WIDEN &
wait
say "=== WAVE done ==="

# clean FPS probe (sequential, 4 seqs, threads=4)
say "=== FPS probe (sequential, 4 seqs) ==="
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
PROBE=$ROOT/fps_probe
mkdir -p "$PROBE"
if [ ! -f "$PROBE/bare/sglatrack/$DS/test/manifest.json" ]; then
  $PY -u tools/run_baseline.py --tracker sglatrack --dataset "$DS" --split test \
    --device cpu --max_sequences 4 --output_dir "$PROBE/bare" || say "  FAIL probe bare"
fi
if [ ! -f "$PROBE/csc/sglatrack/$DS/test/ctrl_hold_probe/metrics.json" ]; then
  $PY -u tools/run_with_csc.py --tracker sglatrack --dataset "$DS" --split test \
    --csc_checkpoint "$CKPT" --calibration_prefix "$CALIB" --device cpu \
    --max_sequences 4 --output_dir "$PROBE/csc/sglatrack/$DS/test" --run_tag ctrl_hold_probe \
    $PRO $HOLD || say "  FAIL probe ctrl_hold"
fi
say "=== probe done ==="
say "================ run_clamp_compare DONE — run tools/agg_clamp.py ================"
