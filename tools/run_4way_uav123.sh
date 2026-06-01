#!/bin/bash
# ============================================================================
# FROM-SCRATCH 4-way (×2 control variants) SGLATrack eval on UAV123.
# User request 2026-06-01: run everything fresh (NO reuse of existing runs),
# return AUC / Pr@20 / FPS + state-dist CC/CU/LA/FC + FCR/FCD/Recovery for:
#   1. bare sglatrack (no CSC)                  -> Teacher(GT) state metrics
#   2. CSC passive                              -> CSC runtime state metrics
#   3. control PROACTIVE + hard-gate            -> CSC runtime
#   4. control REACTIVE  + hard-gate            -> CSC runtime
# Plus the AUC-debug safeguard re-run. ROOT CAUSE of easy-scene regressions:
# the exit-router forces block-9 on LA, and on false-LA frames this thrashes
# the tracker into a drift feedback loop (uav6 LA 15%->40%). Template-freeze is
# INERT for SGLATrack (try_update_template fires <=5x/seq), and wider-search /
# re-detect are NOT wired. So the safeguard = --policy_hold_on_la: on LA DO
# NOTHING / hold (no block-9 override; template still frozen). FC keeps block-9.
#   5. control PROACTIVE + hard-gate + hold_on_la
#   6. control REACTIVE  + hard-gate + hold_on_la
#
# SAFETY: offline SOT benchmark, diagnosis/control-only, inference (NO training).
# UAV123 = final test; default thresholds (proactive 0.7) NOT tuned here.
# The freeze_fc_only safeguard is a POLICY-aligned code fix, not a UAV123 tune.
#
# bash-3.2 safe, resumable (skips a run when its metrics already exist),
# 2 waves of 3 concurrent jobs (limits CPU oversubscription on M4 Pro).
# FPS is measured by a SEPARATE clean sequential probe (Phase 3) so the
# parallel full runs' contention does not corrupt the reported FPS.
# ============================================================================
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
DS=uav123
ROOT=outputs/eval4_fresh
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
CALIB=sglatrack_aerial_v2
CALIB_JSON=outputs/calibration/${CALIB}_confidence.json
mkdir -p outputs/_logs "$ROOT"
LOG=outputs/_logs/run_4way_uav123.log
exec >>"$LOG" 2>&1
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

say "================ run_4way_uav123 START ================"
[ -f "$CKPT" ]       || { say "FATAL ckpt missing $CKPT"; exit 1; }
[ -f "$CALIB_JSON" ] || { say "FATAL calib missing $CALIB_JSON"; exit 1; }

# Per-job thread cap so 3 concurrent torch-cpu jobs don't oversubscribe.
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3

# ---------------------------------------------------------------- downstream
# Build Teacher(GT) labels + tracking metrics (+ paper metrics for CSC runs).
# $1=run_dir  $2=tag  $3=has_states(0/1)
downstream(){
  local RUN="$1" TAG="$2" HAS_STATES="$3"
  local SHADOW=outputs/_live_matrix_base/$TAG/sglatrack/$DS/test
  mkdir -p "$SHADOW"
  ln -sfn "$(pwd)/$RUN/predictions" "$SHADOW/predictions"
  ln -sfn "$(pwd)/$RUN/telemetry"   "$SHADOW/telemetry"
  # Teacher(GT) labels
  if [ ! -f "$RUN/labels_v3/$DS/test/labels.jsonl" ] && [ ! -d "$RUN/labels_v3/$DS/test/labels_per_sequence" ]; then
    $PY -u tools/build_scene_state_labels.py --tracker sglatrack --dataset "$DS" --split test \
      --baseline_dir outputs/_live_matrix_base/"$TAG"/sglatrack \
      --calibration_dir outputs/calibration --calibrator_tag "$CALIB" \
      --output_dir "$RUN/labels_v3" || say "    FAIL labels $TAG"
  fi
  # tracking metrics (AUC / Pr@20 / FPS)
  if [ ! -f "$RUN/tracking_metrics/metrics_summary.json" ]; then
    $PY -u tools/evaluate_tracking_results.py --dataset "$DS" --split test \
      --pred_dir "$RUN/predictions" --telemetry_dir "$RUN/telemetry" \
      --output_dir "$RUN/tracking_metrics" || say "    FAIL trackmetrics $TAG"
  fi
  # paper metrics (runtime FCR/FCD/Recovery from CSC states) — CSC runs only
  if [ "$HAS_STATES" = "1" ] && [ ! -f "$RUN/paper_metrics/paper_metrics.csv" ]; then
    $PY -u tools/compute_paper_metrics.py --tracker sglatrack --dataset "$DS" --split test \
      --predictions_dir "$RUN/predictions" --states_dir "$RUN/states" --telemetry_dir "$RUN/telemetry" \
      --labels_dir "$RUN/labels_v3/$DS/test" --tracking_metrics_dir "$RUN/tracking_metrics" \
      --confidence_calib "$CALIB_JSON" --output_dir "$RUN/paper_metrics" --recovery_k 30 \
      || say "    FAIL papermetrics $TAG"
  fi
  say "    downstream done: $TAG"
}

# ---------------------------------------------------------------- run wrappers
run_bare(){
  local RUN=$ROOT/bare/sglatrack/$DS/test
  if [ ! -f "$RUN/manifest.json" ]; then
    say ">>> [bare] run_baseline sglatrack/$DS (no CSC, MLP router default)"
    $PY -u tools/run_baseline.py --tracker sglatrack --dataset "$DS" --split test \
      --device cpu --output_dir "$ROOT/bare" || { say "  FAIL bare"; return 1; }
  else say ">>> [bare] skip (manifest exists)"; fi
  downstream "$RUN" bare4 0
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

# control flag groups (default exit-router hold/threshold; only LA policy varies)
PRO="--csc_mode control --exit_router --proactive_v3 --proactive_threshold 0.7 --control_risk_gate"
RE="--csc_mode control --exit_router --control_risk_gate"
# Safeguard (user 2026-06-01): on LA do nothing/hold — suppress block-9 thrashing
# (freeze is inert for SGLATrack; wider-search/re-detect not wired). FC keeps block-9.
SAFE="--policy_hold_on_la"

# ---------------------------------------------------------------- Phase 1: WAVE 1
say "=== WAVE 1 (bare, passive, ctrl_pro_default) ==="
run_bare &
run_csc passive --csc_mode passive &
run_csc ctrl_pro_default $PRO &
wait
say "=== WAVE 1 done ==="

# ---------------------------------------------------------------- Phase 2: WAVE 2
say "=== WAVE 2 (ctrl_re_default, ctrl_pro_hold, ctrl_re_hold) ==="
run_csc ctrl_re_default  $RE &
run_csc ctrl_pro_hold    $PRO $SAFE &
run_csc ctrl_re_hold     $RE  $SAFE &
wait
say "=== WAVE 2 done ==="

# ---------------------------------------------------------------- Phase 3: clean FPS probe
# Sequential, alone, threads=4, first 4 sequences -> uncontended comparable FPS.
say "=== Phase 3: clean FPS probe (sequential, 4 seqs) ==="
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
PROBE=$ROOT/fps_probe
mkdir -p "$PROBE"
if [ ! -f "$PROBE/bare/sglatrack/$DS/test/manifest.json" ]; then
  $PY -u tools/run_baseline.py --tracker sglatrack --dataset "$DS" --split test \
    --device cpu --max_sequences 4 --output_dir "$PROBE/bare" || say "  FAIL probe bare"
fi
for cell in "passive_probe --csc_mode passive" "ctrl_pro_hold_probe $PRO $SAFE"; do
  set -- $cell; ptag="$1"; shift
  if [ ! -f "$PROBE/csc/sglatrack/$DS/test/$ptag/metrics.json" ]; then
    $PY -u tools/run_with_csc.py --tracker sglatrack --dataset "$DS" --split test \
      --csc_checkpoint "$CKPT" --calibration_prefix "$CALIB" --device cpu \
      --max_sequences 4 --output_dir "$PROBE/csc/sglatrack/$DS/test" --run_tag "$ptag" "$@" \
      || say "  FAIL probe $ptag"
  fi
done
say "=== Phase 3 done ==="

say "================ run_4way_uav123 DONE — run tools/agg_4way.py ================"
