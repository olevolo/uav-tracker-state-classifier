#!/bin/bash
# ============================================================================
# CLAMP-CORRECTED baseline matrix — UAV123 (final test). NO TRAINING.
#
# Context (2026-06-02): the bbox clamp-to-frame fix (project rule "every adapter
# must clip bbox to frame") was MISSING in 3 adapters; off-frame drift scored
# IoU~0 -> baseline AUC/Pr@20 understated, FCR inflated.
#   sglatrack : fixed this session -> re-run already in outputs/eval5_clamp  (reuse)
#   ostrack   : already clipped natively (_clip_box, lib.utils.box_ops)      (reuse)
#   ortrack   : clamp ADDED this session -> re-run HERE
#   avtrack   : clamp ADDED this session -> re-run HERE
#
# Passive CSC does NOT move the bbox, so the passive tracking-AUC == bare AUC,
# while the same run ALSO yields the CSC diagnostic states (FCR/FCD/Recovery).
# So ONE passive run per tracker gives both the corrected baseline AUC and the
# CSC-side numbers. CSC = R3-fcw3 (validation-selected paper model), reused
# unchanged. Per-tracker aerial_v2 calibrator (GOT10k+DTB70+VisDrone; keeps
# UAV123 clean/held-out) — identical recipe to the prior matrix + eval5.
#
# SAFETY: offline SOT benchmark, diagnosis-only, INFERENCE. UAV123 = final test;
# the clamp is a uniform correctness fix, NOT a UAV123-tuned threshold.
# bash-3.2 safe, resumable (skip on metrics.json), 1 wave of 2 parallel.
# ============================================================================
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
DS=uav123
ROOT=outputs/eval6_matrix
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
mkdir -p outputs/_logs "$ROOT"
LOG=outputs/_logs/run_clamp_matrix.log
exec >>"$LOG" 2>&1
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

say "================ run_clamp_matrix START ================"
[ -f "$CKPT" ] || { say "FATAL ckpt missing $CKPT"; exit 1; }
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3

# don't fight a heavy proc for CPU
while pgrep -f "train_csc|run_baseline\.py|run_with_csc\.py" >/dev/null 2>&1; do
  say "waiting for heavy proc to finish ..."; sleep 30
done
say "CPU free — proceeding."

# downstream: $1=run_dir $2=tag $3=tracker $4=calib_prefix
downstream(){
  local RUN="$1" TAG="$2" T="$3" C="$4"
  local SHADOW=outputs/_clamp_matrix_base/$TAG/$T/$DS/test
  mkdir -p "$SHADOW"
  ln -sfn "$(pwd)/$RUN/predictions" "$SHADOW/predictions"
  ln -sfn "$(pwd)/$RUN/telemetry"   "$SHADOW/telemetry"
  if [ ! -f "$RUN/labels_v3/$DS/test/labels.jsonl" ] && [ ! -d "$RUN/labels_v3/$DS/test/labels_per_sequence" ]; then
    $PY -u tools/build_scene_state_labels.py --tracker "$T" --dataset "$DS" --split test \
      --baseline_dir outputs/_clamp_matrix_base/"$TAG"/"$T" \
      --calibration_dir outputs/calibration --calibrator_tag "$C" \
      --output_dir "$RUN/labels_v3" || say "    FAIL labels $TAG"
  fi
  if [ ! -f "$RUN/tracking_metrics/metrics_summary.json" ]; then
    $PY -u tools/evaluate_tracking_results.py --dataset "$DS" --split test \
      --pred_dir "$RUN/predictions" --telemetry_dir "$RUN/telemetry" \
      --output_dir "$RUN/tracking_metrics" || say "    FAIL trackmetrics $TAG"
  fi
  if [ ! -f "$RUN/paper_metrics/paper_metrics.csv" ]; then
    $PY -u tools/compute_paper_metrics.py --tracker "$T" --dataset "$DS" --split test \
      --predictions_dir "$RUN/predictions" --states_dir "$RUN/states" --telemetry_dir "$RUN/telemetry" \
      --labels_dir "$RUN/labels_v3/$DS/test" --tracking_metrics_dir "$RUN/tracking_metrics" \
      --confidence_calib outputs/calibration/${C}_confidence.json \
      --output_dir "$RUN/paper_metrics" --recovery_k 30 || say "    FAIL papermetrics $TAG"
  fi
  say "    downstream done: $TAG"
}

# run_passive <tracker> <calib_prefix>
run_passive(){
  local T="$1" C="$2"
  local RUN=$ROOT/csc/$T/$DS/test/passive
  [ -f "outputs/calibration/${C}_confidence.json" ] || { say "SKIP $T — calib $C missing"; return 1; }
  if [ ! -f "$RUN/metrics.json" ]; then
    say ">>> [$T] run_with_csc passive (clamped tracker)  calib=$C"
    rm -rf "$RUN/states" "$RUN/paper_metrics" "$RUN/tracking_metrics"
    $PY -u tools/run_with_csc.py --tracker "$T" --dataset "$DS" --split test \
      --csc_checkpoint "$CKPT" --csc_mode passive --calibration_prefix "$C" --device cpu \
      --output_dir "$ROOT/csc/$T/$DS/test" --run_tag passive \
      || { say "  FAIL $T"; return 1; }
  else say ">>> [$T] skip (metrics.json exists)"; fi
  [ -d "$RUN/states" ] && downstream "$RUN" passive "$T" "$C" || say "  no states for $T"
}

say "=== WAVE (ortrack, avtrack) — 2 parallel ==="
run_passive ortrack ortrack_aerial_v2 &
run_passive avtrack avtrack_aerial_v2 &
wait
say "=== WAVE done ==="
say "================ run_clamp_matrix DONE — run tools/agg_clamp_matrix.py ================"
