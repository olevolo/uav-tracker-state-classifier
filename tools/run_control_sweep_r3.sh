#!/bin/bash
# Mode-3 CONTROL sweep — sglatrack x {uav123, uav123_10fps, uavtrack112, dtb70}.
# NO TRAINING (inference only). R3-fcw3 (w32) primary model. Risk-gated proactive
# control, so passive-vs-control differs ONLY in csc_mode (same calibrator =
# sglatrack_aerial_v2, identical to the Mode-2 passive matrix sglatrack_r3_passive
# row -> the control table's implicit baseline IS that passive row).
#
#   control flags : --csc_mode control --exit_router --proactive_v3
#                   --proactive_threshold 0.7 --control_risk_gate
#   run_tag       : sglatrack_r3_control   (build_results_doc picks this first)
#
# INTEGRITY: offline SOT, diagnosis/control-only. uav123(+@10fps) = clean final test;
# uavtrack112 = clean held-out (aerial); dtb70 = IN-SAMPLE/circular (sanity only).
# Resumable (skip on metrics.json). bash-3.2-safe. Logged.
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
mkdir -p outputs/_logs
LOG=outputs/_logs/control_sweep_r3.log
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
CALIB=sglatrack_aerial_v2
TAG=sglatrack_r3_control
exec >>"$LOG" 2>&1
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

say "=== control_sweep_r3 START (R3-fcw3, risk-gated proactive, calib=$CALIB) ==="
[ -f "$CKPT" ] || { say "FATAL: checkpoint missing: $CKPT"; exit 1; }
[ -f "outputs/calibration/${CALIB}_confidence.json" ] || { say "FATAL: calibrator $CALIB missing"; exit 1; }

# wait out any heavy proc
while pgrep -f "train_csc|run_baseline\.py" >/dev/null 2>&1; do
  say "waiting for heavy proc ..."; sleep 30
done

run_ctrl() {
  local ds="$1"
  local RUN=outputs/eval/sglatrack/$ds/test/$TAG
  say ">>> control sglatrack / $ds  tag=$TAG"
  [ "$ds" = "dtb70" ] && say "    NOTE dtb70 = IN-SAMPLE/circular — sanity cell only"
  if [ ! -f "$RUN/metrics.json" ]; then
    rm -rf "$RUN/states" "$RUN/paper_metrics" "$RUN/tracking_metrics"
    $PY -u tools/run_with_csc.py --tracker sglatrack --dataset "$ds" --split test \
      --csc_checkpoint "$CKPT" --csc_mode control \
      --exit_router --proactive_v3 --proactive_threshold 0.7 --control_risk_gate \
      --calibration_prefix "$CALIB" --device cpu \
      --output_dir "outputs/eval/sglatrack/$ds/test" --run_tag "$TAG" \
      || { say "    FAIL run_with_csc control $ds"; return 1; }
  else say "    skip LIVE (metrics.json exists)"; fi
  [ -d "$RUN/states" ] || { say "    no states — skip downstream"; return 1; }

  local SHADOW=outputs/_live_matrix_base/$TAG/sglatrack/$ds/test
  mkdir -p "$SHADOW"
  ln -sfn "$(pwd)/$RUN/predictions" "$SHADOW/predictions"
  ln -sfn "$(pwd)/$RUN/telemetry"   "$SHADOW/telemetry"

  local LAB=$RUN/labels_v3
  if [ ! -f "$LAB/$ds/test/labels.jsonl" ]; then
    $PY -u tools/build_scene_state_labels.py --tracker sglatrack --dataset "$ds" --split test \
      --baseline_dir outputs/_live_matrix_base/"$TAG"/sglatrack --calibration_dir outputs/calibration \
      --calibrator_tag "$CALIB" --output_dir "$LAB" || say "    FAIL labels $ds"
  else say "    skip labels"; fi

  local TM=$RUN/tracking_metrics
  if [ ! -f "$TM/metrics_summary.json" ]; then
    $PY -u tools/evaluate_tracking_results.py --dataset "$ds" --split test \
      --pred_dir "$RUN/predictions" --telemetry_dir "$RUN/telemetry" \
      --output_dir "$TM" || say "    FAIL track-metrics $ds"
  else say "    skip track-metrics"; fi

  if [ ! -f "$RUN/paper_metrics/paper_metrics.json" ] && [ ! -f "$RUN/paper_metrics/paper_metrics.csv" ]; then
    $PY -u tools/compute_paper_metrics.py --tracker sglatrack --dataset "$ds" --split test \
      --predictions_dir "$RUN/predictions" --states_dir "$RUN/states" --telemetry_dir "$RUN/telemetry" \
      --labels_dir "$LAB/$ds/test" --tracking_metrics_dir "$TM" \
      --confidence_calib outputs/calibration/${CALIB}_confidence.json \
      --output_dir "$RUN/paper_metrics" --recovery_k 30 || say "    FAIL paper-metrics $ds"
  else say "    skip paper-metrics"; fi
  say "    control cell done: $ds"
}

for ds in uav123 uav123_10fps uavtrack112 dtb70; do
  run_ctrl "$ds"
done
say "=== control_sweep_r3 DONE — rebuild RESULTS via build_results_doc.py ==="
