#!/bin/bash
# Master final-paper chain — NO TRAINING (mandate 2026-05-31). Sequential,
# resumable (skip-guards), logged. Waits for any live tracker/eval/training proc
# to finish first so it never contends for CPU or evaluates a non-final ckpt.
#
#   STEP 0  wait for R4 eval / replay-validation / training to finish
#   STEP 1  R4 (sglatrack uav123): confusion + 3-state collapse  -> FINAL_REPORT §4
#   STEP 2  UAV123@10fps (final-test #2): sglatrack/avtrack/ortrack LIVE passive
#           (one pass + symlink) -> labels -> tracking + paper metrics -> confusion+3state
#   STEP 3  Coverage: OSTrack uav123 via VALIDATED telemetry-replay (no re-run)
#
# SAFETY: offline SOT benchmarking only; diagnosis-only; UAV123(@10fps) = final
# test, never trained/tuned on. R3-fcw3 is the selected paper model (chosen on
# validation); @10fps reuses it unchanged (no re-selection on the test set).
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
LOG=outputs/_logs/final_chain.log
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
exec >>"$LOG" 2>&1
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

say "=== final_paper_chain START ==="

# ---- STEP 0: wait until CPU is free (no tracker/eval/training procs) ----
while pgrep -f "train_csc|run_with_csc|run_baseline\.py|replay_csc_states" >/dev/null 2>&1; do
  say "waiting for live proc to finish ..."; sleep 30
done
say "CPU free — proceeding."

# bash 3.2 (macOS default /bin/bash) has NO associative arrays — use a function.
calib_for() {
  case "$1" in
    sglatrack) echo sglatrack_all_v2 ;;
    avtrack)   echo avtrack_aerial_v2 ;;
    ortrack)   echo ortrack_aerial_v2 ;;
    ostrack)   echo ostrack_aerial_v2 ;;
    *)         echo "" ;;
  esac
}

# ---- STEP 1: R4 confusion + 3-state (uav123, sglatrack) ----
R4=outputs/eval/sglatrack/uav123/test/sglatrack_r4_passive
if [ -d "$R4/states" ]; then
  say "STEP1 R4 confusion+collapse"
  $PY tools/confusion_uav123.py   --trackers sglatrack --dataset uav123 --run_tag sglatrack_r4_passive --save
  $PY tools/collapse_cu_3state.py --trackers sglatrack --dataset uav123 --run_tag sglatrack_r4_passive --save
else
  say "STEP1 SKIP — R4 states missing ($R4)"
fi

# ---- STEP 2: UAV123@10fps LIVE passive for the 3 core trackers ----
DS=uav123_10fps
for t in sglatrack avtrack ortrack; do
  c=$(calib_for "$t")
  RUN=outputs/eval/$t/$DS/test/${t}_r3_passive
  say "STEP2 @10fps $t (calib $c)"
  if [ ! -f "$RUN/metrics.json" ]; then
    $PY -u tools/run_with_csc.py --tracker $t --dataset $DS --split test \
      --csc_checkpoint "$CKPT" --csc_mode passive --calibration_prefix $c --device cpu \
      --output_dir outputs/eval/$t/$DS/test --run_tag ${t}_r3_passive || { say "  FAIL run_with_csc $t"; continue; }
  else say "  skip passive (metrics.json exists)"; fi
  # symlink baseline structure so build_scene_state_labels can read it
  BASE=outputs/baselines/$t/$DS/test; mkdir -p "$BASE"
  [ -e "$BASE/predictions" ] || ln -s "$(pwd)/$RUN/predictions" "$BASE/predictions"
  [ -e "$BASE/telemetry" ]   || ln -s "$(pwd)/$RUN/telemetry"   "$BASE/telemetry"
  LAB=outputs/eval/$t/$DS/test/labels_v3
  if [ ! -f "$LAB/$DS/test/labels.jsonl" ]; then
    $PY -u tools/build_scene_state_labels.py --tracker $t --dataset $DS --split test \
      --baseline_dir outputs/baselines/$t --calibration_dir outputs/calibration \
      --calibrator_tag $c --output_dir "$LAB" || say "  FAIL labels $t"
  else say "  skip labels"; fi
  TM=$RUN/tracking_metrics
  if [ ! -f "$TM/summary.json" ]; then
    $PY -u tools/evaluate_tracking_results.py --dataset $DS --split test \
      --pred_dir "$RUN/predictions" --output_dir "$TM" || say "  FAIL track-metrics $t"
  else say "  skip track-metrics"; fi
  if [ ! -f "$RUN/paper_metrics/paper_metrics.json" ]; then
    $PY -u tools/compute_paper_metrics.py --tracker $t --dataset $DS --split test \
      --predictions_dir "$RUN/predictions" --states_dir "$RUN/states" --telemetry_dir "$RUN/telemetry" \
      --labels_dir "$LAB/$DS/test" --tracking_metrics_dir "$TM" \
      --confidence_calib outputs/calibration/${c}_confidence.json \
      --output_dir "$RUN/paper_metrics" --recovery_k 30 || say "  FAIL paper-metrics $t"
  else say "  skip paper-metrics"; fi
done
say "STEP2 @10fps confusion+collapse (3 trackers)"
$PY tools/confusion_uav123.py   --trackers sglatrack avtrack ortrack --dataset $DS --save
$PY tools/collapse_cu_3state.py --trackers sglatrack avtrack ortrack --dataset $DS --save

# ---- STEP 3: OSTrack coverage (uav123) via validated telemetry-replay ----
t=ostrack; c=$(calib_for "$t"); RUN=outputs/eval/$t/uav123/test/${t}_r3_passive
say "STEP3 coverage $t (replay, calib $c)"
if [ ! -d "$RUN/states" ]; then
  $PY -u tools/replay_csc_states.py --tracker $t --dataset uav123 \
    --checkpoint "$CKPT" --calibrator $c --run_tag ${t}_r3_passive || say "  FAIL replay $t"
else say "  skip replay (states exist)"; fi
LAB=outputs/eval/$t/uav123/test/labels_v3
if [ ! -f "$LAB/uav123/test/labels.jsonl" ]; then
  $PY -u tools/build_scene_state_labels.py --tracker $t --dataset uav123 --split test \
    --baseline_dir outputs/baselines/$t --calibration_dir outputs/calibration \
    --calibrator_tag $c --output_dir "$LAB" || say "  FAIL labels $t"
fi
TM=$RUN/tracking_metrics
if [ ! -f "$TM/summary.json" ]; then
  $PY -u tools/evaluate_tracking_results.py --dataset uav123 --split test \
    --pred_dir outputs/baselines/$t/uav123/test/predictions --output_dir "$TM" || say "  FAIL track-metrics $t"
fi
if [ ! -f "$RUN/paper_metrics/paper_metrics.json" ]; then
  $PY -u tools/compute_paper_metrics.py --tracker $t --dataset uav123 --split test \
    --predictions_dir outputs/baselines/$t/uav123/test/predictions --states_dir "$RUN/states" \
    --telemetry_dir outputs/baselines/$t/uav123/test/telemetry \
    --labels_dir "$LAB/uav123/test" --tracking_metrics_dir "$TM" \
    --confidence_calib outputs/calibration/${c}_confidence.json \
    --output_dir "$RUN/paper_metrics" --recovery_k 30 || say "  FAIL paper-metrics $t"
fi
$PY tools/confusion_uav123.py   --trackers $t --dataset uav123 --run_tag ${t}_r3_passive --save
$PY tools/collapse_cu_3state.py --trackers $t --dataset uav123 --run_tag ${t}_r3_passive --save

say "=== final_paper_chain DONE — update FINAL_REPORT §2(@10fps)/§4(R4)/§5(OSTrack) ==="
