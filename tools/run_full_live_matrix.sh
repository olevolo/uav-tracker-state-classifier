#!/bin/bash
# FULL LIVE diagnosis matrix — BOTH calibration ways. NO TRAINING.
# Tracker actually runs every cell (LIVE passive run_with_csc; CSC observes only).
# Resumable (skip on metrics.json), logged, bash-3.2-safe (macOS /bin/bash).
#
#   trackers : sglatrack avtrack ortrack ostrack
#              (FARTrack DROPPED; EVPTrack dropped; UETrack STUB — clip OK but
#               model build still fails on a missing pretrain file -> excluded)
#   datasets : uav123  uav123_10fps  uavtrack112  dtb70
#
# CALIBRATION — user: "прожени всі і так і так" + "all trackers same as sglatrack"
#   PRIMARY (uniform/clean): ALL 4 trackers use *_aerial_v2  (GOT10k+DTB70+VisDrone,
#            n~69k, IDENTICAL recipe across trackers). Keeps uav123 + uav123_10fps +
#            UAVTrack112 as clean held-out (none of them in the aerial fit).
#            run_tag = <t>_r3_passive
#   ROBUSTNESS (sglatrack only): also under sglatrack_all_v2 (754k; incl LaSOT/UAVDT/
#            UAVTrack112) -> shows whether the broader fit changes the numbers.
#            run_tag = sglatrack_r3_all_v2
#   av/or/os have NO all_v2 (no LaSOT/UAVDT/UAVTrack112 telemetry) -> aerial only.
#
# INTEGRITY: offline SOT benchmarking, diagnosis-only. R3-fcw3 = validation-selected
# paper model, reused unchanged (no test-set selection). Per dataset:
#   uav123 / uav123_10fps : FINAL TEST   — clean (never trained/tuned/calibrated on)
#   uavtrack112           : held-out     — clean under aerial_v2 (NOT clean under all_v2)
#   dtb70                 : IN-SAMPLE/circular (in classifier training AND aerial fit)
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
mkdir -p outputs/_logs
LOG=outputs/_logs/full_live_matrix.log
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
exec >>"$LOG" 2>&1
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

say "=== full_live_matrix (both-ways) START ==="
[ -f "$CKPT" ] || { say "FATAL: checkpoint missing: $CKPT"; exit 1; }

while pgrep -f "train_csc|run_baseline\.py" >/dev/null 2>&1; do
  say "waiting for heavy proc to finish ..."; sleep 30
done
say "CPU free — proceeding."

aerial_for() {
  case "$1" in
    sglatrack) echo sglatrack_aerial_v2 ;;
    avtrack)   echo avtrack_aerial_v2 ;;
    ortrack)   echo ortrack_aerial_v2 ;;
    ostrack)   echo ostrack_aerial_v2 ;;
    *)         echo "" ;;
  esac
}

# run_cell <tracker> <dataset> <calib_prefix> <run_tag>
# LIVE passive run (skip iff metrics.json present) -> shadow symlink -> labels ->
# tracking metrics -> paper metrics. Each cell self-contained & idempotent.
run_cell() {
  local t="$1" ds="$2" c="$3" tag="$4"
  local RUN=outputs/eval/$t/$ds/test/$tag
  say ">>> $t / $ds  calib=$c  tag=$tag"
  [ "$ds" = "dtb70" ] && say "    NOTE dtb70 = IN-SAMPLE/circular — sanity cell only"
  if [ ! -f "$RUN/metrics.json" ]; then
    rm -rf "$RUN/states" "$RUN/paper_metrics" "$RUN/tracking_metrics"
    $PY -u tools/run_with_csc.py --tracker "$t" --dataset "$ds" --split test \
      --csc_checkpoint "$CKPT" --csc_mode passive --calibration_prefix "$c" --device cpu \
      --output_dir "outputs/eval/$t/$ds/test" --run_tag "$tag" \
      || { say "    FAIL run_with_csc $t/$ds/$tag"; return 1; }
  else say "    skip LIVE (metrics.json exists)"; fi
  [ -d "$RUN/states" ] || { say "    no states — skip downstream"; return 1; }

  local SHADOW=outputs/_live_matrix_base/$tag/$t/$ds/test
  mkdir -p "$SHADOW"
  ln -sfn "$(pwd)/$RUN/predictions" "$SHADOW/predictions"
  ln -sfn "$(pwd)/$RUN/telemetry"   "$SHADOW/telemetry"

  local LAB=$RUN/labels_v3
  if [ ! -f "$LAB/$ds/test/labels.jsonl" ]; then
    $PY -u tools/build_scene_state_labels.py --tracker "$t" --dataset "$ds" --split test \
      --baseline_dir outputs/_live_matrix_base/"$tag"/"$t" --calibration_dir outputs/calibration \
      --calibrator_tag "$c" --output_dir "$LAB" || say "    FAIL labels $t/$ds/$tag"
  else say "    skip labels"; fi

  local TM=$RUN/tracking_metrics
  if [ ! -f "$TM/metrics_summary.json" ]; then
    $PY -u tools/evaluate_tracking_results.py --dataset "$ds" --split test \
      --pred_dir "$RUN/predictions" --telemetry_dir "$RUN/telemetry" \
      --output_dir "$TM" || say "    FAIL track-metrics $t/$ds/$tag"
  else say "    skip track-metrics"; fi

  if [ ! -f "$RUN/paper_metrics/paper_metrics.json" ]; then
    $PY -u tools/compute_paper_metrics.py --tracker "$t" --dataset "$ds" --split test \
      --predictions_dir "$RUN/predictions" --states_dir "$RUN/states" --telemetry_dir "$RUN/telemetry" \
      --labels_dir "$LAB/$ds/test" --tracking_metrics_dir "$TM" \
      --confidence_calib outputs/calibration/${c}_confidence.json \
      --output_dir "$RUN/paper_metrics" --recovery_k 30 || say "    FAIL paper-metrics $t/$ds/$tag"
  else say "    skip paper-metrics"; fi
  say "    cell done: $t/$ds/$tag"
}

TRACKERS="sglatrack avtrack ortrack ostrack"
DATASETS="uav123 uav123_10fps uavtrack112 dtb70"

# sglatrack _r3_passive previously used all_v2 — drop it so the PRIMARY pass
# below re-creates it under aerial_v2 (after we preserve all_v2 separately).
for ds in $DATASETS; do
  R=outputs/eval/sglatrack/$ds/test/sglatrack_r3_passive
  [ -d "$R" ] && { say "reset stale sglatrack/$ds/_r3_passive (was all_v2)"; rm -rf "$R"; }
done

# ---- ROBUSTNESS pass first: sglatrack under all_v2 (preserve the broader fit) ----
for ds in $DATASETS; do
  run_cell sglatrack "$ds" sglatrack_all_v2 sglatrack_r3_all_v2
done

# ---- PRIMARY pass: uniform aerial_v2 for ALL 4 trackers ----
for t in $TRACKERS; do
  c=$(aerial_for "$t")
  [ -f "outputs/calibration/${c}_confidence.json" ] || { say "SKIP $t — aerial calibrator $c missing"; continue; }
  for ds in $DATASETS; do
    run_cell "$t" "$ds" "$c" "${t}_r3_passive"
  done
done

# ---- per-dataset confusion + 3-state collapse over the uniform aerial matrix ----
for ds in $DATASETS; do
  say "confusion+collapse $ds (4 trackers, aerial)"
  $PY tools/confusion_uav123.py   --trackers $TRACKERS --dataset "$ds" --save || say "  FAIL confusion $ds"
  $PY tools/collapse_cu_3state.py --trackers $TRACKERS --dataset "$ds" --save || say "  FAIL collapse $ds"
done

say "=== full_live_matrix (both-ways) DONE — consolidate into ONE doc; clean stale scratch docs ==="
