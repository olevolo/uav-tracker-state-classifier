#!/usr/bin/env bash
# Auto pipeline: wait for SGLATrack baseline (PID $1) → SGLATrack pipeline →
# ORTrack baseline restart → ORTrack pipeline → done.
#
# Long-running (~6-8 hours). Designed to run via nohup in background.
#
# Usage:
#   nohup bash tools/auto_full_pipeline.sh <SGLA_PID> > logs/auto_pipeline.log 2>&1 &
set -u

SGLA_PID="${1:?Usage: $0 <SGLATrack baseline PID>}"
cd "$(dirname "$0")/.."

PY=".venv/bin/python -u"
TS() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[auto_pipeline] $(TS) $*"; }
fail() { log "FATAL: $*"; exit 1; }

require_220() {
  local tracker="$1"
  local n
  n=$(ls "outputs/baselines/$tracker/lasot/train/predictions/" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -lt 220 ]; then
    fail "$tracker baseline only $n/220 sequences. Pipeline aborted."
  fi
  log "$tracker baseline complete: $n/220"
}

wait_for_pid() {
  local pid="$1" name="$2"
  log "Waiting for $name (PID $pid) to finish..."
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
  done
  log "$name (PID $pid) exited."
}

run_pipeline() {
  local tracker="$1"
  log "=== Pipeline for $tracker ==="

  log "Step 1/3: post-baseline pipeline (calibrate + labels + train CSC)"
  bash tools/run_post_baseline_pipeline.sh "$tracker" \
    || fail "post-baseline pipeline failed for $tracker"

  local ckpt="outputs/csc_training/${tracker}_lasot_tcn16/checkpoint_best.pth"
  if [ ! -f "$ckpt" ]; then
    fail "CSC checkpoint missing for $tracker: $ckpt"
  fi

  log "Step 2/3: UAV123 final-eval"
  CSC_NOT_TRAINED_ON_UAV123=1 \
    bash tools/run_uav123_final_eval.sh "$tracker" "$ckpt" \
    || fail "UAV123 final-eval failed for $tracker"

  log "Step 3/3: $tracker pipeline DONE"
  log "  Final report: outputs/eval/${tracker}/uav123/test/FINAL_REPORT.md (or similar)"
}

# ----------------------------------------------------------------------------
log "auto_full_pipeline START — SGLA_PID=$SGLA_PID"

# Stage 1: wait for SGLATrack baseline
wait_for_pid "$SGLA_PID" "SGLATrack baseline"
require_220 "sglatrack"

# Stage 2: SGLATrack pipeline
run_pipeline "sglatrack"

# Stage 3: ORTrack baseline restart
log "=== Launching ORTrack baseline restart ==="
ORTRACK_LOG="logs/baselines/ortrack_lasot_$(date +%Y%m%d_%H%M%S).log"
nohup $PY tools/run_baseline.py \
  --tracker ortrack \
  --dataset lasot \
  --split train \
  --device cpu \
  --skip_existing \
  > "$ORTRACK_LOG" 2>&1 &
ORTRACK_PID=$!
echo "$ORTRACK_PID" > logs/baselines/ortrack.pid
log "ORTrack baseline launched PID=$ORTRACK_PID, log=$ORTRACK_LOG"

# Brief wait + alive check
sleep 10
if ! kill -0 "$ORTRACK_PID" 2>/dev/null; then
  fail "ORTrack baseline died within 10s of launch — check $ORTRACK_LOG"
fi
log "ORTrack baseline confirmed alive"

# Stage 4: wait for ORTrack baseline
wait_for_pid "$ORTRACK_PID" "ORTrack baseline"
require_220 "ortrack"

# Stage 5: ORTrack pipeline
run_pipeline "ortrack"

log "=== ALL DONE ==="
log "SGLATrack final report: outputs/eval/sglatrack/uav123/test/"
log "ORTrack  final report: outputs/eval/ortrack/uav123/test/"
