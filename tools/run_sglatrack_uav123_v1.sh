#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_sglatrack_uav123_v1.sh
# Re-runs SGLATrack UAV123 baseline (full 11-feature telemetry),
# clears stale eval outputs, then runs the full eval pipeline
# with the LaSOT calibrator and CSC v1 checkpoint.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

CSC_CKPT="$PROJECT_ROOT/outputs/csc_training/sglatrack_lasot_tcn16/checkpoint_best.pth"
EVAL_ROOT="$PROJECT_ROOT/outputs/eval/sglatrack/uav123/test"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOG_DIR"

echo ""
echo "=========================================="
echo " SGLATrack UAV123 V1 Full Pipeline"
echo "=========================================="
echo "  Checkpoint: $CSC_CKPT"
echo ""

# Step 1: Re-run baseline (overwrites old 3-field telemetry with full 11-feature schema)
echo "[1/3] Re-running SGLATrack UAV123 baseline..."
"$PYTHON" "$PROJECT_ROOT/tools/run_baseline.py" \
    --tracker sglatrack \
    --dataset uav123 \
    --split test \
    --device cpu
echo "  Baseline done."

# Step 2: Clean stale eval outputs (wrong calibrator + incomplete telemetry)
echo "[2/3] Cleaning stale eval outputs..."
rm -rf "$EVAL_ROOT/passive"
rm -rf "$EVAL_ROOT/labels"
rm -rf "$EVAL_ROOT/tracking_metrics"
rm -rf "$EVAL_ROOT/episode_metrics"
rm -rf "$EVAL_ROOT/paper_metrics"
rm -f  "$EVAL_ROOT/FINAL_REPORT.md"
echo "  Cleaned."

# Step 3: Run eval pipeline (LaSOT calibrator, CSC v1)
echo "[3/3] Running eval pipeline..."
export CSC_NOT_TRAINED_ON_UAV123=1
bash "$PROJECT_ROOT/tools/run_uav123_final_eval.sh" sglatrack "$CSC_CKPT"
echo "  Done."
