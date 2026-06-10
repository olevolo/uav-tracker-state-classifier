#!/usr/bin/env bash
# Post-LaSOT pipeline for one tracker.
# Run after the baseline pass on LaSOT is complete.
# Usage: bash tools/run_post_baseline_pipeline.sh <tracker>
# Example: bash tools/run_post_baseline_pipeline.sh sglatrack
# This is called automatically by run_full_chain.sh but can also be run manually.
set -e
cd "$(dirname "$0")/.."

TRACKER="${1:?Usage: $0 <tracker_name>}"
DATASET="lasot"
SPLIT="train"
PY=".venv/bin/python -u"
BASELINE_DIR="outputs/baselines/$TRACKER/$DATASET/$SPLIT"
CALIB_DIR="outputs/calibration"
LABELS_DIR="outputs/csc_labels/$TRACKER/$DATASET/$SPLIT"

echo "[$(date '+%H:%M:%S')] === Post-baseline pipeline: tracker=$TRACKER dataset=$DATASET ===" >&2

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Fit calibrator from LaSOT telemetry
# ──────────────────────────────────────────────────────────────────────────────
TELEMETRY_DIR="$BASELINE_DIR/telemetry"
if [ ! -f "$CALIB_DIR/${TRACKER}_lasot_confidence.json" ]; then
  echo "[$(date '+%H:%M:%S')] Step 1/3 Calibrate $TRACKER from LaSOT telemetry..." >&2
  $PY -u tools/fit_calibration.py \
    --tracker "$TRACKER" \
    --telemetry_dir "$TELEMETRY_DIR" \
    --output_dir "$CALIB_DIR" \
    --features confidence apce psr
else
  echo "[$(date '+%H:%M:%S')] Step 1/3 Calibrator already exists, skipping." >&2
fi

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Generate CSC state labels from LaSOT predictions + GT
# ──────────────────────────────────────────────────────────────────────────────
if [ ! -d "$LABELS_DIR" ] || [ -z "$(ls -A "$LABELS_DIR" 2>/dev/null)" ]; then
  echo "[$(date '+%H:%M:%S')] Step 2/3 Generate labels for $TRACKER on LaSOT..." >&2
  $PY -u tools/build_scene_state_labels.py \
    --tracker "$TRACKER" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --baseline_dir "outputs/baselines/$TRACKER" \
    --output_dir "$LABELS_DIR" \
    --calibration_dir "$CALIB_DIR"
else
  echo "[$(date '+%H:%M:%S')] Step 2/3 Labels already exist, skipping." >&2
fi

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Train CSC-TCN16 on LaSOT labels
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_OUT="outputs/csc_training/${TRACKER}_lasot_tcn16"
if [ ! -f "$TRAIN_OUT/checkpoint_best.pth" ]; then
  echo "[$(date '+%H:%M:%S')] Step 3/3 Train CSC-TCN16 for $TRACKER on LaSOT..." >&2
  $PY -u tools/train_csc.py \
    --config configs/csc/csc_tcn16.yaml \
    --labels_dir "$LABELS_DIR" \
    --output_dir "$TRAIN_OUT" \
    --device cpu
else
  echo "[$(date '+%H:%M:%S')] Step 3/3 CSC checkpoint already exists, skipping." >&2
fi

echo "[$(date '+%H:%M:%S')] === Pipeline done for $TRACKER ===" >&2
echo "Outputs:"
echo "  Calibrator : $CALIB_DIR/${TRACKER}_lasot_confidence.json"
echo "  Labels     : $LABELS_DIR/"
echo "  CSC model  : $TRAIN_OUT/checkpoint_best.pth"
