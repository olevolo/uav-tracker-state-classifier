#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_ortrack_calibration_baselines.sh
# Runs ORTrack baselines on DTB70 then VisDrone-SOT sequentially
# (same adapter sys.path — cannot run in parallel).
# These are calibration datasets for CSC v2.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

echo ""
echo "=========================================="
echo " ORTrack Calibration Baselines"
echo "=========================================="
echo "  Datasets: DTB70 → VisDrone-SOT"
echo ""

echo "[1/2] ORTrack DTB70..."
"$PYTHON" "$PROJECT_ROOT/tools/run_baseline.py" \
    --tracker ortrack \
    --dataset dtb70 \
    --split test \
    --device cpu \
    --skip_existing
echo "  DTB70 done."

echo "[2/2] ORTrack VisDrone-SOT..."
"$PYTHON" "$PROJECT_ROOT/tools/run_baseline.py" \
    --tracker ortrack \
    --dataset visdrone_sot \
    --split test \
    --device cpu \
    --skip_existing
echo "  VisDrone-SOT done."

echo ""
echo "ORTrack calibration baselines complete."
echo "Next: run tools/fit_calibration.py --tracker ortrack (GOT-10k + DTB70 + VisDrone)"
