#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_tracker_v2_baselines.sh <tracker>
#
# For OSTrack / AVTrack / EVPTrack:
# 1. Run baseline on GOT-10k val + DTB70 + VisDrone-SOT + UAV123
# 2. Fit aerial_v2 calibration (GOT-10k + DTB70 + VisDrone)
#
# UAV123 guard: no calibration from UAV123.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <tracker>  (ostrack|avtrack|evptrack)" >&2
    exit 1
fi
TRACKER="$1"

echo ""
echo "=========================================="
echo " V2 Baseline Pipeline: $TRACKER"
echo "=========================================="

run_ds() {
    local ds="$1" split="$2"
    echo "[$TRACKER] $ds/$split..."
    "$PYTHON" "$PROJECT_ROOT/tools/run_baseline.py" \
        --tracker "$TRACKER" \
        --dataset "$ds" \
        --split "$split" \
        --device cpu \
        --skip_existing
    echo "[$TRACKER] $ds/$split done."
}

run_ds got10k  val
run_ds dtb70   test
run_ds visdrone_sot test
run_ds uav123  test

echo "[$TRACKER] Fitting aerial_v2 calibration..."
"$PYTHON" "$PROJECT_ROOT/tools/fit_calibration.py" \
    --tracker "$TRACKER" \
    --telemetry_dirs \
        "$PROJECT_ROOT/outputs/baselines/$TRACKER/got10k/val/telemetry" \
        "$PROJECT_ROOT/outputs/baselines/$TRACKER/dtb70/test/telemetry" \
        "$PROJECT_ROOT/outputs/baselines/$TRACKER/visdrone_sot/test/telemetry" \
    --tag aerial_v2 \
    --output_dir "$PROJECT_ROOT/outputs/calibration"

echo ""
echo "[$TRACKER] V2 baselines + calibration COMPLETE."
