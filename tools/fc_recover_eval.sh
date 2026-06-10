#!/bin/bash
# Auto-evaluate fc_recover_v1 results: compute AUC on predictions, compare to baseline,
# update FINAL_REPORT macro table. Run after the main background run completes.
#
# Usage:
#   bash tools/fc_recover_eval.sh <run_dir>
# where <run_dir> has the structure: <output_dir>/<tracker>_<dataset>_<split>_<ckpt>/
#
set -euo pipefail
RUN_DIR="${1:-outputs/fc_recover_v1/full_uav123_with_detector/sglatrack_uav123_test_checkpoint_best}"
EVAL_DIR="${RUN_DIR%/sglatrack_uav123_test_checkpoint_best}_eval"
COMPARE_DIR="${RUN_DIR%/sglatrack_uav123_test_checkpoint_best}_compare"
LABEL="${2:-fc_recover + RT-DETR}"

cd "$(git rev-parse --show-toplevel)"

if [ ! -d "$RUN_DIR/predictions" ]; then
  echo "ERROR: $RUN_DIR/predictions not found"; exit 1
fi

echo "[1/3] evaluate predictions -> $EVAL_DIR"
PYTHONPATH=src:csc_uav_tracking_sdk/src:salrtd/src:. .venv/bin/python \
  tools/evaluate_tracking_results.py \
    --dataset uav123 --split test \
    --pred_dir "$RUN_DIR/predictions" \
    --telemetry_dir "$RUN_DIR/states" \
    --output_dir "$EVAL_DIR" 2>&1 | tail -3

echo
echo "[2/3] compare vs baseline (eval5_clamp passive)"
mkdir -p "$COMPARE_DIR"
PYTHONPATH=src:. .venv/bin/python tools/compare_tracking_runs.py \
  --baseline outputs/fc_recover_v1/baseline_eval5_clamp \
  --candidate "$EVAL_DIR" \
  --output "$COMPARE_DIR" \
  --label_baseline "V3 passive (csc_prod)" \
  --label_candidate "$LABEL" 2>&1

echo
echo "[3/3] aggregate fc_recover counters from metrics.json"
PYTHONPATH=src .venv/bin/python -c "
import json
m = json.load(open('$RUN_DIR/metrics.json'))
print('  control_fc_recover_starts:', m.get('control_fc_recover_starts'))
print('  control_fc_recover_switches:', m.get('control_fc_recover_switches'))
print('  control_fc_recover_commits:', m.get('control_fc_recover_commits'))
print('  control_fc_recover_rollbacks:', m.get('control_fc_recover_rollbacks'))
print('  control_fc_recover_aborts:', m.get('control_fc_recover_aborts'))
print('  control_fc_recover_distractor_seeds:', m.get('control_fc_recover_distractor_seeds'))
print('  control_fc_recover_redetect_calls:', m.get('control_fc_recover_redetect_calls'))
print('  mean_total_fps:', m.get('mean_total_fps'))
print('  control_fc_recover_active_frames:', m.get('control_fc_recover_active_frames'))
print('  control_fc_recover_verified_total:', m.get('control_fc_recover_verified_total'))
"
echo
echo "Done. Compare summary: $COMPARE_DIR/compare_summary.json"
