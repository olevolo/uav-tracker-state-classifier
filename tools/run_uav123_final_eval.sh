#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_uav123_final_eval.sh — UAV123 final evaluation pipeline
#
# Usage:
#   tools/run_uav123_final_eval.sh <tracker> <csc_checkpoint>
#
#   <tracker>         : sglatrack | ortrack
#   <csc_checkpoint>  : absolute or relative path to the .pth checkpoint
#
# Safety gate:
#   The script refuses to proceed unless CSC_NOT_TRAINED_ON_UAV123=1 is
#   set in the environment.  This is a deliberate tripwire that forces the
#   caller to acknowledge the constraint from CLAUDE.md §Research Constraints:
#   "UAV123 must be used ONLY for final evaluation — NEVER train CSC on it."
#
#   The checkpoint path is also checked: if it contains "uav123" anywhere
#   the script aborts with an error.
#
# Steps:
#   1. Sanity-check checkpoint provenance (name must not contain "uav123")
#   2. Assert baseline exists (does not re-run it)
#   3. Assert calibrator exists (fails loud if missing)
#   4. Run run_with_csc.py in PASSIVE mode → outputs/eval/<tracker>/uav123/test/passive/
#   5. Run build_scene_state_labels.py for evaluation-only GT labels
#      → outputs/eval/<tracker>/uav123/test/labels/
#   6. Run evaluate_tracking_results.py → AUC / Precision / FPS
#   7. Run evaluate_csc_episodes.py → Recall@5/10, FA, delay
#   8. Run compute_paper_metrics.py → FCR, FCD, TTFC, Recovery@30,
#      State-Conditioned AUC, State Transition Matrix
#   9. Append all results to outputs/eval/<tracker>/uav123/test/FINAL_REPORT.md
#
# Notes:
#   - Never touches outputs/baselines/{sglatrack,ortrack}/lasot/  (baselines running)
#   - UAV123 labels generated here are used for metric computation only,
#     never fed back into model training
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Arg parsing
# ---------------------------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <tracker> <csc_checkpoint>" >&2
    echo "  tracker: sglatrack | ortrack" >&2
    echo "  csc_checkpoint: path to .pth file" >&2
    echo "" >&2
    echo "  Must set: CSC_NOT_TRAINED_ON_UAV123=1  (safety gate)" >&2
    exit 1
fi

TRACKER="$1"
CSC_CKPT="$2"

# ---------------------------------------------------------------------------
# 1. Safety gate — explicit acknowledgement required
# ---------------------------------------------------------------------------
if [[ "${CSC_NOT_TRAINED_ON_UAV123:-0}" != "1" ]]; then
    echo "" >&2
    echo "ERROR: Safety gate not cleared." >&2
    echo "" >&2
    echo "  The CSC model must NEVER be trained on UAV123 (CLAUDE.md §Research Constraints)." >&2
    echo "  Before running this script, verify that your checkpoint was trained on" >&2
    echo "  LaSOT / GOT-10k / DTB70 / VisDrone-SOT only — NEVER on UAV123." >&2
    echo "" >&2
    echo "  Once verified, set the environment variable and re-run:" >&2
    echo "    export CSC_NOT_TRAINED_ON_UAV123=1" >&2
    echo "    $0 $*" >&2
    exit 1
fi

# Checkpoint name must not contain "uav123" (catches obvious path mistakes)
CKPT_LOWER="${CSC_CKPT,,}"
if [[ "$CKPT_LOWER" == *"uav123"* ]]; then
    echo "ERROR: CSC checkpoint path contains 'uav123': $CSC_CKPT" >&2
    echo "  This strongly suggests the model was trained on UAV123." >&2
    echo "  Aborting to protect evaluation integrity." >&2
    exit 1
fi

# Tracker must be sglatrack or ortrack
if [[ "$TRACKER" != "sglatrack" && "$TRACKER" != "ortrack" ]]; then
    echo "ERROR: tracker must be 'sglatrack' or 'ortrack', got: $TRACKER" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

BASELINE_DIR="$PROJECT_ROOT/outputs/baselines/$TRACKER/uav123/test"
CALIBRATION_DIR="$PROJECT_ROOT/outputs/calibration"
EVAL_ROOT="$PROJECT_ROOT/outputs/eval/$TRACKER/uav123/test"
PASSIVE_DIR="$EVAL_ROOT/passive"
LABELS_DIR="$EVAL_ROOT/labels"
TRACKING_EVAL_DIR="$EVAL_ROOT/tracking_metrics"
EPISODE_EVAL_DIR="$EVAL_ROOT/episode_metrics"
PAPER_METRICS_DIR="$EVAL_ROOT/paper_metrics"
FINAL_REPORT="$EVAL_ROOT/FINAL_REPORT.md"

DATASET="uav123"
SPLIT="test"

# Resolve absolute checkpoint path
CSC_CKPT="$(python3 -c "import os, sys; print(os.path.realpath(sys.argv[1]))" "$CSC_CKPT")"

echo ""
echo "=========================================="
echo " UAV123 Final Evaluation Pipeline"
echo "=========================================="
echo "  Tracker    : $TRACKER"
echo "  Checkpoint : $CSC_CKPT"
echo "  Output root: $EVAL_ROOT"
echo ""

mkdir -p "$EVAL_ROOT"

# ---------------------------------------------------------------------------
# Step 1: Assert CSC checkpoint exists
# ---------------------------------------------------------------------------
echo "[1/8] Checking CSC checkpoint..."
if [[ ! -f "$CSC_CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CSC_CKPT" >&2
    exit 1
fi
echo "  OK: $CSC_CKPT"

# ---------------------------------------------------------------------------
# Step 2: Assert baseline exists
# ---------------------------------------------------------------------------
echo "[2/8] Checking baseline at $BASELINE_DIR..."
if [[ ! -d "$BASELINE_DIR/predictions" ]]; then
    echo "ERROR: baseline predictions not found at $BASELINE_DIR/predictions" >&2
    echo "  Run the baseline first (but NOT via this script; baseline is separate)." >&2
    exit 1
fi
N_PREDS=$(ls "$BASELINE_DIR/predictions/"*.txt 2>/dev/null | wc -l | tr -d ' ')
echo "  OK: $N_PREDS prediction files found"
if [[ "$N_PREDS" -lt 120 ]]; then
    echo "WARNING: expected 123 sequences, found only $N_PREDS" >&2
fi

# ---------------------------------------------------------------------------
# Step 3: Assert calibrator exists
# ---------------------------------------------------------------------------
echo "[3/8] Checking calibrator..."
# Determine which calibrator to require based on tracker + training dataset
if [[ "$TRACKER" == "ortrack" ]]; then
    CALIB_CONF="$CALIBRATION_DIR/ortrack_lasot_confidence.json"
    CALIB_MANIFEST="$CALIBRATION_DIR/ortrack_lasot.manifest.json"
elif [[ "$TRACKER" == "sglatrack" ]]; then
    # SGLATrack was calibrated on GOT-10k (Issue 5 in memory — lasot calibrator missing)
    CALIB_CONF="$CALIBRATION_DIR/sglatrack_got10k_confidence.json"
    CALIB_MANIFEST="$CALIBRATION_DIR/sglatrack_got10k.manifest.json"
fi

if [[ ! -f "$CALIB_CONF" ]]; then
    echo "ERROR: calibrator not found: $CALIB_CONF" >&2
    echo "  Run tools/fit_calibration.py first for $TRACKER." >&2
    exit 1
fi
echo "  OK: $CALIB_CONF"

# Emit a warning for SGLATrack: calibrator is GOT-10k not LaSOT (known Issue 5)
if [[ "$TRACKER" == "sglatrack" ]]; then
    echo "  WARNING: SGLATrack calibrator was fitted on GOT-10k, not LaSOT." >&2
    echo "           This is a known gap (Issue 5). Results may have systematically" >&2
    echo "           miscalibrated confidence thresholds for SGLATrack." >&2
fi

# ---------------------------------------------------------------------------
# Step 4: CSC passive inference
# ---------------------------------------------------------------------------
echo "[4/8] Running CSC passive inference → $PASSIVE_DIR ..."
if [[ -f "$PASSIVE_DIR/metrics.json" ]]; then
    echo "  SKIP: $PASSIVE_DIR/metrics.json already exists (delete to rerun)"
else
    "$PYTHON" -u "$PROJECT_ROOT/tools/run_with_csc.py" \
        --tracker "$TRACKER" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --csc_checkpoint "$CSC_CKPT" \
        --csc_mode passive \
        --output_dir "$EVAL_ROOT" \
        --device "${DEVICE:-cpu}"
    # run_with_csc writes to outputs/csc_runs/<run_tag>/; rename to passive/
    # Actually run_with_csc writes inside output_dir/<run_tag>/ — we need to
    # find the latest run_tag directory and symlink/move it.
    RUN_TAG=$(ls -td "$EVAL_ROOT"/*/metrics.json 2>/dev/null | head -1 | xargs -I{} dirname {} | xargs basename)
    if [[ -n "$RUN_TAG" && "$RUN_TAG" != "passive" ]]; then
        echo "  Moving $EVAL_ROOT/$RUN_TAG → $PASSIVE_DIR"
        mv "$EVAL_ROOT/$RUN_TAG" "$PASSIVE_DIR"
    fi
    echo "  Done."
fi

# ---------------------------------------------------------------------------
# Step 5: GT label generation (eval-only, never used for training)
# ---------------------------------------------------------------------------
echo "[5/8] Generating GT state labels for evaluation → $LABELS_DIR ..."
if [[ -d "$LABELS_DIR" ]] && [[ -n "$(ls "$LABELS_DIR"/*.jsonl 2>/dev/null | head -1)" ]]; then
    echo "  SKIP: labels already generated (delete $LABELS_DIR to rerun)"
else
    mkdir -p "$LABELS_DIR"
    "$PYTHON" -u "$PROJECT_ROOT/tools/build_scene_state_labels.py" \
        --tracker "$TRACKER" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --baseline_dir "$PROJECT_ROOT/outputs/baselines/$TRACKER" \
        --calibration_dir "$CALIBRATION_DIR" \
        --output_dir "$LABELS_DIR"
    echo "  Done."
fi

# ---------------------------------------------------------------------------
# Step 6: Standard tracking metrics (AUC / Precision / FPS)
# ---------------------------------------------------------------------------
echo "[6/8] Computing AUC / Precision / FPS → $TRACKING_EVAL_DIR ..."
if [[ -f "$TRACKING_EVAL_DIR/summary.json" ]]; then
    echo "  SKIP: summary.json exists (delete $TRACKING_EVAL_DIR to rerun)"
else
    mkdir -p "$TRACKING_EVAL_DIR"
    "$PYTHON" -u "$PROJECT_ROOT/tools/evaluate_tracking_results.py" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --pred_dir "$BASELINE_DIR/predictions" \
        --telemetry_dir "$BASELINE_DIR/telemetry" \
        --output_dir "$TRACKING_EVAL_DIR"
    echo "  Done."
fi

# ---------------------------------------------------------------------------
# Step 7: Episode-level CSC metrics (Recall@5/10, FA/1000, delay)
# ---------------------------------------------------------------------------
echo "[7/8] Computing episode metrics → $EPISODE_EVAL_DIR ..."
if [[ -f "$EPISODE_EVAL_DIR/episode_summary.json" ]]; then
    echo "  SKIP: episode_summary.json exists (delete $EPISODE_EVAL_DIR to rerun)"
else
    mkdir -p "$EPISODE_EVAL_DIR"
    STATES_DIR="$PASSIVE_DIR/states"
    if [[ ! -d "$STATES_DIR" ]]; then
        echo "ERROR: CSC states dir not found: $STATES_DIR" >&2
        echo "  Step 4 (CSC passive inference) must succeed first." >&2
        exit 1
    fi
    "$PYTHON" -u "$PROJECT_ROOT/tools/evaluate_csc_episodes.py" \
        --labels "$LABELS_DIR" \
        --predictions "$STATES_DIR" \
        --out "$EPISODE_EVAL_DIR"
    echo "  Done."
fi

# ---------------------------------------------------------------------------
# Step 8: Paper metrics (FCR / FCD / TTFC / Recovery@30 / State-Cond AUC / Transition Matrix)
# ---------------------------------------------------------------------------
echo "[8/8] Computing paper metrics → $PAPER_METRICS_DIR ..."
if [[ -f "$PAPER_METRICS_DIR/paper_metrics.csv" ]]; then
    echo "  SKIP: paper_metrics.csv exists (delete $PAPER_METRICS_DIR to rerun)"
else
    mkdir -p "$PAPER_METRICS_DIR"
    "$PYTHON" -u "$PROJECT_ROOT/tools/compute_paper_metrics.py" \
        --tracker "$TRACKER" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --predictions_dir "$BASELINE_DIR/predictions" \
        --telemetry_dir "$BASELINE_DIR/telemetry" \
        --states_dir "$PASSIVE_DIR/states" \
        --labels_dir "$LABELS_DIR" \
        --tracking_metrics_dir "$TRACKING_EVAL_DIR" \
        --output_dir "$PAPER_METRICS_DIR" \
        --confidence_calib "$CALIB_CONF" \
        --recovery_k 30
    echo "  Done."
fi

# ---------------------------------------------------------------------------
# Assemble FINAL_REPORT.md
# ---------------------------------------------------------------------------
echo ""
echo "Assembling FINAL_REPORT.md → $FINAL_REPORT ..."

{
    echo "# UAV123 Final Evaluation Report"
    echo ""
    echo "**Tracker:** $TRACKER"
    echo "**Checkpoint:** $CSC_CKPT"
    echo "**Generated:** $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo ""
    echo "---"
    echo ""
    echo "## Standard Tracking Metrics (AUC / Precision / FPS)"
    echo ""
    if [[ -f "$TRACKING_EVAL_DIR/summary.json" ]]; then
        "$PYTHON" -c "
import json, sys
with open('$TRACKING_EVAL_DIR/summary.json') as f:
    d = json.load(f)
print('| Metric | Value |')
print('|--------|-------|')
for k, v in d.items():
    if isinstance(v, float):
        print(f'| {k} | {v:.4f} |')
    else:
        print(f'| {k} | {v} |')
" 2>/dev/null || echo "_summary.json not parseable_"
    else
        echo "_Not computed yet._"
    fi
    echo ""
    echo "---"
    echo ""
    echo "## CSC Episode Metrics (Recall@5/10, FA/1000, Delay)"
    echo ""
    if [[ -f "$EPISODE_EVAL_DIR/episode_summary.json" ]]; then
        "$PYTHON" -c "
import json
with open('$EPISODE_EVAL_DIR/episode_summary.json') as f:
    d = json.load(f)
print('| Metric | Value |')
print('|--------|-------|')
def _flat(d, prefix=''):
    for k, v in d.items():
        key = f'{prefix}{k}' if prefix else k
        if isinstance(v, dict):
            yield from _flat(v, key+'.')
        elif isinstance(v, float):
            yield key, f'{v:.4f}'
        else:
            yield key, str(v)
for k, v in _flat(d):
    print(f'| {k} | {v} |')
" 2>/dev/null || echo "_episode_summary.json not parseable_"
    else
        echo "_Not computed yet._"
    fi
    echo ""
    echo "---"
    echo ""
    echo "## Paper Metrics (FCR / FCD / TTFC / Recovery@30 / State-Conditioned AUC)"
    echo ""
    if [[ -f "$PAPER_METRICS_DIR/QUALITY_REPORT.md" ]]; then
        cat "$PAPER_METRICS_DIR/QUALITY_REPORT.md"
    else
        echo "_Not computed yet._"
    fi
    echo ""
    echo "---"
    echo ""
    echo "## State Transition Matrix"
    echo ""
    if [[ -f "$PAPER_METRICS_DIR/state_transition_matrix.csv" ]]; then
        "$PYTHON" -c "
import csv, sys
rows = list(csv.reader(open('$PAPER_METRICS_DIR/state_transition_matrix.csv')))
for r in rows:
    print('| ' + ' | '.join(r) + ' |')
" 2>/dev/null || cat "$PAPER_METRICS_DIR/state_transition_matrix.csv"
    else
        echo "_Not computed yet._"
    fi
    echo ""
} > "$FINAL_REPORT"

echo ""
echo "=========================================="
echo " DONE"
echo "=========================================="
echo ""
echo "  Final report: $FINAL_REPORT"
echo "  Paper metrics: $PAPER_METRICS_DIR/paper_metrics.csv"
echo "  Quality report: $PAPER_METRICS_DIR/QUALITY_REPORT.md"
echo ""
echo "  To run (example):"
echo "    CSC_NOT_TRAINED_ON_UAV123=1 $0 $TRACKER $CSC_CKPT"
echo ""
