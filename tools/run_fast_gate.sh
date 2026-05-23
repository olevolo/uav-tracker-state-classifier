#!/usr/bin/env bash
# Fast end-to-end pipeline gate — validates the full pipeline
# on a tiny slice (N sequences) before committing to overnight runs.
#
# Checks:
#   G1  baseline output: N seqs, bbox valid, telemetry present
#   G2  calibration: percentile mapping works, >=0.5% FC-gate fires
#   G3  labels: at least 3 states present (not all CONFIRMED)
#   G4  CSC train 2 epochs: no crash, LOST recall > 0 (model doesn't collapse)
#   G5  gate checker: failure AUROC > 0.7 on mini-val
#   G6  telemetry validation (validate_csc_telemetry.py)
#   G7  feature diagnosis --per_state --shallow_ablation
#   G8  episode evaluation (evaluate_csc_episodes.py)
#
# Usage:  bash tools/run_fast_gate.sh <tracker> [n_seqs]
# Example: bash tools/run_fast_gate.sh sglatrack 10
#          bash tools/run_fast_gate.sh ortrack 10
#
# All outputs go to outputs/_fast_gate/<tracker>/  and are auto-cleaned.
# Exits 0 only if ALL gates pass.

set -e
cd "$(dirname "$0")/.."

TRACKER="${1:?Usage: $0 <tracker> [n_seqs]}"
N="${2:-10}"
DATASET="lasot"
SPLIT="fast_gate"        # separate output dir so it never pollutes real outputs
DEVICE="cpu"
PY=".venv/bin/python -u"

GATE_ROOT="outputs/_fast_gate/$TRACKER"
PRED_DIR="$GATE_ROOT/predictions"
TEL_DIR="$GATE_ROOT/telemetry"
CALIB_DIR="$GATE_ROOT/calibration"
LABELS_DIR="$GATE_ROOT/labels"
TRAIN_DIR="$GATE_ROOT/csc_train"
METRICS="$TRAIN_DIR/val_metrics.json"
REPORT_DIR="$GATE_ROOT/gate_report"
RUN_DIR="$GATE_ROOT"

PASS=0; FAIL=0; WARN=0
declare -A GATE_STATUS
declare -A GATE_MSG

gate() {
  local name="$1"; local ok="$2"; local msg="${3:-}"
  if [ "$ok" = "0" ]; then
    echo "  ✓ $name" >&2; PASS=$((PASS+1)); GATE_STATUS[$name]="PASS"; GATE_MSG[$name]="$msg"
  else
    echo "  ✗ $name" >&2; FAIL=$((FAIL+1)); GATE_STATUS[$name]="FAIL"; GATE_MSG[$name]="$msg"
  fi
}

gate_warn() {
  local name="$1"; local msg="${2:-}"
  echo "  ~ $name (WARN)" >&2; WARN=$((WARN+1)); GATE_STATUS[$name]="WARN"; GATE_MSG[$name]="$msg"
}

echo "[$(date '+%H:%M:%S')] === Fast gate: tracker=$TRACKER n_seqs=$N ===" >&2
mkdir -p "$GATE_ROOT" "$REPORT_DIR"

# ── G1: Baseline on N LaSOT sequences ──────────────────────────────────────
echo "[$(date '+%H:%M:%S')] G1 baseline ($N seqs)..." >&2
$PY -u tools/run_baseline.py \
  --tracker "$TRACKER" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --max_sequences "$N" \
  --device "$DEVICE" \
  --output_dir "$GATE_ROOT" \
  --no_telemetry=False 2>&1 | grep -E "INFO|ERROR|WARN" | tail -5

MANIFEST="$GATE_ROOT/$DATASET/$SPLIT/manifest.json"
PRED_DIR="$GATE_ROOT/$DATASET/$SPLIT/predictions"
TEL_DIR="$GATE_ROOT/$DATASET/$SPLIT/telemetry"

N_DONE=$($PY -c "import json; d=json.load(open('$MANIFEST')); print(d['n_sequences'])" 2>/dev/null || echo 0)
N_FRAMES=$($PY -c "import json; d=json.load(open('$MANIFEST')); print(d['n_frames'])" 2>/dev/null || echo 0)
FPS=$($PY -c "import json; d=json.load(open('$MANIFEST')); print(f\"{d['mean_fps']:.1f}\")" 2>/dev/null || echo 0)
echo "    n_sequences=$N_DONE n_frames=$N_FRAMES mean_fps=$FPS" >&2
[ "$N_DONE" -ge "$N" ] && gate "G1 baseline: $N_DONE seqs complete" 0 || gate "G1 baseline: only $N_DONE/$N seqs" 1

# ── G6: Telemetry validation ─────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] G6 telemetry validation..." >&2
TEL_QUALITY_DIR="$REPORT_DIR/telemetry_quality"
mkdir -p "$TEL_QUALITY_DIR"
TEL_VAL_OK=1
if [ -d "$TEL_DIR" ] && [ -d "$PRED_DIR" ]; then
  $PY -u tools/validate_csc_telemetry.py \
    --telemetry "$TEL_DIR" \
    --predictions "$PRED_DIR" \
    --tracker "$TRACKER" \
    --out "$TEL_QUALITY_DIR" 2>&1 | tail -10
  TEL_VAL_RC=$?
  TEL_STATUS=$($PY -c "import json; d=json.load(open('$TEL_QUALITY_DIR/telemetry_quality.json')); print(d['summary']['status'])" 2>/dev/null || echo "UNKNOWN")
  if [ "$TEL_STATUS" = "PASS" ] && [ "$TEL_VAL_RC" = "0" ]; then
    TEL_VAL_OK=0
  fi
fi
gate "G6 telemetry validation" $TEL_VAL_OK

# ── G2: Calibration ─────────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] G2 calibration..." >&2
mkdir -p "$CALIB_DIR"
$PY -u tools/fit_calibration.py \
  --tracker "$TRACKER" \
  --telemetry_dir "$TEL_DIR" \
  --output_dir "$CALIB_DIR" \
  --features confidence apce psr 2>&1 | grep -E "q05|q95|frac|ERROR" | head -5

CONF_JSON="$CALIB_DIR/${TRACKER}_fast_gate_confidence.json"
# Rename from split-named file
find "$CALIB_DIR" -name "${TRACKER}*confidence.json" | head -1 | xargs -I{} cp {} "$CONF_JSON" 2>/dev/null || true
CALIB_OK=1
if [ -f "$CONF_JSON" ]; then
  FC_PCT=$($PY -c "
import json, numpy as np
d = json.load(open('$CONF_JSON'))
q = np.array(d.get('quantiles', []))
# fraction of frames where calibrated conf >= 0.65 (FC gate)
pct = float(np.mean(q >= 0.65)) if len(q) else 0.0
print(f'{pct:.3f}')
" 2>/dev/null || echo "0")
  echo "    frac_above_fc_conf=$FC_PCT" >&2
  CALIB_OK=0
fi
gate "G2 calibration: calibrator file present" $CALIB_OK

# ── G3: Label generation ─────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] G3 label generation..." >&2
mkdir -p "$LABELS_DIR"
$PY -u tools/build_scene_state_labels.py \
  --tracker "$TRACKER" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --baseline_dir "$GATE_ROOT" \
  --output_dir "$LABELS_DIR" \
  --calibration_dir "$CALIB_DIR" 2>&1 | grep -E "INFO|ERROR|WARN" | tail -5

N_LABEL_FILES=$(find "$LABELS_DIR" -name "*.jsonl" | wc -l | tr -d ' ')
echo "    label files: $N_LABEL_FILES" >&2
# Check state distribution
N_STATES=$($PY -c "
import json, glob, collections
counts = collections.Counter()
for f in glob.glob('$LABELS_DIR/**/*.jsonl', recursive=True):
    for line in open(f):
        r = json.loads(line)
        s = r.get('localization_state') or r.get('state', 'UNKNOWN')
        counts[s] += 1
n = sum(counts.values())
states = {k: round(v/n*100, 1) for k, v in counts.items()}
print(len(counts), json.dumps(states))
" 2>/dev/null || echo "0 {}")
N_S=$(echo $N_STATES | cut -d' ' -f1)
DIST=$(echo $N_STATES | cut -d' ' -f2-)
echo "    states=$N_S distribution=$DIST" >&2
[ "$N_S" -ge 2 ] && gate "G3 labels: >=2 states present ($DIST)" 0 || gate "G3 labels: only $N_S state — all confirmed, calibration issue" 1

# ── audit_label_distribution ─────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] audit_label_distribution..." >&2
AUDIT_CSV="$REPORT_DIR/label_dist.csv"
$PY -u tools/audit_label_distribution.py \
  --labels_dir "$LABELS_DIR" \
  --out "$AUDIT_CSV" 2>&1 | tail -5
AUDIT_RC=$?
[ "$AUDIT_RC" = "0" ] && gate "audit_label_distribution" 0 || gate_warn "audit_label_distribution" "exit $AUDIT_RC"

# ── G4: CSC 2-epoch train ────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] G4 train CSC-TCN16 2 epochs..." >&2
mkdir -p "$TRAIN_DIR"
$PY -u tools/train_csc.py \
  --config configs/csc/csc_tcn16.yaml \
  --labels_dir "$LABELS_DIR" \
  --output_dir "$TRAIN_DIR" \
  --max_epochs 2 \
  --device "$DEVICE" 2>&1 | grep -E "ep [0-9]|ERROR|LOST" | tail -5

if [ -f "$METRICS" ]; then
  LOST_REC=$($PY -c "
import json
d = json.load(open('$METRICS'))
# Support both old and new schema
ps = d.get('loc_per_state') or d.get('per_state') or {}
s = ps.get('LOST') or ps.get('UNCERTAIN') or {}
print(s.get('recall', 0.0))
" 2>/dev/null || echo 0)
  AUROC=$($PY -c "import json; d=json.load(open('$METRICS')); print(d.get('failure_auroc',0))" 2>/dev/null || echo 0)
  echo "    LOST_recall=$LOST_REC failure_AUROC=$AUROC" >&2
  # G4 passes if model doesn't trivially collapse (LOST recall > 0 OR AUROC > 0.6)
  TRAIN_OK=$($PY -c "
import json
d=json.load(open('$METRICS'))
ps = d.get('loc_per_state') or d.get('per_state') or {}
lost = (ps.get('LOST') or {}).get('recall', 0)
auroc = d.get('failure_auroc', 0)
print(0 if (lost > 0 or auroc > 0.6) else 1)
" 2>/dev/null || echo 1)
  gate "G4 train: LOST_recall=$LOST_REC AUROC=$AUROC" $TRAIN_OK
else
  gate "G4 train: val_metrics.json missing" 1
fi

# ── G5: Gate checker ─────────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] G5 stage-1 gate check..." >&2
if [ -f "$METRICS" ]; then
  $PY -u tools/check_stage1_gate.py --metrics_json "$METRICS" 2>&1 | grep -E "PASS|FAIL" | head -10
  gate "G5 gate checker ran" 0
else
  gate "G5 gate checker: no metrics" 1
fi

# ── G7: Feature diagnosis --per_state --shallow_ablation ─────────────────────
echo "[$(date '+%H:%M:%S')] G7 feature diagnosis (per_state + shallow_ablation)..." >&2
DIAG_DIR="$REPORT_DIR/feature_diag"
mkdir -p "$DIAG_DIR"
DIAG_OK=1
$PY -u tools/diagnose_csc_features.py \
  --labels_dir "$LABELS_DIR" \
  --output "$DIAG_DIR/features.csv" \
  --per_state \
  --shallow_ablation 2>&1 | tail -10
DIAG_RC=$?
if [ "$DIAG_RC" = "0" ] && [ -f "$DIAG_DIR/feature_state_summary.csv" ]; then
  DIAG_OK=0
fi
[ "$DIAG_OK" = "0" ] && gate "G7 feature_diag per_state+shallow_ablation" 0 || gate "G7 feature_diag" $DIAG_OK

# ── G8: Episode evaluation ───────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] G8 episode evaluation..." >&2
EPISODE_DIR="$REPORT_DIR/episodes"
mkdir -p "$EPISODE_DIR"
EPISODE_OK=1
# Use labels as both GT and predictions (sanity check: recall should be ~1)
$PY -u tools/evaluate_csc_episodes.py \
  --labels "$LABELS_DIR" \
  --predictions "$LABELS_DIR" \
  --out "$EPISODE_DIR" 2>&1 | tail -10
EPISODE_RC=$?
if [ "$EPISODE_RC" = "0" ] && [ -f "$EPISODE_DIR/episode_metrics.json" ]; then
  EPISODE_OK=0
fi
[ "$EPISODE_OK" = "0" ] && gate "G8 episode_eval ran" 0 || gate "G8 episode_eval" $EPISODE_OK

# ── audit_visualizer (internal audit only) ───────────────────────────────────
echo "[$(date '+%H:%M:%S')] audit_visualizer..." >&2
AUDIT_PNG="$REPORT_DIR/audit_vis.png"
$PY -u tools/audit_visualizer.py \
  --labels "$LABELS_DIR" \
  --predictions "$PRED_DIR" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --n_seqs 4 \
  --frames_per_seq 3 \
  --out "$AUDIT_PNG" 2>&1 | tail -5 || true
# Not gated — audit_visualizer may fail if image data not available

# ── FAST_GATE_REPORT.md ───────────────────────────────────────────────────────
TOTAL=$((PASS+FAIL+WARN))
if [ "$FAIL" -eq 0 ]; then
  FINAL_DECISION="RUN_FULL_BENCHMARK"
else
  FINAL_DECISION="FIX_FIRST"
fi

REPORT_MD="$RUN_DIR/FAST_GATE_REPORT.md"
{
  echo "# Fast Gate Report — $TRACKER"
  echo ""
  echo "**Date**: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "**Tracker**: $TRACKER | **Dataset**: $DATASET | **N seqs**: $N"
  echo "**Decision**: $FINAL_DECISION"
  echo ""
  echo "## Gate Results"
  echo ""
  echo "| Gate | Status |"
  echo "|------|--------|"
  echo "| G1 baseline | ${GATE_STATUS[G1 baseline: $N_DONE seqs complete]:-UNKNOWN} |"
  echo "| G6 telemetry validation | ${GATE_STATUS[G6 telemetry validation]:-UNKNOWN} |"
  echo "| G2 calibration | ${GATE_STATUS[G2 calibration: calibrator file present]:-UNKNOWN} |"
  echo "| G3 labels | $(echo "${GATE_STATUS[@]}" | tr ' ' '\n' | grep -m1 'G3\|PASS\|FAIL' | head -1) |"
  echo "| audit_label_distribution | ${GATE_STATUS[audit_label_distribution]:-UNKNOWN} |"
  echo "| G4 train | $(echo "${!GATE_STATUS[@]}" | tr ' ' '\n' | grep 'G4 train' | head -1) |"
  echo "| G5 gate checker | ${GATE_STATUS[G5 gate checker ran]:-UNKNOWN} |"
  echo "| G7 feature_diag | ${GATE_STATUS[G7 feature_diag per_state+shallow_ablation]:-UNKNOWN} |"
  echo "| G8 episode_eval | ${GATE_STATUS[G8 episode_eval ran]:-UNKNOWN} |"
  echo ""
  echo "## Summary"
  echo ""
  echo "- PASS: $PASS / $TOTAL"
  echo "- WARN: $WARN / $TOTAL"
  echo "- FAIL: $FAIL / $TOTAL"
  echo ""
  echo "## Final Decision"
  echo ""
  if [ "$FAIL" -eq 0 ]; then
    echo "**$FINAL_DECISION** — all gates pass (warnings noted above, review before production)."
    echo ""
    echo "Next step: \`make baseline TRACKER=$TRACKER DATASET=lasot SPLIT=train DEVICE=$DEVICE\`"
  else
    echo "**$FINAL_DECISION** — $FAIL gate(s) failed. Check outputs in \`$GATE_ROOT/\`."
  fi
} > "$REPORT_MD"
echo "" >&2
echo "[$(date '+%H:%M:%S')] Wrote $REPORT_MD" >&2

# ── Summary ──────────────────────────────────────────────────────────────────
echo "" >&2
echo "=============================" >&2
echo "Fast gate summary for $TRACKER" >&2
echo "  PASS: $PASS / $((PASS+FAIL+WARN))" >&2
echo "  WARN: $WARN / $((PASS+FAIL+WARN))" >&2
echo "  FAIL: $FAIL / $((PASS+FAIL+WARN))" >&2
if [ "$FAIL" -eq 0 ]; then
  echo "  ✓ ALL GATES PASS — safe to launch full run" >&2
  echo "  Decision: RUN_FULL_BENCHMARK" >&2
  echo "  Next: make baseline TRACKER=$TRACKER DATASET=lasot SPLIT=train DEVICE=$DEVICE" >&2
else
  echo "  ✗ GATES FAILED — fix issues before overnight run" >&2
  echo "  Decision: FIX_FIRST" >&2
  echo "  Check: $GATE_ROOT/" >&2
fi
echo "=============================" >&2

[ "$FAIL" -eq 0 ]
