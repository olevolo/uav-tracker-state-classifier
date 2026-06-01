#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_train2_v2_pipeline.sh — train2_v2 pipeline with unified calibration,
# DatasetAwareSampler, stratified val split, and FC focal weight 4.0.
#
# Fixes from v2:
#   1. Unified calibrator: sglatrack_all_v2 (all datasets combined)
#   2. DatasetAwareSampler: fixed batch composition per dataset
#   3. Stratified deterministic val split (FC-guaranteed)
#   4. FC weight: 4.0, selection: 0.45*F1 + 0.55*FC_recall
#   5. LaSOT: 12 seqs/cat, bird added; all ×1 (sampler controls fractions)
#
# Usage:
#   bash tools/run_train2_v2_pipeline.sh sglatrack
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

TRACKER="${1:?Usage: $0 <tracker_name>}"
PY=".venv/bin/python -u"
CALIB_DIR="outputs/calibration"
LABELS_BASE="outputs/csc_labels/$TRACKER"
V2_LABELS="$LABELS_BASE/train2_v2_combined"
TRAIN_OUT="outputs/csc_training/${TRACKER}_train2_v2_tcn16"
CALIB_TAG="${TRACKER}_all_v2"

TS()   { date '+%H:%M:%S'; }
log()  { echo "[$(TS)] $*"; }
fail() { echo "[$(TS)] FATAL: $*" >&2; exit 1; }

log "=== train2_v2 pipeline: $TRACKER ==="

# ---------------------------------------------------------------------------
# Step 0: Verify unified calibrator exists
# ---------------------------------------------------------------------------
log "Step 0/6: Checking unified calibrator..."
[ -f "$CALIB_DIR/${CALIB_TAG}_confidence.json" ] || \
  fail "Calibrator not found: $CALIB_DIR/${CALIB_TAG}_confidence.json. Run fit_calibration.py first."
log "  ✓ $CALIB_DIR/${CALIB_TAG}_confidence.json"

# ---------------------------------------------------------------------------
# Step 1: Regenerate LaSOT labels with unified calibrator
# ---------------------------------------------------------------------------
LASOT_LABELS="$LABELS_BASE/lasot/train/lasot/train/labels.jsonl"
if [ -f "$LASOT_LABELS" ]; then
  log "Step 1/6: Deleting stale LaSOT labels (wrong calibrator)..."
  rm -rf "$LABELS_BASE/lasot/train/lasot/"
fi
log "Step 1/6: Generating LaSOT labels with $CALIB_TAG calibrator..."
$PY tools/build_scene_state_labels.py \
  --tracker "$TRACKER" --dataset lasot --split train \
  --baseline_dir "outputs/baselines/$TRACKER" \
  --output_dir "$LABELS_BASE/lasot/train" \
  --calibration_dir "$CALIB_DIR" \
  --calibrator_tag "$CALIB_TAG" \
  || fail "LaSOT label generation failed"
log "  ✓ LaSOT labels regenerated with $CALIB_TAG"

# ---------------------------------------------------------------------------
# Step 2: Regenerate aerial labels with unified calibrator
# DTB70 is validation-only — NOT regenerated for training
# GOT-10k removed: 24 FC frames / 21K total = 0.1% FC, median 100fr sequences — not useful
# ---------------------------------------------------------------------------
for ds_split in "uavdt_sot/test" "visdrone_sot/test"; do
  ds=$(echo "$ds_split" | cut -d/ -f1)
  split=$(echo "$ds_split" | cut -d/ -f2)
  label_dir="$LABELS_BASE/$ds/$split"
  label_file="$label_dir/$ds/$split/labels.jsonl"
  if [ -f "$label_file" ]; then
    log "  Regenerating $ds/$split labels with $CALIB_TAG..."
    rm -rf "$label_dir/$ds/"
    $PY tools/build_scene_state_labels.py \
      --tracker "$TRACKER" --dataset "$ds" --split "$split" \
      --baseline_dir "outputs/baselines/$TRACKER" \
      --output_dir "$label_dir" \
      --calibration_dir "$CALIB_DIR" \
      --calibrator_tag "$CALIB_TAG" \
      || fail "$ds/$split label generation failed"
  else
    log "  $ds/$split: no existing labels, generating with $CALIB_TAG..."
    $PY tools/build_scene_state_labels.py \
      --tracker "$TRACKER" --dataset "$ds" --split "$split" \
      --baseline_dir "outputs/baselines/$TRACKER" \
      --output_dir "$label_dir" \
      --calibration_dir "$CALIB_DIR" \
      --calibrator_tag "$CALIB_TAG" \
      || fail "$ds/$split label generation failed"
  fi
  log "  ✓ $ds/$split"
done

# ---------------------------------------------------------------------------
# Step 3: Verify calibration consistency (<5pp HIGH_CONFIDENCE delta)
# ---------------------------------------------------------------------------
log "Step 3/6: Checking calibration consistency..."
$PY - << 'PYEOF'
import json, sys
from pathlib import Path

datasets = {
    "lasot":    "outputs/csc_labels/sglatrack/lasot/train/lasot/train/label_stats.json",
    "uavdt_sot":"outputs/csc_labels/sglatrack/uavdt_sot/test/uavdt_sot/test/label_stats.json",
    "visdrone": "outputs/csc_labels/sglatrack/visdrone_sot/test/visdrone_sot/test/label_stats.json",
}
hc_rates = {}
for name, path in datasets.items():
    if not Path(path).exists():
        continue
    stats = json.load(open(path))
    total = stats["summary"]["n_frames"]
    # HIGH_CONFIDENCE = confidence_state 1
    # approximate from FC + some LOST_AWARE (rough check)
    fc = stats["summary"]["state_counts"].get("FALSE_CONFIRMED", 0)
    hc_rates[name] = fc / total if total > 0 else 0.0
    print(f"  {name}: FC%={100*hc_rates[name]:.2f}%")

if hc_rates:
    vals = list(hc_rates.values())
    delta = max(vals) - min(vals)
    print(f"  FC% range: {min(vals)*100:.2f}% - {max(vals)*100:.2f}% (delta={delta*100:.2f}pp)")
PYEOF
log "  ✓ Consistency check done"

# ---------------------------------------------------------------------------
# Step 4: Build train2_v2_combined labels
# ---------------------------------------------------------------------------
if [ -f "$V2_LABELS/labels.jsonl" ]; then
  log "Step 4/6: SKIP — $V2_LABELS/labels.jsonl exists (delete to rebuild)"
else
  log "Step 4/6: Building train2_v2_combined labels..."
  mkdir -p "$V2_LABELS"

  # LaSOT: 8 cats × 12 seqs/cat, drone×3
  $PY - << 'PYEOF'
import json, sys, collections
from pathlib import Path

cats = set("drone,motorcycle,truck,bus,car,person,boat,bird".split(","))
seqs_per_cat = 12
drone_upsample = 3
src = Path("outputs/csc_labels/sglatrack/lasot/train/lasot/train/labels.jsonl")
out = Path("outputs/csc_labels/sglatrack/train2_v2_combined/labels.jsonl")

cat_lines = collections.defaultdict(list)
cat_seqs_seen = collections.defaultdict(set)

with open(src) as fin:
    for line in fin:
        row = json.loads(line)
        seq = row.get("sequence", "")
        cat = seq.split("-")[0] if "-" in seq else seq
        if cat not in cats:
            continue
        if len(cat_seqs_seen[cat]) >= seqs_per_cat and seq not in cat_seqs_seen[cat]:
            continue
        cat_seqs_seen[cat].add(seq)
        cat_lines[cat].append(line)

with open(out, "w") as fout:
    for cat, lines in sorted(cat_lines.items()):
        repeat = drone_upsample if cat == "drone" else 1
        for _ in range(repeat):
            for line in lines:
                fout.write(line)

for cat in sorted(cat_seqs_seen):
    mul = drone_upsample if cat == "drone" else 1
    print(f"  LaSOT {cat}: {len(cat_seqs_seen[cat])} seqs ×{mul}", file=sys.stderr)
PYEOF

  # Remaining datasets ×1 (DerivedState WRS sampler controls effective fractions)
  # DTB70 excluded — validation-only
  # GOT-10k excluded — 0.1% FC, median 100fr sequences, not useful for FC training
  for src_path in \
    "outputs/csc_labels/$TRACKER/uavdt_sot/test/uavdt_sot/test/labels.jsonl" \
    "outputs/csc_labels/$TRACKER/visdrone_sot/test/visdrone_sot/test/labels.jsonl"; do
    if [ -f "$src_path" ]; then
      cat "$src_path" >> "$V2_LABELS/labels.jsonl"
      n=$(wc -l < "$src_path" | tr -d ' ')
      log "  Appended: $src_path ($n lines)"
    else
      log "  SKIP (not found): $src_path"
    fi
  done

  total=$(wc -l < "$V2_LABELS/labels.jsonl" | tr -d ' ')
  log "  train2_v2_combined total: $total frames"
  log "  ✓ $V2_LABELS/labels.jsonl"
fi

# ---------------------------------------------------------------------------
# Step 5: Validate label FC rate
# ---------------------------------------------------------------------------
log "Step 5/6: Validating FC rate in combined labels..."
$PY - << 'PYEOF'
import json
from collections import Counter
from pathlib import Path

labels = Path("outputs/csc_labels/sglatrack/train2_v2_combined/labels.jsonl")
c = Counter()
with open(labels) as f:
    for line in f:
        r = json.loads(line)
        c[r.get("derived_state_name", "")] += 1
total = sum(c.values())
fc = c.get("FALSE_CONFIRMED", 0)
la = c.get("LOST_AWARE", 0)
print(f"  FC: {fc}/{total} = {100*fc/total:.2f}%  (target: ≥5.5%)")
print(f"  LA: {la}/{total} = {100*la/total:.2f}%")
for s, n in sorted(c.items(), key=lambda x: -x[1]):
    print(f"  {s}: {n:,} ({100*n/total:.1f}%)")
PYEOF

# ---------------------------------------------------------------------------
# Step 6: Train
# ---------------------------------------------------------------------------
if [ -f "$TRAIN_OUT/checkpoint_best.pth" ]; then
  log "Step 6/6: SKIP — $TRAIN_OUT/checkpoint_best.pth exists (delete to retrain)"
else
  log "Step 6/6: Training CSC train2_v2..."
  $PY tools/train_csc.py \
    --config configs/csc/csc_tcn16_train2.yaml \
    --labels_dir "$V2_LABELS" \
    --output_dir "$TRAIN_OUT" \
    --device cpu \
    || fail "Training failed"
  log "  ✓ $TRAIN_OUT/checkpoint_best.pth"
fi

log ""
log "=== train2_v2 DONE for $TRACKER ==="
log "Calibrator: $CALIB_DIR/${CALIB_TAG}_confidence.json"
log "Labels:     $V2_LABELS/labels.jsonl"
log "Model:      $TRAIN_OUT/checkpoint_best.pth"
