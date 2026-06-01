#!/usr/bin/env bash
# CSC v2 pipeline: LaSOT (curated) + GOT-10k + UAVDT-SOT multi-dataset training.
#
# Goal: ~60% aerial-domain data (UAVDT + GOT-10k) to improve FC detection on UAV123.
#
# Key parameters vs v1:
#   --lasot_cats        LaSOT categories to keep (default: drone,motorcycle,truck,bus,car,person,boat)
#                       Drops: horse(0%FC), bicycle(0.3%FC), dog(0.9%FC), bird(1.2%FC)
#   --lasot_seqs_per_cat Max sequences per LaSOT category (default: 10, not 20)
#                       Reduces LaSOT dominance from 91% to ~43%
#   --uavdt_upsample    UAVDT repeat factor (default: 5) → aerial = ~57%
#
# Prerequisites (must exist before running):
#   outputs/baselines/<tracker>/lasot/train/        ← from v1 pipeline  ✓
#   outputs/csc_labels/<tracker>/lasot/train/       ← from v1 pipeline  ✓
#   outputs/baselines/<tracker>/got10k/val/         ← re-run with full telemetry
#   outputs/baselines/<tracker>/uavdt_sot/test/     ← new baseline
#
# What this script produces:
#   outputs/calibration/<tracker>_v2_confidence.json
#   outputs/csc_labels/<tracker>/got10k/val/labels.jsonl
#   outputs/csc_labels/<tracker>/uavdt_sot/test/labels.jsonl
#   outputs/csc_labels/<tracker>/v2_combined/labels.jsonl  ← merged + curated
#   outputs/csc_training/<tracker>_v2_tcn16/checkpoint_best.pth
#
# Usage:
#   bash tools/run_v2_pipeline.sh sglatrack
#   bash tools/run_v2_pipeline.sh sglatrack --lasot_seqs_per_cat 10 --uavdt_upsample 5
#   bash tools/run_v2_pipeline.sh sglatrack --lasot_cats drone,truck,bus,car,person,motorcycle
#
# DO NOT run until:
#   1. got10k baseline re-run with full telemetry (APCE/PSR present)
#   2. uavdt_sot baseline complete
#   3. FC audit confirms useful scenes in both datasets
#   4. v1 auto_full_pipeline.sh has finished UAV123 eval

set -euo pipefail
cd "$(dirname "$0")/.."

TRACKER="${1:?Usage: $0 <tracker_name> [options]}"

# Defaults: curated LaSOT + drone×3 + got10k×1.5 + uavdt×5 + visdrone×3
# FC density v2: ~7.1% vs v1: 2.87% (+148%)
# Distribution: LaSOT 61% | GOT-10k 4% | UAVDT 23% | VisDrone 12% | Aerial 39%
LASOT_CATS="drone,motorcycle,truck,bus,car,person,boat"
LASOT_SEQS_PER_CAT=20
LASOT_DRONE_UPSAMPLE=3    # drone: 13% FC — most FC-rich LaSOT category
GOT10K_UPSAMPLE=15        # ×1.5 (stored as tenths: 15 = 1.5x)
UAVDT_UPSAMPLE=5
VISDRONE_UPSAMPLE=3       # 34% of seqs have FC>5%, mean FC=8.6%
SKIP_CALIBRATION_REFIT=0

shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lasot_cats)             LASOT_CATS="$2";            shift 2 ;;
    --lasot_seqs_per_cat)     LASOT_SEQS_PER_CAT="$2";    shift 2 ;;
    --lasot_drone_upsample)   LASOT_DRONE_UPSAMPLE="$2";  shift 2 ;;
    --got10k_upsample)        GOT10K_UPSAMPLE="$2";        shift 2 ;;
    --uavdt_upsample)         UAVDT_UPSAMPLE="$2";         shift 2 ;;
    --visdrone_upsample)      VISDRONE_UPSAMPLE="$2";       shift 2 ;;
    --skip_calibration_refit) SKIP_CALIBRATION_REFIT=1;    shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

PY=".venv/bin/python -u"
CALIB_DIR="outputs/calibration"
V2_LABELS="outputs/csc_labels/$TRACKER/v2_combined"
TRAIN_OUT="outputs/csc_training/${TRACKER}_v2_tcn16"
LASOT_LABELS_SRC="outputs/csc_labels/$TRACKER/lasot/train/lasot/train"

TS()   { date '+%H:%M:%S'; }
log()  { echo "[$(TS)] $*" >&2; }
fail() { log "FATAL: $*"; exit 1; }

# ──────────────────────────────────────────────────────────────────────────────
# Safety checks
# ──────────────────────────────────────────────────────────────────────────────

require_baseline() {
  local dataset="$1" split="$2" min="$3"
  local pred_dir="outputs/baselines/$TRACKER/$dataset/$split/predictions"
  local n; n=$(ls "$pred_dir" 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -ge "$min" ] || fail "Baseline incomplete: $pred_dir ($n/$min). Run run_baseline.py first."
  log "  ✓ $TRACKER/$dataset/$split: $n seqs"
}

require_labels() {
  local path="$1"
  [ -f "$path/labels.jsonl" ] || fail "Labels missing: $path/labels.jsonl"
  log "  ✓ labels: $path"
}

log "=== CSC v2 pipeline: $TRACKER ==="
log "Config: lasot_cats=$LASOT_CATS seqs_per_cat=$LASOT_SEQS_PER_CAT uavdt_upsample=${UAVDT_UPSAMPLE}x"
log "Checking prerequisites..."
require_baseline lasot    train 220
require_baseline got10k   val   170
require_baseline uavdt_sot    test  45
require_baseline visdrone_sot test  30
require_labels   "$LASOT_LABELS_SRC"

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Re-fit calibration on combined telemetry (optional, default ON)
# Calibration is for INFERENCE only — not needed for label generation.
# Re-fitting improves APCE/PSR normalisation across LaSOT + aerial domains.
# ──────────────────────────────────────────────────────────────────────────────
V2_CALIB_TAG="${TRACKER}_v2"
if [ "$SKIP_CALIBRATION_REFIT" -eq 0 ] && \
   [ ! -f "$CALIB_DIR/${V2_CALIB_TAG}_confidence.json" ]; then
  log "Step 1/5: Merge telemetry + re-fit calibration..."
  MERGED_TEL="$CALIB_DIR/merged_telemetry_${TRACKER}_v2"
  mkdir -p "$MERGED_TEL"
  for ds_split in "lasot/train" "got10k/val" "uavdt_sot/test" "visdrone_sot/test"; do
    tel="outputs/baselines/$TRACKER/$ds_split/telemetry"
    [ -d "$tel" ] || continue
    tag=$(echo "$ds_split" | tr '/' '_')
    for f in "$tel"/*.jsonl; do
      [ -f "$f" ] && ln -sf "$(realpath "$f")" \
        "$MERGED_TEL/${tag}__$(basename "$f")" 2>/dev/null || true
    done
  done
  # Name merged dir so dataset_tag → v2 (fit_calibration.py derives tag from path)
  V2_TEL_LINK="$CALIB_DIR/${TRACKER}_v2"
  ln -sfn "$(realpath "$MERGED_TEL")" "$V2_TEL_LINK"
  $PY tools/fit_calibration.py \
    --tracker "$TRACKER" \
    --telemetry_dir "$V2_TEL_LINK" \
    --output_dir "$CALIB_DIR" \
    --features confidence apce psr \
    || fail "Calibration refit failed"
  rm -f "$V2_TEL_LINK"
  log "  ✓ $CALIB_DIR/${V2_CALIB_TAG}_confidence.json"
else
  log "Step 1/5: Skipping calibration refit (--skip_calibration_refit or already done)."
fi

# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Generate labels for GOT-10k val
# ──────────────────────────────────────────────────────────────────────────────
GOT10K_LABELS_DIR="outputs/csc_labels/$TRACKER/got10k/val"
GOT10K_LABELS="$GOT10K_LABELS_DIR/got10k/val"
if [ ! -f "$GOT10K_LABELS/labels.jsonl" ]; then
  log "Step 2/5: Labels for GOT-10k val..."
  $PY tools/build_scene_state_labels.py \
    --tracker "$TRACKER" --dataset got10k --split val \
    --baseline_dir "outputs/baselines/$TRACKER" \
    --output_dir "$GOT10K_LABELS_DIR" \
    --calibration_dir "$CALIB_DIR" \
    --calibrator_tag "${TRACKER}_aerial_v2" \
    || fail "GOT-10k labels failed"
  log "  ✓ $GOT10K_LABELS"
else
  log "Step 2/5: GOT-10k labels exist, skipping."
fi

# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Generate labels for UAVDT-SOT
# ──────────────────────────────────────────────────────────────────────────────
UAVDT_LABELS_DIR="outputs/csc_labels/$TRACKER/uavdt_sot/test"
UAVDT_LABELS="$UAVDT_LABELS_DIR/uavdt_sot/test"
if [ ! -f "$UAVDT_LABELS/labels.jsonl" ]; then
  log "Step 3/6: Labels for UAVDT-SOT..."
  $PY tools/build_scene_state_labels.py \
    --tracker "$TRACKER" --dataset uavdt_sot --split test \
    --baseline_dir "outputs/baselines/$TRACKER" \
    --output_dir "$UAVDT_LABELS_DIR" \
    --calibration_dir "$CALIB_DIR" \
    --calibrator_tag "${TRACKER}_aerial_v2" \
    || fail "UAVDT labels failed"
  log "  ✓ $UAVDT_LABELS"
else
  log "Step 3/6: UAVDT labels exist, skipping."
fi

# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Generate labels for VisDrone-SOT (34% seqs FC>5%, mean FC=8.6%)
# ──────────────────────────────────────────────────────────────────────────────
VD_LABELS_DIR="outputs/csc_labels/$TRACKER/visdrone_sot/test"
VD_LABELS="$VD_LABELS_DIR/visdrone_sot/test"
if [ ! -f "$VD_LABELS/labels.jsonl" ]; then
  log "Step 4/6: Labels for VisDrone-SOT..."
  $PY tools/build_scene_state_labels.py \
    --tracker "$TRACKER" --dataset visdrone_sot --split test \
    --baseline_dir "outputs/baselines/$TRACKER" \
    --output_dir "$VD_LABELS_DIR" \
    --calibration_dir "$CALIB_DIR" \
    --calibrator_tag "${TRACKER}_aerial_v2" \
    || fail "VisDrone labels failed"
  log "  ✓ $VD_LABELS"
else
  log "Step 4/6: VisDrone labels exist, skipping."
fi

# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Merge labels with LaSOT curation
# LaSOT: filter to selected categories × max N sequences per category
# UAVDT: upsample K times to boost aerial representation
# ──────────────────────────────────────────────────────────────────────────────
if [ ! -f "$V2_LABELS/labels.jsonl" ]; then
  log "Step 5/6: Merge + curate labels..."
  log "  LaSOT: cats=[$LASOT_CATS] seqs_per_cat=$LASOT_SEQS_PER_CAT drone×$LASOT_DRONE_UPSAMPLE"
  log "  UAVDT: ${UAVDT_UPSAMPLE}x | VisDrone: ${VISDRONE_UPSAMPLE}x"
  mkdir -p "$V2_LABELS"

  # Python inline: curated LaSOT with per-category upsampling (drone×N)
  $PY - <<PYEOF
import json, sys, collections
from pathlib import Path

cats = set("$LASOT_CATS".split(","))
seqs_per_cat = $LASOT_SEQS_PER_CAT
drone_upsample = $LASOT_DRONE_UPSAMPLE
src = Path("$LASOT_LABELS_SRC/labels.jsonl")
out = Path("$V2_LABELS/labels.jsonl")

# Two-pass: collect all lines per category, then write with upsampling
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

total_seqs = sum(len(v) for v in cat_seqs_seen.values())
print(f"  LaSOT curated: {sum(len(v)*( $LASOT_DRONE_UPSAMPLE if k=='drone' else 1) for k,v in cat_lines.items())} frames (drone×{drone_upsample})", file=sys.stderr)
for cat in sorted(cat_seqs_seen):
    mul = drone_upsample if cat == "drone" else 1
    print(f"    {cat}: {len(cat_seqs_seen[cat])} seqs ×{mul} = {len(cat_lines[cat])*mul}", file=sys.stderr)
PYEOF

  # Append GOT-10k (×1.5 = append once + half the lines)
  # GOT10K_UPSAMPLE stored as tenths: 15 = 1.5, 10 = 1.0, 20 = 2.0
  got_lines=$(wc -l < "$GOT10K_LABELS/labels.jsonl" | tr -d ' ')
  half=$(( got_lines * (GOT10K_UPSAMPLE - 10) / 10 ))
  cat "$GOT10K_LABELS/labels.jsonl" >> "$V2_LABELS/labels.jsonl"
  [ "$half" -gt 0 ] && head -n "$half" "$GOT10K_LABELS/labels.jsonl" >> "$V2_LABELS/labels.jsonl"
  log "  GOT-10k: $got_lines frames + $half extra (×$(echo "scale=1; $GOT10K_UPSAMPLE/10" | bc))"

  # Append UAVDT K times
  for i in $(seq 1 "$UAVDT_UPSAMPLE"); do
    cat "$UAVDT_LABELS/labels.jsonl" >> "$V2_LABELS/labels.jsonl"
  done

  # Append VisDrone-SOT K times
  for i in $(seq 1 "$VISDRONE_UPSAMPLE"); do
    cat "$VD_LABELS/labels.jsonl" >> "$V2_LABELS/labels.jsonl"
  done

  # Distribution report
  total=$(wc -l < "$V2_LABELS/labels.jsonl" | tr -d ' ')
  lasot_n=$($PY -c "
import json
from pathlib import Path
cats = set('$LASOT_CATS'.split(','))
n=0
with open('$V2_LABELS/labels.jsonl') as f:
    for i,l in enumerate(f):
        r=json.loads(l)
        if r.get('dataset','') == 'lasot': n+=1
        if i > 0 and i % 100000 == 0: pass
print(n)
" 2>/dev/null || echo "?")
  got10k_n=$(wc -l < "$GOT10K_LABELS/labels.jsonl" | tr -d ' ')
  uavdt_n=$(wc -l < "$UAVDT_LABELS/labels.jsonl" | tr -d ' ')
  uavdt_total=$((uavdt_n * UAVDT_UPSAMPLE))

  log "  Total labels: $total"
  log "  LaSOT (curated): ~$lasot_n"
  log "  GOT-10k:          $got10k_n"
  log "  UAVDT (${UAVDT_UPSAMPLE}x):       $uavdt_total"
  aerial=$((got10k_n + uavdt_total))
  log "  Aerial share:     ~$(( aerial * 100 / total ))%  (target: 60%)"

  cat > "$V2_LABELS/dataset_info.json" <<EOF
{
  "version": "v2",
  "tracker": "$TRACKER",
  "lasot_cats": "$LASOT_CATS",
  "lasot_seqs_per_cat": $LASOT_SEQS_PER_CAT,
  "uavdt_upsample": $UAVDT_UPSAMPLE,
  "total_lines": $total,
  "got10k_lines": $got10k_n,
  "uavdt_lines_per_copy": $uavdt_n,
  "uavdt_total_lines": $uavdt_total
}
EOF
  log "  ✓ $V2_LABELS/labels.jsonl"
else
  log "Step 5/6: Combined v2 labels exist, skipping."
fi

# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Train CSC v2
# ──────────────────────────────────────────────────────────────────────────────
if [ ! -f "$TRAIN_OUT/checkpoint_best.pth" ]; then
  log "Step 6/6: Train CSC-TCN16 v2..."
  $PY tools/train_csc.py \
    --config configs/csc/csc_tcn16.yaml \
    --labels_dir "$V2_LABELS" \
    --output_dir "$TRAIN_OUT" \
    --device cpu \
    || fail "CSC v2 training failed"
  log "  ✓ $TRAIN_OUT/checkpoint_best.pth"
else
  log "Step 6/6: v2 checkpoint exists, skipping."
fi

# ──────────────────────────────────────────────────────────────────────────────
log ""
log "=== CSC v2 DONE for $TRACKER ==="
log "Calibration:  $CALIB_DIR/${V2_CALIB_TAG}_confidence.json"
log "Labels:       $V2_LABELS/labels.jsonl"
log "Model:        $TRAIN_OUT/checkpoint_best.pth"
log ""
log "Next:"
log "  UAV123 eval: CSC_NOT_TRAINED_ON_UAV123=1 bash tools/run_uav123_final_eval.sh \\"
log "               $TRACKER $TRAIN_OUT/checkpoint_best.pth"
log "  FC audit:    python tools/audit_fc_precision.py --dataset dtb70 \\"
log "               --csc_run_dir outputs/csc_runs/..."
