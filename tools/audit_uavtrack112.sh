#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# audit_uavtrack112.sh — автоматичний аудит UAVTrack112 для train2_v2
#
# Що робить:
#   1. Знаходить директорію UAVTrack112
#   2. Підраховує sequences, frames, GT format
#   3. Запускає SGLATrack baseline
#   4. Генерує labels з sglatrack_all_v2 calibrator
#   5. Аудитує FC rate, unique FC sequences, FC pattern (APCE vs conf)
#   6. Записує результат в docs/research/uavtrack112_audit.md
#
# Usage:
#   bash tools/audit_uavtrack112.sh [/path/to/UAVTrack112]
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python -u"
AUDIT_OUT="docs/research/uavtrack112_audit.md"

# Auto-detect dataset path
DATASET_PATH="${1:-}"
if [ -z "$DATASET_PATH" ]; then
    for candidate in \
        ~/uav-tracker-data/UAVTrack112 \
        ~/uav-tracker-data/uavtrack112 \
        ~/uav-tracker-data/UAVTrack-112 \
        ~/Downloads/UAVTrack112; do
        if [ -d "$candidate" ]; then
            DATASET_PATH="$candidate"
            break
        fi
    done
fi

if [ -z "$DATASET_PATH" ] || [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: UAVTrack112 not found. Provide path as argument or place at ~/uav-tracker-data/UAVTrack112"
    exit 1
fi

echo "[$(date '+%H:%M:%S')] UAVTrack112 found at: $DATASET_PATH"

# ---------------------------------------------------------------------------
# Step 1: Dataset structure audit
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Step 1: Analyzing dataset structure..."

$PY - << PYEOF
import os
import json
from pathlib import Path
from collections import Counter, defaultdict

ds_path = Path("$DATASET_PATH")
print(f"Dataset path: {ds_path}")

# Find sequences (look for groundtruth.txt or *.txt annotation files)
seqs = []
for p in sorted(ds_path.rglob("groundtruth.txt")):
    seq_dir = p.parent
    imgs = list(seq_dir.glob("img/*.jpg")) + list(seq_dir.glob("img/*.png")) + \
           list(seq_dir.glob("*.jpg")) + list(seq_dir.glob("*.png"))
    if imgs:
        seqs.append((seq_dir.name, len(imgs), str(p)))

if not seqs:
    # Try alternative: look for annotation files
    for p in sorted(ds_path.rglob("*.txt")):
        if "gt" in p.name.lower() or "anno" in p.name.lower():
            seq_dir = p.parent
            imgs = list(seq_dir.rglob("*.jpg"))[:1]
            if imgs:
                seqs.append((seq_dir.name, -1, str(p)))

print(f"Sequences found: {len(seqs)}")
if seqs:
    frame_counts = [n for _, n, _ in seqs if n > 0]
    if frame_counts:
        print(f"Total frames: {sum(frame_counts)}")
        print(f"Avg frames/seq: {sum(frame_counts)/len(frame_counts):.0f}")
        print(f"Min/Max frames: {min(frame_counts)}/{max(frame_counts)}")

# Check GT format (first sequence)
if seqs:
    gt_path = Path(seqs[0][2])
    lines = gt_path.read_text().strip().split('\n')[:3]
    print(f"\nGT format (first 3 lines of {seqs[0][0]}):")
    for l in lines:
        print(f"  {l}")
    # Detect separator
    sep = ',' if ',' in lines[0] else '\t' if '\t' in lines[0] else ' '
    vals = lines[0].replace('\t', ',').replace(' ', ',').split(',')
    print(f"  → {len(vals)} values per line, separator='{sep}'")
    if len(vals) == 4:
        print("  → Format: xywh (4 values) ✓")
    elif len(vals) == 8:
        print("  → Format: x1y1x2y2x3y3x4y4 (rotated bbox)")

# Top-level structure
toplevel = sorted(set(p.parent.name for p in ds_path.glob("*/*") if p.is_dir()))
print(f"\nTop-level dirs (first 10): {toplevel[:10]}")
PYEOF

# ---------------------------------------------------------------------------
# Step 2: Run SGLATrack baseline
# ---------------------------------------------------------------------------
BASELINE_OUT="outputs/baselines/sglatrack/uavtrack112/test"
if [ -d "$BASELINE_OUT/predictions" ] && \
   [ "$(ls $BASELINE_OUT/predictions/*.txt 2>/dev/null | wc -l)" -ge 100 ]; then
    echo "[$(date '+%H:%M:%S')] Step 2: SKIP — baseline exists"
else
    echo "[$(date '+%H:%M:%S')] Step 2: Running SGLATrack baseline..."
    # Note: requires UAVTrack112 loader to be registered in csc_uav_tracking
    # If not yet registered, this step will fail with a helpful error
    $PY tools/run_baseline.py \
        --tracker sglatrack \
        --dataset uavtrack112 \
        --split test \
        --device cpu \
        || echo "WARNING: baseline failed — UAVTrack112 loader may need to be registered first"
fi

# ---------------------------------------------------------------------------
# Step 3: Generate labels with sglatrack_all_v2 calibrator
# ---------------------------------------------------------------------------
LABELS_OUT="outputs/csc_labels/sglatrack/uavtrack112/test"
if [ -f "$LABELS_OUT/uavtrack112/test/labels.jsonl" ]; then
    echo "[$(date '+%H:%M:%S')] Step 3: SKIP — labels exist"
else
    echo "[$(date '+%H:%M:%S')] Step 3: Generating labels..."
    $PY tools/build_scene_state_labels.py \
        --tracker sglatrack \
        --dataset uavtrack112 \
        --split test \
        --baseline_dir outputs/baselines/sglatrack \
        --calibration_dir outputs/calibration \
        --calibrator_tag sglatrack_all_v2 \
        --output_dir "$LABELS_OUT" \
        || echo "WARNING: label generation failed — baseline may not exist yet"
fi

# ---------------------------------------------------------------------------
# Step 4: FC audit
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Step 4: FC audit..."

$PY - << 'PYEOF'
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

labels_path = Path("outputs/csc_labels/sglatrack/uavtrack112/test/uavtrack112/test/labels_per_sequence")
if not labels_path.exists():
    print("Labels not generated yet — run Step 3 first")
    exit(0)

state_counts = Counter()
fc_seqs = []
conf_fc = []    # calibrated confidence for FC frames
apce_fc = []    # calibrated APCE for FC frames
conf_cc = []
apce_cc = []

for f in sorted(labels_path.glob("*.jsonl")):
    seq_fc = 0
    seq_total = 0
    for line in f.read_text().splitlines():
        if not line: continue
        r = json.loads(line)
        state = r.get("derived_state_name", "")
        state_counts[state] += 1
        seq_total += 1
        conf = r.get("confidence")
        apce = r.get("apce")
        if state == "FALSE_CONFIRMED":
            seq_fc += 1
            if conf: conf_fc.append(conf)
            if apce: apce_fc.append(apce)
        elif state == "CORRECT_CONFIRMED":
            if conf: conf_cc.append(conf)
            if apce: apce_cc.append(apce)
    if seq_fc > 0:
        fc_seqs.append((f.stem, seq_fc, seq_total))

total = sum(state_counts.values())
fc = state_counts.get("FALSE_CONFIRMED", 0)

print(f"\n=== UAVTrack112 FC Audit ===")
print(f"Total frames: {total:,}")
for s, n in sorted(state_counts.items(), key=lambda x: -x[1]):
    print(f"  {s}: {n:,} ({100*n/total:.1f}%)")

print(f"\nFC-positive sequences: {len(fc_seqs)}")
for seq, fc_n, tot in sorted(fc_seqs, key=lambda x: -x[1])[:10]:
    print(f"  {seq}: {fc_n}/{tot} = {100*fc_n/tot:.1f}% FC")

print(f"\nFC signature analysis (calibrated values):")
if conf_fc:
    print(f"  FC confidence: mean={np.mean(conf_fc):.3f}, median={np.median(conf_fc):.3f}")
    print(f"  CC confidence: mean={np.mean(conf_cc):.3f}, median={np.median(conf_cc):.3f}")
if apce_fc:
    print(f"  FC APCE: mean={np.mean(apce_fc):.3f}, median={np.median(apce_fc):.3f}")
    print(f"  CC APCE: mean={np.mean(apce_cc):.3f}, median={np.median(apce_cc):.3f}")

# Key question: is FC signature more like LaSOT (low-conf) or UAVDT (high-APCE)?
if conf_fc and conf_cc:
    fc_hc = sum(1 for c in conf_fc if c >= 0.65) / len(conf_fc)
    cc_hc = sum(1 for c in conf_cc if c >= 0.65) / len(conf_cc)
    print(f"\n  HIGH_CONFIDENCE rate: FC={100*fc_hc:.1f}%  CC={100*cc_hc:.1f}%")
    if fc_hc > 0.5:
        print("  → FC signature: HIGH-CONFIDENCE (like UAVDT) ← aerial pattern")
    else:
        print("  → FC signature: LOW-CONFIDENCE (like LaSOT) ← generic pattern")

print(f"\nUnique FC windows (approx): {fc // 16}")
PYEOF

# ---------------------------------------------------------------------------
# Step 5: Write audit report
# ---------------------------------------------------------------------------
echo "[$(date '+%H:%M:%S')] Step 5: Writing audit report to $AUDIT_OUT..."

cat > "$AUDIT_OUT" << 'MDEOF'
# UAVTrack112 Audit Report

**Date:** $(date '+%Y-%m-%d')
**Dataset path:** $DATASET_PATH
**Purpose:** Evaluate suitability for train2_v2 FC training data

## Structure
(See stdout above for measured values)

## Recommendations

### If FC% >= 3% and FC signature is HIGH-CONFIDENCE (aerial pattern):
→ **Add to train2_v2** at batch fraction 12%
→ Refit `sglatrack_all_v2` calibrator with UAVTrack112 telemetry
→ Regenerate all labels

### If FC% < 1% or FC signature is LOW-CONFIDENCE (LaSOT pattern):
→ **Skip for train2_v2**, add as hard-negative source only
→ Can still be used for CC/CU diversity

### If loader is missing:
→ Implement `csc_uav_tracking/datasets/uavtrack112.py` loader
→ Register in `csc_uav_tracking/__init__.py`

## Integration commands

```bash
# After baseline and labels are generated:
.venv/bin/python tools/fit_calibration.py \
  --tracker sglatrack \
  --telemetry_dirs \
    outputs/baselines/sglatrack/lasot/train/telemetry \
    outputs/baselines/sglatrack/got10k/val/telemetry \
    outputs/baselines/sglatrack/uavdt_sot/test/telemetry \
    outputs/baselines/sglatrack/visdrone_sot/test/telemetry \
    outputs/baselines/sglatrack/dtb70/test/telemetry \
    outputs/baselines/sglatrack/uavtrack112/test/telemetry \
  --tag all_v2 \
  --output_dir outputs/calibration
```
MDEOF

echo "[$(date '+%H:%M:%S')] Audit complete. Report: $AUDIT_OUT"
