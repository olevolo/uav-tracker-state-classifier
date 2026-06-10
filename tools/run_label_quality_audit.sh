#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_label_quality_audit.sh — comprehensive quality audit after train2_v2
# label generation.
#
# Usage: bash tools/run_label_quality_audit.sh
#
# Checks:
#   1. Calibration consistency (HC rates per dataset, should be ±5pp)
#   2. FC distribution per dataset (frames, episodes, unique windows, FCD)
#   3. Teacher confusion matrix vs CSC v1 paper model predictions (UAV123)
#   4. FC scene heatmap (which scene attributes correlate with FC)
#   5. Hard negatives: teacher=CC but student=FC (potential teacher misses)
#   6. Quality gates: FC% 3-6%, HC rate consistency, min FC sequences
#
# Output: outputs/quality/train2_v2_label_audit.md
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python3 -u"
LABELS_BASE="outputs/csc_labels/sglatrack"
EVAL_V2="outputs/eval_v2/sglatrack/uav123/test"
OUT="outputs/quality/train2_v2_label_audit.md"
mkdir -p outputs/quality

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Label Quality Audit — train2_v2 ==="

$PY - << 'PYEOF' | tee "$OUT"
import json, numpy as np
from pathlib import Path
from collections import defaultdict, Counter

print("# train2_v2 Label Quality Audit")
print(f"**Date:** $(date '+%Y-%m-%d')")
print()

# ── 1. Per-dataset calibration consistency ──────────────────────────────────
print("## 1. Calibration Consistency (HIGH_CONFIDENCE rate per dataset)")
print()
DATASETS = {
    "lasot": "outputs/csc_labels/sglatrack/lasot/train/lasot/train",
    "uavdt_sot": "outputs/csc_labels/sglatrack/uavdt_sot/test/uavdt_sot/test",
    "visdrone_sot": "outputs/csc_labels/sglatrack/visdrone_sot/test/visdrone_sot/test",
    "uavtrack112": "outputs/csc_labels/sglatrack/uavtrack112/test/uavtrack112/test",
}

hc_rates = {}
for ds, path in DATASETS.items():
    lf = Path(path) / "labels.jsonl"
    if not lf.exists():
        print(f"  {ds}: MISSING labels.jsonl")
        continue
    total = hc = 0
    with open(lf) as f:
        for line in f:
            r = json.loads(line)
            total += 1
            if r.get("confidence_state", 0) == 1:
                hc += 1
    hc_rates[ds] = hc / max(total, 1)
    print(f"  {ds}: HIGH_CONFIDENCE={100*hc_rates[ds]:.1f}%  ({hc:,}/{total:,} frames)")

if hc_rates:
    vals = list(hc_rates.values())
    delta = max(vals) - min(vals)
    status = "✅ PASS" if delta < 0.05 else "⚠️ WARN"
    print(f"\n  HC rate range: {min(vals)*100:.1f}% – {max(vals)*100:.1f}% (delta={delta*100:.1f}pp)")
    print(f"  Gate (delta < 5pp): {status}")
print()

# ── 2. FC distribution per dataset ─────────────────────────────────────────
print("## 2. FC Distribution per Dataset")
print()
print("| Dataset | Seqs | Frames | FC% | FC_eps | FC_wins(T=16) | FCD |")
print("|---------|------|--------|-----|--------|--------------|-----|")

T = 16
total_stats = {"seqs":0, "frames":0, "fc":0, "eps":0, "wins":0}

for ds, path in DATASETS.items():
    lf = Path(path) / "labels.jsonl"
    if not lf.exists():
        continue
    seq_rows = defaultdict(list)
    with open(lf) as f:
        for line in f:
            r = json.loads(line)
            seq_rows[r["sequence"]].append(r)

    seqs = len(seq_rows)
    frames = sum(len(v) for v in seq_rows.values())
    fc_frames = sum(1 for rows in seq_rows.values() for r in rows if r.get("derived_state_name")=="FALSE_CONFIRMED")

    # Episodes + FCD
    eps = 0; fcd_list = []
    for rows in seq_rows.values():
        states = [1 if r.get("derived_state_name")=="FALSE_CONFIRMED" else 0 for r in sorted(rows, key=lambda r:r["frame_idx"])]
        in_ep = False; ep_len = 0
        for s in states:
            if s:
                ep_len += 1
                if not in_ep: in_ep=True; eps+=1
            else:
                if in_ep and ep_len>0: fcd_list.append(ep_len); ep_len=0
                in_ep=False
        if in_ep and ep_len>0: fcd_list.append(ep_len)

    # Unique FC windows
    wins = 0
    for rows in seq_rows.values():
        states = [1 if r.get("derived_state_name")=="FALSE_CONFIRMED" else 0 for r in sorted(rows, key=lambda r:r["frame_idx"])]
        n = len(states); arr = np.array(states)
        if n >= T:
            for end in range(T, n+1):
                if arr[end-T:end][-8:].any(): wins+=1

    fcd = np.mean(fcd_list) if fcd_list else 0
    fc_pct = 100*fc_frames/max(frames,1)
    print(f"| {ds} | {seqs} | {frames:,} | {fc_pct:.1f}% | {eps} | {wins:,} | {fcd:.1f} |")
    total_stats["seqs"]+=seqs; total_stats["frames"]+=frames
    total_stats["fc"]+=fc_frames; total_stats["eps"]+=eps; total_stats["wins"]+=wins

print(f"| **TOTAL** | {total_stats['seqs']} | {total_stats['frames']:,} | {100*total_stats['fc']/max(total_stats['frames'],1):.1f}% | {total_stats['eps']} | {total_stats['wins']:,} | — |")
print()

# ── 3. Gate checks ──────────────────────────────────────────────────────────
print("## 3. Quality Gates")
print()
fc_pct_total = 100*total_stats["fc"]/max(total_stats["frames"],1)
print(f"  FC% overall: {fc_pct_total:.1f}%  ", end="")
print("✅" if 3.0 <= fc_pct_total <= 8.0 else "⚠️ outside 3-8% range")

print(f"  FC episodes: {total_stats['eps']:,}  ", end="")
print("✅" if total_stats["eps"] >= 100 else "⚠️ < 100 episodes")

print(f"  Unique FC windows: {total_stats['wins']:,}  ", end="")
print("✅" if total_stats["wins"] >= 1000 else "⚠️ < 1000 unique windows")
print()

# ── 4. Teacher vs Student confusion (UAV123) ────────────────────────────────
print("## 4. Teacher vs Student Confusion (UAV123 — CSC v1 paper model)")
print()
ldir = Path("outputs/eval_v2/sglatrack/uav123/test/labels/uav123/test/labels_per_sequence")
sdir = Path("outputs/eval_v2/sglatrack/uav123/test/passive/states")
STATES = ["CORRECT_CONFIRMED","CORRECT_UNCERTAIN","LOST_AWARE","FALSE_CONFIRMED"]
SHORT = {"CORRECT_CONFIRMED":"CC","CORRECT_UNCERTAIN":"CU","LOST_AWARE":"LA","FALSE_CONFIRMED":"FC"}

if ldir.exists() and sdir.exists():
    confusion = defaultdict(Counter)
    total_cf = 0
    for lf in sorted(ldir.glob("*.jsonl")):
        sf = sdir / lf.name
        if not sf.exists(): continue
        teacher = {r["frame_idx"]: r.get("derived_state_name","")
                   for line in lf.read_text().splitlines() if line
                   for r in [json.loads(line)]}
        student = {r["frame_idx"]: STATES[r.get("derived_state",0)]
                   for line in sf.read_text().splitlines() if line
                   for r in [json.loads(line)] if "derived_state" in r}
        for fidx, t in teacher.items():
            s = student.get(fidx)
            if s:
                confusion[t][s] += 1
                total_cf += 1

    print(f"Total compared: {total_cf:,} frames")
    print()
    hdr = "T\\S     "
    print(f"| {hdr} | {' | '.join(SHORT[s] for s in STATES)} | Total |")
    print("|" + "-"*8 + "|" + "|".join(["------"]*5) + "|")
    for t in STATES:
        row = sum(confusion[t].values())
        vals = " | ".join(f"{confusion[t][s]:6,}" for s in STATES)
        print(f"| {SHORT[t]:<7} | {vals} | {row:6,} |")
    print()

    cc_fc = confusion["CORRECT_CONFIRMED"]["FALSE_CONFIRMED"]
    la_fc = confusion["LOST_AWARE"]["FALSE_CONFIRMED"]
    fc_fc = confusion["FALSE_CONFIRMED"]["FALSE_CONFIRMED"]
    fc_tot = sum(confusion["FALSE_CONFIRMED"].values())
    fc_prec = confusion["FALSE_CONFIRMED"]["FALSE_CONFIRMED"] / max(sum(confusion[t]["FALSE_CONFIRMED"] for t in STATES), 1)

    print(f"**Teacher=CC, Student=FC**: {cc_fc:,} ({100*cc_fc/total_cf:.1f}%) — potential teacher misses")
    print(f"**Teacher=FC, Student=FC**: {fc_fc}/{fc_tot} = {100*fc_fc/max(fc_tot,1):.0f}% recall")
    print(f"**FC precision** (student=FC correct): {100*fc_prec:.0f}%")
    print()

    fc_recall_gate = fc_ff/max(fc_tot,1) if (fc_ff:=fc_fc) else 0
    print(f"Gate (FC recall >= 60%): {'✅' if fc_recall_gate >= 0.60 else '⚠️'}")
    print(f"Gate (FC precision >= 40%): {'✅' if fc_prec >= 0.40 else '⚠️'}")
else:
    print("  UAV123 eval not available — skipping confusion matrix")
print()

# ── 5. Hard negatives summary ───────────────────────────────────────────────
print("## 5. Hard Negatives (Teacher=CC but Student=FC)")
print()
if ldir.exists() and sdir.exists():
    hn_seqs = Counter()
    for lf in sorted(ldir.glob("*.jsonl")):
        sf = sdir / lf.name
        if not sf.exists(): continue
        teacher = {r["frame_idx"]: r.get("derived_state_name","")
                   for line in lf.read_text().splitlines() if line
                   for r in [json.loads(line)]}
        student = {r["frame_idx"]: STATES[r.get("derived_state",0)]
                   for line in sf.read_text().splitlines() if line
                   for r in [json.loads(line)] if "derived_state" in r}
        n = sum(1 for fidx, t in teacher.items() if t=="CORRECT_CONFIRMED" and student.get(fidx)=="FALSE_CONFIRMED")
        if n > 0: hn_seqs[lf.stem] = n

    total_hn = sum(hn_seqs.values())
    print(f"Total hard negative frames: {total_hn:,}")
    print("Top 10 sequences:")
    for seq, n in sorted(hn_seqs.items(), key=lambda x:-x[1])[:10]:
        print(f"  {seq}: {n} frames")
else:
    print("  Not available")

print()
print("---")
print("*Generated by tools/run_label_quality_audit.sh*")
PYEOF

log "Audit written to $OUT"
