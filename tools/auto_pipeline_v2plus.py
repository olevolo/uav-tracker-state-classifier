"""
Auto-pipeline: runs after LaSOT full baseline completes.
Monitors telemetry dir, triggers when 220 sequences are done:
  1. Audit sim_initial quality
  2. Regenerate LaSOT labels
  3. Merge all labels into train2_v2plus_full_combined
  4. Launch V2++ full training
"""
import subprocess, time, json, numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
TEL_DIR = ROOT / "outputs/baselines_v2plus/sglatrack/sglatrack/lasot/train/telemetry"
LOG = ROOT / "logs/auto_pipeline.log"

def log(msg):
    print(msg, flush=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

def run(cmd, **kw):
    log(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        log(f"  ERROR rc={r.returncode}")
        sys.exit(1)
    return r

# ── 1. Wait for LaSOT ─────────────────────────────────────────────────────────
log("=== Waiting for LaSOT 220/220 ===")
while True:
    done = len(list(TEL_DIR.glob("*.jsonl")))
    log(f"  {done}/220 sequences")
    if done >= 220:
        break
    time.sleep(120)

# ── 2. Audit ──────────────────────────────────────────────────────────────────
log("\n=== AUDIT: sim_initial quality ===")
missing = 0  # non-init frames without sim
total_tracking = 0
sim_vals = []

for f in sorted(TEL_DIR.glob("*.jsonl")):
    for line in open(f):
        d = json.loads(line)
        if d.get("init"):  # skip init frame
            continue
        total_tracking += 1
        if "initial_template_sim" not in d:
            missing += 1
        else:
            sim_vals.append(d["initial_template_sim"])

sim_vals = np.array(sim_vals)
coverage = (total_tracking - missing) / max(1, total_tracking) * 100
sim_one_pct = (np.abs(sim_vals - 1.0) < 1e-4).mean() * 100

log(f"  Tracking frames: {total_tracking:,}")
log(f"  With real sim: {total_tracking-missing:,} ({coverage:.1f}%)")
log(f"  Frames with sim≈1.0: {(np.abs(sim_vals-1.0)<1e-4).sum()} ({sim_one_pct:.1f}%)")
log(f"  sim range: min={sim_vals.min():.4f} max={sim_vals.max():.4f} mean={sim_vals.mean():.4f}")

if coverage < 95.0:
    log(f"  WARNING: coverage {coverage:.1f}% < 95% — some sequences may be incomplete")
elif sim_one_pct > 5.0:
    log(f"  WARNING: {sim_one_pct:.1f}% frames have sim≈1.0 — check for fake defaults")
else:
    log("  AUDIT PASSED ✓")

# ── 3. Regenerate LaSOT labels ────────────────────────────────────────────────
log("\n=== Regenerate LaSOT labels ===")
run([
    str(ROOT / ".venv/bin/python3"), str(ROOT / "tools/build_scene_state_labels.py"),
    "--dataset", "lasot", "--split", "train",
    "--baseline_dir", str(ROOT / "outputs/baselines_v2plus/sglatrack/sglatrack"),
    "--calibration_dir", str(ROOT / "outputs/calibration"),
    "--output_dir", str(ROOT / "outputs/csc_labels_v2plus"),
], capture_output=True, text=True)
log("  Labels generated ✓")

# ── 4. Merge all labels ───────────────────────────────────────────────────────
log("\n=== Merge labels → train2_v2plus_full_combined ===")

NEW_BASE = ROOT / "outputs/baselines_v2plus/sglatrack/sglatrack"
NEW_LABELS = ROOT / "outputs/csc_labels_v2plus"
OUT_DIR = ROOT / "outputs/csc_labels/sglatrack/train2_v2plus_full_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def patch_sim(labels_path, tel_base, dataset, split):
    tel_dir = tel_base / dataset / split / "telemetry"
    current_seq = None; tel_map = {}; patched = []
    for line in open(labels_path):
        d = json.loads(line)
        seq = d.get("sequence")
        if seq != current_seq:
            tf = tel_dir / f"{seq}.jsonl"
            tel_map = {}
            if tf.exists():
                for tl in open(tf):
                    tr = json.loads(tl)
                    fi = tr.get("frame_idx")
                    if fi is not None:
                        tel_map[fi] = tr
            current_seq = seq
        fi = d.get("frame_idx", -1)
        tr = tel_map.get(fi, {})
        sim = tr.get("initial_template_sim")
        if sim is not None:
            d["initial_template_sim"] = round(float(sim), 6)
        drift = tr.get("appearance_drift")
        if drift is not None:
            d["appearance_drift"] = round(float(drift), 6)
        patched.append(d)
    return patched

total = sim_total = 0
with open(OUT_DIR / "labels.jsonl", "w") as out:
    # existing non-LaSOT: got10k, dtb70, visdrone (already patched)
    for ds, split in [("got10k","val"), ("dtb70","test"), ("visdrone_sot","test")]:
        lpath = NEW_LABELS / ds / split / "labels.jsonl"
        if not lpath.exists():
            log(f"  MISSING {lpath}, skipping")
            continue
        rows = patch_sim(lpath, NEW_BASE, ds, split)
        n_sim = sum(1 for r in rows if "initial_template_sim" in r)
        for r in rows: out.write(json.dumps(r) + "\n")
        total += len(rows); sim_total += n_sim
        log(f"  {ds}/{split}: {len(rows)} rows, {n_sim} with sim ({n_sim/len(rows)*100:.1f}%)")

    # NEW LaSOT (all 220 sequences, all with real sim)
    lpath = NEW_LABELS / "lasot/train/labels.jsonl"
    rows = patch_sim(lpath, NEW_BASE, "lasot", "train")
    n_sim = sum(1 for r in rows if "initial_template_sim" in r)
    for r in rows: out.write(json.dumps(r) + "\n")
    total += len(rows); sim_total += n_sim
    log(f"  lasot/train: {len(rows)} rows, {n_sim} with sim ({n_sim/len(rows)*100:.1f}%)")

log(f"  Total: {total:,} rows, {sim_total:,} with real sim ({sim_total/total*100:.1f}%)")

import shutil
src = ROOT / "outputs/csc_training/sglatrack_train2_v2_tcn16/label_mapping.json"
if src.exists(): shutil.copy(src, OUT_DIR / "label_mapping.json")

# ── 5. Launch V2++ full training ──────────────────────────────────────────────
log("\n=== Launch V2++ full training ===")

config_src = ROOT / "configs/csc/csc_tcn16_train2.yaml"
import yaml
cfg = yaml.safe_load(open(config_src))
cfg["labels_dir"] = str(ROOT / "outputs/csc_labels/sglatrack/train2_v2plus_full_combined")
cfg["output_dir"] = str(ROOT / "outputs/csc_training/sglatrack_v2plus_full2_tcn16")

config_path = "/tmp/csc_v2plus_full2.yaml"
with open(config_path, "w") as f:
    yaml.dump(cfg, f)

(ROOT / "outputs/csc_training/sglatrack_v2plus_full2_tcn16").mkdir(parents=True, exist_ok=True)
train_log = open(ROOT / "logs/train_v2plus_full2.log", "w")
p = subprocess.Popen(
    [str(ROOT / ".venv/bin/python3"), str(ROOT / "tools/train_csc.py"),
     "--config", config_path],
    stdout=train_log, stderr=subprocess.STDOUT
)
log(f"  V2++ training started PID={p.pid}")
log("  Monitor: tail -f logs/train_v2plus_full2.log")
log("\n=== Pipeline complete ===")
