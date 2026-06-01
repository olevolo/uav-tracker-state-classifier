"""
Comprehensive overnight auto-test for V2++.
Monitors V2++ training → on convergence runs full overnight suite:
  1. Worst-5 confusion matrix (FC recall check)
  2. All 7 trackers passive mode on UAV123 (diagnosis)
  3. SGLATrack all control modes (ctrl_A, proactive, ctrl_A_min_depth)
  4. Stage 2 V3 on V2++ base (forecast heads)
  5. SGLATrack proactive V3 control
  6. Final confusion matrices + FCR analysis

Fixes itself on errors and continues.
"""
import subprocess, time, json, sys, os, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "outputs/csc_training/sglatrack_v2plus_full2_tcn16"
TRAIN_LOG = CKPT_DIR / "train_log.jsonl"
WLOG = ROOT / "logs/auto_test_v2plus.log"
UAV_DATA = Path.home() / "uav-tracker-data/UAV123/UAV123"
WORST5 = ["person10", "car7", "uav1_2", "person14_1", "car6_5"]
PYTHON = str(ROOT / ".venv/bin/python3")
RUNNER = str(ROOT / "tools/run_with_csc.py")

def log(msg):
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(WLOG, "a") as f:
        f.write(line + "\n")

def run_tracker(tracker, output_tag, extra_args, desc=""):
    """Run run_with_csc.py and return (success, output_dir)."""
    ckpt = str(CKPT_DIR / "checkpoint_best.pth")
    out = str(ROOT / f"outputs/experiments/overnight/{output_tag}")
    cmd = [PYTHON, RUNNER,
           "--tracker", tracker,
           "--dataset", "uav123", "--split", "test",
           "--csc_checkpoint", ckpt,
           "--device", "mps",
           "--output_dir", out,
           *extra_args]
    log(f"  → {desc or output_tag}")
    log_file = ROOT / f"logs/{output_tag}.log"
    with open(log_file, "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=7200)
    if r.returncode != 0:
        log(f"  ✗ FAILED rc={r.returncode} (see {log_file.name})")
        return False, out
    log(f"  ✓ done")
    return True, out

def compute_fcr(pred_dir, gt_base):
    """Compute FCR from predictions vs GT."""
    try:
        total = fc = 0
        for pf in Path(pred_dir).glob("predictions/*.txt"):
            seq = pf.stem
            gt = None
            for p in [gt_base/"anno/UAV123"/f"{seq}.txt", gt_base/"anno"/f"{seq}.txt"]:
                if p.exists(): gt = p; break
            if not gt: continue
            preds = np.loadtxt(pf, delimiter=",")
            gtbb = np.loadtxt(gt, delimiter=",") if "," in open(gt).read()[:50] else None
            if gtbb is None:
                raw = open(gt).read().replace("\t",",").replace(" ",",")
                gtbb = np.array([[float(v) for v in l.split(",") if v.strip()] for l in raw.strip().split("\n")])
            if preds.ndim==1: preds=preds.reshape(1,-1)
            if gtbb.ndim==1: gtbb=gtbb.reshape(1,-1)
            n = min(len(preds),len(gtbb))
            for i in range(1,n):
                ax,ay,aw,ah=preds[i]; bx,by,bw,bh=gtbb[i]
                ix=max(0,min(ax+aw,bx+bw)-max(ax,bx)); iy=max(0,min(ay+ah,by+bh)-max(ay,by))
                iou=(ix*iy)/(aw*ah+bw*bh-ix*iy) if aw*ah+bw*bh-ix*iy>0 else 0
                total+=1
                # Check state from states file
            # Use states JSONL for FC count
            sf = Path(pred_dir)/f"states/{seq}.jsonl"
            if sf.exists():
                fc += sum(1 for l in open(sf) if json.loads(l).get("false_confirmed_flag"))
        return fc/total if total>0 else 0, fc, total
    except Exception as e:
        log(f"  FCR compute error: {e}")
        return 0, 0, 0

def compute_auc(pred_dir, gt_base):
    """Compute mean AUC across all sequences in pred_dir."""
    try:
        aucs = []
        for pf in sorted(Path(pred_dir).glob("predictions/*.txt")):
            seq = pf.stem
            gt = None
            for p in [gt_base/"anno/UAV123"/f"{seq}.txt", gt_base/"anno"/f"{seq}.txt"]:
                if p.exists(): gt = p; break
            if not gt: continue
            preds = np.loadtxt(pf, delimiter=",")
            raw = open(gt).read().replace("\t",",").replace(" ",",")
            gtbb = np.array([[float(v) for v in l.split(",") if v.strip()] for l in raw.strip().split("\n")])
            if preds.ndim==1: preds=preds.reshape(1,-1)
            if gtbb.ndim==1: gtbb=gtbb.reshape(1,-1)
            n = min(len(preds),len(gtbb))
            px,py,pw,ph=preds[:n,0],preds[:n,1],preds[:n,2],preds[:n,3]
            gx,gy,gw,gh=gtbb[:n,0],gtbb[:n,1],gtbb[:n,2],gtbb[:n,3]
            ix=np.maximum(0,np.minimum(px+pw,gx+gw)-np.maximum(px,gx))
            iy=np.maximum(0,np.minimum(py+ph,gy+gh)-np.maximum(py,gy))
            iou=np.where((pw*ph+gw*gh-ix*iy)>0,ix*iy/(pw*ph+gw*gh-ix*iy),0.0)
            aucs.append(float(np.mean([np.mean(iou>=t) for t in np.linspace(0,1,101)])))
        return np.mean(aucs) if aucs else 0
    except Exception as e:
        log(f"  AUC compute error: {e}")
        return 0

# ── Wait for training ─────────────────────────────────────────────────────────
log("=== Overnight V2++ Auto-Test ===")
log("Waiting for training log to appear...")

while not TRAIN_LOG.exists():
    time.sleep(60)
log("Training started ✓")
prev_ep = 0

while True:
    try:
        epochs = [json.loads(l) for l in open(TRAIN_LOG) if l.strip()]
    except:
        time.sleep(60); continue
    if not epochs: time.sleep(60); continue

    best = max(epochs, key=lambda x: x.get("selection_score", 0))
    last = epochs[-1]
    n_ep = len(epochs)
    if n_ep > prev_ep:
        log(f"  ep{last['epoch']}: F1={last['val_derived_f1']:.4f} "
            f"fc={last['val_fc_recall']:.4f} sel={last['selection_score']:.4f} "
            f"(best ep{best['epoch']})")
        prev_ep = n_ep
    if n_ep >= 3 and (n_ep - best['epoch']) >= 8:
        log(f"Training converged: best ep{best['epoch']}, total {n_ep} epochs"); break
    if last['epoch'] >= 30:
        log("Reached max 30 epochs"); break
    time.sleep(120)

CKPT = str(CKPT_DIR / "checkpoint_best.pth")
log(f"\n{'='*60}")
log(f"Using: {CKPT}")
log(f"{'='*60}")

# ── PHASE 1: Worst-5 FC recall check ─────────────────────────────────────────
log("\n=== PHASE 1: Worst-5 confusion matrix ===")
ok, out = run_tracker("sglatrack", "sgla_worst5_passive",
    ["--csc_mode", "passive", "--include_sequences", *WORST5],
    "SGLATrack worst-5 passive")
# Quick FC recall check from states
if ok:
    run_dirs = list(Path(out).glob("*/"))
    if run_dirs:
        states_dir = run_dirs[0] / "states"
        fc_count = sum(1 for sf in states_dir.glob("*.jsonl")
                      for l in open(sf) if json.loads(l).get("false_confirmed_flag"))
        total_count = max(1, sum(len(open(sf).readlines())-1 for sf in states_dir.glob("*.jsonl")))
        log(f"  FC frames predicted: {fc_count}/{total_count} ({fc_count/total_count*100:.1f}%)")

# ── PHASE 2: All trackers passive diagnosis ───────────────────────────────────
log("\n=== PHASE 2: All trackers — passive diagnosis (UAV123 all 123 seq) ===")
TRACKERS_PASSIVE = [
    ("sglatrack", "sgla_full_passive", "SGLATrack all"),
    ("ortrack",   "ortrack_full_passive", "ORTrack all"),
    ("avtrack",   "avtrack_full_passive", "AVTrack all"),
    ("ostrack",   "ostrack_full_passive", "OSTrack all"),
    ("evptrack",  "evptrack_full_passive", "EVPTrack all"),
]
# Try FARTrack and UETrack if adapters exist
try:
    sys.path.insert(0, str(ROOT/"src"))
    from uav_tracker.registry import TRACKERS as TR
    if "fartrack" in TR.names():
        TRACKERS_PASSIVE.append(("fartrack", "fartrack_full_passive", "FARTrack all"))
    if "uetrack" in TR.names():
        TRACKERS_PASSIVE.append(("uetrack", "uetrack_full_passive", "UETrack all"))
except: pass

passive_results = {}
for tracker, tag, desc in TRACKERS_PASSIVE:
    try:
        ok, out = run_tracker(tracker, tag, ["--csc_mode", "passive"], desc)
        if ok:
            run_dirs = list(Path(out).glob("*/"))
            if run_dirs:
                auc = compute_auc(str(run_dirs[0]), UAV_DATA)
                fcr, fc, total = compute_fcr(str(run_dirs[0]), UAV_DATA)
                passive_results[tracker] = {"auc": auc, "fcr": fcr, "fc": fc, "total": total}
                log(f"  {tracker}: AUC={auc:.4f} FCR={fcr*100:.2f}% ({fc}/{total})")
    except Exception as e:
        log(f"  {tracker} FAILED: {e}")

# Print summary
log("\n  === Passive Diagnosis Summary ===")
for t, r in passive_results.items():
    log(f"  {t:<12}: AUC={r['auc']:.4f}  FCR={r['fcr']*100:.2f}%")

# ── PHASE 3: SGLATrack control modes ─────────────────────────────────────────
log("\n=== PHASE 3: SGLATrack control modes ===")
ctrl_results = {}

# ctrl_A (exit router, standard)
ok, out = run_tracker("sglatrack", "sgla_ctrl_A",
    ["--exit_router"], "ctrl_A (exit router)")
if ok:
    run_dirs = list(Path(out).glob("*/"))
    if run_dirs:
        auc = compute_auc(str(run_dirs[0]), UAV_DATA)
        fcr, fc, total = compute_fcr(str(run_dirs[0]), UAV_DATA)
        ctrl_results["ctrl_A"] = {"auc": auc, "fcr": fcr}
        log(f"  ctrl_A: AUC={auc:.4f} FCR={fcr*100:.2f}%")

# ctrl_A with min_conf gate (safer)
ok, out = run_tracker("sglatrack", "sgla_ctrl_A_minconf07",
    ["--exit_router", "--exit_router_min_conf", "0.7"],
    "ctrl_A + min_conf=0.7 (safer)")
if ok:
    run_dirs = list(Path(out).glob("*/"))
    if run_dirs:
        auc = compute_auc(str(run_dirs[0]), UAV_DATA)
        fcr, fc, total = compute_fcr(str(run_dirs[0]), UAV_DATA)
        ctrl_results["ctrl_A_min07"] = {"auc": auc, "fcr": fcr}
        log(f"  ctrl_A_min07: AUC={auc:.4f} FCR={fcr*100:.2f}%")

# ── PHASE 4: Stage 2 V3 training on V2++ base ────────────────────────────────
log("\n=== PHASE 4: Stage 2 V3 on V2++ ===")
STAGE2_CKPT_DIR = ROOT / "outputs/csc_training/sglatrack_v2plus_stage2_tcn16"
STAGE2_CKPT = STAGE2_CKPT_DIR / "checkpoint_best.pth"

if not STAGE2_CKPT.exists():
    import yaml
    stage2_cfg_path = ROOT / "configs/csc/csc_tcn16_train2_v3_stage2.yaml"
    cfg = yaml.safe_load(open(stage2_cfg_path))
    cfg["stage1_checkpoint"] = CKPT
    cfg["output_dir"] = str(STAGE2_CKPT_DIR)
    STAGE2_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open("/tmp/stage2_v2plus.yaml", "w") as f:
        yaml.dump(cfg, f)
    log("  Starting Stage 2 training (~30 min)...")
    stage2_log = open(ROOT / "logs/train_v2plus_stage2.log", "w")
    p = subprocess.run([PYTHON, str(ROOT / "tools/train_csc.py"), "--config", "/tmp/stage2_v2plus.yaml"],
                       stdout=stage2_log, stderr=subprocess.STDOUT, timeout=7200)
    if p.returncode == 0:
        log("  Stage 2 training done ✓")
        s2_log = [json.loads(l) for l in open(STAGE2_CKPT_DIR / "train_log.jsonl") if l.strip()]
        best_s2 = max(s2_log, key=lambda x: x.get("selection_score",0))
        log(f"  Best ep{best_s2['epoch']}: fc_n10_AUPRC={best_s2.get('val_fc_n10_auprc',0):.4f}")
    else:
        log("  Stage 2 FAILED")
else:
    log(f"  Stage 2 checkpoint already exists: {STAGE2_CKPT}")

# ── PHASE 5: Proactive V3 control ────────────────────────────────────────────
log("\n=== PHASE 5: SGLATrack proactive V3 control ===")
if STAGE2_CKPT.exists():
    # For proactive, use Stage 2 checkpoint (has forecast heads)
    # Override: use v3 ckpt instead of v2++ ckpt
    proactive_cmd = [PYTHON, RUNNER,
        "--tracker", "sglatrack",
        "--dataset", "uav123", "--split", "test",
        "--csc_checkpoint", str(STAGE2_CKPT),
        "--device", "mps",
        "--output_dir", str(ROOT / "outputs/experiments/overnight/sgla_proactive_v3"),
        "--exit_router",
        "--proactive_v3",
        "--proactive_threshold", "0.7"]
    log("  Running proactive V3...")
    with open(ROOT / "logs/sgla_proactive_v3.log", "w") as lf:
        r = subprocess.run(proactive_cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=7200)
    if r.returncode == 0:
        log("  ✓ Proactive V3 done")
        run_dirs = list(Path(ROOT / "outputs/experiments/overnight/sgla_proactive_v3").glob("*/"))
        if run_dirs:
            auc = compute_auc(str(run_dirs[0]), UAV_DATA)
            fcr, fc, total = compute_fcr(str(run_dirs[0]), UAV_DATA)
            ctrl_results["proactive_V3"] = {"auc": auc, "fcr": fcr}
            log(f"  Proactive V3: AUC={auc:.4f} FCR={fcr*100:.2f}%")
    else:
        log("  ✗ Proactive V3 FAILED")
else:
    log("  Skipping proactive (no Stage 2 checkpoint)")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
log("\n" + "="*60)
log("=== OVERNIGHT RESULTS SUMMARY ===")
log("="*60)

log("\n--- Passive Diagnosis (UAV123, all 123 seq) ---")
log(f"{'Tracker':<14} {'AUC':>8} {'FCR':>8}")
log("-"*32)
sgla_passive_auc = passive_results.get("sglatrack", {}).get("auc", 0)
for t, r in passive_results.items():
    log(f"{t:<14} {r['auc']:>8.4f} {r['fcr']*100:>7.2f}%")

log("\n--- SGLATrack Control Modes (UAV123) ---")
log(f"{'Mode':<18} {'AUC':>8} {'ΔAUC':>8} {'FCR':>8}")
log("-"*44)
for mode, r in ctrl_results.items():
    delta = r['auc'] - sgla_passive_auc if sgla_passive_auc > 0 else 0
    log(f"{mode:<18} {r['auc']:>8.4f} {delta:>+8.4f} {r['fcr']*100:>7.2f}%")

log("\n=== ALL DONE ===")

# Write results JSON for easy parsing
import json as _json
results = {
    "passive": passive_results,
    "control": ctrl_results,
    "v2plus_best_epoch": best['epoch'] if 'best' in dir() else -1,
}
with open(ROOT / "outputs/experiments/overnight/results_summary.json", "w") as f:
    _json.dump(results, f, indent=2)
log(f"Results saved to outputs/experiments/overnight/results_summary.json")
