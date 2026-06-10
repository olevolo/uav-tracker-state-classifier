#!/usr/bin/env python3
"""V3-fix full pipeline: train TCN16→TCN32, passive+control eval on 4 trackers.

Run:
    CSC_NOT_TRAINED_ON_UAV123=1 python3 -u tools/v3fix_pipeline.py 2>&1 | tee logs/v3fix_$(date +%Y%m%d_%H%M%S).log

Steps (sequential, idempotent — skips completed):
    1. Smoke-test new features (Cohen's d gate)
    2. Train TCN16 stage1
    3. Train TCN16 stage2
    4. Train TCN32 stage1
    5. Train TCN32 stage2
    6. Passive eval: TCN16 × 4 trackers
    7. Passive eval: TCN32 × 4 trackers
    8. Control mode: TCN16 × sglatrack only
    9. Paper metrics for all runs
   10. FINAL_RESULTS_V3FIX.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# Limit CPU threads to prevent process stacking / thermal throttling
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ["CSC_NOT_TRAINED_ON_UAV123"] = "1"

TRACKERS = ["sglatrack", "ortrack", "avtrack", "uetrack"]

CALIB_TAGS = {
    "sglatrack": "sglatrack_train2_v3",
    "ortrack":   "ortrack_aerial_v2",
    "avtrack":   "avtrack_aerial_v2",
    "uetrack":   "sglatrack_train2_v3",
}

TCN16_STAGE1_CKPT = ROOT / "outputs/csc_training/sglatrack_v3fix_tcn16_stage1/checkpoint_best.pth"
TCN16_STAGE2_CKPT = ROOT / "outputs/csc_training/sglatrack_v3fix_tcn16_stage2/checkpoint_best.pth"
TCN32_STAGE1_CKPT = ROOT / "outputs/csc_training/sglatrack_v3fix_tcn32_stage1/checkpoint_best.pth"
TCN32_STAGE2_CKPT = ROOT / "outputs/csc_training/sglatrack_v3fix_tcn32_stage2/checkpoint_best.pth"

CALIB_DIR  = ROOT / "outputs/calibration"
STATUS_FILE = ROOT / "STATUS_V3FIX.md"


# ---------------------------------------------------------------------------
def ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def ram_gb() -> float:
    m = psutil.virtual_memory()
    return m.used / 1e9, m.total / 1e9


def wait_ram(threshold: float = 0.80, poll: int = 30) -> None:
    while True:
        used, total = ram_gb()
        if used / total < threshold:
            return
        log(f"RAM {used:.1f}/{total:.1f} GB — waiting {poll}s …")
        time.sleep(poll)


def run(cmd: list[str], env_extra: dict | None = None) -> bool:
    env = {**os.environ, **(env_extra or {})}
    log("RUN: " + " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        log(f"  !! FAILED (exit {r.returncode})")
        return False
    return True


def write_status(step: str, detail: str = "") -> None:
    lines = [
        "# V3-Fix Pipeline Status\n",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        f"**Current step:** {step}\n",
    ]
    if detail:
        lines.append(f"\n{detail}\n")
    STATUS_FILE.write_text("".join(lines))


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_gate() -> bool:
    """Quick Cohen's d check: edge_contact and log_area_ratio must separate FC from CC."""
    log("GATE: checking new feature discriminability …")
    result = subprocess.run(
        [PYTHON, "-c", """
import sys, json, glob, numpy as np
from csc_lib.csc.features import build_sequence_features, FEATURE_NAMES
label_base = 'outputs/eval/sglatrack/uav123/test/labels_v3/uav123/test/labels_per_sequence'
fc, cc = [], []
for p in sorted(glob.glob(f'{label_base}/*.jsonl')):
    rows = [json.loads(l) for l in open(p)]
    if not rows: continue
    feats = build_sequence_features(rows, (1920, 1080))
    for t, r in enumerate(rows):
        s = r.get('derived_state_name','')
        if s == 'FALSE_CONFIRMED': fc.append(feats[t])
        elif s == 'CORRECT_CONFIRMED': cc.append(feats[t])
fc=np.array(fc); cc=np.array(cc)
def d(a,b):
    s=np.sqrt((np.var(a)+np.var(b))/2); return abs(np.mean(a)-np.mean(b))/(s+1e-9)
checks = {
    'edge_contact_score':    d(fc[:,10], cc[:,10]),
    'log_area_ratio_to_init':d(fc[:,12], cc[:,12]),
    'motion_angle_change':   d(fc[:,13], cc[:,13]),
    'log_w_ratio_to_init':   d(fc[:,11], cc[:,11]),
    'log_h_ratio_to_init':   d(fc[:,14], cc[:,14]),
}
good = sum(1 for v in checks.values() if v > 0.3)
for k,v in checks.items():
    print(f"  {k}: d={v:.3f} {'OK' if v>0.3 else 'FAIL'}")
print(f"GATE: {good}/5 features pass d>0.3")
sys.exit(0 if good >= 4 else 1)
"""],
        cwd=ROOT,
    )
    if result.returncode != 0:
        log("GATE FAILED — abort. Fix features before training.")
        return False
    log("GATE PASSED")
    return True


def step_train(config: str, label: str) -> bool:
    ckpt = ROOT / "outputs/csc_training" / f"sglatrack_{config.replace('configs/csc/csc_', '').replace('.yaml', '')}" / "checkpoint_best.pth"
    # Determine expected output dir from config
    cfg_text = (ROOT / config).read_text()
    for line in cfg_text.splitlines():
        if line.strip().startswith("output_dir:"):
            out_dir = ROOT / line.split(":", 1)[1].strip()
            ckpt = out_dir / "checkpoint_best.pth"
            break

    if ckpt.exists():
        log(f"SKIP {label} (checkpoint exists: {ckpt})")
        return True

    wait_ram(0.78)
    log(f"TRAIN {label}")
    write_status(f"Training {label}")
    ok = run([PYTHON, "-u", str(ROOT / "tools/train_csc.py"),
              "--config", str(ROOT / config)])
    if ok and ckpt.exists():
        # Read val F1 from val_metrics.json
        vm = ckpt.parent / "val_metrics.json"
        if vm.exists():
            m = json.loads(vm.read_text())
            log(f"  {label}: F1={m.get('derived_macro_f1',0):.4f} "
                f"FC_recall={m.get('derived_per_state',{}).get('FALSE_CONFIRMED',{}).get('recall','?')}")
    return ok


def step_passive(tracker: str, ckpt: Path, run_tag: str) -> Path | None:
    base = ROOT / "outputs/eval" / tracker / "uav123/test"
    passive_dir = base / run_tag

    if (passive_dir / "states").exists() and \
       len(list((passive_dir / "states").glob("*.jsonl"))) == 123:
        log(f"SKIP passive {tracker}/{run_tag} (123 state files exist)")
        return passive_dir

    if not ckpt.exists():
        log(f"SKIP passive {tracker}/{run_tag} — checkpoint missing: {ckpt}")
        return None

    wait_ram(0.82)
    log(f"PASSIVE {tracker} / {run_tag}")
    write_status(f"Passive eval: {tracker} / {run_tag}")
    ok = run([PYTHON, "-u", str(ROOT / "tools/run_with_csc.py"),
              "--tracker", tracker,
              "--dataset", "uav123", "--split", "test",
              "--csc_checkpoint", str(ckpt),
              "--csc_mode", "passive",
              "--device", "cpu",
              "--output_dir", str(base)],
             env_extra={"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})

    if not ok:
        return None

    # Rename auto-generated subdir to run_tag
    import shutil
    candidates = sorted(base.glob(f"{tracker}_uav123_test_*"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if c.name != run_tag and not (base / run_tag).exists():
            shutil.move(str(c), str(base / run_tag))
            break

    return passive_dir if (base / run_tag).exists() else None


def step_control(ckpt: Path, run_tag: str) -> Path | None:
    tracker = "sglatrack"
    base = ROOT / "outputs/eval" / tracker / "uav123/test"
    ctrl_dir = base / run_tag

    if (ctrl_dir / "states").exists() and \
       len(list((ctrl_dir / "states").glob("*.jsonl"))) == 123:
        log(f"SKIP control sglatrack/{run_tag} (123 state files exist)")
        return ctrl_dir

    if not ckpt.exists():
        log(f"SKIP control — checkpoint missing")
        return None

    wait_ram(0.82)
    log(f"CONTROL sglatrack / {run_tag}")
    write_status(f"Control mode: sglatrack / {run_tag}")
    ok = run([PYTHON, "-u", str(ROOT / "tools/run_with_csc.py"),
              "--tracker", tracker,
              "--dataset", "uav123", "--split", "test",
              "--csc_checkpoint", str(ckpt),
              "--csc_mode", "control",
              "--proactive_v3", "--proactive_threshold", "0.7",
              "--device", "cpu",
              "--output_dir", str(base)],
             env_extra={"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})

    if not ok:
        return None

    import shutil
    candidates = sorted(base.glob(f"{tracker}_uav123_test_*"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for c in candidates:
        if c.name != run_tag and not (base / run_tag).exists():
            shutil.move(str(c), str(base / run_tag))
            break

    return ctrl_dir if (base / run_tag).exists() else None


def step_paper_metrics(tracker: str, run_dir: Path, calib_tag: str) -> bool:
    out = run_dir / "paper_metrics" / "paper_metrics.csv"
    if out.exists():
        log(f"SKIP paper_metrics {tracker}/{run_dir.name}")
        return True

    labels_dir = ROOT / "outputs/eval" / tracker / "uav123/test/labels_v3/uav123/test"
    tracking_metrics = run_dir / "tracking_metrics"

    # Build tracking metrics if missing
    if not (tracking_metrics / "metrics_summary.json").exists():
        run([PYTHON, "-u", str(ROOT / "tools/evaluate_tracking_results.py"),
             "--dataset", "uav123", "--split", "test",
             "--pred_dir", str(run_dir / "predictions"),
             "--output_dir", str(tracking_metrics)])

    calib_file = CALIB_DIR / f"{calib_tag}_confidence.json"
    if not calib_file.exists():
        calib_file = CALIB_DIR / "sglatrack_train2_v3_confidence.json"

    return run([PYTHON, "-u", str(ROOT / "tools/compute_paper_metrics.py"),
                "--tracker", tracker, "--dataset", "uav123", "--split", "test",
                "--predictions_dir", str(run_dir / "predictions"),
                "--states_dir", str(run_dir / "states"),
                "--labels_dir", str(labels_dir),
                "--tracking_metrics_dir", str(tracking_metrics),
                "--confidence_calib", str(calib_file),
                "--output_dir", str(run_dir / "paper_metrics"),
                "--recovery_k", "30"])


def build_final_report() -> None:
    import csv
    lines = ["# V3-Fix Final Results\n\n",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"]

    for model_tag, ckpt in [("v3fix_tcn16", TCN16_STAGE2_CKPT),
                             ("v3fix_tcn32", TCN32_STAGE2_CKPT)]:
        if not ckpt.exists():
            continue
        vm = ckpt.parent / "val_metrics.json"
        if vm.exists():
            m = json.loads(vm.read_text())
            lines.append(f"## {model_tag}  F1={m.get('derived_macro_f1',0):.4f}\n\n")

        lines.append(f"| Tracker | AUC | Prec@20 | FCR% | FCD | Rec@30 | SC-AUC(CC) |\n")
        lines.append(f"|---|---|---|---|---|---|---|\n")

        passive_tag = f"passive_{model_tag}"
        for tracker in TRACKERS:
            pm_path = ROOT / "outputs/eval" / tracker / "uav123/test" / passive_tag / "paper_metrics/paper_metrics.csv"
            tm_path = ROOT / "outputs/eval" / tracker / "uav123/test" / passive_tag / "tracking_metrics/metrics_summary.json"
            try:
                rows_pm = list(csv.DictReader(open(pm_path)))
                import numpy as np
                total = sum(int(r['n_frames']) for r in rows_pm)
                fc_f = sum(int(r['n_fc_frames']) for r in rows_pm)
                fcr = fc_f / total * 100 if total else 0
                fcd_v = [float(r['fcd']) for r in rows_pm if float(r.get('fcd', 0)) > 0]
                fcd = np.mean(fcd_v) if fcd_v else 0
                rec_v = [float(r['recovery_at_30']) for r in rows_pm if r.get('recovery_at_30')]
                rec = np.mean(rec_v) if rec_v else 0
                cc_v = [float(r['auc_CORRECT_CONFIRMED']) for r in rows_pm if r.get('auc_CORRECT_CONFIRMED')]
                sc = np.mean(cc_v) if cc_v else 0
                tm = json.loads(open(tm_path).read())
                auc = tm['macro']['auc']
                prec = tm['macro']['precision_20']
                lines.append(f"| {tracker} | {auc:.4f} | {prec:.4f} | {fcr:.2f}% | {fcd:.1f} | {rec:.4f} | {sc:.4f} |\n")
            except Exception as e:
                lines.append(f"| {tracker} | — (missing: {e}) |\n")
        lines.append("\n")

    out = ROOT / "FINAL_RESULTS_V3FIX.md"
    out.write_text("".join(lines))
    log(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log("=" * 60)
    log("V3-FIX PIPELINE START")
    log(f"RAM: {ram_gb()[0]:.1f}/{ram_gb()[1]:.1f} GB")
    log("=" * 60)

    # --- Step 1: gate ---
    if not step_gate():
        sys.exit(1)

    # --- Steps 2-3: TCN16 ---
    if not step_train("configs/csc/csc_tcn16_v3fix_stage1.yaml", "TCN16-stage1"):
        log("TCN16 stage1 failed — aborting")
        sys.exit(1)
    if not step_train("configs/csc/csc_tcn16_v3fix_stage2.yaml", "TCN16-stage2"):
        log("TCN16 stage2 failed — aborting")
        sys.exit(1)

    # --- Steps 4-5: TCN32 ---
    step_train("configs/csc/csc_tcn32_v3fix_stage1.yaml", "TCN32-stage1")
    step_train("configs/csc/csc_tcn32_v3fix_stage2.yaml", "TCN32-stage2")

    # --- Steps 6-7: Passive eval ---
    for tracker in TRACKERS:
        calib = CALIB_TAGS[tracker]
        step_passive(tracker, TCN16_STAGE2_CKPT, "passive_v3fix_tcn16")
        step_paper_metrics(tracker, ROOT / "outputs/eval" / tracker / "uav123/test/passive_v3fix_tcn16", calib)

    for tracker in TRACKERS:
        calib = CALIB_TAGS[tracker]
        step_passive(tracker, TCN32_STAGE2_CKPT, "passive_v3fix_tcn32")
        step_paper_metrics(tracker, ROOT / "outputs/eval" / tracker / "uav123/test/passive_v3fix_tcn32", calib)

    # --- Step 8: Control mode (sglatrack only, TCN16) ---
    step_control(TCN16_STAGE2_CKPT, "control_v3fix_tcn16")
    ctrl_dir = ROOT / "outputs/eval/sglatrack/uav123/test/control_v3fix_tcn16"
    if ctrl_dir.exists():
        step_paper_metrics("sglatrack", ctrl_dir, CALIB_TAGS["sglatrack"])

    # --- Step 9: Final report ---
    build_final_report()
    write_status("DONE", "All steps completed. See FINAL_RESULTS_V3FIX.md")
    log("=" * 60)
    log("PIPELINE COMPLETE — see FINAL_RESULTS_V3FIX.md")
    log("=" * 60)


if __name__ == "__main__":
    main()
