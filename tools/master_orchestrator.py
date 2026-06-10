#!/usr/bin/env python3
"""Master autonomous orchestrator for V3 evaluation pipeline.

Runs without user intervention:
  Phase 1: Generate fartrack/uetrack baselines
  Phase 2: Paper metrics for 4 done trackers (sglatrack/ortrack/avtrack/ostrack)
  Phase 3: Wait for TCN-16 f16 training, then full eval for all 6 trackers
  Phase 4: Train TCN-32 f16 (stage1 → stage2), then eval
  Phase 5: Assemble FINAL_RESULTS.md + quality analysis

Usage:
    CSC_NOT_TRAINED_ON_UAV123=1 .venv/bin/python -u tools/master_orchestrator.py \\
        2>&1 | tee logs/master_$(date +%Y%m%d_%H%M%S).log
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
# Make tools/ importable as a flat namespace
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
TOOLS = ROOT / "tools"
CONFIGS = ROOT / "configs/csc"
CALIB_DIR = ROOT / "outputs/calibration"
BASELINES_DIR = ROOT / "outputs/baselines"
TRAINING_DIR = ROOT / "outputs/csc_training"

CKPT_TCN16_F16  = TRAINING_DIR / "sglatrack_train2_v3_stage2_tcn16_f16/checkpoint_best.pth"
CKPT_TCN32_S1   = TRAINING_DIR / "sglatrack_train2_v3_tcn32_f16_stage1/checkpoint_best.pth"
CKPT_TCN32_F16  = TRAINING_DIR / "sglatrack_train2_v3_tcn32_f16_stage2/checkpoint_best.pth"

TRACKERS = {
    "sglatrack": "sglatrack_train2_v3",
    "ortrack":   "ortrack_aerial_v2",
    "avtrack":   "avtrack_aerial_v2",
    "ostrack":   "ostrack_aerial_v2",
    "fartrack":  "sglatrack_train2_v3",
    "uetrack":   "sglatrack_train2_v3",
}

STATUS_FILE = ROOT / "STATUS.md"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], **kw) -> int:
    log(f"RUN: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kw)
    return result.returncode


def _wait_if_running(tracker: str) -> None:
    """If run_with_csc.py for this tracker is already running, wait for it."""
    while True:
        r = subprocess.run(
            ["pgrep", "-f", f"run_with_csc.py.*--tracker {tracker}"],
            capture_output=True,
        )
        if r.returncode != 0:
            break
        log(f"  WAIT: run_with_csc.py for {tracker} already running, waiting 60s...")
        time.sleep(60)


def run_check(cmd: list[str], **kw) -> None:
    rc = run(cmd, **kw)
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {' '.join(str(c) for c in cmd)}")


def write_status(lines: list[str]) -> None:
    with open(STATUS_FILE, "w") as f:
        f.write(f"# Pipeline Status — {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("\n".join(lines))
        f.write("\n")


def find_latest_run_subdir(parent: Path, tracker: str) -> Path | None:
    candidates = sorted(
        parent.glob(f"{tracker}_uav123_test_*"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Phase 1: Baselines for fartrack + uetrack
# ---------------------------------------------------------------------------

def phase1_baselines() -> None:
    log("=== PHASE 1: Baselines for fartrack + uetrack ===")
    for tracker in ("fartrack", "uetrack"):
        pred_dir = BASELINES_DIR / tracker / "uav123/test/predictions"
        if pred_dir.exists() and any(pred_dir.glob("*.txt")):
            log(f"  SKIP {tracker} baselines (exist)")
            continue
        log(f"  Generating baseline for {tracker} ...")
        run_check([PYTHON, "-u", str(TOOLS / "run_baseline.py"),
                   "--tracker", tracker,
                   "--dataset", "uav123", "--split", "test",
                   "--device", "cpu", "--skip_existing",
                   "--output_dir", str(BASELINES_DIR)])


# ---------------------------------------------------------------------------
# Phase 2: Paper metrics for 4 already-done trackers
# ---------------------------------------------------------------------------

def phase2_paper_metrics_interim() -> None:
    """Run paper metrics on the 4 completed passive_v3 runs (11-feature checkpoint).

    These serve as interim results until the f16 checkpoint is ready.
    The passive_v3 predictions already exist; only metrics need recomputing.
    """
    log("=== PHASE 2: Interim paper metrics for 4 done trackers ===")
    # Import inline to reuse helpers from run_v3_eval_pipeline
    from run_v3_eval_pipeline import (
        run_tracking_metrics, run_episode_metrics, run_paper_metrics
    )
    for tracker, calib_tag in list(TRACKERS.items())[:4]:  # sglatrack/ortrack/avtrack/ostrack
        base = ROOT / "outputs/eval" / tracker / "uav123/test"
        passive_v3 = base / "passive_v3"
        labels_v3  = base / "labels_v3"
        if not passive_v3.exists():
            log(f"  SKIP {tracker} — passive_v3 not found")
            continue
        try:
            track_m = run_tracking_metrics(tracker, passive_v3)
            run_episode_metrics(tracker, passive_v3, labels_v3)
            run_paper_metrics(tracker, calib_tag, passive_v3, labels_v3, track_m)
        except Exception as e:
            log(f"  ERROR {tracker}: {e}")


# ---------------------------------------------------------------------------
# Phase 3: Full eval with TCN-16 f16 checkpoint
# ---------------------------------------------------------------------------

def _run_passive(tracker: str, calib_tag: str, base: Path, ckpt: Path,
                 run_tag: str) -> Path:
    out_dir = base / run_tag
    if (out_dir / "metrics.json").exists():
        log(f"  SKIP passive {run_tag} ({tracker})")
        return out_dir

    # Also skip if the auto-generated alt dir already has a completed run
    alt_dir = base / f"{tracker}_uav123_test_{ckpt.stem}"
    if (alt_dir / "metrics.json").exists():
        log(f"  SKIP passive {run_tag} ({tracker}) — found completed run in {alt_dir.name}")
        return alt_dir

    run_check([PYTHON, "-u", str(TOOLS / "run_with_csc.py"),
               "--tracker", tracker, "--dataset", "uav123", "--split", "test",
               "--csc_checkpoint", str(ckpt), "--csc_mode", "passive",
               "--device", "cpu", "--run_tag", run_tag,
               "--output_dir", str(base)])

    return out_dir


def _full_metrics(tracker: str, calib_tag: str, run_dir: Path, labels_v3: Path) -> None:
    from run_v3_eval_pipeline import (
        run_build_labels, run_tracking_metrics, run_episode_metrics, run_paper_metrics
    )
    base = run_dir.parent
    labels_v3 = run_build_labels(tracker, calib_tag, base)
    track_m   = run_tracking_metrics(tracker, run_dir)
    run_episode_metrics(tracker, run_dir, labels_v3)
    run_paper_metrics(tracker, calib_tag, run_dir, labels_v3, track_m)


def phase3_eval_tcn16_f16() -> None:
    log("=== PHASE 3: Full eval with TCN-16 f16 checkpoint ===")
    ckpt = CKPT_TCN16_F16
    run_tag = "passive_v3_f16"

    for tracker, calib_tag in TRACKERS.items():
        base = ROOT / "outputs/eval" / tracker / "uav123/test"
        base.mkdir(parents=True, exist_ok=True)
        labels_v3 = base / "labels_v3"
        try:
            run_dir = _run_passive(tracker, calib_tag, base, ckpt, run_tag)
            _full_metrics(tracker, calib_tag, run_dir, labels_v3)
        except Exception as e:
            log(f"  ERROR {tracker}: {e}")
            continue

    # SGLATrack control run
    _run_control_proactive(CKPT_TCN16_F16, "control_v3_proactive_f16")


def _run_control_proactive(ckpt: Path, run_tag: str) -> None:
    base = ROOT / "outputs/eval/sglatrack/uav123/test"
    ctrl = base / run_tag
    if (ctrl / "metrics.json").exists():
        log(f"  SKIP {run_tag}")
        return
    run_check([PYTHON, "-u", str(TOOLS / "run_with_csc.py"),
               "--tracker", "sglatrack", "--dataset", "uav123", "--split", "test",
               "--csc_checkpoint", str(ckpt), "--csc_mode", "control",
               "--exit_router", "--proactive_v3", "--proactive_threshold", "0.7",
               "--device", "cpu", "--output_dir", str(base)])
    latest = find_latest_run_subdir(base, "sglatrack")
    if latest and latest != ctrl:
        shutil.move(str(latest), str(ctrl))
    labels_v3 = base / "labels_v3"
    if labels_v3.exists():
        from run_v3_eval_pipeline import run_tracking_metrics, run_paper_metrics
        track_m = run_tracking_metrics("sglatrack", ctrl)
        run_paper_metrics("sglatrack", "sglatrack_train2_v3", ctrl, labels_v3, track_m)


# ---------------------------------------------------------------------------
# Phase 4: TCN-32 training + eval
# ---------------------------------------------------------------------------

def phase4_tcn32_train_and_eval() -> None:
    log("=== PHASE 4: TCN-32 f16 training (stage1 → stage2) ===")

    # Stage 1
    if not CKPT_TCN32_S1.exists():
        log("  Starting TCN-32 stage1 training ...")
        run_check([PYTHON, "-u", str(TOOLS / "train_csc.py"),
                   "--config", str(CONFIGS / "csc_tcn32_train2_v3_f16_stage1.yaml")])
    else:
        log("  TCN-32 stage1 DONE")

    # Stage 2
    if not CKPT_TCN32_F16.exists():
        log("  Starting TCN-32 stage2 training ...")
        run_check([PYTHON, "-u", str(TOOLS / "train_csc.py"),
                   "--config", str(CONFIGS / "csc_tcn32_train2_v3_f16_stage2.yaml")])
    else:
        log("  TCN-32 stage2 DONE")

    log("=== PHASE 4b: Full eval with TCN-32 f16 checkpoint ===")
    run_tag = "passive_v3_tcn32_f16"
    for tracker, calib_tag in TRACKERS.items():
        base = ROOT / "outputs/eval" / tracker / "uav123/test"
        base.mkdir(parents=True, exist_ok=True)
        labels_v3 = base / "labels_v3"
        try:
            run_dir = _run_passive(tracker, calib_tag, base, CKPT_TCN32_F16, run_tag)
            _full_metrics(tracker, calib_tag, run_dir, labels_v3)
        except Exception as e:
            log(f"  ERROR {tracker}: {e}")

    _run_control_proactive(CKPT_TCN32_F16, "control_v3_proactive_tcn32_f16")


# ---------------------------------------------------------------------------
# Phase 5: Assemble FINAL_RESULTS.md
# ---------------------------------------------------------------------------

def _load_paper_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("aggregate", data)


def _load_tracking_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def phase5_assemble_results() -> None:
    log("=== PHASE 5: Assembling FINAL_RESULTS.md ===")

    models = [
        ("TCN-16 f16 passive",   "passive_v3_f16"),
        ("TCN-32 f16 passive",   "passive_v3_tcn32_f16"),
        ("TCN-16 f16 control",   "control_v3_proactive_f16"),
        ("TCN-32 f16 control",   "control_v3_proactive_tcn32_f16"),
    ]

    rows = []
    issues = []

    for model_name, run_tag in models:
        for tracker in TRACKERS:
            base = ROOT / "outputs/eval" / tracker / "uav123/test" / run_tag
            pm = _load_paper_metrics(base / "paper_metrics/paper_metrics.json")
            tm = _load_tracking_summary(base / "tracking_metrics/summary.json")
            if not pm and not tm:
                continue
            fcr   = pm.get("FCR", pm.get("fcr", "N/A"))
            fcd   = pm.get("FCD", pm.get("fcd", "N/A"))
            rec30 = pm.get("Recovery@30", pm.get("recovery_at_30", "N/A"))
            auc   = tm.get("auc", tm.get("success_auc", "N/A"))
            prec  = tm.get("precision_20", tm.get("precision", "N/A"))
            rows.append(f"| {model_name} | {tracker} | {auc} | {prec} | {fcr} | {fcd} | {rec30} |")

            # Flag suspicious results
            if isinstance(fcr, float) and fcr == 0.0:
                issues.append(f"⚠️  FCR=0.0 for {model_name}/{tracker} — GT labels may be missing")

    header = "| Model | Tracker | AUC | Prec@20 | FCR | FCD | Recovery@30 |\n"
    header += "|---|---|---|---|---|---|---|\n"

    with open(ROOT / "FINAL_RESULTS.md", "w") as f:
        f.write(f"# Final Results — V3 Eval (generated {time.strftime('%Y-%m-%d %H:%M:%S')})\n\n")
        f.write("## Paper Metrics\n\n")
        f.write(header)
        f.write("\n".join(rows) + "\n")
        if issues:
            f.write("\n## Quality Issues\n\n")
            f.write("\n".join(issues) + "\n")

    log(f"  Wrote FINAL_RESULTS.md ({len(rows)} rows, {len(issues)} issues)")
    for issue in issues:
        log(f"  {issue}")


# ---------------------------------------------------------------------------
# Wait helper
# ---------------------------------------------------------------------------

def wait_for_checkpoint(ckpt: Path, poll_secs: int = 60, max_hours: float = 8.0) -> bool:
    deadline = time.time() + max_hours * 3600
    log(f"  Waiting for {ckpt.name} (max {max_hours}h) ...")
    while not ckpt.exists():
        if time.time() > deadline:
            log(f"  TIMEOUT waiting for {ckpt}")
            return False
        log(f"  ... still training ({ckpt.name}) ...")
        update_status()
        time.sleep(poll_secs)
    log(f"  Checkpoint ready: {ckpt}")
    return True


# ---------------------------------------------------------------------------
# Status update (called by cron)
# ---------------------------------------------------------------------------

def update_status() -> None:
    lines = []

    def ck(label: str, path: Path) -> str:
        return f"- [{'x' if path.exists() else ' '}] {label}"

    lines.append("## Training")
    lines.append(ck("TCN-16 f16 stage2", CKPT_TCN16_F16))
    lines.append(ck("TCN-32 f16 stage1", CKPT_TCN32_S1))
    lines.append(ck("TCN-32 f16 stage2", CKPT_TCN32_F16))

    lines.append("\n## Baselines")
    for t in ("fartrack", "uetrack"):
        lines.append(ck(f"{t} baselines", BASELINES_DIR / t / "uav123/test/predictions"))

    lines.append("\n## Eval (passive_v3_f16)")
    for t in TRACKERS:
        p = ROOT / "outputs/eval" / t / "uav123/test/passive_v3_f16/paper_metrics/paper_metrics.json"
        lines.append(ck(t, p))

    lines.append("\n## Eval (passive_v3_tcn32_f16)")
    for t in TRACKERS:
        p = ROOT / "outputs/eval" / t / "uav123/test/passive_v3_tcn32_f16/paper_metrics/paper_metrics.json"
        lines.append(ck(t, p))

    lines.append("\n## Control")
    lines.append(ck("sglatrack control_v3_proactive_f16",
                    ROOT / "outputs/eval/sglatrack/uav123/test/control_v3_proactive_f16/paper_metrics/paper_metrics.json"))
    lines.append(ck("sglatrack control_v3_proactive_tcn32_f16",
                    ROOT / "outputs/eval/sglatrack/uav123/test/control_v3_proactive_tcn32_f16/paper_metrics/paper_metrics.json"))

    lines.append("\n## Final")
    lines.append(ck("FINAL_RESULTS.md", ROOT / "FINAL_RESULTS.md"))

    write_status(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(ROOT / "logs", exist_ok=True)
    log("=== Master Orchestrator started ===")
    update_status()

    # Phase 1: baselines for fartrack/uetrack (can run immediately)
    try:
        phase1_baselines()
    except Exception as e:
        log(f"Phase 1 error: {e}")

    # Phase 2: interim paper metrics for 4 done trackers
    try:
        phase2_paper_metrics_interim()
    except Exception as e:
        log(f"Phase 2 error: {e}")

    update_status()

    # Phase 3: wait for TCN-16 f16, then full eval
    log("=== Waiting for TCN-16 f16 stage2 training ===")
    if wait_for_checkpoint(CKPT_TCN16_F16, poll_secs=120, max_hours=6.0):
        try:
            phase3_eval_tcn16_f16()
        except Exception as e:
            log(f"Phase 3 error: {e}")
    else:
        log("WARN: TCN-16 f16 training did not finish in time — skipping phase 3")

    update_status()

    # Phase 4: TCN-32 training + eval
    try:
        phase4_tcn32_train_and_eval()
    except Exception as e:
        log(f"Phase 4 error: {e}")

    update_status()

    # Phase 5: assemble results
    try:
        phase5_assemble_results()
    except Exception as e:
        log(f"Phase 5 error: {e}")

    update_status()
    log("=== Master Orchestrator complete ===")


if __name__ == "__main__":
    main()
