#!/usr/bin/env python3
"""Autonomous status checker + healer — runs every 10 min via cron.

Checks:
  - Training progress (log tail, checkpoint age)
  - Baseline completeness (count .txt files vs 123 expected)
  - Passive eval completeness (states/*.jsonl count)
  - Paper metrics quality (FCR != 0, AUC reasonable, GT labels loaded)
  - Detects stale/wrong outputs and re-triggers steps

Writes STATUS.md with ✅/❌/⚠️ for each step.
Writes HEALTH_LOG.jsonl for trend analysis.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
TOOLS = ROOT / "tools"
TRAINING_DIR = ROOT / "outputs/csc_training"
BASELINES_DIR = ROOT / "outputs/baselines"
EVAL_DIR = ROOT / "outputs/eval"

EXPECTED_SEQS = 123  # UAV123

CKPT_TCN16_F16 = TRAINING_DIR / "sglatrack_train2_v3_stage2_tcn16_f16/checkpoint_best.pth"
CKPT_TCN32_S1  = TRAINING_DIR / "sglatrack_train2_v3_tcn32_f16_stage1/checkpoint_best.pth"
CKPT_TCN32_F16 = TRAINING_DIR / "sglatrack_train2_v3_tcn32_f16_stage2/checkpoint_best.pth"

TRACKERS = ["sglatrack", "ortrack", "avtrack", "ostrack", "fartrack", "uetrack"]
TRACKERS_CALIB = {
    "sglatrack": "sglatrack_train2_v3",
    "ortrack":   "ortrack_aerial_v2",
    "avtrack":   "avtrack_aerial_v2",
    "ostrack":   "ostrack_aerial_v2",
    "fartrack":  "sglatrack_train2_v3",
    "uetrack":   "sglatrack_train2_v3",
}

STATUS_FILE = ROOT / "STATUS.md"
HEALTH_LOG  = ROOT / "logs/health.jsonl"

NOW = time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{NOW}] {msg}", flush=True)


def count_files(d: Path, glob: str) -> int:
    if not d.exists():
        return 0
    return len(list(d.glob(glob)))


def read_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def tail_log(log_file: Path, n: int = 5) -> list[str]:
    if not log_file.exists():
        return []
    try:
        lines = log_file.read_text().splitlines()
        return lines[-n:]
    except Exception:
        return []


def is_already_running(tracker: str, script: str = "run_with_csc.py") -> bool:
    """Return True if a process matching script+tracker is already running."""
    r = subprocess.run(
        ["pgrep", "-f", f"{script}.*--tracker {tracker}"],
        capture_output=True,
    )
    return r.returncode == 0


def run_bg(cmd: list[str]) -> None:
    log(f"  LAUNCH: {' '.join(str(c) for c in cmd)}")
    env = {**os.environ, "CSC_NOT_TRAINED_ON_UAV123": "1"}
    log_path = ROOT / f"logs/heal_{time.strftime('%Y%m%d_%H%M%S')}_{Path(cmd[2]).stem}.log"
    with open(log_path, "w") as lf:
        subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)
    log(f"  → log: {log_path.name}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

class Check:
    def __init__(self, name: str):
        self.name = name
        self.status = "?"   # ✅ ❌ ⚠️ 🔄
        self.detail = ""
        self.healed = False

    def ok(self, detail: str = "") -> None:
        self.status, self.detail = "✅", detail

    def fail(self, detail: str) -> None:
        self.status, self.detail = "❌", detail

    def warn(self, detail: str) -> None:
        self.status, self.detail = "⚠️", detail

    def running(self, detail: str = "") -> None:
        self.status, self.detail = "🔄", detail

    def __str__(self) -> str:
        h = " [HEALED]" if self.healed else ""
        return f"{self.status} **{self.name}**{h}: {self.detail}"


# ---------------------------------------------------------------------------
# Training checks
# ---------------------------------------------------------------------------

def check_training(checks: list[Check]) -> None:
    training_items = [
        ("TCN-16 f16 stage2", CKPT_TCN16_F16,
         TRAINING_DIR / "sglatrack_train2_v3_stage2_tcn16_f16"),
        ("TCN-32 f16 stage1", CKPT_TCN32_S1,
         TRAINING_DIR / "sglatrack_train2_v3_tcn32_f16_stage1"),
        ("TCN-32 f16 stage2", CKPT_TCN32_F16,
         TRAINING_DIR / "sglatrack_train2_v3_tcn32_f16_stage2"),
    ]
    for name, ckpt, out_dir in training_items:
        c = Check(f"Train {name}")
        if ckpt.exists():
            vm = read_json(out_dir / "val_metrics.json")
            f1 = vm.get("derived_macro_f1") or vm.get("macro_f1") or "?"
            c.ok(f"checkpoint exists, val_f1={f1}")
        else:
            # Check if training log exists and is progressing
            logs = sorted((ROOT / "logs").glob(f"train_*f16*.log"))
            if logs:
                tail = tail_log(logs[-1], 3)
                last = tail[-1] if tail else ""
                if "epoch" in last.lower() or "loss" in last.lower() or "val" in last.lower():
                    c.running(f"training in progress — {last[:80]}")
                elif "error" in last.lower() or "traceback" in last.lower():
                    c.fail(f"training error: {last[:80]}")
                else:
                    c.running(f"log exists: {last[:60]}")
            else:
                c.fail("no checkpoint, no training log — not started")
        checks.append(c)


# ---------------------------------------------------------------------------
# Baseline checks
# ---------------------------------------------------------------------------

def check_baselines(checks: list[Check]) -> None:
    for tracker in TRACKERS:
        c = Check(f"Baseline {tracker}")
        pred_dir = BASELINES_DIR / tracker / "uav123/test/predictions"
        n = count_files(pred_dir, "*.txt")
        if n == EXPECTED_SEQS:
            c.ok(f"{n}/{EXPECTED_SEQS} predictions")
        elif n > 0:
            c.warn(f"{n}/{EXPECTED_SEQS} predictions — incomplete")
        else:
            c.fail(f"0 predictions")
            if tracker in ("fartrack", "uetrack"):
                log(f"  HEAL: launching baseline for {tracker}")
                run_bg([PYTHON, "-u", str(TOOLS / "run_baseline.py"),
                        "--tracker", tracker, "--dataset", "uav123",
                        "--split", "test", "--device", "cpu",
                        "--skip_existing", "--output_dir", str(BASELINES_DIR)])
                c.healed = True
        checks.append(c)


# ---------------------------------------------------------------------------
# Passive eval checks
# ---------------------------------------------------------------------------

def check_passive(checks: list[Check], run_tag: str, ckpt: Path) -> None:
    for tracker in TRACKERS:
        c = Check(f"Passive {run_tag} — {tracker}")
        base = EVAL_DIR / tracker / "uav123/test"
        run_dir = base / run_tag
        states_dir = run_dir / "states"
        n_states = count_files(states_dir, "*.jsonl")

        # Fallback: check the auto-generated run_tag dir (older runs wrote there)
        if n_states == 0 and ckpt.exists():
            alt_run_tag = f"{tracker}_uav123_test_{ckpt.stem}"
            alt_states = base / alt_run_tag / "states"
            n_states_alt = count_files(alt_states, "*.jsonl")
            if n_states_alt > 0:
                n_states = n_states_alt
                log(f"  NOTE: {tracker} states found in alt path ({alt_run_tag}), {n_states} files")

        if n_states == EXPECTED_SEQS:
            c.ok(f"{n_states} state files")
        elif n_states > 0:
            c.warn(f"{n_states}/{EXPECTED_SEQS} state files — incomplete, may still running")
        else:
            if not ckpt.exists():
                c.warn(f"checkpoint not ready yet: {ckpt.name}")
            else:
                c.fail(f"0 state files — not started or failed")
                # Check if baseline exists before triggering
                pred_dir = BASELINES_DIR / tracker / "uav123/test/predictions"
                if count_files(pred_dir, "*.txt") == EXPECTED_SEQS:
                    if is_already_running(tracker):
                        c.warn(f"0 state files — already running, waiting")
                    else:
                        log(f"  HEAL: launching passive eval {run_tag} for {tracker}")
                        run_bg([PYTHON, "-u", str(TOOLS / "run_with_csc.py"),
                                "--tracker", tracker, "--dataset", "uav123",
                                "--split", "test", "--csc_checkpoint", str(ckpt),
                                "--csc_mode", "passive", "--device", "cpu",
                                "--run_tag", run_tag,
                                "--output_dir", str(base)])
                        c.healed = True
                else:
                    c.warn(f"baseline missing — waiting")
        checks.append(c)


# ---------------------------------------------------------------------------
# Paper metrics checks
# ---------------------------------------------------------------------------

def check_paper_metrics(checks: list[Check], run_tag: str, ckpt: Path) -> None:
    for tracker in TRACKERS:
        c = Check(f"PaperMetrics {run_tag} — {tracker}")
        base = EVAL_DIR / tracker / "uav123/test"
        pm_path = base / run_tag / "paper_metrics/paper_metrics.json"
        pm = read_json(pm_path)

        if not pm_path.exists():
            # Check canonical and alt paths for states count
            states_n = count_files(base / run_tag / "states", "*.jsonl")
            if states_n == 0 and ckpt.exists():
                alt_run_tag = f"{tracker}_uav123_test_{ckpt.stem}"
                states_n = count_files(base / alt_run_tag / "states", "*.jsonl")
            if states_n == EXPECTED_SEQS:
                c.warn(f"states done but paper_metrics missing — will trigger on next check")
            else:
                c.warn(f"not ready (states={states_n})")
            checks.append(c)
            continue

        agg = pm.get("aggregate", pm)
        fcr = agg.get("FCR") or agg.get("fcr")
        auc_path = base / run_tag / "tracking_metrics/summary.json"
        tm = read_json(auc_path)
        auc = tm.get("auc") or tm.get("success_auc")
        gt_seqs = agg.get("n_sequences_with_gt", agg.get("n_gt_seqs", 0))

        issues = []
        if fcr is not None and float(fcr) == 0.0 and gt_seqs == 0:
            issues.append("FCR=0 AND gt_seqs=0 → GT labels not loaded")
        if auc is not None and float(auc) < 0.1:
            issues.append(f"AUC={auc:.3f} suspiciously low")
        if auc is None:
            issues.append("AUC missing")

        if issues:
            c.warn(f"FCR={fcr}, AUC={auc}, gt_seqs={gt_seqs} | ISSUES: {'; '.join(issues)}")
            # If GT labels missing: delete stale paper_metrics and rerun
            if "GT labels not loaded" in " ".join(issues):
                labels_inner = base / "labels_v3/uav123/test"
                if count_files(labels_inner / "labels_per_sequence", "*.jsonl") > 10:
                    log(f"  HEAL: deleting stale paper_metrics for {run_tag}/{tracker} and re-triggering")
                    pm_path.unlink(missing_ok=True)
                    (base / run_tag / "episode_metrics/episode_metrics.json").unlink(missing_ok=True)
                    c.healed = True
                    c.warn(f"deleted stale metrics, will recompute")
        else:
            c.ok(f"FCR={fcr:.4f}, AUC={auc:.3f}, gt_seqs={gt_seqs}")

        checks.append(c)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(ROOT / "logs", exist_ok=True)
    checks: list[Check] = []

    log("=== Status check ===")
    check_training(checks)
    check_baselines(checks)
    check_passive(checks, "passive_v3_f16", CKPT_TCN16_F16)
    check_passive(checks, "passive_v3_tcn32_f16", CKPT_TCN32_F16)
    check_paper_metrics(checks, "passive_v3_f16", CKPT_TCN16_F16)
    check_paper_metrics(checks, "passive_v3_tcn32_f16", CKPT_TCN32_F16)

    # Write STATUS.md
    lines = [f"# Pipeline Status — {NOW}\n"]
    lines.append("## Summary\n")
    done   = sum(1 for c in checks if c.status == "✅")
    failed = sum(1 for c in checks if c.status == "❌")
    warn   = sum(1 for c in checks if c.status == "⚠️")
    run_   = sum(1 for c in checks if c.status == "🔄")
    lines.append(f"✅ {done}  ❌ {failed}  ⚠️ {warn}  🔄 {run_}\n")
    lines.append("## Details\n")
    for c in checks:
        lines.append(str(c))
    STATUS_FILE.write_text("\n".join(lines) + "\n")

    # Append to health log
    record = {
        "ts": NOW,
        "done": done, "failed": failed, "warn": warn, "running": run_,
        "checks": [{"name": c.name, "status": c.status, "detail": c.detail,
                    "healed": c.healed} for c in checks]
    }
    with open(HEALTH_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    log(f"STATUS.md updated — ✅{done} ❌{failed} ⚠️{warn} 🔄{run_}")

    # Print failures
    for c in checks:
        if c.status in ("❌", "⚠️"):
            log(f"  {c}")


if __name__ == "__main__":
    main()
