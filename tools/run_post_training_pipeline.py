#!/usr/bin/env python3
"""Autonomous post-training pipeline: compare CSC models → final UAV123 eval → report.

Auto-triggered by tools/monitor_v3fix.py after all 5 training stages are DONE
(R1_S1, R1_S2, R2_S1, R2_S2, R15_S1).

Pipeline:
  1. Run tools/compare_csc_models.py — rank runs on held-out val (NEVER UAV123).
  2. Run tools/run_v3_eval_pipeline.py --run-tag r1
       (passive on sglatrack/ortrack/avtrack/ostrack + sglatrack proactive control).
  3. Run tools/run_v3_eval_pipeline.py --run-tag r2
       (same suite for V2 features ckpt).
  4. Write FINAL_REPORT.md aggregating paper metrics (FCR/FCD/Recovery@30/SC-AUC)
     for both run-tags + tracker AUC + control mode delta.

Sentinels (logs/v3fix_full/):
  post_pipeline.running   — blocks re-fire while running (cron-safe)
  post_pipeline.done      — blocks re-fire after success
  FINAL_REPORT.md         — human summary (this is what the user reads)

CLAUDE.md compliance:
  - UAV123 used ONLY for final evaluation (passive + control). No training, no tuning.
  - Numbers reported are MEASURED (no fabrication).
  - Control mode evaluated on SGLATrack only (per project paper-strategy memory).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "v3fix_full"
PYTHON = sys.executable

RUNNING = LOG_DIR / "post_pipeline.running"
DONE = LOG_DIR / "post_pipeline.done"
REPORT = ROOT / "FINAL_REPORT.md"
COMPARE_OUT = LOG_DIR / "compare_models.txt"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], log_path: Path) -> int:
    log(f"RUN: {' '.join(cmd)} > {log_path}")
    with log_path.open("a") as f:
        f.write(f"\n\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} === {' '.join(cmd)}\n\n")
        f.flush()
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT)).returncode
    log(f"  rc={rc}")
    return rc


def step_compare() -> str:
    """Run compare_csc_models.py, return stdout text."""
    log("Step 1: compare CSC models on held-out val")
    text = subprocess.check_output(
        [PYTHON, str(ROOT / "tools/compare_csc_models.py")],
        cwd=str(ROOT), text=True, stderr=subprocess.STDOUT,
    )
    COMPARE_OUT.write_text(text)
    log(f"  → {COMPARE_OUT}")
    return text


def step_eval(run_tag: str) -> int:
    """Run full V3-Fix eval pipeline for one run-tag (passive 4 trackers + sgla control)."""
    log(f"Step 2/3: full eval pipeline for {run_tag}")
    env = os.environ.copy()
    env["CSC_NOT_TRAINED_ON_UAV123"] = "1"
    log_path = LOG_DIR / f"eval_pipeline_{run_tag}.log"
    cmd = [PYTHON, "-u", str(ROOT / "tools/run_v3_eval_pipeline.py"),
           "--run-tag", run_tag]
    log(f"RUN: {' '.join(cmd)} > {log_path}")
    with log_path.open("a") as f:
        f.write(f"\n\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} === {' '.join(cmd)}\n\n")
        f.flush()
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                            cwd=str(ROOT)).returncode
    log(f"  rc={rc} for {run_tag}")
    return rc


def _safe_load(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _paper_metrics_path(tracker: str, run_tag: str, kind: str = "passive") -> Path:
    """outputs/eval_v3fix/<tracker>/uav123/test/<passive|control_<rt>_proactive>/paper_metrics/paper_metrics.json"""
    base = ROOT / "outputs/eval_v3fix" / tracker / "uav123/test"
    if kind == "passive":
        return base / f"passive_{run_tag}/paper_metrics/paper_metrics.json"
    return base / f"control_{run_tag}_proactive/paper_metrics/paper_metrics.json"


def _track_metrics_path(tracker: str, run_tag: str, kind: str = "passive") -> Path:
    base = ROOT / "outputs/eval_v3fix" / tracker / "uav123/test"
    sub = f"passive_{run_tag}" if kind == "passive" else f"control_{run_tag}_proactive"
    return base / sub / "tracking_metrics/summary.json"


def step_report(compare_text: str, ran_tags: list[str]) -> None:
    """Aggregate paper metrics + tracking metrics into FINAL_REPORT.md."""
    log("Step 4: write FINAL_REPORT.md")
    trackers = ["sglatrack", "ortrack", "avtrack", "ostrack"]

    lines: list[str] = []
    lines.append("# V3-Fix Final Report — UAV123 Test")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\nMeasured numbers — no fabrication. UAV123 used only for final evaluation.\n")

    lines.append("## Held-out val ranking (compare_csc_models.py)\n")
    lines.append("```")
    lines.append(compare_text.strip())
    lines.append("```\n")

    for run_tag in ran_tags:
        lines.append(f"## Run tag: `{run_tag}` — Paper Metrics on UAV123\n")
        lines.append("| Tracker | AUC | Prec@20 | FCR | FCD | TTFC | Recovery@30 | SC-AUC(FC) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for tk in trackers:
            pm = _safe_load(_paper_metrics_path(tk, run_tag, "passive"))
            tm = _safe_load(_track_metrics_path(tk, run_tag, "passive"))
            def g(d, *keys, default="—"):
                cur = d
                for k in keys:
                    if not isinstance(cur, dict) or k not in cur:
                        return default
                    cur = cur[k]
                if isinstance(cur, (int, float)):
                    return f"{cur:.4f}" if isinstance(cur, float) else str(cur)
                return str(cur) if cur is not None else default
            auc = g(tm, "auc")
            p20 = g(tm, "precision_at_20")
            fcr = g(pm, "fcr")
            fcd = g(pm, "fcd")
            ttfc = g(pm, "ttfc")
            rk = g(pm, "recovery_at_30")
            scauc = g(pm, "state_conditioned_auc", "false_confirmed")
            lines.append(f"| {tk} | {auc} | {p20} | {fcr} | {fcd} | {ttfc} | {rk} | {scauc} |")
        lines.append("")

        # Control mode delta on SGLATrack
        ctrl_pm = _safe_load(_paper_metrics_path("sglatrack", run_tag, "control"))
        passive_pm = _safe_load(_paper_metrics_path("sglatrack", run_tag, "passive"))
        if ctrl_pm and passive_pm:
            lines.append(f"### SGLATrack Control Mode (proactive_v3, threshold=0.7) vs Passive — `{run_tag}`\n")
            lines.append("| Metric | Passive | Control | Δ |")
            lines.append("|---|---|---|---|")
            for k, label in [("fcr", "FCR"), ("fcd", "FCD"), ("ttfc", "TTFC"),
                             ("recovery_at_30", "Recovery@30")]:
                p = passive_pm.get(k, None)
                c = ctrl_pm.get(k, None)
                if isinstance(p, (int, float)) and isinstance(c, (int, float)):
                    delta = c - p
                    pct = f" ({delta/p*100:+.1f}%)" if p != 0 else ""
                    lines.append(f"| {label} | {p:.4f} | {c:.4f} | {delta:+.4f}{pct} |")
                else:
                    lines.append(f"| {label} | {p} | {c} | — |")
            lines.append("")

    lines.append("## Source\n")
    lines.append("- Per-tracker paper metrics: `outputs/eval_v3fix/<tracker>/uav123/test/passive_<run>/paper_metrics/`")
    lines.append("- SGLATrack control: `outputs/eval_v3fix/sglatrack/uav123/test/control_<run>_proactive/`")
    lines.append("- Held-out val ranking: `logs/v3fix_full/compare_models.txt`")
    lines.append("- Pipeline logs: `logs/v3fix_full/eval_pipeline_<run>.log`")
    lines.append("- Training memory: `project_csc_feature_dispatch_bugs.md`, `csc-cosine-lr-fix.md`\n")

    REPORT.write_text("\n".join(lines))
    log(f"  → {REPORT}")


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if DONE.exists():
        log("Already done — exiting.")
        return 0
    if RUNNING.exists():
        # Stale-check via PID file: if PID dead, allow restart.
        pid_str = RUNNING.read_text().strip()
        if pid_str.isdigit():
            try:
                os.kill(int(pid_str), 0)  # signal 0 = check liveness
                log(f"Already running (pid={pid_str}) — exiting.")
                return 0
            except (ProcessLookupError, PermissionError):
                log(f"Stale lock (pid={pid_str} dead) — restarting.")
                RUNNING.unlink()
    RUNNING.write_text(str(os.getpid()))

    try:
        compare_text = step_compare()

        ran_tags: list[str] = []
        ckpt_paths = {
            "r1": ROOT / "outputs/csc_training/sglatrack_v3fix_tcn16_stage2/checkpoint_best.pth",
            "r2": ROOT / "outputs/csc_training/sglatrack_run2_scalectx_tcn16_stage2/checkpoint_best.pth",
            "r25": ROOT / "outputs/csc_training/sglatrack_r25_fcw3_tcn16_stage2/checkpoint_best.pth",
        }
        for tag in ("r1", "r2", "r25"):
            ckpt = ckpt_paths[tag]
            if not ckpt.exists():
                log(f"WARN: {ckpt} missing — skipping {tag} eval")
                continue
            rc = step_eval(tag)
            if rc == 0:
                ran_tags.append(tag)
            else:
                log(f"  {tag} eval returned non-zero rc={rc} (continuing — partial results may exist)")
                ran_tags.append(tag)  # report partial anyway

        step_report(compare_text, ran_tags)
        DONE.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        log(f"PIPELINE DONE — see {REPORT}")
        return 0
    except Exception as e:
        log(f"FATAL: {type(e).__name__}: {e}")
        return 1
    finally:
        if RUNNING.exists():
            RUNNING.unlink()


if __name__ == "__main__":
    sys.exit(main())
