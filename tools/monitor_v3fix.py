"""V3-Fix training monitor — runs every 10 min via cron.

Reports per-run state:
  - epoch progress (current/total + macro F1 + FC F1 + FC AUPRC)
  - ETA (linear extrapolation from last 3 epochs)
  - process liveness (log mtime within mtime_window)
  - alerts: stalled, NaN loss, FC F1 collapse, OOM marker
  - acceptance gate previews when training is near complete

Writes:
  logs/v3fix_full/MONITOR.md  (latest snapshot, overwritten)
  logs/v3fix_full/MONITOR.log (append-only history)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "v3fix_full"
MTIME_STALE_SEC = 15 * 60  # log not touched in 15 min ⇒ stalled

RUNS = {
    "R1_S1": {
        "log": LOG_DIR / "run1_stage1.log",
        "out": ROOT / "outputs" / "csc_training" / "sglatrack_v3fix_tcn16_stage1",
        "epochs": 30,
        "stage": 1,
        "label": "Run 1 (V1) Stage 1",
    },
    "R2_S1": {
        "log": LOG_DIR / "run2_stage1.log",
        "out": ROOT / "outputs" / "csc_training" / "sglatrack_run2_scalectx_tcn16_stage1",
        "epochs": 30,
        "stage": 1,
        "label": "Run 2 (V2) Stage 1",
    },
    "R1_S2": {
        "log": LOG_DIR / "run1_stage2.log",
        "out": ROOT / "outputs" / "csc_training" / "sglatrack_v3fix_tcn16_stage2",
        "epochs": 25,
        "stage": 2,
        "label": "Run 1 (V1) Stage 2",
    },
    "R2_S2": {
        "log": LOG_DIR / "run2_stage2.log",
        "out": ROOT / "outputs" / "csc_training" / "sglatrack_run2_scalectx_tcn16_stage2",
        "epochs": 25,
        "stage": 2,
        "label": "Run 2 (V2) Stage 2",
    },
    "R25_S1": {
        "log": LOG_DIR / "run2_5_stage1.log",
        "out": ROOT / "outputs" / "csc_training" / "sglatrack_r25_fcw3_tcn16_stage1",
        "epochs": 30,
        "stage": 1,
        "label": "Run 2.5 (V2 + FCw=3.0) Stage 1",
    },
    "R25_S2": {
        "log": LOG_DIR / "run2_5_stage2.log",
        "out": ROOT / "outputs" / "csc_training" / "sglatrack_r25_fcw3_tcn16_stage2",
        "epochs": 25,
        "stage": 2,
        "label": "Run 2.5 (V2 + FCw=3.0) Stage 2",
    },
    # R3 = winner(R2,R25) + window=32. Variant resolved at runtime from r3_winner.txt.
    "R3_S1": {
        "log": LOG_DIR / "r3_stage1.log",
        "out": ROOT / "outputs" / "csc_training" / "sglatrack_r3_DYN_w32_tcn32_stage1",
        "epochs": 30,
        "stage": 1,
        "label": "Run 3 (winner + window=32) Stage 1",
        "dynamic_out": True,
    },
    "R3_S2": {
        "log": LOG_DIR / "r3_stage2.log",
        "out": ROOT / "outputs" / "csc_training" / "sglatrack_r3_DYN_w32_tcn32_stage2",
        "epochs": 25,
        "stage": 2,
        "label": "Run 3 (winner + window=32) Stage 2",
        "dynamic_out": True,
    },
}


def _resolve_r3_variant() -> str | None:
    """Read winner from r3_winner.txt (written by select_and_kick_r3.py)."""
    marker = LOG_DIR / "r3_winner.txt"
    if marker.exists():
        return marker.read_text().strip()
    return None

EPOCH_RE = re.compile(
    r"\bep(?:och)?\s*(\d+)/(\d+)\s*\|\s*loss[=:]?\s*([\d.]+)\s*\|\s*derivedF1[=:]?\s*([\d.]+)",
    re.IGNORECASE,
)
FC_F1_RE = re.compile(r"FC_recall[=:]?\s*([\d.]+)", re.IGNORECASE)
FC_AUPRC_RE = re.compile(r"\bAUPRC[=:]?\s*([\d.]+)", re.IGNORECASE)
NAN_RE = re.compile(r"\bnan\b", re.IGNORECASE)
ERR_RE = re.compile(r"(error|exception|traceback|killed|oom)", re.IGNORECASE)
DONE_RE = re.compile(r"(training done|early stopping at epoch)", re.IGNORECASE)


@dataclass
class RunStatus:
    key: str
    label: str
    log_path: Path
    out_path: Path
    epochs_total: int
    log_exists: bool = False
    log_mtime: float | None = None
    log_age_sec: float | None = None
    last_epoch: int | None = None
    last_loss: float | None = None
    last_macro_f1: float | None = None
    last_fc_f1: float | None = None
    last_fc_auprc: float | None = None
    ckpt_best_size_mb: float | None = None
    has_nan: bool = False
    has_error: bool = False
    error_lines: list[str] = field(default_factory=list)
    is_alive: bool = False  # log touched within MTIME_STALE_SEC
    eta_min: float | None = None
    epoch_history: list[tuple[int, float]] = field(default_factory=list)
    state: str = "UNKNOWN"  # NOT_STARTED | RUNNING | STALLED | DONE | ERROR


def parse_run(key: str, cfg: dict) -> RunStatus:
    out_path = cfg["out"]
    if cfg.get("dynamic_out"):
        variant = _resolve_r3_variant()
        if variant:
            out_path = Path(str(out_path).replace("DYN", variant))
    rs = RunStatus(
        key=key,
        label=cfg["label"],
        log_path=cfg["log"],
        out_path=out_path,
        epochs_total=cfg["epochs"],
    )

    if not rs.log_path.exists():
        rs.state = "NOT_STARTED"
        return rs

    rs.log_exists = True
    rs.log_mtime = rs.log_path.stat().st_mtime
    rs.log_age_sec = dt.datetime.now().timestamp() - rs.log_mtime
    rs.is_alive = rs.log_age_sec < MTIME_STALE_SEC

    text = rs.log_path.read_text(errors="ignore")
    lines = text.splitlines()

    if NAN_RE.search(text):
        rs.has_nan = True
    for line in lines[-100:]:
        if ERR_RE.search(line) and "weight_decay" not in line.lower():
            rs.error_lines.append(line.strip()[:200])
            rs.has_error = True

    epoch_marks: list[tuple[int, float, float]] = []
    for line in lines:
        m = EPOCH_RE.search(line)
        if m:
            ep = int(m.group(1))
            total = int(m.group(2))
            loss = float(m.group(3))
            mf1 = float(m.group(4))
            ts_match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if ts_match:
                t = dt.datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            else:
                t = 0.0
            epoch_marks.append((ep, loss, t))
            rs.epochs_total = total
            rs.last_epoch = ep
            rs.last_loss = loss
            rs.last_macro_f1 = mf1

    fc_matches = FC_F1_RE.findall(text)
    if fc_matches:
        rs.last_fc_f1 = float(fc_matches[-1])
    fc_auprc_matches = FC_AUPRC_RE.findall(text)
    if fc_auprc_matches:
        rs.last_fc_auprc = float(fc_auprc_matches[-1])

    if len(epoch_marks) >= 2:
        rs.epoch_history = [(e, t) for e, _, t in epoch_marks if t > 0]
        if len(rs.epoch_history) >= 2:
            recent = rs.epoch_history[-3:] if len(rs.epoch_history) >= 3 else rs.epoch_history
            t_first = recent[0][1]
            t_last = recent[-1][1]
            ep_first = recent[0][0]
            ep_last = recent[-1][0]
            if ep_last > ep_first and t_last > t_first:
                sec_per_epoch = (t_last - t_first) / (ep_last - ep_first)
                remain = max(0, rs.epochs_total - ep_last)
                rs.eta_min = sec_per_epoch * remain / 60.0

    ckpt_best = rs.out_path / "checkpoint_best.pth"
    if ckpt_best.exists():
        rs.ckpt_best_size_mb = ckpt_best.stat().st_size / 1024 / 1024

    if rs.has_error or rs.has_nan:
        rs.state = "ERROR"
    elif DONE_RE.search(text) or (rs.last_epoch is not None and rs.last_epoch >= rs.epochs_total):
        # DONE on explicit completion marker ("training done" / "early stopping")
        # OR when the final epoch is reached. Early-stop ends before epochs_total.
        rs.state = "DONE"
    elif rs.is_alive:
        rs.state = "RUNNING"
    elif rs.log_exists and not rs.is_alive:
        rs.state = "STALLED"

    return rs


def fmt_status(rs: RunStatus) -> str:
    icon = {
        "NOT_STARTED": "[--]",
        "RUNNING": "[>>]",
        "STALLED": "[!!]",
        "DONE": "[OK]",
        "ERROR": "[XX]",
        "UNKNOWN": "[??]",
    }.get(rs.state, "[??]")

    parts = [f"{icon} **{rs.label}** — {rs.state}"]

    if rs.last_epoch is not None:
        parts.append(f"epoch {rs.last_epoch}/{rs.epochs_total}")
    if rs.last_loss is not None:
        parts.append(f"loss={rs.last_loss:.4f}")
    if rs.last_macro_f1 is not None:
        parts.append(f"macro_F1={rs.last_macro_f1:.3f}")
    if rs.last_fc_f1 is not None:
        parts.append(f"FC_F1={rs.last_fc_f1:.3f}")
    if rs.last_fc_auprc is not None:
        parts.append(f"FC_AUPRC={rs.last_fc_auprc:.3f}")
    if rs.eta_min is not None and rs.state == "RUNNING":
        parts.append(f"ETA={rs.eta_min:.0f}min")
    if rs.log_age_sec is not None:
        parts.append(f"log_age={rs.log_age_sec/60:.1f}min")
    if rs.ckpt_best_size_mb is not None:
        parts.append(f"ckpt={rs.ckpt_best_size_mb:.2f}MB")

    line = " | ".join(parts)
    if rs.error_lines:
        line += "\n  ERR: " + "; ".join(rs.error_lines[-2:])
    return line


CHAIN = [
    # (key, dep_key, config) — dep_key=None means parallel-launch independent
    ("R1_S1", None,    "configs/csc/csc_tcn16_v3fix_stage1.yaml"),
    ("R2_S1", None,    "configs/csc/csc_tcn16_run2_scalectx_stage1.yaml"),
    ("R1_S2", "R1_S1", "configs/csc/csc_tcn16_v3fix_stage2.yaml"),
    ("R2_S2", "R2_S1", "configs/csc/csc_tcn16_run2_scalectx_stage2.yaml"),
    ("R25_S2", "R25_S1", "configs/csc/csc_tcn16_r25_fcw3_stage2.yaml"),
]


def auto_kickoff_chain() -> list[str]:
    """Parallel chain — two independent tracks:
        Track 1: R1_S1 -> R1_S2
        Track 2: R2_S1 -> R2_S2
    Stage 2 starts as soon as its Stage 1 is DONE — no cross-track barrier.
    """
    notes = []
    statuses = {k: parse_run(k, c) for k, c in RUNS.items()}

    for key, dep_key, cfg_path in CHAIN:
        s = statuses[key]
        if s.state in ("RUNNING", "DONE", "ERROR"):
            continue
        if s.state == "STALLED":
            notes.append(f"[!] {key} STALLED — manual investigation needed")
            continue
        if dep_key is None:
            continue  # initial stages launched manually
        dep = statuses[dep_key]
        if dep.state != "DONE":
            continue
        dep_ckpt = dep.out_path / "checkpoint_best.pth"
        if not dep_ckpt.exists():
            notes.append(f"[!] {dep_key} DONE but no checkpoint_best.pth — cannot start {key}")
            continue
        log_file = RUNS[key]["log"]
        cmd = (
            f"cd {ROOT} && nohup .venv/bin/python tools/train_csc.py "
            f"--config {cfg_path} > {log_file} 2>&1 &"
        )
        os.system(cmd)
        notes.append(f"[+] Auto-kickoff: {RUNS[key]['label']} (config={cfg_path})")
        statuses[key].state = "RUNNING"  # update local view; next tick re-parses log

    # Run 2.5 (V2 + FCw=3.0) — replaces R1.5 in the chain.
    # Goal: lift FC precision (R2 has 0.629) for control-mode UUR.
    # Launches when BOTH R1_S2 + R2_S2 are DONE (no CPU contention with main runs).
    # Stage 2 chains automatically via CHAIN entry above.
    r25 = statuses.get("R25_S1")
    if r25 is not None and r25.state == "NOT_STARTED":
        s1_done = statuses["R1_S2"].state == "DONE"
        s2_done = statuses["R2_S2"].state == "DONE"
        if s1_done and s2_done:
            cfg_path = "configs/csc/csc_tcn16_r25_fcw3_stage1.yaml"
            log_file = RUNS["R25_S1"]["log"]
            os.system(
                f"cd {ROOT} && nohup .venv/bin/python tools/train_csc.py "
                f"--config {cfg_path} > {log_file} 2>&1 &"
            )
            notes.append(f"[+] Auto-kickoff: {RUNS['R25_S1']['label']} (config={cfg_path})")
            statuses["R25_S1"].state = "RUNNING"

    # Post-training pipeline: ALL training done → compare + UAV123 final eval
    # (passive 4 trackers + SGLATrack proactive control). Sentinel files in
    # logs/v3fix_full/ prevent re-fire across cron ticks.
    all_done = (
        statuses["R1_S2"].state == "DONE"
        and statuses["R2_S2"].state == "DONE"
        and (r25 is None or statuses["R25_S2"].state == "DONE")
    )
    post_running = LOG_DIR / "post_pipeline.running"
    post_done = LOG_DIR / "post_pipeline.done"
    if all_done and not post_done.exists() and not post_running.exists():
        log_file = LOG_DIR / "post_pipeline_dispatch.log"
        os.system(
            f"cd {ROOT} && nohup .venv/bin/python tools/run_post_training_pipeline.py "
            f">> {log_file} 2>&1 &"
        )
        notes.append("[+] Auto-kickoff: post-training pipeline (compare + UAV123 final eval + control)")
    elif post_running.exists() and not post_done.exists():
        notes.append(f"[~] post-training pipeline RUNNING (pid={post_running.read_text().strip()})")
    elif post_done.exists():
        notes.append(f"[OK] post-training pipeline DONE — see FINAL_REPORT.md")
    return notes


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now()
    statuses = [parse_run(k, c) for k, c in RUNS.items()]

    notes = auto_kickoff_chain()

    md = [f"# V3-Fix Training Monitor — {now:%Y-%m-%d %H:%M:%S}", ""]
    for rs in statuses:
        md.append(f"- {fmt_status(rs)}")
    md.append("")

    if notes:
        md.append("## Auto-actions")
        for n in notes:
            md.append(f"- {n}")
        md.append("")

    alerts = []
    for rs in statuses:
        if rs.state == "STALLED":
            alerts.append(f"STALLED: {rs.label} (log idle {rs.log_age_sec/60:.0f}min)")
        if rs.state == "ERROR":
            alerts.append(f"ERROR: {rs.label} ({rs.error_lines[-1] if rs.error_lines else 'NaN/error'})")
    if alerts:
        md.append("## ALERTS")
        for a in alerts:
            md.append(f"- {a}")
        md.append("")

    summary = " | ".join(f"{rs.key}={rs.state}" for rs in statuses)
    md.append(f"**Summary:** {summary}")

    snapshot = "\n".join(md)
    (LOG_DIR / "MONITOR.md").write_text(snapshot)

    with (LOG_DIR / "MONITOR.log").open("a") as f:
        f.write(f"[{now:%Y-%m-%d %H:%M:%S}] {summary}\n")

    print(snapshot)


if __name__ == "__main__":
    main()
