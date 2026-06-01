#!/usr/bin/env python3
"""V3 Watchdog: runs every 10 min, monitors all V3 pipeline processes.

Starts master_orchestrator if not running.
Calls status_checker every 10 min for health + self-healing.
"""
from __future__ import annotations
import os, subprocess, sys, time, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
WDOG_LOG = ROOT / "logs/watchdog.log"
CHECK_INTERVAL = 600  # 10 minutes

ORCHESTRATOR_PID_FILE = ROOT / "logs/orchestrator.pid"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] WATCHDOG: {msg}"
    print(line, flush=True)
    with open(WDOG_LOG, "a") as f:
        f.write(line + "\n")


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def is_process_running(pattern: str) -> bool:
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True)
    return r.returncode == 0


def start_orchestrator() -> int:
    env = {**os.environ, "CSC_NOT_TRAINED_ON_UAV123": "1"}
    log_path = ROOT / f"logs/master_{time.strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_path, "a") as lf:
        p = subprocess.Popen(
            [PYTHON, "-u", str(ROOT / "tools/master_orchestrator.py")],
            stdout=lf, stderr=subprocess.STDOUT,
            env=env, start_new_session=True, cwd=str(ROOT),
        )
    ORCHESTRATOR_PID_FILE.write_text(str(p.pid))
    log(f"Started orchestrator pid={p.pid}, log={log_path.name}")
    return p.pid


def run_status_check() -> None:
    env = {**os.environ, "CSC_NOT_TRAINED_ON_UAV123": "1"}
    subprocess.run(
        [PYTHON, "-u", str(ROOT / "tools/status_checker.py")],
        env=env, cwd=str(ROOT),
    )


def log_training_progress() -> None:
    """Tail training logs to show current epoch/loss."""
    for tag, path in [
        ("TCN16-f16", ROOT / "outputs/csc_training/sglatrack_train2_v3_stage2_tcn16_f16/train_log.jsonl"),
        ("TCN32-s1",  ROOT / "outputs/csc_training/sglatrack_train2_v3_tcn32_f16_stage1/train_log.jsonl"),
        ("TCN32-f16", ROOT / "outputs/csc_training/sglatrack_train2_v3_tcn32_f16_stage2/train_log.jsonl"),
    ]:
        if not path.exists():
            continue
        try:
            lines = [json.loads(l) for l in open(path) if l.strip()]
            if not lines:
                continue
            last = lines[-1]
            ep = last.get("epoch", "?")
            f1 = last.get("derived_macro_f1") or last.get("macro_f1") or "?"
            log(f"  {tag}: ep={ep} macro_f1={f1}")
        except Exception:
            pass


def main() -> None:
    os.makedirs(ROOT / "logs", exist_ok=True)
    log("=== V3 Watchdog started (no-orchestrator mode) ===")

    # Run first status check immediately
    run_status_check()

    while True:
        time.sleep(CHECK_INTERVAL)
        log("=== 10-min check ===")
        log_training_progress()

        try:
            run_status_check()
        except Exception as e:
            log(f"status_checker error: {e}")

        log(f"Next check in {CHECK_INTERVAL // 60} min")


if __name__ == "__main__":
    main()
