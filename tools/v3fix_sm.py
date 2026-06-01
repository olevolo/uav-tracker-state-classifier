"""V3-fix autonomous pipeline state machine.

Run by cron every 10 min:
    python3 tools/v3fix_sm.py

Reads STATUS_V3FIX_FULL.md (current state), takes action for that state,
writes back. For long-running ops (training/eval), launches in background
via nohup; subsequent ticks check PID and advance when complete.

For state == 'init' (code work), exits with code 1 to signal Claude needs
to dispatch parallel agents. All other states are deterministic shell ops.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path("/Users/voleksiuk/projects/uav-tracker-detector")
STATUS = REPO / "STATUS_V3FIX_FULL.md"

# State definitions (ordered)
STATES = [
    "init",                # T1-T5: code changes (Claude required)
    "smoke_pending",       # T6-T9: smoke tests + V1 cohort distribution check (Claude required)
    "smoke_ok",            # smoke + V1 cohort gates passed → ready for run1
    "train_r1_s1",         # waiting for run1 stage1
    "r1_s1_done",          # ready to launch run1 s2
    "train_r1_s2",         # waiting for run1 stage2
    "r1_s2_done",          # ready to eval run1
    "eval_r1_running",     # waiting for run1 eval
    "eval_r1_done",        # ready to gate (FCR<10%)
    "r2_ready",            # ready to launch run2 s1
    "train_r2_s1",
    "r2_s1_done",
    "train_r2_s2",
    "r2_s2_done",
    "eval_r2_running",
    "eval_r2_done",        # Claude: gate (FCR ∈ [1,5]% AND recall AND F1)
    "v4_stress_running",   # Level 4 synthetic stress test (auto, after Run 2 gate pass)
    "v4_stress_done",      # all 3 scenarios pass → final_eval_ready; else manual
    "final_eval_ready",
    "final_eval_running",
    "final_eval_done",
    "done",
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_status() -> dict:
    """Parse STATUS file. Returns dict with state, pid, log."""
    if not STATUS.exists():
        return {"state": "init", "pid": None, "raw": ""}
    content = STATUS.read_text()
    m = re.search(r"^## Current State:\s*(\S+)", content, re.MULTILINE)
    state = m.group(1) if m else "init"
    m2 = re.search(r"^PID:\s*(\d+)", content, re.MULTILINE)
    pid = int(m2.group(1)) if m2 else None
    return {"state": state, "pid": pid, "raw": content}


def write_status(state: str, pid: int | None = None, log_line: str = "", error: str = ""):
    """Update STATUS file with new state, PID, append log line."""
    existing = STATUS.read_text() if STATUS.exists() else ""
    # Extract existing log section
    log_match = re.search(r"## Action Log\n(.*?)(?=\n## |\Z)", existing, re.DOTALL)
    existing_log = log_match.group(1).strip() if log_match else ""
    err_match = re.search(r"## Errors\n(.*?)(?=\n## |\Z)", existing, re.DOTALL)
    existing_err = err_match.group(1).strip() if err_match else ""

    new_log = existing_log
    if log_line:
        new_log = (existing_log + f"\n- [{now()}] {state}: {log_line}").strip()
    new_err = existing_err
    if error:
        new_err = (existing_err + f"\n- [{now()}] {state}: {error}").strip()

    pid_line = f"PID: {pid}" if pid else "PID: -"
    content = f"""# V3-Fix Full Pipeline Status

## Current State: {state}
Last updated: {now()}
{pid_line}

## Action Log
{new_log}

## Errors
{new_err}
"""
    STATUS.write_text(content)


def is_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def launch_bg(cmd: list[str], log_path: str) -> int:
    """Launch command in background via nohup, return PID."""
    log = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        cmd, cwd=REPO, stdout=log, stderr=log, start_new_session=True,
    )
    return proc.pid


def file_exists(p: str) -> bool:
    return Path(p).exists() if Path(p).is_absolute() else (REPO / p).exists()


def handle_state(s: dict) -> int:
    """Take action for current state. Returns exit code (0=ok, 1=needs Claude)."""
    state = s["state"]
    pid = s["pid"]

    # ----- States requiring Claude (code work / smoke tests) -----
    if state == "init":
        write_status("init", log_line="State 'init' requires Claude to do code work (T1-T5).")
        print(f"STATE: {state} | NEEDS_CLAUDE: code changes per playbook")
        return 1

    if state == "smoke_pending":
        write_status("smoke_pending", log_line="State 'smoke_pending' requires Claude: T6-T9 smoke tests + V1 cohort distribution check.")
        print(f"STATE: {state} | NEEDS_CLAUDE: run smoke tests + V1 cohort check")
        return 1

    # ----- Deterministic states -----
    if state == "smoke_ok":
        # Launch Run 1 stage 1
        cmd = ["python3", "-u", "tools/train_csc.py",
               "--config", "configs/csc/csc_tcn16_run1_hardneg_stage1.yaml"]
        log_path = "/tmp/v3fix_run1_s1.log"
        new_pid = launch_bg(cmd, log_path)
        write_status("train_r1_s1", pid=new_pid,
                     log_line=f"Launched Run 1 Stage 1 (PID={new_pid}, log={log_path})")
        print(f"STATE: train_r1_s1 | PID: {new_pid}")
        return 0

    if state == "train_r1_s1":
        if pid and is_alive(pid):
            tail = subprocess.run(["tail", "-3", "/tmp/v3fix_run1_s1.log"],
                                   capture_output=True, text=True).stdout.strip()
            write_status("train_r1_s1", pid=pid, log_line=f"PID alive, tail: {tail[:200]}")
            print(f"STATE: train_r1_s1 | PID {pid} alive | waiting")
            return 0
        # Process dead — check for checkpoint
        ckpt = "outputs/csc_training/sglatrack_run1_hardneg_tcn16_stage1/checkpoint_best.pth"
        if file_exists(ckpt):
            write_status("r1_s1_done", log_line="Stage 1 done, checkpoint exists.")
            print("STATE: r1_s1_done")
            return 0
        write_status("r1_s1_failed", error="Stage1 process died without checkpoint. See /tmp/v3fix_run1_s1.log")
        print("STATE: r1_s1_failed | NEEDS_CLAUDE")
        return 1

    if state == "r1_s1_done":
        cmd = ["python3", "-u", "tools/train_csc.py",
               "--config", "configs/csc/csc_tcn16_run1_hardneg_stage2.yaml"]
        log_path = "/tmp/v3fix_run1_s2.log"
        new_pid = launch_bg(cmd, log_path)
        write_status("train_r1_s2", pid=new_pid,
                     log_line=f"Launched Run 1 Stage 2 (PID={new_pid})")
        print(f"STATE: train_r1_s2 | PID: {new_pid}")
        return 0

    if state == "train_r1_s2":
        if pid and is_alive(pid):
            tail = subprocess.run(["tail", "-3", "/tmp/v3fix_run1_s2.log"],
                                   capture_output=True, text=True).stdout.strip()
            write_status("train_r1_s2", pid=pid, log_line=f"PID alive, tail: {tail[:200]}")
            print("STATE: train_r1_s2 | waiting")
            return 0
        ckpt = "outputs/csc_training/sglatrack_run1_hardneg_tcn16_stage2/checkpoint_best.pth"
        if file_exists(ckpt):
            write_status("r1_s2_done", log_line="Run 1 Stage 2 done, checkpoint exists.")
            print("STATE: r1_s2_done")
            return 0
        write_status("r1_s2_failed", error="Stage2 process died without checkpoint.")
        return 1

    if state == "r1_s2_done":
        # Launch Run 1 UAV123 eval (sglatrack passive)
        cmd = ["python3", "-u", "tools/run_with_csc.py",
               "--tracker", "sglatrack",
               "--csc-checkpoint", "outputs/csc_training/sglatrack_run1_hardneg_tcn16_stage2/checkpoint_best.pth",
               "--dataset", "uav123",
               "--output", "outputs/v3fix_run1_eval"]
        log_path = "/tmp/v3fix_run1_eval.log"
        new_pid = launch_bg(cmd, log_path)
        write_status("eval_r1_running", pid=new_pid, log_line="Launched Run 1 UAV123 eval")
        print(f"STATE: eval_r1_running | PID: {new_pid}")
        return 0

    if state == "eval_r1_running":
        if pid and is_alive(pid):
            print("STATE: eval_r1_running | waiting")
            return 0
        write_status("eval_r1_done", log_line="Eval done, ready to compute FCR + gate")
        print("STATE: eval_r1_done | NEEDS_CLAUDE: compute FCR")
        return 1  # Claude computes FCR + gate

    if state == "eval_r1_done":
        # Claude has updated state to either r2_ready or r2_skip via STATUS file
        print("STATE: eval_r1_done | NEEDS_CLAUDE: gate decision")
        return 1

    if state == "r2_ready":
        cmd = ["python3", "-u", "tools/train_csc.py",
               "--config", "configs/csc/csc_tcn16_run2_scalectx_stage1.yaml"]
        log_path = "/tmp/v3fix_run2_s1.log"
        new_pid = launch_bg(cmd, log_path)
        write_status("train_r2_s1", pid=new_pid, log_line=f"Launched Run 2 Stage 1 (PID={new_pid})")
        return 0

    if state == "train_r2_s1":
        if pid and is_alive(pid):
            return 0
        ckpt = "outputs/csc_training/sglatrack_run2_scalectx_tcn16_stage1/checkpoint_best.pth"
        if file_exists(ckpt):
            write_status("r2_s1_done", log_line="Run 2 Stage 1 done")
            return 0
        write_status("r2_s1_failed", error="Stage1 died w/o ckpt")
        return 1

    if state == "r2_s1_done":
        cmd = ["python3", "-u", "tools/train_csc.py",
               "--config", "configs/csc/csc_tcn16_run2_scalectx_stage2.yaml"]
        log_path = "/tmp/v3fix_run2_s2.log"
        new_pid = launch_bg(cmd, log_path)
        write_status("train_r2_s2", pid=new_pid, log_line=f"Launched Run 2 Stage 2 (PID={new_pid})")
        return 0

    if state == "train_r2_s2":
        if pid and is_alive(pid):
            return 0
        ckpt = "outputs/csc_training/sglatrack_run2_scalectx_tcn16_stage2/checkpoint_best.pth"
        if file_exists(ckpt):
            write_status("r2_s2_done", log_line="Run 2 Stage 2 done")
            return 0
        write_status("r2_s2_failed", error="Stage2 died w/o ckpt")
        return 1

    if state == "r2_s2_done":
        cmd = ["python3", "-u", "tools/run_with_csc.py",
               "--tracker", "sglatrack",
               "--csc-checkpoint", "outputs/csc_training/sglatrack_run2_scalectx_tcn16_stage2/checkpoint_best.pth",
               "--dataset", "uav123",
               "--output", "outputs/v3fix_run2_eval"]
        new_pid = launch_bg(cmd, "/tmp/v3fix_run2_eval.log")
        write_status("eval_r2_running", pid=new_pid, log_line="Launched Run 2 UAV123 eval")
        return 0

    if state == "eval_r2_running":
        if pid and is_alive(pid):
            return 0
        write_status("eval_r2_done", log_line="Run 2 eval done")
        print("STATE: eval_r2_done | NEEDS_CLAUDE: Run 2 gate (FCR ∈ [1,5]% AND recall ≥ R1−5pp AND F1 ≥ R1−0.02)")
        return 1

    if state == "eval_r2_done":
        return 1

    # ----- V4 Synthetic stress test (auto, after Claude passes Run 2 gate) -----
    if state == "v4_stress_running":
        if pid and is_alive(pid):
            return 0
        out = "outputs/v3fix_diag/synthetic_stress.json"
        if file_exists(out):
            write_status("v4_stress_done", log_line="V4 stress test complete; check JSON for verdict")
            print("STATE: v4_stress_done | NEEDS_CLAUDE: review V4 verdicts (3/3 PASS = go to final_eval)")
            return 1
        write_status("v4_stress_failed", error="V4 stress test ran but no JSON output")
        return 1

    if state == "v4_stress_done":
        return 1  # Claude reviews verdict and sets final_eval_ready or manual

    # When Claude sets state to v4_stress_pending, this kicks off the script:
    if state == "v4_stress_pending":
        cmd = ["python3", "-u", "tools/synthetic_stress_test.py",
               "--ckpt", "outputs/csc_training/sglatrack_run2_scalectx_tcn16_stage2/checkpoint_best.pth",
               "--out", "outputs/v3fix_diag/synthetic_stress.json"]
        new_pid = launch_bg(cmd, "/tmp/v3fix_v4_stress.log")
        write_status("v4_stress_running", pid=new_pid, log_line="Launched V4 synthetic stress test")
        print(f"STATE: v4_stress_running | PID: {new_pid}")
        return 0

    if state == "final_eval_ready":
        # Run full pipeline (passive + control on sglatrack, passive on others)
        cmd = ["python3", "-u", "tools/v3fix_pipeline.py"]
        new_pid = launch_bg(cmd, "/tmp/v3fix_final_pipeline.log")
        write_status("final_eval_running", pid=new_pid, log_line="Launched final pipeline")
        return 0

    if state == "final_eval_running":
        if pid and is_alive(pid):
            return 0
        write_status("final_eval_done", log_line="Final pipeline done")
        return 1  # Claude writes report

    if state == "final_eval_done":
        return 1  # Claude writes FINAL_RESULTS_V3FIX_FULL.md

    if state == "done":
        print("STATE: done — pipeline complete, no-op")
        return 0

    # Failed states
    if state.endswith("_failed"):
        print(f"STATE: {state} | NEEDS_CLAUDE: manual intervention")
        return 1

    # Unknown state
    write_status(state, error=f"Unknown state: {state}")
    print(f"UNKNOWN STATE: {state}")
    return 1


def main():
    s = read_status()
    rc = handle_state(s)
    sys.exit(rc)


if __name__ == "__main__":
    main()
