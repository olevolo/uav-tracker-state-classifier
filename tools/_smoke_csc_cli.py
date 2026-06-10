#!/usr/bin/env python
"""Smoke test for tools/csc.py CLI dispatcher and Makefile targets.

Run via:
    perl -e 'alarm 240; exec @ARGV' .venv/bin/python -u tools/_smoke_csc_cli.py

Exit code: 0 if all checks pass, 1 on first failure.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PY = str(_REPO / ".venv" / "bin" / "python")

# Try to import torch for CUDA check
try:
    import torch
    print(f"CUDA= {torch.cuda.is_available()}", flush=True)
except ImportError:
    print("CUDA= (torch not importable in smoke runner)", flush=True)


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _run(label: str, cmd: list[str], *, check_stdout: list[str] | None = None) -> bool:
    print(f"\n[{_ts()}] CHECK: {label}", flush=True)
    print(f"  $ {' '.join(cmd)}", flush=True)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_REPO),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("  FAIL — timed out", flush=True)
        return False

    combined = result.stdout + result.stderr
    print(f"  exit={result.returncode}", flush=True)
    if result.stdout.strip():
        # Print up to 30 lines of stdout
        lines = result.stdout.strip().splitlines()
        for ln in lines[:30]:
            print(f"    {ln}", flush=True)
        if len(lines) > 30:
            print(f"    ... ({len(lines) - 30} more lines)", flush=True)

    if result.returncode != 0:
        if result.stderr.strip():
            for ln in result.stderr.strip().splitlines()[-10:]:
                print(f"  STDERR: {ln}", flush=True)
        print(f"  FAIL — non-zero exit", flush=True)
        return False

    if check_stdout:
        for needle in check_stdout:
            if needle not in combined:
                print(f"  FAIL — expected string not found in output: {needle!r}", flush=True)
                return False

    print("  PASS", flush=True)
    return True


def main() -> int:
    failures: list[str] = []

    # ------------------------------------------------------------------
    # 1. python -u tools/csc.py  →  exit 0, contains all subcommand names
    # ------------------------------------------------------------------
    label = "csc.py (no args) → summary help"
    expected_subcommands = [
        "baseline", "with-csc", "labels", "calibrate", "train",
        "eval-csc", "eval-tracker", "profile", "diagnose-features",
        "audit-labels", "audit-viz", "gate", "verify-trackers", "pipeline",
    ]
    ok = _run(label, [_PY, "-u", "tools/csc.py"],
              check_stdout=expected_subcommands)
    if not ok:
        failures.append(label)

    # ------------------------------------------------------------------
    # 2. python -u tools/csc.py baseline -h  →  wrapped help, exit 0
    # ------------------------------------------------------------------
    label = "csc.py baseline -h → wrapped tool help"
    ok = _run(label, [_PY, "-u", "tools/csc.py", "baseline", "-h"],
              check_stdout=["--tracker", "--dataset"])
    if not ok:
        failures.append(label)

    # ------------------------------------------------------------------
    # 3. python -u tools/csc.py calibrate -h  →  wrapped help, exit 0
    # ------------------------------------------------------------------
    label = "csc.py calibrate -h → wrapped tool help"
    ok = _run(label, [_PY, "-u", "tools/csc.py", "calibrate", "-h"],
              check_stdout=["--telemetry_dir"])
    if not ok:
        failures.append(label)

    # ------------------------------------------------------------------
    # 4. python -u tools/csc.py pipeline -h  →  built-in pipeline help
    # ------------------------------------------------------------------
    label = "csc.py pipeline -h → pipeline subparser help"
    ok = _run(label, [_PY, "-u", "tools/csc.py", "pipeline", "-h"],
              check_stdout=["--tracker", "--dataset", "--csc-config"])
    if not ok:
        failures.append(label)

    # ------------------------------------------------------------------
    # 5. make -n verify  →  prints command, exit 0
    # ------------------------------------------------------------------
    label = "make -n verify → echoes command"
    ok = _run(label, ["make", "-n", "verify"],
              check_stdout=["verify-trackers"])
    if not ok:
        failures.append(label)

    # ------------------------------------------------------------------
    # 6. make -n baseline TRACKER=sglatrack DATASET=got10k SPLIT=val
    #    →  echoes right command, exit 0
    # ------------------------------------------------------------------
    label = "make -n baseline TRACKER=sglatrack DATASET=got10k SPLIT=val"
    ok = _run(label,
              ["make", "-n", "baseline",
               "TRACKER=sglatrack", "DATASET=got10k", "SPLIT=val"],
              check_stdout=["sglatrack", "got10k", "val"])
    if not ok:
        failures.append(label)

    # ------------------------------------------------------------------
    # 7. make -n pipeline TRACKER=sglatrack DATASET=got10k
    #    →  echoes right command
    # ------------------------------------------------------------------
    label = "make -n pipeline TRACKER=sglatrack DATASET=got10k"
    ok = _run(label,
              ["make", "-n", "pipeline",
               "TRACKER=sglatrack", "DATASET=got10k", "SPLIT=val"],
              check_stdout=["pipeline", "sglatrack"])
    if not ok:
        failures.append(label)

    # ------------------------------------------------------------------
    # 8. make help  →  exit 0, contains all target names
    # ------------------------------------------------------------------
    label = "make help → lists all targets"
    ok = _run(label, ["make", "help"],
              check_stdout=["verify", "baseline", "pipeline", "all-trackers-on"])
    if not ok:
        failures.append(label)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}", flush=True)
    if failures:
        print(f"FAIL — {len(failures)} check(s) failed:", flush=True)
        for f in failures:
            print(f"  - {f}", flush=True)
        return 1
    else:
        print("PASS — all smoke checks passed", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
