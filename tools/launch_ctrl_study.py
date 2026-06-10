"""Launch parallel control-mode study on worst-5 UAV123 sequences.

Runs 9 variants × 2 CSC models (V1, V2) = 18 jobs in parallel.
Each job gets its own --output_dir so results never collide.

Usage:
    python tools/launch_ctrl_study.py [--dry_run] [--device mps]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CKPTS = {
    "v1": str(ROOT / "outputs/csc_training/sglatrack_lasot_tcn16/checkpoint_best.pth"),
    "v2": str(ROOT / "outputs/csc_training/sglatrack_train2_v2_tcn16/checkpoint_best.pth"),
}

WORST_SEQS = ["person10", "car7", "uav1_2", "person14_1", "car6_5"]

# Each variant: (name, extra_args_list)
VARIANTS = [
    ("passive",       ["--csc_mode", "passive"]),
    ("ctrl_B",        ["--csc_mode", "control"]),
    ("ctrl_A",        ["--exit_router"]),
    ("ctrl_C",        ["--csc_mode", "control", "--csc_advisor"]),           # fixed: needs control mode
    ("ctrl_AC",       ["--csc_mode", "control", "--exit_router", "--csc_advisor"]),  # fixed
    ("fix_B_fc_only", ["--csc_mode", "control", "--policy_freeze_fc_only"]),
    ("fix_A_gated",   ["--exit_router", "--policy_tau_fc", "0.75"]),
    ("fix_B_streak",  ["--csc_mode", "control", "--policy_fc_streak", "3"]),
    ("fix_C_fc_only", ["--csc_mode", "control", "--csc_advisor", "--policy_freeze_fc_only"]),
]


def build_cmd(model: str, variant: str, extra: list[str], device: str) -> list[str]:
    out_dir = str(ROOT / "outputs/experiments/ctrl_study" / model / variant)
    return [
        sys.executable, str(ROOT / "tools/run_with_csc.py"),
        "--tracker",        "sglatrack",
        "--dataset",        "uav123",
        "--split",          "test",
        "--csc_checkpoint", CKPTS[model],
        "--device",         device,
        "--output_dir",     out_dir,
        "--include_sequences", *WORST_SEQS,
        *extra,
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run",  action="store_true")
    ap.add_argument("--device",   default="mps")
    ap.add_argument("--models",   nargs="+", default=["v1", "v2"], choices=["v1", "v2"])
    ap.add_argument("--variants", nargs="+", default=None,
                    help="Subset of variant names to run (default: all)")
    args = ap.parse_args()

    log_dir = ROOT / "logs/ctrl_study"
    log_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/experiments/ctrl_study").mkdir(parents=True, exist_ok=True)

    variants = [(n, e) for n, e in VARIANTS
                if args.variants is None or n in args.variants]

    jobs: list[tuple[str, subprocess.Popen, Path]] = []
    for model in args.models:
        for vname, extra in variants:
            tag = f"{model}_{vname}"
            cmd = build_cmd(model, vname, extra, args.device)
            log_path = log_dir / f"{tag}.log"
            if args.dry_run:
                print(f"[DRY] {tag}:")
                print("  " + " ".join(cmd))
                continue
            log_f = open(log_path, "w")
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            jobs.append((tag, proc, log_path))
            print(f"[{tag}] PID={proc.pid}  log={log_path.relative_to(ROOT)}")

    if args.dry_run or not jobs:
        return

    print(f"\nLaunched {len(jobs)} jobs. Waiting…\n")
    t0 = time.time()
    failed = []
    while jobs:
        still_running = []
        for tag, proc, log_path in jobs:
            rc = proc.poll()
            if rc is None:
                still_running.append((tag, proc, log_path))
            else:
                elapsed = time.time() - t0
                status = "✓" if rc == 0 else f"✗ (rc={rc})"
                print(f"  {status}  {tag}  ({elapsed:.0f}s)  log={log_path.relative_to(ROOT)}")
                if rc != 0:
                    failed.append(tag)
        jobs = still_running
        if jobs:
            time.sleep(10)

    print(f"\nDone in {time.time()-t0:.0f}s. Failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
