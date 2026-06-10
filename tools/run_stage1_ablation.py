"""End-to-end Stage-1 ablation runner.

Sequence:
    1. Run bare SGLATrack baseline.
    2. Build weak scene-state labels from baseline + GT.
    3. Train CSC on the labels (sequence-level split).
    4. Evaluate baseline tracking metrics.
    5. Standalone CSC eval (causal, no GT at runtime).
    6. Run SGLATrack + CSC passive.
    7. Optionally run SGLATrack + CSC control.
    8. Evaluate CSC-equipped runs.
    9. Emit comparison_summary.csv / .md.

The script wraps the other tools/ via subprocess so each step is also
runnable independently.
"""
from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], log: logging.Logger) -> int:
    log.info("$ %s", shlex.join(cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        log.error("command failed: %s", " ".join(cmd))
    return proc.returncode


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _format_row(method: str, summary: dict, csc_summary: Optional[dict]) -> dict:
    macro = summary.get("macro", {})
    runtime = summary.get("runtime", {})
    out = {
        "method": method,
        "n_sequences": summary.get("n_sequences", 0),
        "n_frames": summary.get("n_frames", 0),
        "auc": macro.get("auc"),
        "precision_20": macro.get("precision_20"),
        "norm_precision_auc": macro.get("norm_precision_auc"),
        "ao": macro.get("ao"),
        "sr_50": macro.get("sr_50"),
        "sr_75": macro.get("sr_75"),
        "mean_fps": macro.get("fps") or runtime.get("mean_fps"),
        "p95_latency_ms": runtime.get("p95_ms"),
        "n_failures": summary.get("n_failures_total"),
        "total_failure_frames": summary.get("total_failure_frames"),
    }
    if csc_summary:
        out.update({
            "csc_macro_f1": csc_summary.get("macro_f1"),
            "csc_failure_auroc": csc_summary.get("failure_auroc"),
            "csc_failure_auprc": csc_summary.get("failure_auprc"),
            "csc_early_warning": csc_summary.get("early_warning_recall_k"),
            "csc_fa_per_1000": csc_summary.get("false_alarms_per_1000"),
        })
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 ablation runner.")
    p.add_argument("--dataset", required=True, choices=["lasot", "got10k", "uav123", "dtb70", "visdrone_sot"])
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_sequences", type=int, default=None)
    p.add_argument("--csc_config", default="configs/csc/csc_gru.yaml")
    p.add_argument("--label_config", default="configs/csc/labeling.yaml")
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--skip_labels", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_passive", action="store_true")
    p.add_argument("--run_control", action="store_true")
    p.add_argument("--out_root", default="outputs/stage1_ablation")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("stage1")
    args = parse_args()

    py = sys.executable
    ds, sp = args.dataset, args.split
    out_root = Path(args.out_root) / ds
    out_root.mkdir(parents=True, exist_ok=True)

    baseline_root = Path("outputs/baselines/sglatrack") / ds / sp
    labels_root = Path("outputs/csc_labels") / ds / sp
    csc_train_root = Path("outputs/csc_training/csc_gru") / ds
    passive_root = Path("outputs/csc_runs/sglatrack_csc_passive") / ds / sp
    control_root = Path("outputs/csc_runs/sglatrack_csc_control") / ds / sp

    # 1. Baseline ---------------------------------------------------
    if not args.skip_baseline:
        cmd = [
            py, "tools/run_sglatrack_baseline.py",
            "--dataset", ds, "--split", sp,
            "--device", args.device, "--seed", str(args.seed),
            "--output_dir", "outputs/baselines/sglatrack",
        ]
        if args.max_sequences:
            cmd += ["--max_sequences", str(args.max_sequences)]
        if _run(cmd, log) != 0:
            return 1

    # 2. Labels -----------------------------------------------------
    if not args.skip_labels:
        cmd = [
            py, "tools/build_scene_state_labels.py",
            "--dataset", ds, "--split", sp,
            "--baseline_dir", "outputs/baselines/sglatrack",
            "--output_dir", "outputs/csc_labels",
            "--threshold_config", args.label_config,
        ]
        if args.max_sequences:
            cmd += ["--max_sequences", str(args.max_sequences)]
        if _run(cmd, log) != 0:
            return 1

    # 3. Train CSC --------------------------------------------------
    if not args.skip_train:
        cmd = [
            py, "tools/train_csc.py",
            "--config", args.csc_config,
            "--labels_dir", str(labels_root),
            "--output_dir", str(csc_train_root),
            "--device", args.device, "--seed", str(args.seed),
        ]
        if _run(cmd, log) != 0:
            return 1

    csc_ckpt = csc_train_root / "checkpoint_best.pth"

    # 4. Evaluate baseline tracking metrics ------------------------
    eval_baseline_dir = out_root / "baseline_eval"
    cmd = [
        py, "tools/evaluate_tracking_results.py",
        "--dataset", ds, "--split", sp,
        "--pred_dir", str(baseline_root / "predictions"),
        "--telemetry_dir", str(baseline_root / "telemetry"),
        "--output_dir", str(eval_baseline_dir),
    ]
    if args.max_sequences:
        cmd += ["--max_sequences", str(args.max_sequences)]
    if _run(cmd, log) != 0:
        return 1

    # 5. Standalone CSC eval ---------------------------------------
    csc_eval_dir = out_root / "csc_standalone_eval"
    if csc_ckpt.exists():
        cmd = [
            py, "tools/evaluate_csc_standalone.py",
            "--checkpoint", str(csc_ckpt),
            "--labels_dir", str(labels_root),
            "--output_dir", str(csc_eval_dir),
            "--device", args.device,
        ]
        if _run(cmd, log) != 0:
            return 1

    # 6. SGLATrack + CSC passive -----------------------------------
    if not args.skip_passive and csc_ckpt.exists():
        cmd = [
            py, "tools/run_sglatrack_with_csc.py",
            "--dataset", ds, "--split", sp,
            "--csc_config", args.csc_config,
            "--csc_checkpoint", str(csc_ckpt),
            "--output_dir", "outputs/csc_runs/sglatrack_csc_passive",
            "--device", args.device, "--seed", str(args.seed),
            "--csc_mode", "passive",
        ]
        if args.max_sequences:
            cmd += ["--max_sequences", str(args.max_sequences)]
        if _run(cmd, log) != 0:
            return 1

        cmd = [
            py, "tools/evaluate_tracking_results.py",
            "--dataset", ds, "--split", sp,
            "--pred_dir", str(passive_root / "predictions"),
            "--telemetry_dir", str(passive_root / "telemetry"),
            "--output_dir", str(out_root / "csc_passive_eval"),
        ]
        if args.max_sequences:
            cmd += ["--max_sequences", str(args.max_sequences)]
        if _run(cmd, log) != 0:
            return 1

    # 7. SGLATrack + CSC control (optional) ------------------------
    if args.run_control and csc_ckpt.exists():
        cmd = [
            py, "tools/run_sglatrack_with_csc.py",
            "--dataset", ds, "--split", sp,
            "--csc_config", args.csc_config,
            "--csc_checkpoint", str(csc_ckpt),
            "--output_dir", "outputs/csc_runs/sglatrack_csc_control",
            "--device", args.device, "--seed", str(args.seed),
            "--csc_mode", "control",
        ]
        if args.max_sequences:
            cmd += ["--max_sequences", str(args.max_sequences)]
        if _run(cmd, log) != 0:
            return 1
        cmd = [
            py, "tools/evaluate_tracking_results.py",
            "--dataset", ds, "--split", sp,
            "--pred_dir", str(control_root / "predictions"),
            "--telemetry_dir", str(control_root / "telemetry"),
            "--output_dir", str(out_root / "csc_control_eval"),
        ]
        if args.max_sequences:
            cmd += ["--max_sequences", str(args.max_sequences)]
        if _run(cmd, log) != 0:
            return 1

    # 8. Comparison summary ----------------------------------------
    csc_summary = _read_summary(out_root / "csc_standalone_eval" / "csc_metrics_summary.json") or None
    rows = []
    rows.append(_format_row("baseline", _read_summary(out_root / "baseline_eval" / "metrics_summary.json"), csc_summary if False else None))
    if (out_root / "csc_passive_eval" / "metrics_summary.json").exists():
        rows.append(_format_row("csc_passive", _read_summary(out_root / "csc_passive_eval" / "metrics_summary.json"), csc_summary))
    if (out_root / "csc_control_eval" / "metrics_summary.json").exists():
        rows.append(_format_row("csc_control", _read_summary(out_root / "csc_control_eval" / "metrics_summary.json"), csc_summary))

    if rows:
        keys = list(rows[0].keys())
        with open(out_root / "comparison_summary.csv", "w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + "\n")
        # Markdown table
        with open(out_root / "comparison_summary.md", "w") as fh:
            fh.write("| " + " | ".join(keys) + " |\n")
            fh.write("|" + "|".join("---" for _ in keys) + "|\n")
            for r in rows:
                fh.write("| " + " | ".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + " |\n")
        log.info("comparison summary -> %s", out_root)
    else:
        log.warning("no rows to compare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
