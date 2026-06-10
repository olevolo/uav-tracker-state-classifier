#!/usr/bin/env python
"""CSC unified CLI dispatcher.

Usage
-----
    python -u tools/csc.py <subcommand> [args...]
    python -u tools/csc.py --help

Each subcommand is a thin pass-through to the corresponding tool script in
tools/.  No tool logic is duplicated here — this is purely an orchestration
entry-point.

The ``pipeline`` subcommand is the exception: it is implemented here because
it coordinates *multiple* tool invocations, each as its own subprocess.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "tools"
_PY = str(_REPO / ".venv" / "bin" / "python")


# ---------------------------------------------------------------------------
# Subcommand registry
# ---------------------------------------------------------------------------

# Each entry: (name, script_path_relative_to_REPO, one_line_description)
_SUBCOMMANDS: list[tuple[str, str, str]] = [
    ("baseline",          "tools/run_baseline.py",                "Run any tracker on a dataset, save predictions + telemetry"),
    ("with-csc",          "tools/run_with_csc.py",                "Run any tracker with CSC inference (passive or control mode)"),
    ("labels",            "tools/build_scene_state_labels.py",    "Generate weak scene-state labels from baseline predictions + GT"),
    ("calibrate",         "tools/fit_calibration.py",             "Fit per-tracker percentile calibrators from telemetry JSONL"),
    ("train",             "tools/train_csc.py",                   "Train the CSC model (GRU / MLP / TCN16 / TCN32)"),
    ("eval-csc",          "tools/evaluate_csc_standalone.py",     "Standalone CSC eval — localization + confidence + derived metrics"),
    ("eval-tracker",      "tools/evaluate_tracking_results.py",   "Tracker AUC / Precision against GT annotations"),
    ("profile",           "tools/profile_pipeline.py",            "GFLOPs + per-stage latency profiler for tracker + CSC pipeline"),
    ("diagnose-features", "tools/diagnose_csc_features.py",       "Per-feature AUROC / AUPRC / Pearson-r usefulness analysis"),
    ("audit-labels",      "tools/audit_label_distribution.py",    "Label distribution auditor — per-state counts and health check"),
    ("audit-viz",         "tools/audit_visualizer.py",            "Manual audit grid PNG: GT vs predicted bboxes per frame"),
    ("gate",              "tools/check_stage1_gate.py",           "Stage-1 gate checker — 9 criteria from CSC.md §2"),
    ("verify-trackers",   "tools/_verify_no_stubs.py",            "Verify all 5 trackers load real weights (one subprocess each)"),
    ("pipeline",          "__builtin__",                          "Full end-to-end pipeline: baseline→calibrate→labels→train→eval→profile→gate"),
]

_SCRIPT_MAP: dict[str, str] = {name: script for name, script, _ in _SUBCOMMANDS}
_DESC_MAP:   dict[str, str] = {name: desc  for name, _, desc  in _SUBCOMMANDS}


# ---------------------------------------------------------------------------
# Forwarding helper
# ---------------------------------------------------------------------------

def _forward(script_rel: str, rest: list[str]) -> None:
    """Replace the current process with: .venv/bin/python -u <script> [rest...]."""
    script = str(_REPO / script_rel)
    cmd = [_PY, "-u", script] + rest
    # execvp replaces the current process — no return on success.
    # Fall back to subprocess on Windows where execvp isn't available.
    if hasattr(os, "execvp"):
        os.execvp(cmd[0], cmd)
    else:
        sys.exit(subprocess.run(cmd, cwd=str(_REPO)).returncode)


# ---------------------------------------------------------------------------
# Top-level help
# ---------------------------------------------------------------------------

def _print_summary() -> None:
    print("CSC evaluation framework — single-entry CLI dispatcher")
    print()
    print("Usage:")
    print("  python -u tools/csc.py <subcommand> [args...]")
    print("  python -u tools/csc.py <subcommand> -h    # wrapped tool's own help")
    print()
    print("Subcommands:")
    width = max(len(n) for n, _, _ in _SUBCOMMANDS)
    for name, _, desc in _SUBCOMMANDS:
        print(f"  {name:<{width + 2}} {desc}")
    print()
    print("Notes:")
    print("  - One tracker per Python process (tracker libs conflict in sys.path).")
    print("  - Use 'pipeline' for the full end-to-end run (baseline→gate).")
    print("  - See RUNNING.md for recipes and expected output paths.")


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def _run_step(
    step_num: int,
    label: str,
    cmd: list[str],
    *,
    dry_run: bool = False,
) -> None:
    """Print + run one pipeline step.  Exit the process on failure."""
    display = " ".join(cmd)
    print(f"\n[step {step_num}] {label}", flush=True)
    print(f"  $ {display}", flush=True)
    if dry_run:
        print("  (DRY RUN — skipped)", flush=True)
        return
    result = subprocess.run(cmd, cwd=str(_REPO))
    if result.returncode != 0:
        print(
            f"\n[pipeline] FAILED at step {step_num} ({label}) — exit {result.returncode}",
            flush=True,
        )
        sys.exit(result.returncode)


def _pipeline_state_path(run_dir: Path) -> Path:
    return run_dir / ".pipeline_state.json"


def _load_pipeline_state(run_dir: Path) -> dict:
    p = _pipeline_state_path(run_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _save_pipeline_state(run_dir: Path, state: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _pipeline_state_path(run_dir).write_text(json.dumps(state, indent=2))


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_fresh(state: dict, step_key: str, dependency_path: Optional[Path] = None) -> bool:
    """Return True if this step was completed more recently than dependency_path's mtime."""
    ts_str = state.get(step_key)
    if not ts_str:
        return False
    if dependency_path is not None and dependency_path.exists():
        dep_mtime = dependency_path.stat().st_mtime
        try:
            from datetime import timezone as _tz
            import dateutil.parser  # type: ignore[import]
            step_ts = dateutil.parser.parse(ts_str).timestamp()
        except Exception:
            # dateutil not available — compare by naive iso string
            return False
        if dep_mtime > step_ts:
            return False
    return True


def _run_pipeline(args: argparse.Namespace) -> None:
    tracker: str = args.tracker
    dataset: str = args.dataset
    split: str = args.split
    csc_config: str = args.csc_config
    max_seq: Optional[int] = args.max_sequences
    dry_run: bool = getattr(args, "dry_run", False)

    csc_config_stem = Path(csc_config).stem  # e.g. csc_tcn16

    # Output directories
    baseline_dir = _REPO / "outputs" / "baselines" / tracker / dataset / split
    manifest_path = baseline_dir / "manifest.json"
    calibration_dir = _REPO / "outputs" / "calibration"
    calibration_json = calibration_dir / f"{tracker}_{dataset}_confidence.json"
    labels_root = _REPO / "outputs" / "csc_labels" / dataset / split
    train_out = _REPO / "outputs" / "csc_training" / f"{tracker}_{dataset}_{csc_config_stem}"
    val_metrics = train_out / "val_metrics.json"
    best_ckpt = train_out / "checkpoint_best.pth"
    profile_out = _REPO / "outputs" / "profile" / f"{tracker}_{dataset}_{csc_config_stem}.json"

    # Pipeline state
    run_dir = _REPO / "outputs" / f"_pipeline_{tracker}_{dataset}_{split}_{csc_config_stem}"
    state = _load_pipeline_state(run_dir)

    print(f"\n{'='*70}", flush=True)
    print(f"CSC PIPELINE: tracker={tracker} dataset={dataset} split={split}", flush=True)
    print(f"  csc_config   : {csc_config}", flush=True)
    print(f"  max_sequences: {max_seq}", flush=True)
    print(f"  run_dir      : {run_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    step = 0

    # ------------------------------------------------------------------
    # Step 1: verify-trackers (subset for this tracker)
    # ------------------------------------------------------------------
    step += 1
    if not _is_fresh(state, "verify"):
        cmd = [_PY, "-u", str(_TOOLS / "_verify_no_stubs.py")]
        _run_step(step, f"verify tracker {tracker}", cmd, dry_run=dry_run)
        state["verify"] = _ts()
        _save_pipeline_state(run_dir, state)
    else:
        print(f"\n[step {step}] verify — fresh (skipping)", flush=True)

    # ------------------------------------------------------------------
    # Step 2: baseline
    # ------------------------------------------------------------------
    step += 1
    # Check adapter file freshness
    adapter_file = _REPO / "src" / "uav_tracker" / "trackers" / f"{tracker}.py"
    need_baseline = not manifest_path.exists() or not _is_fresh(
        state, "baseline", adapter_file if adapter_file.exists() else None
    )
    if need_baseline:
        cmd = [_PY, "-u", str(_TOOLS / "run_baseline.py"),
               "--tracker", tracker,
               "--dataset", dataset,
               "--split", split]
        if max_seq:
            cmd += ["--max_sequences", str(max_seq)]
        _run_step(step, f"baseline: {tracker}/{dataset}/{split}", cmd, dry_run=dry_run)
        state["baseline"] = _ts()
        _save_pipeline_state(run_dir, state)
    else:
        print(f"\n[step {step}] baseline — fresh manifest found (skipping)", flush=True)

    # ------------------------------------------------------------------
    # Step 3: calibrate
    # ------------------------------------------------------------------
    step += 1
    if not calibration_json.exists() or not _is_fresh(state, "calibrate"):
        telemetry_dir = baseline_dir / "telemetry"
        cmd = [_PY, "-u", str(_TOOLS / "fit_calibration.py"),
               "--tracker", tracker,
               "--telemetry_dir", str(telemetry_dir),
               "--output_dir", str(calibration_dir)]
        _run_step(step, f"calibrate: {tracker}/{dataset}", cmd, dry_run=dry_run)
        state["calibrate"] = _ts()
        _save_pipeline_state(run_dir, state)
    else:
        print(f"\n[step {step}] calibrate — existing calibrator found (skipping)", flush=True)

    # ------------------------------------------------------------------
    # Step 4: labels
    # ------------------------------------------------------------------
    step += 1
    labels_jsonl = labels_root / "labels.jsonl"
    if not labels_jsonl.exists() or not _is_fresh(state, "labels"):
        cmd = [_PY, "-u", str(_TOOLS / "build_scene_state_labels.py"),
               "--dataset", dataset,
               "--split", split,
               "--baseline_dir", str(_REPO / "outputs" / "baselines" / tracker),
               "--tracker", tracker,
               "--calibration_dir", str(calibration_dir)]
        if max_seq:
            cmd += ["--max_sequences", str(max_seq)]
        _run_step(step, f"labels: {dataset}/{split}", cmd, dry_run=dry_run)
        state["labels"] = _ts()
        _save_pipeline_state(run_dir, state)
    else:
        print(f"\n[step {step}] labels — fresh labels.jsonl found (skipping)", flush=True)

    # ------------------------------------------------------------------
    # Step 5: train
    # ------------------------------------------------------------------
    step += 1
    if not best_ckpt.exists() or not _is_fresh(state, "train"):
        cmd = [_PY, "-u", str(_TOOLS / "train_csc.py"),
               "--config", csc_config,
               "--labels_dir", str(labels_root),
               "--output_dir", str(train_out)]
        _run_step(step, f"train: {csc_config_stem} on {dataset}/{split}", cmd, dry_run=dry_run)
        state["train"] = _ts()
        _save_pipeline_state(run_dir, state)
    else:
        print(f"\n[step {step}] train — fresh checkpoint found (skipping)", flush=True)

    # ------------------------------------------------------------------
    # Step 6: eval-csc
    # ------------------------------------------------------------------
    step += 1
    if not val_metrics.exists() or not _is_fresh(state, "eval_csc"):
        cmd = [_PY, "-u", str(_TOOLS / "evaluate_csc_standalone.py"),
               "--labels_dir", str(labels_root),
               "--checkpoint", str(best_ckpt),
               "--output", str(val_metrics)]
        _run_step(step, "eval-csc: standalone metrics", cmd, dry_run=dry_run)
        state["eval_csc"] = _ts()
        _save_pipeline_state(run_dir, state)
    else:
        print(f"\n[step {step}] eval-csc — fresh val_metrics.json found (skipping)", flush=True)

    # ------------------------------------------------------------------
    # Step 7: profile
    # ------------------------------------------------------------------
    step += 1
    if not profile_out.exists() or not _is_fresh(state, "profile"):
        profile_out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [_PY, "-u", str(_TOOLS / "profile_pipeline.py"),
               "--tracker", tracker,
               "--dataset", dataset,
               "--split", split,
               "--csc_checkpoint", str(best_ckpt),
               "--output", str(profile_out)]
        if max_seq:
            cmd += ["--max_sequences", str(max_seq)]
        _run_step(step, f"profile: {tracker} + CSC", cmd, dry_run=dry_run)
        state["profile"] = _ts()
        _save_pipeline_state(run_dir, state)
    else:
        print(f"\n[step {step}] profile — fresh profile JSON found (skipping)", flush=True)

    # ------------------------------------------------------------------
    # Step 8: gate
    # ------------------------------------------------------------------
    step += 1
    cmd = [_PY, "-u", str(_TOOLS / "check_stage1_gate.py"),
           "--metrics_json", str(val_metrics)]
    _run_step(step, "gate: Stage-1 criteria check", cmd, dry_run=dry_run)
    state["gate"] = _ts()
    _save_pipeline_state(run_dir, state)

    print(f"\n{'='*70}", flush=True)
    print("CSC PIPELINE COMPLETE", flush=True)
    print(f"  predictions : outputs/baselines/{tracker}/{dataset}/{split}/predictions/", flush=True)
    print(f"  telemetry   : outputs/baselines/{tracker}/{dataset}/{split}/telemetry/", flush=True)
    print(f"  labels      : outputs/csc_labels/{dataset}/{split}/", flush=True)
    print(f"  calibrators : outputs/calibration/", flush=True)
    print(f"  train       : {train_out}/", flush=True)
    print(f"  profile     : {profile_out}", flush=True)
    print(f"  gate report : {train_out}/gate_report.json", flush=True)
    print(f"{'='*70}\n", flush=True)


# ---------------------------------------------------------------------------
# Pipeline subparser
# ---------------------------------------------------------------------------

def _add_pipeline_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "pipeline",
        help=_DESC_MAP["pipeline"],
        description=_DESC_MAP["pipeline"],
        add_help=True,
    )
    p.add_argument("--tracker", required=True,
                   choices=["sglatrack", "ostrack", "ortrack", "avtrack", "evptrack"])
    p.add_argument("--dataset", required=True,
                   choices=["lasot", "got10k", "uav123", "dtb70", "visdrone_sot"])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--csc-config", dest="csc_config",
                   default="configs/csc/csc_tcn16.yaml",
                   help="Path to CSC YAML config (default: configs/csc/csc_tcn16.yaml)")
    p.add_argument("--max-sequences", dest="max_sequences", type=int, default=None,
                   help="Cap the number of sequences (smoke / debug)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="Print commands without executing them")
    p.set_defaults(func=_run_pipeline)


# ---------------------------------------------------------------------------
# Generic forwarding subparser
# ---------------------------------------------------------------------------

def _make_forward_parser(name: str, script_rel: str, desc: str, subparsers) -> None:
    p = subparsers.add_parser(
        name,
        help=desc,
        description=desc,
        # Don't let argparse consume -h — forward it to the wrapped tool.
        add_help=False,
    )
    p.set_defaults(_script=script_rel)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    # No args → print summary
    if not argv:
        _print_summary()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        prog="csc",
        description="CSC evaluation framework — unified CLI dispatcher",
        add_help=True,
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # Register forwarding subparsers for every non-builtin subcommand
    for name, script_rel, desc in _SUBCOMMANDS:
        if name == "pipeline":
            _add_pipeline_parser(subparsers)
        else:
            _make_forward_parser(name, script_rel, desc, subparsers)

    # Peek at the first token
    first = argv[0]
    if first in ("-h", "--help"):
        _print_summary()
        sys.exit(0)

    known, rest = parser.parse_known_args(argv)

    if known.subcommand is None:
        _print_summary()
        sys.exit(0)

    if known.subcommand == "pipeline":
        # Parse the full pipeline args (rest already consumed by subparser above)
        pipeline_args, extra = parser.parse_known_args(argv)
        if extra:
            print(f"warning: unrecognised pipeline args: {extra}", file=sys.stderr)
        pipeline_args.func(pipeline_args)
        sys.exit(0)

    # All other subcommands: forward remaining argv unchanged to wrapped tool
    script_rel = getattr(known, "_script", None)
    if script_rel is None:
        parser.print_help()
        sys.exit(1)

    # rest = everything after the subcommand token
    _forward(script_rel, rest)


if __name__ == "__main__":
    main()
