#!/usr/bin/env python3
"""Full V3-Fix evaluation pipeline: passive eval + paper metrics for all trackers on UAV123.

Usage:
    # Run 1 (V1 features) eval:
    CSC_NOT_TRAINED_ON_UAV123=1 python -u tools/run_v3_eval_pipeline.py --run-tag r1

    # Run 2 (V2 features) eval:
    CSC_NOT_TRAINED_ON_UAV123=1 python -u tools/run_v3_eval_pipeline.py --run-tag r2

    # Override checkpoint path:
    python -u tools/run_v3_eval_pipeline.py --run-tag r1 --ckpt path/to/best.pth

Steps per tracker:
    1. Passive CSC run with V3-Fix stage2 checkpoint
    2. Build GT state labels (eval-only)
    3. Evaluate tracking results (AUC/Precision)
    4. Evaluate CSC episodes (Recall@K, FA, delay)
    5. Compute paper metrics (FCR, FCD, TTFC, Recovery@30, SC-AUC, STM)
       — with threshold-aware metrics (FC/LOST AUPRC + recall@FPR≤1%/3%)
       — with sentinel regression check (configs/csc/sentinels.yaml)

After all passive runs, runs SGLATrack control with proactive V3 forecast heads.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

CKPT_BY_RUN = {
    "r1":  ROOT / "outputs/csc_training/sglatrack_v3fix_tcn16_stage2/checkpoint_best.pth",
    "r2":  ROOT / "outputs/csc_training/sglatrack_run2_scalectx_tcn16_stage2/checkpoint_best.pth",
    "r25": ROOT / "outputs/csc_training/sglatrack_r25_fcw3_tcn16_stage2/checkpoint_best.pth",
}

CALIB_DIR = ROOT / "outputs/calibration"
BASELINES_DIR = ROOT / "outputs/baselines"
DEFAULT_SENTINELS = ROOT / "configs/csc/sentinels.yaml"

# Tracker → calibrator tag (verified against outputs/calibration/)
TRACKERS: dict[str, str] = {
    "sglatrack": "sglatrack_all_v2",
    "ortrack":   "ortrack_aerial_v2",
    "avtrack":   "avtrack_aerial_v2",
    "ostrack":   "ostrack_aerial_v2",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], **kw) -> None:
    log(f"RUN: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kw)


def find_latest_run_subdir(parent: Path, tracker: str) -> Path | None:
    """Find the most recently created run subdir matching tracker_uav123_test_*."""
    pattern = f"{tracker}_uav123_test_*"
    candidates = sorted(parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def run_passive(tracker: str, calib_tag: str, base: Path, ckpt: Path, run_tag: str) -> Path:
    """Run passive CSC eval, return path to passive_<run_tag>/ dir."""
    passive_dir = base / f"passive_{run_tag}"
    if (passive_dir / "metrics.json").exists():
        log(f"  SKIP passive_{run_tag} for {tracker} (metrics.json exists)")
        return passive_dir

    run([PYTHON, "-u", str(ROOT / "tools/run_with_csc.py"),
         "--tracker", tracker,
         "--dataset", "uav123",
         "--split", "test",
         "--csc_checkpoint", str(ckpt),
         "--csc_mode", "passive",
         "--calibration_prefix", calib_tag,
         "--device", "cpu",
         "--output_dir", str(base)])

    latest = find_latest_run_subdir(base, tracker)
    if latest is None:
        raise RuntimeError(f"Could not find run output subdir under {base} for {tracker}")
    log(f"  Renaming {latest.name} → passive_{run_tag}")
    shutil.move(str(latest), str(passive_dir))
    return passive_dir


def run_baseline_if_needed(tracker: str) -> None:
    """Generate baseline predictions+telemetry if not already present."""
    pred_dir = BASELINES_DIR / tracker / "uav123/test/predictions"
    if pred_dir.exists() and any(pred_dir.glob("*.txt")):
        log(f"  SKIP baseline for {tracker} (predictions exist)")
        return
    log(f"  Generating baseline for {tracker} ...")
    run([PYTHON, "-u", str(ROOT / "tools/run_baseline.py"),
         "--tracker", tracker,
         "--dataset", "uav123",
         "--split", "test",
         "--device", "cpu",
         "--skip_existing",
         "--output_dir", str(BASELINES_DIR)])


def run_build_labels(tracker: str, calib_tag: str, base: Path) -> Path:
    labels_v3 = base / "labels_v3"
    seq_labels = labels_v3 / "uav123/test/labels_per_sequence"
    if seq_labels.exists() and any(seq_labels.glob("*.jsonl")):
        log(f"  SKIP labels_v3 for {tracker}")
        return labels_v3

    run([PYTHON, "-u", str(ROOT / "tools/build_scene_state_labels.py"),
         "--tracker", tracker,
         "--dataset", "uav123",
         "--split", "test",
         "--baseline_dir", str(BASELINES_DIR / tracker),
         "--calibration_dir", str(CALIB_DIR),
         "--calibrator_tag", calib_tag,
         "--output_dir", str(labels_v3)])
    return labels_v3


def run_tracking_metrics(tracker: str, passive_v3: Path) -> Path:
    track_metrics = passive_v3 / "tracking_metrics"
    if (track_metrics / "summary.json").exists():
        log(f"  SKIP tracking_metrics for {tracker}")
        return track_metrics

    run([PYTHON, "-u", str(ROOT / "tools/evaluate_tracking_results.py"),
         "--dataset", "uav123",
         "--split", "test",
         "--pred_dir", str(passive_v3 / "predictions"),
         "--output_dir", str(track_metrics)])
    return track_metrics


def run_episode_metrics(tracker: str, passive_v3: Path, labels_v3: Path) -> Path:
    ep_metrics = passive_v3 / "episode_metrics"
    if (ep_metrics / "episode_metrics.json").exists():
        log(f"  SKIP episode_metrics for {tracker}")
        return ep_metrics

    # build_scene_state_labels nests output under <dataset>/<split>
    labels_inner = labels_v3 / "uav123/test"
    run([PYTHON, "-u", str(ROOT / "tools/evaluate_csc_episodes.py"),
         "--labels", str(labels_inner),
         "--predictions", str(passive_v3 / "states"),
         "--out", str(ep_metrics)])
    return ep_metrics


def run_paper_metrics(tracker: str, calib_tag: str, passive_dir: Path,
                      labels_v3: Path, track_metrics: Path,
                      sentinels: Path | None) -> Path:
    paper = passive_dir / "paper_metrics"
    if (paper / "paper_metrics.json").exists():
        log(f"  SKIP paper_metrics for {tracker}")
        return paper

    # build_scene_state_labels nests output under <dataset>/<split>
    labels_inner = labels_v3 / "uav123/test"
    cmd = [PYTHON, "-u", str(ROOT / "tools/compute_paper_metrics.py"),
         "--tracker", tracker,
         "--dataset", "uav123",
         "--split", "test",
         "--predictions_dir", str(passive_dir / "predictions"),
         "--states_dir", str(passive_dir / "states"),
         "--labels_dir", str(labels_inner),
         "--tracking_metrics_dir", str(track_metrics),
         "--confidence_calib", str(CALIB_DIR / f"{calib_tag}_confidence.json"),
         "--output_dir", str(paper),
         "--recovery_k", "30"]
    if sentinels and sentinels.exists():
        cmd.extend(["--sentinels", str(sentinels)])
    run(cmd)
    return paper


def run_control_proactive(base: Path, ckpt: Path, run_tag: str, calib_tag: str) -> Path:
    """SGLATrack only: proactive control with V3 forecast heads."""
    ctrl = base / f"control_{run_tag}_proactive"
    if (ctrl / "metrics.json").exists():
        log(f"  SKIP control_{run_tag}_proactive (metrics.json exists)")
        return ctrl

    run([PYTHON, "-u", str(ROOT / "tools/run_with_csc.py"),
         "--tracker", "sglatrack",
         "--dataset", "uav123",
         "--split", "test",
         "--csc_checkpoint", str(ckpt),
         "--csc_mode", "control",
         "--exit_router",
         "--proactive_v3",
         "--proactive_threshold", "0.7",
         "--calibration_prefix", calib_tag,
         "--device", "cpu",
         "--output_dir", str(base)])

    latest = find_latest_run_subdir(base, "sglatrack")
    if latest is None:
        raise RuntimeError(f"Could not find control run output subdir under {base}")
    log(f"  Renaming {latest.name} → control_{run_tag}_proactive")
    shutil.move(str(latest), str(ctrl))
    return ctrl


def main() -> None:
    parser = argparse.ArgumentParser(description="V3-Fix eval pipeline")
    parser.add_argument("--run-tag", choices=["r1", "r2", "r25"], required=True,
                        help="r1 = V1 features (csc_tcn16_v3fix), r2 = V2 features (run2_scalectx), "
                             "r25 = V2 + FCw=3.0 (r25_fcw3)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Override checkpoint path (defaults from CKPT_BY_RUN[run_tag])")
    parser.add_argument("--sentinels", type=str, default=str(DEFAULT_SENTINELS),
                        help="Path to sentinels YAML (default: configs/csc/sentinels.yaml)")
    parser.add_argument("--skip-control", action="store_true",
                        help="Skip SGLATrack proactive control run")
    args = parser.parse_args()

    run_tag = args.run_tag
    ckpt = Path(args.ckpt) if args.ckpt else CKPT_BY_RUN[run_tag]
    sentinels = Path(args.sentinels) if args.sentinels else None

    assert ckpt.exists(), f"Checkpoint not found: {ckpt}"
    if sentinels and not sentinels.exists():
        log(f"  WARN: sentinels file not found, skipping: {sentinels}")
        sentinels = None

    log(f"=== V3-Fix Eval Pipeline | run_tag={run_tag} | ckpt={ckpt.name} ===")
    os.makedirs(ROOT / "logs", exist_ok=True)

    for tracker, calib_tag in TRACKERS.items():
        log(f"\n{'='*60}")
        log(f"TRACKER: {tracker}  calibrator: {calib_tag}")
        log(f"{'='*60}")

        base = ROOT / "outputs/eval_v3fix" / tracker / "uav123/test"
        base.mkdir(parents=True, exist_ok=True)

        try:
            run_baseline_if_needed(tracker)
            passive_dir = run_passive(tracker, calib_tag, base, ckpt, run_tag)
            labels_v3 = run_build_labels(tracker, calib_tag, base)
            track_metrics = run_tracking_metrics(tracker, passive_dir)
            run_episode_metrics(tracker, passive_dir, labels_v3)
            run_paper_metrics(tracker, calib_tag, passive_dir, labels_v3, track_metrics, sentinels)
        except subprocess.CalledProcessError as e:
            log(f"ERROR on {tracker}: {e}")
            log("Continuing with next tracker...")
            continue

    if args.skip_control:
        log("\nSkipping proactive control run (--skip-control)")
    else:
        # SGLATrack proactive control run
        log(f"\n{'='*60}")
        log(f"SGLATrack proactive control run ({run_tag} forecast heads)")
        log(f"{'='*60}")
        sgla_base = ROOT / "outputs/eval_v3fix/sglatrack/uav123/test"
        try:
            ctrl = run_control_proactive(sgla_base, ckpt, run_tag, TRACKERS["sglatrack"])
            labels_v3 = sgla_base / "labels_v3"
            if labels_v3.exists():
                run_paper_metrics("sglatrack", TRACKERS["sglatrack"], ctrl, labels_v3,
                                  ctrl / "tracking_metrics", sentinels)
        except subprocess.CalledProcessError as e:
            log(f"ERROR on sglatrack control: {e}")

    log(f"\n[DONE] V3-Fix eval pipeline ({run_tag}) complete!")
    log(f"Results in: outputs/eval_v3fix/<tracker>/uav123/test/passive_{run_tag}/paper_metrics/")


if __name__ == "__main__":
    main()
