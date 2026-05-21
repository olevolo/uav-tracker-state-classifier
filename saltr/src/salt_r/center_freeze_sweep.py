"""center_freeze_sweep.py — Sweep center-freeze hyperparameters on hard sequences.

Sweeps (pfc_threshold, freeze_max_frames, release_pfc) and finds the config
that maximises hard-subset AUC delta without hurting the full benchmark.

Usage
-----
    PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.center_freeze_sweep \\
        --advisory saltr/checkpoints/production/saltrd_best.pt \\
        --output saltr/results/center_freeze_sweep.json

Success gate (checked after sweep):
  - hard_subset mean AUC delta >= +0.03 for at least one config
  - full UAV123 mean AUC delta >= -0.005 (no big regression)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure source trees are on path
_REPO_ROOT = Path(__file__).parents[4]  # .../uav-tracker-detector
_SRC = _REPO_ROOT / "src"
_SALTR_SRC = _REPO_ROOT / "saltr" / "src"
for _p in (_SRC, _SALTR_SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------

SWEEP_CONFIG: dict[str, list] = {
    "pfc_threshold":     [0.55, 0.60, 0.65],
    "freeze_max_frames": [5, 10, 20],
    "release_pfc":       [0.30, 0.35, 0.40],
}

HARD_SEQUENCES = [
    "bike2", "uav2", "uav3", "uav4", "uav5", "uav6", "uav7", "uav8",
    "group2_1", "group3_2", "person14_1", "person19_3",
    "person1_s", "person7_1", "wakeboard5",
]

# Standard 6-sequence fast-bench set for regression check
STANDARD_SEQUENCES = ["car13", "uav2", "bike2", "car7", "building1", "truck1"]

MAX_FRAMES = 500


# ---------------------------------------------------------------------------
# Dataset / runner helpers
# ---------------------------------------------------------------------------

def _build_dataset(max_frames: int = MAX_FRAMES):
    from uav_tracker.datasets.uav123 import UAV123Dataset
    return UAV123Dataset(max_frames=max_frames)


def _compute_auc(preds, gt, n):
    from uav_tracker.metrics.success import compute_auc
    gta = np.array([[b.x, b.y, b.w, b.h] for b in gt[:n]], dtype=np.float64)
    pra = np.array([[b.x, b.y, b.w, b.h] for b in preds[:n]], dtype=np.float64)
    return float(compute_auc(gta, pra))


def _build_runner(advisory_ckpt: str, fc_block: float,
                  freeze_max_frames: int, freeze_release_pfc: float):
    """Build a fresh SALTRunner with center-freeze configured."""
    from uav_tracker.salt_runner import SALTRunner
    from salt_r.advisor import SALTRDAdvisor

    runner = SALTRunner.from_config(str(_REPO_ROOT / "configs" / "prod" / "salt.yaml"))
    advisor = SALTRDAdvisor(
        advisory_ckpt,
        device="cpu",
        fc_block=fc_block,
        freeze_max_frames=freeze_max_frames,
        freeze_release_pfc=freeze_release_pfc,
    )
    runner.tracker.set_salt_rd_advisor(advisor)
    return runner


def _run_sequence(runner, seq) -> tuple[float, int]:
    """Run SALTRunner on a sequence and return (auc, n_freeze_frames)."""
    _advisor = getattr(runner.tracker, '_salt_rd_advisor', None)
    if _advisor is not None:
        _advisor.reset()

    entries = list(runner.run(seq))
    preds = [e.bbox for e in entries]
    n = len(entries)

    auc = _compute_auc(preds, seq.ground_truth, n)

    # Count freeze frames from telemetry
    n_freeze = sum(1 for e in entries if e.aux.get("center_freeze_active", False))

    return auc, n_freeze


def _run_baseline(seq_map: dict, sequences: list[str]) -> dict[str, float]:
    """Run the baseline (no advisor) and return per-sequence AUC."""
    from uav_tracker.salt_runner import SALTRunner
    from collections import Counter

    runner = SALTRunner.from_config(str(_REPO_ROOT / "configs" / "prod" / "salt.yaml"))

    baseline_aucs: dict[str, float] = {}
    for sname in sequences:
        if sname not in seq_map:
            continue
        seq = seq_map[sname]
        if hasattr(runner.tracker, 'reset'):
            runner.tracker.reset()

        t0 = time.perf_counter()
        entries = list(runner.run(seq))
        elapsed = time.perf_counter() - t0

        preds = [e.bbox for e in entries]
        n = len(entries)
        auc = _compute_auc(preds, seq.ground_truth, n)
        baseline_aucs[sname] = auc
        print(f"    baseline  {sname:<16} AUC={auc:.3f}  ({elapsed:.1f}s)")

    return baseline_aucs


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    advisory_ckpt: str,
    output_path: str,
    max_frames: int = MAX_FRAMES,
    hard_only: bool = False,
) -> dict:
    """Run the full center-freeze parameter sweep.

    Returns the best config dict and writes results to output_path.
    """
    print(f"\n{'='*70}")
    print(f"  Center-Freeze Sweep — advisory={advisory_ckpt}")
    print(f"  hard_seqs={len(HARD_SEQUENCES)}  max_frames={max_frames}")
    print(f"{'='*70}")

    ds = _build_dataset(max_frames=max_frames)
    seq_map = {s.name: s for s in ds}

    # Determine which sequences are available
    hard_available = [s for s in HARD_SEQUENCES if s in seq_map]
    std_available = [s for s in STANDARD_SEQUENCES if s in seq_map]

    if not hard_available:
        print("ERROR: No hard sequences found in dataset.")
        return {}

    all_sweep_seqs = list(dict.fromkeys(hard_available + std_available))

    print(f"\n  Available: hard={len(hard_available)}/{len(HARD_SEQUENCES)}  "
          f"std={len(std_available)}/{len(STANDARD_SEQUENCES)}")

    # --- Baseline (no advisor) ---
    print("\n  Running baseline (no advisor)...")
    baseline_aucs = _run_baseline(seq_map, all_sweep_seqs)

    hard_baseline_mean = np.mean([baseline_aucs[s] for s in hard_available
                                   if s in baseline_aucs]) if hard_available else 0.0
    std_baseline_mean = np.mean([baseline_aucs[s] for s in std_available
                                  if s in baseline_aucs]) if std_available else 0.0

    print(f"\n  Baseline: hard_mean={hard_baseline_mean:.3f}  std_mean={std_baseline_mean:.3f}")

    # --- Sweep ---
    keys = list(SWEEP_CONFIG.keys())
    values = list(SWEEP_CONFIG.values())
    configs = list(product(*values))

    print(f"\n  Sweeping {len(configs)} configs x {len(all_sweep_seqs)} sequences...\n")

    all_results: list[dict] = []
    best_config = None
    best_hard_delta = -999.0

    for config_vals in configs:
        cfg = dict(zip(keys, config_vals))
        pfc = cfg["pfc_threshold"]
        max_fr = cfg["freeze_max_frames"]
        rel_pfc = cfg["release_pfc"]

        runner = _build_runner(advisory_ckpt, pfc, max_fr, rel_pfc)

        config_aucs: dict[str, float] = {}
        config_freeze_counts: dict[str, int] = {}

        for sname in all_sweep_seqs:
            if sname not in seq_map:
                continue
            seq = seq_map[sname]
            auc, n_freeze = _run_sequence(runner, seq)
            config_aucs[sname] = auc
            config_freeze_counts[sname] = n_freeze

        # Compute metrics
        hard_aucs = [config_aucs[s] for s in hard_available if s in config_aucs]
        std_aucs = [config_aucs[s] for s in std_available if s in config_aucs]

        hard_mean = float(np.mean(hard_aucs)) if hard_aucs else 0.0
        std_mean = float(np.mean(std_aucs)) if std_aucs else 0.0

        hard_delta = hard_mean - hard_baseline_mean
        std_delta = std_mean - std_baseline_mean
        total_freeze = sum(config_freeze_counts.values())

        result = {
            "pfc_threshold": pfc,
            "freeze_max_frames": max_fr,
            "release_pfc": rel_pfc,
            "hard_auc_mean": hard_mean,
            "hard_auc_delta": hard_delta,
            "std_auc_mean": std_mean,
            "std_auc_delta": std_delta,
            "n_freeze_frames_total": total_freeze,
            "per_seq_auc": config_aucs,
            "per_seq_freeze": config_freeze_counts,
        }
        all_results.append(result)

        print(f"  pfc={pfc:.2f} max_fr={max_fr:>2} rel={rel_pfc:.2f} | "
              f"hard_delta={hard_delta:+.3f}  std_delta={std_delta:+.3f}  "
              f"freeze_frames={total_freeze}")

        # Update best: maximise hard_delta while keeping std_delta >= -0.005
        if hard_delta > best_hard_delta and std_delta >= -0.005:
            best_hard_delta = hard_delta
            best_config = result

    # --- Summary ---
    print(f"\n{'─'*70}")
    if best_config is not None:
        print(f"  BEST CONFIG:")
        print(f"    pfc_threshold     = {best_config['pfc_threshold']}")
        print(f"    freeze_max_frames = {best_config['freeze_max_frames']}")
        print(f"    release_pfc       = {best_config['release_pfc']}")
        print(f"    hard_AUC_delta    = {best_config['hard_auc_delta']:+.3f}")
        print(f"    std_AUC_delta     = {best_config['std_auc_delta']:+.3f}")
        print(f"    freeze_frames     = {best_config['n_freeze_frames_total']}")

        # Check success gate
        gate_hard = best_config['hard_auc_delta'] >= 0.03
        gate_full = best_config['std_auc_delta'] >= -0.005
        print(f"\n  Success gates:")
        print(f"    hard_AUC_delta >= +0.03  : {'PASS' if gate_hard else 'FAIL'}")
        print(f"    std_AUC_delta  >= -0.005 : {'PASS' if gate_full else 'FAIL'}")
    else:
        print("  WARNING: No config passed the std_delta >= -0.005 regression gate.")
        # Pick best hard_delta regardless
        best_config = max(all_results, key=lambda r: r["hard_auc_delta"])
        print(f"  Best (unconstrained) config: pfc={best_config['pfc_threshold']} "
              f"max_fr={best_config['freeze_max_frames']} "
              f"rel={best_config['release_pfc']} "
              f"hard_delta={best_config['hard_auc_delta']:+.3f}")

    # --- Save results ---
    output = {
        "advisory_ckpt": advisory_ckpt,
        "hard_sequences": hard_available,
        "std_sequences": std_available,
        "baseline_hard_mean": float(hard_baseline_mean),
        "baseline_std_mean": float(std_baseline_mean),
        "baseline_per_seq": baseline_aucs,
        "best_config": best_config,
        "all_results": all_results,
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved to {out_path}")
    print(f"{'='*70}\n")

    return best_config or {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Sweep center-freeze hyperparameters.")
    ap.add_argument("--advisory", required=True, metavar="CHECKPOINT",
                    help="Path to SALT-RD advisory checkpoint")
    ap.add_argument("--output", default="saltr/results/center_freeze_sweep.json",
                    metavar="PATH", help="Output JSON path")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES, metavar="N",
                    help=f"Max frames per sequence (default: {MAX_FRAMES})")
    ap.add_argument("--hard-only", action="store_true",
                    help="Sweep on hard sequences only (skip std regression check)")
    args = ap.parse_args()

    run_sweep(
        advisory_ckpt=args.advisory,
        output_path=args.output,
        max_frames=args.max_frames,
        hard_only=args.hard_only,
    )


if __name__ == "__main__":
    main()
