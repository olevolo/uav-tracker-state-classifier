"""GFLOPs + per-stage latency profiler for tracker + CSC pipeline.

CLI usage
---------
python tools/profile_pipeline.py \
    --tracker sglatrack \
    --dataset got10k \
    --split val \
    --max_sequences 5 \
    --n_warmup 5 \
    --n_frames 200 \
    --device cpu \
    --output outputs/profile/sglatrack.json

Outputs a JSON file with per-stage FPS / p50 / p95 latency and
GFLOPs/frame for the tracker and CSC model.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "salrtd" / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


log = logging.getLogger("profile_pipeline")


# ---------------------------------------------------------------------------
# Tracker loading helpers
# ---------------------------------------------------------------------------

def _load_tracker(name: str, device: str):
    """Build tracker from the uav_tracker registry by name."""
    # Import all known adapters so they self-register
    from uav_tracker.trackers import sglatrack  # noqa: F401
    try:
        from uav_tracker.trackers import avtrack  # noqa: F401
    except Exception:
        pass
    try:
        from uav_tracker.trackers import ortrack  # noqa: F401
    except Exception:
        pass
    try:
        from uav_tracker.trackers import evptrack  # noqa: F401
    except Exception:
        pass
    try:
        from uav_tracker.trackers import ostrack  # noqa: F401
    except Exception:
        pass
    try:
        from uav_tracker.trackers import kcf_henriques  # noqa: F401
    except Exception:
        pass
    from uav_tracker.registry import TRACKERS
    if name not in TRACKERS.names():
        raise SystemExit(f"tracker {name!r} not registered. Known: {TRACKERS.names()}")
    return TRACKERS.build(name, device=device)


def _get_tracker_flops(name: str) -> Optional[float]:
    """Return _FLOPS_PER_UPDATE from the tracker module, or None."""
    mod_map = {
        "sglatrack": "uav_tracker.trackers.sglatrack",
        "avtrack": "uav_tracker.trackers.avtrack",
        "ortrack": "uav_tracker.trackers.ortrack",
        "evptrack": "uav_tracker.trackers.evptrack",
        "ostrack": "uav_tracker.trackers.ostrack",
        "kcf": "uav_tracker.trackers.kcf_henriques",
    }
    mod_name = mod_map.get(name)
    if mod_name is None:
        return None
    import importlib
    try:
        mod = importlib.import_module(mod_name)
        val = getattr(mod, "_FLOPS_PER_UPDATE", None)
        return float(val) if val is not None else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(dataset: str, split: str):
    import csc_uav_tracking  # noqa: F401 — triggers registration
    from csc_uav_tracking.registry import DATASETS
    if dataset not in DATASETS.names():
        raise SystemExit(f"unknown dataset {dataset!r}. registered: {DATASETS.names()}")
    if dataset == "got10k":
        return DATASETS.build(dataset, split=split)
    return DATASETS.build(dataset)


# ---------------------------------------------------------------------------
# CSC loading
# ---------------------------------------------------------------------------

def _load_csc(checkpoint: Optional[str], device: str):
    """Load a CSCRuntime from checkpoint; returns None if no checkpoint."""
    if checkpoint is None:
        return None
    from csc_lib.csc.inference import load_runtime
    return load_runtime(Path(checkpoint), device=device)


# ---------------------------------------------------------------------------
# Bbox helpers
# ---------------------------------------------------------------------------

def _bbox_tuple(b) -> tuple[float, float, float, float]:
    if b is None:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(b.x), float(b.y), float(b.w), float(b.h))


def _state_telem(state):
    """Extract telemetry fields from TrackState for CSC step."""
    if state is None:
        return {}
    out = {}
    for attr in ("confidence", "apce", "psr"):
        v = getattr(state, attr, None)
        if v is not None:
            try:
                out[attr] = float(v)
            except (TypeError, ValueError):
                pass
    return out


# ---------------------------------------------------------------------------
# Profiling core
# ---------------------------------------------------------------------------

def _profile_sequence(
    tracker,
    sequence,
    *,
    csc_runtime,
    n_warmup: int,
    n_frames: int,
    device: str,
) -> dict[str, Any]:
    """Run one sequence, collect per-frame tracker + CSC latencies.

    Returns tracker_ms and csc_ms lists (length == n_frames timed).
    """
    frames_iter = iter(sequence.frames)
    init_bbox = sequence.init_bbox

    try:
        first_frame = next(frames_iter)
    except StopIteration:
        return {"tracker_ms": [], "csc_ms": []}

    # Init tracker (not timed)
    tracker.init(first_frame, init_bbox)

    if csc_runtime is not None:
        csc_runtime.reset()

    tracker_ms: list[float] = []
    csc_ms: list[float] = []
    n_done = 0

    for frame in frames_iter:
        timed = n_done >= n_warmup

        # --- Tracker step ---
        t0 = time.perf_counter()
        state = tracker.update(frame)
        t_dt = (time.perf_counter() - t0) * 1000.0

        # --- CSC step ---
        c_dt = 0.0
        if csc_runtime is not None:
            telem = _state_telem(state)
            bbox = _bbox_tuple(getattr(state, "bbox", None))
            t1 = time.perf_counter()
            csc_runtime.step(
                confidence=telem.get("confidence"),
                apce=telem.get("apce"),
                psr=telem.get("psr"),
                pred_bbox=bbox,
            )
            c_dt = (time.perf_counter() - t1) * 1000.0

        if timed:
            tracker_ms.append(t_dt)
            csc_ms.append(c_dt)

        n_done += 1
        if n_done >= n_warmup + n_frames:
            break

    return {"tracker_ms": tracker_ms, "csc_ms": csc_ms}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile tracker + CSC pipeline GFLOPs and latency.")
    p.add_argument("--tracker", required=True,
                   choices=["sglatrack", "avtrack", "ortrack", "evptrack", "ostrack", "kcf"],
                   help="Tracker name to profile.")
    p.add_argument("--csc_checkpoint", default=None,
                   help="Optional path to a CSC checkpoint (.pt). "
                        "If omitted, CSC stage is skipped (csc latency = 0).")
    p.add_argument("--dataset", default="got10k",
                   choices=["lasot", "got10k", "uav123", "dtb70", "visdrone_sot"])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--max_sequences", type=int, default=5,
                   help="Cap on number of sequences to use (streaming stop).")
    p.add_argument("--n_warmup", type=int, default=5,
                   help="Frames to discard before timing starts per sequence.")
    p.add_argument("--n_frames", type=int, default=200,
                   help="Total timed frames across all sequences.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default=None,
                   help="Output JSON path. Defaults to outputs/profile/<tracker>.json")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "outputs" / "profile" / f"{args.tracker}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load tracker ---
    log.info("[1/4] loading tracker %s on %s", args.tracker, args.device)
    tracker = _load_tracker(args.tracker, args.device)

    # --- Load CSC (optional) ---
    log.info("[2/4] loading CSC checkpoint: %s", args.csc_checkpoint or "(none)")
    csc_runtime = _load_csc(args.csc_checkpoint, args.device)

    # --- Load dataset ---
    log.info("[3/4] loading dataset %s/%s", args.dataset, args.split)
    dataset = _load_dataset(args.dataset, args.split)
    sequences = list(dataset)
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]

    # --- Profile ---
    log.info("[4/4] profiling %d sequence(s), n_warmup=%d, n_frames=%d",
             len(sequences), args.n_warmup, args.n_frames)

    all_tracker_ms: list[float] = []
    all_csc_ms: list[float] = []
    remaining = args.n_frames

    for i, seq in enumerate(sequences):
        if remaining <= 0:
            break
        log.info("  [%d/%d] %s", i + 1, len(sequences), seq.name)
        r = _profile_sequence(
            tracker, seq,
            csc_runtime=csc_runtime,
            n_warmup=args.n_warmup,
            n_frames=remaining,
            device=args.device,
        )
        all_tracker_ms.extend(r["tracker_ms"])
        all_csc_ms.extend(r["csc_ms"])
        remaining -= len(r["tracker_ms"])

    # --- Latency stats ---
    from csc_lib.eval.custom_metrics.runtime_metrics import summarise_latencies

    tracker_stats = summarise_latencies(all_tracker_ms)
    csc_stats = summarise_latencies(all_csc_ms)
    total_ms = [t + c for t, c in zip(all_tracker_ms, all_csc_ms)]
    total_stats = summarise_latencies(total_ms)

    # --- GFLOPs ---
    tracker_flops_raw = _get_tracker_flops(args.tracker)
    tracker_gflops = (tracker_flops_raw / 1e9) if tracker_flops_raw is not None else None
    tracker_note = None if tracker_gflops is not None else "adapter did not declare _FLOPS_PER_UPDATE"

    csc_gflops: Optional[float] = None
    csc_params: Optional[int] = None
    if csc_runtime is not None:
        from csc_lib.eval.custom_metrics.runtime_metrics import count_csc_flops
        from csc_lib.csc.features import FEATURE_DIM
        win = csc_runtime.feature_cfg.window_size
        fdim = FEATURE_DIM
        flops = count_csc_flops(csc_runtime.model, window=win, feature_dim=fdim)
        csc_gflops = flops / 1e9
        csc_params = sum(p.numel() for p in csc_runtime.model.parameters())

    # --- Overhead metrics ---
    total_tracker_ms = sum(all_tracker_ms)
    total_csc_ms = sum(all_csc_ms)
    total_pipeline_ms = total_tracker_ms + total_csc_ms

    csc_overhead_pct_runtime: Optional[float] = None
    if total_pipeline_ms > 0:
        csc_overhead_pct_runtime = 100.0 * total_csc_ms / total_pipeline_ms

    csc_overhead_pct_gflops: Optional[float] = None
    if tracker_gflops is not None and csc_gflops is not None:
        denom = tracker_gflops + csc_gflops
        if denom > 0:
            csc_overhead_pct_gflops = 100.0 * csc_gflops / denom

    # --- Build output ---
    result: dict[str, Any] = {
        "tracker": args.tracker,
        "csc_checkpoint": args.csc_checkpoint,
        "device": args.device,
        "dataset": args.dataset,
        "split": args.split,
        "n_warmup": args.n_warmup,
        "n_frames_timed": len(all_tracker_ms),
        "tracker_stage": {
            **tracker_stats,
            "gflops_per_frame": tracker_gflops,
            "note": tracker_note,
        },
        "csc": {
            **csc_stats,
            "gflops_per_frame": csc_gflops,
            "params": csc_params,
        },
        "total": total_stats,
        "csc_overhead_pct_runtime": csc_overhead_pct_runtime,
        "csc_overhead_pct_gflops": csc_overhead_pct_gflops,
    }

    with open(output_path, "w") as fh:
        json.dump(result, fh, indent=2)

    log.info("wrote profile to %s", output_path)
    log.info("tracker: mean=%.1f ms (%.0f FPS), gflops=%.4f",
             tracker_stats.get("p50_ms", 0), tracker_stats.get("mean_fps", 0),
             tracker_gflops or 0.0)
    if csc_gflops is not None:
        log.info("csc:     mean=%.1f ms (%.0f FPS), gflops=%.6f",
                 csc_stats.get("p50_ms", 0), csc_stats.get("mean_fps", 0), csc_gflops)
    log.info("total:   mean_fps=%.0f, csc_overhead=%.1f%%",
             total_stats.get("mean_fps", 0), csc_overhead_pct_runtime or 0.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
