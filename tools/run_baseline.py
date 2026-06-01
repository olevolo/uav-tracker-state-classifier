"""Run any tracker adapter on a dataset and save predictions + telemetry.

Saves:
    outputs/baselines/<tracker>/<dataset>/<split>/predictions/<seq>.txt
    outputs/baselines/<tracker>/<dataset>/<split>/telemetry/<seq>.jsonl
    outputs/baselines/<tracker>/<dataset>/<split>/manifest.json

Tracker adapters live in ``salrtd/src/uav_tracker/trackers/<name>.py``.
This script imports each adapter explicitly before building via the registry
so the ``@TRACKERS.register(...)`` decorator fires at the right time.

Manifest keys (preserved for backward-compat with calibrator / label generator):
    tracker, dataset, split, device, git_commit, weights_path, seed, datetime,
    n_sequences, n_frames, total_time_s, mean_fps, sequences[].
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import psutil as _psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Make both the new src package and the archived SALT-RD package importable.
# NEW src must come BEFORE salrtd/src so updated adapters (with identity features) take priority.
sys.path.insert(0, str(PROJECT_ROOT / "salrtd" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))   # takes priority over salrtd
sys.path.insert(0, str(PROJECT_ROOT))

_TRACKER_NAMES = ["sglatrack", "ostrack", "ortrack", "avtrack", "evptrack", "fartrack", "uetrack"]


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _load_dataset(name: str, split: str, max_frames_per_seq: Optional[int] = None):
    """Load a dataset from csc_uav_tracking.registry."""
    import csc_uav_tracking  # noqa: F401  triggers registration
    from csc_uav_tracking.registry import DATASETS

    if name not in DATASETS.names():
        raise SystemExit(
            f"unknown dataset {name!r}. registered: {DATASETS.names()}"
        )
    if name == "got10k":
        return DATASETS.build(name, split=split)
    if name == "lasot" and max_frames_per_seq is not None:
        return DATASETS.build(name, max_frames=max_frames_per_seq)
    return DATASETS.build(name)


def _load_tracker(name: str, weights_path: Optional[str], device: str,
                  force_layer: int = -1):
    """Import the named adapter module (registering it) then build via TRACKERS registry."""
    # Explicit import triggers the @TRACKERS.register(...) decorator.
    _adapter_map = {
        "sglatrack": "uav_tracker.trackers.sglatrack",
        "ostrack":   "uav_tracker.trackers.ostrack",
        "ortrack":   "uav_tracker.trackers.ortrack",
        "avtrack":   "uav_tracker.trackers.avtrack",
        "evptrack":  "uav_tracker.trackers.evptrack",
        "fartrack":  "uav_tracker.trackers.fartrack",
        "uetrack":   "uav_tracker.trackers.uetrack",
    }
    if name not in _adapter_map:
        raise SystemExit(
            f"unknown tracker {name!r}. supported: {_TRACKER_NAMES}"
        )

    try:
        import importlib
        importlib.import_module(_adapter_map[name])
        from uav_tracker.registry import TRACKERS
    except Exception as exc:
        raise SystemExit(
            f"Failed to import tracker adapter {name!r} from salrtd/src/uav_tracker.\n"
            f"Underlying error: {exc!r}\n"
            "Ensure salrtd/ exists and the relevant external tracker repo is present."
        )

    if name not in TRACKERS:
        raise SystemExit(
            f"{name!r} not in registry after import. Known: {TRACKERS.names()}"
        )

    kwargs: dict[str, Any] = {"device": device}
    if weights_path:
        kwargs["weights_path"] = weights_path
    # SGLATrack supports force_layer_idx for exit-block ablation
    if name == "sglatrack" and force_layer != -1:
        kwargs["force_layer_idx"] = force_layer
    return TRACKERS.build(name, **kwargs)

def _bbox_tuple(b) -> tuple[float, float, float, float]:
    if b is None:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(b.x), float(b.y), float(b.w), float(b.h))


def _telemetry_from_state(state) -> dict[str, Any]:
    """Best-effort extraction of confidence/APCE/PSR/score-map stats from TrackState."""
    out: dict[str, Any] = {}
    if state is None:
        return out
    # Direct attributes on TrackState (all adapters fill these)
    for attr in ("confidence", "apce", "psr", "response_entropy"):
        v = getattr(state, attr, None)
        if v is not None:
            try:
                out[attr] = float(v)
            except (TypeError, ValueError):
                pass
    raw = getattr(state, "raw", {}) or getattr(state, "aux", {}) or {}
    # Flatten the nested {"raw": {...}} structure used by some adapters
    if isinstance(raw, dict) and "raw" in raw:
        raw = raw["raw"]
    for key in (
        "score",
        "score_max",
        "response_max",
        "response_mean",
        "response_std",
        "token_keep_ratio",
        "active_layers",
        "search_factor",
        "template_age",
        "selected_block",
        "force_layer_idx",
    ):
        if key in raw and raw[key] is not None:
            try:
                out[key] = float(raw[key])
            except (TypeError, ValueError):
                pass
    sm = raw.get("score_map_stats") if isinstance(raw, dict) else None
    if isinstance(sm, dict):
        for k, v in sm.items():
            if k == "candidates":
                # candidates is a list-of-dicts — skip, not a scalar
                continue
            try:
                out[f"sm_{k}"] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def _run_one_sequence(
    tracker,
    sequence,
    *,
    save_telemetry: bool,
    pred_path: Optional[Path] = None,
    tel_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Run tracker on one sequence.

    If ``pred_path`` / ``tel_path`` are provided the function writes each row
    directly to disk (streaming mode) and returns empty lists — this caps
    per-sequence RAM regardless of sequence length.  If paths are None it
    falls back to the legacy in-memory list behaviour (used by callers that
    inspect the returned lists).
    """
    _proc = _psutil.Process(os.getpid())

    bboxes: list[tuple[float, float, float, float]] = []
    telemetry: list[dict] = []
    latencies_ms: list[float] = []

    streaming = pred_path is not None

    # Open output files before the frame loop when streaming
    pred_fh = None
    tel_fh = None
    if streaming:
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred_fh = open(pred_path, "w")
        if save_telemetry and tel_path is not None:
            tel_path.parent.mkdir(parents=True, exist_ok=True)
            tel_fh = open(tel_path, "w")

    log = logging.getLogger("run_baseline")

    def _write_bbox(b: tuple[float, float, float, float]) -> None:
        line = ",".join(f"{v:.4f}" for v in b) + "\n"
        if pred_fh is not None:
            pred_fh.write(line)
        else:
            bboxes.append(b)

    def _write_tel(row: dict) -> None:
        if tel_fh is not None:
            tel_fh.write(json.dumps(row) + "\n")
        else:
            telemetry.append(row)

    init_bbox = sequence.init_bbox
    frames_iter = iter(sequence.frames)
    try:
        first_frame = next(frames_iter)
    except StopIteration:
        if pred_fh:
            pred_fh.close()
        if tel_fh:
            tel_fh.close()
        return {"bboxes": [], "telemetry": [], "latencies_ms": []}

    t0 = time.perf_counter()
    tracker.init(first_frame, init_bbox)
    init_ms = (time.perf_counter() - t0) * 1000.0
    _write_bbox(_bbox_tuple(init_bbox))
    latencies_ms.append(init_ms)
    if save_telemetry:
        _write_tel({"frame_idx": 0, "init": True})
    del first_frame

    frame_idx = 1
    for frame in frames_iter:
        t0 = time.perf_counter()
        state = tracker.update(frame)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        _write_bbox(_bbox_tuple(getattr(state, "bbox", None)))
        if save_telemetry:
            row = _telemetry_from_state(state)
            row["frame_idx"] = frame_idx
            row["latency_ms"] = dt_ms
            # Identity / appearance drift signals from SGLATrack (others return default 1.0/0.0)
            row["initial_template_sim"] = float(getattr(tracker, "_initial_template_sim", 1.0))
            row["last_cosine_sim"]       = float(getattr(tracker, "_last_cosine_sim", 1.0))
            row["appearance_drift"]      = float(1.0 - row["initial_template_sim"])
            _write_tel(row)
            if pred_fh and frame_idx % 200 == 0:
                pred_fh.flush()
            if tel_fh and frame_idx % 200 == 0:
                tel_fh.flush()

        # Log RSS every 100 frames to detect linear growth
        if frame_idx % 100 == 0:
            rss_mb = _proc.memory_info().rss / 1024 / 1024
            log.info("    mem frame=%d rss=%.0fMB", frame_idx, rss_mb)

        # Explicit cleanup — release image array and state after each frame
        del frame
        del state
        if (frame_idx & 0xFF) == 0:  # every 256 frames
            gc.collect()

        frame_idx += 1

    if pred_fh:
        pred_fh.close()
    if tel_fh:
        tel_fh.close()

    return {"bboxes": bboxes, "telemetry": telemetry, "latencies_ms": latencies_ms}


def _save_predictions_txt(
    path: Path, bboxes: list[tuple[float, float, float, float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for b in bboxes:
            fh.write(",".join(f"{v:.4f}" for v in b) + "\n")


def _save_telemetry_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run any tracker adapter on a dataset.")
    p.add_argument(
        "--tracker",
        required=True,
        choices=_TRACKER_NAMES,
        help="Tracker adapter name.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=["lasot", "got10k", "uav123", "dtb70", "visdrone_sot", "uavdt_sot", "uavtrack112", "uav123_10fps"],
    )
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output_dir",
        default=None,
        help="Root output dir. Defaults to outputs/baselines/<tracker>.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Cap the number of sequences (smoke / debug).",
    )
    p.add_argument("--no_telemetry", action="store_true")
    p.add_argument(
        "--weights_path",
        default=None,
        help="Override path to tracker weights checkpoint.",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip sequences whose prediction file already exists (resume after crash).",
    )
    p.add_argument(
        "--skip_sequences",
        nargs="*",
        default=[],
        help="Sequence names to skip entirely (e.g. bicycle-18 bird-18 for OOM-prone seqs).",
    )
    p.add_argument(
        "--max_frames_per_seq",
        type=int,
        default=None,
        help="Cap frames processed per sequence (mitigates SGLATrack OOM on 1000+ frame LaSOT seqs).",
    )
    p.add_argument(
        "--force_layer",
        type=int,
        default=-1,
        help=(
            "SGLATrack only: force exit block index (0=block6, 3=block9, 5=block11). "
            "Default -1 = MLP router decides (default SGLATrack behavior)."
        ),
    )
    p.add_argument(
        "--include_sequences",
        nargs="*",
        default=None,
        help="If set, only run these sequence names (whitelist).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger("run_baseline")
    args = parse_args(argv)
    _set_seed(args.seed)

    # Always include tracker name so concurrent runs never overwrite each other.
    # Idempotent: if --output_dir already ends with <tracker>/<dataset>/<split>,
    # don't append again (prevents path-doubling when launchers pass full path).
    output_dir = args.output_dir or str(PROJECT_ROOT / "outputs" / "baselines")
    expected_tail = Path(args.tracker) / args.dataset / args.split
    out_root = Path(output_dir)
    if not str(out_root).endswith(str(expected_tail)):
        out_root = out_root / expected_tail
    pred_dir = out_root / "predictions"
    tel_dir = out_root / "telemetry"
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("loading dataset %s/%s", args.dataset, args.split)
    dataset = _load_dataset(args.dataset, args.split, args.max_frames_per_seq)

    log.info("loading tracker %s on %s", args.tracker, args.device)
    tracker = _load_tracker(args.tracker, args.weights_path, args.device,
                            force_layer=args.force_layer)

    sequences = list(dataset)
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]
    # Filter to whitelist when --include_sequences is given
    if args.include_sequences is not None:
        include_set = set(args.include_sequences)
        sequences = [s for s in sequences if s.name in include_set]
        log.info("--include_sequences: kept %d sequences out of %d",
                 len(sequences), len(list(dataset)))

    n_seq = len(sequences)
    total_frames = 0
    total_time_ms = 0.0
    seq_records: list[dict] = []

    n_skipped = 0
    skip_set = set(args.skip_sequences)
    for i, sequence in enumerate(sequences):
        if sequence.name in skip_set:
            log.info("[%d/%d] %s SKIP (--skip_sequences)", i + 1, n_seq, sequence.name)
            n_skipped += 1
            continue
        if args.skip_existing and (pred_dir / f"{sequence.name}.txt").exists():
            log.info("[%d/%d] %s SKIP (exists)", i + 1, n_seq, sequence.name)
            n_skipped += 1
            continue
        log.info("[%d/%d] %s", i + 1, n_seq, sequence.name)
        # Build output paths for streaming mode (writes each row inline, capping RAM)
        seq_pred_path = pred_dir / f"{sequence.name}.txt"
        seq_tel_path = tel_dir / f"{sequence.name}.jsonl" if not args.no_telemetry else None
        try:
            result = _run_one_sequence(
                tracker,
                sequence,
                save_telemetry=not args.no_telemetry,
                pred_path=seq_pred_path,
                tel_path=seq_tel_path,
            )
        except Exception as exc:
            log.exception("sequence %s failed: %s", sequence.name, exc)
            continue

        # In streaming mode bboxes/telemetry lists are empty; count frames from latencies.
        n_frames = len(result["latencies_ms"])
        seq_time_ms = float(sum(result["latencies_ms"]))
        total_frames += n_frames
        total_time_ms += seq_time_ms

        # In streaming mode the files are already written; skip the post-loop saves.
        if result["bboxes"]:
            _save_predictions_txt(seq_pred_path, result["bboxes"])
        if not args.no_telemetry and result["telemetry"]:
            _save_telemetry_jsonl(seq_tel_path, result["telemetry"])

        seq_records.append(
            {
                "sequence": sequence.name,
                "n_frames": n_frames,
                "time_ms": seq_time_ms,
                "fps": (n_frames / (seq_time_ms / 1000.0)) if seq_time_ms > 0 else 0.0,
            }
        )

    manifest = {
        "tracker": args.tracker,
        "dataset": args.dataset,
        "split": args.split,
        "device": args.device,
        "git_commit": _git_commit(),
        "weights_path": args.weights_path,
        "seed": args.seed,
        "datetime": datetime.now(timezone.utc).isoformat(),
        "n_sequences": len(seq_records),
        "n_frames": total_frames,
        "total_time_s": total_time_ms / 1000.0,
        "mean_fps": (
            (total_frames / (total_time_ms / 1000.0)) if total_time_ms > 0 else 0.0
        ),
        "sequences": seq_records,
    }
    with open(out_root / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    log.info(
        "done: %d sequences, %d frames, mean FPS=%.1f -> %s",
        len(seq_records),
        total_frames,
        manifest["mean_fps"],
        out_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
