"""Run any tracker adapter with CSC inference (passive or control mode).

Output schema:
    outputs/csc_runs/<tracker>_<dataset>_<split>_<csc_model>/
        predictions/<seq>.txt
        telemetry/<seq>.jsonl
        states/<seq>.jsonl          ← per-frame CSC output
        metrics.json                ← runtime / FPS summary

``states/<seq>.jsonl`` row keys (one per non-init frame):
    frame_idx, localization_probs, confidence_probs,
    predicted_localization, predicted_confidence,
    derived_state, false_confirmed_flag,
    risk_score, aux_probs,
    should_freeze_template, should_expand_search,
    should_request_redetection, should_skip_template_update,
    tracker_latency_ms, csc_latency_ms

Control mode (--csc_mode control):
    Only template-update freeze is implemented for trackers that expose
    ``set_update_enabled(bool)``.  Search-expansion and redetection hints
    are logged but produce no behaviour change (TODO for Phase 3 / CSC §10).
"""
from __future__ import annotations

import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "salrtd" / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

_TRACKER_NAMES = ["sglatrack", "ostrack", "ortrack", "avtrack", "evptrack"]


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


def _load_dataset(name: str, split: str):
    import csc_uav_tracking  # noqa: F401
    from csc_uav_tracking.registry import DATASETS

    if name not in DATASETS.names():
        raise SystemExit(
            f"unknown dataset {name!r}. registered: {DATASETS.names()}"
        )
    if name == "got10k":
        return DATASETS.build(name, split=split)
    return DATASETS.build(name)


def _load_tracker(name: str, weights_path: Optional[str], device: str):
    """Import the named adapter (triggers @TRACKERS.register) then build."""
    _adapter_map = {
        "sglatrack": "uav_tracker.trackers.sglatrack",
        "ostrack":   "uav_tracker.trackers.ostrack",
        "ortrack":   "uav_tracker.trackers.ortrack",
        "avtrack":   "uav_tracker.trackers.avtrack",
        "evptrack":  "uav_tracker.trackers.evptrack",
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
            f"Failed to import tracker adapter {name!r}: {exc!r}"
        )
    if name not in TRACKERS:
        raise SystemExit(
            f"{name!r} not in registry after import. Known: {TRACKERS.names()}"
        )
    kwargs: dict[str, Any] = {"device": device}
    if weights_path:
        kwargs["weights_path"] = weights_path
    return TRACKERS.build(name, **kwargs)


def _bbox_tuple(b) -> tuple[float, float, float, float]:
    if b is None:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(b.x), float(b.y), float(b.w), float(b.h))


def _telemetry_from_state(state) -> dict[str, Any]:
    """Best-effort extraction of confidence/APCE/PSR/score-map stats."""
    out: dict[str, Any] = {}
    if state is None:
        return out
    for attr in ("confidence", "apce", "psr", "response_entropy"):
        v = getattr(state, attr, None)
        if v is not None:
            try:
                out[attr] = float(v)
            except (TypeError, ValueError):
                pass
    raw = getattr(state, "raw", {}) or getattr(state, "aux", {}) or {}
    if isinstance(raw, dict) and "raw" in raw:
        raw = raw["raw"]
    for key in (
        "score", "score_max", "response_max", "response_mean", "response_std",
        "token_keep_ratio", "active_layers", "search_factor", "template_age",
    ):
        if key in raw and raw[key] is not None:
            try:
                out[key] = float(raw[key])
            except (TypeError, ValueError):
                pass
    sm = raw.get("score_map_stats") if isinstance(raw, dict) else None
    if isinstance(sm, dict):
        for k, v in sm.items():
            try:
                out[f"sm_{k}"] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def _try_set_update_enabled(tracker, enabled: bool) -> bool:
    """Apply template-update freeze if the tracker exposes the hook."""
    fn = getattr(tracker, "set_update_enabled", None)
    if callable(fn):
        try:
            fn(enabled)
            return True
        except Exception:
            return False
    # TODO (Phase 3): wire freeze into tracker adapters that carry template-update
    # logic internally (SGLATrack.try_update_template / OSTrack template refresh).
    return False


def _csc_model_name(checkpoint_path: str) -> str:
    """Derive a short run-tag from the checkpoint path stem."""
    return Path(checkpoint_path).stem


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run any tracker adapter with CSC.")
    p.add_argument(
        "--tracker",
        required=True,
        choices=_TRACKER_NAMES,
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=["lasot", "got10k", "uav123", "dtb70", "visdrone_sot"],
    )
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--csc_checkpoint", required=True, help="Path to CSC .pth checkpoint.")
    p.add_argument(
        "--csc_mode",
        choices=["passive", "control"],
        default="passive",
        help="passive = observe only; control = apply freeze hint.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output_dir",
        default=None,
        help="Root output dir. Defaults to outputs/csc_runs.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Cap the number of sequences (smoke / debug).",
    )
    p.add_argument(
        "--weights_path",
        default=None,
        help="Override path to tracker weights checkpoint.",
    )
    p.add_argument(
        "--exit_router",
        action="store_true",
        default=False,
        help=(
            "SGLATrack only. Activate StateExitRouter: CSC-predicted state "
            "overrides SGLATrack's (collapsed) MLP exit-router per frame. "
            "No effect on other trackers. Requires --csc_mode control or "
            "passive (routing happens regardless of template-freeze control). "
            "Default OFF — preserves baseline behaviour."
        ),
    )
    p.add_argument(
        "--exit_router_hold",
        type=int,
        default=5,
        help="StateExitRouter min_hold_frames hysteresis (default 5).",
    )
    p.add_argument(
        "--exit_router_min_conf",
        type=float,
        default=0.0,
        help=(
            "StateExitRouter min CSC state confidence to trust a risky-state "
            "prediction. 0.0 = trust all (default). 0.6 = downgrade low-"
            "confidence risky states to uncertain."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    import torch
    from csc_lib.csc.inference import load_runtime

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger("run_with_csc")
    args = parse_args(argv)
    _set_seed(args.seed)

    csc_model_tag = _csc_model_name(args.csc_checkpoint)
    run_tag = f"{args.tracker}_{args.dataset}_{args.split}_{csc_model_tag}"

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "outputs" / "csc_runs")

    out_root = Path(output_dir) / run_tag
    pred_dir = out_root / "predictions"
    tel_dir = out_root / "telemetry"
    states_dir = out_root / "states"
    for d in (pred_dir, tel_dir, states_dir):
        d.mkdir(parents=True, exist_ok=True)

    log.info("loading dataset %s/%s", args.dataset, args.split)
    dataset = _load_dataset(args.dataset, args.split)

    log.info("loading tracker %s on %s", args.tracker, args.device)
    tracker = _load_tracker(args.tracker, args.weights_path, args.device)

    log.info("loading CSC from %s", args.csc_checkpoint)
    csc = load_runtime(Path(args.csc_checkpoint), device=args.device)

    # --- StateExitRouter (SGLATrack only, default OFF) ---
    exit_router = None
    if args.exit_router:
        if args.tracker != "sglatrack":
            log.warning(
                "--exit_router is only supported for sglatrack; "
                "ignoring (tracker=%s)",
                args.tracker,
            )
        else:
            from uav_tracker.trackers.exit_router import StateExitRouter
            exit_router = StateExitRouter(
                min_hold_frames=args.exit_router_hold,
                min_state_confidence=args.exit_router_min_conf,
            )
            log.info(
                "StateExitRouter enabled (min_hold=%d, min_conf=%.2f)",
                args.exit_router_hold,
                args.exit_router_min_conf,
            )

    sequences = list(dataset)
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]

    n_seq = len(sequences)
    total_frames = 0

    # Per-stage latency accumulators (ms, across all frames of all sequences)
    tracker_latencies_ms: list[float] = []
    csc_latencies_ms: list[float] = []
    total_latencies_ms: list[float] = []

    seq_records: list[dict] = []
    update_freezes = 0
    exit_router_switches_total = 0

    for i, seq in enumerate(sequences):
        log.info("[%d/%d] %s", i + 1, n_seq, seq.name)

        # Determine image size from first frame for CSC reset
        frames_iter = iter(seq.frames)
        try:
            first_frame = next(frames_iter)
        except StopIteration:
            log.warning("sequence %s is empty, skipping", seq.name)
            continue
        h, w = first_frame.shape[:2]
        csc.reset(image_size=(int(w), int(h)))

        # Reset exit router for each new sequence so hysteresis state doesn't
        # carry over from previous sequences.
        if exit_router is not None:
            exit_router.reset()

        bboxes: list[tuple[float, float, float, float]] = []
        telemetry: list[dict] = []
        states_rows: list[dict] = []
        seq_tracker_ms: list[float] = []
        seq_csc_ms: list[float] = []

        # --- Init frame 0 ---
        t0 = time.perf_counter()
        tracker.init(first_frame, seq.init_bbox)
        init_ms = (time.perf_counter() - t0) * 1000.0

        bboxes.append(_bbox_tuple(seq.init_bbox))
        telemetry.append({"frame_idx": 0, "init": True, "latency_ms": init_ms})
        states_rows.append({"frame_idx": 0, "init": True})

        # --- Subsequent frames ---
        frame_idx = 1
        for frame in frames_iter:
            # --- StateExitRouter: apply force_layer_idx from previous frame's CSC
            # prediction BEFORE calling tracker.update().  On frame 1 the router
            # has not yet produced a prediction, so it returns DEFAULT_EXIT_IDX
            # (-1 = let MLP decide), which is the same as the tracker default.
            if exit_router is not None:
                # _force_layer_idx is read on every update() call.  Direct
                # attribute set is safe — no method encapsulation needed here.
                tracker._force_layer_idx = exit_router.current_idx  # type: ignore[attr-defined]

            t_track = time.perf_counter()
            state = tracker.update(frame)
            track_ms = (time.perf_counter() - t_track) * 1000.0

            tel = _telemetry_from_state(state)
            tel["frame_idx"] = frame_idx
            tel["latency_ms"] = track_ms
            # Capture the block that SGLATrack actually used (patched backbone
            # returns this in aux["selected_block"]; -1 if not available).
            _applied_force = getattr(tracker, "_force_layer_idx", -1)
            _selected_block = None
            _aux = getattr(state, "aux", None)
            if isinstance(_aux, dict):
                _sb = _aux.get("selected_block")
                if _sb is not None:
                    _selected_block = int(_sb)
            telemetry.append(tel)

            bbox = _bbox_tuple(getattr(state, "bbox", None))
            bboxes.append(bbox)

            with torch.no_grad():
                csc_pred = csc.step(
                    confidence=tel.get("confidence"),
                    apce=tel.get("apce"),
                    psr=tel.get("psr"),
                    pred_bbox=bbox,
                )

            # --- StateExitRouter: advance router with this frame's CSC state.
            # The returned force_idx will be applied at the NEXT frame's
            # tracker.update() call (causal — no look-ahead).
            exit_force_idx: Optional[int] = None
            exit_state_str: Optional[str] = None
            if exit_router is not None:
                from uav_tracker.trackers.exit_router import (
                    DERIVED_INT_TO_ROUTER_STATE,
                    _DERIVED_FALLBACK_STATE,
                )
                exit_state_str = DERIVED_INT_TO_ROUTER_STATE.get(
                    int(csc_pred.derived_state), _DERIVED_FALLBACK_STATE
                )
                # Use the argmax of localization_probs as state confidence proxy
                # (not a true per-class confidence, but a reasonable signal).
                _loc_conf = float(
                    csc_pred.localization_probs[csc_pred.predicted_localization]
                )
                exit_force_idx = exit_router.step(
                    exit_state_str, state_confidence=_loc_conf
                )

            # --- Control hooks (Phase 1: template-update freeze only) ---
            if args.csc_mode == "control":
                if csc_pred.should_skip_template_update:
                    if _try_set_update_enabled(tracker, False):
                        update_freezes += 1
                # TODO (Phase 3): wire should_expand_search into tracker search
                # expansion hooks once adapters expose them.
                # TODO (Phase 3): wire should_request_redetection into the
                # redetection pipeline once the detector integration is ready.

            row: dict = {
                "frame_idx": frame_idx,
                "localization_probs": csc_pred.localization_probs.tolist(),
                "confidence_probs": csc_pred.confidence_probs.tolist(),
                "predicted_localization": int(csc_pred.predicted_localization),
                "predicted_confidence": int(csc_pred.predicted_confidence),
                "derived_state": int(csc_pred.derived_state),
                "false_confirmed_flag": bool(csc_pred.false_confirmed_flag),
                "risk_score": float(csc_pred.risk_score),
                "aux_probs": csc_pred.aux_probs,
                "should_freeze_template": bool(csc_pred.should_freeze_template),
                "should_expand_search": bool(csc_pred.should_expand_search),
                "should_request_redetection": bool(
                    csc_pred.should_request_redetection
                ),
                "should_skip_template_update": bool(
                    csc_pred.should_skip_template_update
                ),
                "tracker_latency_ms": float(track_ms),
                "csc_latency_ms": float(csc_pred.latency_ms),
            }
            # Exit router telemetry (only populated when --exit_router is active)
            if exit_router is not None:
                row["exit_router_state"] = exit_state_str
                row["exit_router_force_idx"] = exit_force_idx
                row["exit_router_applied_force"] = _applied_force
                row["exit_router_selected_block"] = _selected_block
            states_rows.append(row)

            seq_tracker_ms.append(track_ms)
            seq_csc_ms.append(float(csc_pred.latency_ms))
            tracker_latencies_ms.append(track_ms)
            csc_latencies_ms.append(float(csc_pred.latency_ms))
            total_latencies_ms.append(track_ms + float(csc_pred.latency_ms))
            frame_idx += 1

        # --- Save sequence outputs ---
        with open(pred_dir / f"{seq.name}.txt", "w") as fh:
            for b in bboxes:
                fh.write(",".join(f"{v:.4f}" for v in b) + "\n")
        with open(tel_dir / f"{seq.name}.jsonl", "w") as fh:
            for r in telemetry:
                fh.write(json.dumps(r) + "\n")
        with open(states_dir / f"{seq.name}.jsonl", "w") as fh:
            for r in states_rows:
                fh.write(json.dumps(r) + "\n")

        n_frames = len(bboxes)
        total_frames += n_frames
        seq_total_ms = float(sum(seq_tracker_ms) + sum(seq_csc_ms))
        seq_rec: dict = {
            "sequence": seq.name,
            "n_frames": n_frames,
            "time_ms": seq_total_ms,
            "fps": (
                n_frames / (seq_total_ms / 1000.0) if seq_total_ms > 0 else 0.0
            ),
        }
        if exit_router is not None:
            seq_rec["exit_router_stats"] = exit_router.stats_dict()
            exit_router_switches_total += exit_router.stats.n_switched
        seq_records.append(seq_rec)

    # --- Runtime metrics ---
    from csc_lib.eval.custom_metrics.runtime_metrics import latency_summary

    tracker_stats = latency_summary(np.array(tracker_latencies_ms))
    csc_stats = latency_summary(np.array(csc_latencies_ms))
    total_stats = latency_summary(np.array(total_latencies_ms))

    metrics = {
        "tracker": args.tracker,
        "dataset": args.dataset,
        "split": args.split,
        "csc_mode": args.csc_mode,
        "csc_checkpoint": args.csc_checkpoint,
        "weights_path": args.weights_path,
        "device": args.device,
        "git_commit": _git_commit(),
        "datetime": datetime.now(timezone.utc).isoformat(),
        "n_sequences": len(seq_records),
        "n_frames": total_frames,
        "mean_tracker_fps": tracker_stats["mean_fps"],
        "mean_csc_fps": csc_stats["mean_fps"],
        "mean_total_fps": total_stats["mean_fps"],
        "control_template_update_freezes": update_freezes,
        "tracker_latency": {
            k: tracker_stats[k]
            for k in ("mean_ms", "p95_ms", "p99_ms", "median_ms", "p90_ms")
        },
        "csc_latency": {
            k: csc_stats[k]
            for k in ("mean_ms", "p95_ms", "p99_ms", "median_ms", "p90_ms")
        },
        "total_latency": {
            k: total_stats[k]
            for k in ("mean_ms", "p95_ms", "p99_ms", "median_ms", "p90_ms")
        },
        "sequences": seq_records,
    }

    if exit_router is not None:
        metrics["exit_router_enabled"] = True
        metrics["exit_router_min_hold"] = args.exit_router_hold
        metrics["exit_router_min_conf"] = args.exit_router_min_conf
        metrics["exit_router_switches_total"] = exit_router_switches_total
        metrics["exit_router_policy"] = exit_router.policy_summary()
    else:
        metrics["exit_router_enabled"] = False

    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(
        "done: %d seq, %d frames — tracker %.1f fps, csc %.1f fps, total %.1f fps"
        " (freezes=%d, exit_switches=%d) -> %s",
        len(seq_records),
        total_frames,
        metrics["mean_tracker_fps"],
        metrics["mean_csc_fps"],
        metrics["mean_total_fps"],
        update_freezes,
        exit_router_switches_total,
        out_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
