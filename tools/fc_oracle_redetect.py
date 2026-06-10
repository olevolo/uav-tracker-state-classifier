#!/usr/bin/env python
"""Offline oracle study: can SGLATrack redetect recover true-FC episodes?

The trigger is fixed from an existing passive CSC run: a frame must be predicted
FC (derived_state == 3) and be truly wrong by GT (IoU < --fc_iou). At the first
frame of each such passive FC segment, this script asks the live SGLATrack
adapter for top-K wide/global redetect candidates. GT is used ONLY to select the
best candidate and decide whether to apply it. After that one oracle switch, the
tracker runs normally again.

This is an upper-bound / controllability experiment, not deployable control.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "salrtd" / "src", PROJECT_ROOT / "src", PROJECT_ROOT):
    sys.path.insert(0, str(_p))

import csc_uav_tracking  # noqa: F401
from csc_uav_tracking.registry import DATASETS
from uav_tracker.registry import TRACKERS
from uav_tracker.types import BBox


DEFAULT_PASSIVE = PROJECT_ROOT / "outputs/eval5_clamp/csc/sglatrack/uav123/test/passive"

# Per-tracker passive baseline directories (used when --passive_dir is not set
# and --tracker is something other than 'sglatrack'). Each must contain
# ``predictions/<seq>.txt`` and ``states/<seq>.jsonl`` with derived_state.
_TRACKER_PASSIVE_DIRS: dict = {
    "sglatrack": DEFAULT_PASSIVE,
    "ortrack": PROJECT_ROOT / "outputs/eval_paperv2/ortrack_uav123_passive_frozen/passive_frozen",
    "avtrack": PROJECT_ROOT / "outputs/eval_paperv2/avtrack_uav123_passive_frozen/passive_frozen",
}


def _bbox_tuple(b) -> tuple[float, float, float, float]:
    if isinstance(b, dict):
        b = b["bbox"]
    if all(hasattr(b, attr) for attr in ("x", "y", "w", "h")):
        return float(b.x), float(b.y), float(b.w), float(b.h)
    return tuple(float(x) for x in b[:4])


def _iou(a, b) -> float:
    ax, ay, aw, ah = _bbox_tuple(a)
    bx, by, bw, bh = _bbox_tuple(b)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _read_preds(path: Path) -> list[tuple[float, float, float, float]]:
    out = []
    for line in open(path):
        vals = [float(x) for x in line.strip().replace("\t", ",").split(",")[:4]]
        out.append(tuple(vals))
    return out


def _read_states(path: Path) -> dict[int, int]:
    out = {}
    for line in open(path):
        row = json.loads(line)
        if row.get("derived_state") is not None:
            out[int(row["frame_idx"])] = int(row["derived_state"])
    return out


def _segment_starts(frames: Iterable[int]) -> set[int]:
    ordered = sorted(set(frames))
    return {t for i, t in enumerate(ordered) if i == 0 or t != ordered[i - 1] + 1}


def _failure_runs(ious: list[float], threshold: float) -> list[int]:
    runs, cur = [], 0
    for v in ious:
        if v < threshold:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def _first_recovery(ious: list[float], start: int, threshold: float, horizon: int) -> int | None:
    for dt, v in enumerate(ious[start:min(len(ious), start + horizon + 1)]):
        if v >= threshold:
            return dt
    return None


def _load_tracker(name: str, device: str, weights_path: str | None):
    """Generic tracker loader; mirrors run_with_csc.py registry pattern."""
    if name == "ortrack":
        importlib.import_module("uav_tracker.trackers.transformer.ortrack")
    elif name == "avtrack":
        importlib.import_module("uav_tracker.trackers.avtrack")
    elif name == "sglatrack":
        importlib.import_module("uav_tracker.trackers.sglatrack")
    elif name == "mobiletrack":
        importlib.import_module("uav_tracker.trackers.siamese.mobiletrack")
    else:
        # Fallback: try the registry-name pattern.
        importlib.import_module(f"uav_tracker.trackers.{name}")
    kwargs = {"device": device}
    if weights_path:
        kwargs["weights_path"] = weights_path
    return TRACKERS.build(name, **kwargs)


def _load_dataset():
    try:
        return list(DATASETS.build("uav123", split="test"))
    except TypeError:
        return list(DATASETS.build("uav123"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", default="sglatrack",
                    choices=["sglatrack", "ortrack", "avtrack", "mobiletrack"],
                    help="Tracker name (registry key). passive_dir defaults from per-tracker map.")
    ap.add_argument("--passive_dir", default=None,
                    help="Override per-tracker default passive_dir. Must contain "
                         "predictions/<seq>.txt and states/<seq>.jsonl with derived_state.")
    ap.add_argument("--output_dir", default="outputs/fc_oracle_redetect")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights_path", default=None)
    ap.add_argument("--include_sequences", nargs="*", default=None)
    ap.add_argument("--max_sequences", type=int, default=5)
    ap.add_argument(
        "--trigger_every_true_fc",
        action="store_true",
        help="run redetect on every passive true-FC frame, not only segment starts",
    )
    ap.add_argument("--fc_iou", type=float, default=0.20)
    ap.add_argument("--candidate_iou", type=float, default=0.20)
    ap.add_argument("--min_improvement", type=float, default=0.05)
    ap.add_argument("--recovery_iou", type=float, default=0.50)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--factors", default="8,12,16")
    ap.add_argument("--grid_size", type=int, default=3)
    ap.add_argument("--max_candidates_per_crop", type=int, default=5)
    ap.add_argument(
        "--shadow_probation_frames",
        type=int,
        default=0,
        help="Oracle safety upper bound: keep the incumbent live and track a redetect "
             "candidate in a second SGLATrack instance for N frames before commit. 0=off.",
    )
    ap.add_argument(
        "--shadow_commit_margin",
        type=float,
        default=0.05,
        help="Shadow branch must beat incumbent mean IoU by this margin during probation. "
             "GT is used only for this offline oracle safety experiment.",
    )
    ap.add_argument(
        "--shadow_abort_frames",
        type=int,
        default=10,
        help="After shadow commit, keep the incumbent branch alive for N frames and oracle-"
             "rollback if it becomes better again. Only used with --shadow_probation_frames.",
    )
    ap.add_argument("--shadow_abort_margin", type=float, default=0.05)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("fc_oracle_redetect")
    # Resolve per-tracker passive_dir default if not explicitly given.
    if args.passive_dir is None:
        if args.tracker not in _TRACKER_PASSIVE_DIRS:
            raise SystemExit(
                f"--passive_dir is required for tracker {args.tracker!r} "
                f"(no default registered)."
            )
        passive = _TRACKER_PASSIVE_DIRS[args.tracker]
    else:
        passive = Path(args.passive_dir)
    log.info("tracker=%s passive_dir=%s", args.tracker, passive)
    out = Path(args.output_dir)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    factors = tuple(float(x) for x in args.factors.split(",") if x.strip())

    seq_index = {s.name: s for s in _load_dataset()}
    eligible = []
    for name, seq in seq_index.items():
        pp = passive / "predictions" / f"{name}.txt"
        sp = passive / "states" / f"{name}.jsonl"
        if not pp.exists() or not sp.exists():
            continue
        preds = _read_preds(pp)
        states = _read_states(sp)
        n = min(len(preds), len(seq.ground_truth))
        true_fc = [
            t for t in range(n)
            if states.get(t) == 3 and _iou(preds[t], seq.ground_truth[t]) < args.fc_iou
        ]
        if true_fc:
            eligible.append((len(true_fc), name, true_fc))
    eligible.sort(reverse=True)
    if args.include_sequences:
        wanted = set(args.include_sequences)
        eligible = [x for x in eligible if x[1] in wanted]
    elif args.max_sequences:
        eligible = eligible[:args.max_sequences]
    if not eligible:
        raise SystemExit("no passive true-FC sequences found")

    tracker = _load_tracker(args.tracker, args.device, args.weights_path)
    shadow_tracker = (
        _load_tracker(args.tracker, args.device, args.weights_path)
        if args.shadow_probation_frames > 0 else None
    )
    all_events, seq_reports = [], []
    for seq_i, (_, name, true_fc_frames) in enumerate(eligible, 1):
        seq = seq_index[name]
        passive_preds = _read_preds(passive / "predictions" / f"{name}.txt")
        trigger_frames = (
            set(true_fc_frames) if args.trigger_every_true_fc
            else _segment_starts(true_fc_frames)
        )
        frames = iter(seq.frames)
        first = next(frames)
        tracker.init(first, seq.init_bbox)
        if shadow_tracker is not None:
            shadow_tracker.init(first, seq.init_bbox)
        oracle_preds = [_bbox_tuple(seq.init_bbox)]
        oracle_ious = [_iou(seq.init_bbox, seq.ground_truth[0])]
        applied_events = []
        probation = None
        fallback_remaining = 0
        active_commit_event = None
        log.info("[%d/%d] %s: %d true-FC frames, %d trigger segments",
                 seq_i, len(eligible), name, len(true_fc_frames), len(trigger_frames))

        for t, frame in enumerate(frames, 1):
            state = tracker.update(frame)
            current = _bbox_tuple(state.bbox)
            current_iou = _iou(current, seq.ground_truth[t])
            output_bbox = current

            # True shadow branch: the incumbent is untouched during probation.
            # GT decides commit/abort only because this script is an offline
            # controllability upper bound, not a deployable verifier.
            if probation is not None and shadow_tracker is not None:
                shadow_state = shadow_tracker.update(frame)
                shadow_bbox = _bbox_tuple(shadow_state.bbox)
                shadow_iou = _iou(shadow_bbox, seq.ground_truth[t])
                probation["incumbent_ious"].append(current_iou)
                probation["shadow_ious"].append(shadow_iou)
                probation["frames"] += 1
                if probation["frames"] >= args.shadow_probation_frames:
                    inc_mean = float(np.mean(probation["incumbent_ious"]))
                    sh_mean = float(np.mean(probation["shadow_ious"]))
                    committed = bool(
                        sh_mean >= inc_mean + args.shadow_commit_margin
                        and shadow_iou >= args.candidate_iou
                    )
                    event = probation["event"]
                    event["shadow_incumbent_mean_iou"] = inc_mean
                    event["shadow_candidate_mean_iou"] = sh_mean
                    event["shadow_committed"] = committed
                    if committed:
                        tracker, shadow_tracker = shadow_tracker, tracker
                        output_bbox, current_iou = shadow_bbox, shadow_iou
                        event["applied"] = True
                        applied_events.append(event)
                        active_commit_event = event
                        fallback_remaining = int(args.shadow_abort_frames)
                    probation = None
            elif fallback_remaining > 0 and shadow_tracker is not None:
                fallback_state = shadow_tracker.update(frame)
                fallback_bbox = _bbox_tuple(fallback_state.bbox)
                fallback_iou = _iou(fallback_bbox, seq.ground_truth[t])
                fallback_remaining -= 1
                if fallback_iou >= current_iou + args.shadow_abort_margin:
                    tracker, shadow_tracker = shadow_tracker, tracker
                    output_bbox, current_iou = fallback_bbox, fallback_iou
                    if active_commit_event is not None:
                        active_commit_event["shadow_rolled_back"] = True
                        active_commit_event["shadow_rollback_frame"] = t
                    active_commit_event = None
                    fallback_remaining = 0
                elif fallback_remaining == 0:
                    active_commit_event = None

            if t in trigger_frames and probation is None and fallback_remaining == 0:
                t0 = time.perf_counter()
                candidates = tracker.redetect(
                    frame,
                    factors=factors,
                    include_current=True,
                    grid_size=args.grid_size,
                    max_candidates=args.max_candidates_per_crop,
                    top_k=args.top_k,
                    rank_by="quality",
                    frame_idx=t,
                ) or []
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if isinstance(candidates, dict):
                    candidates = [candidates]
                scored = sorted(
                    [(_iou(c["bbox"], seq.ground_truth[t]), c) for c in candidates],
                    key=lambda x: x[0],
                    reverse=True,
                )
                best_iou, best = scored[0] if scored else (0.0, None)
                applied = bool(
                    best is not None
                    and best_iou >= args.candidate_iou
                    and best_iou >= current_iou + args.min_improvement
                )
                shadow_started = bool(applied and shadow_tracker is not None)
                if applied and shadow_tracker is None:
                    bx, by, bw, bh = _bbox_tuple(best["bbox"])
                    tracker.override_search_center(bx + bw / 2.0, by + bh / 2.0, bw, bh)
                    output_bbox = (bx, by, bw, bh)
                    current_iou = best_iou
                elif shadow_started:
                    bx, by, bw, bh = _bbox_tuple(best["bbox"])
                    shadow_tracker.init(frame, BBox(x=bx, y=by, w=bw, h=bh))
                event = {
                    "sequence": name,
                    "frame_idx": t,
                    "passive_iou": _iou(passive_preds[t], seq.ground_truth[t]),
                    "pre_switch_iou": _iou(state.bbox, seq.ground_truth[t]),
                    "best_candidate_iou": best_iou,
                    "topk_hit_0_2": bool(best_iou >= 0.20),
                    "topk_hit_0_5": bool(best_iou >= 0.50),
                    "n_candidates": len(candidates),
                    "applied": bool(applied and shadow_tracker is None),
                    "shadow_started": shadow_started,
                    "shadow_committed": False,
                    "shadow_rolled_back": False,
                    "latency_ms": elapsed_ms,
                    "best_candidate": best,
                }
                all_events.append(event)
                if shadow_started:
                    probation = {
                        "event": event,
                        "frames": 1,
                        "incumbent_ious": [current_iou],
                        "shadow_ious": [best_iou],
                    }
                elif applied:
                    applied_events.append(event)
            oracle_preds.append(output_bbox)
            oracle_ious.append(current_iou)

        n = min(len(passive_preds), len(seq.ground_truth), len(oracle_ious))
        passive_ious = [_iou(passive_preds[t], seq.ground_truth[t]) for t in range(n)]
        oracle_ious = oracle_ious[:n]
        for event in applied_events:
            t = event["frame_idx"]
            event["passive_next30_iou"] = float(np.mean(passive_ious[t:min(n, t + args.horizon + 1)]))
            event["oracle_next30_iou"] = float(np.mean(oracle_ious[t:min(n, t + args.horizon + 1)]))
            event["passive_recovery_dt"] = _first_recovery(
                passive_ious, t, args.recovery_iou, args.horizon)
            event["oracle_recovery_dt"] = _first_recovery(
                oracle_ious, t, args.recovery_iou, args.horizon)

        passive_runs = _failure_runs(passive_ious, args.fc_iou)
        oracle_runs = _failure_runs(oracle_ious, args.fc_iou)
        report = {
            "sequence": name,
            "n_frames": n,
            "n_true_fc_segments": len(_segment_starts(true_fc_frames)),
            "n_trigger_frames": len(trigger_frames),
            "n_oracle_switches": len(applied_events),
            "n_shadow_starts": sum(e.get("shadow_started", False) for e in all_events if e["sequence"] == name),
            "n_shadow_rollbacks": sum(e.get("shadow_rolled_back", False) for e in all_events if e["sequence"] == name),
            "passive_auc": float(np.mean(passive_ious)),
            "oracle_auc": float(np.mean(oracle_ious)),
            "delta_auc": float(np.mean(oracle_ious) - np.mean(passive_ious)),
            "passive_failure_frames": int(sum(v < args.fc_iou for v in passive_ious)),
            "oracle_failure_frames": int(sum(v < args.fc_iou for v in oracle_ious)),
            "passive_mean_failure_run": float(np.mean(passive_runs)) if passive_runs else 0.0,
            "oracle_mean_failure_run": float(np.mean(oracle_runs)) if oracle_runs else 0.0,
        }
        seq_reports.append(report)
        with open(out / "predictions" / f"{name}.txt", "w") as fh:
            for b in oracle_preds[:n]:
                fh.write(",".join(f"{x:.6f}" for x in b) + "\n")
        log.info("%s: switches=%d ΔAUC=%+.4f failure_frames=%d->%d",
                 name, len(applied_events), report["delta_auc"],
                 report["passive_failure_frames"], report["oracle_failure_frames"])

    summary = {
        "protocol": "passive-predicted true-FC segment starts; GT selects top-K candidate",
        "config": vars(args),
        "n_sequences": len(seq_reports),
        "n_events": len(all_events),
        "n_topk_hit_0_2": sum(e["topk_hit_0_2"] for e in all_events),
        "n_topk_hit_0_5": sum(e["topk_hit_0_5"] for e in all_events),
        "n_oracle_switches": sum(e["applied"] for e in all_events),
        "n_shadow_starts": sum(e.get("shadow_started", False) for e in all_events),
        "n_shadow_rollbacks": sum(e.get("shadow_rolled_back", False) for e in all_events),
        "topk_recall_0_2": float(np.mean([e["topk_hit_0_2"] for e in all_events])) if all_events else 0.0,
        "topk_recall_0_5": float(np.mean([e["topk_hit_0_5"] for e in all_events])) if all_events else 0.0,
        "mean_delta_auc": float(np.mean([r["delta_auc"] for r in seq_reports])),
        "total_failure_frames_passive": sum(r["passive_failure_frames"] for r in seq_reports),
        "total_failure_frames_oracle": sum(r["oracle_failure_frames"] for r in seq_reports),
        "sequences": seq_reports,
        "events": all_events,
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2))

    print("\n=== FC-triggered oracle SGLA redetect ===")
    print(f"events={summary['n_events']} switches={summary['n_oracle_switches']} "
          f"topK recall@IoU.2={summary['topk_recall_0_2']:.3f} "
          f"recall@IoU.5={summary['topk_recall_0_5']:.3f}")
    print(f"mean seq ΔAUC={summary['mean_delta_auc']:+.4f}  failure frames "
          f"{summary['total_failure_frames_passive']} -> {summary['total_failure_frames_oracle']}")
    print(f"{'sequence':<16}{'switch':>8}{'baseAUC':>9}{'oracleAUC':>11}{'dAUC':>9}{'fail base':>11}{'fail oracle':>13}")
    for r in sorted(seq_reports, key=lambda x: x["delta_auc"], reverse=True):
        print(f"{r['sequence']:<16}{r['n_oracle_switches']:>8}{r['passive_auc']:>9.3f}"
              f"{r['oracle_auc']:>11.3f}{r['delta_auc']:>+9.3f}"
              f"{r['passive_failure_frames']:>11}{r['oracle_failure_frames']:>13}")
    print(f"report: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
