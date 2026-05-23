"""Episode-level evaluation for CSC predictions.

An episode is a contiguous run of frames with the same derived_state label.
For each target state (LOST_AWARE, FALSE_CONFIRMED, CORRECT_UNCERTAIN), computes:
  - GT episode count / predicted episode count
  - Episode Recall@K (K=5, K=10): GT episode detected if CSC predicts same state
    within K frames after episode start
  - Episode Precision
  - Mean / Median detection delay (frames) for detected episodes
  - False alarm episodes per 1000 frames
  - Mean episode duration

Usage
-----
    python tools/evaluate_csc_episodes.py \\
        --labels <labels_dir> \\
        --predictions <csc_predictions_dir> \\
        --out <episodes_dir>

    The <labels_dir> must contain labels.jsonl files (searched recursively).
    The <csc_predictions_dir> must contain <sequence>.jsonl files with at least
    a ``derived_state`` field per row; falls back to labels if no separate CSC
    predictions are given (then recall = precision = 1.0 — sanity check mode).

Exit codes
----------
    0  analysis complete (metrics written)
    1  no data found
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Primary target states per spec (plus CORRECT_UNCERTAIN for context)
TARGET_STATES = {
    2: "LOST_AWARE",
    3: "FALSE_CONFIRMED",
    1: "CORRECT_UNCERTAIN",
}

DERIVED_STATE_NAMES = {
    0: "CORRECT_CONFIRMED",
    1: "CORRECT_UNCERTAIN",
    2: "LOST_AWARE",
    3: "FALSE_CONFIRMED",
}

RECALL_K_VALUES = [5, 10]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _collect_labels(labels_dir: Path) -> dict[str, list[dict]]:
    """Load all labels.jsonl files, return dict keyed by 'dataset/sequence'."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(labels_dir.rglob("labels.jsonl")):
        rows = _load_jsonl(p)
        for r in rows:
            key = f"{r['dataset']}/{r['sequence']}"
            groups[key].append(r)
    # Also try per-sequence jsonl files named after sequences
    for p in sorted(labels_dir.rglob("*.jsonl")):
        if p.name == "labels.jsonl":
            continue
        rows = _load_jsonl(p)
        if not rows:
            continue
        first = rows[0]
        if "dataset" in first and "sequence" in first:
            key = f"{first['dataset']}/{first['sequence']}"
            if key not in groups:
                groups[key].extend(rows)
    # Sort by frame_idx
    for key in groups:
        groups[key].sort(key=lambda r: r.get("frame_idx", 0))
    return dict(groups)


def _collect_predictions(predictions_dir: Path) -> dict[str, list[dict]]:
    """Load per-sequence CSC prediction JSONLs.

    Each JSONL row must have ``derived_state`` (int) and ideally ``frame_idx``.
    """
    groups: dict[str, list[dict]] = {}
    for p in sorted(predictions_dir.rglob("*.jsonl")):
        rows = _load_jsonl(p)
        if not rows:
            continue
        first = rows[0]
        if "dataset" in first and "sequence" in first:
            key = f"{first['dataset']}/{first['sequence']}"
        else:
            # Fallback: use stem as key (may not match label key)
            key = p.stem
        rows.sort(key=lambda r: r.get("frame_idx", 0))
        groups[key] = rows
    return groups


# ---------------------------------------------------------------------------
# Episode extraction
# ---------------------------------------------------------------------------


def _extract_episodes(states: list[int]) -> list[tuple[int, int, int]]:
    """Return list of (start, end, state) tuples for contiguous runs.

    ``end`` is inclusive.
    """
    if not states:
        return []
    episodes: list[tuple[int, int, int]] = []
    start = 0
    cur = states[0]
    for i in range(1, len(states)):
        if states[i] != cur:
            episodes.append((start, i - 1, cur))
            start = i
            cur = states[i]
    episodes.append((start, len(states) - 1, cur))
    return episodes


# ---------------------------------------------------------------------------
# Episode matching
# ---------------------------------------------------------------------------


def _match_episodes(
    gt_episodes: list[tuple[int, int, int]],   # (start, end, state)
    pred_states: list[int],
    target_state: int,
    k: int,
) -> tuple[list[int], list[int]]:
    """Return (detection_delays_for_detected, undetected_starts).

    An episode is detected if ``pred_states[start : start+k]`` contains
    at least one frame with the target state.  Detection delay is the
    offset from episode start to the first such prediction frame.
    """
    detected_delays: list[int] = []
    undetected_starts: list[int] = []
    for start, end, state in gt_episodes:
        if state != target_state:
            continue
        window = pred_states[start : start + k]
        found = False
        for offset, ps in enumerate(window):
            if ps == target_state:
                detected_delays.append(offset)
                found = True
                break
        if not found:
            undetected_starts.append(start)
    return detected_delays, undetected_starts


# ---------------------------------------------------------------------------
# Per-sequence metric computation
# ---------------------------------------------------------------------------


def _compute_sequence_metrics(
    gt_states: list[int],
    pred_states: list[int],
    target_state: int,
) -> dict:
    """Compute episode-level metrics for one sequence and one target state."""
    n = min(len(gt_states), len(pred_states))
    gt_states = gt_states[:n]
    pred_states = pred_states[:n]

    gt_episodes = _extract_episodes(gt_states)
    pred_episodes = _extract_episodes(pred_states)

    gt_target_episodes = [(s, e, st) for s, e, st in gt_episodes if st == target_state]
    pred_target_episodes = [(s, e, st) for s, e, st in pred_episodes if st == target_state]

    n_gt = len(gt_target_episodes)
    n_pred = len(pred_target_episodes)

    # Episode Recall@K
    recall_at = {}
    for k in RECALL_K_VALUES:
        detected, _ = _match_episodes(gt_target_episodes, pred_states, target_state, k)
        recall_at[k] = len(detected) / max(1, n_gt) if n_gt > 0 else float("nan")

    # Detection delays (at K=10 for summary)
    detected_delays_10, _ = _match_episodes(gt_target_episodes, pred_states, target_state, 10)
    if detected_delays_10:
        mean_delay = float(np.mean(detected_delays_10))
        median_delay = float(np.median(detected_delays_10))
    else:
        mean_delay = float("nan")
        median_delay = float("nan")

    # Episode Precision: among predicted target-state episodes, how many
    # overlap with a GT target-state episode (1+ frame overlap)?
    n_tp_pred = 0
    for ps, pe, _ in pred_target_episodes:
        overlap = any(
            not (pe < gs or ps > ge)  # intervals overlap
            for gs, ge, _ in gt_target_episodes
        )
        if overlap:
            n_tp_pred += 1
    episode_precision = n_tp_pred / max(1, n_pred) if n_pred > 0 else float("nan")

    # False alarm episodes per 1000 frames
    n_fa = n_pred - n_tp_pred
    fa_per_1000 = n_fa / max(1, n) * 1000.0

    # Mean GT episode duration
    gt_durations = [e - s + 1 for s, e, _ in gt_target_episodes]
    mean_gt_duration = float(np.mean(gt_durations)) if gt_durations else float("nan")

    return {
        "n_gt_episodes": n_gt,
        "n_pred_episodes": n_pred,
        **{f"recall_at_{k}": recall_at[k] for k in RECALL_K_VALUES},
        "episode_precision": episode_precision,
        "mean_detection_delay": mean_delay,
        "median_detection_delay": median_delay,
        "fa_per_1000_frames": fa_per_1000,
        "mean_gt_episode_duration": mean_gt_duration,
        "n_frames": n,
    }


# ---------------------------------------------------------------------------
# Top-10 timeline examples
# ---------------------------------------------------------------------------


def _collect_timeline_examples(
    seq_key: str,
    gt_states: list[int],
    pred_states: list[int],
    target_state: int,
    max_examples: int = 10,
) -> list[dict]:
    """Collect representative episode timeline rows for this sequence."""
    n = min(len(gt_states), len(pred_states))
    gt_states = gt_states[:n]
    pred_states = pred_states[:n]

    gt_episodes = _extract_episodes(gt_states)
    gt_target_episodes = [(s, e, st) for s, e, st in gt_episodes if st == target_state]

    rows: list[dict] = []
    for start, end, _ in gt_target_episodes[:max_examples]:
        window = pred_states[start : start + 10]
        delay = None
        pred_state_at_start = pred_states[start] if start < len(pred_states) else -1
        for offset, ps in enumerate(window):
            if ps == target_state:
                delay = offset
                break
        rows.append({
            "seq_id": seq_key,
            "frame_start": start,
            "frame_end": end,
            "gt_state": DERIVED_STATE_NAMES.get(target_state, str(target_state)),
            "pred_state_at_start": DERIVED_STATE_NAMES.get(pred_state_at_start, str(pred_state_at_start)),
            "detection_delay": delay if delay is not None else -1,
        })
    return rows


# ---------------------------------------------------------------------------
# Aggregate across sequences
# ---------------------------------------------------------------------------


def _aggregate_metrics(
    per_seq: list[dict],
    target_state: int,
) -> dict:
    """Pool metrics across sequences."""
    if not per_seq:
        return {
            "target_state": DERIVED_STATE_NAMES.get(target_state, str(target_state)),
            "n_sequences": 0,
        }

    total_gt = sum(r["n_gt_episodes"] for r in per_seq)
    total_pred = sum(r["n_pred_episodes"] for r in per_seq)
    total_frames = sum(r["n_frames"] for r in per_seq)

    recall_at = {}
    for k in RECALL_K_VALUES:
        # Weight by n_gt_episodes
        weighted_sum = sum(
            r[f"recall_at_{k}"] * r["n_gt_episodes"]
            for r in per_seq
            if not math.isnan(r[f"recall_at_{k}"])
        )
        total_weight = sum(r["n_gt_episodes"] for r in per_seq if not math.isnan(r[f"recall_at_{k}"]))
        recall_at[k] = weighted_sum / max(1, total_weight) if total_weight > 0 else float("nan")

    precision_vals = [r["episode_precision"] for r in per_seq if not math.isnan(r["episode_precision"])]
    ep_precision = float(np.mean(precision_vals)) if precision_vals else float("nan")

    all_delays = [r["mean_detection_delay"] for r in per_seq if not math.isnan(r["mean_detection_delay"])]
    mean_delay = float(np.mean(all_delays)) if all_delays else float("nan")
    median_delay = float(np.median(all_delays)) if all_delays else float("nan")

    fa_vals = [r["fa_per_1000_frames"] for r in per_seq]
    fa_per_1000 = float(np.mean(fa_vals)) if fa_vals else float("nan")

    dur_vals = [r["mean_gt_episode_duration"] for r in per_seq if not math.isnan(r["mean_gt_episode_duration"])]
    mean_duration = float(np.mean(dur_vals)) if dur_vals else float("nan")

    result = {
        "target_state": DERIVED_STATE_NAMES.get(target_state, str(target_state)),
        "n_sequences": len(per_seq),
        "total_gt_episodes": total_gt,
        "total_pred_episodes": total_pred,
        "total_frames": total_frames,
        "episode_precision": round(ep_precision, 6) if not math.isnan(ep_precision) else None,
        "mean_detection_delay_frames": round(mean_delay, 3) if not math.isnan(mean_delay) else None,
        "median_detection_delay_frames": round(median_delay, 3) if not math.isnan(median_delay) else None,
        "fa_per_1000_frames": round(fa_per_1000, 3) if not math.isnan(fa_per_1000) else None,
        "mean_gt_episode_duration": round(mean_duration, 2) if not math.isnan(mean_duration) else None,
    }
    for k in RECALL_K_VALUES:
        rv = recall_at[k]
        result[f"recall_at_{k}"] = round(rv, 6) if not math.isnan(rv) else None
    return result


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_episode_metrics_md(metrics: list[dict], out_path: Path) -> None:
    lines = ["# Episode-Level CSC Metrics\n"]
    for m in metrics:
        state = m["target_state"]
        lines.append(f"## {state}\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Sequences | {m['n_sequences']} |")
        lines.append(f"| GT episodes | {m.get('total_gt_episodes', 'N/A')} |")
        lines.append(f"| Pred episodes | {m.get('total_pred_episodes', 'N/A')} |")
        for k in RECALL_K_VALUES:
            v = m.get(f"recall_at_{k}")
            lines.append(f"| Recall@{k} | {v if v is not None else 'N/A'} |")
        lines.append(f"| Episode Precision | {m.get('episode_precision', 'N/A')} |")
        lines.append(f"| Mean Detection Delay | {m.get('mean_detection_delay_frames', 'N/A')} frames |")
        lines.append(f"| Median Detection Delay | {m.get('median_detection_delay_frames', 'N/A')} frames |")
        lines.append(f"| FA per 1000 frames | {m.get('fa_per_1000_frames', 'N/A')} |")
        lines.append(f"| Mean GT episode duration | {m.get('mean_gt_episode_duration', 'N/A')} frames |")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


def _write_timeline_csv(examples: list[dict], out_path: Path) -> None:
    fieldnames = [
        "seq_id", "frame_start", "frame_end",
        "gt_state", "pred_state_at_start", "detection_delay",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(examples)
    print(f"Wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Episode-level evaluation of CSC derived-state predictions."
    )
    parser.add_argument(
        "--labels", required=True, type=Path,
        help="Directory with labels.jsonl files (searched recursively).",
    )
    parser.add_argument(
        "--predictions", required=True, type=Path,
        help="Directory with CSC prediction JSONLs (one per sequence, must have derived_state).",
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output directory for episode metrics.",
    )
    args = parser.parse_args()

    labels_dir: Path = args.labels
    predictions_dir: Path = args.predictions
    out_dir: Path = args.out

    if not labels_dir.exists():
        print(f"ERROR: labels dir {labels_dir} does not exist", file=sys.stderr)
        return 1

    # Load labels
    print(f"[evaluate_csc_episodes] Loading labels from {labels_dir}...", flush=True)
    label_groups = _collect_labels(labels_dir)
    if not label_groups:
        print("ERROR: no label data found", file=sys.stderr)
        return 1
    print(f"  {len(label_groups)} sequences loaded.", flush=True)

    # Load predictions (or fall back to labels as pseudo-predictions for sanity mode)
    pred_groups: dict[str, list[dict]] = {}
    if predictions_dir.exists():
        print(f"[evaluate_csc_episodes] Loading predictions from {predictions_dir}...", flush=True)
        pred_groups = _collect_predictions(predictions_dir)
        print(f"  {len(pred_groups)} prediction sequences loaded.", flush=True)
    else:
        print(
            f"WARNING: predictions dir {predictions_dir} does not exist — "
            "using GT labels as predictions (sanity mode, recall=1 expected).",
            flush=True,
        )
        pred_groups = label_groups  # sanity check mode

    # Per-sequence, per-state metrics
    all_metrics_by_state: dict[int, list[dict]] = {s: [] for s in TARGET_STATES}
    all_timeline_examples: list[dict] = []
    n_matched = 0

    for seq_key, gt_rows in sorted(label_groups.items()):
        gt_states = [int(r.get("derived_state", 0)) for r in gt_rows]

        pred_rows = pred_groups.get(seq_key)
        if pred_rows is None:
            print(f"  WARNING: no predictions for {seq_key} — skipping.", flush=True)
            continue

        pred_states = [int(r.get("derived_state", 0)) for r in pred_rows]
        n_matched += 1

        for target_state in TARGET_STATES:
            metrics = _compute_sequence_metrics(gt_states, pred_states, target_state)
            metrics["sequence"] = seq_key
            all_metrics_by_state[target_state].append(metrics)

            # Collect timeline examples for this state
            if metrics["n_gt_episodes"] > 0:
                examples = _collect_timeline_examples(
                    seq_key, gt_states, pred_states, target_state
                )
                all_timeline_examples.extend(examples)

    print(f"\n  Matched {n_matched}/{len(label_groups)} sequences.", flush=True)

    if n_matched == 0:
        print("ERROR: no sequences matched between labels and predictions.", file=sys.stderr)
        return 1

    # Aggregate
    aggregated: list[dict] = []
    for target_state in TARGET_STATES:
        agg = _aggregate_metrics(all_metrics_by_state[target_state], target_state)
        aggregated.append(agg)
        state_name = DERIVED_STATE_NAMES.get(target_state, str(target_state))
        print(f"\n  [{state_name}]", flush=True)
        print(f"    GT episodes:   {agg.get('total_gt_episodes', 0)}", flush=True)
        for k in RECALL_K_VALUES:
            print(f"    Recall@{k}:     {agg.get(f'recall_at_{k}')}", flush=True)
        print(f"    Precision:     {agg.get('episode_precision')}", flush=True)
        print(f"    Mean delay:    {agg.get('mean_detection_delay_frames')} frames", flush=True)
        print(f"    FA/1000:       {agg.get('fa_per_1000_frames')}", flush=True)

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "episode_metrics.json"
    with open(json_path, "w") as fh:
        json.dump(aggregated, fh, indent=2, default=str)
    print(f"\nWrote {json_path}", flush=True)

    md_path = out_dir / "episode_metrics.md"
    _write_episode_metrics_md(aggregated, md_path)
    print(f"Wrote {md_path}", flush=True)

    # Top-10 timeline examples across all target states (sort by delay ascending)
    timeline_sorted = sorted(
        [e for e in all_timeline_examples if e["detection_delay"] >= 0],
        key=lambda e: e["detection_delay"],
    )[:10]
    timeline_path = out_dir / "episode_timeline_examples.csv"
    _write_timeline_csv(timeline_sorted, timeline_path)

    print("\n[evaluate_csc_episodes] Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
