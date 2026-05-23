#!/usr/bin/env python
"""Manual audit visualizer for CSC predictions vs labels.

For each sampled sequence, picks frames_per_seq *hardest* frames (lowest
IoU among LOST / FALSE_CONFIRMED frames, falling back to all frames if none
exist).  Composites an n_seqs × frames_per_seq PNG grid with:
  - frame image (or blank if not available on disk)
  - GT bbox (green)
  - predicted bbox (red)
  - state label caption

Usage
-----
    python tools/audit_visualizer.py \
        --predictions outputs/baselines/sglatrack/got10k/val/predictions \
        --labels outputs/csc_labels/got10k \
        --dataset got10k \
        --split val \
        --n_seqs 20 \
        --frames_per_seq 6 \
        --hardest \
        --out outputs/audit/got10k_val.png

Optional flags:
    --data_root   override UAV_DATA_ROOT env var
    --seq_names   comma-separated list of specific sequences to include
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

# matplotlib import — suppress backend warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data helpers
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


def _find_labels_for_seq(labels_dir: Path, dataset: str, split: str, seq_name: str) -> list[dict]:
    """Find label rows for a sequence.  Handles flat and nested layouts."""
    # Try: labels_dir/<dataset>/<split>/labels_per_sequence/<seq>.jsonl
    candidate = labels_dir / dataset / split / "labels_per_sequence" / f"{seq_name}.jsonl"
    if candidate.exists():
        return _load_jsonl(candidate)

    # Try: labels_dir/<dataset>/<split>/<seq>.jsonl
    candidate2 = labels_dir / dataset / split / f"{seq_name}.jsonl"
    if candidate2.exists():
        return _load_jsonl(candidate2)

    # Try: labels_dir/<split>/labels_per_sequence/<seq>.jsonl
    candidate3 = labels_dir / split / "labels_per_sequence" / f"{seq_name}.jsonl"
    if candidate3.exists():
        return _load_jsonl(candidate3)

    # Try: labels_dir/labels_per_sequence/<seq>.jsonl
    candidate4 = labels_dir / "labels_per_sequence" / f"{seq_name}.jsonl"
    if candidate4.exists():
        return _load_jsonl(candidate4)

    # Recursive rglob search as last resort
    for jf in labels_dir.rglob(f"{seq_name}.jsonl"):
        return _load_jsonl(jf)

    return []


def _load_predictions(pred_file: Path) -> list[tuple[float, float, float, float] | None]:
    """Load xywh predictions from a text file (one bbox per line)."""
    bboxes: list[tuple[float, float, float, float] | None] = []
    with open(pred_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                bboxes.append(None)
                continue
            parts = line.replace(",", " ").split()
            if len(parts) >= 4:
                try:
                    x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    bboxes.append((x, y, w, h))
                except ValueError:
                    bboxes.append(None)
            else:
                bboxes.append(None)
    return bboxes


def _iou(b1: tuple, b2: tuple) -> float:
    """xywh IoU."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Image + overlay
# ---------------------------------------------------------------------------


def _find_frame_image(data_root: Path, dataset: str, split: str, seq_name: str, frame_idx: int) -> Path | None:
    """Try to locate frame image on disk."""
    # GOT-10k style: data_root/GOT_10k/<split>/<seq>/<frame+1:08d>.jpg
    frame_num = frame_idx + 1

    got_style = data_root / "GOT_10k" / split / seq_name / f"{frame_num:08d}.jpg"
    if got_style.exists():
        return got_style

    # UAV123 style: data_root/uav123/data_seq/<seq>/img<frame+1:05d>.jpg
    for ext in ["jpg", "png", "jpeg"]:
        for stem in [f"img{frame_num:05d}", f"{frame_num:05d}", f"{frame_num:06d}", f"{frame_num:08d}"]:
            for base_dir in [
                data_root / "uav123" / "data_seq" / seq_name,
                data_root / "UAV123" / "data_seq" / seq_name,
                data_root / dataset / split / seq_name,
                data_root / dataset / seq_name,
                data_root / seq_name,
            ]:
                p = base_dir / f"{stem}.{ext}"
                if p.exists():
                    return p

    # DTB70 / VisDrone style: data_root/<dataset>/<seq>/img<frame:05d>.jpg
    for base in [data_root / "DTB70" / seq_name / "img",
                 data_root / "VisDrone-SOT" / seq_name / "img",
                 data_root / "LaSOT" / seq_name / "img"]:
        for ext in ["jpg", "png"]:
            p = base.parent / f"img{frame_num:05d}.{ext}"
            if p.exists():
                return p

    return None


def _draw_bbox(img: np.ndarray, bbox: tuple | None, color: tuple, thickness: int = 2) -> np.ndarray:
    """Draw xywh bbox on an image (in-place, returns img)."""
    if bbox is None:
        return img
    x, y, w, h = [int(v) for v in bbox]
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img - 1, x + w), min(h_img - 1, y + h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def _make_cell(
    frame_path: Path | None,
    gt_bbox: tuple | None,
    pred_bbox: tuple | None,
    caption: str,
    cell_w: int = 200,
    cell_h: int = 160,
) -> np.ndarray:
    """Build a single composite cell: image + overlaid bboxes + caption bar."""
    cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    if frame_path is not None and frame_path.exists():
        img = cv2.imread(str(frame_path))
        if img is not None:
            img = cv2.resize(img, (cell_w, cell_h - 20))
            cell[:cell_h - 20] = img

    # Draw bboxes on the image portion
    img_area = cell[:cell_h - 20]
    if gt_bbox is not None:
        oh, ow = img_area.shape[:2]
        # Scale bbox to cell size — assume original image dims unknown; use raw
        _draw_bbox(img_area, gt_bbox, color=(0, 200, 0), thickness=2)  # green GT
    if pred_bbox is not None:
        _draw_bbox(img_area, pred_bbox, color=(0, 0, 255), thickness=2)  # red pred

    # Caption bar
    caption_bar = cell[cell_h - 20:]
    caption_bar[:] = (30, 30, 30)
    cv2.putText(
        caption_bar,
        caption[:28],
        (2, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return cell


# ---------------------------------------------------------------------------
# Sequence discovery
# ---------------------------------------------------------------------------


def _list_sequences(pred_dir: Path, labels_dir: Path, dataset: str, split: str) -> list[str]:
    """List sequence names from prediction files."""
    seqs = []
    for txt in sorted(pred_dir.glob("*.txt")):
        seqs.append(txt.stem)
    if not seqs:
        # Fall back to label files
        for jf in labels_dir.rglob("*.jsonl"):
            seqs.append(jf.stem)
    return sorted(set(seqs))


# ---------------------------------------------------------------------------
# Hard frame selection
# ---------------------------------------------------------------------------


def _select_hardest_frames(
    rows: list[dict],
    pred_bboxes: list[tuple | None],
    k: int,
) -> list[int]:
    """Select k hardest frames: LOST/FALSE_CONFIRMED frames with lowest IoU.

    Fallback: if fewer than k such frames exist, also include frames with
    the lowest IoU overall (per-frame from labels).  If no IoU data,
    samples uniformly.
    """
    # Collect (frame_idx, iou, is_failure)
    scored: list[tuple[float, bool, int]] = []
    for r in rows:
        fidx = r.get("frame_idx", 0)
        iou_val = r.get("iou", 1.0) or 1.0
        # Detect failure state — new schema or old schema
        is_failure = False
        if "localization_state" in r:
            loc = r.get("localization_state", 0)
            is_failure = loc >= 2  # LOST
        elif "derived_state" in r:
            derived = r.get("derived_state", 0)
            is_failure = derived in (2, 3)  # LOST_AWARE or FALSE_CONFIRMED
        elif "state" in r:
            state = r.get("state", 0)
            is_failure = state >= 3  # LOST / DISTRACTOR / FALSE_CONFIRMED in old schema
        scored.append((float(iou_val), is_failure, fidx))

    # Sort failures first (lowest IoU), then other hard frames
    failures = sorted(
        [(iou, fidx) for iou, is_failure, fidx in scored if is_failure],
        key=lambda x: x[0],
    )
    others = sorted(
        [(iou, fidx) for iou, is_failure, fidx in scored if not is_failure],
        key=lambda x: x[0],
    )

    selected = [fidx for _, fidx in failures[:k]]
    if len(selected) < k:
        extra_needed = k - len(selected)
        selected += [fidx for _, fidx in others[:extra_needed]]

    # If still not enough, sample uniformly
    if len(selected) < k and rows:
        all_idxs = [r.get("frame_idx", i) for i, r in enumerate(rows)]
        step = max(1, len(all_idxs) // k)
        selected += all_idxs[::step]

    # Deduplicate while preserving order
    seen: set[int] = set()
    out: list[int] = []
    for fidx in selected:
        if fidx not in seen:
            seen.add(fidx)
            out.append(fidx)
    return out[:k]


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------


def visualize(
    pred_dir: Path,
    labels_dir: Path,
    dataset: str,
    split: str,
    n_seqs: int,
    frames_per_seq: int,
    hardest: bool,
    out_path: Path,
    data_root: Path,
    seq_names: list[str] | None = None,
) -> int:
    """Build the audit PNG.  Returns number of panels rendered."""
    if seq_names:
        all_seqs = seq_names
    else:
        all_seqs = _list_sequences(pred_dir, labels_dir, dataset, split)

    if not all_seqs:
        print(f"[viz] ERROR: no sequences found in {pred_dir}", flush=True)
        return 0

    # Sample sequences
    import random
    random.seed(0)
    chosen = all_seqs if len(all_seqs) <= n_seqs else random.sample(all_seqs, n_seqs)
    chosen = sorted(chosen)

    cell_w, cell_h = 200, 170
    n_cols = frames_per_seq
    n_rows = len(chosen)
    canvas_w = n_cols * cell_w + (n_cols + 1) * 2
    canvas_h = n_rows * cell_h + n_rows * 20  # extra 20px per row for seq label

    # Build a large blank canvas
    canvas = np.full((canvas_h, canvas_w, 3), 15, dtype=np.uint8)

    panels_rendered = 0
    for row_idx, seq_name in enumerate(chosen):
        # Load labels
        label_rows = _find_labels_for_seq(labels_dir, dataset, split, seq_name)
        if not label_rows:
            print(f"[viz] WARNING: no labels for {seq_name}, skipping", flush=True)
            continue

        # Load predictions
        pred_file = pred_dir / f"{seq_name}.txt"
        pred_bboxes: list[tuple | None] = []
        if pred_file.exists():
            pred_bboxes = _load_predictions(pred_file)

        # Build frame_idx → (label_row, pred_bbox) map
        row_map: dict[int, dict] = {r.get("frame_idx", i): r for i, r in enumerate(label_rows)}
        pred_map: dict[int, tuple | None] = {}
        for i, pb in enumerate(pred_bboxes):
            pred_map[i] = pb

        # Select hard frames
        frame_idxs = _select_hardest_frames(label_rows, pred_bboxes, k=frames_per_seq)

        # Sequence label row (dark bar with seq name)
        y_seq_top = row_idx * (cell_h + 20)
        seq_bar = canvas[y_seq_top : y_seq_top + 20, :]
        seq_bar[:] = (40, 40, 80)
        cv2.putText(
            seq_bar,
            f"{seq_name}",
            (4, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )

        # Per-frame state summary for print
        state_counts: dict[str, int] = {}

        for col_idx, fidx in enumerate(frame_idxs[:n_cols]):
            row_data = row_map.get(fidx, {})
            gt_bbox = row_data.get("gt_bbox")
            if gt_bbox and len(gt_bbox) >= 4:
                gt_bbox = tuple(float(v) for v in gt_bbox[:4])
            else:
                gt_bbox = None

            pred_bbox = pred_map.get(fidx) or row_data.get("pred_bbox")
            if pred_bbox and len(pred_bbox) >= 4:
                pred_bbox = tuple(float(v) for v in pred_bbox[:4])
            else:
                pred_bbox = None

            iou_val = row_data.get("iou", 0.0) or 0.0

            # Determine state label for caption
            if "localization_state_name" in row_data:
                state_label = row_data["localization_state_name"]
                derived = row_data.get("derived_state_name", "")
                if derived:
                    state_label = derived[:10]
            elif "state_name" in row_data:
                state_label = row_data["state_name"]
            elif "localization_state" in row_data:
                ls_names = ["STABLE", "UNCERTAIN", "LOST"]
                ls_id = row_data.get("localization_state", 0)
                state_label = ls_names[ls_id] if 0 <= ls_id < len(ls_names) else f"loc{ls_id}"
            else:
                state_label = "?"

            state_counts[state_label] = state_counts.get(state_label, 0) + 1

            caption = f"f{fidx} IoU={iou_val:.2f} {state_label[:8]}"

            # Find frame image
            frame_path = _find_frame_image(data_root, dataset, split, seq_name, fidx)

            # Build cell
            cell = _make_cell(frame_path, gt_bbox, pred_bbox, caption, cell_w, cell_h)

            # Place cell on canvas
            y0 = y_seq_top + 20
            x0 = col_idx * cell_w + (col_idx + 1) * 2
            canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = cell
            panels_rendered += 1

        print(
            f"[viz] {seq_name}: {len(frame_idxs)} frames, state dist: {state_counts}",
            flush=True,
        )

    if panels_rendered == 0:
        print("[viz] ERROR: no panels rendered", flush=True)
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert BGR to RGB for matplotlib
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(canvas_w / 72, canvas_h / 72), dpi=72)
    ax.imshow(canvas_rgb)
    ax.axis("off")

    # Legend
    legend_handles = [
        mpatches.Patch(color="green", label="GT bbox"),
        mpatches.Patch(color="red", label="Pred bbox"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=6,
              framealpha=0.7, ncol=2)

    fig.tight_layout(pad=0)
    fig.savefig(str(out_path), dpi=72, bbox_inches="tight")
    plt.close(fig)

    print(f"[viz] PNG written: {out_path} ({panels_rendered} panels)", flush=True)
    return panels_rendered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="CSC manual audit visualizer")
    parser.add_argument("--predictions", required=True, type=Path,
                        help="Directory containing <seq>.txt prediction files")
    parser.add_argument("--labels", required=True, type=Path,
                        help="Root directory for CSC labels (*.jsonl)")
    parser.add_argument("--dataset", required=True, type=str,
                        help="Dataset name (e.g. got10k, uav123)")
    parser.add_argument("--split", required=True, type=str,
                        help="Split name (e.g. val, test)")
    parser.add_argument("--n_seqs", type=int, default=20,
                        help="Number of sequences to sample (default 20)")
    parser.add_argument("--frames_per_seq", type=int, default=6,
                        help="Number of frames per sequence (default 6)")
    parser.add_argument("--hardest", action="store_true", default=True,
                        help="Select hardest (lowest IoU / failure) frames (default True)")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output PNG path")
    parser.add_argument("--data_root", type=Path,
                        default=Path(os.environ.get("UAV_DATA_ROOT", str(Path.home() / "uav-tracker-data"))),
                        help="Root of dataset images (default: UAV_DATA_ROOT env or ~/uav-tracker-data)")
    parser.add_argument("--seq_names", type=str, default=None,
                        help="Comma-separated list of specific sequence names to visualize")
    args = parser.parse_args()

    seq_names = [s.strip() for s in args.seq_names.split(",")] if args.seq_names else None

    n_panels = visualize(
        pred_dir=args.predictions,
        labels_dir=args.labels,
        dataset=args.dataset,
        split=args.split,
        n_seqs=args.n_seqs,
        frames_per_seq=args.frames_per_seq,
        hardest=args.hardest,
        out_path=args.out,
        data_root=args.data_root,
        seq_names=seq_names,
    )

    sys.exit(0 if n_panels > 0 else 1)


if __name__ == "__main__":
    main()
