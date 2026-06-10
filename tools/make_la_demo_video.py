"""Render an MP4 visualisation of one tracker run on a UAV123 sequence.

Reads the per-frame trace produced by ``run_with_csc.py`` (predictions/,
telemetry/, states/) plus the source UAV123 frames + GT and writes an
annotated MP4 with:

  - GT bbox (thin white)
  - Predicted bbox, colour-coded by CSC state (CC=green / CU=yellow /
    LA=orange / FC=red)
  - Top-left badge: sequence name, frame index, CSC state, IoU
  - Bottom-right gauges: confidence, APCE, last_cosine_sim,
    sm_local_top2_ratio, response_entropy
  - Top-right title (the --label string, e.g. "BASELINE" / "MOTION_BRIDGE")

Usage::

    python tools/make_la_demo_video.py \
        --run_dir outputs/.../passive/.../ \
        --sequence group3_2 \
        --label BASELINE \
        --out demo/group3_2_baseline.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_tracker.viz.video import write_mp4  # noqa: E402

STATE_NAMES = {0: "CC", 1: "CU", 2: "LA", 3: "FC"}
STATE_COLOURS = {
    0: (0, 220, 0),       # green   — confirmed
    1: (0, 220, 220),     # yellow  — uncertain
    2: (0, 140, 255),     # orange  — lost-aware
    3: (0, 0, 255),       # red     — false-confirmed
    -1: (180, 180, 180),  # grey    — init / unknown
}

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def load_predictions(p: Path) -> np.ndarray:
    """Load predictions/<seq>.txt as (N, 4) xywh array."""
    rows = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[,\s]+", line)
            rows.append([float(x) for x in parts[:4]])
    return np.asarray(rows, dtype=np.float64)


def load_jsonl(p: Path) -> list[dict]:
    out = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_gt(p: Path) -> list[Optional[tuple[float, float, float, float]]]:
    """Parse UAV123 anno; NaN row → None."""
    out: list[Optional[tuple[float, float, float, float]]] = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[,\s]+", line)
            try:
                vals = [float(x) for x in parts[:4]]
            except ValueError:
                out.append(None)
                continue
            if any(v != v for v in vals):
                out.append(None)
            else:
                out.append((vals[0], vals[1], vals[2], vals[3]))
    return out


def parse_config_seqs(config_path: Path) -> dict[str, tuple[str, int, int]]:
    out: dict[str, tuple[str, int, int]] = {}
    if not config_path.exists():
        return out
    content = config_path.read_text()
    pattern = (
        r"struct\('name','(\w+)','path','.*?\\(\w+)\\','startFrame',(\d+),"
        r"'endFrame',(\d+)"
    )
    for m in re.finditer(pattern, content):
        out[m.group(1)] = (m.group(2), int(m.group(3)), int(m.group(4)))
    return out


# --------------------------------------------------------------------------- #
# IoU
# --------------------------------------------------------------------------- #


def iou_xywh(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #


def draw_bbox(img: np.ndarray, box, colour, thickness: int = 2) -> None:
    if box is None:
        return
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return
    x0 = int(round(x))
    y0 = int(round(y))
    x1 = int(round(x + w))
    y1 = int(round(y + h))
    cv2.rectangle(img, (x0, y0), (x1, y1), colour, thickness, cv2.LINE_AA)


def draw_translucent_box(img, p0, p1, alpha: float = 0.55) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, p0, p1, BLACK, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def put_text(
    img,
    text: str,
    org: tuple[int, int],
    scale: float = 0.55,
    colour=WHITE,
    thickness: int = 1,
) -> tuple[int, int]:
    """Put text and return its (width, height)."""
    cv2.putText(img, text, org, FONT, scale, colour, thickness, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return tw, th


def draw_top_badge(img, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    """Render a stacked top-left badge."""
    pad_x, pad_y = 8, 6
    line_h = 22
    scale = 0.6
    thickness = 1
    # measure max width
    max_w = 0
    for text, _ in lines:
        (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
        max_w = max(max_w, tw)
    box_w = pad_x * 2 + max_w
    box_h = pad_y * 2 + line_h * len(lines)
    draw_translucent_box(img, (0, 0), (box_w, box_h), alpha=0.55)
    y = pad_y + 16
    for text, colour in lines:
        cv2.putText(img, text, (pad_x, y), FONT, scale, colour, thickness, cv2.LINE_AA)
        y += line_h


def draw_title(img, label: str, colour=WHITE) -> None:
    """Top-right big label."""
    pad_x, pad_y = 10, 8
    scale = 0.9
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, FONT, scale, thickness)
    h, w = img.shape[:2]
    x1 = w - pad_x
    x0 = x1 - tw - pad_x
    y0 = 0
    y1 = pad_y * 2 + th
    draw_translucent_box(img, (x0 - 4, y0), (x1 + 4, y1), alpha=0.55)
    cv2.putText(img, label, (x0, pad_y + th), FONT, scale, colour, thickness, cv2.LINE_AA)


def draw_gauges(
    img,
    items: list[tuple[str, float, float]],
    state_colour=(0, 220, 0),
) -> None:
    """Bottom-right vertical gauge stack.

    Each item is (label, value, max_value_for_full_bar).
    """
    pad = 8
    bar_w = 140
    bar_h = 8
    line_h = 24
    scale = 0.5
    thickness = 1
    n = len(items)
    box_w = bar_w + pad * 2 + 90  # 90 for value text
    box_h = pad * 2 + line_h * n
    h, w = img.shape[:2]
    x0 = w - box_w - 6
    y0 = h - box_h - 6
    x1 = x0 + box_w
    y1 = y0 + box_h
    draw_translucent_box(img, (x0, y0), (x1, y1), alpha=0.55)

    cy = y0 + pad + 14
    for label, value, vmax in items:
        # label
        cv2.putText(img, label, (x0 + pad, cy), FONT, scale, WHITE, thickness, cv2.LINE_AA)
        # value text
        if value != value:  # nan
            vtxt = "  --"
        elif abs(value) >= 100:
            vtxt = f"{value:7.1f}"
        else:
            vtxt = f"{value:7.3f}"
        cv2.putText(
            img, vtxt,
            (x0 + pad + 84, cy),
            FONT, scale, WHITE, thickness, cv2.LINE_AA,
        )
        # bar
        bar_y0 = cy + 4
        bar_y1 = bar_y0 + bar_h
        bar_x0 = x0 + pad
        bar_x1 = bar_x0 + bar_w
        cv2.rectangle(img, (bar_x0, bar_y0), (bar_x1, bar_y1), (60, 60, 60), -1)
        if value == value and vmax > 0:
            frac = float(np.clip(value / vmax, 0.0, 1.0))
            fill_w = int(round(frac * bar_w))
            if fill_w > 0:
                cv2.rectangle(
                    img, (bar_x0, bar_y0), (bar_x0 + fill_w, bar_y1),
                    state_colour, -1,
                )
        cy += line_h


def draw_legend(img) -> None:
    """Bottom-left mini legend: GT / pred."""
    pad = 8
    scale = 0.5
    h, w = img.shape[:2]
    items = [
        ("GT", WHITE, 1, "thin"),
        ("pred (state-coloured)", (0, 220, 0), 2, "thick"),
    ]
    line_h = 22
    box_w = 240
    box_h = pad * 2 + line_h * len(items)
    x0 = 6
    y0 = h - box_h - 6
    x1 = x0 + box_w
    y1 = y0 + box_h
    draw_translucent_box(img, (x0, y0), (x1, y1), alpha=0.55)
    cy = y0 + pad + 14
    for text, colour, thick, _ in items:
        cv2.line(img, (x0 + pad, cy - 4), (x0 + pad + 22, cy - 4), colour, thick)
        cv2.putText(
            img, text,
            (x0 + pad + 32, cy),
            FONT, scale, WHITE, 1, cv2.LINE_AA,
        )
        cy += line_h


# --------------------------------------------------------------------------- #
# Main render
# --------------------------------------------------------------------------- #


def render_video(
    run_dir: Path,
    sequence: str,
    out_path: Path,
    uav123_root: Path,
    label: str,
    fps: int = 30,
    downscale: float = 1.0,
    start: int = 0,
    end: Optional[int] = None,
) -> None:
    pred_path = run_dir / "predictions" / f"{sequence}.txt"
    tel_path = run_dir / "telemetry" / f"{sequence}.jsonl"
    state_path = run_dir / "states" / f"{sequence}.jsonl"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    preds = load_predictions(pred_path)
    tel = load_jsonl(tel_path) if tel_path.exists() else []
    states = load_jsonl(state_path) if state_path.exists() else []

    anno_path = uav123_root / "anno" / "UAV123" / f"{sequence}.txt"
    gt = load_gt(anno_path)

    cfg = parse_config_seqs(uav123_root / "configSeqs.m")
    if sequence not in cfg:
        raise KeyError(f"{sequence} not in configSeqs.m")
    folder, start_frame, end_frame = cfg[sequence]
    frame_dir = uav123_root / "data_seq" / "UAV123" / folder
    n_frames_seq = end_frame - start_frame + 1

    n = min(len(preds), len(gt), n_frames_seq, len(states) if states else 10**9)
    if end is not None:
        n = min(n, end)
    if start < 0:
        start = 0

    # Build telemetry / state lookup by frame_idx
    tel_by_idx: dict[int, dict] = {r.get("frame_idx", -1): r for r in tel}
    state_by_idx: dict[int, dict] = {r.get("frame_idx", -1): r for r in states}

    # iterator that yields rendered BGR frames
    def gen():
        for i in range(start, n):
            fnum = start_frame + i
            img_path = frame_dir / f"{fnum:06d}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                # placeholder
                img = np.zeros((360, 640, 3), dtype=np.uint8)

            if downscale != 1.0:
                img = cv2.resize(
                    img, None, fx=downscale, fy=downscale,
                    interpolation=cv2.INTER_AREA,
                )

            # scale boxes
            scale_box = downscale
            pred_box = preds[i].copy()
            pred_box *= scale_box
            gt_box = gt[i]
            if gt_box is not None:
                gt_box = tuple(v * scale_box for v in gt_box)

            st = state_by_idx.get(i, {})
            if st.get("init", False):
                state_id = -1
            else:
                state_id = int(st.get("derived_state", -1))
            colour = STATE_COLOURS.get(state_id, STATE_COLOURS[-1])

            # GT first (so pred draws on top)
            draw_bbox(img, gt_box, WHITE, thickness=1)
            draw_bbox(img, tuple(pred_box), colour, thickness=2)

            iou = iou_xywh(tuple(pred_box), gt_box)
            iou_txt = f"{iou:.2f}" if iou == iou else "  --"

            state_name = STATE_NAMES.get(state_id, "init") if state_id >= 0 else "init"
            badge_lines = [
                (f"{sequence}   frame {i+1}/{n}", WHITE),
                (f"state: {state_name}", colour),
                (f"IoU: {iou_txt}", WHITE),
            ]
            draw_top_badge(img, badge_lines)
            draw_title(img, label, WHITE)
            draw_legend(img)

            t = tel_by_idx.get(i, {})
            conf = float(t.get("confidence", float("nan")))
            apce = float(t.get("apce", float("nan")))
            cosine = float(t.get("last_cosine_sim", t.get("cosine_sim", float("nan"))))
            sm_top1 = float(t.get("sm_top1", float("nan")))
            sm_t2r = float(t.get("sm_local_top2_ratio", float("nan")))
            r_entropy = float(t.get("response_entropy", float("nan")))

            gauges = [
                ("conf",          conf,      1.0),
                ("APCE",          apce,      400.0),
                ("cosine",        cosine,    1.0),
                ("sm_top1",       sm_top1,   1.0),
                ("sm_top2_ratio", sm_t2r,    1.0),
                ("resp_entropy",  r_entropy, 6.0),
            ]
            draw_gauges(img, gauges, state_colour=colour)

            yield img

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(gen(), out_path, fps=fps)
    print(f"[ok] wrote {out_path}  ({n - start} frames @ {fps} fps)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=Path, required=True,
                   help="Directory with predictions/, telemetry/, states/")
    p.add_argument("--sequence", required=True, help="e.g. group3_2")
    p.add_argument("--out", type=Path, required=True, help="Output MP4 path")
    p.add_argument("--label", default="",
                   help="Big top-right label, e.g. BASELINE")
    p.add_argument(
        "--uav123_root",
        type=Path,
        default=Path(os.path.expanduser(
            "~/uav-tracker-data/UAV123/UAV123",
        )),
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--downscale", type=float, default=1.0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    render_video(
        run_dir=args.run_dir,
        sequence=args.sequence,
        out_path=args.out,
        uav123_root=args.uav123_root,
        label=args.label or args.run_dir.name,
        fps=args.fps,
        downscale=args.downscale,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
