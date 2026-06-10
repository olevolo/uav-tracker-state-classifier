#!/usr/bin/env python
"""Diagnostic: identity verifier discrimination on car7 (SGLATrack v2, ΔAUC -0.59).

For the car7 FC-switch frame, score sim_to_init at THREE locations:
  (a) GT bbox (where the true target IS)
  (b) wrong-lock incumbent (tracker prediction at FC frame; the wrong car)
  (c) RT-DETR candidate position (whatever the recover policy switched to)

If (a) >> (b)/(c): verifier could work; we just need a denser search.
If (a) ~ (b) ~ (c): verifier is structurally blind — switching to DINOv2/CLIP
or a learned ReID head is required.

Usage:
  PYTHONPATH=src:csc_uav_tracking_sdk/src:. .venv/bin/python \\
    tools/identity_verifier_diagnostic.py
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    sys.path.insert(0, str(_p))

import csc_uav_tracking  # noqa: F401  registers DATASETS
from csc_uav_tracking.registry import DATASETS
from uav_tracker.registry import TRACKERS
from uav_tracker.types import BBox


SEQ_NAME = "car7"
SWITCH_FRAME = 192
RUN_DIR = PROJECT_ROOT / "outputs/fc_recover_v1/full_uav123_v2/sglatrack_uav123_test_checkpoint_best"


def _read_predictions(path: Path) -> list[tuple[float, float, float, float]]:
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            out.append((0.0, 0.0, 0.0, 0.0))
            continue
        parts = [p for p in line.replace("\t", ",").split(",") if p]
        try:
            vals = [float(p) for p in parts[:4]]
        except ValueError:
            vals = [0.0, 0.0, 0.0, 0.0]
        if len(vals) < 4:
            vals = vals + [0.0] * (4 - len(vals))
        out.append(tuple(vals))
    return out


def _gt_bbox(seq, frame_idx: int) -> BBox:
    bb = seq.ground_truth[frame_idx]
    return BBox(x=float(bb.x), y=float(bb.y), w=float(bb.w), h=float(bb.h))


def main() -> int:
    importlib.import_module("uav_tracker.trackers.sglatrack")
    tracker = TRACKERS.build("sglatrack", device="mps")

    seq = next(s for s in DATASETS.build("uav123", split="test") if s.name == SEQ_NAME)
    # seq.frames may be a lazy iterable; materialise to list of np.ndarray.
    frames = list(seq.frames)
    print(f"sequence: {seq.name}  frames={len(frames)}  switch_frame={SWITCH_FRAME}")

    predictions = _read_predictions(RUN_DIR / "predictions" / f"{SEQ_NAME}.txt")
    incumbent_bbox_xywh = predictions[SWITCH_FRAME]
    incumbent = BBox(*[float(v) for v in incumbent_bbox_xywh])
    gt = _gt_bbox(seq, SWITCH_FRAME)
    gt_init = _gt_bbox(seq, 0)

    # Init tracker on frame 0 with GT (canonical setup).
    tracker.init(frames[0], gt_init)
    # Step tracker through frames 1..SWITCH_FRAME so embedding cache is current.
    for t in range(1, SWITCH_FRAME + 1):
        tracker.update(frames[t])

    print(f"\nframe {SWITCH_FRAME} bboxes:")
    print(f"  GT incumbent: x={gt.x:.1f} y={gt.y:.1f} w={gt.w:.1f} h={gt.h:.1f}")
    print(f"  predicted  : x={incumbent.x:.1f} y={incumbent.y:.1f} "
          f"w={incumbent.w:.1f} h={incumbent.h:.1f}  (the wrong car)")

    # IoU(prediction, GT) at the switch frame — sanity check for "wrong-lock".
    def _iou(a: BBox, b: BBox) -> float:
        x1 = max(a.x, b.x); y1 = max(a.y, b.y)
        x2 = min(a.x + a.w, b.x + b.w); y2 = min(a.y + a.h, b.y + b.h)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = a.w * a.h + b.w * b.h - inter
        return float(inter / union) if union > 0 else 0.0

    iou_pred_gt = _iou(incumbent, gt)
    print(f"  IoU(predicted, GT) = {iou_pred_gt:.3f}  "
          f"(<0.2 == truly FC; >0.5 == not really wrong-locked)")

    # Score sim_to_init at the GT location: ask tracker to redetect with GT as
    # anchor. With factor=4 it crops a small window around GT centre, runs the
    # template/search forward, and reports per-candidate sim_to_init relative to
    # the FROZEN frame-0 template embedding.
    frame = frames[SWITCH_FRAME]
    print("\nidentity scores (sim_to_init = cosine to frozen frame-0 template):")
    for label, anchor in [("GT (true target)", gt),
                          ("predicted (wrong-lock)", incumbent)]:
        cands = tracker.redetect(
            frame,
            factors=(4.0,),
            anchor_bboxes=[anchor],
            include_current=False,
            grid_size=0,
            max_candidates=3,
            top_k=3,
            rank_by="quality",
        )
        if cands is None:
            print(f"  {label:<28}: no candidates returned")
            continue
        if isinstance(cands, dict):
            cands = [cands]
        for i, c in enumerate(cands[:3]):
            sim = c.get("sim_to_init", float("nan"))
            score = c.get("score", float("nan"))
            ratio = c.get("score_ratio", float("nan"))
            cx, cy = c.get("center", (float("nan"), float("nan")))
            print(f"  {label:<28} #{i}: sim_to_init={sim:+.4f}  "
                  f"score={score:.3f}  ratio={ratio:.3f}  centre=({cx:.0f},{cy:.0f})")

    # Also: explicit full-frame redetect to grab the candidate the recover policy
    # would have chosen (top SGLA-internal candidate at the switch frame).
    print()
    cands_full = tracker.redetect(
        frame, factors=(8.0, 12.0, 16.0),
        grid_size=3, max_candidates=5, top_k=5,
        rank_by="quality",
    )
    if isinstance(cands_full, dict):
        cands_full = [cands_full]
    print(f"top SGLA-internal candidates (factors=8/12/16, grid=3, top_k=5):")
    for i, c in enumerate(cands_full[:5]):
        sim = c.get("sim_to_init", float("nan"))
        score = c.get("score", float("nan"))
        ratio = c.get("score_ratio", float("nan"))
        cx, cy = c.get("center", (float("nan"), float("nan")))
        # IoU vs GT for context.
        cw = c.get("bbox", [0, 0, 0, 0])[2]
        ch = c.get("bbox", [0, 0, 0, 0])[3]
        cand_bbox = BBox(x=cx - cw / 2, y=cy - ch / 2, w=cw, h=ch)
        iou_gt = _iou(cand_bbox, gt)
        print(f"  #{i}: sim_to_init={sim:+.4f}  score={score:.3f}  "
              f"ratio={ratio:.3f}  centre=({cx:.0f},{cy:.0f})  IoU(GT)={iou_gt:.3f}")

    print()
    print("interpretation:")
    print("  GT-vs-wrong sim_to_init close (within ~0.05) => verifier blind.")
    print("  GT >> wrong (gap >0.10) => verifier OK; need denser candidate pool.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
