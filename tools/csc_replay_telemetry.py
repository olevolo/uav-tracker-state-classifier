#!/usr/bin/env python3
"""Replay saved tracker telemetry through a trained CSC checkpoint — no tracker re-run.

Reads baseline predictions (bbox) + telemetry (confidence/apce/psr) that were
saved during the baseline pass, feeds them frame-by-frame through the CSC
runtime (production path, with calibrators), and compares the predicted
derived-state against the GT state labels.

This is a CPU-light smoke check (single-thread, no tracker forward pass) so it
can run alongside training without contention.

Usage:
    .venv/bin/python tools/csc_replay_telemetry.py \
        --checkpoint outputs/csc_training/sglatrack_v3fix_tcn16_stage1/checkpoint_best.pth \
        --sequences bird1_2 car1_s
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from csc_lib.csc.inference import load_runtime
from csc_lib.csc.labeling.label_schema import DerivedState
from csc_lib.csc.labeling.weak_labeler import label_frame, LabelingThresholds

ROOT = Path(__file__).resolve().parent.parent
DERIVED_NAMES = {0: "CC", 1: "CU", 2: "LA", 3: "FC"}
FC_IDX = int(DerivedState.FALSE_CONFIRMED)  # 3

BASE_DIR = ROOT / "outputs/baselines/sglatrack/uav123/test"
GT_DIR = ROOT / "outputs/eval_v2/sglatrack/uav123/test/labels/uav123/test/labels_per_sequence"
IMG_ROOT = Path.home() / "uav-tracker-data/uav123/UAV123/data_seq/UAV123"
CALIB_DIR = ROOT / "outputs/calibration"

_SUFFIX_RE = re.compile(r"_(s|\d+)$")


def base_seq_name(seq: str) -> str:
    """car1_s -> car1, bird1_2 -> bird1, group2_1 -> group2, car13 -> car13."""
    return _SUFFIX_RE.sub("", seq)


def image_size_for(seq: str) -> tuple[int, int]:
    base = base_seq_name(seq)
    folder = IMG_ROOT / base
    if folder.is_dir():
        imgs = sorted(folder.glob("*.jpg")) or sorted(folder.glob("*.png"))
        if imgs:
            with Image.open(imgs[0]) as im:
                return im.size  # (W, H)
    return (1280, 720)


def load_predictions(seq: str) -> list[tuple[float, float, float, float]]:
    p = BASE_DIR / "predictions" / f"{seq}.txt"
    boxes = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[,\s]+", line)
            boxes.append(tuple(float(x) for x in parts[:4]))
    return boxes


def load_telemetry(seq: str) -> dict[int, dict]:
    p = BASE_DIR / "telemetry" / f"{seq}.jsonl"
    tel = {}
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            tel[int(r["frame_idx"])] = r
    return tel


def load_gt(seq: str, relabel: bool = False) -> dict[int, int]:
    """Load GT derived_state per frame.

    relabel=False: use the stored derived_state from eval_v2 (may be the OLD
                   OR-gate label set — inflates FC).
    relabel=True:  recompute derived_state with the current strict-AND rule
                   (weak_labeler.label_frame) from the stored iou/confidence.
    """
    p = GT_DIR / f"{seq}.jsonl"
    gt = {}
    if not relabel:
        with p.open() as f:
            for line in f:
                r = json.loads(line)
                s = r.get("derived_state")
                if s is not None:
                    gt[int(r["frame_idx"])] = int(s)
        return gt

    th = LabelingThresholds()
    consec_low = 0
    with p.open() as f:
        for line in f:
            r = json.loads(line)
            iou = r.get("iou")
            if iou is not None and iou < th.tau_lost_iou:
                consec_low += 1
            else:
                consec_low = 0
            aux = r.get("aux") or {}
            _, _, derived, _, _, _ = label_frame(
                iou=iou,
                confidence=r.get("confidence"),  # calibrated; tau_high=0.65
                full_occlusion=bool(aux.get("occlusion", False)),
                out_of_view=bool(aux.get("out_of_view", False)),
                absent=bool(r.get("absent", False)),
                consecutive_low_iou=consec_low,
                apce=r.get("apce"),
                psr=r.get("psr"),
                thresholds=th,
            )
            gt[int(r["frame_idx"])] = int(derived)
    return gt


def replay_sequence(runtime, seq: str, relabel: bool = False) -> dict:
    img_size = image_size_for(seq)
    runtime.reset(image_size=img_size)

    preds = load_predictions(seq)
    tel = load_telemetry(seq)
    gt = load_gt(seq, relabel=relabel)

    n = len(preds)
    pred_states = np.full(n, -1, dtype=int)
    risk = np.zeros(n, dtype=float)

    for i in range(n):
        t = tel.get(i, {})
        out = runtime.step(
            confidence=t.get("confidence"),
            apce=t.get("apce"),
            psr=t.get("psr"),
            pred_bbox=preds[i],
        )
        pred_states[i] = out.derived_state
        risk[i] = out.risk_score

    # Confusion (4x4) over frames with GT
    conf = np.zeros((4, 4), dtype=int)
    for i, g in gt.items():
        if 0 <= i < n and pred_states[i] >= 0:
            conf[g, pred_states[i]] += 1

    n_gt = int(conf.sum())
    # Per-class metrics
    metrics = {}
    for c in range(4):
        tp = conf[c, c]
        fn = conf[c, :].sum() - tp
        fp = conf[:, c].sum() - tp
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec and rec and not np.isnan(prec) and not np.isnan(rec) and (prec + rec)) else float("nan")
        metrics[DERIVED_NAMES[c]] = dict(tp=int(tp), fn=int(fn), fp=int(fp), recall=rec, precision=prec, f1=f1)

    pred_dist = {DERIVED_NAMES[c]: int((pred_states == c).sum()) for c in range(4)}
    gt_dist = {DERIVED_NAMES[c]: int(conf[c, :].sum()) for c in range(4)}
    fcr_pred = pred_dist["FC"] / n if n else 0.0

    return dict(
        seq=seq, img_size=img_size, n_frames=n, n_gt=n_gt,
        pred_dist=pred_dist, gt_dist=gt_dist, fcr_pred=fcr_pred,
        confusion=conf.tolist(), metrics=metrics, mean_risk=float(risk.mean()),
    )


def fmt_report(r: dict) -> str:
    lines = []
    lines.append(f"\n{'='*64}")
    lines.append(f"SEQ: {r['seq']}  ({r['img_size'][0]}x{r['img_size'][1]}, {r['n_frames']} frames, {r['n_gt']} GT-labeled)")
    lines.append(f"{'='*64}")
    lines.append(f"  GT dist:   " + "  ".join(f"{k}={v}" for k, v in r["gt_dist"].items()))
    lines.append(f"  Pred dist: " + "  ".join(f"{k}={v}" for k, v in r["pred_dist"].items()))
    lines.append(f"  FCR (pred): {r['fcr_pred']*100:.2f}%   mean risk: {r['mean_risk']:.3f}")
    lines.append("")
    lines.append(f"  {'class':<6}{'recall':>9}{'prec':>9}{'f1':>9}{'tp':>7}{'fn':>7}{'fp':>7}")
    for c in ("CC", "CU", "LA", "FC"):
        m = r["metrics"][c]
        def g(x): return f"{x:.3f}" if isinstance(x, float) and not np.isnan(x) else "  -  "
        lines.append(f"  {c:<6}{g(m['recall']):>9}{g(m['precision']):>9}{g(m['f1']):>9}{m['tp']:>7}{m['fn']:>7}{m['fp']:>7}")
    lines.append("")
    lines.append(f"  Confusion (rows=GT, cols=Pred) [CC CU LA FC]:")
    for c in range(4):
        lines.append(f"    {DERIVED_NAMES[c]}: {r['confusion'][c]}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "outputs/csc_training/sglatrack_v3fix_tcn16_stage1/checkpoint_best.pth"))
    ap.add_argument("--sequences", nargs="+", default=["bird1_2", "car1_s"])
    ap.add_argument("--calibrator", default="sglatrack_all_v2")
    ap.add_argument("--relabel", action="store_true",
                    help="Recompute GT with current strict-AND rule (eval_v2 labels are stale OR-gate)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"

    # epoch info
    import torch
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    epoch = raw.get("epoch", "?") if isinstance(raw, dict) else "?"
    feat_ver = raw.get("config", {}).get("feature", {}).get("feature_version", "?") if isinstance(raw, dict) else "?"

    print(f"Checkpoint: {ckpt}")
    print(f"  epoch={epoch}  feature_version={feat_ver}  calibrator={args.calibrator}")
    print(f"  GT label mode: {'strict-AND RELABEL' if args.relabel else 'stored eval_v2 (may be stale OR-gate)'}")

    runtime = load_runtime(
        ckpt,
        device="cpu",
        calibration_dir=CALIB_DIR,
        tracker_name=args.calibrator,
    )

    reports = [replay_sequence(runtime, s, relabel=args.relabel) for s in args.sequences]
    for r in reports:
        print(fmt_report(r))

    # aggregate FC
    print(f"\n{'='*64}")
    print("AGGREGATE (FC class):")
    tot_tp = sum(r["metrics"]["FC"]["tp"] for r in reports)
    tot_fn = sum(r["metrics"]["FC"]["fn"] for r in reports)
    tot_fp = sum(r["metrics"]["FC"]["fp"] for r in reports)
    rec = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else float("nan")
    prec = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else float("nan")
    print(f"  FC recall={rec:.3f}  precision={prec:.3f}  (tp={tot_tp} fn={tot_fn} fp={tot_fp})")


if __name__ == "__main__":
    main()
