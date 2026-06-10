#!/usr/bin/env python3
"""GENERAL telemetry-replay: feed a tracker's SAVED baseline telemetry through a
CSC checkpoint and write states/<seq>.jsonl in the live run_with_csc format —
NO tracker re-run.

Why this is paper-grade (not merely "exploratory"): in PASSIVE mode the CSC never
modifies tracker behaviour, so the tracker's per-frame output is *identical* to its
baseline run. Replaying the baseline telemetry through the exact same CSCRuntime
therefore reproduces the passive-diagnosis states. (Validate once against a known
live row with tools/confusion_uav123.py before trusting on new trackers/datasets.)

Generalises tools/replay_csc_states_uav123.py to ANY (tracker, dataset):
  reads   outputs/baselines/<tracker>/<dataset>/test/{predictions,telemetry}
  writes  outputs/eval/<tracker>/<dataset>/test/<run_tag>/states/<seq>.jsonl

Always passes the full telemetry row via step(extra=...), so it works for BOTH
v2 (16-dim) and v3 (23-dim, response-structure) checkpoints — the v2 builder
ignores `extra`, the v3 builder consumes the sm_*/response_entropy fields.

Single-thread, CPU-light (no tracker forward pass) so it can run beside training.
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
torch.set_num_threads(1)

from csc_lib.csc.inference import load_runtime

CALIB_DIR = ROOT / "outputs/calibration"
# UAV123 images; uav123_10fps is the SAME frames (strided) so the same root + the
# same per-sequence (W,H) apply.
IMG_ROOT = Path.home() / "uav-tracker-data/uav123/UAV123/data_seq/UAV123"
_SUFFIX_RE = re.compile(r"_(s|\d+)$")


def base_seq_name(seq: str) -> str:
    """car1_s -> car1, bird1_2 -> bird1, group2_1 -> group2, car13 -> car13."""
    return _SUFFIX_RE.sub("", seq)


def image_size_for(seq: str) -> tuple[int, int]:
    folder = IMG_ROOT / base_seq_name(seq)
    if folder.is_dir():
        imgs = sorted(folder.glob("*.jpg")) or sorted(folder.glob("*.png"))
        if imgs:
            with Image.open(imgs[0]) as im:
                return im.size  # (W, H)
    return (1280, 720)


def load_predictions(path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"[,\s]+", line)
            boxes.append(tuple(float(x) for x in parts[:4]))
    return boxes


def load_telemetry(path: Path) -> dict[int, dict]:
    tel = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            tel[int(r["frame_idx"])] = r
    return tel


def replay_one(runtime, seq: str, pred_path: Path, tel_path: Path) -> list[dict]:
    runtime.reset(image_size=image_size_for(seq))
    preds = load_predictions(pred_path)
    tel = load_telemetry(tel_path)
    rows = []
    for i in range(len(preds)):
        if i == 0:
            rows.append({"frame_idx": 0, "init": True})
            continue
        t = tel.get(i, {})
        out = runtime.step(
            confidence=t.get("confidence"),
            apce=t.get("apce"),
            psr=t.get("psr"),
            pred_bbox=preds[i],
            extra=t,  # v3 builder consumes sm_*/response_entropy; v2 ignores it
        )
        lp = np.asarray(out.localization_probs, dtype=float).ravel()
        cp = np.asarray(out.confidence_probs, dtype=float).ravel()
        dp = np.asarray(out.derived_probs, dtype=float).ravel()
        rows.append({
            "frame_idx": i,
            "localization_probs": lp.tolist(),
            "confidence_probs": cp.tolist(),
            "derived_probs": dp.tolist(),
            "predicted_localization": int(lp.argmax()),
            "predicted_confidence": int(cp.argmax()),
            "derived_state": int(out.derived_state),
            "false_confirmed_flag": bool(out.false_confirmed_flag),
            "risk_score": float(out.risk_score),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--dataset", default="uav123")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--calibrator", required=True,
                    help="calibrator prefix, e.g. sglatrack_all_v2 / ostrack_aerial_v2")
    ap.add_argument("--run_tag", required=True,
                    help="output under outputs/eval/<tracker>/<dataset>/test/<run_tag>/states")
    ap.add_argument("--baselines_root", default=str(ROOT / "outputs/baselines"))
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    feat_ver = raw.get("config", {}).get("feature", {}).get("feature_version", "?") if isinstance(raw, dict) else "?"

    base = Path(args.baselines_root) / args.tracker / args.dataset / "test"
    tel_dir = base / "telemetry"
    pred_dir = base / "predictions"
    assert tel_dir.is_dir(), f"missing baseline telemetry: {tel_dir}"

    print(f"[replay] {args.tracker}/{args.dataset} ckpt={ckpt.name} feat={feat_ver} "
          f"calib={args.calibrator} run_tag={args.run_tag}", flush=True)

    runtime = load_runtime(ckpt, device="cpu", calibration_dir=CALIB_DIR,
                           tracker_name=args.calibrator)

    out_dir = ROOT / "outputs/eval" / args.tracker / args.dataset / "test" / args.run_tag / "states"
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = sorted(p.stem for p in tel_dir.glob("*.jsonl"))
    print(f"  {len(seqs)} sequences -> {out_dir}", flush=True)
    n_frames = 0
    for k, seq in enumerate(seqs, 1):
        pf = pred_dir / f"{seq}.txt"
        tf = tel_dir / f"{seq}.jsonl"
        if not pf.is_file():
            print(f"  [skip] {seq}: no predictions", flush=True)
            continue
        rows = replay_one(runtime, seq, pf, tf)
        with open(out_dir / f"{seq}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_frames += len(rows)
        if k % 25 == 0 or k == len(seqs):
            print(f"  [{k}/{len(seqs)}] {seq} ({len(rows)} fr) total={n_frames}", flush=True)
    print(f"[replay DONE] {n_frames} frames -> {out_dir.parent}", flush=True)


if __name__ == "__main__":
    main()
