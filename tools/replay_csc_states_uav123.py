#!/usr/bin/env python3
"""Replay saved SGLATrack UAV123 telemetry through a CSC checkpoint and WRITE
states/<seq>.jsonl in the live run_with_csc format — no tracker re-run.

EXPLORATORY: lets us score R2 / R25 / R3-fcw4 on UAV123 with the *same* metric
code as the selected R3-fcw3, without re-running the tracker. Single-thread so it
does not contend with training. GT for any downstream confusion is the fixed
labels_v3 stored derived_state (checkpoint-independent), so all checkpoints are
compared apples-to-apples.

Telemetry/prediction source: outputs/baselines/sglatrack/uav123/test/ (shared by
every checkpoint). The same checkpoint replayed here vs its live run is a
fidelity control.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
torch.set_num_threads(1)

from csc_lib.csc.inference import load_runtime
from tools.csc_replay_telemetry import (
    load_predictions, load_telemetry, image_size_for, BASE_DIR, CALIB_DIR,
)


def replay_one(runtime, seq: str) -> list[dict]:
    runtime.reset(image_size=image_size_for(seq))
    preds = load_predictions(seq)
    tel = load_telemetry(seq)
    n = len(preds)
    rows = []
    for i in range(n):
        if i == 0:
            rows.append({"frame_idx": 0, "init": True})
            continue
        t = tel.get(i, {})
        out = runtime.step(
            confidence=t.get("confidence"),
            apce=t.get("apce"),
            psr=t.get("psr"),
            pred_bbox=preds[i],
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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--calibrator", default="sglatrack_all_v2")
    ap.add_argument("--run_tag", required=True,
                    help="output dir under outputs/eval/sglatrack/uav123/test/<run_tag>")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    feat_ver = raw.get("config", {}).get("feature", {}).get("feature_version", "?") if isinstance(raw, dict) else "?"
    print(f"[replay] {ckpt}\n  feature_version={feat_ver} calibrator={args.calibrator} run_tag={args.run_tag}")

    runtime = load_runtime(ckpt, device="cpu", calibration_dir=CALIB_DIR, tracker_name=args.calibrator)

    out_dir = ROOT / "outputs/eval/sglatrack/uav123/test" / args.run_tag / "states"
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = sorted(p.stem for p in (BASE_DIR / "telemetry").glob("*.jsonl"))
    print(f"  {len(seqs)} sequences -> {out_dir}")
    n_frames = 0
    for k, seq in enumerate(seqs, 1):
        rows = replay_one(runtime, seq)
        with open(out_dir / f"{seq}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_frames += len(rows)
        if k % 25 == 0 or k == len(seqs):
            print(f"  [{k}/{len(seqs)}] {seq} ({len(rows)} fr) total={n_frames}")
    print(f"[replay DONE] {n_frames} frames written")


if __name__ == "__main__":
    main()
