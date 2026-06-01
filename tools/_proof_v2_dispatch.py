"""PROOF: V2-trained CSC model fed V1 feature layout (the old runtime bug)
collapses predictions to CC. Same model, same telemetry rows — only the
feature builder differs (V1 slots 8/11/14/15 vs V2).

This isolates the train/inference feature-version mismatch from any
live-tracking variance.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csc_lib.csc.config import CSCTrainConfig
from csc_lib.csc.features import build_sequence_features, build_sequence_features_v2
from csc_lib.csc.model import build_model

CKPT = ROOT / "outputs/csc_training/sglatrack_run2_scalectx_tcn16_stage2/checkpoint_best.pth"
LABELS = ROOT / "outputs/csc_labels/sglatrack/v3fix_combined"
IMG = (1280, 720)
SCENES = [("dtb70", "BMX3"), ("uavdt_sot", "S1308"),
          ("lasot", "bird-10"), ("visdrone_sot", "uav0000180_00050_s")]
STATE = ["CC", "CU", "LA", "FC"]


def load_rows(ds, seq):
    rows = []
    for jl in LABELS.glob("*/labels.jsonl"):
        with jl.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("dataset") == ds and r.get("sequence") == seq:
                    rows.append(r)
    rows.sort(key=lambda r: r.get("frame_idx", 0))
    return rows


def run(model, feats):
    x = torch.from_numpy(feats).unsqueeze(0)
    with torch.inference_mode():
        out = model.predict(x, last_step_only=False)
    der = out["derived_probs"][0].cpu().numpy()  # (T, 4)
    return der.argmax(axis=1), der[:, 3]  # states, p_fc


def main():
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = CSCTrainConfig.from_dict(blob["config"])
    print(f"ckpt feature_version = {getattr(cfg.feature, 'feature_version', 'v1')}")
    model = build_model(cfg.model)
    model.load_state_dict(blob["state_dict"])
    model.eval()

    print(f"\n{'scene':<32} {'N':>5}  {'GT_FC':>6}  "
          f"{'V1bld_FC':>9} {'V1_pFCmax':>10}  {'V2bld_FC':>9} {'V2_pFCmax':>10}")
    print("-" * 96)
    for ds, seq in SCENES:
        rows = load_rows(ds, seq)
        if len(rows) < 16:
            print(f"{ds+'/'+seq:<32} (skip — {len(rows)} rows)")
            continue
        gt = np.array([r.get("derived_state", -1) for r in rows])
        gt_fc = int((gt == 3).sum())

        f_v1 = build_sequence_features(rows, IMG, cfg=cfg.feature)
        f_v2 = build_sequence_features_v2(rows, IMG, cfg=cfg.feature)
        s_v1, p_v1 = run(model, f_v1)
        s_v2, p_v2 = run(model, f_v2)

        print(f"{ds+'/'+seq:<32} {len(rows):>5}  {gt_fc:>6}  "
              f"{int((s_v1==3).sum()):>9} {p_v1.max():>10.4f}  "
              f"{int((s_v2==3).sum()):>9} {p_v2.max():>10.4f}")

    print("\nV1bld = old runtime layout (the bug). V2bld = correct (matches training).")
    print("If V2bld_FC >> V1bld_FC, the dispatch bug was suppressing all FC detections.")


if __name__ == "__main__":
    main()
