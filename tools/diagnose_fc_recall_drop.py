"""Diagnose where R2 misses true FC frames — is it threshold issue or broken?

For each scene with low R2 FC_recall:
  1. Run R1 (V1) and R2 (V2) on the sequence
  2. Compute p_FC per frame (sigmoid of derived FC head)
  3. On TRUE FC frames only:
     - print p_FC distribution (mean, median, q05/q95)
     - threshold sweep: what threshold yields what FC recall?
     - cross-tab: at threshold matching 80% recall, what's CC→FC?

If R2 has p_FC ≈ 0.0 on true FC → genuinely broken
If R2 has p_FC ≈ 0.3-0.45 on true FC → close to threshold; full training will fix
If R2 has bimodal p_FC (some low, some high) → mixed; training + threshold tune helps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from csc_lib.csc.config import CSCTrainConfig  # noqa: E402
from csc_lib.csc.features import (  # noqa: E402
    build_sequence_features,
    build_sequence_features_v2,
)
from csc_lib.csc.model import build_model  # noqa: E402

DEFAULT_SCENES = [
    ("lasot",         "bird-10"),
    ("uavdt_sot",     "S0201"),
    ("visdrone_sot",  "uav0000180_00050_s"),
    ("lasot",         "car-13"),
    ("lasot",         "drone-19"),
]
N_STATES = 4
DEFAULT_IMAGE_SIZE = (1280, 720)


def load_ckpt(ckpt_path: Path):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = CSCTrainConfig.from_dict(blob["config"])
    sd = blob["state_dict"]
    proj_w = sd.get("proj.0.weight", sd.get("input_proj.weight"))
    if proj_w is not None:
        cfg.model.feature_dim = int(proj_w.shape[1])
    model = build_model(cfg.model)
    model.load_state_dict(sd)
    model.eval()
    return model, cfg


def load_seq_rows(labels_dir: Path, ds: str, seq: str):
    rows = []
    for jsonl in labels_dir.glob("*/labels.jsonl"):
        with jsonl.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("dataset") == ds and r.get("sequence") == seq:
                    rows.append(r)
    rows.sort(key=lambda r: r.get("frame_idx", 0))
    return rows


def predict_probs(model, cfg, rows):
    fv = getattr(cfg.feature, "feature_version", "v1")
    builder = build_sequence_features_v2 if fv == "v2" else build_sequence_features
    feats = builder(rows, DEFAULT_IMAGE_SIZE, cfg=cfg.feature)
    x = torch.from_numpy(feats).unsqueeze(0)
    with torch.inference_mode():
        out = model.predict(x, last_step_only=False)
    probs = out["derived_probs"][0].cpu().numpy()
    return probs, fv


def threshold_sweep(p_fc: np.ndarray, true_is_fc: np.ndarray, true_is_cc: np.ndarray):
    """Compute (threshold, fc_recall, cc_to_fc) over a grid."""
    rows = []
    for thr in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        pred_fc = p_fc >= thr
        n_fc = max(true_is_fc.sum(), 1)
        n_cc = max(true_is_cc.sum(), 1)
        fc_rec = (pred_fc & true_is_fc).sum() / n_fc
        cc_to_fc = (pred_fc & true_is_cc).sum() / n_cc
        rows.append({"thr": thr, "fc_recall": float(fc_rec), "cc_to_fc": float(cc_to_fc),
                     "fc_pred_total": int(pred_fc.sum())})
    return rows


def diagnose_scene(ds, seq, model_r1, cfg_r1, model_r2, cfg_r2, labels_dir):
    print(f"\n{'='*100}")
    print(f"SCENE: {ds}/{seq}")
    print("=" * 100)
    rows = load_seq_rows(labels_dir, ds, seq)
    if not rows or len(rows) < 16:
        print("  (skip — too few frames)")
        return
    true = np.array([r.get("derived_state", -1) for r in rows], dtype=np.int32)
    n_fc = int((true == 3).sum())
    n_cc = int((true == 0).sum())
    print(f"  n_frames={len(rows)}  n_CC={n_cc}  n_FC={n_fc}")

    if n_fc == 0:
        print("  no FC frames in this seq — skip diagnosis")
        return

    probs1, fv1 = predict_probs(model_r1, cfg_r1, rows)
    probs2, fv2 = predict_probs(model_r2, cfg_r2, rows)
    n = min(len(true), len(probs1), len(probs2))
    true = true[:n]
    p_fc1 = probs1[:n, 3]
    p_fc2 = probs2[:n, 3]
    is_fc = true == 3
    is_cc = true == 0

    # Distribution of p_FC on TRUE FC frames
    print(f"\n  p_FC distribution on TRUE FC frames (n={int(is_fc.sum())}):")
    for tag, p in [("R1 V1", p_fc1), ("R2 V2", p_fc2)]:
        v = p[is_fc]
        print(f"    {tag:<7}  mean={v.mean():.3f}  med={np.median(v):.3f}  "
              f"q05={np.quantile(v, 0.05):.3f}  q25={np.quantile(v, 0.25):.3f}  "
              f"q75={np.quantile(v, 0.75):.3f}  q95={np.quantile(v, 0.95):.3f}  "
              f">=0.5: {(v>=0.5).sum()}/{len(v)}")

    # Distribution of p_FC on TRUE CC frames (false-alarm side)
    print(f"\n  p_FC distribution on TRUE CC frames (n={int(is_cc.sum())}):")
    for tag, p in [("R1 V1", p_fc1), ("R2 V2", p_fc2)]:
        v = p[is_cc]
        print(f"    {tag:<7}  mean={v.mean():.3f}  med={np.median(v):.3f}  "
              f"q95={np.quantile(v, 0.95):.3f}  q99={np.quantile(v, 0.99):.3f}  "
              f">=0.5: {(v>=0.5).sum()}/{len(v)}  >=0.3: {(v>=0.3).sum()}/{len(v)}")

    # Threshold sweep
    print(f"\n  Threshold sweep — FC_recall vs CC→FC false-alarm:")
    print(f"    {'thr':<7}{'R1_rec':>10}{'R1_cc→fc':>11}{'R2_rec':>10}{'R2_cc→fc':>11}")
    sw1 = threshold_sweep(p_fc1, is_fc, is_cc)
    sw2 = threshold_sweep(p_fc2, is_fc, is_cc)
    for r1, r2 in zip(sw1, sw2):
        print(f"    {r1['thr']:<7.2f}{r1['fc_recall']:>10.3f}{r1['cc_to_fc']:>11.4f}"
              f"{r2['fc_recall']:>10.3f}{r2['cc_to_fc']:>11.4f}")

    # Find threshold for matched FC recall (target: R1 default argmax recall)
    pred1 = probs1[:n].argmax(-1)
    r1_recall = ((pred1 == 3) & is_fc).sum() / max(is_fc.sum(), 1)
    print(f"\n  R1 default argmax FC recall: {r1_recall:.3f}")
    target = r1_recall
    # For R2: find lowest threshold where recall >= target
    matched = None
    for r in sw2:
        if r["fc_recall"] >= target * 0.9:  # within 10% of R1
            matched = r
            break
    if matched:
        print(f"  R2 to match (within 10% of) R1 recall {target:.3f}: "
              f"thr={matched['thr']:.2f} → R2 recall={matched['fc_recall']:.3f}, "
              f"R2 CC→FC={matched['cc_to_fc']:.4f}")
    else:
        print(f"  R2 cannot reach 90% of R1 recall {target:.3f} at any tested threshold "
              f"(max R2 recall over grid: {max(r['fc_recall'] for r in sw2):.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-r1", type=Path, required=True)
    ap.add_argument("--ckpt-r2", type=Path, required=True)
    ap.add_argument("--labels-dir", type=Path,
                    default=Path("outputs/csc_labels/sglatrack/v3fix_combined"))
    ap.add_argument("--scenes", default=None)
    args = ap.parse_args()

    if args.scenes:
        scenes = [tuple(s.split(":", 1)) for s in args.scenes.split(",")]
    else:
        scenes = DEFAULT_SCENES

    print(f"[load] R1 ckpt={args.ckpt_r1}")
    model_r1, cfg_r1 = load_ckpt(args.ckpt_r1)
    print(f"[load] R2 ckpt={args.ckpt_r2}")
    model_r2, cfg_r2 = load_ckpt(args.ckpt_r2)

    for ds, seq in scenes:
        diagnose_scene(ds, seq, model_r1, cfg_r1, model_r2, cfg_r2, args.labels_dir)


if __name__ == "__main__":
    main()
