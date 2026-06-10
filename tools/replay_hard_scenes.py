"""Replay specific hard scenes side-by-side on two ckpts (R1 vs R2).

For each named (dataset, sequence):
  - Build features per ckpt's feature_version (V1 vs V2)
  - Predict full sequence
  - Print: n_CC/n_LA/n_FC, CC→FC false-alarm rate, FC/LA recall
  - Print: first 5 frames where pred=FC begins (frame_idx, true_state, pred,
    p_FC, log_area_ratio, scale_smoothness if V2)

Default scene list — covers FC-stuck, FC-chaotic, aerial CC-with-natural-growth,
and the worst R1 case visdrone_sot/uav0000184_00625_s.
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
    ("lasot",         "bird-10"),                 # FC-stuck classic
    ("lasot",         "drone-11"),                # large FC, FC-stuck
    ("lasot",         "car-13"),                  # FC-chaotic classic
    ("lasot",         "drone-19"),                # mixed CC + FC large
    ("uavdt_sot",     "S1308"),                   # aerial FC-stuck
    ("uavdt_sot",     "S0201"),                   # aerial mix large CC
    ("visdrone_sot",  "uav0000180_00050_s"),      # aerial FC-chaotic
    ("visdrone_sot",  "uav0000184_00625_s"),      # R1 worst case (CC→FC 44%)
    ("visdrone_sot",  "uav0000074_06312_s"),      # R1 also problematic
    ("dtb70",         "BMX3"),                    # clean dtb70
]

DEFAULT_IMAGE_SIZE = (1280, 720)
N_STATES = 4
STATE_NAMES = ["CC", "CU", "LA", "FC"]


def load_ckpt(ckpt_path: Path, device: str = "cpu"):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = CSCTrainConfig.from_dict(blob["config"])
    state_dict = blob["state_dict"]
    proj_w = state_dict.get("proj.0.weight", state_dict.get("input_proj.weight"))
    if proj_w is not None:
        cfg.model.feature_dim = int(proj_w.shape[1])
    model = build_model(cfg.model)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, cfg


def load_sequence_rows(labels_dir: Path, ds: str, seq: str):
    candidates = list(labels_dir.glob(f"*/labels.jsonl"))
    rows = []
    for jsonl in candidates:
        with jsonl.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("dataset") == ds and r.get("sequence") == seq:
                    rows.append(r)
    rows.sort(key=lambda r: r.get("frame_idx", 0))
    return rows


def predict(model, cfg, rows):
    fv = getattr(cfg.feature, "feature_version", "v1")
    if fv == "v2":
        feats = build_sequence_features_v2(rows, DEFAULT_IMAGE_SIZE, cfg=cfg.feature)
    else:
        feats = build_sequence_features(rows, DEFAULT_IMAGE_SIZE, cfg=cfg.feature)
    x = torch.from_numpy(feats).unsqueeze(0)
    with torch.inference_mode():
        out = model.predict(x, last_step_only=False)
    probs = out["derived_probs"][0].cpu().numpy()  # (T, 4)
    return probs, feats, fv


def confusion(true: np.ndarray, pred: np.ndarray):
    c = np.zeros((N_STATES, N_STATES), dtype=np.int64)
    valid = (true >= 0) & (true < N_STATES) & (pred >= 0) & (pred < N_STATES)
    for t, p in zip(true[valid], pred[valid]):
        c[t, p] += 1
    return c


def summarize_seq(name_tag, ds, seq, true, probs, feats, fv):
    pred = probs.argmax(-1)
    n = min(len(true), len(pred))
    true, pred = true[:n], pred[:n]
    c = confusion(true, pred)
    n_cc, n_cu, n_la, n_fc = c.sum(axis=1).tolist()
    cc_to_fc = c[0, 3] / max(n_cc, 1)
    cc_to_la = c[0, 2] / max(n_cc, 1)
    fc_recall = c[3, 3] / max(n_fc, 1) if n_fc else float("nan")
    la_recall = c[2, 2] / max(n_la, 1) if n_la else float("nan")
    cc_recall = c[0, 0] / max(n_cc, 1) if n_cc else float("nan")
    print(f"  [{name_tag} fv={fv}] n_fr={n}  n_CC={n_cc}  n_LA={n_la}  n_FC={n_fc}  "
          f"CC→FC={cc_to_fc:.4f}  CC→LA={cc_to_la:.4f}  "
          f"CC_rec={cc_recall:.3f}  FC_rec={fc_recall:.3f}  LA_rec={la_recall:.3f}")
    return c, pred


def show_fc_onset(name_tag, true, pred, probs, feats, fv, n_show=5):
    """Print first frames where prediction transitions to FC."""
    fc_idx = STATE_NAMES.index("FC")
    fc_pred_mask = pred == fc_idx
    onset_indices = []
    prev = False
    for i, b in enumerate(fc_pred_mask):
        if b and not prev:
            onset_indices.append(i)
        prev = b
    if not onset_indices:
        return
    print(f"    [{name_tag}] pred=FC onsets at frames {onset_indices[:n_show]}"
          + (f" (... +{len(onset_indices) - n_show} more)" if len(onset_indices) > n_show else ""))
    n = min(len(true), len(pred), len(probs))
    extra_cols = ""
    if fv == "v2":
        # V2 slots: 11=edge_pressure, 12=log_area, 14=scale_smooth, 15=aspect_inst
        extra_cols = "  edge_press  log_area  scale_sm  aspect_inst"
    else:
        # V1 slots: 10=edge_contact, 12=log_area
        extra_cols = "  edge_cont   log_area"
    print(f"      frame   true_pred  p_CC   p_LA   p_FC   {extra_cols.strip()}")
    for i in onset_indices[:n_show]:
        if i >= n:
            continue
        t, p = int(true[i]), int(pred[i])
        t_n = STATE_NAMES[t] if 0 <= t < N_STATES else "?"
        p_n = STATE_NAMES[p]
        ps = probs[i]
        if fv == "v2":
            ep = feats[i, 11]
            la = feats[i, 12]
            ss = feats[i, 14]
            ai = feats[i, 15]
            extra = f"  {ep:9.4f}  {la:8.3f}  {ss:8.4f}  {ai:9.4f}"
        else:
            ec = feats[i, 10]
            la = feats[i, 12]
            extra = f"  {ec:9.4f}  {la:8.3f}"
        print(f"      {i:>5}   {t_n}→{p_n:<4}  {ps[0]:.3f}  {ps[2]:.3f}  {ps[3]:.3f}{extra}")


def replay_one_scene(ds, seq, ckpt_r1, ckpt_r2, cfg_r1, cfg_r2, model_r1, model_r2, labels_dir):
    print(f"\n{'='*112}")
    print(f"SCENE: {ds}/{seq}")
    print("=" * 112)
    rows = load_sequence_rows(labels_dir, ds, seq)
    if not rows:
        print(f"  (no rows found)")
        return
    if len(rows) < 16:
        print(f"  (too short: {len(rows)} frames)")
        return
    true = np.array([r.get("derived_state", -1) for r in rows], dtype=np.int32)
    n_cc_true = int((true == 0).sum())
    n_fc_true = int((true == 3).sum())
    print(f"  true labels: n_frames={len(rows)}  n_CC={n_cc_true}  n_FC={n_fc_true}")

    probs1, feats1, fv1 = predict(model_r1, cfg_r1, rows)
    probs2, feats2, fv2 = predict(model_r2, cfg_r2, rows)
    c1, pred1 = summarize_seq("R1", ds, seq, true, probs1, feats1, fv1)
    c2, pred2 = summarize_seq("R2", ds, seq, true, probs2, feats2, fv2)

    show_fc_onset("R1", true, pred1, probs1, feats1, fv1)
    show_fc_onset("R2", true, pred2, probs2, feats2, fv2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-r1", type=Path, required=True)
    ap.add_argument("--ckpt-r2", type=Path, required=True)
    ap.add_argument("--labels-dir", type=Path,
                    default=Path("outputs/csc_labels/sglatrack/v3fix_combined"))
    ap.add_argument("--scenes", type=str, default=None,
                    help="Comma-separated list of dataset:sequence pairs. "
                         "Default uses 10 hard scenes from breakdown analysis.")
    args = ap.parse_args()

    if args.scenes:
        scenes = [tuple(s.split(":", 1)) for s in args.scenes.split(",")]
    else:
        scenes = DEFAULT_SCENES

    print(f"[load] R1 ckpt={args.ckpt_r1}")
    model_r1, cfg_r1 = load_ckpt(args.ckpt_r1)
    print(f"[load] R2 ckpt={args.ckpt_r2}")
    model_r2, cfg_r2 = load_ckpt(args.ckpt_r2)

    print(f"\n[run] {len(scenes)} hard scenes")
    for ds, seq in scenes:
        replay_one_scene(ds, seq, args.ckpt_r1, args.ckpt_r2,
                         cfg_r1, cfg_r2, model_r1, model_r2,
                         args.labels_dir)


if __name__ == "__main__":
    main()
