"""Per-sequence eval on smoke checkpoints — sanity check before full training.

For each val sequence:
  1. Build V1 (Run 1) or V2 (Run 2) features per ckpt config.
  2. Forward pass through model.
  3. Compute per-sequence 4×4 confusion (CC/CU/LA/FC).

Output: top-K simple sequences (low CC→FC false alarm) and top-K complex
(high CC→FC) for each ckpt; aggregate per-dataset confusion.

Usage:
  python tools/eval_per_sequence_smoke.py \\
      --ckpt /tmp/smoke_logs/t6_run1_s1/checkpoint_best.pth \\
      --labels-dir outputs/csc_labels/sglatrack/v3fix_combined \\
      --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from csc_lib.csc.config import CSCTrainConfig  # noqa: E402
from csc_lib.csc.features import (  # noqa: E402
    build_sequence_features,
    build_sequence_features_v2,
)
from csc_lib.csc.labeling.label_schema import DerivedState  # noqa: E402
from csc_lib.csc.model import build_model  # noqa: E402

AERIAL_DATASETS = {"dtb70", "uavdt_sot", "visdrone_sot", "uavtrack112"}
N_STATES = 4
STATE_NAMES = ["CC", "CU", "LA", "FC"]
DEFAULT_IMAGE_SIZE = (1280, 720)


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


def load_val_split(ckpt_dir: Path) -> set[tuple[str, str]]:
    info = json.loads((ckpt_dir / "split_info.json").read_text())
    return {(ds, seq) for ds, seq in info["val_sequences"]}


def load_sequences(labels_dir: Path):
    seq_rows = defaultdict(list)
    for sub in sorted(labels_dir.iterdir()):
        jsonl = sub / "labels.jsonl"
        if not jsonl.exists():
            continue
        with jsonl.open() as f:
            for line in f:
                r = json.loads(line)
                key = (r.get("dataset", sub.name), r.get("sequence", "?"))
                seq_rows[key].append(r)
    return seq_rows


def predict_sequence(model, cfg, rows, image_size, device):
    feature_version = getattr(cfg.feature, "feature_version", "v1")
    if feature_version == "v2":
        feats = build_sequence_features_v2(rows, image_size, cfg=cfg.feature)
    else:
        feats = build_sequence_features(rows, image_size, cfg=cfg.feature)
    x = torch.from_numpy(feats).unsqueeze(0).to(device)
    with torch.inference_mode():
        out = model.predict(x, last_step_only=False)
    pred = out["derived_probs"][0].argmax(-1).cpu().numpy()
    return pred


def per_seq_confusion(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    conf = np.zeros((N_STATES, N_STATES), dtype=np.int64)
    valid = (true >= 0) & (true < N_STATES) & (pred >= 0) & (pred < N_STATES)
    for t, p in zip(true[valid], pred[valid]):
        conf[t, p] += 1
    return conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--labels-dir", type=Path,
                    default=Path("outputs/csc_labels/sglatrack/v3fix_combined"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out-json", type=Path,
                    default=None,
                    help="Optional output JSON path (default: <ckpt-dir>/per_sequence_eval.json)")
    args = ap.parse_args()

    ckpt_path = args.ckpt
    ckpt_dir = ckpt_path.parent
    out_json = args.out_json or (ckpt_dir / "per_sequence_eval.json")

    print(f"[load] ckpt={ckpt_path}")
    model, cfg = load_ckpt(ckpt_path)
    feature_version = getattr(cfg.feature, "feature_version", "v1")
    print(f"[ckpt] feature_dim={cfg.model.feature_dim}  feature_version={feature_version}")

    val_set = load_val_split(ckpt_dir)
    print(f"[split] {len(val_set)} val sequences")

    print(f"[load] sequences from {args.labels_dir}")
    all_rows = load_sequences(args.labels_dir)

    seq_results = []
    aggregate_conf = np.zeros((N_STATES, N_STATES), dtype=np.int64)
    per_dataset_conf = defaultdict(lambda: np.zeros((N_STATES, N_STATES), dtype=np.int64))

    val_pairs = sorted([k for k in all_rows if k in val_set])
    for ds, seq in tqdm(val_pairs, desc="seq"):
        rows = all_rows[(ds, seq)]
        rows.sort(key=lambda r: r.get("frame_idx", 0))
        if len(rows) < 16:
            continue
        # Use per-row image_size if present, else default. (Fall back consistent with builder.)
        img_size = DEFAULT_IMAGE_SIZE
        for r in rows:
            if "image_size" in r and isinstance(r["image_size"], (list, tuple)) and len(r["image_size"]) == 2:
                img_size = (int(r["image_size"][0]), int(r["image_size"][1]))
                break

        true = np.array([r.get("derived_state", -1) for r in rows], dtype=np.int32)
        pred = predict_sequence(model, cfg, rows, img_size, device="cpu")
        # Builder returns features for full sequence; pred is same length as features
        # If lengths mismatch (some builders emit T-1 due to diff features), align by min
        n = min(len(true), len(pred))
        true, pred = true[:n], pred[:n]
        conf = per_seq_confusion(true, pred)

        n_cc, n_cu, n_la, n_fc = conf.sum(axis=1).tolist()
        cc_to_fc = conf[0, 3] / max(n_cc, 1)
        fc_recall = conf[3, 3] / max(n_fc, 1) if n_fc > 0 else None
        la_recall = conf[2, 2] / max(n_la, 1) if n_la > 0 else None
        cc_recall = conf[0, 0] / max(n_cc, 1) if n_cc > 0 else None

        seq_results.append({
            "dataset": ds,
            "sequence": seq,
            "n_frames": int(n),
            "n_CC": int(n_cc),
            "n_CU": int(n_cu),
            "n_LA": int(n_la),
            "n_FC": int(n_fc),
            "cc_to_fc_rate": float(cc_to_fc),
            "fc_recall": None if fc_recall is None else float(fc_recall),
            "la_recall": None if la_recall is None else float(la_recall),
            "cc_recall": None if cc_recall is None else float(cc_recall),
            "confusion": conf.tolist(),
        })
        aggregate_conf += conf
        per_dataset_conf[ds] += conf

    print(f"\n[done] processed {len(seq_results)} val sequences\n")

    # ----------------------------------------------------------------- TOP K HARD
    by_cc_to_fc = sorted([s for s in seq_results if s["n_CC"] >= 50],
                         key=lambda s: s["cc_to_fc_rate"], reverse=True)
    print("=" * 110)
    print(f"TOP {args.top_k} HARD sequences (highest CC→FC false alarm rate, n_CC≥50):")
    print(f"{'dataset':<14}{'sequence':<28}{'n_fr':>6}{'n_CC':>6}{'n_FC':>6}"
          f"{'CC→FC':>8}{'FC_rec':>8}{'LA_rec':>8}{'CC_rec':>8}")
    print("-" * 110)
    for s in by_cc_to_fc[:args.top_k]:
        fr = "—" if s["fc_recall"] is None else f"{s['fc_recall']:.3f}"
        lr = "—" if s["la_recall"] is None else f"{s['la_recall']:.3f}"
        cr = "—" if s["cc_recall"] is None else f"{s['cc_recall']:.3f}"
        print(f"{s['dataset']:<14}{s['sequence']:<28}{s['n_frames']:>6}{s['n_CC']:>6}"
              f"{s['n_FC']:>6}{s['cc_to_fc_rate']:>8.3f}{fr:>8}{lr:>8}{cr:>8}")

    # ----------------------------------------------------------------- TOP K CLEAN
    print("\n" + "=" * 110)
    print(f"TOP {args.top_k} CLEAN sequences (lowest CC→FC, n_CC≥50):")
    print(f"{'dataset':<14}{'sequence':<28}{'n_fr':>6}{'n_CC':>6}{'n_FC':>6}"
          f"{'CC→FC':>8}{'FC_rec':>8}{'LA_rec':>8}{'CC_rec':>8}")
    print("-" * 110)
    for s in sorted(by_cc_to_fc, key=lambda s: s["cc_to_fc_rate"])[:args.top_k]:
        fr = "—" if s["fc_recall"] is None else f"{s['fc_recall']:.3f}"
        lr = "—" if s["la_recall"] is None else f"{s['la_recall']:.3f}"
        cr = "—" if s["cc_recall"] is None else f"{s['cc_recall']:.3f}"
        print(f"{s['dataset']:<14}{s['sequence']:<28}{s['n_frames']:>6}{s['n_CC']:>6}"
              f"{s['n_FC']:>6}{s['cc_to_fc_rate']:>8.3f}{fr:>8}{lr:>8}{cr:>8}")

    # ----------------------------------------------------------------- AERIAL HARD
    aerial = [s for s in by_cc_to_fc if s["dataset"] in AERIAL_DATASETS]
    if aerial:
        print("\n" + "=" * 110)
        print(f"TOP {min(args.top_k, len(aerial))} AERIAL HARD sequences (CC→FC, aerial datasets only):")
        print(f"{'dataset':<14}{'sequence':<28}{'n_fr':>6}{'n_CC':>6}{'n_FC':>6}"
              f"{'CC→FC':>8}{'FC_rec':>8}{'LA_rec':>8}{'CC_rec':>8}")
        print("-" * 110)
        for s in aerial[:args.top_k]:
            fr = "—" if s["fc_recall"] is None else f"{s['fc_recall']:.3f}"
            lr = "—" if s["la_recall"] is None else f"{s['la_recall']:.3f}"
            cr = "—" if s["cc_recall"] is None else f"{s['cc_recall']:.3f}"
            print(f"{s['dataset']:<14}{s['sequence']:<28}{s['n_frames']:>6}{s['n_CC']:>6}"
                  f"{s['n_FC']:>6}{s['cc_to_fc_rate']:>8.3f}{fr:>8}{lr:>8}{cr:>8}")

    # ----------------------------------------------------------------- AGGREGATE
    print("\n" + "=" * 110)
    print("AGGREGATE confusion (rows=true, cols=pred):")
    print(f"             {' '.join(f'pred_{n:<5}' for n in STATE_NAMES)}")
    for i, name in enumerate(STATE_NAMES):
        row = aggregate_conf[i]
        total = max(row.sum(), 1)
        rates = " ".join(f"{r/total:.3f}" for r in row)
        print(f"  true_{name:<5}{row.sum():>8}  {rates}")

    print("\nPer-dataset CC→FC false alarm rate (true_CC predicted as FC):")
    for ds in sorted(per_dataset_conf):
        c = per_dataset_conf[ds]
        n_cc = c[0].sum()
        cc_to_fc = c[0, 3] / max(n_cc, 1)
        n_fc = c[3].sum()
        fc_recall = c[3, 3] / max(n_fc, 1) if n_fc > 0 else float("nan")
        marker = "[AERIAL]" if ds in AERIAL_DATASETS else ""
        print(f"  {ds:<16} n_CC={n_cc:>7}  CC→FC={cc_to_fc:.4f}  n_FC={n_fc:>6}  "
              f"FC_recall={fc_recall:.3f}  {marker}")

    # Aerial vs LaSOT vs other roll-up
    def _roll(filter_fn):
        c = np.zeros((N_STATES, N_STATES), dtype=np.int64)
        for ds, mat in per_dataset_conf.items():
            if filter_fn(ds):
                c += mat
        return c

    aer_c = _roll(lambda d: d in AERIAL_DATASETS)
    lasot_c = _roll(lambda d: d == "lasot")
    other_c = _roll(lambda d: d not in AERIAL_DATASETS and d != "lasot")
    print("\nGroup roll-up:")
    for label, c in [("aerial", aer_c), ("lasot", lasot_c), ("other", other_c)]:
        n_cc = c[0].sum()
        n_fc = c[3].sum()
        if n_cc == 0 and n_fc == 0:
            continue
        cc_to_fc = c[0, 3] / max(n_cc, 1)
        fc_recall = c[3, 3] / max(n_fc, 1) if n_fc > 0 else float("nan")
        cc_recall = c[0, 0] / max(n_cc, 1)
        print(f"  {label:<8} n_CC={n_cc:>7}  CC→FC={cc_to_fc:.4f}  CC_recall={cc_recall:.3f}  "
              f"n_FC={n_fc:>6}  FC_recall={fc_recall:.3f}")

    out_data = {
        "ckpt": str(ckpt_path),
        "feature_version": feature_version,
        "n_val_sequences": len(seq_results),
        "aggregate_confusion": aggregate_conf.tolist(),
        "per_dataset_confusion": {ds: c.tolist() for ds, c in per_dataset_conf.items()},
        "sequences": seq_results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_data, indent=2))
    print(f"\n[wrote] {out_json}")


if __name__ == "__main__":
    main()
