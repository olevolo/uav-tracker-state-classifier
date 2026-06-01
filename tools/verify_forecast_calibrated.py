"""Calibrated forecast-head lead-time comparison.

Why this exists:
  verify_forecast_heads.py uses a fixed probability threshold (default 0.50)
  for ALL models. That is invalid for cross-model comparison — a trigger-happy
  model wins on lead time trivially.

  This script reads calibrated thresholds from fc_n10_threshold_aware.json
  (where the threshold is chosen per model to give FPR ≤ 3% on the val set)
  and measures lead_strict / lead_any of the p_fc head at THAT threshold.

If R1 still beats R2 here → V1 features carry real early-warning signal.
If the advantage disappears → V1's "advantage" was just calibration noise.

Usage:
  python tools/verify_forecast_calibrated.py \\
    --threshold-json logs/v3fix_full/fc_n10_threshold_aware.json \\
    --signal p_fc --fpr 0.03 \\
    --labels-dir outputs/csc_labels/sglatrack/v3fix_combined
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csc_lib.csc.config import CSCTrainConfig  # noqa: E402
from csc_lib.csc.features import (  # noqa: E402
    build_sequence_features,
    build_sequence_features_v2,
)
from csc_lib.csc.model import build_model  # noqa: E402

DEFAULT_SCENES = [
    ("lasot",         "bird-10"),
    ("lasot",         "drone-11"),
    ("lasot",         "car-13"),
    ("lasot",         "drone-19"),
    ("uavdt_sot",     "S1308"),
    ("uavdt_sot",     "S0201"),
    ("visdrone_sot",  "uav0000180_00050_s"),
    ("visdrone_sot",  "uav0000184_00625_s"),
    ("visdrone_sot",  "uav0000074_06312_s"),
    ("dtb70",         "BMX3"),
]

DEFAULT_IMAGE_SIZE = (1280, 720)
FC_LABEL = 3
SIGNAL_TO_KEY = {
    "p_fc":   "false_confirmed_next_10_prob",
    "p_fail": "failure_next_10_prob",
    "p_la":   "lost_aware_next_10_prob",
}


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


def load_seq_rows(labels_dir: Path, ds: str, seq: str) -> list[dict]:
    rows: list[dict] = []
    for jsonl in labels_dir.glob("*/labels.jsonl"):
        with jsonl.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("dataset") == ds and r.get("sequence") == seq:
                    rows.append(r)
    rows.sort(key=lambda r: r.get("frame_idx", 0))
    return rows


def predict(model, cfg, rows: list[dict], signal_key: str) -> np.ndarray:
    fv = getattr(cfg.feature, "feature_version", "v1")
    builder = build_sequence_features_v2 if fv == "v2" else build_sequence_features
    feats = builder(rows, DEFAULT_IMAGE_SIZE, cfg=cfg.feature)
    x = torch.from_numpy(feats).unsqueeze(0)
    with torch.inference_mode():
        out = model.predict(x, last_step_only=False)
    if signal_key not in out:
        raise RuntimeError(
            f"Ckpt missing forecast key '{signal_key}'. "
            f"Available: {list(out.keys())}. "
            f"This must be a Stage 2 ckpt with enable_forecast_heads=True."
        )
    return out[signal_key][0].cpu().numpy()


def find_fc_onsets(states: np.ndarray) -> list[int]:
    onsets: list[int] = []
    for t in range(1, len(states)):
        if states[t] == FC_LABEL and states[t - 1] != FC_LABEL:
            onsets.append(t)
    return onsets


def lead_strict(p: np.ndarray, t_fc: int, thr: float, window: int) -> int:
    start = max(0, t_fc - window)
    if t_fc - 1 < start:
        return 0
    k = 0
    t = t_fc - 1
    while t >= start and p[t] >= thr:
        k += 1
        t -= 1
    return k


def lead_any(p: np.ndarray, t_fc: int, thr: float, window: int) -> int:
    start = max(0, t_fc - window)
    end = t_fc
    if start >= end:
        return 0
    above = np.where(p[start:end] >= thr)[0]
    if len(above) == 0:
        return 0
    return t_fc - (start + int(above[0]))


def analyze(model, cfg, labels_dir: Path, scenes: list[tuple[str, str]],
            signal_key: str, thr: float, window: int, tag: str) -> dict:
    leads_strict, leads_any = [], []
    onset_count = 0
    for ds, seq in scenes:
        rows = load_seq_rows(labels_dir, ds, seq)
        if not rows or len(rows) < 16:
            continue
        states = np.array([r.get("derived_state", -1) for r in rows], dtype=np.int32)
        p = predict(model, cfg, rows, signal_key)
        n = min(len(states), len(p))
        states, p = states[:n], p[:n]
        onsets = find_fc_onsets(states)
        onset_count += len(onsets)
        for t_fc in onsets:
            leads_strict.append(lead_strict(p, t_fc, thr, window))
            leads_any.append(lead_any(p, t_fc, thr, window))
    if not leads_strict:
        return {"tag": tag, "n_onsets": 0, "thr": thr, "signal": signal_key}
    ls = np.array(leads_strict)
    la = np.array(leads_any)
    return {
        "tag": tag,
        "signal": signal_key,
        "thr": float(thr),
        "n_onsets": int(onset_count),
        "mean_strict": float(ls.mean()),
        "median_strict": float(np.median(ls)),
        "p25_strict": float(np.percentile(ls, 25)),
        "p75_strict": float(np.percentile(ls, 75)),
        "frac_ge5_strict": float((ls >= 5).mean()),
        "mean_any": float(la.mean()),
        "median_any": float(np.median(la)),
        "frac_ge5_any": float((la >= 5).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold-json", type=Path,
                    default=ROOT / "logs/v3fix_full/fc_n10_threshold_aware.json")
    ap.add_argument("--fpr", type=float, default=0.03,
                    help="Which calibrated FPR budget to use (0.01 or 0.03 or 0.05).")
    ap.add_argument("--signal", default="p_fc", choices=list(SIGNAL_TO_KEY.keys()))
    ap.add_argument("--labels-dir", type=Path,
                    default=ROOT / "outputs/csc_labels/sglatrack/v3fix_combined")
    ap.add_argument("--window-before", type=int, default=15)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "logs/v3fix_full/lead_time_calibrated.json")
    args = ap.parse_args()

    thr_data = json.loads(args.threshold_json.read_text())
    thr_key = {0.01: "thr_fpr_01", 0.03: "thr_fpr_03", 0.05: "thr_fpr_05"}[args.fpr]
    rec_key = {0.01: "recall_at_fpr_01", 0.03: "recall_at_fpr_03", 0.05: "recall_at_fpr_05"}[args.fpr]

    signal_key = SIGNAL_TO_KEY[args.signal]
    print(f"[config] signal={args.signal} (model key={signal_key})  "
          f"FPR budget={args.fpr*100:.0f}%  window_before={args.window_before}")

    results = []
    for entry in thr_data:
        ckpt = Path(entry["ckpt"])
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        thr = entry[thr_key]
        recall = entry[rec_key]
        feat = entry["feature_version"]
        tag = f"{ckpt.parent.name} ({feat})"

        print(f"\n[load] {tag}  thr={thr:.4f}  recall@FPR<={args.fpr:.2%} = {recall:.3f}")
        model, cfg = load_ckpt(ckpt)
        res = analyze(model, cfg, args.labels_dir, DEFAULT_SCENES,
                      signal_key, thr, args.window_before, tag)
        res["recall_at_op"] = recall
        res["ckpt"] = str(ckpt)
        res["feature_version"] = feat
        results.append(res)

        if res["n_onsets"] == 0:
            print(f"  (no onsets across scenes — skipping summary)")
            continue
        print(f"  n_onsets={res['n_onsets']}  "
              f"mean_strict={res['mean_strict']:.2f}  med={res['median_strict']:.1f}  "
              f"frac(>=5)={res['frac_ge5_strict']:.3f}  "
              f"mean_any={res['mean_any']:.2f}  frac_any(>=5)={res['frac_ge5_any']:.3f}")

    print("\n" + "=" * 90)
    print(f"CALIBRATED COMPARISON  ({args.signal} @ FPR<={args.fpr:.2%})")
    print("=" * 90)
    valid = [r for r in results if r["n_onsets"] > 0]
    if len(valid) >= 1:
        print(f"  {'model':<40} {'thr':>7} {'rec@OP':>7} {'mean_LS':>8} "
              f"{'med_LS':>7} {'frac>=5':>8} {'mean_LA':>8}")
        for r in valid:
            print(f"  {r['tag']:<40} {r['thr']:>7.4f} {r['recall_at_op']:>7.3f} "
                  f"{r['mean_strict']:>8.2f} {r['median_strict']:>7.1f} "
                  f"{r['frac_ge5_strict']:>8.3f} {r['mean_any']:>8.2f}")

    if len(valid) == 2:
        r1, r2 = valid[0], valid[1]
        d_strict = r2["mean_strict"] - r1["mean_strict"]
        d_frac = r2["frac_ge5_strict"] - r1["frac_ge5_strict"]
        print(f"\n  Δ (R2 − R1) mean_strict = {d_strict:+.2f}  "
              f"Δ frac(>=5) = {d_frac:+.3f}")
        if abs(d_strict) < 1.0 and abs(d_frac) < 0.10:
            print("  → calibrated lead time is essentially equal — "
                  "uncalibrated R1 advantage was an artifact.")
        elif d_strict > 0:
            print("  → R2 leads even at calibrated threshold.")
        else:
            print("  → R1 STILL leads at calibrated threshold — "
                  "real signal in V1 features worth investigating.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
