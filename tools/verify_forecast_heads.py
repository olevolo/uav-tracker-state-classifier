"""Measure lead time of CSC V3 forecast heads — how many frames ahead do they
warn before the actual FC onset?

For each scene with FC onsets:
  1. Run R1 (V1 features) and R2 (V2 features) on the sequence
  2. Extract p_failure_next_10, p_fc_next_10, p_la_next_10 (forecast heads)
  3. Find FC onsets: state[t-1] != 3 AND state[t] == 3
  4. For each onset compute:
     - lead_time_strict: frames the model warned ahead with CONTIGUOUS over-thr
     - lead_time_any: frames the model first crossed thr (not necessarily cont.)
  5. Aggregate across scenes — paper claim: warn 5+ frames ahead.
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

REQUIRED_KEYS = (
    "failure_next_10_prob",
    "false_confirmed_next_10_prob",
    "lost_aware_next_10_prob",
)


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


def predict_forecasts(model, cfg, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    fv = getattr(cfg.feature, "feature_version", "v1")
    builder = build_sequence_features_v2 if fv == "v2" else build_sequence_features
    feats = builder(rows, DEFAULT_IMAGE_SIZE, cfg=cfg.feature)
    x = torch.from_numpy(feats).unsqueeze(0)
    with torch.inference_mode():
        out = model.predict(x, last_step_only=False)
    missing = [k for k in REQUIRED_KEYS if k not in out]
    if missing:
        raise RuntimeError(
            f"Checkpoint output is missing forecast-head keys {missing}. "
            f"This appears to be a Stage 1 ckpt (no forecast heads enabled). "
            f"Use a Stage 2 ckpt with enable_forecast_heads=True."
        )
    p_fail = out["failure_next_10_prob"][0].cpu().numpy()
    p_fc   = out["false_confirmed_next_10_prob"][0].cpu().numpy()
    p_la   = out["lost_aware_next_10_prob"][0].cpu().numpy()
    return p_fail, p_fc, p_la, fv


def find_fc_onsets(states: np.ndarray) -> list[int]:
    """Return frame indices where a new FC block begins (state[t-1]!=FC, state[t]==FC).
    Skip the very first frame (no t-1)."""
    onsets: list[int] = []
    for t in range(1, len(states)):
        if states[t] == FC_LABEL and states[t - 1] != FC_LABEL:
            onsets.append(t)
    return onsets


def compute_lead_strict(p: np.ndarray, t_fc: int, threshold: float, window_before: int) -> int:
    """Largest k>=0 such that p[t_fc-k .. t_fc-1] are ALL >= threshold,
    with k <= window_before. Returns 0 if p[t_fc-1] < threshold (no warning at onset)."""
    start = max(0, t_fc - window_before)
    if t_fc - 1 < start:
        return 0
    # Walk backwards from t_fc-1 while still above threshold
    k = 0
    t = t_fc - 1
    while t >= start and p[t] >= threshold:
        k += 1
        t -= 1
    return k


def compute_lead_any(p: np.ndarray, t_fc: int, threshold: float, window_before: int) -> int:
    """t_fc - t_first, where t_first is the smallest t in [t_fc-window_before, t_fc-1]
    with p[t] >= threshold. 0 if no crossing in the window."""
    start = max(0, t_fc - window_before)
    end = t_fc  # exclusive
    if start >= end:
        return 0
    region = p[start:end]
    above = np.where(region >= threshold)[0]
    if len(above) == 0:
        return 0
    t_first = start + int(above[0])
    return t_fc - t_first


def safe_at(p: np.ndarray, t: int) -> float:
    if t < 0 or t >= len(p):
        return float("nan")
    return float(p[t])


def analyze_scene(
    ds: str,
    seq: str,
    model,
    cfg,
    labels_dir: Path,
    threshold: float,
    window_before: int,
    window_after: int,
    tag: str,
) -> list[dict]:
    print(f"\n  [{tag}] scene {ds}/{seq}")
    rows = load_seq_rows(labels_dir, ds, seq)
    if not rows or len(rows) < 16:
        print(f"    (skip — too few frames: {len(rows)})")
        return []
    states = np.array([r.get("derived_state", -1) for r in rows], dtype=np.int32)
    p_fail, p_fc, p_la, fv = predict_forecasts(model, cfg, rows)
    n = min(len(states), len(p_fail))
    states = states[:n]
    p_fail, p_fc, p_la = p_fail[:n], p_fc[:n], p_la[:n]

    onsets = find_fc_onsets(states)
    if not onsets:
        print(f"    no FC onsets in this seq — skip (n_fr={n}, fv={fv})")
        return []

    print(f"    fv={fv}  n_fr={n}  n_onsets={len(onsets)}  threshold={threshold:.2f}  "
          f"window_before={window_before}")
    print(f"    {'t_fc':>6}  {'lt_strict':>9}  {'lt_any':>6}  "
          f"{'pF@-10':>7}  {'pF@-5':>7}  {'pF@0':>6}  "
          f"{'pFC@-10':>8}  {'pFC@-5':>7}  {'pFC@0':>7}")
    records: list[dict] = []
    for t_fc in onsets:
        lt_strict = compute_lead_strict(p_fail, t_fc, threshold, window_before)
        lt_any = compute_lead_any(p_fail, t_fc, threshold, window_before)
        pf_m10 = safe_at(p_fail, t_fc - 10)
        pf_m5  = safe_at(p_fail, t_fc - 5)
        pf_0   = safe_at(p_fail, t_fc)
        pfc_m10 = safe_at(p_fc, t_fc - 10)
        pfc_m5  = safe_at(p_fc, t_fc - 5)
        pfc_0   = safe_at(p_fc, t_fc)
        print(f"    {t_fc:>6d}  {lt_strict:>9d}  {lt_any:>6d}  "
              f"{pf_m10:>7.3f}  {pf_m5:>7.3f}  {pf_0:>6.3f}  "
              f"{pfc_m10:>8.3f}  {pfc_m5:>7.3f}  {pfc_0:>7.3f}")
        records.append({
            "dataset": ds, "sequence": seq, "tag": tag,
            "t_fc": int(t_fc),
            "lead_strict": int(lt_strict),
            "lead_any": int(lt_any),
        })

    leads_strict = np.array([r["lead_strict"] for r in records])
    leads_any = np.array([r["lead_any"] for r in records])
    frac_ge5 = float((leads_strict >= 5).mean())
    print(f"    summary: n_onsets={len(records)}  "
          f"mean_lt_strict={leads_strict.mean():.2f}  med_lt_strict={np.median(leads_strict):.1f}  "
          f"mean_lt_any={leads_any.mean():.2f}  med_lt_any={np.median(leads_any):.1f}  "
          f"frac(lt_strict>=5)={frac_ge5:.2f}")
    return records


def aggregate(records: list[dict], tag: str, threshold: float) -> dict:
    if not records:
        print(f"\n  [{tag}] no onsets collected.")
        return {"tag": tag, "n_onsets": 0}
    leads_strict = np.array([r["lead_strict"] for r in records])
    leads_any = np.array([r["lead_any"] for r in records])
    summary = {
        "tag": tag,
        "n_onsets": int(len(records)),
        "mean_strict": float(leads_strict.mean()),
        "median_strict": float(np.median(leads_strict)),
        "p25_strict": float(np.percentile(leads_strict, 25)),
        "p75_strict": float(np.percentile(leads_strict, 75)),
        "mean_any": float(leads_any.mean()),
        "median_any": float(np.median(leads_any)),
        "frac_ge5_strict": float((leads_strict >= 5).mean()),
    }
    print(f"\n  [{tag}] aggregate over {summary['n_onsets']} onsets "
          f"(threshold={threshold:.2f}):")
    print(f"    lead_strict  mean={summary['mean_strict']:.2f}  "
          f"med={summary['median_strict']:.1f}  "
          f"p25={summary['p25_strict']:.1f}  p75={summary['p75_strict']:.1f}")
    print(f"    lead_any     mean={summary['mean_any']:.2f}  "
          f"med={summary['median_any']:.1f}")
    print(f"    fraction with lead_strict >= 5: {summary['frac_ge5_strict']:.3f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-r1", type=Path,
                    default=Path("/tmp/smoke_logs/t7_run1_s2/checkpoint_best.pth"),
                    help="R1 (V1 features) Stage 2 ckpt with forecast heads.")
    ap.add_argument("--ckpt-r2", type=Path,
                    default=Path("/tmp/smoke_logs/t9_run2_s2/checkpoint_best.pth"),
                    help="R2 (V2 features) Stage 2 ckpt with forecast heads.")
    ap.add_argument("--labels-dir", type=Path,
                    default=Path("outputs/csc_labels/sglatrack/v3fix_combined"))
    ap.add_argument("--scenes", default=None,
                    help="Comma-separated list of dataset:sequence pairs.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--window-before", type=int, default=15)
    ap.add_argument("--window-after", type=int, default=5)
    args = ap.parse_args()

    if args.scenes:
        scenes = [tuple(s.split(":", 1)) for s in args.scenes.split(",")]
    else:
        scenes = DEFAULT_SCENES

    print(f"[load] R1 ckpt={args.ckpt_r1}")
    model_r1, cfg_r1 = load_ckpt(args.ckpt_r1)
    print(f"[load] R2 ckpt={args.ckpt_r2}")
    model_r2, cfg_r2 = load_ckpt(args.ckpt_r2)

    print(f"\n[run] {len(scenes)} scenes  threshold={args.threshold}  "
          f"window_before={args.window_before}  window_after={args.window_after}")

    records_r1: list[dict] = []
    records_r2: list[dict] = []
    for ds, seq in scenes:
        print(f"\n{'=' * 100}")
        print(f"SCENE: {ds}/{seq}")
        print("=" * 100)
        records_r1.extend(analyze_scene(
            ds, seq, model_r1, cfg_r1, args.labels_dir,
            args.threshold, args.window_before, args.window_after, "R1",
        ))
        records_r2.extend(analyze_scene(
            ds, seq, model_r2, cfg_r2, args.labels_dir,
            args.threshold, args.window_before, args.window_after, "R2",
        ))

    print(f"\n{'=' * 100}")
    print("AGGREGATE")
    print("=" * 100)
    s1 = aggregate(records_r1, "R1 (V1)", args.threshold)
    s2 = aggregate(records_r2, "R2 (V2)", args.threshold)

    if s1.get("n_onsets") and s2.get("n_onsets"):
        print("\n  side-by-side R1 vs R2:")
        print(f"    {'metric':<22}{'R1 (V1)':>14}{'R2 (V2)':>14}")
        for k, label in [
            ("n_onsets",         "n_onsets"),
            ("mean_strict",      "mean lead_strict"),
            ("median_strict",    "median lead_strict"),
            ("p25_strict",       "p25 lead_strict"),
            ("p75_strict",       "p75 lead_strict"),
            ("mean_any",         "mean lead_any"),
            ("median_any",       "median lead_any"),
            ("frac_ge5_strict",  "frac(lead>=5)"),
        ]:
            v1 = s1.get(k)
            v2 = s2.get(k)
            if isinstance(v1, float):
                print(f"    {label:<22}{v1:>14.3f}{v2:>14.3f}")
            else:
                print(f"    {label:<22}{v1:>14}{v2:>14}")


if __name__ == "__main__":
    main()
