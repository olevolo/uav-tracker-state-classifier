"""Control-mode A/B (passive vs proactive control) on easy/hard SGLATrack scenes.

Datasets used here are HELD-OUT from the CSC *classifier* training set:
  - UAV123      : ZERO exposure (not in classifier training, not in calibrator) -> headline
  - UAVTrack112 : not in classifier training; calibrator-only telemetry exposure
DELIBERATELY EXCLUDED (leaked into the classifier training set -> circular):
  - DTB70    : 70 seqs in v3fix_combined/dtb70 shard
  - UAVDT    : 50 seqs in the base shard (verified 2026-05-31)
  - VisDrone : 35 seqs in the base shard (verified 2026-05-31)

Per dataset:
  1. Rank sequences by SGLATrack BASELINE GT-fail rate = fraction(IoU < TAU_FAIL),
     read from cached baseline predictions (NO tracker re-run).
  2. Pick N easiest + N hardest.
  3. Run SGLATrack PASSIVE (control off) and CONTROL (proactive_v3 + exit_router)
     on those sequences via tools/run_with_csc.py.
  4. Compare Success-AUC + FCR (CSC states, GT-validated), split easy vs hard.

The proactive threshold is the DEFAULT 0.7 -- it is NOT tuned on any eval set
(CLAUDE.md: no threshold tuning on UAV123).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

PYTHON = ROOT / ".venv/bin/python"
TAU_FAIL = 0.2
CALIB_PREFIX = "sglatrack_all_v2"
R2_CKPT = ROOT / "outputs/csc_training/sglatrack_run2_scalectx_tcn16_stage2/checkpoint_best.pth"
BASELINES = ROOT / "outputs/baselines/sglatrack"
OUT_ROOT = ROOT / "outputs/eval_control_ab"
PROACTIVE_THRESHOLD = 0.7

# datasets that leaked into CLASSIFIER training -> never use for held-out eval
FORBIDDEN = {"dtb70", "got10k", "lasot", "uavdt_sot", "visdrone_sot"}


def iou_xywh(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    px, py, pw, ph = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    gx, gy, gw, gh = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    ix1 = np.maximum(px, gx); iy1 = np.maximum(py, gy)
    ix2 = np.minimum(px + pw, gx + gw); iy2 = np.minimum(py + ph, gy + gh)
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = pw * ph + gw * gh - inter
    return np.where(union > 0, inter / union, 0.0)


def center_error_xywh(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-frame Euclidean center distance in pixels (for Precision@20px)."""
    pcx = pred[:, 0] + pred[:, 2] / 2.0
    pcy = pred[:, 1] + pred[:, 3] / 2.0
    gcx = gt[:, 0] + gt[:, 2] / 2.0
    gcy = gt[:, 1] + gt[:, 3] / 2.0
    return np.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2)


def load_gt(dataset: str) -> dict[str, np.ndarray]:
    import csc_uav_tracking  # noqa: F401
    from csc_uav_tracking.registry import DATASETS
    try:
        ds = DATASETS.build(dataset, split="test")
    except TypeError:
        ds = DATASETS.build(dataset)
    return {s.name: np.array([[b.x, b.y, b.w, b.h] for b in s.ground_truth], dtype=np.float64)
            for s in ds}


def read_preds(path: Path) -> np.ndarray:
    return np.array([[float(v) for v in ln.split(",")]
                     for ln in path.read_text().splitlines() if ln.strip()],
                    dtype=np.float64)


def success_auc(ious: np.ndarray) -> float:
    """Standard OPE success AUC: mean over thresholds 0..1 of (IoU >= t)."""
    if ious.size == 0:
        return float("nan")
    thr = np.linspace(0.0, 1.0, 21)
    return float(np.mean([(ious >= t).mean() for t in thr]))


def rank_dataset(dataset: str) -> list[dict]:
    """Per-seq baseline GT-fail rate from cached predictions. Sorted easy->hard."""
    gt_by = load_gt(dataset)
    pred_dir = BASELINES / dataset / "test" / "predictions"
    rows = []
    for pf in sorted(pred_dir.glob("*.txt")):
        name = pf.stem
        if name not in gt_by:
            continue
        preds = read_preds(pf)
        gt = gt_by[name]
        n = min(len(preds), len(gt))
        if n < 5:
            continue
        ious = iou_xywh(preds[:n], gt[:n])
        rows.append({"name": name, "n": int(n),
                     "fail_rate": float((ious < TAU_FAIL).mean()),
                     "baseline_auc": success_auc(ious)})
    rows.sort(key=lambda r: (r["fail_rate"], r["name"]))
    return rows


def run_csc(dataset: str, seqs: list[str], mode: str, run_tag: str,
            gate: bool = False) -> Path:
    cmd = [str(PYTHON), "-u", str(ROOT / "tools/run_with_csc.py"),
           "--tracker", "sglatrack", "--dataset", dataset, "--split", "test",
           "--csc_checkpoint", str(R2_CKPT), "--csc_mode", mode,
           "--calibration_prefix", CALIB_PREFIX, "--device", "cpu",
           "--output_dir", str(OUT_ROOT), "--run_tag", run_tag,
           "--include_sequences", *seqs]
    if mode == "control":
        cmd += ["--exit_router", "--proactive_v3",
                "--proactive_threshold", str(PROACTIVE_THRESHOLD)]
    if gate:
        # "if FCR ~ 0 -> do nothing": bypass ALL control on frames with no
        # recent LA/FC risk (causal trailing-window gate). Recovers easy-scene
        # AUC lost to template-update drift; leaves hard-scene control intact.
        cmd += ["--control_risk_gate"]
    tag = f"{mode}+gate" if gate else mode
    print(f"  RUN [{tag}] {dataset} ({len(seqs)} seqs) -> {run_tag}", flush=True)
    subprocess.run(cmd, check=True)
    return OUT_ROOT / run_tag


def read_metric(run_dir: Path, key: str):
    m = run_dir / "metrics.json"
    if not m.exists():
        return None
    try:
        return json.loads(m.read_text()).get(key)
    except Exception:
        return None


def evaluate_run(run_dir: Path, gt_by: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    states_dir = run_dir / "states"
    pred_dir = run_dir / "predictions"
    for sf in sorted(states_dir.glob("*.jsonl")):
        name = sf.stem
        if name not in gt_by:
            continue
        st = [json.loads(l) for l in sf.open() if not json.loads(l).get("init")]
        states = np.array([r["derived_state"] for r in st])
        preds = read_preds(pred_dir / f"{name}.txt")
        gt = gt_by[name]
        n = min(len(states), len(preds) - 1, len(gt) - 1)
        if n < 5:
            continue
        ious = iou_xywh(preds[1:n + 1], gt[1:n + 1])
        states = states[:n]
        out[name] = {
            "n": int(n),
            "auc": success_auc(ious),
            "prec20": float((center_error_xywh(preds[1:n + 1], gt[1:n + 1]) <= 20.0).mean()),
            "fcr": float((states == 3).mean()),       # FALSE_CONFIRMED
            "la_rate": float((states == 2).mean()),    # LOST_AWARE
            "mean_iou": float(ious.mean()),
        }
    return out


def agg(d: dict[str, dict], names: list[str], key: str) -> float:
    vals = [d[n][key] for n in names if n in d]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    global R2_CKPT, OUT_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["uav123", "uavtrack112"])
    ap.add_argument("--n-easy", type=int, default=10)
    ap.add_argument("--n-hard", type=int, default=10)
    ap.add_argument("--rank-only", action="store_true",
                    help="Only rank + select; do NOT run passive/control.")
    ap.add_argument("--checkpoint", default=str(R2_CKPT),
                    help="CSC checkpoint (default R2). Pass R3-fcw3 for the R3 A/B.")
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="Output dir; use a fresh dir to avoid overwriting the R2 run.")
    ap.add_argument("--ckpt-label", default="R2",
                    help="Short checkpoint label shown in summary.md.")
    args = ap.parse_args()

    R2_CKPT = Path(args.checkpoint)
    OUT_ROOT = Path(args.out_root)

    for ds in args.datasets:
        assert ds not in FORBIDDEN, f"{ds} leaked into classifier training -- not a held-out set"
    assert R2_CKPT.exists(), f"ckpt missing: {R2_CKPT}"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    report = {"threshold": PROACTIVE_THRESHOLD, "ckpt": str(R2_CKPT),
              "ckpt_label": args.ckpt_label, "datasets": {}}

    for ds in args.datasets:
        print(f"\n{'='*70}\nDATASET: {ds}\n{'='*70}", flush=True)
        ranked = rank_dataset(ds)
        easy = [r["name"] for r in ranked[:args.n_easy]]
        hard = [r["name"] for r in ranked[-args.n_hard:]]
        sel = easy + hard
        print(f"  {len(ranked)} seqs ranked. EASY (fail%): "
              + ", ".join(f"{r['name']}={100*r['fail_rate']:.0f}" for r in ranked[:args.n_easy]))
        print(f"  HARD (fail%): "
              + ", ".join(f"{r['name']}={100*r['fail_rate']:.0f}" for r in ranked[-args.n_hard:]))

        ds_rec = {"easy": easy, "hard": hard,
                  "ranked_failrate": {r["name"]: r["fail_rate"] for r in ranked}}

        if not args.rank_only:
            gt_by = load_gt(ds)
            passive_dir = run_csc(ds, sel, "passive", f"{ds}_eh_passive")
            control_dir = run_csc(ds, sel, "control", f"{ds}_eh_control")
            # Risk-gate runs split by cohort so each gets a clean
            # risk_gate_closed_frac (easy should close ~all frames, hard stay open).
            gate_easy_dir = run_csc(ds, easy, "control", f"{ds}_eh_gate_easy", gate=True)
            gate_hard_dir = run_csc(ds, hard, "control", f"{ds}_eh_gate_hard", gate=True)
            ev_p = evaluate_run(passive_dir, gt_by)
            ev_c = evaluate_run(control_dir, gt_by)
            ev_g = {**evaluate_run(gate_easy_dir, gt_by),
                    **evaluate_run(gate_hard_dir, gt_by)}
            gate_closed = {
                "easy": read_metric(gate_easy_dir, "risk_gate_closed_frac"),
                "hard": read_metric(gate_hard_dir, "risk_gate_closed_frac"),
            }
            ds_rec["per_seq"] = {"passive": ev_p, "control": ev_c, "control_gate": ev_g}
            ds_rec["gate_closed_frac"] = gate_closed
            for label, names in [("easy", easy), ("hard", hard)]:
                ds_rec[f"{label}_summary"] = {
                    "passive_auc": agg(ev_p, names, "auc"),
                    "control_auc": agg(ev_c, names, "auc"),
                    "control_gate_auc": agg(ev_g, names, "auc"),
                    "passive_prec20": agg(ev_p, names, "prec20"),
                    "control_prec20": agg(ev_c, names, "prec20"),
                    "control_gate_prec20": agg(ev_g, names, "prec20"),
                    "passive_fcr": agg(ev_p, names, "fcr"),
                    "control_fcr": agg(ev_c, names, "fcr"),
                    "control_gate_fcr": agg(ev_g, names, "fcr"),
                    "passive_la": agg(ev_p, names, "la_rate"),
                    "control_la": agg(ev_c, names, "la_rate"),
                    "control_gate_la": agg(ev_g, names, "la_rate"),
                    "gate_closed_frac": gate_closed[label],
                }
        report["datasets"][ds] = ds_rec

    (OUT_ROOT / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"\n[DONE] wrote {OUT_ROOT/'summary.json'}", flush=True)

    if not args.rank_only:
        lines = ["# Control A/B (passive vs proactive control vs control+risk-gate) — held-out scenes",
                 f"\nProactive threshold = {PROACTIVE_THRESHOLD} (default, NOT tuned on eval sets). "
                 f"ckpt = {args.ckpt_label}. Risk-gate: control fully bypassed (=passive) on frames with no "
                 f"recent LA/FC risk — recovers easy-scene AUC lost to template drift.\n",
                 "| dataset | subset | AUC p→c→+gate | Pr@20 p→c→+gate | FCR p→c→+gate | LA p→c→+gate | gate_closed |",
                 "|---|---|---|---|---|---|---|"]
        for ds, rec in report["datasets"].items():
            for label in ("easy", "hard"):
                s = rec.get(f"{label}_summary")
                if not s:
                    continue
                gc = s.get("gate_closed_frac")
                gc_str = f"{gc:.3f}" if isinstance(gc, (int, float)) else "—"
                lines.append(
                    f"| {ds} | {label} "
                    f"| {s['passive_auc']:.3f}→{s['control_auc']:.3f}→{s['control_gate_auc']:.3f} "
                    f"| {s['passive_prec20']:.3f}→{s['control_prec20']:.3f}→{s['control_gate_prec20']:.3f} "
                    f"| {s['passive_fcr']:.3f}→{s['control_fcr']:.3f}→{s['control_gate_fcr']:.3f} "
                    f"| {s['passive_la']:.3f}→{s['control_la']:.3f}→{s['control_gate_la']:.3f} "
                    f"| {gc_str} |")
        (OUT_ROOT / "summary.md").write_text("\n".join(lines) + "\n")
        print(f"[DONE] wrote {OUT_ROOT/'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
