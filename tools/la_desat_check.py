#!/usr/bin/env python
"""De-saturation check for the lost_aware_next_10 forecast head.

Question: did the stage-2 retrain (focal-BCE gamma + higher lost weight) give the
lost_aware_next_10 head enough DYNAMIC RANGE to serve as a control gate that
separates TRUE-LOSS (recoverable, IoU<0.2) from FALSE-LA (tracker fine, IoU>=0.5)
frames — vs the PROD head which saturated ~0.99 on everything?

Method (apples-to-apples, no GT in the model path):
  - Replay the SAME passive telemetry (confidence/apce/psr/pred_bbox) through BOTH
    the prod and the new runtime, with the SAME calibrators + per-seq image size.
  - Cross-check: prod-runtime reproduction vs the prod prob stored in states/*.jsonl
    (should match closely → confirms stepping+calibration are faithful).
  - On frames where PROD predicted LA (derived_state==2): compare prob distributions
    + AUROC(prob -> true-loss vs false-LA) + a tau-sweep (true-recall vs false-fire).

Pure diagnosis of model behavior; UAV123 used only as an exploratory probe (NOT
threshold tuning for final results).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "tools"):
    sys.path.insert(0, str(_p))
_argv = sys.argv
sys.argv = [sys.argv[0]]
import csc_uav_tracking  # noqa: F401
import la_smoke as ls
from csc_lib.csc.inference import load_runtime
sys.argv = _argv

PROD = "outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth"
NEW = "outputs/csc_training/sglatrack_r3_fcw3lost_w32_tcn32_stage2/checkpoint_best.pth"
PASSIVE = PROJECT_ROOT / "outputs/eval_control_ab_r3/uav123_eh_passive"
CALIB_DIR = PROJECT_ROOT / "outputs/calibration"
CALIB_TAG = "sglatrack_all_v2"


def load_states(path):
    d = {}
    if not path.exists():
        return d
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        t = int(r.get("frame_idx", -1))
        d[t] = (r.get("derived_state"), r.get("lost_aware_next_10_prob"))
    return d


def load_telem(path):
    rows = {}
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln)
        rows[int(r.get("frame_idx", -1))] = r
    return rows


def run_runtime_over_seq(runtime, seq, telem, preds):
    """Replay telemetry+bbox through runtime.step(); return {frame_idx: lost_aware_prob}."""
    try:
        first = next(iter(seq.frames))
        h, w = first.shape[:2]
        image_size = (int(w), int(h))
    except Exception:
        image_size = (1280, 720)
    runtime.reset(image_size=image_size)
    out = {}
    for t in sorted(telem):
        row = telem[t]
        bbox = None
        if 0 <= t < len(preds):
            b = preds[t]
            if np.all(np.isfinite(b)):
                bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        pred = runtime.step(
            confidence=row.get("confidence"),
            apce=row.get("apce"),
            psr=row.get("psr"),
            pred_bbox=bbox,
        )
        out[t] = pred.lost_aware_next_10_prob
    return out


def auroc(scores, labels):
    """Rank-AUROC: P(score[pos] > score[neg]). labels: 1=positive(true-loss)."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = {}
    start = 0
    for i, c in enumerate(cnt):
        avg[i] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[i] for i in inv])
    n_pos = len(pos)
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * len(neg)))


def describe(probs):
    a = np.asarray([p for p in probs if p is not None], float)
    if len(a) == 0:
        return "  (none)"
    return (f"n={len(a):5d}  med={np.median(a):.3f}  "
            f"p10={np.percentile(a,10):.3f}  p90={np.percentile(a,90):.3f}  "
            f"≥.99={np.mean(a>=0.99):.2f}  ≥.95={np.mean(a>=0.95):.2f}  ≥.90={np.mean(a>=0.90):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", default=PROD)
    ap.add_argument("--new", default=NEW)
    ap.add_argument("--passive", default=str(PASSIVE))
    args = ap.parse_args()
    passive = Path(args.passive)

    print("loading uav123 + runtimes ...", file=sys.stderr)
    idx = ls.build_index()
    rt_prod = load_runtime(Path(args.prod), device="cpu", calibration_dir=CALIB_DIR, tracker_name=CALIB_TAG)
    rt_new = load_runtime(Path(args.new), device="cpu", calibration_dir=CALIB_DIR, tracker_name=CALIB_TAG)

    tel_dir = passive / "telemetry"
    st_dir = passive / "states"
    pr_dir = passive / "predictions"
    seqs = sorted(p.stem for p in tel_dir.glob("*.jsonl"))
    print(f"telemetry seqs available: {len(seqs)} -> {seqs}", file=sys.stderr)

    # Pooled per-frame records: (iou, prod_derived, prod_states_prob, prod_repro_prob, new_prob)
    rec = []
    xcheck = []  # |prod_repro - states_prod| on shared frames
    for name in seqs:
        seq = idx.get(name)
        if seq is None:
            continue
        telem = load_telem(tel_dir / f"{name}.jsonl")
        states = load_states(st_dir / f"{name}.jsonl")
        preds = ls.read_preds(pr_dir / f"{name}.txt")
        ious, n = ls.seq_iou(seq, pr_dir)
        if ious is None:
            continue
        prod_repro = run_runtime_over_seq(rt_prod, seq, telem, preds)
        new_repro = run_runtime_over_seq(rt_new, seq, telem, preds)
        for t in range(len(ious)):
            iou = ious[t]
            if not np.isfinite(iou):
                continue
            ds, sp = states.get(t, (None, None))
            pr = prod_repro.get(t)
            nw = new_repro.get(t)
            if pr is not None and sp is not None:
                xcheck.append(abs(pr - sp))
            rec.append((iou, ds, sp, pr, nw))

    if not rec:
        print("NO RECORDS — abort"); return
    ious = np.array([r[0] for r in rec])
    dsv = np.array([(-1 if r[1] is None else r[1]) for r in rec])
    prod_s = np.array([(np.nan if r[2] is None else r[2]) for r in rec])
    prod_r = np.array([(np.nan if r[3] is None else r[3]) for r in rec])
    new_r = np.array([(np.nan if r[4] is None else r[4]) for r in rec])

    xc = np.array(xcheck)
    print("\n" + "=" * 78)
    print("CROSS-CHECK  prod-runtime reproduction vs states-file prod prob")
    print(f"  shared frames={len(xc)}  mean|Δ|={xc.mean():.4f}  p95|Δ|={np.percentile(xc,95):.4f}  max|Δ|={xc.max():.4f}")
    print("  (small => stepping+calibration faithfully reproduce the prod run)")

    # Population: frames PROD called LA (derived_state==2)
    la = (dsv == 2)
    true_loss = la & (ious < 0.2)
    false_la = la & (ious >= 0.5)
    print("\n" + "=" * 78)
    print(f"LA-PREDICTED frames (prod derived_state==2): {la.sum()}  "
          f"| true-loss(IoU<0.2)={true_loss.sum()}  false-LA(IoU>=0.5)={false_la.sum()}")

    print("\n--- PROD lost_aware_next_10 (repro) ---")
    print("  true-loss:", describe(prod_r[true_loss]))
    print("  false-LA :", describe(prod_r[false_la]))
    print("--- NEW lost_aware_next_10 ---")
    print("  true-loss:", describe(new_r[true_loss]))
    print("  false-LA :", describe(new_r[false_la]))

    # Separation AUROC within LA frames: prob -> P(true-loss) vs false-LA
    mask = (true_loss | false_la)
    lbl = true_loss[mask].astype(int)
    print("\n" + "=" * 78)
    print("SEPARATION  AUROC(prob -> true-loss vs false-LA) within LA frames  (higher=better gate)")
    a_prod = auroc(prod_r[mask], lbl)
    a_new = auroc(new_r[mask], lbl)
    print(f"  PROD head : AUROC={a_prod:.3f}")
    print(f"  NEW  head : AUROC={a_new:.3f}")

    # tau-sweep: true-loss recall vs false-LA fire-rate
    print("\n" + "=" * 78)
    print("TAU-SWEEP  (gate = prob>=tau)   true-loss RECALL  /  false-LA FIRE-rate")
    print(f"  {'tau':>6} | {'PROD recall':>11} {'PROD false':>11} | {'NEW recall':>10} {'NEW false':>10}")
    for tau in (0.80, 0.90, 0.95, 0.97, 0.99, 0.995, 0.999):
        def rr(arr, m):
            v = arr[m]; v = v[np.isfinite(v)]
            return np.mean(v >= tau) if len(v) else float("nan")
        print(f"  {tau:>6.3f} | {rr(prod_r,true_loss):>11.3f} {rr(prod_r,false_la):>11.3f} | "
              f"{rr(new_r,true_loss):>10.3f} {rr(new_r,false_la):>10.3f}")
    print("\nGOAL: a tau where NEW recall>=~0.7 on true-loss while NEW false<=~0.2 on false-LA.")
    print("If PROD can't separate (recall and false move together) but NEW can, the retrain worked.")


if __name__ == "__main__":
    main()
