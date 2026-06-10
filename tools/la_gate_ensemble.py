#!/usr/bin/env python
"""Gate-ensemble comparison on the LA true-loss vs false-LA population.

We established the lost_aware_next_10 forecast head separates true-loss(IoU<0.2)
from false-LA(IoU>=0.5) within LA frames at AUROC~0.73 — worse than the structural
gate (~0.85). This script measures, on the SAME population, the best achievable
gate: per-feature AUROC for every runtime telemetry signal, the deployed K-of-N
structural vote, and a grouped-CV logistic ensemble (structural-only vs
structural+forecast) — to decide the strongest "act vs do-nothing" gate and
whether the de-saturated forecast head ADDS anything.

Offline analysis on existing passive telemetry; UAV123 used as exploratory probe.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "salrtd" / "src", PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "tools"):
    sys.path.insert(0, str(_p))
_argv = sys.argv
sys.argv = [sys.argv[0]]
import csc_uav_tracking  # noqa: F401
import la_smoke as ls
from csc_lib.csc.inference import load_runtime
sys.argv = _argv

NEW = "outputs/csc_training/sglatrack_r3_fcw3lost_w32_tcn32_stage2/checkpoint_best.pth"
PASSIVE = PROJECT_ROOT / "outputs/eval_control_ab_r3/uav123_eh_passive"
CALIB_DIR = PROJECT_ROOT / "outputs/calibration"
CALIB_TAG = "sglatrack_all_v2"

# Telemetry features to evaluate as gate signals (raw, runtime-available).
FEATS = [
    "sm_local_top2_ratio", "last_cosine_sim", "apce", "response_entropy",
    "sm_local_peak_margin", "sm_peak_distance", "psr", "confidence",
    "appearance_drift", "initial_template_sim", "sm_heatmap_mass_topk",
    "sm_n_secondary", "sm_peak_width",
]


def auroc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    m = np.isfinite(scores)
    scores, labels = scores[m], labels[m]
    n_pos = int((labels == 1).sum()); n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    avg = {}; start = 0
    for i, c in enumerate(cnt):
        avg[i] = (start + 1 + start + c) / 2.0; start += c
    ranks = np.array([avg[i] for i in inv])
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def load_telem(path):
    rows = {}
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln); rows[int(r.get("frame_idx", -1))] = r
    return rows


def load_states(path):
    d = {}
    if not path.exists():
        return d
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        r = json.loads(ln); d[int(r.get("frame_idx", -1))] = r.get("derived_state")
    return d


def run_new_forecast(runtime, seq, telem, preds):
    try:
        first = next(iter(seq.frames)); h, w = first.shape[:2]; image_size = (int(w), int(h))
    except Exception:
        image_size = (1280, 720)
    runtime.reset(image_size=image_size)
    out = {}
    for t in sorted(telem):
        row = telem[t]; bbox = None
        if 0 <= t < len(preds):
            b = preds[t]
            if np.all(np.isfinite(b)):
                bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        p = runtime.step(confidence=row.get("confidence"), apce=row.get("apce"),
                         psr=row.get("psr"), pred_bbox=bbox)
        out[t] = (p.lost_aware_next_10_prob, p.failure_next_10_prob)
    return out


def struct_vote(row):
    """Replicate the deployed combined K-of-N vote (count of true-loss-leaning signals)."""
    v = 0
    g = lambda k, d=np.nan: float(row.get(k, d)) if row.get(k) is not None else np.nan
    if g("sm_local_top2_ratio") >= 0.30: v += 1
    if g("apce") <= 110: v += 1
    if g("last_cosine_sim") <= 0.85: v += 1
    if g("response_entropy") >= 4.0: v += 1
    if g("sm_local_peak_margin") <= 0.35: v += 1
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default=NEW)
    ap.add_argument("--state", type=int, default=2, help="derived_state to analyse: 2=LA, 3=FC")
    ap.add_argument("--passive", default=str(PASSIVE), help="passive run dir w/ telemetry/states/predictions")
    ap.add_argument("--no_forecast", action="store_true",
                    help="skip running the model (forecast cols=nan); fast for 123-seq sweeps")
    args = ap.parse_args()
    passive = Path(args.passive)
    ST = int(args.state); SNAME = {2: "LA", 3: "FC"}.get(ST, str(ST))
    print(f"loading uav123 (state={SNAME}, passive={passive}) ...", file=sys.stderr)
    idx = ls.build_index()
    rt = None
    if not args.no_forecast:
        rt = load_runtime(Path(args.new), device="cpu", calibration_dir=CALIB_DIR, tracker_name=CALIB_TAG)

    tel_dir = passive / "telemetry"; st_dir = passive / "states"; pr_dir = passive / "predictions"
    seqs = sorted(p.stem for p in tel_dir.glob("*.jsonl"))

    rows_feat = []   # per-LA-frame dict of features + forecast + vote
    labels = []      # 1=true-loss, 0=false-LA
    groups = []      # seq name (for grouped CV)
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
        fc = run_new_forecast(rt, seq, telem, preds) if rt is not None else {}
        for t in range(len(ious)):
            iou = ious[t]
            if not np.isfinite(iou):
                continue
            if states.get(t) != ST:           # only target-state frames (LA or FC)
                continue
            if 0.2 <= iou < 0.5:              # ambiguous middle — drop
                continue
            row = telem.get(t, {})
            la10, fail10 = fc.get(t, (np.nan, np.nan))
            rec = {k: (float(row[k]) if row.get(k) is not None else np.nan) for k in FEATS}
            rec["forecast_lost10"] = la10 if la10 is not None else np.nan
            rec["forecast_fail10"] = fail10 if fail10 is not None else np.nan
            rec["struct_vote"] = struct_vote(row)
            rows_feat.append(rec)
            labels.append(1 if iou < 0.2 else 0)
            groups.append(name)

    labels = np.array(labels, int); groups = np.array(groups)
    print(f"\n{SNAME} frames: {len(labels)}  true(IoU<0.2)={int((labels==1).sum())}  false(IoU>=0.5)={int((labels==0).sum())}")
    print(f"seqs contributing: {len(set(groups))}")

    allk = FEATS + ["forecast_lost10", "forecast_fail10", "struct_vote"]
    print("\n=== single-signal AUROC (true-loss vs false-LA, oriented higher=>true-loss) ===")
    scored = []
    for k in allk:
        x = np.array([r[k] for r in rows_feat], float)
        a = auroc(x, labels)
        if np.isfinite(a) and a < 0.5:
            a = 1.0 - a; ori = "(inv)"
        else:
            ori = "     "
        scored.append((a, k, ori))
    for a, k, ori in sorted(scored, reverse=True, key=lambda z: (-1 if np.isnan(z[0]) else z[0])):
        print(f"  {k:24s} {ori} AUROC={a:.3f}")

    # ---- grouped-CV logistic ensembles ----
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
        have_sk = True
    except Exception as e:
        print(f"\n(sklearn unavailable: {e} — skipping ensemble)"); have_sk = False

    if have_sk and len(set(groups)) >= 3:
        def cv_auroc(featset):
            X = np.array([[r[k] for k in featset] for r in rows_feat], float)
            X = np.nan_to_num(X, nan=0.0)
            y = labels
            n_splits = min(5, len(set(groups)))
            gkf = GroupKFold(n_splits=n_splits)
            oof = np.full(len(y), np.nan)
            for tr, te in gkf.split(X, y, groups):
                if len(set(y[tr])) < 2:
                    continue
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(max_iter=2000, class_weight="balanced")
                clf.fit(sc.transform(X[tr]), y[tr])
                oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
            return auroc(oof, y)

        struct = ["sm_local_top2_ratio", "last_cosine_sim", "apce", "response_entropy", "sm_local_peak_margin"]
        print("\n=== grouped-CV logistic ensemble AUROC (out-of-fold, honest) ===")
        print(f"  structural(5)            : {cv_auroc(struct):.3f}")
        print(f"  structural + forecast    : {cv_auroc(struct + ['forecast_lost10','forecast_fail10']):.3f}")
        print(f"  structural + ALL telem   : {cv_auroc(struct + ['sm_peak_distance','psr','confidence','appearance_drift','initial_template_sim','sm_heatmap_mass_topk','sm_n_secondary','sm_peak_width']):.3f}")
        print(f"  ALL signals (incl fcast) : {cv_auroc(allk[:-1]):.3f}")
        print(f"  forecast-only(2)         : {cv_auroc(['forecast_lost10','forecast_fail10']):.3f}")

    # struct_vote operating points
    print("\n=== struct_vote operating points (deployed gate) ===")
    sv = np.array([r["struct_vote"] for r in rows_feat], int)
    for k in range(1, 6):
        fire = sv >= k
        rec = np.mean(fire[labels == 1]) if (labels == 1).any() else np.nan
        fls = np.mean(fire[labels == 0]) if (labels == 0).any() else np.nan
        print(f"  vote>={k}: true-loss recall={rec:.3f}  false-LA fire={fls:.3f}")

    # last_cosine_sim is the dominant single signal (AUROC ~0.925) — pick its operating point.
    print("\n=== last_cosine_sim threshold sweep (gate = cosine <= tau) ===")
    cosv = np.array([r["last_cosine_sim"] for r in rows_feat], float)
    print(f"  {'tau':>7} | {'recall(true)':>12} {'fire(false-LA)':>14} {'fires of all LA':>16}")
    for tau in (0.65, 0.70, 0.75, 0.78, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925):
        fire = cosv <= tau
        rec = np.mean(fire[labels == 1]) if (labels == 1).any() else float("nan")
        fls = np.mean(fire[labels == 0]) if (labels == 0).any() else float("nan")
        print(f"  {tau:>7.3f} | {rec:>12.3f} {fls:>14.3f} {fire.mean():>16.3f}")
    print("  GOAL: tau with recall(true)>=~0.8 AND fire(false-LA)<=~0.15.")


if __name__ == "__main__":
    main()
