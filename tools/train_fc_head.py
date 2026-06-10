#!/usr/bin/env python
"""Decoupled geometry FC head (CSC-v4 module).

WHY: a single joint head can't learn FALSE_CONFIRMED — FC needs geometry (bbox
drifted vs init) to separate from CC and confidence/peakiness to separate from LA,
and a joint model shortcuts to one and abandons the other (val AUROC 0.49-0.60).
A PARSIMONIOUS geometry-only classifier generalizes (held-out FC-vs-CC 0.82) where
the 41-feature TCN overfits (0.485). This head is meant to run HIERARCHICALLY:
the main model predicts CC/CU/LA; this head flags FC among the confident (non-LA)
frames. Train-set only (NEVER UAV123). Offline.

Outputs a small joblib bundle (scaler + logistic weights + feature names + a few
calibrated operating-point thresholds) under the out dir.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT):
    sys.path.insert(0, str(_p))
from csc_lib.csc.v4.features_v4 import FEATURE_NAMES_V4

FC, CC, LA = 3, 0, 2
# Geometry / shape-vs-init slots (percentile view — monotone, no clip saturation).
GEOM_FC_FEATURES = [
    "log_w_ratio_to_init_pct", "log_h_ratio_to_init_pct", "log_area_ratio_to_init_pct",
    "aspect_ratio_pct", "conf_ema_trend_pct",
]


def _split(rows, seed, val_frac):
    rng = np.random.default_rng(seed)
    sr = defaultdict(list)
    for r in rows:
        sr[(r["dataset"], r["sequence"])].append(r)
    bk = defaultdict(list)
    for sk, rs in sr.items():
        d = [x["derived"] for x in rs]
        bk[(sk[0], any(x == FC for x in d), any(x == LA for x in d))].append(sk)
    val = set()
    for k in sorted(bk):
        bs = sorted(bk[k]); rng.shuffle(bs); nb = len(bs); kk = int(round(nb * val_frac))
        if nb >= 2:
            kk = min(max(kk, 1), nb - 1)
        val.update(bs[:kk])
    return val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="outputs/csc_labels_v4/train_shards.jsonl")
    ap.add_argument("--out_dir", default="outputs/csc_training_v4/fc_geom_head")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.shards) if l.strip()]
    idx = {n: i for i, n in enumerate(FEATURE_NAMES_V4)}
    cols = [idx[n] for n in GEOM_FC_FEATURES]
    X = np.array([[r[f"feat_{c}"] for c in cols] for r in rows], np.float64)
    Y = np.array([r["derived"] for r in rows])
    val = _split(rows, args.seed, args.val_fraction)
    inval = np.array([(r["dataset"], r["sequence"]) in val for r in rows])

    Xtr, Ytr = X[~inval], Y[~inval]
    Xva, Yva = X[inval], Y[inval]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5).fit(sc.transform(Xtr), (Ytr == FC))
    pva = clf.predict_proba(sc.transform(Xva))[:, 1]

    def auroc(mask):
        m = mask & inval
        y = (Y[m] == FC); s = clf.predict_proba(sc.transform(X[m]))[:, 1]
        return roc_auc_score(y, s) if y.any() and not y.all() else float("nan")
    allm = np.ones(len(Y), bool)
    cc_m = np.isin(Y, [FC, CC])
    la_m = np.isin(Y, [FC, LA])
    nonla_m = Y != LA   # the HIERARCHICAL regime: FC flagged among confident (non-LA) frames

    print(f"feature set ({len(GEOM_FC_FEATURES)}): {GEOM_FC_FEATURES}", file=sys.stderr)
    print(f"train {len(Ytr)} (FC {int((Ytr==FC).sum())}) | val {len(Yva)} (FC {int((Yva==FC).sum())})", file=sys.stderr)
    print("\n=== held-out val AUROC ===")
    print(f"  FC vs ALL          : {auroc(allm):.3f}   AUPRC {average_precision_score((Yva==FC),pva):.3f}")
    print(f"  FC vs CC           : {auroc(cc_m):.3f}   <- safety-critical (looks-confirmed-but-wrong)")
    print(f"  FC vs LA           : {auroc(la_m):.3f}")
    print(f"  FC vs ALL non-LA   : {auroc(nonla_m):.3f}   <- HIERARCHICAL regime (gate LA first)")

    # operating points on the hierarchical (non-LA) regime
    m = nonla_m & inval
    y = (Y[m] == FC); s = clf.predict_proba(sc.transform(X[m]))[:, 1]
    print("\n=== operating points (FC among non-LA val frames) ===")
    print("  thr   recall  precision  #flagged")
    ops = {}
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        pred = s >= thr
        tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
        rec = tp / max(int(y.sum()), 1); prec = tp / max(tp + fp, 1)
        ops[f"{thr:.1f}"] = {"recall": round(rec, 3), "precision": round(prec, 3), "flagged": int(pred.sum())}
        print(f"  {thr:.1f}  {rec:.3f}    {prec:.3f}     {int(pred.sum())}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.dump({"scaler": sc, "clf": clf, "features": GEOM_FC_FEATURES,
                     "feature_cols": cols, "fc_label": FC}, out / "fc_geom_head.joblib")
        saved = "fc_geom_head.joblib"
    except Exception as exc:
        saved = f"(joblib failed: {exc}) — coeffs in json"
    (out / "fc_geom_head.json").write_text(json.dumps({
        "features": GEOM_FC_FEATURES,
        "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
        "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
        "val_auroc_fc_vs_cc": round(auroc(cc_m), 4), "val_auroc_fc_vs_nonla": round(auroc(nonla_m), 4),
        "operating_points_nonLA": ops,
    }, indent=1))
    print(f"\nsaved -> {out}/ ({saved} + fc_geom_head.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
