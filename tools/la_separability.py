#!/usr/bin/env python
"""THE gate-feasibility question: within CSC-predicted LA frames, does ANY
runtime signal separate true-loss (IoU<0.2, recoverable) from false-LA
(IoU>=0.5, must-not-touch)?

If no signal separates, every gated re-detection approach is doomed (re-confirms
the precision wall). If something separates, it tells us WHICH signal to gate on.

We test every per-frame telemetry feature + the CSC head outputs (risk_score,
lost_aware_next_10_prob, localization LA-prob) as a discriminator via rank-AUROC.
Reported globally AND within the HARD difficulty tercile (the slice that matters:
EASY/MEDIUM LA is almost all false-LA, so the real test is separating inside HARD).

UAV123 clamped SGLATrack passive run — already on disk, no re-run.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

# Replicate run_with_csc.py sys.path so src/ csc_uav_tracking + csc_lib resolve
# (src/ is the canonical lib path).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

sys.argv = [sys.argv[0]]  # keep imported arg parsers happy
import csc_uav_tracking  # noqa: F401
from csc_uav_tracking.registry import DATASETS
from csc_lib.eval.custom_metrics.tracking_metrics import compute_per_frame_arrays

RUN = Path("outputs/eval5_clamp/csc/sglatrack/uav123/test/passive")
TRUE_LA_IOU = 0.20    # genuine loss — recoverable headroom
FALSE_LA_IOU = 0.50   # CSC wrong, target fine — must NOT intervene
EXCLUDE_TEL = {"frame_idx", "init", "latency_ms"}


def gt_array(seq):
    out = []
    for bb in seq.ground_truth:
        ok = bb is not None and getattr(bb, "valid", True)
        out.append([float(bb.x), float(bb.y), float(bb.w), float(bb.h)] if ok else [0.0] * 4)
    return np.asarray(out, float)


def read_preds(p):
    rows = []
    for ln in open(p):
        ln = ln.strip()
        v = [float(x) for x in ln.replace("\t", ",").split(",")[:4] if x] if ln else []
        rows.append((v + [0.0] * 4)[:4])
    return np.asarray(rows, float)


def read_states(p, n):
    """Return (derived_state[n], extra_signals dict of name->array[n])."""
    st = np.full(n, -1, int)
    extra = {k: np.full(n, np.nan) for k in
             ("risk_score", "lost_aware_next_10_prob", "failure_next_10_prob",
              "false_confirmed_next_10_prob", "loc_LA_prob", "derived_LA_prob")}
    if not p.exists():
        return st, extra
    for ln in open(p):
        if not ln.strip():
            continue
        d = json.loads(ln); t = int(d.get("frame_idx", -1))
        if not (0 <= t < n):
            continue
        v = d.get("derived_state")
        if v is not None:
            st[t] = int(v)
        for k in ("risk_score", "lost_aware_next_10_prob", "failure_next_10_prob",
                  "false_confirmed_next_10_prob"):
            if d.get(k) is not None:
                extra[k][t] = float(d[k])
        lp = d.get("localization_probs")
        if isinstance(lp, list) and len(lp) >= 3:
            extra["loc_LA_prob"][t] = float(lp[2])
        dp = d.get("derived_probs")
        if isinstance(dp, list) and len(dp) >= 3:
            extra["derived_LA_prob"][t] = float(dp[2])
    return st, extra


def read_telemetry(p, n, feats):
    out = {k: np.full(n, np.nan) for k in feats}
    if not p.exists():
        return out
    for ln in open(p):
        if not ln.strip():
            continue
        d = json.loads(ln); t = int(d.get("frame_idx", -1))
        if not (0 <= t < n):
            continue
        for k in feats:
            v = d.get(k)
            if isinstance(v, (int, float)):
                out[k][t] = float(v)
    return out


def rankdata_avg(a):
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i, m = 0, len(a)
    while i < m:
        j = i
        while j + 1 < m and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(pos, neg):
    """AUROC for separating pos (true-LA) from neg (false-LA). 1=perfect (high
    feature -> true-LA), 0=perfectly inverted (low feature -> true-LA), 0.5=none."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = rankdata_avg(allv)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


# discover telemetry feature names from first content line of any file
def discover_feats():
    for f in sorted((RUN / "telemetry").glob("*.jsonl")):
        for ln in open(f):
            d = json.loads(ln)
            if d.get("init"):
                continue
            return sorted(k for k, v in d.items()
                          if k not in EXCLUDE_TEL and isinstance(v, (int, float)))
    return []


print("loading uav123 ...", file=sys.stderr)
ds = list(DATASETS.build("uav123", split="test"))
TEL_FEATS = discover_feats()
print(f"  telemetry feats: {TEL_FEATS}", file=sys.stderr)

# per-seq collection
seqs = []  # (name, iou[n], state[n], difficulty, {feat: arr})
for seq in ds:
    pf = RUN / "predictions" / f"{seq.name}.txt"
    if not pf.exists():
        continue
    gt = gt_array(seq); pr = read_preds(pf); n = min(len(pr), len(gt))
    if n == 0:
        continue
    try:
        first = next(iter(seq.frames)); diag = float(np.hypot(*first.shape[1::-1]))
    except Exception:
        diag = 1280.0
    ious, _, _ = compute_per_frame_arrays(pr[:n], gt[:n], image_diag=diag)
    st, extra = read_states(RUN / "states" / f"{seq.name}.jsonl", n)
    tel = read_telemetry(RUN / "telemetry" / f"{seq.name}.jsonl", n, TEL_FEATS)
    feat = {**tel, **extra}
    fin = np.isfinite(ious)
    diff = float((ious[fin] < 0.2).mean()) if fin.any() else 0.0
    seqs.append((seq.name, ious, st, diff, feat, fin))

ALL_FEATS = TEL_FEATS + ["risk_score", "lost_aware_next_10_prob", "failure_next_10_prob",
                         "false_confirmed_next_10_prob", "loc_LA_prob", "derived_LA_prob"]

# difficulty terciles
seqs_sorted = sorted(seqs, key=lambda s: s[3]); t = len(seqs_sorted) // 3
TERCILE = {nm: set() for nm in ("EASY", "MEDIUM", "HARD")}
for i, s in enumerate(seqs_sorted):
    nm = "EASY" if i < t else ("MEDIUM" if i < 2 * t else "HARD")
    TERCILE[nm].add(s[0])


def collect(scope):
    """Return dict feat->(pos_vals, neg_vals) for LA frames in scope ('ALL' or tercile),
    plus counts (n_true, n_false, n_amb, n_la)."""
    pos = {f: [] for f in ALL_FEATS}; neg = {f: [] for f in ALL_FEATS}
    n_true = n_false = n_amb = n_la = 0
    for name, ious, st, diff, feat, fin in seqs:
        if scope != "ALL" and name not in TERCILE[scope]:
            continue
        la = (st == 2) & fin
        for idx in np.where(la)[0]:
            n_la += 1
            iou = ious[idx]
            if iou < TRUE_LA_IOU:
                bucket, n_true = "pos", n_true + 1
            elif iou >= FALSE_LA_IOU:
                bucket, n_false = "neg", n_false + 1
            else:
                n_amb += 1; continue
            for f in ALL_FEATS:
                (pos if bucket == "pos" else neg)[f].append(feat[f][idx])
    return pos, neg, (n_true, n_false, n_amb, n_la)


def report_scope(scope, L):
    pos, neg, (n_true, n_false, n_amb, n_la) = collect(scope)
    L.append(f"\n## {scope}  — LA frames: {n_la}  (true-loss<{TRUE_LA_IOU}: {n_true}, "
             f"false-LA>={FALSE_LA_IOU}: {n_false}, ambiguous: {n_amb})")
    if n_true < 10 or n_false < 10:
        L.append(f"> too few in one bucket for a stable AUROC (need >=10 each).")
        if n_true < 10 and scope != "EASY":
            return
    rows = []
    for f in ALL_FEATS:
        a = auroc(pos[f], neg[f])
        if np.isnan(a):
            continue
        sep = abs(a - 0.5)
        pv = np.asarray(pos[f], float); nv = np.asarray(neg[f], float)
        pv = pv[np.isfinite(pv)]; nv = nv[np.isfinite(nv)]
        rows.append((sep, a, f, np.median(pv) if len(pv) else np.nan,
                     np.median(nv) if len(nv) else np.nan))
    rows.sort(reverse=True)
    L.append("| feature | AUROC | separation | median(true-LA) | median(false-LA) | direction |")
    L.append("|---|---|---|---|---|---|")
    for sep, a, f, mt, mf in rows:
        direction = "high→true-loss" if a > 0.5 else "low→true-loss"
        flag = " **<-- strong**" if sep >= 0.20 else (" *<- weak*" if sep >= 0.12 else "")
        L.append(f"| {f} | {a:.3f} | {sep:.3f} | {mt:.4f} | {mf:.4f} | {direction}{flag} |")


L = ["# LA gate-feasibility — can any runtime signal separate true-loss from false-LA?\n",
     f"Within CSC-predicted LA frames (UAV123 clamped SGLATrack passive). "
     f"positive=true-loss (IoU<{TRUE_LA_IOU}, recoverable), negative=false-LA "
     f"(IoU>={FALSE_LA_IOU}, must-not-touch). AUROC=0.5 -> no signal (wall confirmed); "
     f">=0.70 (separation>=0.20) -> a usable gate exists.\n"]
for scope in ("ALL", "HARD", "MEDIUM", "EASY"):
    report_scope(scope, L)

out = "\n".join(L)
print(out)
Path("outputs/eval6_matrix").mkdir(parents=True, exist_ok=True)
Path("outputs/eval6_matrix/LA_separability.md").write_text(out)
print("\n[saved] outputs/eval6_matrix/LA_separability.md", file=sys.stderr)
