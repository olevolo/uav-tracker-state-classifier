#!/usr/bin/env python
"""Pick the HARD-true-LA gate operating point OFFLINE (pure telemetry function —
no tracker runs needed). For every gate preset/threshold, over all CSC-predicted
LA frames of the UAV123 passive run, report:
  - true-LA recall   : fires / (true-loss LA frames, IoU<0.2)   — want HIGH
  - false-LA firerate: fires / (false-LA frames, IoU>=0.5)      — want LOW
  - uav6 firerate    : fires on the canonical false-LA guard    — want ~0
  - EASY/MED false fire-rate (the do-no-harm constraint)
The gate is exactly run_with_csc.py:_hard_la_gate, so the chosen config drops
straight into --gate_preset / --gate_* without re-deriving anything.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
_argv = sys.argv
sys.argv = [sys.argv[0]]
import csc_uav_tracking  # noqa: F401
from csc_uav_tracking.registry import DATASETS
from csc_lib.eval.custom_metrics.tracking_metrics import compute_per_frame_arrays
sys.argv = _argv

RUN = PROJECT_ROOT / "outputs/eval5_clamp/csc/sglatrack/uav123/test/passive"
TRUE_IOU, FALSE_IOU = 0.20, 0.50
HARD_GT_FAIL = 0.30


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


def load_frame_records():
    """Return list of per-LA-frame dicts: {feat..., iou, bucket, seq, hard}."""
    ds = list(DATASETS.build("uav123", split="test"))
    recs = []
    for seq in ds:
        pf = RUN / "predictions" / f"{seq.name}.txt"
        sf = RUN / "states" / f"{seq.name}.jsonl"
        tf = RUN / "telemetry" / f"{seq.name}.jsonl"
        if not (pf.exists() and sf.exists() and tf.exists()):
            continue
        gt = gt_array(seq); pr = read_preds(pf); n = min(len(pr), len(gt))
        if n == 0:
            continue
        try:
            first = next(iter(seq.frames)); diag = float(np.hypot(*first.shape[1::-1]))
        except Exception:
            diag = 1280.0
        ious, _, _ = compute_per_frame_arrays(pr[:n], gt[:n], image_diag=diag)
        fin = np.isfinite(ious)
        gtfail = float((ious[fin] < 0.2).mean()) if fin.any() else 0.0
        hard = gtfail >= HARD_GT_FAIL
        states = {}
        for ln in open(sf):
            if not ln.strip():
                continue
            d = json.loads(ln); t = int(d.get("frame_idx", -1))
            if 0 <= t < n:
                states[t] = (d.get("derived_state"), d.get("lost_aware_next_10_prob"))
        tel = {}
        for ln in open(tf):
            if not ln.strip():
                continue
            d = json.loads(ln); t = int(d.get("frame_idx", -1))
            if 0 <= t < n:
                tel[t] = d
        for t in range(n):
            sd = states.get(t)
            if not sd or sd[0] != 2 or not np.isfinite(ious[t]):
                continue  # only CSC-predicted LA frames
            iou = float(ious[t])
            bucket = "true" if iou < TRUE_IOU else ("false" if iou >= FALSE_IOU else "amb")
            rec = dict(tel.get(t, {}))
            rec["lost_aware_next_10_prob"] = sd[1]
            rec["iou"] = iou; rec["bucket"] = bucket; rec["seq"] = seq.name; rec["hard"] = hard
            recs.append(rec)
    return recs


def gate(rec, preset, k=3, top2=0.30, apce=110.0, cos=0.85, ent=4.0, pm=0.35, la10=0.99):
    g = rec.get
    top2r, a, c, e, pmg = g("sm_local_top2_ratio"), g("apce"), g("last_cosine_sim"), g("response_entropy"), g("sm_local_peak_margin")
    l = g("lost_aware_next_10_prob")
    if preset == "csc_head":
        return l is not None and float(l) >= la10
    if preset == "appearance":
        return (c is not None and c <= cos) and (a is not None and a <= apce)
    if preset == "response":
        return (top2r is not None and top2r >= top2) and (e is not None and e >= ent) and (pmg is not None and pmg <= pm)
    v = 0
    if top2r is not None and top2r >= top2: v += 1
    if a is not None and a <= apce: v += 1
    if c is not None and c <= cos: v += 1
    if e is not None and e >= ent: v += 1
    if pmg is not None and pmg <= pm: v += 1
    return v >= k


def evaluate(recs, label, **kw):
    preset = kw.pop("preset")
    buckets = {"true": [0, 0], "false": [0, 0], "easyfalse": [0, 0], "medfalse": [0, 0], "uav6": [0, 0]}
    for r in recs:
        fired = gate(r, preset, **kw)
        b = r["bucket"]
        if b == "true":
            buckets["true"][0] += fired; buckets["true"][1] += 1
        elif b == "false":
            buckets["false"][0] += fired; buckets["false"][1] += 1
            if not r["hard"]:
                buckets["easyfalse"][0] += fired; buckets["easyfalse"][1] += 1
        if r["seq"] == "uav6":
            buckets["uav6"][0] += fired; buckets["uav6"][1] += 1

    def rate(b):
        f, n = buckets[b]
        return (f / n) if n else float("nan"), n
    tr, trn = rate("true"); fa, fan = rate("false"); u6, u6n = rate("uav6"); ef, efn = rate("easyfalse")
    return (f"| {label:<34} | {tr:5.2f} ({trn}) | {fa:5.2f} ({fan}) | "
            f"{ef:5.2f} ({efn}) | {u6:5.2f} ({u6n}) |")


print("loading uav123 + telemetry ...", file=sys.stderr)
recs = load_frame_records()
n_true = sum(1 for r in recs if r["bucket"] == "true")
n_false = sum(1 for r in recs if r["bucket"] == "false")
print(f"  {len(recs)} LA frames: {n_true} true-loss, {n_false} false-LA", file=sys.stderr)

L = ["# HARD-true-LA gate operating points (offline, UAV123 passive)\n",
     "Want: true-LA recall HIGH (recover genuine loss), false-LA/uav6/easy-false rate LOW "
     "(do no harm). (n) = #frames in that bucket.\n",
     "| config | true recall | false-LA fire | EASY-false fire | uav6 fire |",
     "|---|---|---|---|---|"]
# combined: vote-K sweep
for k in (2, 3, 4, 5):
    L.append(evaluate(recs, f"combined K={k} (default thr)", preset="combined", k=k))
# combined tighter thresholds at K=3
L.append(evaluate(recs, "combined K=3 cos<=0.82 apce<=90", preset="combined", k=3, cos=0.82, apce=90.0))
L.append(evaluate(recs, "combined K=4 cos<=0.82 apce<=90", preset="combined", k=4, cos=0.82, apce=90.0))
# appearance sweep
for cos in (0.88, 0.85, 0.82):
    for ap in (150.0, 110.0, 80.0):
        L.append(evaluate(recs, f"appearance cos<={cos} apce<={ap:.0f}", preset="appearance", cos=cos, apce=ap))
# response (defaults)
L.append(evaluate(recs, "response (default thr)", preset="response"))
L.append(evaluate(recs, "response top2>=0.4 ent>=4.2", preset="response", top2=0.40, ent=4.2))
# csc_head sweep
for la in (0.97, 0.99, 0.995, 0.999):
    L.append(evaluate(recs, f"csc_head lostaware>={la}", preset="csc_head", la10=la))

out = "\n".join(L)
print(out)
Path("outputs/eval6_matrix/LA_gate_tune.md").write_text(out)
print("\n[saved] outputs/eval6_matrix/LA_gate_tune.md", file=sys.stderr)
