#!/usr/bin/env python
"""Aggregate the CLAMP-CORRECTED baseline matrix — UAV123 (final test).

Per tracker: corrected (clamped) AUC/Pr@20 vs prior un-clamped, ΔAUC from the
bbox-clamp fix, plus CSC-runtime diagnostic (state dist, FCR/FCD/Recovery) and
GT-ruled FCR. All rows are PASSIVE runs (CSC observes only) so the tracking AUC
== bare AUC while the same run yields the CSC states.

Sources (all CLAMPED unless noted):
  sglatrack : outputs/eval5_clamp/csc/sglatrack/uav123/test/passive
  ortrack   : outputs/eval6_matrix/csc/ortrack/uav123/test/passive   (clamp added)
  avtrack   : outputs/eval6_matrix/csc/avtrack/uav123/test/passive   (clamp added)
  ostrack   : outputs/eval/ostrack/uav123/test/ostrack_r3_passive    (native _clip_box; no bug)
Prior un-clamped (for ΔAUC): outputs/eval/<t>/uav123/test/<t>_r3_passive
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

DS, SPLIT = "uav123", "test"
NAMES = {0: "CC", 1: "CU", 2: "LA", 3: "FC"}
OUT = Path("outputs/eval6_matrix/SUMMARY_clamp_matrix.md")

# tracker -> (clamped_run_dir, old_unclamped_dir, had_bug)
RUNS = {
    "sglatrack": (Path("outputs/eval5_clamp/csc/sglatrack")/DS/SPLIT/"passive",
                  Path("outputs/eval/sglatrack")/DS/SPLIT/"sglatrack_r3_passive", True),
    "ortrack":   (Path("outputs/eval6_matrix/csc/ortrack")/DS/SPLIT/"passive",
                  Path("outputs/eval/ortrack")/DS/SPLIT/"ortrack_r3_passive", True),
    "avtrack":   (Path("outputs/eval6_matrix/csc/avtrack")/DS/SPLIT/"passive",
                  Path("outputs/eval/avtrack")/DS/SPLIT/"avtrack_r3_passive", True),
    "ostrack":   (Path("outputs/eval/ostrack")/DS/SPLIT/"ostrack_r3_passive",
                  Path("outputs/eval/ostrack")/DS/SPLIT/"ostrack_r3_passive", False),
}

def track_auc(run: Path):
    f = run/"tracking_metrics"/"metrics_summary.json"
    if not f.exists(): return None, None, None
    s = json.loads(f.read_text()); m = s.get("macro", {})
    return m.get("auc"), m.get("precision_20"), s.get("frame_weighted", {}).get("auc")

def derived_states(run: Path):
    d = run/"states"; out = {}
    if not d.is_dir(): return out
    for f in sorted(d.glob("*.jsonl")):
        seq = [int(json.loads(l)["derived_state"]) for l in f.read_text().splitlines()
               if l.strip() and json.loads(l).get("derived_state") is not None]
        out[f.stem] = seq
    return out

def teacher_states(run: Path):
    base = run/"labels_v3"/DS/SPLIT; out = {}
    psd = base/"labels_per_sequence"
    if psd.is_dir():
        for f in sorted(psd.glob("*.jsonl")):
            out[f.stem] = [int(json.loads(l)["derived_state"]) for l in f.read_text().splitlines()
                           if l.strip() and json.loads(l).get("derived_state") is not None]
        return out
    flat = base/"labels.jsonl"
    if flat.exists():
        cur, name = [], None
        for l in flat.read_text().splitlines():
            if not l.strip(): continue
            d = json.loads(l); nm = d.get("sequence") or d.get("seq") or d.get("name")
            if nm != name:
                if name is not None: out[name] = cur
                name, cur = nm, []
            v = d.get("derived_state")
            if v is not None: cur.append(int(v))
        if name is not None: out[name] = cur
    return out

def state_dist(per_seq):
    from collections import Counter
    flat = [s for seq in per_seq.values() for s in seq]; n = len(flat) or 1
    c = Counter(flat)
    return {NAMES[k]: 100.0*c.get(k, 0)/n for k in range(4)}

def fcr_fcd_rec(per_seq, k=30):
    total = fc = 0; runs = []; er = et = 0
    for seq in per_seq.values():
        total += len(seq); fc += sum(1 for s in seq if s == 3)
        r = 0
        for s in seq:
            if s == 3: r += 1
            elif r: runs.append(r); r = 0
        if r: runs.append(r)
        i, L = 0, len(seq)
        while i < L:
            if seq[i] in (2, 3):
                j = i
                while j < L and seq[j] in (2, 3): j += 1
                et += 1
                if any(seq[t] == 0 for t in range(j, min(j+k, L))): er += 1
                i = j
            else: i += 1
    return (fc/total if total else 0.0,
            float(np.mean(runs)) if runs else 0.0, er/et if et else 0.0)

def paper_runtime(run: Path):
    f = run/"paper_metrics"/"paper_metrics.csv"
    if not f.exists(): return None
    rows = list(csv.DictReader(f.open()))
    agg = [r for r in rows if r.get("sequence") == "__aggregate__"]
    r = agg[0] if agg else (rows[-1] if rows else {})
    def g(k):
        try: return float(r[k])
        except Exception: return None
    return {"fcr": g("fcr"), "fcd": g("fcd"), "rec": g("recovery_at_30")}

def f(x, n=3): return "—" if x is None else f"{x:.{n}f}"

def main():
    rows = []
    for t, (clamped, old, had_bug) in RUNS.items():
        auc, pr20, aucfw = track_auc(clamped)
        old_auc, old_pr20, _ = track_auc(old)
        ds = derived_states(clamped); dist = state_dist(ds) if ds else {}
        pr = paper_runtime(clamped)
        if pr and pr["fcr"] is not None:
            fcr, fcd, rec = pr["fcr"], pr["fcd"], pr["rec"]
        else:
            fcr, fcd, rec = fcr_fcd_rec(ds) if ds else (None, None, None)
        tch = teacher_states(clamped); gt_fcr = fcr_fcd_rec(tch)[0] if tch else None
        d_auc = (auc-old_auc) if (auc is not None and old_auc is not None and had_bug) else (0.0 if not had_bug else None)
        rows.append(dict(t=t, auc=auc, pr20=pr20, aucfw=aucfw, old_auc=old_auc,
                         old_pr20=old_pr20, d_auc=d_auc, had_bug=had_bug, dist=dist,
                         fcr=fcr, fcd=fcd, rec=rec, gt_fcr=gt_fcr,
                         ready=clamped.joinpath("tracking_metrics/metrics_summary.json").exists()))
    L = ["# CLAMP-corrected baseline + CSC diagnostic matrix — UAV123 (final test)\n",
         "All rows = PASSIVE runs on the CLAMPED tracker (CSC observes only; tracking "
         "AUC == bare AUC). The bbox clamp-to-frame fix restores faithful behaviour "
         "(off-frame drift was scoring IoU~0). ostrack clipped natively (no bug).\n",
         "## Corrected tracking baseline (the paper's main-table numbers)\n",
         "| Tracker | AUC (un-clamped) | AUC (clamped) | ΔAUC | Pr@20 (un→clamped) | clamp |",
         "|---|---|---|---|---|---|"]
    for r in rows:
        clamp = "added" if r["had_bug"] else "native (no change)"
        d_auc_s = "—" if r["d_auc"] is None else f"{r['d_auc']:+.4f}"
        L.append(f"| {r['t']} | {f(r['old_auc'])} | **{f(r['auc'])}** | {d_auc_s} | "
                 f"{f(r['old_pr20'])}→{f(r['pr20'])} | {clamp} |")
    L += ["",
          "## CSC-runtime diagnostic on the clamped trackers\n",
          "| Tracker | %CC | %CU | %LA | %FC | FCR(CSC) | FCD | Rec@30 | FCR(GT) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = r["dist"]
        L.append(f"| {r['t']} | {f(d.get('CC'),1)} | {f(d.get('CU'),1)} | {f(d.get('LA'),1)} | "
                 f"{f(d.get('FC'),1)} | {f(r['fcr'],4)} | {f(r['fcd'],2)} | {f(r['rec'],3)} | {f(r['gt_fcr'],4)} |")
    notready = [r["t"] for r in rows if not r["ready"]]
    if notready:
        L.append(f"\n> ⏳ NOT READY (run still in progress / missing): {', '.join(notready)}")
    out = "\n".join(L); print(out)
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(out)
    print(f"\n[saved] {OUT}", file=sys.stderr)

if __name__ == "__main__":
    main()
