#!/usr/bin/env python
"""Aggregate the CLAMP + LOST-action UAV123 comparison (tools/run_clamp_compare.sh).

Reads outputs/eval5_clamp/: bare / passive / ctrl_hold / ctrl_widen (all CLAMPED tracker).
Also reads the prior UN-clamped bare (outputs/eval4_fresh/bare) to quantify the
bbox-clamp's effect on the baseline. Outputs main table + GT-FCR + per-seq
easy/hard tercile ΔAUC + regression counts. Saves SUMMARY_clamp.md.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("outputs/eval5_clamp")
DS, SPLIT = "uav123", "test"
NAMES = {0: "CC", 1: "CU", 2: "LA", 3: "FC"}
UNCLAMPED_BARE = Path("outputs/eval4_fresh/bare/sglatrack") / DS / SPLIT  # for clamp delta

RUNS = [
    ("1. bare (clamped)",       ROOT / "bare/sglatrack" / DS / SPLIT,                 "bare"),
    ("2. CSC passive",          ROOT / "csc/sglatrack" / DS / SPLIT / "passive",      "csc"),
    ("3. ctrl HOLD-on-LA (safe)",ROOT / "csc/sglatrack" / DS / SPLIT / "ctrl_hold",   "csc"),
    ("4. ctrl WIDEN-on-LA (abl)",ROOT / "csc/sglatrack" / DS / SPLIT / "ctrl_widen",  "csc"),
]
PROBE = ROOT / "fps_probe"

def track_metrics(run: Path):
    f = run / "tracking_metrics" / "metrics_summary.json"
    if not f.exists(): return None
    s = json.loads(f.read_text()); m = s.get("macro", {})
    return {"auc": m.get("auc"), "pr20": m.get("precision_20"),
            "auc_fw": s.get("frame_weighted", {}).get("auc")}

def per_seq_auc(run: Path):
    f = run / "tracking_metrics" / "metrics_per_sequence.csv"
    if not f.exists(): return {}
    out = {}
    for r in csv.DictReader(f.open()):
        try:
            out[r["sequence"]] = {"auc": float(r["auc"]), "pr20": float(r["precision_20"]),
                                  "n": int(r["n_frames"]), "fail": int(r["total_failure_frames"])}
        except Exception: pass
    return out

def derived_states(states_dir: Path):
    out = {}
    if not states_dir.is_dir(): return out
    for f in sorted(states_dir.glob("*.jsonl")):
        seq = []
        for ln in f.read_text().splitlines():
            if not ln.strip(): continue
            v = json.loads(ln).get("derived_state")
            if v is not None: seq.append(int(v))
        out[f.stem] = seq
    return out

def teacher_states(run: Path):
    base = run / "labels_v3" / DS / SPLIT
    out = {}; psd = base / "labels_per_sequence"
    if psd.is_dir():
        for f in sorted(psd.glob("*.jsonl")):
            seq = [int(json.loads(l).get("derived_state", -1)) for l in f.read_text().splitlines() if l.strip()]
            out[f.stem] = [s for s in seq if s >= 0]
        return out
    flat = base / "labels.jsonl"
    if flat.exists():
        cur_name, cur = None, []
        for l in flat.read_text().splitlines():
            if not l.strip(): continue
            d = json.loads(l); nm = d.get("sequence") or d.get("seq") or d.get("name")
            if nm != cur_name:
                if cur_name is not None: out[cur_name] = cur
                cur_name, cur = nm, []
            v = d.get("derived_state")
            if v is not None: cur.append(int(v))
        if cur_name is not None: out[cur_name] = cur
    return out

def state_dist(per_seq):
    from collections import Counter
    flat = [s for seq in per_seq.values() for s in seq]; n = len(flat) or 1
    c = Counter(flat)
    return {NAMES[k]: 100.0 * c.get(k, 0) / n for k in range(4)}

def fcr_fcd_rec(per_seq, k=30):
    total = fc = 0; fc_runs = []; er = et = 0
    for seq in per_seq.values():
        total += len(seq); fc += sum(1 for s in seq if s == 3)
        run = 0
        for s in seq:
            if s == 3: run += 1
            elif run: fc_runs.append(run); run = 0
        if run: fc_runs.append(run)
        i, L = 0, len(seq)
        while i < L:
            if seq[i] in (2, 3):
                j = i
                while j < L and seq[j] in (2, 3): j += 1
                et += 1
                if any(seq[t] == 0 for t in range(j, min(j + k, L))): er += 1
                i = j
            else: i += 1
    return (fc/total if total else 0.0,
            float(np.mean(fc_runs)) if fc_runs else 0.0,
            er/et if et else 0.0)

def paper_runtime(run: Path):
    f = run / "paper_metrics" / "paper_metrics.csv"
    if not f.exists(): return None
    rows = list(csv.DictReader(f.open()))
    agg = [r for r in rows if r.get("sequence") == "__aggregate__"]
    r = agg[0] if agg else (rows[-1] if rows else {})
    def g(k):
        try: return float(r[k])
        except Exception: return None
    return {"fcr": g("fcr"), "fcd": g("fcd"), "rec": g("recovery_at_30")}

def fps_of(manifest_or_metrics: Path):
    if not manifest_or_metrics.exists(): return None
    s = json.loads(manifest_or_metrics.read_text())
    if "mean_fps" in s: return s["mean_fps"]
    return s.get("mean_total_fps") or s.get("mean_tracker_fps")

def fmt(x, n=3): return "—" if x is None else f"{x:.{n}f}"

def main():
    rows = []
    per_seq_by = {}
    gt_fcr = {}
    for label, run, kind in RUNS:
        tm = track_metrics(run); psa = per_seq_auc(run); per_seq_by[label] = psa
        if kind == "bare":
            st = teacher_states(run); src = "Teacher(GT)"
            fcr, fcd, rec = fcr_fcd_rec(st) if st else (None, None, None)
        else:
            st = derived_states(run / "states"); src = "CSC runtime"
            pr = paper_runtime(run)
            fcr, fcd, rec = (pr["fcr"], pr["fcd"], pr["rec"]) if (pr and pr["fcr"] is not None) \
                            else (fcr_fcd_rec(st) if st else (None, None, None))
        dist = state_dist(st) if st else {}
        tgt = teacher_states(run)
        if tgt: gt_fcr[label] = fcr_fcd_rec(tgt)[0]
        rows.append({"label": label, "src": src,
                     "auc": tm["auc"] if tm else None, "pr20": tm["pr20"] if tm else None,
                     "auc_fw": tm["auc_fw"] if tm else None,
                     "dist": dist, "fcr": fcr, "fcd": fcd, "rec": rec})

    L = []
    L.append("# CLAMP + LOST-action comparison — UAV123 (final test)\n")
    # clamp delta vs un-clamped bare
    f = UNCLAMPED_BARE / "tracking_metrics" / "metrics_summary.json"
    if f.exists():
        old = json.loads(f.read_text()).get("macro", {})
        new = rows[0]["auc"]
        if old.get("auc") is not None and new is not None:
            L.append(f"**bbox-clamp baseline effect:** un-clamped bare AUC "
                     f"{old['auc']:.3f} → clamped bare AUC {new:.3f} "
                     f"(**Δ {new-old['auc']:+.3f}**). The clamp is an unconditional "
                     f"correctness fix (no CSC dependency).\n")
    L.append("State source: row 1 = Teacher(GT); rows 2-4 = CSC runtime.\n")
    L.append("| Run | AUC | AUC(fw) | Pr@20 | %CC | %CU | %LA | %FC | FCR | FCD | Rec@30 | src |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        d = r["dist"]
        L.append("| {lab} | {auc} | {afw} | {pr} | {cc} | {cu} | {la} | {fc} | {fcr} | {fcd} | {rec} | {src} |".format(
            lab=r["label"], auc=fmt(r["auc"]), afw=fmt(r["auc_fw"]), pr=fmt(r["pr20"]),
            cc=fmt(d.get("CC"),1), cu=fmt(d.get("CU"),1), la=fmt(d.get("LA"),1), fc=fmt(d.get("FC"),1),
            fcr=fmt(r["fcr"],4), fcd=fmt(r["fcd"],2), rec=fmt(r["rec"],3), src=r["src"]))
    L.append("")
    L.append("## GT-ruled FCR per run (honest causal measure on each run's bboxes)\n")
    L.append("| Run | FCR(GT) |"); L.append("|---|---|")
    for r in rows: L.append(f"| {r['label']} | {fmt(gt_fcr.get(r['label']),4)} |")
    L.append("")

    bare_ps = per_seq_by.get("1. bare (clamped)", {})
    if bare_ps:
        diff = {s: (v["fail"]/v["n"] if v["n"] else 0.0) for s, v in bare_ps.items()}
        ranked = sorted(diff, key=lambda s: diff[s]); t = len(ranked)//3
        bins = {"EASY": ranked[:t], "MEDIUM": ranked[t:2*t], "HARD": ranked[2*t:]}
        ctrl = [r["label"] for r in rows if "ctrl" in r["label"]]
        L.append(f"## Per-sequence easy/hard ΔAUC vs clamped-bare ({len(ranked)} seqs)\n")
        L.append("Difficulty = bare GT failure-rate (terciles). (+) control helped, (−) hurt.\n")
        L.append("| bin | n | bare AUC | " + " | ".join(c.replace("ctrl ", "") for c in ctrl) + " |")
        L.append("|---|---|---|" + "---|"*len(ctrl))
        for bn, seqs in bins.items():
            ba = np.mean([bare_ps[s]["auc"] for s in seqs]) if seqs else float("nan")
            cells = []
            for c in ctrl:
                cps = per_seq_by.get(c, {})
                ds = [cps[s]["auc"]-bare_ps[s]["auc"] for s in seqs if s in cps]
                cells.append(f"{np.mean(ds):+.4f}" if ds else "—")
            L.append(f"| {bn} | {len(seqs)} | {ba:.3f} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("### Regressed / improved vs bare (|ΔAUC|>0.01)\n")
        L.append("| run | improved | unchanged | regressed | net ΔAUC |"); L.append("|---|---|---|---|---|")
        for c in ctrl:
            cps = per_seq_by.get(c, {})
            ds = [cps[s]["auc"]-bare_ps[s]["auc"] for s in bare_ps if s in cps]
            if not ds: L.append(f"| {c} | — | — | — | — |"); continue
            imp = sum(1 for d in ds if d > 0.01); reg = sum(1 for d in ds if d < -0.01)
            L.append(f"| {c} | {imp} | {len(ds)-imp-reg} | {reg} | {np.mean(ds):+.4f} |")
        L.append("")

    out = "\n".join(L); print(out)
    (ROOT / "SUMMARY_clamp.md").write_text(out)
    print(f"\n[saved] {ROOT/'SUMMARY_clamp.md'}", file=sys.stderr)

if __name__ == "__main__":
    main()
