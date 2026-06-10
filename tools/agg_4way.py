#!/usr/bin/env python
"""Aggregate the from-scratch 4-way UAV123 SGLATrack eval into the final table.

Reads runs produced by tools/run_4way_uav123.sh under outputs/eval4_fresh/:
  bare/...                          (no CSC)      -> Teacher(GT) state metrics
  csc/.../passive                   (CSC observe) -> CSC runtime state metrics
  csc/.../ctrl_pro_default          (block-9 on LA, regresses)
  csc/.../ctrl_re_default
  csc/.../ctrl_pro_hold             (--policy_hold_on_la safeguard)
  csc/.../ctrl_re_hold

Outputs (stdout + outputs/eval4_fresh/SUMMARY_4way.md):
  * main table: AUC / Pr@20 / FPS / %CC %CU %LA %FC / FCR / FCD / Recovery@30
  * GT-Teacher FCR per run (the honest causal measure: did control cut TRUE FC?)
  * per-sequence easy/hard tercile analysis (difficulty = bare GT fail-rate) with
    ΔAUC / ΔPr@20 / ΔFCR(GT) per bin for each control run vs passive/bare.
Robust to missing runs (prints '—').  Run any time; re-run when more cells land.
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path("outputs/eval4_fresh")
DS, SPLIT = "uav123", "test"
NAMES = {0: "CC", 1: "CU", 2: "LA", 3: "FC"}

# (label, run_dir, kind)   kind: 'bare' | 'csc'
RUNS = [
    ("1. bare (no CSC)",        ROOT / "bare/sglatrack" / DS / SPLIT,                 "bare"),
    ("2. CSC passive",          ROOT / "csc/sglatrack" / DS / SPLIT / "passive",      "csc"),
    ("3. ctrl PROACTIVE (def)", ROOT / "csc/sglatrack" / DS / SPLIT / "ctrl_pro_default", "csc"),
    ("4. ctrl REACTIVE (def)",  ROOT / "csc/sglatrack" / DS / SPLIT / "ctrl_re_default",  "csc"),
    ("5. ctrl PROACTIVE +hold", ROOT / "csc/sglatrack" / DS / SPLIT / "ctrl_pro_hold", "csc"),
    ("6. ctrl REACTIVE +hold",  ROOT / "csc/sglatrack" / DS / SPLIT / "ctrl_re_hold",  "csc"),
]
PROBE = ROOT / "fps_probe"
PROBE_FPS = {  # label-substring -> probe manifest/metrics path
    "bare":     PROBE / "bare/sglatrack" / DS / SPLIT / "manifest.json",
    "passive":  PROBE / "csc/sglatrack" / DS / SPLIT / "passive_probe" / "metrics.json",
    "ctrl":     PROBE / "csc/sglatrack" / DS / SPLIT / "ctrl_pro_hold_probe" / "metrics.json",
}

# ----------------------------------------------------------------- readers
def track_metrics(run: Path):
    f = run / "tracking_metrics" / "metrics_summary.json"
    if not f.exists():
        return None
    s = json.loads(f.read_text())
    m = s.get("macro", {})
    return {"auc": m.get("auc"), "pr20": m.get("precision_20"),
            "auc_fw": s.get("frame_weighted", {}).get("auc"), "fps_self": m.get("fps")}

def per_seq_auc(run: Path):
    f = run / "tracking_metrics" / "metrics_per_sequence.csv"
    if not f.exists():
        return {}
    out = {}
    for r in csv.DictReader(f.open()):
        try:
            out[r["sequence"]] = {"auc": float(r["auc"]), "pr20": float(r["precision_20"]),
                                  "n": int(r["n_frames"]), "fail": int(r["total_failure_frames"])}
        except Exception:
            pass
    return out

def derived_states_per_seq(states_dir: Path):
    """Return {seq: [derived_state,...]} from a states/ dir of <seq>.jsonl."""
    out = {}
    if not states_dir.is_dir():
        return out
    for f in sorted(states_dir.glob("*.jsonl")):
        seq = []
        for ln in f.read_text().splitlines():
            if not ln.strip():
                continue
            d = json.loads(ln)
            v = d.get("derived_state")
            if v is not None:
                seq.append(int(v))
        out[f.stem] = seq
    return out

def teacher_states_per_seq(run: Path):
    """{seq: [derived_state,...]} from Teacher(GT) labels_v3."""
    base = run / "labels_v3" / DS / SPLIT
    out = {}
    psd = base / "labels_per_sequence"
    if psd.is_dir():
        for f in sorted(psd.glob("*.jsonl")):
            seq = [int(json.loads(l).get("derived_state", -1))
                   for l in f.read_text().splitlines() if l.strip()]
            out[f.stem] = [s for s in seq if s >= 0]
        return out
    flat = base / "labels.jsonl"
    if flat.exists():
        cur_name, cur = None, []
        for l in flat.read_text().splitlines():
            if not l.strip():
                continue
            d = json.loads(l)
            nm = d.get("sequence") or d.get("seq") or d.get("name")
            if nm != cur_name:
                if cur_name is not None:
                    out[cur_name] = cur
                cur_name, cur = nm, []
            v = d.get("derived_state")
            if v is not None:
                cur.append(int(v))
        if cur_name is not None:
            out[cur_name] = cur
    return out

# ----------------------------------------------------------------- metrics
def state_dist(per_seq_states: dict):
    flat = [s for seq in per_seq_states.values() for s in seq]
    n = len(flat) or 1
    from collections import Counter
    c = Counter(flat)
    return {NAMES[k]: 100.0 * c.get(k, 0) / n for k in range(4)}, n

def fcr_fcd_recovery(per_seq_states: dict, k: int = 30):
    """Compute FCR, FCD, Recovery@K from per-seq derived_state lists.
    FCR = #FC / total.  FCD = mean length of consecutive FC runs.
    Recovery@K = fraction of failure episodes (maximal {LA,FC} runs) whose end is
    followed by a CC within K frames."""
    total = fc = 0
    fc_runs, episodes_rec, episodes_tot = [], 0, 0
    for seq in per_seq_states.values():
        total += len(seq)
        fc += sum(1 for s in seq if s == 3)
        # FC runs
        run = 0
        for s in seq:
            if s == 3:
                run += 1
            elif run:
                fc_runs.append(run); run = 0
        if run:
            fc_runs.append(run)
        # failure episodes = maximal runs of {2,3}
        i, L = 0, len(seq)
        while i < L:
            if seq[i] in (2, 3):
                j = i
                while j < L and seq[j] in (2, 3):
                    j += 1
                episodes_tot += 1
                if any(seq[t] == 0 for t in range(j, min(j + k, L))):
                    episodes_rec += 1
                i = j
            else:
                i += 1
    fcr = fc / total if total else 0.0
    fcd = float(np.mean(fc_runs)) if fc_runs else 0.0
    rec = episodes_rec / episodes_tot if episodes_tot else 0.0
    return fcr, fcd, rec

def paper_runtime(run: Path):
    f = run / "paper_metrics" / "paper_metrics.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(f.open()))
    agg = [r for r in rows if r.get("sequence") == "__aggregate__"]
    r = agg[0] if agg else (rows[-1] if rows else {})
    def g(k):
        try: return float(r[k])
        except Exception: return None
    return {"fcr": g("fcr"), "fcd": g("fcd"), "rec": g("recovery_at_30")}

def fps_lookup(label: str):
    key = "bare" if "bare" in label else ("passive" if "passive" in label else "ctrl")
    f = PROBE_FPS[key]
    if not f.exists():
        return None
    s = json.loads(f.read_text())
    # baseline manifest: mean_fps; csc metrics.json: nested runtime or 'fps'
    if "mean_fps" in s:
        return s["mean_fps"]
    for path in [("runtime", "mean_fps"), ("macro", "fps")]:
        d = s
        ok = True
        for p in path:
            if isinstance(d, dict) and p in d:
                d = d[p]
            else:
                ok = False; break
        if ok and isinstance(d, (int, float)):
            return d
    return s.get("fps")

# ----------------------------------------------------------------- build
def fmt(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"

def main():
    rows_out = []
    teacher_fcr_gt = {}     # label -> GT-ruled FCR on that run's bboxes
    per_seq_by_label = {}   # label -> per_seq_auc dict
    for label, run, kind in RUNS:
        tm = track_metrics(run)
        psa = per_seq_auc(run)
        per_seq_by_label[label] = psa
        # state source
        if kind == "bare":
            st = teacher_states_per_seq(run)
            src = "Teacher(GT)"
            fcr, fcd, rec = fcr_fcd_recovery(st) if st else (None, None, None)
        else:
            st = derived_states_per_seq(run / "states")
            src = "CSC runtime"
            pr = paper_runtime(run)
            if pr and pr["fcr"] is not None:
                fcr, fcd, rec = pr["fcr"], pr["fcd"], pr["rec"]
            else:
                fcr, fcd, rec = fcr_fcd_recovery(st) if st else (None, None, None)
        dist, nfr = state_dist(st) if st else ({}, 0)
        # GT-ruled FCR on this run's bboxes (honest causal measure, all runs)
        tgt = teacher_states_per_seq(run)
        if tgt:
            g_fcr, _, _ = fcr_fcd_recovery(tgt)
            teacher_fcr_gt[label] = g_fcr
        rows_out.append({
            "label": label, "src": src, "ready": tm is not None,
            "auc": tm["auc"] if tm else None, "pr20": tm["pr20"] if tm else None,
            "auc_fw": tm["auc_fw"] if tm else None,
            "fps": fps_lookup(label),
            "dist": dist, "fcr": fcr, "fcd": fcd, "rec": rec,
        })

    L = []
    L.append("# From-scratch 4-way SGLATrack eval — UAV123 (final test)\n")
    L.append("State source: row 1 = **Teacher(GT-rule)**; rows 2-6 = **CSC runtime**. "
             "FPS = clean sequential probe (4 seqs, alone, threads=4) — full runs ran "
             "3-way parallel so their self-FPS is contended.\n")
    L.append("| Run | AUC | AUC(fw) | Pr@20 | FPS | %CC | %CU | %LA | %FC | FCR | FCD | Rec@30 | state src |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows_out:
        d = r["dist"]
        L.append("| {lab} | {auc} | {aucfw} | {pr} | {fps} | {cc} | {cu} | {la} | {fc} | {fcr} | {fcd} | {rec} | {src} |".format(
            lab=r["label"], auc=fmt(r["auc"]), aucfw=fmt(r["auc_fw"]), pr=fmt(r["pr20"]),
            fps=fmt(r["fps"], 1),
            cc=fmt(d.get("CC"), 1), cu=fmt(d.get("CU"), 1), la=fmt(d.get("LA"), 1), fc=fmt(d.get("FC"), 1),
            fcr=fmt(r["fcr"], 4), fcd=fmt(r["fcd"], 2), rec=fmt(r["rec"], 3), src=r["src"]))
    L.append("")

    # GT-Teacher FCR per run (honest: did control cut TRUE false-confirmed?)
    L.append("## GT-ruled FCR per run (honest causal measure on each run's bboxes)\n")
    L.append("| Run | FCR(GT) |")
    L.append("|---|---|")
    for r in rows_out:
        L.append(f"| {r['label']} | {fmt(teacher_fcr_gt.get(r['label']), 4)} |")
    L.append("")

    # ---- per-sequence easy/hard tercile analysis ----
    bare_ps = per_seq_by_label.get("1. bare (no CSC)", {})
    if bare_ps:
        diff = {s: (v["fail"] / v["n"] if v["n"] else 0.0) for s, v in bare_ps.items()}
        ranked = sorted(diff, key=lambda s: diff[s])
        t = len(ranked) // 3
        bins = {"EASY": ranked[:t], "MEDIUM": ranked[t:2*t], "HARD": ranked[2*t:]}
        L.append(f"## Per-sequence easy/hard analysis ({len(ranked)} seqs)\n")
        L.append("Difficulty = **baseline (bare) GT failure-rate** = total_failure_frames / n_frames "
                 "(the fraction of frames the bare tracker has already lost the target per GT). "
                 "EASY/MEDIUM/HARD = terciles of that rate.\n")
        L.append("Per control run: **mean per-seq ΔAUC vs the bare baseline**, by difficulty bin. "
                 "(+) = control helped, (−) = control hurt.\n")
        ctrl_labels = [r["label"] for r in rows_out if "ctrl" in r["label"]]
        hdr = "| bin | n | bareAUC | " + " | ".join(f"Δ{c.split('.')[0].strip()}" for c in ctrl_labels) + " |"
        L.append("| bin | n | bare AUC | " + " | ".join(c.replace("ctrl ", "").replace(" (def)", "·def").replace(" +hold", "·hold") for c in ctrl_labels) + " |")
        L.append("|---|---|" + "---|" * (len(ctrl_labels) + 1))
        for bn, seqs in bins.items():
            cells = []
            bare_auc = np.mean([bare_ps[s]["auc"] for s in seqs]) if seqs else float("nan")
            for c in ctrl_labels:
                cps = per_seq_by_label.get(c, {})
                deltas = [cps[s]["auc"] - bare_ps[s]["auc"] for s in seqs if s in cps]
                cells.append(f"{np.mean(deltas):+.4f}" if deltas else "—")
            L.append(f"| {bn} | {len(seqs)} | {bare_auc:.3f} | " + " | ".join(cells) + " |")
        L.append("")
        # regression counts per control run
        L.append("### Sequences regressed / improved vs bare (|ΔAUC|>0.01)\n")
        L.append("| run | improved | unchanged | regressed | net ΔAUC |")
        L.append("|---|---|---|---|---|")
        for c in ctrl_labels:
            cps = per_seq_by_label.get(c, {})
            ds = [cps[s]["auc"] - bare_ps[s]["auc"] for s in bare_ps if s in cps]
            if not ds:
                L.append(f"| {c} | — | — | — | — |"); continue
            imp = sum(1 for d in ds if d > 0.01); reg = sum(1 for d in ds if d < -0.01)
            unc = len(ds) - imp - reg
            L.append(f"| {c} | {imp} | {unc} | {reg} | {np.mean(ds):+.4f} |")
        L.append("")

    out = "\n".join(L)
    print(out)
    dst = ROOT / "SUMMARY_4way.md"
    dst.write_text(out)
    print(f"\n[saved] {dst}", file=sys.stderr)

if __name__ == "__main__":
    main()
