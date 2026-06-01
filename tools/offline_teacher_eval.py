#!/usr/bin/env python3
"""Mode 1 — OFFLINE GT-referenced evaluation of trackers (the "CSC-Teacher").

Applies the teacher rules (weak_labeler) view: per tracker, the GT-derived state
distribution (CC/CU/LA/FC) + GT-rule FCR/FCD, alongside standard Success AUC /
Precision@20. This evaluates the TRACKERS themselves (how often each is lost /
false-confirmed per ground truth) — it is NOT the online model's prediction.

Reads (from the completed LIVE matrix, run_tag <t>_r3_passive):
  labels_v3/<ds>/test/labels(.jsonl | _per_sequence/*.jsonl)  -> derived_state (GT rule)
  tracking_metrics/metrics_summary.json                       -> AUC, Precision@20
Writes a markdown table to stdout and outputs/eval/_offline_teacher_eval_<ds>.md
"""
from __future__ import annotations
import csv, glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "outputs/eval"
TRACKERS = ["sglatrack", "avtrack", "ortrack", "ostrack"]
ST = {0: "CC", 1: "CU", 2: "LA", 3: "FC"}


def load_states(run: Path, ds: str) -> list[list[int]]:
    """Return list of per-sequence derived_state sequences (GT-rule labels)."""
    base = run / "labels_v3" / ds / "test"
    seqs = []
    psd = base / "labels_per_sequence"
    if psd.is_dir():
        for fp in sorted(psd.glob("*.jsonl")):
            s = []
            for line in fp.open():
                try:
                    s.append(int(json.loads(line).get("derived_state", -1)))
                except Exception:
                    pass
            seqs.append(s)
    else:
        flat = base / "labels.jsonl"
        if flat.is_file():
            cur, key = [], None
            for line in flat.open():
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("sequence") != key:
                    if cur:
                        seqs.append(cur)
                    cur, key = [], d.get("sequence")
                cur.append(int(d.get("derived_state", -1)))
            if cur:
                seqs.append(cur)
    return seqs


def fc_episodes(seq: list[int]) -> list[int]:
    runs, cur = [], 0
    for s in seq:
        if s == 3:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    return runs


def eval_tracker(t: str, ds: str) -> dict | None:
    run = EVAL / t / ds / "test" / f"{t}_r3_passive"
    seqs = load_states(run, ds)
    if not seqs:
        return None
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total = 0
    episodes = []
    for s in seqs:
        for v in s:
            if v in counts:
                counts[v] += 1; total += 1
        episodes += fc_episodes(s)
    if total == 0:
        return None
    d = {"tracker": t, "n_seq": len(seqs), "n_frames": total}
    for k, name in ST.items():
        d[f"pct_{name}"] = 100.0 * counts[k] / total
    d["fcr_gt"] = counts[3] / total
    d["fcd_gt"] = (sum(episodes) / len(episodes)) if episodes else 0.0
    d["n_fc_episodes"] = len(episodes)
    # standard tracking metrics
    ms = run / "tracking_metrics" / "metrics_summary.json"
    if ms.is_file():
        m = json.loads(ms.read_text()).get("macro", {})
        d["auc"] = m.get("auc"); d["pr20"] = m.get("precision_20")
    return d


def f(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def main() -> int:
    datasets = sys.argv[1:] or ["uav123", "uav123_10fps"]
    out = ["# Mode 1 — Offline GT-referenced tracker evaluation (CSC-Teacher)", "",
           "Teacher = weak-labeler rules applied to ground truth. Columns are the tracker's "
           "**intrinsic** per-GT-rule behaviour (state distribution + FCR/FCD), not the online "
           "model's prediction. Success AUC / Pr@20 are standard tracking metrics for reference.", ""]
    hdr = ("| Tracker | AUC | Pr@20 | %CC | %CU | %LA | %FC | FCR(GT) | FCD(GT) | #FC-ep |\n"
           "|---|---|---|---|---|---|---|---|---|---|")
    for ds in datasets:
        out += [f"## {ds}", "", hdr]
        for t in TRACKERS:
            d = eval_tracker(t, ds)
            if not d:
                out.append(f"| {t} | — | — | — | — | — | — | — | — | — |"); continue
            out.append("| " + " | ".join([
                t, f(d.get("auc")), f(d.get("pr20")),
                f(d["pct_CC"], 1), f(d["pct_CU"], 1), f(d["pct_LA"], 1), f(d["pct_FC"], 2),
                f(d["fcr_gt"]), f(d["fcd_gt"], 2), str(d["n_fc_episodes"]),
            ]) + " |")
            print(f"  {ds:13s} {t:10s} AUC={f(d.get('auc'))} %FC={d['pct_FC']:.2f} "
                  f"FCR(GT)={d['fcr_gt']:.4f} FCD={d['fcd_gt']:.2f} ep={d['n_fc_episodes']}")
        out.append("")
        (EVAL / f"_offline_teacher_eval_{ds}.md").write_text("\n".join(out) + "\n")
    print("wrote per-dataset md under outputs/eval/_offline_teacher_eval_*.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
