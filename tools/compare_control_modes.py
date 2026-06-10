#!/usr/bin/env python3
"""Compare SGLATrack + AVTrack control-mode variants on UAV123.

Aggregates existing SGLATrack runs and AVTrack runs (if available).
Shows: ALL / EASY / MID / HARD AUC breakdown, top-10 hardest + easiest,
FCR/FCD from state files, consolidated side-by-side tables.

Usage:
  python tools/compare_control_modes.py          # SGLATrack only
  python tools/compare_control_modes.py --avtrack # include AVTrack (needs eval10_avtrack)
"""
from __future__ import annotations
import sys, pathlib, json, argparse
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT, ROOT / "tools"):
    sys.path.insert(0, str(_p))

sys.argv = [sys.argv[0]]
import csc_uav_tracking  # noqa
import la_smoke as ls

# ── run directories ─────────────────────────────────────────────────────────
# eval10_sgla = new full 123-seq ablation with uniform flags (run_sgla_ablation_full.sh)
# Fallback to eval7_gated for existing runs (passive=eval5_clamp, combo=eval7_gated)
PASSIVE_SGLA = ROOT / "outputs/eval5_clamp/csc/sglatrack/uav123/test/passive"
EVAL10_SGLA  = ROOT / "outputs/eval10_sgla/csc/sglatrack/uav123/test"
EVAL7        = ROOT / "outputs/eval7_gated/csc/sglatrack/uav123/test"

def _sgla(tag: str) -> pathlib.Path:
    """Prefer eval10_sgla; fall back to known older dir."""
    p10 = EVAL10_SGLA / tag
    if p10.exists():
        return p10
    fallback = {
        "passive": PASSIVE_SGLA,
        "combo":   EVAL7 / "full_combo",
    }
    return fallback.get(tag, p10)  # returns non-existent path if no fallback

RUNS_SGLA: dict[str, pathlib.Path] = {
    "sgla_passive":  PASSIVE_SGLA,        # always use proven eval5_clamp baseline
    "sgla_la_only":  _sgla("la_only"),
    "sgla_fc_only":  _sgla("fc_only"),
    "sgla_combo":    _sgla("combo"),
}

AVTRACK_BASE = ROOT / "outputs/baselines/avtrack/uav123/test"
AVTRACK_CTRL = ROOT / "outputs/eval10_avtrack/csc/avtrack/uav123/test"
RUNS_AV: dict[str, pathlib.Path] = {
    "av_passive":   AVTRACK_BASE,
    "av_la_only":   AVTRACK_CTRL / "la_only",
    "av_fc_only":   AVTRACK_CTRL / "fc_only",
    "av_combo":     AVTRACK_CTRL / "combo",
}


# ── helpers ──────────────────────────────────────────────────────────────────
def seq_auc_gtfail(seq, preds_dir: pathlib.Path):
    ious, n = ls.seq_iou(seq, preds_dir / "predictions")
    if ious is None:
        return None, None
    fin = np.isfinite(ious)
    if not fin.any():
        return None, None
    return float(ious[fin].mean()), float((ious[fin] < 0.2).mean())


def bucket(gtfail: float) -> str:
    return "EASY" if gtfail < 0.10 else ("HARD" if gtfail >= 0.30 else "MID")


def build_run_rows(idx: dict, run_dir: pathlib.Path) -> list[tuple]:
    """Returns [(name, auc, gtfail), ...]."""
    rows = []
    for name, seq in idx.items():
        auc, gtfail = seq_auc_gtfail(seq, run_dir)
        if auc is not None:
            rows.append((name, auc, gtfail))
    return rows


def load_state_fcr(run_dir: pathlib.Path, seq_name: str) -> dict | None:
    states_dir = run_dir / "states"
    if not states_dir.exists():
        return None
    # jsonl format (eval10 runs)
    f_jsonl = states_dir / f"{seq_name}.jsonl"
    if f_jsonl.exists():
        try:
            import json as _json
            rows = [_json.loads(l) for l in f_jsonl.read_text().splitlines() if l.strip()]
            rows = [r for r in rows if not r.get("init")]
            total = len(rows)
            if not total:
                return None
            fc = sum(1 for r in rows if r.get("false_confirmed_flag") or r.get("derived_state") == 3)
            la = sum(1 for r in rows if r.get("derived_state") == 2)
            cc = sum(1 for r in rows if r.get("derived_state") == 0)
            gate = sum(1 for r in rows if r.get("gate_fired"))
            redet = sum(1 for r in rows if r.get("sgla_redetect_called"))
            states_arr = [1 if (r.get("false_confirmed_flag") or r.get("derived_state") == 3) else 0 for r in rows]
            segs: list[int] = []
            i = 0
            while i < len(states_arr):
                if states_arr[i]:
                    j = i
                    while j < len(states_arr) and states_arr[j]:
                        j += 1
                    segs.append(j - i)
                    i = j
                else:
                    i += 1
            fcd = float(np.mean(segs)) if segs else 0.0
            return {"total": total, "fc": fc, "la": la, "cc": cc, "fcd": fcd,
                    "gate": gate, "redet": redet}
        except Exception:
            pass
    # legacy csv format
    f_csv = states_dir / f"{seq_name}.csv"
    if f_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(f_csv)
            if "csc_state" in df.columns:
                total = len(df)
                fc = int((df["csc_state"] == 3).sum())   # state 3 = FC (4-class: CC/CU/LA/FC)
                la = int((df["csc_state"] == 2).sum())
                cc = int((df["csc_state"] == 0).sum())
                s = df["csc_state"].values
                segs = []
                i = 0
                while i < len(s):
                    if s[i] == 3:
                        j = i
                        while j < len(s) and s[j] == 3:
                            j += 1
                        segs.append(j - i)
                        i = j
                    else:
                        i += 1
                fcd = float(np.mean(segs)) if segs else 0.0
                return {"total": total, "fc": fc, "la": la, "cc": cc, "fcd": fcd,
                        "gate": 0, "redet": 0}
        except Exception:
            pass
    return None


def fcr_table(idx: dict, runs: dict[str, pathlib.Path]) -> None:
    """Print FCR / FCD / LA-rate / gate-fires summary."""
    print(f"\n  {'run':<22}  {'frames':>8}  {'FCR%':>6}  {'LA%':>6}  {'FCD':>6}  {'gate':>6}  {'redet':>6}")
    print("  " + "-" * 72)
    for tag, run_dir in runs.items():
        if not run_dir.exists():
            print(f"  {tag:<22}  (missing)")
            continue
        tf = tc = 0; fc_n = la_n = cc_n = gate_n = redet_n = 0; fcd_v = []
        for name, seq in idx.items():
            d = load_state_fcr(run_dir, name)
            if d:
                tf += d["total"]; fc_n += d["fc"]; la_n += d["la"]; cc_n += d["cc"]
                gate_n += d.get("gate", 0); redet_n += d.get("redet", 0)
                if d["fcd"] > 0:
                    fcd_v.append(d["fcd"])
                tc += 1
        if tf:
            fcr = fc_n / tf * 100; lar = la_n / tf * 100
            fcd = float(np.mean(fcd_v)) if fcd_v else 0.0
            print(f"  {tag:<22}  {tf:>8}  {fcr:>6.3f}  {lar:>6.3f}  {fcd:>6.2f}  {gate_n:>6}  {redet_n:>6}")
        else:
            print(f"  {tag:<22}  (no state files)")


def print_run_comparison(
    idx: dict,
    runs: dict[str, pathlib.Path],
    passive_key: str,
    title: str,
) -> dict[str, list]:
    """Print EASY/MID/HARD table vs passive. Returns {tag: rows}."""
    print(f"\n{'#'*70}")
    print(f"  {title}")
    print(f"{'#'*70}")

    all_rows: dict[str, list] = {}
    for tag, run_dir in runs.items():
        if not run_dir.exists():
            print(f"\n  [{tag}]  SKIPPED — {run_dir.name} not found")
            continue
        rows = build_run_rows(idx, run_dir)
        all_rows[tag] = rows

    if passive_key not in all_rows:
        print(f"  ERROR: passive run {passive_key!r} missing.")
        return all_rows

    base_dict = {n: a for n, a, _ in all_rows[passive_key]}

    for tag, rows in all_rows.items():
        if tag == passive_key:
            avg = np.mean([a for _, a, _ in rows])
            print(f"\n  [{tag}]  baseline  mean AUC = {avg:.4f}  ({len(rows)} seqs)")
            continue

        per_bucket: dict[str, list] = {"EASY": [], "MID": [], "HARD": []}
        for name, auc, gtfail in rows:
            if name not in base_dict:
                continue
            d = auc - base_dict[name]
            per_bucket[bucket(gtfail)].append((name, gtfail, base_dict[name], auc, d))

        all_b = [r for bk in per_bucket.values() for r in bk]
        if not all_b:
            print(f"\n  [{tag}]  no matching sequences")
            continue

        print(f"\n  [{tag}]")
        print(f"    {'tercile':<6} {'n':>4} {'passive':>8} {'ctrl':>8} {'ΔAUC':>8} {'wins':>5} {'losses':>7}")
        for bk in ("EASY", "MID", "HARD"):
            bk_r = per_bucket[bk]
            if not bk_r:
                continue
            bp = np.mean([r[2] for r in bk_r])
            cp = np.mean([r[3] for r in bk_r])
            d = np.mean([r[4] for r in bk_r])
            wins = sum(1 for r in bk_r if r[4] > 0.02)
            losses = sum(1 for r in bk_r if r[4] < -0.02)
            print(f"    {bk:<6} {len(bk_r):>4} {bp:>8.3f} {cp:>8.3f} {d:>+8.4f} {wins:>5} {losses:>7}")
        bp_all = np.mean([r[2] for r in all_b])
        cp_all = np.mean([r[3] for r in all_b])
        d_all = np.mean([r[4] for r in all_b])
        wins_all = sum(1 for r in all_b if r[4] > 0.02)
        losses_all = sum(1 for r in all_b if r[4] < -0.02)
        print(f"    {'ALL':<6} {len(all_b):>4} {bp_all:>8.3f} {cp_all:>8.3f} {d_all:>+8.4f} {wins_all:>5} {losses_all:>7}")

    return all_rows


def top10_seq_table(
    idx: dict,
    runs: dict[str, pathlib.Path],
    passive_key: str,
    hardest_first: bool,
    n: int = 10,
    title: str = "",
) -> None:
    """Print per-sequence ΔAUC table sorted by passive difficulty."""
    all_rows: dict[str, list] = {}
    for tag, run_dir in runs.items():
        if run_dir.exists():
            all_rows[tag] = build_run_rows(idx, run_dir)

    if passive_key not in all_rows:
        return

    base_dict = {nm: (a, gf) for nm, a, gf in all_rows[passive_key]}
    run_maps = {tag: {nm: a for nm, a, _ in rows} for tag, rows in all_rows.items()}

    ctrl_keys = [k for k in runs if k != passive_key and k in run_maps]

    seqs_sorted = sorted(base_dict.items(), key=lambda kv: (-kv[1][1] if hardest_first else kv[1][1]))[:n]

    print(f"\n  {title}:")
    header = f"    {'seq':<14} {'pass.AUC':>9}"
    for k in ctrl_keys:
        short = k.split("_", 1)[-1][:8]
        header += f" {short:>9}"
    print(header)
    print("    " + "-" * (14 + 9 + 10 * len(ctrl_keys)))
    for name, (pas_auc, gtf) in seqs_sorted:
        line = f"    {name:<14} {pas_auc:>9.4f}"
        for k in ctrl_keys:
            ctrl_auc = run_maps[k].get(name)
            if ctrl_auc is not None:
                d = ctrl_auc - pas_auc
                sym = "▲" if d > 0.02 else ("▼" if d < -0.02 else " ")
                line += f" {sym}{d:>+7.4f}"
            else:
                line += "      ---"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avtrack", action="store_true", help="Include AVTrack section")
    ap.add_argument("--no_sgla", action="store_true", help="Skip SGLATrack section")
    args = ap.parse_args(sys.argv[1:])

    print("=" * 70)
    print("  CONTROL MODE COMPARISON  —  SGLATrack + AVTrack  (UAV123)")
    print("=" * 70)

    idx = ls.build_index("uav123")
    print(f"  UAV123: {len(idx)} sequences")

    # Difficulty from SGLATrack passive (shared reference for both trackers)
    sgla_pass_rows = build_run_rows(idx, PASSIVE_SGLA)
    diff_map = {n: gf for n, _, gf in sgla_pass_rows}
    hard_seqs_names = [n for n, _ in sorted(diff_map.items(), key=lambda kv: -kv[1])[:10]]
    easy_seqs_names = [n for n, _ in sorted(diff_map.items(), key=lambda kv: kv[1])[:10]]

    print(f"\n  TOP-10 HARDEST (SGLATrack gtfail):")
    for nm in hard_seqs_names:
        a = next((a for n, a, _ in sgla_pass_rows if n == nm), None)
        print(f"    {nm:<14} gtfail={diff_map[nm]:.3f}  SGLA-passive AUC={a:.4f}")
    print(f"\n  TOP-10 EASIEST:")
    for nm in easy_seqs_names:
        a = next((a for n, a, _ in sgla_pass_rows if n == nm), None)
        print(f"    {nm:<14} gtfail={diff_map[nm]:.3f}  SGLA-passive AUC={a:.4f}")

    # ── SGLATrack ────────────────────────────────────────────────────────────
    if not args.no_sgla:
        sgla_rows = print_run_comparison(idx, RUNS_SGLA, "sgla_passive", "SGLATRACK — control modes")

        print(f"\n  [SGLATRACK FCR/FCD from state files]")
        fcr_table(idx, RUNS_SGLA)

        top10_seq_table(idx, RUNS_SGLA, "sgla_passive",
                        hardest_first=True, n=15,
                        title="SGLATrack  TOP-15 HARDEST sequences")
        top10_seq_table(idx, RUNS_SGLA, "sgla_passive",
                        hardest_first=False, n=10,
                        title="SGLATrack  TOP-10 EASIEST sequences (guard)")

    # ── AVTrack ──────────────────────────────────────────────────────────────
    if args.avtrack:
        print(f"\n\n{'#'*70}")
        print("  AVTRACK — checking control run availability")
        missing = [k for k, p in RUNS_AV.items() if not p.exists()]
        if missing:
            print(f"  MISSING runs: {missing}")
            print(f"  Run:  bash tools/run_avtrack_control_uav123.sh")
            # Still show passive if available
            if AVTRACK_BASE.exists():
                av_pass_rows = build_run_rows(idx, AVTRACK_BASE)
                avg_av = np.mean([a for _, a, _ in av_pass_rows])
                avg_sg = np.mean([a for _, a, _ in sgla_pass_rows])
                print(f"\n  AVTrack passive   mean AUC = {avg_av:.4f}  ({len(av_pass_rows)} seqs)")
                print(f"  SGLATrack passive mean AUC = {avg_sg:.4f}  (reference)")
                print(f"\n  AVTrack vs SGLATrack on 15 hardest sequences:")
                av_map = {n: a for n, a, _ in av_pass_rows}
                sg_map = {n: a for n, a, _ in sgla_pass_rows}
                print(f"    {'seq':<14} {'SGLA':>8} {'AVTrack':>8} {'AV-SG':>8}")
                for nm in hard_seqs_names + [n for n, _ in sorted(diff_map.items(), key=lambda kv: -kv[1])[10:15]]:
                    sg = sg_map.get(nm); av = av_map.get(nm)
                    if sg and av:
                        print(f"    {nm:<14} {sg:>8.4f} {av:>8.4f} {av-sg:>+8.4f}")
        else:
            av_rows = print_run_comparison(idx, RUNS_AV, "av_passive", "AVTRACK — control modes")
            print(f"\n  [AVTRACK FCR/FCD]")
            fcr_table(idx, RUNS_AV)
            top10_seq_table(idx, RUNS_AV, "av_passive",
                            hardest_first=True, n=15,
                            title="AVTrack  TOP-15 HARDEST sequences")
            top10_seq_table(idx, RUNS_AV, "av_passive",
                            hardest_first=False, n=10,
                            title="AVTrack  TOP-10 EASIEST sequences (guard)")

    # ── Cross-tracker summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  CROSS-TRACKER AUC SUMMARY  (passive → best-control)")
    print(f"  {'tracker':<16} {'passive':>9} {'la_only':>9} {'fc_only':>9} {'combo':>9}")
    print("  " + "-" * 55)

    # SGLA
    if not args.no_sgla:
        sg_rows_map = {k: {n: a for n, a, _ in build_run_rows(idx, p)} for k, p in RUNS_SGLA.items() if p.exists()}
        sg_avg = lambda k: np.mean(list(sg_rows_map[k].values())) if k in sg_rows_map else float("nan")
        print(f"  {'SGLATrack':<16} {sg_avg('sgla_passive'):>9.4f} {sg_avg('sgla_la_only'):>9.4f} "
              f"{sg_avg('sgla_fc_only'):>9.4f} {sg_avg('sgla_combo'):>9.4f}")

    # AVTRACK
    if args.avtrack:
        av_rows_map = {k: {n: a for n, a, _ in build_run_rows(idx, p)} for k, p in RUNS_AV.items() if p.exists()}
        av_avg = lambda k: np.mean(list(av_rows_map[k].values())) if k in av_rows_map else float("nan")
        print(f"  {'AVTrack':<16} {av_avg('av_passive'):>9.4f} {av_avg('av_la_only'):>9.4f} "
              f"{av_avg('av_fc_only'):>9.4f} {av_avg('av_combo'):>9.4f}")


if __name__ == "__main__":
    main()
