"""Summarize CSC passive-diagnosis run: per-sequence state distribution,
FC/LA episodes, forecast peaks. Reads outputs/csc_runs_diag/<run_tag>/states/*.jsonl.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

STATE_NAMES = ["CC", "CU", "LA", "FC"]


def episodes(flags: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end_exclusive) for contiguous True runs."""
    out: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(flags):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            out.append((start, i))
            in_run = False
    if in_run:
        out.append((start, len(flags)))
    return out


def summarize_seq(jsonl_path: Path) -> dict:
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("init"):
                continue
            rows.append(r)
    if not rows:
        return {"sequence": jsonl_path.stem, "n_frames": 0}

    states = np.array([r["derived_state"] for r in rows], dtype=np.int32)
    p_fc = np.array([r.get("false_confirmed_next_10_prob") if r.get("false_confirmed_next_10_prob") is not None
                     else float("nan") for r in rows], dtype=np.float64)
    p_fail = np.array([r.get("failure_next_10_prob") if r.get("failure_next_10_prob") is not None
                       else float("nan") for r in rows], dtype=np.float64)
    risk = np.array([r.get("risk_score", float("nan")) for r in rows],
                    dtype=np.float64)
    track_ms = np.array([r["tracker_latency_ms"] for r in rows])
    csc_ms = np.array([r["csc_latency_ms"] for r in rows])

    counts = Counter(int(s) for s in states)
    n = len(rows)
    state_pct = {STATE_NAMES[i]: 100.0 * counts.get(i, 0) / n for i in range(4)}

    fc_runs = episodes(states == 3)
    la_runs = episodes(states == 2)

    return {
        "sequence": jsonl_path.stem,
        "n_frames": n,
        "state_pct": state_pct,
        "n_fc_episodes": len(fc_runs),
        "max_fc_dur": max((e - s for s, e in fc_runs), default=0),
        "n_la_episodes": len(la_runs),
        "max_la_dur": max((e - s for s, e in la_runs), default=0),
        "p_fc_mean": float(np.nanmean(p_fc)) if np.isfinite(p_fc).any() else None,
        "p_fc_max":  float(np.nanmax(p_fc)) if np.isfinite(p_fc).any() else None,
        "p_fc_p95":  float(np.nanpercentile(p_fc, 95)) if np.isfinite(p_fc).any() else None,
        "p_fail_max": float(np.nanmax(p_fail)) if np.isfinite(p_fail).any() else None,
        "risk_mean": float(np.nanmean(risk)) if np.isfinite(risk).any() else None,
        "risk_max":  float(np.nanmax(risk)) if np.isfinite(risk).any() else None,
        "tracker_ms_mean": float(track_ms.mean()),
        "csc_ms_mean": float(csc_ms.mean()),
        "fc_episodes": [(int(s), int(e), int(e - s)) for s, e in fc_runs],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="outputs/csc_runs_diag/<run_tag>")
    args = ap.parse_args()

    states_dir = args.run_dir / "states"
    if not states_dir.exists():
        raise SystemExit(f"no states dir at {states_dir}")

    files = sorted(states_dir.glob("*.jsonl"))
    rows = [summarize_seq(p) for p in files]

    print(f"\n{'sequence':<18} {'N':>5}  "
          f"{'CC%':>5} {'CU%':>5} {'LA%':>5} {'FC%':>5}  "
          f"{'#FC':>3} {'maxFC':>5}  {'#LA':>3} {'maxLA':>5}  "
          f"{'pFC_mx':>7} {'pFC_p95':>8} {'pFail_mx':>9}  "
          f"{'risk_mx':>7}  {'trk_ms':>6} {'csc_ms':>6}")
    print("-" * 140)
    for r in rows:
        if r["n_frames"] == 0:
            continue
        sp = r["state_pct"]
        print(f"{r['sequence']:<18} {r['n_frames']:>5}  "
              f"{sp['CC']:>5.1f} {sp['CU']:>5.1f} {sp['LA']:>5.1f} {sp['FC']:>5.1f}  "
              f"{r['n_fc_episodes']:>3} {r['max_fc_dur']:>5}  "
              f"{r['n_la_episodes']:>3} {r['max_la_dur']:>5}  "
              f"{(r['p_fc_max'] or 0):>7.3f} {(r['p_fc_p95'] or 0):>8.3f} "
              f"{(r['p_fail_max'] or 0):>9.3f}  "
              f"{(r['risk_max'] or 0):>7.3f}  "
              f"{r['tracker_ms_mean']:>6.1f} {r['csc_ms_mean']:>6.1f}")

    print("\n--- FC episodes per sequence ---")
    for r in rows:
        if r["n_fc_episodes"] > 0:
            print(f"{r['sequence']:<18}  {r['fc_episodes']}")

    out_md = args.run_dir / "diag_summary.md"
    with out_md.open("w") as f:
        f.write("# CSC Diagnosis — passive R2 on DTB70 (6 sequences)\n\n")
        f.write("| sequence | N | CC% | CU% | LA% | FC% | #FC | maxFC | #LA | maxLA |"
                " pFC_max | pFC_p95 | pFail_max | risk_max | trk_ms | csc_ms |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            if r["n_frames"] == 0:
                continue
            sp = r["state_pct"]
            f.write(f"| {r['sequence']} | {r['n_frames']} | "
                    f"{sp['CC']:.1f} | {sp['CU']:.1f} | {sp['LA']:.1f} | {sp['FC']:.1f} | "
                    f"{r['n_fc_episodes']} | {r['max_fc_dur']} | "
                    f"{r['n_la_episodes']} | {r['max_la_dur']} | "
                    f"{r['p_fc_max'] or 0:.3f} | {r['p_fc_p95'] or 0:.3f} | "
                    f"{r['p_fail_max'] or 0:.3f} | {r['risk_max'] or 0:.3f} | "
                    f"{r['tracker_ms_mean']:.1f} | {r['csc_ms_mean']:.1f} |\n")
        f.write("\n## FC episodes (frame_start, frame_end, duration)\n\n")
        for r in rows:
            if r["n_fc_episodes"] > 0:
                f.write(f"- **{r['sequence']}**: {r['fc_episodes']}\n")
    print(f"\n→ {out_md}")


if __name__ == "__main__":
    main()
