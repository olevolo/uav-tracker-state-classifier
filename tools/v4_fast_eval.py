#!/usr/bin/env python
"""CSC-v4 module A11 (tool 2) — FAST ΔAUC / FCR / FCD harness for v4 run dirs.

A specialized, quick scoring pass for v4 control runs: per-sequence ΔAUC vs a
passive baseline, an EASY / MID / HARD difficulty-tercile breakdown (mirroring
``tools/agg_full.py``), runtime FCR / FCD change (mirroring
``la_smoke.fc_stats``), and a uav6-style **false-LA guard** check (the canonical
"do not regress the easy/false-LA scene" safeguard from la_smoke).

AUC == mean finite per-frame IoU (OPE success area). FCR / FCD read the v4 run's
``states/<seq>.jsonl`` (``derived_state==3`` == FC; FCD = mean FC run length) —
identical semantics to V3 so v4 numbers sit in the same table. Reuses the
la_smoke loaders (``build_index``, ``seq_iou``, ``fc_stats``); read-only.

Tercile cutoffs match agg_full: difficulty = passive GT-fail rate
(``frac IoU<0.2``); EASY ``<0.10``, MID ``[0.10,0.30)``, HARD ``>=0.30``.

CLI
---
  python tools/v4_fast_eval.py --run_dir <v4_run> --baseline <passive> --dataset uav123
  python tools/v4_fast_eval.py --selftest      # synthetic arrays, NO dataset

Read-only; offline single-object-tracking benchmark only.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ---- sys.path: mirror tools/la_smoke.py header EXACTLY (live tracker shadows) ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "salrtd" / "src", PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "tools"):
    sys.path.insert(0, str(_p))

# v4 backbone: the canonical FC state index (==3) lives in the shared enum, so
# our FCR/FCD definition stays in lock-step with the rest of v4.
from csc_lib.csc.v4.v4types import DerivedStateV4  # noqa: E402

FC_STATE = int(DerivedStateV4.FC)        # 3
LA_STATE = int(DerivedStateV4.LA)        # 2

# Difficulty-tercile cutoffs (identical to tools/agg_full.py).
EASY_MAX = 0.10
HARD_MIN = 0.30
# A regression that trips the per-seq win/loss + guard verdict.
DELTA_EPS = 0.02
GUARD_FLOOR = -0.01  # uav6-style guard: ΔAUC must stay >= this (la_smoke convention)
GUARD_SEQS = ("uav6",)  # canonical false-LA guard sequence(s)


@dataclass
class SeqRow:
    """One sequence's fast-eval result."""
    name: str
    gtfail: float           # passive GT-fail rate (frac IoU<0.2) = difficulty
    base_auc: float
    run_auc: float
    base_fcr: float
    run_fcr: float
    base_fcd: float
    run_fcd: float
    base_la: float
    run_la: Optional[float]

    @property
    def d_auc(self) -> float:
        return self.run_auc - self.base_auc

    @property
    def d_fcr_pp(self) -> float:           # percentage-points
        return 100.0 * (self.run_fcr - self.base_fcr)

    @property
    def d_fcd(self) -> float:
        return self.run_fcd - self.base_fcd

    @property
    def bucket(self) -> str:
        return tercile_of(self.gtfail)


# ----------------------------------------------------------------------
# Pure math (tested by --selftest with NO dataset).
# ----------------------------------------------------------------------
def auc_of(ious: np.ndarray) -> float:
    """Mean finite per-frame IoU (OPE success area). All-NaN -> 0.0."""
    ious = np.asarray(ious, dtype=np.float64).ravel()
    fin = np.isfinite(ious)
    return float(ious[fin].mean()) if fin.any() else 0.0


def gtfail_of(ious: np.ndarray, thr: float = 0.2) -> float:
    """Fraction of finite frames with IoU < ``thr`` (per-seq difficulty)."""
    ious = np.asarray(ious, dtype=np.float64).ravel()
    fin = np.isfinite(ious)
    return float((ious[fin] < thr).mean()) if fin.any() else 0.0


def tercile_of(gtfail: float) -> str:
    """EASY/MID/HARD difficulty bucket from a GT-fail rate (agg_full cutoffs)."""
    if gtfail < EASY_MAX:
        return "EASY"
    if gtfail < HARD_MIN:
        return "MID"
    return "HARD"


def fcr_fcd_from_states(states: list[int]) -> tuple[float, float]:
    """(FCR, FCD) from a derived-state list. FCR = #FC/len; FCD = mean FC run.

    Pure-list twin of ``la_smoke.fc_stats`` (which reads a .jsonl file). Kept here
    so the math is unit-testable on synthetic lists without touching disk.
    """
    if not states:
        return 0.0, 0.0
    fc = sum(1 for s in states if s == FC_STATE)
    runs: list[int] = []
    r = 0
    for s in states:
        if s == FC_STATE:
            r += 1
        elif r:
            runs.append(r)
            r = 0
    if r:
        runs.append(r)
    return fc / len(states), (float(np.mean(runs)) if runs else 0.0)


def tercile_summary(rows: list[SeqRow]) -> dict[str, dict]:
    """Aggregate rows into EASY/MID/HARD + ALL stats (mirrors agg_full table)."""
    out: dict[str, dict] = {}
    for tname in ("EASY", "MID", "HARD", "ALL"):
        bk = rows if tname == "ALL" else [r for r in rows if r.bucket == tname]
        if not bk:
            continue
        d = [r.d_auc for r in bk]
        out[tname] = {
            "n": len(bk),
            "base_auc": float(np.mean([r.base_auc for r in bk])),
            "run_auc": float(np.mean([r.run_auc for r in bk])),
            "d_auc": float(np.mean(d)),
            "wins": sum(1 for x in d if x > DELTA_EPS),
            "losses": sum(1 for x in d if x < -DELTA_EPS),
            "d_fcr_pp": float(np.mean([r.d_fcr_pp for r in bk])),
            "d_fcd": float(np.mean([r.d_fcd for r in bk])),
        }
    return out


def guard_verdict(rows: list[SeqRow], guard_names=GUARD_SEQS) -> list[tuple[str, float, str]]:
    """For each guard seq present, return (name, ΔAUC, 'OK'|'REGRESSION')."""
    by = {r.name: r for r in rows}
    res = []
    for g in guard_names:
        if g in by:
            d = by[g].d_auc
            res.append((g, d, "OK" if d >= GUARD_FLOOR else "REGRESSION"))
    return res


# ----------------------------------------------------------------------
# Dataset-backed scoring (reuses la_smoke loaders).
# ----------------------------------------------------------------------
def score_run(run_dir: Path, baseline_dir: Path, dataset: str, split: str) -> list[SeqRow]:
    """Build SeqRows for every seq present in BOTH run and baseline predictions."""
    import la_smoke as ls

    idx = ls.build_index(dataset, split=split)
    rows: list[SeqRow] = []
    for name, seq in idx.items():
        b_iou, bn = ls.seq_iou(seq, baseline_dir / "predictions")
        r_iou, rn = ls.seq_iou(seq, run_dir / "predictions")
        if b_iou is None or r_iou is None:
            continue
        n = min(bn, rn)
        b_fcr, b_fcd = ls.fc_stats(baseline_dir / "states" / f"{name}.jsonl", n)
        r_fcr, r_fcd = ls.fc_stats(run_dir / "states" / f"{name}.jsonl", n)
        rows.append(SeqRow(
            name=name,
            gtfail=gtfail_of(b_iou),
            base_auc=auc_of(b_iou),
            run_auc=auc_of(r_iou),
            base_fcr=b_fcr or 0.0,
            run_fcr=r_fcr or 0.0,
            base_fcd=b_fcd or 0.0,
            run_fcd=r_fcd or 0.0,
            base_la=ls.la_fraction(baseline_dir / "states" / f"{name}.jsonl", bn) or 0.0,
            run_la=ls.la_fraction(run_dir / "states" / f"{name}.jsonl", rn),
        ))
    return rows


# ----------------------------------------------------------------------
# Reporting (mirrors agg_full + la_smoke output style).
# ----------------------------------------------------------------------
def print_report(rows: list[SeqRow], run_dir: Path, baseline_dir: Path, tag: str) -> None:
    if not rows:
        print("no rows (no overlapping predictions between run and baseline)")
        return

    summ = tercile_summary(rows)
    print(f"\n================ v4 fast-eval: {tag}  ({len(rows)} seqs) "
          f"vs {baseline_dir.name} ================")
    print(f"{'tercile':<6}{'n':>4}{'base AUC':>10}{'run AUC':>9}{'ΔAUC':>9}"
          f"{'wins':>6}{'loss':>5}{'ΔFCR pp':>9}{'ΔFCD':>8}")
    for tname in ("EASY", "MID", "HARD", "ALL"):
        s = summ.get(tname)
        if not s:
            continue
        print(f"{tname:<6}{s['n']:>4}{s['base_auc']:>10.3f}{s['run_auc']:>9.3f}"
              f"{s['d_auc']:>+9.4f}{s['wins']:>6}{s['losses']:>5}"
              f"{s['d_fcr_pp']:>+9.2f}{s['d_fcd']:>+8.2f}")

    # uav6-style false-LA guard
    print()
    for g, d, verdict in guard_verdict(rows):
        print(f"{g} false-LA GUARD ΔAUC: {d:+.4f}  [{verdict}]  (must stay >= {GUARD_FLOOR})")
    if not guard_verdict(rows):
        print(f"(guard seq {GUARD_SEQS} not in this run — guard not evaluated)")

    # EASY-scene regressions (agg_full's 'do nothing on easy' check)
    easy_reg = sorted((r for r in rows if r.gtfail < EASY_MAX and r.d_auc < -DELTA_EPS),
                      key=lambda r: r.d_auc)
    print(f"\nEASY-scene regressions (gtfail<{EASY_MAX}, ΔAUC<-{DELTA_EPS}): {len(easy_reg)}")
    for r in easy_reg[:12]:
        print(f"  {r.name:<14} gtfail={r.gtfail:.3f}  {r.base_auc:.3f} -> {r.run_auc:.3f}  Δ{r.d_auc:+.4f}")

    print("\nTOP WINNERS (ΔAUC):")
    for r in sorted(rows, key=lambda r: -r.d_auc)[:10]:
        print(f"  {r.name:<14} gtfail={r.gtfail:.3f}  {r.base_auc:.3f} -> {r.run_auc:.3f}  "
              f"Δ{r.d_auc:+.4f}  ΔFCR{r.d_fcr_pp:+.2f}pp")
    print("TOP LOSERS (ΔAUC):")
    for r in sorted(rows, key=lambda r: r.d_auc)[:10]:
        print(f"  {r.name:<14} gtfail={r.gtfail:.3f}  {r.base_auc:.3f} -> {r.run_auc:.3f}  "
              f"Δ{r.d_auc:+.4f}  ΔFCR{r.d_fcr_pp:+.2f}pp")

    # FCR/FCD aggregate (la_smoke.fc_stats semantics)
    a = summ["ALL"]
    bfcr = float(np.mean([r.base_fcr for r in rows]))
    rfcr = float(np.mean([r.run_fcr for r in rows]))
    bfcd = float(np.mean([r.base_fcd for r in rows]))
    rfcd = float(np.mean([r.run_fcd for r in rows]))
    print(f"\nmean FCR: base {100*bfcr:.2f}%  ->  run {100*rfcr:.2f}%  (Δ {100*(rfcr-bfcr):+.2f}pp)")
    print(f"mean FCD: base {bfcd:.2f}  ->  run {rfcd:.2f}  (Δ {rfcd-bfcd:+.2f})")
    print(f"overall ΔAUC (ALL {a['n']} seqs): {a['d_auc']:+.4f}  "
          f"(wins {a['wins']} / losses {a['losses']})")


# ----------------------------------------------------------------------
# Self-test: synthetic arrays, NO dataset. Asserts AUC/FCR/FCD + tercile math.
# ----------------------------------------------------------------------
def selftest() -> None:
    # --- auc_of ---
    assert abs(auc_of(np.array([0.0, 1.0])) - 0.5) < 1e-12
    assert abs(auc_of(np.array([0.4, np.nan, 0.6])) - 0.5) < 1e-12  # NaN dropped
    assert auc_of(np.array([np.nan, np.nan])) == 0.0

    # --- gtfail_of ---
    iou = np.array([0.0, 0.1, 0.3, 0.9])  # 2 of 4 below 0.2
    assert abs(gtfail_of(iou) - 0.5) < 1e-12
    assert gtfail_of(np.array([np.nan])) == 0.0

    # --- tercile_of (agg_full cutoffs) ---
    assert tercile_of(0.0) == "EASY" and tercile_of(0.099) == "EASY"
    assert tercile_of(0.10) == "MID" and tercile_of(0.29) == "MID"
    assert tercile_of(0.30) == "HARD" and tercile_of(0.9) == "HARD"

    # --- fcr_fcd_from_states (FC==3) ---
    # states: CC CC FC FC FC CC FC  -> 4 FC of 7; runs [3,1] -> FCD = 2.0
    st = [0, 0, 3, 3, 3, 0, 3]
    fcr, fcd = fcr_fcd_from_states(st)
    assert abs(fcr - 4 / 7) < 1e-12, fcr
    assert abs(fcd - 2.0) < 1e-12, fcd
    # trailing run counted
    assert fcr_fcd_from_states([3, 3]) == (1.0, 2.0)
    # no FC
    assert fcr_fcd_from_states([0, 1, 2]) == (0.0, 0.0)
    # empty
    assert fcr_fcd_from_states([]) == (0.0, 0.0)
    # cross-check vs la_smoke.fc_stats by writing a tiny states file
    import json
    import tempfile

    import la_smoke as ls  # safe: import only (no dataset load)
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "toy.jsonl"
        with open(sp, "w") as fh:
            for t, s in enumerate(st):
                fh.write(json.dumps({"frame_idx": t, "derived_state": s}) + "\n")
        ls_fcr, ls_fcd = ls.fc_stats(sp, len(st))
        assert abs(ls_fcr - fcr) < 1e-12 and abs(ls_fcd - fcd) < 1e-12, (ls_fcr, ls_fcd)

    # --- SeqRow deltas + bucket ---
    row = SeqRow(name="x", gtfail=0.5, base_auc=0.30, run_auc=0.42,
                 base_fcr=0.05, run_fcr=0.02, base_fcd=4.0, run_fcd=2.0,
                 base_la=0.1, run_la=0.2)
    assert abs(row.d_auc - 0.12) < 1e-12
    assert abs(row.d_fcr_pp - (-3.0)) < 1e-12   # (0.02-0.05)*100
    assert abs(row.d_fcd - (-2.0)) < 1e-12
    assert row.bucket == "HARD"

    # --- tercile_summary on a synthetic 3-seq set (one per bucket) ---
    rows = [
        SeqRow("easy_win", 0.02, 0.80, 0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # +0.05 EASY win
        SeqRow("mid_loss", 0.20, 0.50, 0.45, 0.02, 0.04, 1.0, 2.0, 0.1, 0.1),  # -0.05 MID loss
        SeqRow("hard_win", 0.60, 0.20, 0.40, 0.10, 0.03, 5.0, 1.0, 0.3, 0.2),  # +0.20 HARD win
    ]
    summ = tercile_summary(rows)
    assert summ["EASY"]["n"] == 1 and summ["EASY"]["wins"] == 1 and summ["EASY"]["losses"] == 0
    assert summ["MID"]["losses"] == 1 and summ["MID"]["wins"] == 0
    assert summ["HARD"]["wins"] == 1
    assert summ["ALL"]["n"] == 3
    assert abs(summ["ALL"]["d_auc"] - np.mean([0.05, -0.05, 0.20])) < 1e-9
    # ΔFCR pp aggregate sign: hard cut FCR (-7pp) dominates
    assert summ["HARD"]["d_fcr_pp"] < 0

    # --- guard_verdict ---
    guard_rows = [
        SeqRow("uav6", 0.02, 0.70, 0.705, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # +0.005 OK
        SeqRow("other", 0.5, 0.2, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    gv = guard_verdict(guard_rows)
    assert gv and gv[0][0] == "uav6" and gv[0][2] == "OK", gv
    # a regressing guard trips REGRESSION
    bad = [SeqRow("uav6", 0.02, 0.70, 0.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]  # -0.10
    assert guard_verdict(bad)[0][2] == "REGRESSION"
    # absent guard -> empty
    assert guard_verdict([SeqRow("z", 0.5, 0.2, 0.2, 0, 0, 0, 0, 0, 0)]) == []

    # --- end-to-end report renders on synthetic rows (no dataset) ---
    print_report(rows + guard_rows, Path("/tmp/run"), Path("/tmp/passive"), tag="selftest")

    print("\nOK v4_fast_eval selftest: AUC/gtfail/tercile/FCR/FCD/guard math verified "
          "(incl. la_smoke.fc_stats cross-check)")


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="CSC-v4 FAST ΔAUC/FCR/FCD harness for v4 run dirs.")
    ap.add_argument("--run_dir", default=None, help="v4 control run dir (predictions/ + states/).")
    ap.add_argument("--baseline", default=None,
                    help="passive baseline run dir for ΔAUC. Default: la_smoke.PASSIVE.")
    ap.add_argument("--dataset", default="uav123")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="v4_run")
    ap.add_argument("--selftest", action="store_true",
                    help="run synthetic-array math self-test (NO dataset) and exit.")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    assert args.run_dir, "need --run_dir (or --selftest)"
    import la_smoke as ls  # resolves PASSIVE default
    run = Path(args.run_dir)
    baseline = Path(args.baseline) if args.baseline else Path(ls.PASSIVE)
    print(f"loading {args.dataset}/{args.split} + scoring {args.tag} ...", file=sys.stderr)
    rows = score_run(run, baseline, args.dataset, args.split)
    print_report(rows, run, baseline, tag=args.tag)


if __name__ == "__main__":
    main()
