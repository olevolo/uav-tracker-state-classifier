"""action_audit.py — Compare baseline vs advisory benchmark logs to compute
intervention effectiveness per sequence.

Usage:
    python -m salt_r.action_audit \\
        --baseline-json saltr/results/hard_bench_baseline.json \\
        --advisory-json saltr/results/hard_bench_v2_no_flow.json \\
        --output saltr/results/action_audit_uav123.json

Or called after hard_bench.py has run both baseline and advisory and saved
JSON results. If both JSONs are the same file (combined output from
hard_bench.py), the script will use the embedded comparison.

The audit table format:
    seq | auc_base | auc_adv | delta | interventions | changed_frames | best_action_type

Gate evaluation (printed):
    hard_subset AUC delta >= +0.03  → PASS/FAIL
    full UAV123 regression <= -0.005 → PASS/FAIL (if available)
    changed_frames > 0.5% hard frames → PASS/FAIL
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Gate thresholds (must match hard_bench.py)
# ---------------------------------------------------------------------------

_GATE_AUC_DELTA = 0.03
_GATE_FULL_REGRESSION = -0.005
_GATE_CHANGED_PCT = 0.005


def _load_bench_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _merge_baseline_advisory(base: dict, adv: dict) -> tuple[list, dict]:
    """Merge two separate bench JSONs (baseline + advisory) into audit rows."""
    base_seqs = {r["seq"]: r for r in base.get("sequences", [])}
    adv_seqs = {r["seq"]: r for r in adv.get("sequences", [])}
    all_seqs = sorted(set(list(base_seqs) + list(adv_seqs)))

    rows = []
    for sname in all_seqs:
        br = base_seqs.get(sname, {})
        ar = adv_seqs.get(sname, {})
        auc_base = br.get("auc_base", float("nan"))
        auc_adv = ar.get("auc_base", float("nan"))  # advisory run's auc_base IS its AUC
        # If the advisory JSON has auc_adv (from a combined run), use that instead
        if ar.get("auc_adv") is not None:
            auc_adv = ar["auc_adv"]
        delta = auc_adv - auc_base if not (np.isnan(auc_base) or np.isnan(auc_adv)) else float("nan")
        rows.append({
            "seq": sname,
            "auc_base": auc_base,
            "auc_adv": auc_adv,
            "delta": delta,
            "n_frames": br.get("n_frames") or ar.get("n_frames"),
            "interventions": ar.get("interventions", 0),
            "changed_frames": ar.get("changed_frames", 0),
            "mean_iou_delta_changed": ar.get("mean_iou_delta_changed", 0.0),
            "best_action_type": ar.get("best_action_type", "n/a"),
        })
    return rows, {}


def _audit_from_combined(data: dict) -> tuple[list, dict]:
    """Extract audit rows from a combined hard_bench.py JSON output."""
    rows = []
    for r in data.get("sequences", []):
        auc_base = r.get("auc_base", float("nan"))
        auc_adv = r.get("auc_adv")
        delta = r.get("delta")
        if auc_adv is None:
            auc_adv = float("nan")
        if delta is None:
            delta = float("nan")
        rows.append({
            "seq": r.get("seq", "?"),
            "auc_base": auc_base,
            "auc_adv": auc_adv,
            "delta": delta,
            "n_frames": r.get("n_frames"),
            "interventions": r.get("interventions", 0),
            "changed_frames": r.get("changed_frames", 0),
            "mean_iou_delta_changed": r.get("mean_iou_delta_changed", 0.0),
            "best_action_type": r.get("best_action_type", "n/a"),
        })
    summary = data.get("summary", {})
    return rows, summary


def _print_table(rows: list, summary: dict) -> None:
    """Print a formatted audit table."""
    print(f"\n{'='*80}")
    print("  Action Audit Table")
    print(f"{'─'*80}")
    print(f"  {'Sequence':<16} {'AUC_base':>8} {'AUC_adv':>8} {'Delta':>7} "
          f"{'Interv':>7} {'Changed':>8} {'IoUΔ(ch)':>9} {'Best Action':<14}")
    print(f"{'─'*80}")

    for r in rows:
        auc_base = r["auc_base"]
        auc_adv = r["auc_adv"]
        delta = r["delta"]

        def _fmt(v, fmt=".3f"):
            return format(v, fmt) if not np.isnan(v) else "  n/a"

        def _fmtd(v):
            return format(v, "+.3f") if not np.isnan(v) else "   n/a"

        print(
            f"  {r['seq']:<16} {_fmt(auc_base):>8} {_fmt(auc_adv):>8} "
            f"{_fmtd(delta):>7} {r['interventions']:>7} {r['changed_frames']:>8} "
            f"{r['mean_iou_delta_changed']:>+9.3f} {r['best_action_type']:<14}"
        )

    print(f"{'─'*80}")

    # Print summary if available
    hb = summary.get("hard_mean_auc_base")
    ha = summary.get("hard_mean_auc_adv")
    hd = summary.get("hard_delta")
    ti = summary.get("total_interventions", 0)
    tc = summary.get("total_changed_frames", 0)
    changed_pct = summary.get("changed_frames_pct", 0.0)

    if hb is not None:
        hb_s = f"{hb:.3f}"
        ha_s = f"{ha:.3f}" if ha is not None else "  n/a"
        hd_s = f"{hd:+.3f}" if hd is not None else "   n/a"
        print(f"  {'HARD MEAN':<16} {hb_s:>8} {ha_s:>8} {hd_s:>7} {ti:>7} {tc:>8}")
        print(f"  Changed frames: {tc} ({changed_pct:.2f}% of hard subset)")
    else:
        # Compute from rows
        base_aucs = [r["auc_base"] for r in rows if not np.isnan(r["auc_base"])]
        adv_aucs = [r["auc_adv"] for r in rows if not np.isnan(r["auc_adv"])]
        hb_s = f"{np.mean(base_aucs):.3f}" if base_aucs else "n/a"
        ha_s = f"{np.mean(adv_aucs):.3f}" if adv_aucs else "n/a"
        total_frames = sum(r.get("n_frames") or 0 for r in rows)
        tc = sum(r["changed_frames"] for r in rows)
        ti = sum(r["interventions"] for r in rows)
        changed_pct = tc / max(total_frames, 1) * 100
        print(f"  {'HARD MEAN':<16} {hb_s:>8} {ha_s:>8} {'':>7} {ti:>7} {tc:>8}")
        print(f"  Changed frames: {tc} ({changed_pct:.2f}% of hard subset)")

    print(f"{'='*80}")


def _print_gates(rows: list, summary: dict) -> dict:
    """Print Phase 1+2 gates and return gate results."""
    # Compute values from rows or summary
    hd = summary.get("hard_delta")
    full_d = summary.get("full_delta")
    changed_pct = summary.get("changed_frames_pct")

    if hd is None:
        deltas = [r["delta"] for r in rows if not np.isnan(r["delta"])]
        base_aucs = [r["auc_base"] for r in rows if not np.isnan(r["auc_base"])]
        adv_aucs = [r["auc_adv"] for r in rows if not np.isnan(r["auc_adv"])]
        if base_aucs and adv_aucs:
            hd = float(np.mean(adv_aucs)) - float(np.mean(base_aucs))

    if changed_pct is None:
        tc = sum(r["changed_frames"] for r in rows)
        total_frames = sum(r.get("n_frames") or 0 for r in rows)
        changed_pct = tc / max(total_frames, 1) * 100

    print(f"\n{'='*70}")
    print("Phase 1 Hard Benchmark Gates:")

    gates = {}

    if hd is not None:
        gate1 = hd >= _GATE_AUC_DELTA
        gates["hard_auc_delta_pass"] = gate1
        print(f"  hard_subset AUC delta >= +{_GATE_AUC_DELTA:.2f}:   "
              f"{'PASS' if gate1 else 'FAIL'} (delta={hd:+.3f})")
    else:
        gates["hard_auc_delta_pass"] = None
        print(f"  hard_subset AUC delta >= +{_GATE_AUC_DELTA:.2f}:   SKIP (no advisory data)")

    if full_d is not None:
        gate2 = full_d >= _GATE_FULL_REGRESSION
        gates["full_regression_pass"] = gate2
        print(f"  full UAV123 regression <= {_GATE_FULL_REGRESSION:.3f}:  "
              f"{'PASS' if gate2 else 'FAIL'} (delta={full_d:+.3f})")
    else:
        gates["full_regression_pass"] = None
        print(f"  full UAV123 regression <= {_GATE_FULL_REGRESSION:.3f}:  SKIP (full run not available)")

    gate3 = changed_pct > _GATE_CHANGED_PCT * 100
    gates["changed_frames_pass"] = gate3
    tc = summary.get("total_changed_frames", sum(r["changed_frames"] for r in rows))
    print(f"  changed_frames > {_GATE_CHANGED_PCT*100:.1f}% hard frames:  "
          f"{'PASS' if gate3 else 'FAIL'} ({changed_pct:.2f}% = {tc} frames)")

    print(f"\nPhase 2 Action Audit:")
    tc_all = sum(r["changed_frames"] for r in rows)
    print(f"  Advisory changes anything at all:  {'YES' if tc_all > 0 else 'NO'}")

    improved = sorted(
        [(r["seq"], r["delta"]) for r in rows if (r["delta"] or 0) > 0.005],
        key=lambda x: -x[1]
    )
    regressed = sorted(
        [(r["seq"], r["delta"]) for r in rows if (r["delta"] or 0) < -0.005],
        key=lambda x: x[1]
    )
    print(f"  Hard sequences most improved ({len(improved)}):")
    for sname, d in improved[:5]:
        print(f"    {sname:<16} delta={d:+.3f}")
    if not improved:
        print("    (none > 0.005)")
    print(f"  Hard sequences regressed ({len(regressed)}):")
    for sname, d in regressed[:5]:
        print(f"    {sname:<16} delta={d:+.3f}")
    if not regressed:
        print("    (none < -0.005)")

    print(f"{'='*70}\n")
    return gates


def main():
    ap = argparse.ArgumentParser(
        description="Audit action effectiveness from hard_bench.py outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--baseline-json", default=None, metavar="PATH",
        help="Baseline hard_bench JSON output (no advisory).",
    )
    ap.add_argument(
        "--advisory-json", default=None, metavar="PATH",
        help="Advisory hard_bench JSON output.",
    )
    ap.add_argument(
        "--combined-json", default=None, metavar="PATH",
        help="Combined hard_bench JSON (single file with both base and adv).",
    )
    ap.add_argument(
        "--output", default="saltr/results/action_audit_uav123.json", metavar="PATH",
        help="Output audit JSON path.",
    )
    args = ap.parse_args()

    # Try to auto-discover inputs if none given
    if args.combined_json is None and args.advisory_json is None and args.baseline_json is None:
        # Look for the default hard_bench output
        _root = Path(__file__).parent.parent.parent.parent  # project root from saltr/src/salt_r/
        candidates = [
            _root / "saltr" / "results" / "hard_bench_v2_no_flow.json",
            _root / "saltr" / "results" / "hard_bench_v2_corrected.json",
            _root / "saltr" / "results" / "hard_bench_baseline.json",
        ]
        for c in candidates:
            if c.exists():
                print(f"  Auto-discovered: {c}")
                args.combined_json = str(c)
                break
        if args.combined_json is None:
            print("ERROR: No input files provided and no auto-discovered files found.")
            print("  Run scripts/hard_bench.py first, or provide --combined-json / --advisory-json")
            sys.exit(1)

    # Load and merge
    if args.combined_json:
        data = _load_bench_json(args.combined_json)
        rows, summary = _audit_from_combined(data)
    else:
        base_data = _load_bench_json(args.baseline_json) if args.baseline_json else {"sequences": []}
        adv_data = _load_bench_json(args.advisory_json) if args.advisory_json else {"sequences": []}
        rows, summary = _merge_baseline_advisory(base_data, adv_data)

    # Print table
    _print_table(rows, summary)

    # Print gates
    gates = _print_gates(rows, summary)

    # Build output JSON
    # Clean up rows: remove non-serializable floats (nan)
    clean_rows = []
    for r in rows:
        cr = dict(r)
        for k in ["auc_base", "auc_adv", "delta"]:
            if k in cr and isinstance(cr[k], float) and np.isnan(cr[k]):
                cr[k] = None
        clean_rows.append(cr)

    # Recompute summary if missing
    if not summary:
        base_aucs = [r["auc_base"] for r in rows if r.get("auc_base") is not None and not np.isnan(r["auc_base"])]
        adv_aucs = [r["auc_adv"] for r in rows if r.get("auc_adv") is not None and not np.isnan(r["auc_adv"])]
        total_frames = sum(r.get("n_frames") or 0 for r in rows)
        tc = sum(r["changed_frames"] for r in rows)
        ti = sum(r["interventions"] for r in rows)
        summary = {
            "hard_mean_auc_base": round(float(np.mean(base_aucs)), 4) if base_aucs else None,
            "hard_mean_auc_adv": round(float(np.mean(adv_aucs)), 4) if adv_aucs else None,
            "hard_delta": round(float(np.mean(adv_aucs)) - float(np.mean(base_aucs)), 4) if (base_aucs and adv_aucs) else None,
            "full_auc_base": None,
            "full_auc_adv": None,
            "full_delta": None,
            "total_interventions": ti,
            "total_changed_frames": tc,
            "sequences_improved": sum(1 for r in rows if (r.get("delta") or 0) > 0),
            "sequences_regressed": sum(1 for r in rows if (r.get("delta") or 0) < 0),
            "changed_frames_pct": round(tc / max(total_frames, 1) * 100, 2),
        }

    output_data = {
        "sequences": clean_rows,
        "summary": summary,
        "gates": gates,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  Audit saved to: {output_path}")


if __name__ == "__main__":
    main()
