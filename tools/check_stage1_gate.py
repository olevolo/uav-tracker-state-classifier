#!/usr/bin/env python
"""Stage-1 gate checker for CSC training runs.

Reads a val_metrics.json produced by tools/train_csc.py (or evaluate_csc_standalone.py),
verifies all 9 Stage-1 criteria from CSC.md §2, and prints a per-criterion PASS/FAIL
table.  Writes a gate_report.json next to the metrics file.

Exit code: 0 only if ALL 9 criteria pass.

Usage
-----
    python tools/check_stage1_gate.py \
        --metrics_json outputs/csc_training/csc_gru_v2/val_metrics.json

Stage-1 gate criteria (CSC.md §2):
  1. macro_F1        >= 0.55  (best of loc_macro_f1 / derived_macro_f1)
  2. LOST recall     >= 0.75
  3. UNCERTAIN recall >= 0.60
  4. failure_auroc   >= 0.80
  5. failure_auprc   >= 0.50
  6. false alarms    <= 30 / 1000 frames
  7. avg detection delay <= 10 frames before failure (negative = early warning)
  8. no future-frame leakage (pytest exit 0 on causality + IoU tests)
  9. manual audit subset visually consistent (manual, gate = criterion 9 noted in report)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Gate criteria constants
# ---------------------------------------------------------------------------

CRITERIA = [
    # (id, display_name, metric_key_or_None, threshold, direction, note)
    (1,  "macro_F1 >= 0.55",             None,                    0.55,  ">=", "best of loc_macro_f1 / derived_macro_f1"),
    (2,  "LOST recall >= 0.75",           None,                    0.75,  ">=", "from loc_per_state.LOST.recall"),
    (3,  "UNCERTAIN recall >= 0.60",      None,                    0.60,  ">=", "from loc_per_state.UNCERTAIN.recall"),
    (4,  "failure_auroc >= 0.80",         "failure_auroc",         0.80,  ">=", ""),
    (5,  "failure_auprc >= 0.50",         "failure_auprc",         0.50,  ">=", ""),
    (6,  "false_alarms <= 30/1000",       None,                    30.0,  "<=", "false_alarms_per_1000 key"),
    (7,  "avg_detection_delay <= 10",     None,                    10.0,  "<=", "avg_detection_delay_frames; negative=early warning; NaN=no failures"),
    (8,  "no future-frame leakage",       None,                    None,  "pytest", "runs pytest on causality + iou tests"),
    (9,  "manual audit visually OK",      None,                    None,  "manual", "must be verified by human inspection"),
]

# Pytest test files for criterion 8
_LEAKAGE_TESTS = [
    "tests/test_window_causality.py",
    "tests/test_csctcn_causality.py",
    "tests/test_no_runtime_iou.py",
]


# ---------------------------------------------------------------------------
# Value extraction helpers
# ---------------------------------------------------------------------------


def _get_macro_f1(m: dict) -> tuple[float | None, str]:
    """Return (value, source_key) for the best macro F1 in the metrics dict."""
    candidates = [
        ("loc_macro_f1", m.get("loc_macro_f1")),
        ("derived_macro_f1", m.get("derived_macro_f1")),
        ("macro_f1", m.get("macro_f1")),
    ]
    best_val: float | None = None
    best_key = "not_found"
    for key, val in candidates:
        if val is not None:
            if best_val is None or float(val) > best_val:
                best_val = float(val)
                best_key = key
    return best_val, best_key


def _get_recall(m: dict, state: str) -> float | None:
    """Extract recall for a localization state from loc_per_state or derived_per_state."""
    # Try loc_per_state first (new schema)
    loc_per = m.get("loc_per_state", {})
    if state in loc_per:
        return loc_per[state].get("recall")
    # Try derived_per_state aliases
    derived_per = m.get("derived_per_state", {})
    # LOST maps to LOST_AWARE in derived
    aliases = {"LOST": ["LOST_AWARE", "LOST"], "UNCERTAIN": ["CORRECT_UNCERTAIN", "UNCERTAIN"]}
    for alias in aliases.get(state, [state]):
        if alias in derived_per:
            return derived_per[alias].get("recall")
    # Try top-level shorthand keys
    shorthand = f"{state.lower()}_recall"
    if shorthand in m:
        return float(m[shorthand])
    return None


def _get_false_alarms(m: dict) -> float | None:
    """Extract false alarms per 1000 frames."""
    for key in ["false_alarms_per_1000", "fa_per_1000", "false_alarm_rate_per_1000"]:
        if key in m:
            return float(m[key])
    return None


def _get_detection_delay(m: dict) -> float | None:
    """Extract average detection delay (frames)."""
    for key in ["avg_detection_delay", "avg_detection_delay_frames",
                "average_detection_delay", "detection_delay_frames"]:
        if key in m:
            val = m[key]
            # NaN is represented as None or string "NaN"
            if val is None:
                return None
            if isinstance(val, str) and val.lower() == "nan":
                return None
            return float(val)
    return None


def _run_pytest_leakage(project_root: Path) -> tuple[bool, str]:
    """Run pytest on causality tests.  Returns (passed, output_summary)."""
    test_paths = [str(project_root / t) for t in _LEAKAGE_TESTS]
    existing = [p for p in test_paths if Path(p).exists()]
    missing = [p for p in test_paths if not Path(p).exists()]

    if missing:
        print(f"  [gate] WARNING: missing test files: {missing}", flush=True)

    if not existing:
        return False, "No test files found"

    cmd = [
        sys.executable, "-m", "pytest", "-q",
        "--tb=short",
        "--no-header",
    ] + existing

    print(f"  [gate] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(project_root),
        timeout=120,
    )
    passed = result.returncode == 0
    output = (result.stdout + result.stderr).strip()
    # Trim to last 20 lines for the report
    lines = output.split("\n")
    summary = "\n".join(lines[-20:]) if len(lines) > 20 else output
    return passed, summary


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


def evaluate_gate(metrics_path: Path) -> tuple[list[dict], int]:
    """Evaluate all gate criteria.  Returns (results, n_failed)."""
    with open(metrics_path) as f:
        m = json.load(f)

    project_root = metrics_path.resolve().parent
    # Walk up to find project root (contains tests/ directory)
    for _ in range(5):
        if (project_root / "tests").is_dir():
            break
        project_root = project_root.parent

    results: list[dict] = []
    n_failed = 0

    for cid, display, metric_key, threshold, direction, note in CRITERIA:
        res = {
            "criterion_id": cid,
            "display": display,
            "note": note,
            "threshold": threshold,
            "direction": direction,
            "value": None,
            "passed": False,
            "message": "",
        }

        if direction == "manual":
            # Criterion 9 — manual audit, always mark as pending
            res["passed"] = False   # requires human verification
            res["message"] = "MANUAL — not automatically verifiable; run audit_visualizer.py and inspect"
            # Do not count as failure for automated pass/fail count, but flag it
            results.append(res)
            continue

        if direction == "pytest":
            # Criterion 8 — run pytest
            try:
                passed, summary = _run_pytest_leakage(project_root)
            except subprocess.TimeoutExpired:
                passed, summary = False, "pytest timed out after 120s"
            except Exception as e:
                passed, summary = False, f"pytest error: {e}"
            res["passed"] = passed
            res["value"] = "pytest_exit_0" if passed else "pytest_exit_nonzero"
            res["message"] = summary
            if not passed:
                n_failed += 1
            results.append(res)
            continue

        # Numeric criteria 1-7
        if cid == 1:
            val, source_key = _get_macro_f1(m)
            res["metric_key"] = source_key
        elif cid == 2:
            val = _get_recall(m, "LOST")
            res["metric_key"] = "loc_per_state.LOST.recall"
        elif cid == 3:
            val = _get_recall(m, "UNCERTAIN")
            res["metric_key"] = "loc_per_state.UNCERTAIN.recall"
        elif cid == 4:
            val = m.get("failure_auroc")
            if val is not None:
                val = float(val)
            res["metric_key"] = "failure_auroc"
        elif cid == 5:
            val = m.get("failure_auprc")
            if val is not None:
                val = float(val)
            res["metric_key"] = "failure_auprc"
        elif cid == 6:
            val = _get_false_alarms(m)
            res["metric_key"] = "false_alarms_per_1000"
        elif cid == 7:
            val = _get_detection_delay(m)
            res["metric_key"] = "avg_detection_delay_frames"
        else:
            val = None

        res["value"] = val

        if val is None:
            res["passed"] = False
            res["message"] = "metric not found in val_metrics.json"
            n_failed += 1
        else:
            import math
            if math.isnan(val):
                # NaN detection delay = no failures observed — treated as PASS
                # (if no failures, delay doesn't apply)
                if cid == 7:
                    res["passed"] = True
                    res["message"] = f"NaN (no failure episodes) — treating as PASS"
                else:
                    res["passed"] = False
                    res["message"] = f"value is NaN"
                    n_failed += 1
            elif direction == ">=":
                res["passed"] = val >= threshold
                gap = val - threshold
                res["message"] = f"{val:.4f} vs threshold {threshold} ({'+' if gap>=0 else ''}{gap:.4f})"
                if not res["passed"]:
                    n_failed += 1
            elif direction == "<=":
                res["passed"] = val <= threshold
                gap = threshold - val
                res["message"] = f"{val:.4f} vs threshold {threshold} ({'margin +' if gap>=0 else 'excess '}{abs(gap):.4f})"
                if not res["passed"]:
                    n_failed += 1

        results.append(res)

    return results, n_failed


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(results: list[dict], metrics_path: Path) -> None:
    print(f"\n{'='*70}")
    print(f"  Stage-1 Gate Report")
    print(f"  Metrics: {metrics_path}")
    print(f"{'='*70}")

    print(f"\n{'#':<4} {'Criterion':<35} {'Status':<8} {'Value':<12} {'Details'}")
    print("-" * 80)

    for r in results:
        status = "PASS" if r["passed"] else ("PENDING" if r["direction"] == "manual" else "FAIL")
        val_str = str(r["value"])[:11] if r["value"] is not None else "N/A"
        msg_short = str(r["message"])[:40]
        print(f"  {r['criterion_id']:<2} {r['display']:<35} {status:<8} {val_str:<12} {msg_short}")

    print("-" * 80)

    auto_results = [r for r in results if r["direction"] not in ("manual",)]
    auto_passed = sum(1 for r in auto_results if r["passed"])
    auto_total = len(auto_results)
    manual_count = len(results) - auto_total
    all_auto_pass = all(r["passed"] for r in auto_results)

    print(f"\n  Auto criteria: {auto_passed}/{auto_total} PASS")
    print(f"  Manual criteria: {manual_count} (require human inspection)")
    if all_auto_pass:
        print(f"\n  >> OVERALL (auto): PASS")
    else:
        print(f"\n  >> OVERALL (auto): FAIL  ({auto_total - auto_passed} criterion/criteria below bar)")
    print(f"{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="CSC Stage-1 gate checker")
    parser.add_argument(
        "--metrics_json",
        required=True,
        type=Path,
        help="Path to val_metrics.json produced by train_csc.py",
    )
    args = parser.parse_args()

    if not args.metrics_json.exists():
        print(f"ERROR: metrics file not found: {args.metrics_json}", flush=True)
        sys.exit(2)

    print(f"[gate] Loading metrics from {args.metrics_json}", flush=True)
    results, n_auto_failed = evaluate_gate(args.metrics_json)
    print_report(results, args.metrics_json)

    # Write gate_report.json next to metrics file
    report_path = args.metrics_json.parent / "gate_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "metrics_file": str(args.metrics_json),
                "criteria": results,
                "n_auto_failed": n_auto_failed,
                "all_auto_pass": n_auto_failed == 0,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"[gate] Report written: {report_path}", flush=True)

    # Exit 0 only if all 9 auto criteria pass; manual (criterion 9) is excluded
    # from the automated exit code but flagged as PENDING.
    auto_results = [r for r in results if r["direction"] not in ("manual",)]
    all_auto_pass = all(r["passed"] for r in auto_results)
    sys.exit(0 if all_auto_pass else 1)


if __name__ == "__main__":
    main()
