#!/usr/bin/env python
"""Label distribution auditor for CSC labels.

Reads every *.jsonl under --labels_dir, computes per-state counts and
percentages (both globally and per-sequence), writes a CSV + sibling
summary.json.  Exits non-zero if any state has < 0.5 % support OR if all
sequences collapse to a single dominant state (useful failure signal per
CSC.md §2).

Usage
-----
    python tools/audit_label_distribution.py \
        --labels_dir outputs/csc_labels/got10k \
        --out outputs/audit/got10k_dist.csv

The tool supports both old-schema labels (field 'state') and new-schema
labels (fields 'localization_state', 'confidence_state', 'derived_state').
It reports per-axis distributions where available.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

# Old schema (pre-composite) state names
_OLD_STATE_NAMES = ["CONFIRMED", "UNCERTAIN", "OCCLUDED", "LOST", "DISTRACTOR", "FALSE_CONFIRMED"]

# New schema — LocalizationState
_LOC_NAMES = ["STABLE", "UNCERTAIN", "LOST"]

# New schema — DerivedState
_DERIVED_NAMES = ["CORRECT_CONFIRMED", "CORRECT_UNCERTAIN", "LOST_AWARE", "FALSE_CONFIRMED"]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _detect_schema(rows: list[dict]) -> str:
    """Return 'new' if rows have localization_state, else 'old'."""
    for r in rows[:20]:
        if "localization_state" in r:
            return "new"
    return "old"


def _count_states(
    rows: list[dict],
    schema: str,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Return (loc_counts, derived_counts, old_counts) dicts."""
    loc_counts: dict[str, int] = defaultdict(int)
    derived_counts: dict[str, int] = defaultdict(int)
    old_counts: dict[str, int] = defaultdict(int)

    for r in rows:
        if schema == "new":
            loc_id = r.get("localization_state", 0)
            loc_name = r.get("localization_state_name") or (
                _LOC_NAMES[loc_id] if 0 <= loc_id < len(_LOC_NAMES) else f"loc_{loc_id}"
            )
            loc_counts[loc_name] += 1

            derived_id = r.get("derived_state", 0)
            derived_name = r.get("derived_state_name") or (
                _DERIVED_NAMES[derived_id] if 0 <= derived_id < len(_DERIVED_NAMES) else f"derived_{derived_id}"
            )
            derived_counts[derived_name] += 1
        else:
            old_id = r.get("state", 0)
            old_name = r.get("state_name") or (
                _OLD_STATE_NAMES[old_id] if 0 <= old_id < len(_OLD_STATE_NAMES) else f"state_{old_id}"
            )
            old_counts[old_name] += 1

    return dict(loc_counts), dict(derived_counts), dict(old_counts)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def audit(labels_dir: Path, out_path: Path) -> int:
    """Audit label distribution.  Returns 0 on OK, 1 on bad distribution."""
    jsonl_files = sorted(labels_dir.rglob("*.jsonl"))
    if not jsonl_files:
        print(f"[audit] ERROR: no *.jsonl files found under {labels_dir}", flush=True)
        return 2

    print(f"[audit] Found {len(jsonl_files)} JSONL files under {labels_dir}", flush=True)

    all_rows: list[dict] = []
    per_seq_stats: list[dict] = []

    for jf in jsonl_files:
        rows = _load_jsonl(jf)
        if not rows:
            continue
        schema = _detect_schema(rows)
        loc_c, derived_c, old_c = _count_states(rows, schema)
        all_rows.extend(rows)

        # Identify sequence name from rows or file stem
        seq_name = rows[0].get("sequence", jf.stem)
        dataset = rows[0].get("dataset", "unknown")

        per_seq_stats.append({
            "sequence": seq_name,
            "dataset": dataset,
            "n_frames": len(rows),
            "schema": schema,
            "loc_counts": loc_c,
            "derived_counts": derived_c,
            "old_counts": old_c,
        })

    if not all_rows:
        print("[audit] ERROR: all JSONL files were empty", flush=True)
        return 2

    total = len(all_rows)
    schema = _detect_schema(all_rows)
    loc_total, derived_total, old_total = _count_states(all_rows, schema)

    print(f"[audit] Total frames: {total:,}", flush=True)
    print(f"[audit] Schema: {schema}", flush=True)

    if schema == "new":
        print("\n  Localization state distribution:")
        for name, cnt in sorted(loc_total.items()):
            pct = 100.0 * cnt / total
            print(f"    {name:30s}: {cnt:8,}  ({pct:.2f}%)", flush=True)
        print("\n  Derived state distribution:")
        for name, cnt in sorted(derived_total.items()):
            pct = 100.0 * cnt / total
            print(f"    {name:30s}: {cnt:8,}  ({pct:.2f}%)", flush=True)
    else:
        print("\n  State distribution (old schema):")
        for name, cnt in sorted(old_total.items()):
            pct = 100.0 * cnt / total
            print(f"    {name:30s}: {cnt:8,}  ({pct:.2f}%)", flush=True)

    # --- Write CSV ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "sequence", "dataset", "n_frames", "schema",
            "STABLE_n", "STABLE_pct",
            "UNCERTAIN_n", "UNCERTAIN_pct",
            "LOST_n", "LOST_pct",
            "CORRECT_CONFIRMED_n", "CORRECT_CONFIRMED_pct",
            "CORRECT_UNCERTAIN_n", "CORRECT_UNCERTAIN_pct",
            "LOST_AWARE_n", "LOST_AWARE_pct",
            "FALSE_CONFIRMED_n", "FALSE_CONFIRMED_pct",
            "OLD_state_dominant",
        ])
        for s in per_seq_stats:
            n = s["n_frames"]
            lc = s["loc_counts"]
            dc = s["derived_counts"]
            oc = s["old_counts"]

            def pct(d: dict, k: str) -> str:
                v = d.get(k, 0)
                return f"{100.0 * v / n:.2f}" if n > 0 else "0.00"

            old_dom = max(oc, key=oc.get, default="") if oc else ""
            writer.writerow([
                s["sequence"], s["dataset"], n, s["schema"],
                lc.get("STABLE", 0), pct(lc, "STABLE"),
                lc.get("UNCERTAIN", 0), pct(lc, "UNCERTAIN"),
                lc.get("LOST", 0), pct(lc, "LOST"),
                dc.get("CORRECT_CONFIRMED", 0), pct(dc, "CORRECT_CONFIRMED"),
                dc.get("CORRECT_UNCERTAIN", 0), pct(dc, "CORRECT_UNCERTAIN"),
                dc.get("LOST_AWARE", 0), pct(dc, "LOST_AWARE"),
                dc.get("FALSE_CONFIRMED", 0), pct(dc, "FALSE_CONFIRMED"),
                old_dom,
            ])

    print(f"\n[audit] CSV written: {out_path}", flush=True)

    # --- Write summary.json ---
    summary: dict = {
        "labels_dir": str(labels_dir),
        "total_frames": total,
        "n_sequences": len(per_seq_stats),
        "n_jsonl_files": len(jsonl_files),
        "schema": schema,
    }
    if schema == "new":
        summary["localization_distribution"] = {
            k: {"count": v, "pct": round(100.0 * v / total, 4)}
            for k, v in sorted(loc_total.items())
        }
        summary["derived_distribution"] = {
            k: {"count": v, "pct": round(100.0 * v / total, 4)}
            for k, v in sorted(derived_total.items())
        }
    else:
        summary["state_distribution"] = {
            k: {"count": v, "pct": round(100.0 * v / total, 4)}
            for k, v in sorted(old_total.items())
        }

    summary_path = out_path.with_suffix(".json").with_stem(out_path.stem + "_summary")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[audit] Summary JSON written: {summary_path}", flush=True)

    # --- Exit-code checks ---
    exit_code = 0

    # Check 1: any state < 0.5% support?
    check_counts = loc_total if schema == "new" else old_total
    for state_name, cnt in check_counts.items():
        pct_val = 100.0 * cnt / total
        if pct_val < 0.5:
            print(
                f"[audit] WARNING: state '{state_name}' has only {pct_val:.2f}% support "
                f"({cnt:,} / {total:,} frames) — below 0.5% threshold.",
                flush=True,
            )
            exit_code = 1

    # Check 2: all sequences mapped to single dominant state?
    dominant_states: set[str] = set()
    for s in per_seq_stats:
        if schema == "new":
            lc = s["loc_counts"]
            if lc:
                dominant_states.add(max(lc, key=lc.get))
        else:
            oc = s["old_counts"]
            if oc:
                dominant_states.add(max(oc, key=oc.get))

    if len(dominant_states) == 1 and len(per_seq_stats) > 1:
        print(
            f"[audit] WARNING: all {len(per_seq_stats)} sequences have the same "
            f"dominant state: '{next(iter(dominant_states))}'. "
            "This likely indicates a degenerate label set.",
            flush=True,
        )
        exit_code = 1

    if exit_code == 0:
        print("[audit] Distribution OK — all states >= 0.5% and multiple dominant states found.",
              flush=True)
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="CSC label distribution auditor")
    parser.add_argument("--labels_dir", required=True, type=Path,
                        help="Root directory containing *.jsonl label files")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output CSV path (sibling summary.json also written)")
    args = parser.parse_args()

    code = audit(args.labels_dir, args.out)
    sys.exit(code)


if __name__ == "__main__":
    main()
