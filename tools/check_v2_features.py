"""Level 1 distribution check for CSC V2 features with cohort-based Cohen's d.

Compares the feature distribution of the FALSE_CONFIRMED cohort against
several CORRECT_CONFIRMED cohorts (and LOST_AWARE) to verify that the
new V2 slots actually separate the FC failure mode from natural
"object growing in frame" cases.

Cohorts
-------
- FC                       : derived_state == 3
- CC_all                   : derived_state == 0
- CC_aerial                : CC AND dataset in {dtb70, uavdt_sot,
                             visdrone_sot, uavtrack112}
- CC_large_bbox            : CC AND log_area_ratio_to_init > p75 of CC
- CC_aerial_large_bbox     : intersection of CC_aerial and CC_large_bbox
- LA                       : derived_state == 2

Gates (V2 critical features)
----------------------------
- scale_smoothness_8 (slot 14)   : d(FC, CC_large_bbox)        >= 0.2
- aspect_instability_8 (slot 15) : d(FC, CC_large_bbox)        >= 0.2
- edge_pressure_score (slot 11)  : d(FC, CC_aerial)            >= 0.5
- log_aspect_ratio (slot 8)      : var(FC) > 0.01 AND var(CC_all) > 0.01

Exit code is 0 if all gates pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csc_lib.csc.features import (  # noqa: E402
    FEATURE_NAMES,
    FEATURE_NAMES_V2,
    build_sequence_features,
    build_sequence_features_v2,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AERIAL_DATASETS: frozenset[str] = frozenset({
    "dtb70",
    "uavdt_sot",
    "visdrone_sot",
    "uavtrack112",
})

# Derived-state codes (paper schema):
#   0 = CORRECT_CONFIRMED  (CC)
#   1 = CORRECT_UNCERTAIN
#   2 = LOST_AWARE         (LA)
#   3 = FALSE_CONFIRMED    (FC)
STATE_CC: int = 0
STATE_LA: int = 2
STATE_FC: int = 3

# Slot index of log_area_ratio_to_init in BOTH V1 and V2 feature vectors.
LOG_AREA_RATIO_SLOT: int = 12

# Default fallback image size when label rows lack image_size.
DEFAULT_IMAGE_SIZE: tuple[int, int] = (1280, 720)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _group_by_sequence(jsonl_path: Path) -> dict[tuple[str, str], list[dict]]:
    """Group label rows by (dataset, sequence) and sort by frame_idx."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in _iter_jsonl(jsonl_path):
        key = (row.get("dataset", ""), row.get("sequence", ""))
        groups[key].append(row)
    for key, rows in groups.items():
        rows.sort(key=lambda r: int(r.get("frame_idx", 0)))
    return groups


def _discover_label_files(labels_dir: Path) -> list[Path]:
    """Return sorted list of all labels.jsonl files under ``labels_dir``."""
    paths = sorted(labels_dir.rglob("labels.jsonl"))
    return paths


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _select_builder(feature_version: str):
    if feature_version == "v1":
        return build_sequence_features, FEATURE_NAMES
    if feature_version == "v2":
        return build_sequence_features_v2, FEATURE_NAMES_V2
    raise ValueError(
        f"Unknown feature_version={feature_version!r} (expected 'v1' or 'v2')"
    )


def _resolve_image_size(rows: list[dict], warn_state: dict) -> tuple[int, int]:
    """Pick image_size from the first row if present, else the default.

    ``warn_state`` is used to emit the missing-image-size warning at most
    once per process.
    """
    for r in rows:
        sz = r.get("image_size")
        if sz is None:
            continue
        if isinstance(sz, (list, tuple)) and len(sz) == 2:
            return int(sz[0]), int(sz[1])
    if not warn_state.get("warned"):
        print(
            f"[warn] no image_size in label rows; falling back to "
            f"{DEFAULT_IMAGE_SIZE} for all sequences without it",
            file=sys.stderr,
        )
        warn_state["warned"] = True
    return DEFAULT_IMAGE_SIZE


# ---------------------------------------------------------------------------
# Cohort accumulation
# ---------------------------------------------------------------------------


class _CohortAccumulator:
    """Streams float32 feature rows into a list of arrays per cohort.

    Memory-friendly compromise: we keep per-sequence chunks rather than
    one big array and concatenate at the end.  For ~10M frames x 16 cols
    x 4 bytes that is ~640 MB peak, which is acceptable.
    """

    def __init__(self) -> None:
        self.chunks: dict[str, list[np.ndarray]] = defaultdict(list)

    def add(self, name: str, rows: np.ndarray) -> None:
        if rows.size == 0:
            return
        self.chunks[name].append(rows)

    def materialize(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, parts in self.chunks.items():
            if not parts:
                out[name] = np.zeros((0, 0), dtype=np.float32)
            else:
                out[name] = np.concatenate(parts, axis=0)
        return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _cohens_d(a: np.ndarray, b: np.ndarray, *, min_n: int = 30) -> float | None:
    """Cohen's d using the unbiased pooled std formula d = (mu_a - mu_b) /
    sqrt((var_a + var_b) / 2).  Returns None when either group has fewer
    than ``min_n`` samples, or when the pooled std is zero / NaN.
    """
    if a.size < min_n or b.size < min_n:
        return None
    var_a = float(np.var(a, ddof=1)) if a.size > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if b.size > 1 else 0.0
    denom = (var_a + var_b) / 2.0
    if denom <= 0.0 or not np.isfinite(denom):
        return None
    d = (float(np.mean(a)) - float(np.mean(b))) / float(np.sqrt(denom))
    if not np.isfinite(d):
        return None
    return d


def _fmt(d: float | None) -> str:
    return "  n/a " if d is None else f"{d:+.3f}"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def _build_cohorts(
    labels_dir: Path,
    feature_version: str,
    max_sequences: int | None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Stream sequences from ``labels_dir`` and return materialized cohorts.

    Cohort sizes (frame counts) are returned alongside the feature arrays.
    Note that CC_large_bbox is computed in a second pass after all CC
    rows are collected (because the p75 threshold depends on the full
    CC distribution).
    """
    builder, feat_names = _select_builder(feature_version)
    label_files = _discover_label_files(labels_dir)
    if not label_files:
        raise FileNotFoundError(
            f"No labels.jsonl under {labels_dir!r}. "
            "Run the labeling pipeline first or pass --labels-dir."
        )

    # First we collect (state, dataset) and the feature row separately.
    # We need three groups before the second pass:
    #   * FC rows (always kept as-is)
    #   * LA rows (always kept as-is)
    #   * CC rows + their dataset tag (used for CC_all / CC_aerial / etc.)
    fc_chunks: list[np.ndarray] = []
    la_chunks: list[np.ndarray] = []
    cc_chunks: list[np.ndarray] = []
    cc_dataset_chunks: list[np.ndarray] = []  # str array, same length as cc_chunks

    # Plan: total sequences across all label files (for tqdm).
    seq_total = 0
    seq_groups_per_file: list[
        tuple[Path, dict[tuple[str, str], list[dict]]]
    ] = []
    for lf in label_files:
        groups = _group_by_sequence(lf)
        seq_groups_per_file.append((lf, groups))
        seq_total += len(groups)
    if max_sequences is not None:
        seq_total = min(seq_total, max_sequences)

    warn_state: dict = {}
    seqs_processed = 0
    progress = tqdm(total=seq_total, desc=f"sequences ({feature_version})", unit="seq")
    try:
        for lf, groups in seq_groups_per_file:
            for (dataset, sequence), rows in groups.items():
                if max_sequences is not None and seqs_processed >= max_sequences:
                    break
                if not rows:
                    progress.update(1)
                    seqs_processed += 1
                    continue
                image_size = _resolve_image_size(rows, warn_state)
                try:
                    feats = builder(rows, image_size).astype(np.float32, copy=False)
                except Exception as exc:  # pragma: no cover (defensive)
                    print(
                        f"[warn] feature build failed for {dataset}/{sequence}: {exc}",
                        file=sys.stderr,
                    )
                    progress.update(1)
                    seqs_processed += 1
                    continue
                states = np.fromiter(
                    (int(r.get("derived_state", 0)) for r in rows),
                    dtype=np.int64,
                    count=len(rows),
                )

                fc_mask = states == STATE_FC
                la_mask = states == STATE_LA
                cc_mask = states == STATE_CC

                if fc_mask.any():
                    fc_chunks.append(feats[fc_mask])
                if la_mask.any():
                    la_chunks.append(feats[la_mask])
                if cc_mask.any():
                    cc_feats = feats[cc_mask]
                    cc_chunks.append(cc_feats)
                    # Tag every CC row with its dataset name.
                    cc_dataset_chunks.append(
                        np.full(cc_feats.shape[0], dataset, dtype=object)
                    )

                progress.update(1)
                seqs_processed += 1
            if max_sequences is not None and seqs_processed >= max_sequences:
                break
    finally:
        progress.close()

    # Materialize.
    fc_arr = (
        np.concatenate(fc_chunks, axis=0)
        if fc_chunks
        else np.zeros((0, len(feat_names)), dtype=np.float32)
    )
    la_arr = (
        np.concatenate(la_chunks, axis=0)
        if la_chunks
        else np.zeros((0, len(feat_names)), dtype=np.float32)
    )
    cc_arr = (
        np.concatenate(cc_chunks, axis=0)
        if cc_chunks
        else np.zeros((0, len(feat_names)), dtype=np.float32)
    )
    cc_datasets = (
        np.concatenate(cc_dataset_chunks, axis=0)
        if cc_dataset_chunks
        else np.zeros((0,), dtype=object)
    )

    # Build the derived CC subsets.
    aerial_mask = np.array(
        [d in AERIAL_DATASETS for d in cc_datasets], dtype=bool
    )
    if cc_arr.shape[0] > 0:
        log_area = cc_arr[:, LOG_AREA_RATIO_SLOT]
        if log_area.size >= 2:
            p75 = float(np.percentile(log_area, 75.0))
        else:
            p75 = float(log_area.max()) if log_area.size else 0.0
        large_mask = log_area > p75
    else:
        p75 = 0.0
        large_mask = np.zeros((0,), dtype=bool)

    cohorts: dict[str, np.ndarray] = {
        "FC": fc_arr,
        "CC_all": cc_arr,
        "CC_aerial": cc_arr[aerial_mask] if cc_arr.size else cc_arr,
        "CC_large_bbox": cc_arr[large_mask] if cc_arr.size else cc_arr,
        "CC_aerial_large_bbox": (
            cc_arr[aerial_mask & large_mask] if cc_arr.size else cc_arr
        ),
        "LA": la_arr,
    }
    sizes = {name: int(arr.shape[0]) for name, arr in cohorts.items()}
    sizes["_cc_p75_log_area_ratio"] = -1  # placeholder slot
    sizes["__p75__"] = -1
    sizes_meta = dict(sizes)
    sizes_meta["__p75__"] = round(p75, 6)

    return cohorts, sizes_meta


def _compute_d_table(
    cohorts: dict[str, np.ndarray], feat_names: tuple[str, ...]
) -> list[dict]:
    """Compute d(FC, X) for each X in the comparison set, per slot."""
    fc = cohorts["FC"]
    comparisons = ["CC_all", "CC_aerial", "CC_large_bbox", "CC_aerial_large_bbox", "LA"]
    rows: list[dict] = []
    for slot, name in enumerate(feat_names):
        row: dict = {"slot": slot, "name": name}
        if fc.size:
            fc_col = fc[:, slot]
            row["fc_mean"] = float(np.mean(fc_col))
            row["fc_var"] = float(np.var(fc_col, ddof=1)) if fc_col.size > 1 else 0.0
        else:
            row["fc_mean"] = float("nan")
            row["fc_var"] = float("nan")
        for cohort_name in comparisons:
            arr = cohorts.get(cohort_name)
            if arr is None or arr.size == 0 or fc.size == 0:
                row[f"d_{cohort_name}"] = None
                continue
            row[f"d_{cohort_name}"] = _cohens_d(fc[:, slot], arr[:, slot])
        rows.append(row)
    return rows


def _print_table(d_rows: list[dict]) -> None:
    header_cols = [
        ("slot", 4),
        ("name", 28),
        ("d(FC,CC_all)", 14),
        ("d(FC,CC_aerial)", 16),
        ("d(FC,CC_lg_bbox)", 18),
        ("d(FC,CC_aer_lg)", 17),
        ("d(FC,LA)", 11),
    ]
    header = "  ".join(f"{h:<{w}}" for h, w in header_cols)
    print(header)
    print("-" * len(header))
    for r in d_rows:
        line = "  ".join([
            f"{r['slot']:<4d}",
            f"{r['name']:<28}",
            f"{_fmt(r['d_CC_all']):<14}",
            f"{_fmt(r['d_CC_aerial']):<16}",
            f"{_fmt(r['d_CC_large_bbox']):<18}",
            f"{_fmt(r['d_CC_aerial_large_bbox']):<17}",
            f"{_fmt(r['d_LA']):<11}",
        ])
        print(line)


# ---------------------------------------------------------------------------
# Gates (V2 only)
# ---------------------------------------------------------------------------


def _evaluate_gates(
    cohorts: dict[str, np.ndarray],
    d_rows: list[dict],
    feature_version: str,
) -> tuple[list[dict], bool]:
    """Apply the V2 gates.  Returns (gate_records, all_pass).

    Gates only apply to V2 features.  When ``feature_version == 'v1'``
    the gate evaluation is skipped (returns empty list, passes=True).
    """
    if feature_version != "v2":
        return [], True

    name_to_row = {r["name"]: r for r in d_rows}
    fc = cohorts.get("FC")
    cc_all = cohorts.get("CC_all")

    def _row_d(feat_name: str, key: str) -> float | None:
        r = name_to_row.get(feat_name)
        if r is None:
            return None
        return r.get(key)

    def _all_d(feat_name: str) -> list[float]:
        """Return d values across all 5 cohort comparisons (filter None)."""
        r = name_to_row.get(feat_name)
        if r is None:
            return []
        keys = ("d_CC_all", "d_CC_aerial", "d_CC_large_bbox",
                "d_CC_aerial_large_bbox", "d_LA")
        return [r.get(k) for k in keys if r.get(k) is not None]

    def _sign_consistent(feat_name: str) -> bool:
        """True if d has same sign across all cohorts (strong inverse signal counts)."""
        ds = _all_d(feat_name)
        if not ds:
            return False
        signs = {(1 if d > 0 else (-1 if d < 0 else 0)) for d in ds}
        signs.discard(0)
        return len(signs) == 1

    def _slot_var(arr: np.ndarray | None, feat_name: str) -> float | None:
        if arr is None or arr.size == 0:
            return None
        if feat_name not in name_to_row:
            return None
        slot = name_to_row[feat_name]["slot"]
        col = arr[:, slot]
        if col.size <= 1:
            return 0.0
        return float(np.var(col, ddof=1))

    gates: list[dict] = []

    # Gate 1: scale_smoothness_8 — |d(FC, CC_large_bbox)| >= 0.2 + sign consistent
    d1 = _row_d("scale_smoothness_8", "d_CC_large_bbox")
    g1_consistent = _sign_consistent("scale_smoothness_8")
    gates.append({
        "name": "scale_smoothness_8 :: |d(FC, CC_large_bbox)| >= 0.2 AND sign consistent",
        "value": d1,
        "abs_value": abs(d1) if d1 is not None else None,
        "sign_consistent": g1_consistent,
        "threshold": 0.2,
        "pass": (d1 is not None and abs(d1) >= 0.2 and g1_consistent),
    })

    # Gate 2: aspect_instability_8 — |d(FC, CC_large_bbox)| >= 0.2 + sign consistent
    d2 = _row_d("aspect_instability_8", "d_CC_large_bbox")
    g2_consistent = _sign_consistent("aspect_instability_8")
    gates.append({
        "name": "aspect_instability_8 :: |d(FC, CC_large_bbox)| >= 0.2 AND sign consistent",
        "value": d2,
        "abs_value": abs(d2) if d2 is not None else None,
        "sign_consistent": g2_consistent,
        "threshold": 0.2,
        "pass": (d2 is not None and abs(d2) >= 0.2 and g2_consistent),
    })

    # Gate 3: edge_pressure_score — |d(FC, CC_aerial)| >= 0.5
    d3 = _row_d("edge_pressure_score", "d_CC_aerial")
    gates.append({
        "name": "edge_pressure_score :: |d(FC, CC_aerial)| >= 0.5",
        "value": d3,
        "abs_value": abs(d3) if d3 is not None else None,
        "threshold": 0.5,
        "pass": (d3 is not None and abs(d3) >= 0.5),
    })

    # Gate 4: log_aspect_ratio — var(FC) > 0.01 AND var(CC_all) > 0.01
    var_fc = _slot_var(fc, "log_aspect_ratio")
    var_cc = _slot_var(cc_all, "log_aspect_ratio")
    g4_pass = (
        var_fc is not None and var_cc is not None
        and var_fc > 0.01 and var_cc > 0.01
    )
    gates.append({
        "name": "log_aspect_ratio :: var(FC) > 0.01 AND var(CC_all) > 0.01",
        "value": {"var_FC": var_fc, "var_CC_all": var_cc},
        "threshold": 0.01,
        "pass": g4_pass,
    })

    all_pass = all(g["pass"] for g in gates)
    return gates, all_pass


def _print_gates(gates: list[dict]) -> None:
    if not gates:
        print("\n[gates] feature_version=v1 — no V2 gates to evaluate.")
        return
    print("\nGates:")
    for g in gates:
        verdict = "PASS" if g["pass"] else "FAIL"
        v = g["value"]
        if isinstance(v, dict):
            v_str = ", ".join(
                f"{k}={vv:.4g}" if isinstance(vv, float) else f"{k}={vv}"
                for k, vv in v.items()
            )
        elif v is None:
            v_str = "n/a"
        else:
            v_str = f"{v:+.3f}"
        print(f"  [{verdict}] {g['name']}  (got: {v_str})")


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _save_report(
    out_path: Path,
    *,
    args: argparse.Namespace,
    feat_names: tuple[str, ...],
    cohort_sizes: dict[str, int],
    d_rows: list[dict],
    gates: list[dict],
    all_pass: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels_dir": str(args.labels_dir),
        "feature_version": args.feature_version,
        "max_sequences": args.max_sequences,
        "feature_names": list(feat_names),
        "cohort_sizes": cohort_sizes,
        "per_slot_cohens_d": d_rows,
        "gates": gates,
        "all_gates_pass": bool(all_pass),
    }
    with open(out_path, "w") as fh:
        json.dump(_json_safe(payload), fh, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("outputs/csc_labels/sglatrack/v3fix_combined"),
        help="Path to the label root containing per-dataset labels.jsonl files.",
    )
    p.add_argument(
        "--feature-version",
        type=str,
        choices=["v1", "v2"],
        default="v2",
        help="Which feature builder to use.",
    )
    p.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Optional cap for fast iteration. Default: process all sequences.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/v3fix_diag/cohort_distribution.json"),
        help="Path to output JSON report.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    labels_dir = args.labels_dir
    if not labels_dir.exists():
        print(
            f"[fatal] --labels-dir does not exist: {labels_dir}\n"
            "        Pass an existing label root, e.g.\n"
            "        --labels-dir outputs/csc_labels/sglatrack/v3fix_combined",
            file=sys.stderr,
        )
        return 2

    print(
        f"[info] labels_dir={labels_dir}  feature_version={args.feature_version}  "
        f"max_sequences={args.max_sequences}"
    )

    cohorts, sizes_meta = _build_cohorts(
        labels_dir=labels_dir,
        feature_version=args.feature_version,
        max_sequences=args.max_sequences,
    )

    # Print cohort sizes.
    print("\nCohort sizes (frames):")
    for name in ("FC", "CC_all", "CC_aerial", "CC_large_bbox",
                 "CC_aerial_large_bbox", "LA"):
        n = sizes_meta.get(name, 0)
        print(f"  {name:<24} {n:>10d}")
    p75_val = sizes_meta.get("__p75__")
    if p75_val is not None and p75_val != -1:
        print(f"  (CC log_area_ratio p75 = {p75_val:+.4f})")

    # Per-slot Cohen's d table.
    _, feat_names = _select_builder(args.feature_version)
    d_rows = _compute_d_table(cohorts, feat_names)

    print(f"\nCohen's d (FC vs cohort) — feature_version={args.feature_version}")
    _print_table(d_rows)

    # Gates (V2 only).
    gates, all_pass = _evaluate_gates(cohorts, d_rows, args.feature_version)
    _print_gates(gates)

    # Save report.
    cohort_sizes_json = {
        k: v for k, v in sizes_meta.items() if not k.startswith("_")
    }
    cohort_sizes_json["cc_log_area_ratio_p75"] = sizes_meta.get("__p75__")
    _save_report(
        args.out,
        args=args,
        feat_names=feat_names,
        cohort_sizes=cohort_sizes_json,
        d_rows=d_rows,
        gates=gates,
        all_pass=all_pass,
    )
    print(f"\n[report] wrote {args.out}")

    if args.feature_version != "v2":
        print("[note] feature_version=v1 — gates were not evaluated.")
        return 0
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
