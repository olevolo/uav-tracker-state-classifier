#!/usr/bin/env python3
"""Join V3 response-structure telemetry fields into CSC training labels (R4).

The weak-labeler wrote label rows with {iou, confidence, apce, psr, ...} but NOT
the 7 V3 response-structure fields (response_entropy + sm_*). Those fields ARE
present in the per-sequence baseline telemetry. This script joins them back in by
(dataset, split, sequence, frame_idx) so the V3 feature builder
(build_sequence_features_v3, which reads ``extra=row``) can consume them.

No tracker rerun is needed — the telemetry already exists on disk.

Join key per label row:
    telemetry = outputs/baselines/<tracker>/<dataset>/<split>/telemetry/<sequence>.jsonl
    matched by frame_idx.

Missing fields (e.g. the init frame, or a tracker that never emitted sm_*) are
left ABSENT in the output row — the builder degrades them to 0.0 safely.

Usage:
    python tools/join_v3_features_to_labels.py \
        --src outputs/csc_labels/sglatrack/v3fix_combined \
        --dst outputs/csc_labels/sglatrack/v3fix_combined_v3 \
        --tracker sglatrack
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

# The 7 fields the V3 feature builder reads (csc_lib/csc/features.py: V3_EXTRA_FIELDS).
V3_FIELDS = (
    "response_entropy",
    "sm_local_peak_margin",
    "sm_n_secondary",
    "sm_top1",
    "sm_heatmap_mass_topk",
    "sm_peak_margin",
    "sm_local_top2_ratio",
)

log = logging.getLogger("join_v3")


def _telemetry_path(baselines_root: Path, tracker: str, dataset: str,
                    split: str, sequence: str) -> Path:
    return baselines_root / tracker / dataset / split / "telemetry" / f"{sequence}.jsonl"


def _load_telemetry_v3(path: Path) -> dict[int, dict]:
    """frame_idx -> {present V3 fields}. Empty dict if file missing."""
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        fi = row.get("frame_idx")
        if fi is None:
            continue
        present = {k: row[k] for k in V3_FIELDS if k in row and row[k] is not None}
        if present:
            out[int(fi)] = present
    return out


def process_shard(shard_src: Path, shard_dst: Path, baselines_root: Path,
                  tracker: str) -> dict:
    src_labels = shard_src / "labels.jsonl"
    rows = [json.loads(l) for l in src_labels.read_text().splitlines() if l.strip()]

    # Group row indices by (dataset, split, sequence) so each telemetry file
    # is read exactly once.
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["dataset"], r["split"], r["sequence"])].append(i)

    n_rows = len(rows)
    n_filled = 0
    n_missing_file = 0
    missing_seqs: set[str] = set()

    for (dataset, split, sequence), idxs in groups.items():
        tel = _load_telemetry_v3(
            _telemetry_path(baselines_root, tracker, dataset, split, sequence)
        )
        if not tel:
            n_missing_file += 1
            missing_seqs.add(f"{dataset}/{split}/{sequence}")
            continue
        for i in idxs:
            fi = int(rows[i]["frame_idx"])
            fields = tel.get(fi)
            if fields:
                rows[i].update(fields)
                n_filled += 1

    shard_dst.mkdir(parents=True, exist_ok=True)
    with (shard_dst / "labels.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    return {
        "shard": shard_src.name,
        "n_rows": n_rows,
        "n_filled": n_filled,
        "fill_frac": round(n_filled / max(n_rows, 1), 4),
        "n_groups": len(groups),
        "n_missing_telemetry_files": n_missing_file,
        "missing_example": sorted(missing_seqs)[:5],
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source labels dir (v3fix_combined).")
    ap.add_argument("--dst", required=True, help="Dest labels dir (v3fix_combined_v3).")
    ap.add_argument("--tracker", default="sglatrack")
    ap.add_argument("--baselines_root", default="outputs/baselines")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    baselines_root = Path(args.baselines_root)

    shards = sorted(p.parent for p in src.rglob("labels.jsonl"))
    if not shards:
        raise SystemExit(f"no labels.jsonl found under {src}")
    log.info("Found %d shard(s): %s", len(shards), [s.name for s in shards])

    summary = []
    for shard_src in shards:
        rel = shard_src.relative_to(src)
        shard_dst = dst / rel
        log.info("Joining shard %s ...", rel)
        stats = process_shard(shard_src, shard_dst, baselines_root, args.tracker)
        log.info(
            "  %s: %d rows, %d filled (%.1f%%), %d seqs, %d missing-telemetry-files %s",
            stats["shard"], stats["n_rows"], stats["n_filled"],
            100 * stats["fill_frac"], stats["n_groups"],
            stats["n_missing_telemetry_files"],
            stats["missing_example"] if stats["n_missing_telemetry_files"] else "",
        )
        summary.append(stats)

    (dst / "join_v3_report.json").write_text(json.dumps(summary, indent=2))
    total_rows = sum(s["n_rows"] for s in summary)
    total_filled = sum(s["n_filled"] for s in summary)
    log.info("DONE. total %d rows, %d filled (%.1f%%). Report -> %s",
             total_rows, total_filled, 100 * total_filled / max(total_rows, 1),
             dst / "join_v3_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
