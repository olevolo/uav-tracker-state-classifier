"""Re-derive CSC labels with the NEW weak_labeler (strict confidence-required).

Operates on existing labels.jsonl rows that already contain calibrated
confidence / APCE / PSR values.  Re-applies label_frame() per frame and
recomputes the V3 forecast labels (failure_next_10, false_confirmed_next_10,
lost_aware_next_10).

This is functionally identical to a full build_scene_state_labels.py rerun
because:
  1. existing rows store CALIBRATED telemetry (already mapped to [0,1]
     percentile rank by PercentileConfidenceCalibrator);
  2. the only labeling change is the OR-gate → strict-AND patch in
     weak_labeler.py:106-122;
  3. forecast labels are derived purely from the derived_state sequence.

Run:
  .venv/bin/python tools/relabel_v3fix_inplace.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from csc_lib.csc.labeling.label_schema import DerivedState  # noqa: E402
from csc_lib.csc.labeling.risk_labeler import build_future_risk_labels  # noqa: E402
from csc_lib.csc.labeling.weak_labeler import (  # noqa: E402
    LabelingThresholds,
    label_frame,
)

LABELS_ROOT = Path("outputs/csc_labels/sglatrack/v3fix_combined")
BACKUP_ROOT = Path("outputs/csc_labels/sglatrack/_ARCHIVE/v3fix_pre_relabel_2026_05_29")
HORIZON = 10
TH = LabelingThresholds()


def relabel_jsonl(jsonl_path: Path, dry_run: bool = False) -> dict:
    """Re-derive labels for one jsonl file.  Returns stats dict."""
    target = jsonl_path.resolve()
    print(f"[load] {jsonl_path}")
    print(f"       resolved → {target}")

    t0 = time.time()
    rows = []
    with target.open() as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"       {len(rows):,} rows  ({time.time()-t0:.1f}s)")

    # Group by (dataset, sequence) — keep original positions for write-back order.
    by_seq: dict = defaultdict(list)
    pos_index: dict = {}
    for pos, r in enumerate(rows):
        key = (r["dataset"], r["sequence"])
        by_seq[key].append(pos)
        pos_index[pos] = r

    old_state_counts: dict = defaultdict(int)
    new_state_counts: dict = defaultdict(int)

    t0 = time.time()
    for (ds, seq), positions in by_seq.items():
        positions.sort(key=lambda p: pos_index[p].get("frame_idx", 0))
        consecutive_low_iou = 0
        # ---- Stage 1: per-frame relabel
        for pos in positions:
            r = pos_index[pos]
            old_state = r.get("derived_state", -1)
            if 0 <= old_state <= 3:
                old_state_counts[old_state] += 1

            iou = r.get("iou")
            if iou is not None and iou < TH.tau_lost_iou:
                consecutive_low_iou += 1
            else:
                consecutive_low_iou = 0

            aux_old = r.get("aux") or {}
            absent = bool(r.get("absent", False))
            full_occlusion = bool(aux_old.get("occlusion", False))
            out_of_view = bool(aux_old.get("out_of_view", False))

            loc, conf, derived, aux, source, noisy = label_frame(
                iou=iou,
                confidence=r.get("confidence"),
                apce=r.get("apce"),
                psr=r.get("psr"),
                full_occlusion=full_occlusion,
                out_of_view=out_of_view,
                absent=absent,
                consecutive_low_iou=consecutive_low_iou,
                thresholds=TH,
            )
            r["localization_state"] = int(loc)
            r["confidence_state"] = int(conf)
            r["derived_state"] = int(derived)
            r["false_confirmed_flag"] = derived == DerivedState.FALSE_CONFIRMED
            r["aux"] = aux
            r["label_source"] = int(source)
            r["label_noisy"] = noisy
            new_state_counts[int(derived)] += 1

        # ---- Stage 2: forecast labels (V3)
        derived_seq = [pos_index[p]["derived_state"] for p in positions]
        risk = build_future_risk_labels(derived_seq, horizon=HORIZON)
        for pos, rl in zip(positions, risk):
            r = pos_index[pos]
            r["failure_next_10"] = int(rl["failure_next_10"])
            r["false_confirmed_next_10"] = int(rl["false_confirmed_next_10"])
            r["lost_aware_next_10"] = int(rl["lost_aware_next_10"])
            r["ignore_forecast"] = int(rl["ignore_forecast"])
    print(f"[relabel]  {time.time()-t0:.1f}s")

    # ---- Print per-jsonl distribution
    print(f"  state distribution (CC/CU/LA/FC):")
    n = sum(old_state_counts.values())
    print(
        "    OLD:  "
        + "  ".join(
            f"{['CC','CU','LA','FC'][s]}={100*old_state_counts[s]/max(n,1):>5.2f}%"
            for s in range(4)
        )
    )
    print(
        "    NEW:  "
        + "  ".join(
            f"{['CC','CU','LA','FC'][s]}={100*new_state_counts[s]/max(n,1):>5.2f}%"
            for s in range(4)
        )
    )

    if dry_run:
        print(f"[dry-run] would write {len(rows):,} rows to {target}")
        return {"old": dict(old_state_counts), "new": dict(new_state_counts), "n": n}

    # ---- Atomic write: tmp file then rename
    t0 = time.time()
    tmp = target.with_suffix(target.suffix + ".relabel_tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(target)
    print(f"[write]    {len(rows):,} rows to {target}  ({time.time()-t0:.1f}s)")

    return {"old": dict(old_state_counts), "new": dict(new_state_counts), "n": n}


def backup_originals():
    """Resolve all symlinks under v3fix_combined and copy resolved files to BACKUP_ROOT."""
    if BACKUP_ROOT.exists():
        print(f"[backup]  {BACKUP_ROOT} already exists — skipping (won't overwrite)")
        return
    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    for jsonl in sorted(LABELS_ROOT.glob("*/labels.jsonl")):
        target = jsonl.resolve()
        rel = jsonl.relative_to(LABELS_ROOT)
        dst = BACKUP_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, dst)
        size_mb = dst.stat().st_size / (1024 * 1024)
        print(f"[backup]  {target} → {dst}  ({size_mb:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute new labels but do not write back.")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip backing up originals (NOT recommended).")
    args = ap.parse_args()

    if not args.dry_run and not args.no_backup:
        backup_originals()

    agg_old: dict = defaultdict(int)
    agg_new: dict = defaultdict(int)
    for jsonl in sorted(LABELS_ROOT.glob("*/labels.jsonl")):
        stats = relabel_jsonl(jsonl, dry_run=args.dry_run)
        for k, v in stats["old"].items():
            agg_old[k] += v
        for k, v in stats["new"].items():
            agg_new[k] += v
        print()

    n = sum(agg_old.values())
    print("=" * 80)
    print("AGGREGATE state distribution (CC/CU/LA/FC):")
    print(
        "  OLD:  "
        + "  ".join(
            f"{['CC','CU','LA','FC'][s]}={100*agg_old[s]/max(n,1):>5.2f}%  ({agg_old[s]:,})"
            for s in range(4)
        )
    )
    print(
        "  NEW:  "
        + "  ".join(
            f"{['CC','CU','LA','FC'][s]}={100*agg_new[s]/max(n,1):>5.2f}%  ({agg_new[s]:,})"
            for s in range(4)
        )
    )


if __name__ == "__main__":
    main()
