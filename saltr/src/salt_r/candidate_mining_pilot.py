"""SGLATrack top-K candidate mining pilot for SALT-RD false-confirmed cases.

This script answers one narrow architectural question before we build another
heavy teacher sidecar:

    When SGLATrack is false-confirmed, is the true target still present among
    the tracker score-map alternatives?

If the answer is yes, SALT-RD can learn a candidate-aware verifier using
top-K score-map candidates plus point/identity teacher signals.  If the answer
is no, the tracker response map itself has lost the target and we need an
external candidate generator (detector/SAM/TAM), not another memory embedding.

The script uses the candidate boxes exposed by ``SGLATracker.score_map_stats``
and evaluates oracle top-K recall against offline GT boxes from the SALT-RD NPZ.
It does not train a model and does not alter tracker behavior.

Example
-------
python -m salt_r.candidate_mining_pilot \\
  --npz saltr/data/salt_rd_v2_labels.npz \\
  --config-path configs/prod/salt.yaml \\
  --output saltr/results/candidate_mining_pilot_diagnostic.json \\
  --splits diagnostic
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class PilotSequence:
    key: str
    dataset: str
    name: str
    split: str


class _WrappedSequence:
    def __init__(self, name: str, frames: list, ground_truth: list) -> None:
        self.name = name
        self.frames = frames
        self.ground_truth = ground_truth

    @property
    def init_bbox(self):
        return self.ground_truth[0]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_paths() -> None:
    root = _repo_root()
    for extra in (root / "src", root / "saltr" / "src"):
        s = str(extra)
        if s not in sys.path:
            sys.path.insert(0, s)


def _bbox_iou_xywh(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _label_index(data: np.lib.npyio.NpzFile, label_name: str) -> int:
    names = [str(x) for x in data["label_names"].tolist()]
    if label_name not in names:
        raise ValueError(f"label {label_name!r} not found in label_names={names}")
    return names.index(label_name)


def _dataset_from_key(data: np.lib.npyio.NpzFile, key: str) -> str:
    ds_key = f"dataset/{key}"
    if ds_key in data.files:
        return str(data[ds_key])
    return key.split("/", 1)[0]


def _name_from_key(data: np.lib.npyio.NpzFile, key: str) -> str:
    name_key = f"sequence_name/{key}"
    if name_key in data.files:
        return str(data[name_key])
    return key.split("/", 1)[-1]


def _select_sequences(
    data: np.lib.npyio.NpzFile,
    splits: set[str],
    explicit_seqs: list[str] | None = None,
    max_val_seqs: int = 0,
    smoke_test: int = 0,
) -> list[PilotSequence]:
    keys = sorted(k[len("features/"):] for k in data.files if k.startswith("features/"))

    if explicit_seqs:
        wanted = set(explicit_seqs)
        missing = sorted(wanted - set(keys))
        if missing:
            raise ValueError(f"Requested sequences missing from NPZ: {missing}")
        keys = [k for k in keys if k in wanted]

    selected: list[PilotSequence] = []
    val_pool: dict[str, list[PilotSequence]] = collections.defaultdict(list)

    for key in keys:
        split = str(data[f"split/{key}"])
        dataset = _dataset_from_key(data, key)
        seq = PilotSequence(key=key, dataset=dataset, name=_name_from_key(data, key), split=split)
        if explicit_seqs or split in splits:
            if split == "val" and max_val_seqs > 0 and not explicit_seqs:
                val_pool[dataset].append(seq)
            elif split != "val" or max_val_seqs <= 0:
                selected.append(seq)

    if "val" in splits and max_val_seqs > 0 and not explicit_seqs:
        datasets = sorted(val_pool)
        cursor = {ds: 0 for ds in datasets}
        while len([s for s in selected if s.split == "val"]) < max_val_seqs:
            made_progress = False
            for ds in datasets:
                if len([s for s in selected if s.split == "val"]) >= max_val_seqs:
                    break
                idx = cursor[ds]
                if idx < len(val_pool[ds]):
                    selected.append(val_pool[ds][idx])
                    cursor[ds] += 1
                    made_progress = True
            if not made_progress:
                break

    selected.sort(key=lambda s: (s.split, s.dataset, s.name))
    if smoke_test > 0:
        selected = selected[:smoke_test]
    return selected


def _get_dataset_sequences(dataset_name: str) -> dict[str, object]:
    _ensure_paths()
    if dataset_name == "uav123":
        from uav_tracker.datasets.uav123 import UAV123Dataset
        return {seq.name: seq for seq in UAV123Dataset(root=None)}
    if dataset_name == "visdrone_sot":
        from uav_tracker.datasets.visdrone_sot import VisDroneSOTDataset
        return {seq.name: seq for seq in VisDroneSOTDataset(root=None)}
    if dataset_name == "dtb70":
        from uav_tracker.datasets.dtb70 import DTB70Dataset
        return {seq.name: seq for seq in DTB70Dataset(root=None)}
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _candidate_frame_record(
    frame_idx: int,
    candidates: list[dict],
    gt_bbox_xywh: np.ndarray,
    tracker_bbox_xywh: np.ndarray,
    false_confirmed: bool,
) -> dict[str, Any]:
    candidate_ious = [
        _bbox_iou_xywh(c.get("bbox", [0, 0, 0, 0]), gt_bbox_xywh)
        for c in candidates
    ]
    best_iou = max(candidate_ious) if candidate_ious else 0.0
    best_rank = int(np.argmax(candidate_ious)) if candidate_ious else -1
    top1_iou = candidate_ious[0] if candidate_ious else 0.0
    tracker_iou = _bbox_iou_xywh(tracker_bbox_xywh, gt_bbox_xywh)

    return {
        "frame_idx": int(frame_idx),
        "false_confirmed": bool(false_confirmed),
        "tracker_iou": float(tracker_iou),
        "n_candidates": int(len(candidates)),
        "top1_iou": float(top1_iou),
        "best_iou": float(best_iou),
        "best_rank": int(best_rank),
        "top3_hit_iou03": bool(any(iou >= 0.3 for iou in candidate_ious[:3])),
        "top5_hit_iou03": bool(any(iou >= 0.3 for iou in candidate_ious[:5])),
        "top3_hit_iou05": bool(any(iou >= 0.5 for iou in candidate_ious[:3])),
        "top5_hit_iou05": bool(any(iou >= 0.5 for iou in candidate_ious[:5])),
    }


def _aggregate_records(records: list[dict]) -> dict[str, Any]:
    if not records:
        return {
            "n_frames": 0,
            "mean_candidate_count": None,
            "top1_iou_mean": None,
            "best_iou_mean": None,
            "oracle_top3_recall_iou03": None,
            "oracle_top5_recall_iou03": None,
            "oracle_top3_recall_iou05": None,
            "oracle_top5_recall_iou05": None,
        }

    def mean(name: str) -> float:
        return float(np.mean([float(r[name]) for r in records]))

    return {
        "n_frames": int(len(records)),
        "mean_candidate_count": mean("n_candidates"),
        "top1_iou_mean": mean("top1_iou"),
        "best_iou_mean": mean("best_iou"),
        "oracle_top3_recall_iou03": mean("top3_hit_iou03"),
        "oracle_top5_recall_iou03": mean("top5_hit_iou03"),
        "oracle_top3_recall_iou05": mean("top3_hit_iou05"),
        "oracle_top5_recall_iou05": mean("top5_hit_iou05"),
        "tracker_iou_mean": mean("tracker_iou"),
    }


def _summarize_records(records: list[dict]) -> dict[str, Any]:
    fc = [r for r in records if r["false_confirmed"]]
    non_fc = [r for r in records if not r["false_confirmed"]]
    return {
        "all": _aggregate_records(records),
        "false_confirmed": _aggregate_records(fc),
        "not_false_confirmed": _aggregate_records(non_fc),
        "n_false_confirmed": int(len(fc)),
        "false_confirmed_rate": float(len(fc) / len(records)) if records else float("nan"),
    }


def _process_sequence(
    runner: object,
    seq_info: PilotSequence,
    seq_obj: object,
    data: np.lib.npyio.NpzFile,
    fc_idx: int,
    max_frames: int = 0,
    include_frame_records: bool = False,
) -> dict[str, Any]:
    frames = list(seq_obj.frames)
    gt_bboxes = list(seq_obj.ground_truth)
    if max_frames > 0:
        frames = frames[:max_frames]
        gt_bboxes = gt_bboxes[:max_frames]

    seq_for_run = _WrappedSequence(seq_obj.name, frames, gt_bboxes)
    entries = list(runner.run(seq_for_run))
    labels = data[f"labels/{seq_info.key}"][:len(entries)]
    gt_arr = data[f"bbox_gt/{seq_info.key}"][:len(entries)]
    pred_arr = data[f"bbox_pred/{seq_info.key}"][:len(entries)]

    records: list[dict] = []
    for t, entry in enumerate(entries):
        sms = entry.aux.get("score_map_stats", {})
        candidates = sms.get("candidates", []) or []
        records.append(_candidate_frame_record(
            frame_idx=t,
            candidates=candidates,
            gt_bbox_xywh=gt_arr[t],
            tracker_bbox_xywh=pred_arr[t],
            false_confirmed=bool(labels[t, fc_idx] > 0.5),
        ))

    result: dict[str, Any] = {
        "key": seq_info.key,
        "dataset": seq_info.dataset,
        "name": seq_info.name,
        "split": seq_info.split,
        "summary": _summarize_records(records),
        "_records": records,
    }
    if include_frame_records:
        result["frames"] = records
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_paths()
    npz_path = Path(args.npz)
    data = np.load(npz_path, allow_pickle=True)
    fc_idx = _label_index(data, "false_confirmed")
    splits = set(args.splits)
    seqs = _select_sequences(
        data,
        splits=splits,
        explicit_seqs=args.seqs,
        max_val_seqs=args.max_val_seqs,
        smoke_test=args.smoke_test,
    )

    if args.dry_run:
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "npz": str(npz_path),
            "dry_run": True,
            "selected_sequences": [s.__dict__ for s in seqs],
        }

    from uav_tracker.salt_runner import SALTRunner

    runner = SALTRunner.from_config(args.config_path)
    dataset_cache: dict[str, dict[str, object]] = {}
    sequence_results: list[dict[str, Any]] = []

    for i, seq_info in enumerate(seqs, start=1):
        print(
            f"[candidate_mining] {i}/{len(seqs)} {seq_info.key} split={seq_info.split}",
            flush=True,
        )
        dataset_cache.setdefault(seq_info.dataset, _get_dataset_sequences(seq_info.dataset))
        seq_obj = dataset_cache[seq_info.dataset][seq_info.name]
        sequence_results.append(_process_sequence(
            runner=runner,
            seq_info=seq_info,
            seq_obj=seq_obj,
            data=data,
            fc_idx=fc_idx,
            max_frames=args.max_frames,
            include_frame_records=args.include_frame_records,
        ))

    all_records: list[dict] = []
    by_dataset_records: dict[str, list[dict]] = collections.defaultdict(list)
    for seq_result in sequence_results:
        records = seq_result.pop("_records", [])
        by_dataset_records[seq_result["dataset"]].extend(records)
        all_records.extend(records)

    result: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "npz": str(npz_path),
        "config_path": args.config_path,
        "splits": sorted(splits),
        "n_sequences": len(sequence_results),
        "sequences": sequence_results,
        "gate": {
            "candidate_oracle_top3_iou03_min": 0.50,
            "candidate_oracle_top5_iou03_min": 0.60,
            "decision_rule": (
                "GO if false_confirmed oracle_top3_recall_iou03 >= 0.50 "
                "on diagnostic; otherwise use external candidate generator."
            ),
        },
    }
    if all_records:
        result["overall_summary"] = _summarize_records(all_records)
        result["per_dataset_summary"] = {
            ds: _summarize_records(records)
            for ds, records in sorted(by_dataset_records.items())
        }

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--config-path", default="configs/prod/salt.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["diagnostic"],
        choices=["train", "val", "diagnostic"],
    )
    parser.add_argument("--seqs", nargs="*", default=None)
    parser.add_argument("--max-val-seqs", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--smoke-test", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-frame-records", action="store_true")
    args = parser.parse_args(argv)

    result = run(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[candidate_mining] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
