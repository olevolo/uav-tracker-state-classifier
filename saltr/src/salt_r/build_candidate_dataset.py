"""build_candidate_dataset.py — collect and label per-candidate reinit events.

Runs SALTRunner over all training sequences with the CandidateEventLogger enabled.
For each proposed reinit candidate (accepted or rejected by geometry guard), records:
    source, bbox, scores, geometry ratio, crop_sim (MobileNetV3), aspect_ratio_delta,
    size_delta_ratio
Then labels each event offline:
    candidate_iou   = IoU(candidate_bbox, gt_bbox[frame])
    future_iou_gain = mean(iou_trace[t+1:t+20]) - iou_trace[t]
    label_good_candidate = int(candidate_iou > 0.3 and future_iou_gain > 0)

Output: saltr/data/candidate_events_labeled.npz
    keys: 'events'  — object array of dicts, one per event
          'stats'   — summary statistics (positive_rate, n_events, etc.)

Offline gate (required before BUG-26(b)/(c) training):
    positive_rate (IoU > 0.3) >= 5% of reinit events

Usage:
    PYTHONPATH=src:saltr/src python saltr/src/salt_r/build_candidate_dataset.py \\
        --config     configs/prod/salt.yaml \\
        --oracle-npz saltr/data/salt_rd_v2_labels.npz \\
        --dataset    uav123 \\
        --split      diagnostic \\
        --output     saltr/data/candidate_events_labeled.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.models as _M
import torch.nn.functional as _F

# ---------------------------------------------------------------------------
# MobileNetV3 identity embedding — OFFLINE USE ONLY
# Must NOT be imported into controller.py or any runtime path.
# ---------------------------------------------------------------------------

try:
    _EMBED_MODEL = _M.mobilenet_v3_small(weights="IMAGENET1K_V1").features.eval()
    for _p in _EMBED_MODEL.parameters():
        _p.requires_grad_(False)
    _EMBED_AVAILABLE = True
except Exception as _e:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "MobileNetV3 weights unavailable (%s). crop_sim will be 0.0. "
        "To fix: manually place weights at "
        "~/.cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth "
        "(download from https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth)",
        _e,
    )
    _EMBED_MODEL = None
    _EMBED_AVAILABLE = False
_EMBED_MEAN = torch.tensor([.485, .456, .406])[:, None, None]
_EMBED_STD  = torch.tensor([.229, .224, .225])[:, None, None]


def _crop_embed(frame_bgr: "np.ndarray", bbox_xywh: tuple, size: int = 64):
    """Return flattened MobileNetV3 embedding for a crop, or None if unavailable."""
    if _EMBED_MODEL is None:
        return None
    x, y, w, h = (max(0, int(v)) for v in bbox_xywh)
    crop = frame_bgr[y:y+h, x:x+w]
    if crop.size == 0 or w < 4 or h < 4:
        return None
    crop = cv2.resize(crop, (size, size))
    t = torch.from_numpy(crop[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.
    t = (t - _EMBED_MEAN) / _EMBED_STD
    with torch.no_grad():
        return _EMBED_MODEL(t.unsqueeze(0)).flatten()


def _compute_crop_sim(frame_bgr: "np.ndarray", candidate_bbox: tuple, template_bbox: tuple) -> float:
    """Cosine similarity between MobileNetV3 crops of candidate and template bboxes."""
    f1 = _crop_embed(frame_bgr, template_bbox)
    f2 = _crop_embed(frame_bgr, candidate_bbox)
    if f1 is None or f2 is None:
        return 0.0
    return _F.cosine_similarity(f1, f2, dim=0).item()


def _iou(a: tuple, b: tuple) -> float:
    """IoU between two (x,y,w,h) tuples."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-6)


def _sequences_for_split(oracle_npz_path: str, split: str) -> list[str]:
    """Return sequence keys in the oracle NPZ for a given split."""
    data = np.load(oracle_npz_path, allow_pickle=True)
    split_key = f"splits/{split}"
    if split_key in data.files:
        return [str(s) for s in data[split_key]]
    # Fallback: return all feature keys
    return [k.replace("features/", "") for k in data.files if k.startswith("features/")]


def run(
    config_path: str,
    oracle_npz_path: str,
    dataset: str,
    split: str,
    output_path: str,
    max_seqs: int = 0,
    sequences: list[str] | None = None,
    max_frames: int = 0,
) -> dict[str, Any]:
    """Collect and label candidate events. Returns summary stats."""
    from uav_tracker.salt_runner import SALTRunner
    from salt_r.candidate_events import CandidateEventLogger

    oracle_data = np.load(oracle_npz_path, allow_pickle=True)
    runner = SALTRunner.from_config(config_path)
    if runner.candidate_logger is None:
        runner.candidate_logger = CandidateEventLogger(enabled=True)
    else:
        runner.candidate_logger.enabled = True

    # Load dataset — uses __iter__ protocol
    if dataset == "uav123":
        from uav_tracker.datasets.uav123 import UAV123Dataset
        ds = UAV123Dataset()
    elif dataset == "dtb70":
        from uav_tracker.datasets.dtb70 import DTB70Dataset
        ds = DTB70Dataset()
    elif dataset == "visdrone_sot":
        from uav_tracker.datasets.visdrone_sot import VisDroneSOTDataset
        ds = VisDroneSOTDataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Filter to sequences present in oracle NPZ (they have feature/label data)
    oracle_seq_names = {
        k.replace(f"features/{dataset}/", "")
        for k in oracle_data.files
        if k.startswith(f"features/{dataset}/")
    }

    all_events: list[dict] = []
    t0 = time.time()
    n_processed = 0

    for seq in ds:
        if max_seqs > 0 and n_processed >= max_seqs:
            break
        seq_name = seq.name
        if sequences and seq_name not in sequences:
            continue  # explicit sequence filter (smoke run)
        if oracle_seq_names and seq_name not in oracle_seq_names:
            continue  # skip sequences not in oracle (e.g. diagnostic hold-outs)

        iou_key = f"iou_trace/{dataset}/{seq_name}"
        iou_trace = oracle_data[iou_key] if iou_key in oracle_data.files else None

        # GT bboxes come directly from the sequence object
        gt_bboxes = seq.ground_truth  # list or array of (x,y,w,h)

        runner.candidate_logger.reset(seq_id=seq_name)
        try:
            frame_count = 0
            for _ in runner.run(seq):
                frame_count += 1
                if max_frames > 0 and frame_count >= max_frames:
                    break
        except Exception as exc:
            print(f"  [skip] {seq_name}: {exc}", file=sys.stderr)
            continue

        n_processed += 1

        # Build frame index map and rolling template bbox for crop_sim computation.
        # Iterate seq.frames a second time (offline — no runtime cost concern).
        # Rolling template: most recent frame where IoU(pred, GT) > 0.7 per iou_trace.
        frame_map: dict[int, np.ndarray] = {}
        rolling_template_bbox: tuple | None = None  # (x, y, w, h) from GT
        try:
            for fi, frame_bgr in enumerate(seq.frames):
                if max_frames > 0 and fi >= max_frames:
                    break
                frame_map[fi] = frame_bgr
                # Update rolling template from GT when tracker was on-target
                if iou_trace is not None and fi < len(iou_trace) and iou_trace[fi] > 0.7:
                    if fi < len(gt_bboxes):
                        _gt = gt_bboxes[fi]
                        if hasattr(_gt, 'x'):
                            rolling_template_bbox = (float(_gt.x), float(_gt.y), float(_gt.w), float(_gt.h))
                        else:
                            rolling_template_bbox = (float(_gt[0]), float(_gt[1]), float(_gt[2]), float(_gt[3]))
        except Exception as exc:
            print(f"  [warn] {seq_name}: frame re-iteration failed: {exc}", file=sys.stderr)
            frame_map = {}

        # Label collected events with GT IoU, future utility, and v2 features
        for ev in runner.candidate_logger.events():
            d = ev.to_dict()
            t = ev.frame_idx

            if t < len(gt_bboxes):
                gt = gt_bboxes[t]
                # GT may be a BBox dataclass or array-like
                if hasattr(gt, 'x'):
                    gt_tuple = (float(gt.x), float(gt.y), float(gt.w), float(gt.h))
                else:
                    gt_tuple = (float(gt[0]), float(gt[1]), float(gt[2]), float(gt[3]))
                d["candidate_iou"] = _iou(tuple(float(v) for v in ev.candidate_bbox), gt_tuple)
            else:
                d["candidate_iou"] = 0.0

            if iou_trace is not None and t < len(iou_trace):
                future = iou_trace[t + 1: t + 21]
                d["future_iou_gain"] = float(np.mean(future) - iou_trace[t]) if len(future) > 0 else 0.0
            else:
                d["future_iou_gain"] = 0.0

            d["label_good_candidate"] = int(
                d["candidate_iou"] > 0.3 and d["future_iou_gain"] > 0.0
            )
            d["candidate_correct_iou03"] = int(d["candidate_iou"] >= 0.30)
            d["candidate_correct_iou05"] = int(d["candidate_iou"] >= 0.50)

            # ----------------------------------------------------------------
            # v2 candidate features — computed in post-processing pass
            # ----------------------------------------------------------------
            cand_bb = ev.candidate_bbox  # [x, y, w, h]
            cand_w = float(cand_bb[2]) if len(cand_bb) > 2 else 0.0
            cand_h = float(cand_bb[3]) if len(cand_bb) > 3 else 0.0

            # aspect_ratio_delta
            if rolling_template_bbox is not None:
                tmpl_x, tmpl_y, tmpl_w, tmpl_h = rolling_template_bbox
            else:
                # Fall back to first GT bbox if no high-IoU frame seen yet
                _g0 = gt_bboxes[0] if len(gt_bboxes) > 0 else None
                if _g0 is not None:
                    if hasattr(_g0, 'x'):
                        tmpl_x, tmpl_y, tmpl_w, tmpl_h = float(_g0.x), float(_g0.y), float(_g0.w), float(_g0.h)
                    else:
                        tmpl_x, tmpl_y, tmpl_w, tmpl_h = float(_g0[0]), float(_g0[1]), float(_g0[2]), float(_g0[3])
                else:
                    tmpl_x, tmpl_y, tmpl_w, tmpl_h = 0.0, 0.0, 1.0, 1.0

            cand_ar = cand_w / max(cand_h, 1e-3)
            tmpl_ar = tmpl_w / max(tmpl_h, 1e-3)
            aspect_ratio_delta = abs(cand_ar - tmpl_ar)

            # size_delta_ratio
            cand_area = cand_w * cand_h
            tmpl_area = tmpl_w * tmpl_h
            size_delta_ratio = min(abs(cand_area - tmpl_area) / max(tmpl_area, 1e-3), 1.0)

            # crop_sim — uses frame pixels and rolling template bbox
            candidate_bbox_xywh = (float(cand_bb[0]), float(cand_bb[1]), cand_w, cand_h)
            template_bbox_xywh = (tmpl_x, tmpl_y, tmpl_w, tmpl_h)
            if t in frame_map:
                try:
                    crop_sim = _compute_crop_sim(frame_map[t], candidate_bbox_xywh, template_bbox_xywh)
                except Exception:
                    crop_sim = 0.0
            else:
                crop_sim = 0.0

            d["crop_sim"] = crop_sim
            d["aspect_ratio_delta"] = aspect_ratio_delta
            d["size_delta_ratio"] = size_delta_ratio

            all_events.append(d)

        print(f"  {seq_name}: {len(runner.candidate_logger.events())} events", flush=True)

    # Summary statistics
    n_total = len(all_events)
    n_accepted = sum(1 for e in all_events if e.get("accepted", False))
    n_good = sum(1 for e in all_events if e.get("label_good_candidate", 0))
    n_correct_iou03 = sum(1 for e in all_events if e.get("candidate_correct_iou03", 0))
    n_correct_iou05 = sum(1 for e in all_events if e.get("candidate_correct_iou05", 0))
    positive_rate = n_good / max(n_accepted, 1)
    correct_iou03_rate = n_correct_iou03 / max(n_accepted, 1)

    # Source-separated distance stats
    det_events = [e for e in all_events if e.get("source") == "detector"]
    sm_events = [e for e in all_events if e.get("source") != "detector"]
    det_dist_nonzero = sum(1 for e in det_events if e.get("dist_from_last", 0.0) > 0.0)
    det_dist_nonzero_rate = det_dist_nonzero / max(len(det_events), 1)
    sm_dist_nonzero = sum(1 for e in sm_events if e.get("dist_from_last", 0.0) > 0.0)
    sm_dist_nonzero_rate = sm_dist_nonzero / max(len(sm_events), 1)

    # Provenance gates
    seq_nonblank_rate = sum(1 for e in all_events if e.get("seq_id", "")) / max(n_total, 1)
    frame_dims_nonzero_rate = sum(
        1 for e in all_events if e.get("frame_w", 0) > 0 and e.get("frame_h", 0) > 0
    ) / max(n_total, 1)
    crop_sim_nonzero_rate = sum(
        1 for e in all_events if e.get("crop_sim", 0.0) != 0.0
    ) / max(n_total, 1)

    stats = {
        "n_events": n_total,
        "n_accepted": n_accepted,
        "n_good_candidate": n_good,
        "n_correct_iou03": n_correct_iou03,
        "n_correct_iou05": n_correct_iou05,
        "positive_rate_of_accepted": positive_rate,
        "correct_iou03_rate": correct_iou03_rate,
        "seq_nonblank_rate": seq_nonblank_rate,
        "frame_dims_nonzero_rate": frame_dims_nonzero_rate,
        "det_dist_nonzero_rate": det_dist_nonzero_rate,
        "sm_dist_nonzero_rate": sm_dist_nonzero_rate,
        "n_detector_events": len(det_events),
        "n_scoremap_events": len(sm_events),
        "crop_sim_nonzero_rate": crop_sim_nonzero_rate,
        "elapsed_s": time.time() - t0,
        "gate_pass": correct_iou03_rate >= 0.05 and seq_nonblank_rate == 1.0 and frame_dims_nonzero_rate == 1.0,
    }

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        events=np.array(all_events, dtype=object),
        stats=np.array([stats], dtype=object),
    )

    jsonl_path = output_path.with_suffix(".jsonl")
    with jsonl_path.open("w") as f:
        for ev in all_events:
            f.write(json.dumps(ev) + "\n")

    print(f"\nSaved {n_total} events → {output_path}", flush=True)
    print(f"  accepted={n_accepted}  correct_iou03={n_correct_iou03}  correct_iou05={n_correct_iou05}  legacy_good={n_good}", flush=True)
    print(f"  correct_iou03_rate={correct_iou03_rate:.3f}  (legacy positive_rate={positive_rate:.3f})", flush=True)
    print(f"  seq_nonblank={seq_nonblank_rate:.3f}  frame_dims={frame_dims_nonzero_rate:.3f}  det_dist_nonzero={det_dist_nonzero_rate:.3f}  sm_dist_nonzero={sm_dist_nonzero_rate:.3f}", flush=True)
    print(f"  crop_sim_nonzero={crop_sim_nonzero_rate:.3f}", flush=True)
    print(f"  gate ({'PASS' if stats['gate_pass'] else 'FAIL'}): need correct_iou03_rate >= 0.05 + seq_nonblank == 1.0 + frame_dims == 1.0", flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="SALTRunner config YAML. Defaults to configs/prod/saltrd_<dataset>.yaml.")
    ap.add_argument(
        "--oracle-npz", default=None,
        help="Per-dataset oracle NPZ. Defaults to saltr/results/reinit_oracle_<dataset>.npz. "
             "Fall back to saltr/results/reinit_oracle_dataset.npz if per-dataset file absent."
    )
    ap.add_argument("--dataset", default="uav123", choices=["uav123", "dtb70", "visdrone_sot"])
    ap.add_argument("--split", default="diagnostic")
    ap.add_argument(
        "--output", default=None,
        help="Output NPZ path. Defaults to saltr/data/candidate_events_v5_<dataset>.npz."
    )
    ap.add_argument("--max-seqs", type=int, default=0, help="0 = all sequences")
    ap.add_argument(
        "--sequences", nargs="+", default=None,
        help="Explicit sequence names to run (e.g. car7 truck1 uav7). Overrides --split filter."
    )
    ap.add_argument(
        "--max-frames", type=int, default=0,
        help="Max frames per sequence (0 = all). Use for fast smoke runs."
    )
    args = ap.parse_args()

    # Auto-resolve per-dataset oracle path
    oracle_npz = args.oracle_npz
    if oracle_npz is None:
        per_ds = Path(f"saltr/results/reinit_oracle_{args.dataset}.npz")
        combined = Path("saltr/results/reinit_oracle_dataset.npz")
        oracle_npz = str(per_ds) if per_ds.exists() else str(combined)
        print(f"[build_candidate_dataset] oracle: {oracle_npz}", flush=True)

    # Auto-resolve config path
    config = args.config
    if config is None:
        config = f"configs/prod/saltrd_{args.dataset}.yaml"
        print(f"[build_candidate_dataset] config: {config}", flush=True)

    # Auto-resolve output path
    output = args.output
    if output is None:
        output = f"saltr/data/candidate_events_v5_{args.dataset}.npz"
        print(f"[build_candidate_dataset] output: {output}", flush=True)

    stats = run(
        config_path=config,
        oracle_npz_path=oracle_npz,
        dataset=args.dataset,
        split=args.split,
        output_path=output,
        max_seqs=args.max_seqs,
        sequences=args.sequences,
        max_frames=args.max_frames,
    )
    sys.exit(0 if stats["gate_pass"] else 1)


if __name__ == "__main__":
    main()
