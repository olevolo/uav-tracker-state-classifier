"""Level-4 synthetic stress test for trained CSC checkpoints (Run 2 / V2 features).

Verifies that a trained CSC model behaves correctly on three crafted scenarios:

  Scenario 1 — Approaching object (NATURAL).
      Pick a CC sequence whose GT bbox area grows by 2x or more across the
      sequence (continuous CORRECT_CONFIRMED).  CSC should NOT cry FC.
      Pass criterion: max(p_FC) < 0.20 over the entire sequence.

  Scenario 2 — Edge crawl (NATURAL).
      Pick a CC sequence where the GT bbox stays close to a frame edge for
      a long stretch (>=50 frames, min normalised distance < 0.05) but the
      tracker remains correct (IoU >= 0.5).  CSC should NOT cry FC.
      Pass criterion: mean(p_FC) over the edge-crawl portion < 0.30.

  Scenario 3 — Real FC injection (ADVERSARIAL).
      Take any CC sequence; at frame T..T+30 splice in a fake FC by
      replacing pred_bbox with a random non-overlapping crop and forcing
      confidence to >=0.7.  CSC SHOULD detect FC after a small startup
      delay.  Pass criterion: max(p_FC) over T+5..T+30 > 0.70.

Inputs
------
- ``--ckpt``      : trained checkpoint (default lookup path is the Run 2
                    Stage-2 V2 model).
- ``--labels-dir``: root of weak-labels dir (default: V3-fix combined).
                    Must contain ``<dataset>/labels.jsonl`` files.

Outputs
-------
- Per-scenario verdict (PASS/FAIL) printed to stdout.
- JSON report at ``--out`` with per-scenario metrics.
- Exit code 0 if all 3 pass, 1 if any fail, 2 on environment error
  (missing checkpoint / no suitable sequence).

This is OFFLINE inference — uses the V2 feature builder
``csc_lib.csc.features.build_sequence_features_v2`` and runs the model
in a single forward pass over the full sequence (causal, since the
underlying TCN/GRU only attends to past frames).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csc_lib.csc.config import CSCFeatureConfig, CSCTrainConfig  # noqa: E402
from csc_lib.csc.features import (  # noqa: E402
    FEATURE_DIM_V2,
    build_sequence_features_v2,
)
from csc_lib.csc.labeling.label_schema import DerivedState  # noqa: E402
from csc_lib.csc.model import build_model  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_SIZE = (1280, 720)

# Pass thresholds — scenario specs from the user.
S1_FC_MAX_THRESHOLD = 0.20      # max(p_FC) must be below this
S2_FC_MEAN_THRESHOLD = 0.30     # mean(p_FC) over edge portion must be below this
S3_FC_MAX_THRESHOLD = 0.70      # max(p_FC) over injected window must be above this

# Scenario 1 — area growth ratio
S1_MIN_AREA_GROWTH = 2.0

# Scenario 2 — edge crawl
S2_EDGE_DIST_THRESHOLD = 0.05    # normalised distance to nearest edge
S2_MIN_EDGE_FRAMES = 50          # sustained edge frames required
S2_MIN_IOU = 0.5

# Scenario 3 — FC injection
S3_INJECT_LEN = 30               # number of frames to inject
S3_INJECT_CONF = 0.85            # forced confidence in injected region
S3_INJECT_DELAY = 5              # ignore the first 5 frames after injection

logger = logging.getLogger("synthetic_stress_test")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Level-4 synthetic stress test for CSC checkpoints",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to trained CSC checkpoint (.pth) — Run 2 Stage-2 V2 expected.",
    )
    p.add_argument(
        "--labels-dir",
        type=Path,
        default=ROOT / "outputs" / "csc_labels" / "sglatrack" / "v3fix_combined",
        help="Root weak-labels directory (default: outputs/csc_labels/sglatrack/v3fix_combined).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs" / "v3fix_diag" / "synthetic_stress.json",
        help="Path for JSON report (default: outputs/v3fix_diag/synthetic_stress.json).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the FC injection in Scenario 3 (default: 42).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu).",
    )
    p.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=list(DEFAULT_IMAGE_SIZE),
        metavar=("W", "H"),
        help="Frame width/height when not present in labels (default: 1280 720).",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Label IO
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _group_by_sequence(
    rows: list[dict],
) -> dict[tuple[str, str], list[dict]]:
    """Group rows by ``(dataset, sequence)`` and sort each group by frame_idx."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = (r.get("dataset", "?"), r.get("sequence", "?"))
        groups[key].append(r)
    for k in groups:
        groups[k].sort(key=lambda r: int(r.get("frame_idx", 0)))
    return groups


def _discover_label_files(labels_dir: Path) -> list[Path]:
    return sorted(labels_dir.rglob("labels.jsonl"))


def _resolve_image_size(rows: list[dict], default: tuple[int, int]) -> tuple[int, int]:
    for r in rows:
        sz = r.get("image_size")
        if isinstance(sz, (list, tuple)) and len(sz) == 2:
            return int(sz[0]), int(sz[1])
    return default


def _is_correct_confirmed_sequence(rows: list[dict]) -> bool:
    """A sequence is CC-dominant if >= 90% of frames are CORRECT_CONFIRMED.

    Empty / very short sequences are excluded.
    """
    if len(rows) < 30:
        return False
    cc = int(DerivedState.CORRECT_CONFIRMED)
    n_cc = sum(1 for r in rows if int(r.get("derived_state", -1)) == cc)
    return (n_cc / len(rows)) >= 0.90


# ---------------------------------------------------------------------------
# Sequence selection helpers
# ---------------------------------------------------------------------------


def _bbox_area(bbox: Optional[list[float] | tuple[float, ...]]) -> float:
    if bbox is None:
        return 0.0
    if len(bbox) < 4:
        return 0.0
    w = float(bbox[2])
    h = float(bbox[3])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h


def _min_edge_distance_normalised(
    bbox: Optional[list[float] | tuple[float, ...]],
    image_size: tuple[int, int],
) -> Optional[float]:
    """Min normalised distance of bbox to any frame edge."""
    if bbox is None or len(bbox) < 4:
        return None
    img_w, img_h = float(image_size[0]), float(image_size[1])
    if img_w <= 0 or img_h <= 0:
        return None
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    x1_n = x / img_w
    y1_n = y / img_h
    x2_n = (x + w) / img_w
    y2_n = (y + h) / img_h
    return float(min(x1_n, y1_n, 1.0 - x2_n, 1.0 - y2_n))


def find_approaching_sequence(
    seq_rows: dict[tuple[str, str], list[dict]],
    image_size: tuple[int, int],
    *,
    prefer_dataset: str = "dtb70",
) -> Optional[tuple[tuple[str, str], list[dict]]]:
    """Find a CC sequence with GT bbox area growth >= S1_MIN_AREA_GROWTH.

    Prefers sequences in *prefer_dataset* (DTB70 by default).
    Returns ``(key, rows)`` or ``None``.
    """
    candidates: list[tuple[float, tuple[str, str], list[dict]]] = []
    for key, rows in seq_rows.items():
        dataset = key[0]
        if not _is_correct_confirmed_sequence(rows):
            continue
        # Compute GT bbox area at start vs end.  Use median over first/last
        # 10 frames to be robust to single-frame jitter.
        n = len(rows)
        if n < 50:
            continue
        head = [r.get("gt_bbox") for r in rows[:10]]
        tail = [r.get("gt_bbox") for r in rows[-10:]]
        head_areas = [a for a in (_bbox_area(b) for b in head) if a > 0]
        tail_areas = [a for a in (_bbox_area(b) for b in tail) if a > 0]
        if not head_areas or not tail_areas:
            continue
        a0 = float(np.median(head_areas))
        a1 = float(np.median(tail_areas))
        if a0 <= 0:
            continue
        ratio = a1 / a0
        if ratio < S1_MIN_AREA_GROWTH:
            continue
        candidates.append((ratio, key, rows))

    if not candidates:
        return None

    # Sort: prefer dataset == prefer_dataset, then by area ratio descending.
    def _rank(c):
        ratio, key, _rows = c
        ds_pref = 0 if key[0] == prefer_dataset else 1
        return (ds_pref, -ratio)

    candidates.sort(key=_rank)
    _, key, rows = candidates[0]
    return key, rows


def find_edge_crawl_sequence(
    seq_rows: dict[tuple[str, str], list[dict]],
    image_size: tuple[int, int],
    *,
    prefer_dataset: str = "dtb70",
) -> Optional[tuple[tuple[str, str], list[dict], int, int]]:
    """Find a CC sequence with sustained edge contact + IoU >= 0.5 segment.

    Returns ``(key, rows, edge_start, edge_end)`` or ``None``.
    The slice ``rows[edge_start:edge_end]`` is the qualifying edge run.
    """
    candidates: list[tuple[int, tuple[str, str], list[dict], int, int]] = []
    for key, rows in seq_rows.items():
        if not _is_correct_confirmed_sequence(rows):
            continue
        # Build per-frame mask: edge AND IoU>=0.5.
        mask = []
        for r in rows:
            gt = r.get("gt_bbox")
            iou = r.get("iou")
            d = _min_edge_distance_normalised(gt, image_size)
            ok = (
                d is not None
                and d < S2_EDGE_DIST_THRESHOLD
                and iou is not None
                and float(iou) >= S2_MIN_IOU
            )
            mask.append(ok)

        # Find longest contiguous True run.
        best_len = 0
        best_start = -1
        best_end = -1
        cur_start = -1
        for i, ok in enumerate(mask):
            if ok and cur_start < 0:
                cur_start = i
            if (not ok or i == len(mask) - 1) and cur_start >= 0:
                cur_end = i if not ok else i + 1
                run_len = cur_end - cur_start
                if run_len > best_len:
                    best_len = run_len
                    best_start = cur_start
                    best_end = cur_end
                cur_start = -1

        if best_len >= S2_MIN_EDGE_FRAMES:
            candidates.append((best_len, key, rows, best_start, best_end))

    if not candidates:
        return None

    def _rank(c):
        run_len, key, _rows, _s, _e = c
        ds_pref = 0 if key[0] == prefer_dataset else 1
        return (ds_pref, -run_len)

    candidates.sort(key=_rank)
    _, key, rows, s, e = candidates[0]
    return key, rows, s, e


def find_any_cc_sequence(
    seq_rows: dict[tuple[str, str], list[dict]],
    *,
    prefer_dataset: str = "dtb70",
    min_len: int = 200,
) -> Optional[tuple[tuple[str, str], list[dict]]]:
    """Find any CC-dominant sequence with at least *min_len* frames."""
    candidates: list[tuple[int, tuple[str, str], list[dict]]] = []
    for key, rows in seq_rows.items():
        if not _is_correct_confirmed_sequence(rows):
            continue
        if len(rows) < min_len:
            continue
        candidates.append((len(rows), key, rows))

    if not candidates:
        return None

    def _rank(c):
        n, key, _rows = c
        ds_pref = 0 if key[0] == prefer_dataset else 1
        return (ds_pref, -n)

    candidates.sort(key=_rank)
    _, key, rows = candidates[0]
    return key, rows


# ---------------------------------------------------------------------------
# Scenario 3 — FC injection
# ---------------------------------------------------------------------------


def _bbox_iou_xywh(a: tuple[float, float, float, float],
                   b: tuple[float, float, float, float]) -> float:
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2]) * max(0.0, a[3])
    area_b = max(0.0, b[2]) * max(0.0, b[3])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _random_non_overlapping_bbox(
    ref_bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    rng: random.Random,
    *,
    iou_max: float = 0.0,
    max_tries: int = 200,
) -> tuple[float, float, float, float]:
    """Sample a random bbox of similar size that does not overlap *ref_bbox*."""
    img_w, img_h = float(image_size[0]), float(image_size[1])
    w = max(1.0, float(ref_bbox[2]))
    h = max(1.0, float(ref_bbox[3]))
    # If the bbox is bigger than the frame, shrink it.
    w = min(w, img_w * 0.9)
    h = min(h, img_h * 0.9)
    for _ in range(max_tries):
        x = rng.uniform(0.0, max(0.0, img_w - w))
        y = rng.uniform(0.0, max(0.0, img_h - h))
        cand = (x, y, w, h)
        if _bbox_iou_xywh(cand, ref_bbox) <= iou_max:
            return cand
    # Fallback — corner opposite to the ref bbox.
    cx = ref_bbox[0] + ref_bbox[2] / 2
    cy = ref_bbox[1] + ref_bbox[3] / 2
    if cx < img_w / 2:
        x = max(0.0, img_w - w)
    else:
        x = 0.0
    if cy < img_h / 2:
        y = max(0.0, img_h - h)
    else:
        y = 0.0
    return (x, y, w, h)


def inject_fc_segment(
    rows: list[dict],
    inject_start: int,
    image_size: tuple[int, int],
    rng: random.Random,
) -> list[dict]:
    """Return a deep-ish copy of *rows* with frames [t, t+S3_INJECT_LEN) overridden.

    Each injected frame:
      - pred_bbox replaced with a random non-overlapping crop of similar size,
      - confidence forced to S3_INJECT_CONF.

    The original rows are not mutated.
    """
    out: list[dict] = []
    end = min(len(rows), inject_start + S3_INJECT_LEN)
    for i, r in enumerate(rows):
        if inject_start <= i < end:
            # Reference bbox to avoid: prefer GT, fallback to pred.
            ref = r.get("gt_bbox") or r.get("pred_bbox")
            if ref is None or len(ref) < 4:
                # No reference — fall back to a random bbox in the frame.
                ref = (
                    image_size[0] / 4,
                    image_size[1] / 4,
                    image_size[0] / 4,
                    image_size[1] / 4,
                )
            new_bbox = _random_non_overlapping_bbox(
                tuple(ref), image_size, rng,
            )
            new_row = dict(r)
            new_row["pred_bbox"] = list(new_bbox)
            new_row["confidence"] = S3_INJECT_CONF
            out.append(new_row)
        else:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def load_csc_model(ckpt_path: Path, device: str):
    """Load model + feature config from checkpoint.

    Returns ``(model, feature_cfg, train_cfg)``.  The model is in eval mode
    on *device*.
    """
    import torch

    blob = torch.load(ckpt_path, map_location=device)
    cfg = CSCTrainConfig.from_dict(blob["config"])
    state_dict = blob["state_dict"]
    # Honour the actual feature_dim recorded in the checkpoint.
    proj_w = state_dict.get("proj.0.weight", state_dict.get("input_proj.weight"))
    if proj_w is not None:
        cfg.model.feature_dim = int(proj_w.shape[1])
    model = build_model(cfg.model)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, cfg.feature, cfg


def run_csc_on_sequence(
    model,
    feature_cfg: CSCFeatureConfig,
    rows: list[dict],
    image_size: tuple[int, int],
    device: str,
) -> np.ndarray:
    """Build V2 features for *rows* and return per-frame FC probability.

    Output is a (T,) numpy float32 array of P(FALSE_CONFIRMED) from the
    derived head softmax.
    """
    import torch

    feats = build_sequence_features_v2(rows, image_size, cfg=feature_cfg)
    if feats.shape[1] != FEATURE_DIM_V2:
        raise RuntimeError(
            f"V2 builder returned {feats.shape[1]} features but FEATURE_DIM_V2={FEATURE_DIM_V2}"
        )
    # Sanity check vs model input dim.
    expected_in = int(model.proj[0].weight.shape[1])
    if feats.shape[1] != expected_in:
        raise RuntimeError(
            f"feature dim mismatch: builder produced {feats.shape[1]}, "
            f"model expects {expected_in}. Is the checkpoint really trained with "
            "feature_version=v2?"
        )
    x = torch.from_numpy(feats).unsqueeze(0).to(device)  # (1, T, F)
    with torch.inference_mode():
        out = model.predict(x, last_step_only=False)
    der_probs = out["derived_probs"][0].cpu().numpy()  # (T, 4)
    fc_idx = int(DerivedState.FALSE_CONFIRMED)
    return der_probs[:, fc_idx].astype(np.float32)


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------


def run_scenario_1(
    model,
    feature_cfg: CSCFeatureConfig,
    seq_rows: dict[tuple[str, str], list[dict]],
    default_image_size: tuple[int, int],
    device: str,
) -> dict:
    pick = find_approaching_sequence(seq_rows, default_image_size)
    if pick is None:
        return {
            "name": "approaching_object_natural",
            "verdict": "FAIL",
            "error": (
                f"No CC sequence found with GT area growth >= {S1_MIN_AREA_GROWTH}x. "
                "Check that --labels-dir contains DTB70 labels."
            ),
            "pass_criterion": f"max(p_FC) < {S1_FC_MAX_THRESHOLD}",
        }
    key, rows = pick
    image_size = _resolve_image_size(rows, default_image_size)
    p_fc = run_csc_on_sequence(model, feature_cfg, rows, image_size, device)
    measured = float(p_fc.max()) if p_fc.size else 0.0
    verdict = "PASS" if measured < S1_FC_MAX_THRESHOLD else "FAIL"

    # Extra context: area growth ratio at the chosen sequence.
    head = [r.get("gt_bbox") for r in rows[:10]]
    tail = [r.get("gt_bbox") for r in rows[-10:]]
    head_areas = [a for a in (_bbox_area(b) for b in head) if a > 0]
    tail_areas = [a for a in (_bbox_area(b) for b in tail) if a > 0]
    growth = (
        float(np.median(tail_areas)) / float(np.median(head_areas))
        if head_areas and tail_areas else 0.0
    )

    return {
        "name": "approaching_object_natural",
        "dataset": key[0],
        "sequence": key[1],
        "frame_count": len(rows),
        "image_size": list(image_size),
        "area_growth_ratio": growth,
        "pass_criterion": f"max(p_FC) < {S1_FC_MAX_THRESHOLD}",
        "measured_max_p_fc": measured,
        "measured_mean_p_fc": float(p_fc.mean()) if p_fc.size else 0.0,
        "verdict": verdict,
    }


def run_scenario_2(
    model,
    feature_cfg: CSCFeatureConfig,
    seq_rows: dict[tuple[str, str], list[dict]],
    default_image_size: tuple[int, int],
    device: str,
) -> dict:
    pick = find_edge_crawl_sequence(seq_rows, default_image_size)
    if pick is None:
        return {
            "name": "edge_crawl_natural",
            "verdict": "FAIL",
            "error": (
                f"No CC sequence with sustained edge contact "
                f">= {S2_MIN_EDGE_FRAMES} frames + IoU >= {S2_MIN_IOU} found."
            ),
            "pass_criterion": f"mean(p_FC) over edge run < {S2_FC_MEAN_THRESHOLD}",
        }
    key, rows, edge_start, edge_end = pick
    image_size = _resolve_image_size(rows, default_image_size)
    p_fc = run_csc_on_sequence(model, feature_cfg, rows, image_size, device)
    edge_probs = p_fc[edge_start:edge_end]
    measured = float(edge_probs.mean()) if edge_probs.size else 0.0
    verdict = "PASS" if measured < S2_FC_MEAN_THRESHOLD else "FAIL"

    return {
        "name": "edge_crawl_natural",
        "dataset": key[0],
        "sequence": key[1],
        "frame_count": len(rows),
        "edge_start": int(edge_start),
        "edge_end": int(edge_end),
        "edge_length": int(edge_end - edge_start),
        "image_size": list(image_size),
        "pass_criterion": f"mean(p_FC) over edge run < {S2_FC_MEAN_THRESHOLD}",
        "measured_mean_p_fc_edge": measured,
        "measured_max_p_fc_edge": float(edge_probs.max()) if edge_probs.size else 0.0,
        "verdict": verdict,
    }


def run_scenario_3(
    model,
    feature_cfg: CSCFeatureConfig,
    seq_rows: dict[tuple[str, str], list[dict]],
    default_image_size: tuple[int, int],
    device: str,
    seed: int,
) -> dict:
    pick = find_any_cc_sequence(seq_rows)
    if pick is None:
        return {
            "name": "fc_injection_adversarial",
            "verdict": "FAIL",
            "error": "No CC-dominant sequence (>=200 frames) found in labels-dir.",
            "pass_criterion": f"max(p_FC) over injected window > {S3_FC_MAX_THRESHOLD}",
        }
    key, rows = pick
    image_size = _resolve_image_size(rows, default_image_size)

    # Inject at frame 100, or mid-sequence if the seq is too short.
    inject_start = 100 if len(rows) >= 100 + S3_INJECT_LEN else max(0, len(rows) // 2)
    inject_end = min(len(rows), inject_start + S3_INJECT_LEN)

    rng = random.Random(seed)
    injected_rows = inject_fc_segment(rows, inject_start, image_size, rng)
    p_fc = run_csc_on_sequence(model, feature_cfg, injected_rows, image_size, device)

    win_start = inject_start + S3_INJECT_DELAY
    win_end = inject_end
    if win_end <= win_start:
        win_start = inject_start
    window_probs = p_fc[win_start:win_end]
    measured = float(window_probs.max()) if window_probs.size else 0.0
    verdict = "PASS" if measured > S3_FC_MAX_THRESHOLD else "FAIL"

    return {
        "name": "fc_injection_adversarial",
        "dataset": key[0],
        "sequence": key[1],
        "frame_count": len(rows),
        "image_size": list(image_size),
        "inject_start": int(inject_start),
        "inject_end": int(inject_end),
        "inject_window_for_metric": [int(win_start), int(win_end)],
        "inject_confidence": S3_INJECT_CONF,
        "pass_criterion": f"max(p_FC) over [t+{S3_INJECT_DELAY}..t+{S3_INJECT_LEN}] > {S3_FC_MAX_THRESHOLD}",
        "measured_max_p_fc_injected": measured,
        "measured_mean_p_fc_injected": (
            float(window_probs.mean()) if window_probs.size else 0.0
        ),
        "baseline_max_p_fc_pre_inject": (
            float(p_fc[:inject_start].max()) if inject_start > 0 else 0.0
        ),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print_scenario(report: dict) -> None:
    name = report.get("name", "?")
    verdict = report.get("verdict", "?")
    crit = report.get("pass_criterion", "")
    print(f"\n[{verdict}] {name}")
    print(f"  criterion : {crit}")
    if "error" in report:
        print(f"  error     : {report['error']}")
        return
    print(f"  sequence  : {report.get('dataset')}/{report.get('sequence')} "
          f"(n={report.get('frame_count')})")
    if name.startswith("approaching"):
        print(f"  growth    : {report.get('area_growth_ratio'):.2f}x")
        print(f"  measured  : max(p_FC)={report.get('measured_max_p_fc'):.4f}, "
              f"mean(p_FC)={report.get('measured_mean_p_fc'):.4f}")
    elif name.startswith("edge"):
        print(f"  edge run  : [{report.get('edge_start')}..{report.get('edge_end')}] "
              f"({report.get('edge_length')} frames)")
        print(f"  measured  : mean(p_FC)={report.get('measured_mean_p_fc_edge'):.4f}, "
              f"max(p_FC)={report.get('measured_max_p_fc_edge'):.4f}")
    elif name.startswith("fc_injection"):
        print(f"  injected  : frames "
              f"[{report.get('inject_start')}..{report.get('inject_end')}]")
        print(f"  measured  : max(p_FC) on window={report.get('measured_max_p_fc_injected'):.4f}, "
              f"baseline pre-inject max={report.get('baseline_max_p_fc_pre_inject'):.4f}")


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. Validate checkpoint.
    if not args.ckpt.is_file():
        print(
            f"[error] checkpoint not found: {args.ckpt}\n"
            "Run training first or pass a valid --ckpt path.",
            file=sys.stderr,
        )
        return 2

    # 2. Validate labels dir + load.
    if not args.labels_dir.is_dir():
        print(
            f"[error] labels-dir not found: {args.labels_dir}\n"
            "Pass --labels-dir to a directory containing labels.jsonl files.",
            file=sys.stderr,
        )
        return 2

    label_files = _discover_label_files(args.labels_dir)
    if not label_files:
        print(
            f"[error] no labels.jsonl files under {args.labels_dir}",
            file=sys.stderr,
        )
        return 2

    logger.info("found %d labels.jsonl files under %s", len(label_files), args.labels_dir)
    all_rows: list[dict] = []
    for p in label_files:
        all_rows.extend(_load_jsonl(p))
    logger.info("loaded %d total label rows", len(all_rows))
    seq_rows = _group_by_sequence(all_rows)
    logger.info("grouped into %d sequences", len(seq_rows))

    # 3. Load model.
    logger.info("loading checkpoint %s", args.ckpt)
    try:
        model, feature_cfg, _train_cfg = load_csc_model(args.ckpt, args.device)
    except Exception as exc:
        print(f"[error] failed to load checkpoint {args.ckpt}: {exc}", file=sys.stderr)
        return 2
    logger.info(
        "model loaded: kind=%s feature_dim=%d hidden_dim=%d window_size=%d",
        getattr(_train_cfg.model, "kind", "?"),
        getattr(_train_cfg.model, "feature_dim", -1),
        getattr(_train_cfg.model, "hidden_dim", -1),
        getattr(feature_cfg, "window_size", -1),
    )

    default_image_size = (int(args.image_size[0]), int(args.image_size[1]))

    # 4. Run scenarios.
    report1 = run_scenario_1(model, feature_cfg, seq_rows, default_image_size, args.device)
    report2 = run_scenario_2(model, feature_cfg, seq_rows, default_image_size, args.device)
    report3 = run_scenario_3(
        model, feature_cfg, seq_rows, default_image_size, args.device, args.seed,
    )

    for r in (report1, report2, report3):
        _print_scenario(r)

    # 5. Aggregate report.
    overall = {
        "checkpoint": str(args.ckpt),
        "labels_dir": str(args.labels_dir),
        "seed": args.seed,
        "image_size_default": list(default_image_size),
        "feature_version": "v2",
        "feature_dim": int(FEATURE_DIM_V2),
        "scenarios": [report1, report2, report3],
    }
    n_pass = sum(1 for r in (report1, report2, report3) if r.get("verdict") == "PASS")
    n_total = 3
    overall["n_pass"] = n_pass
    overall["n_total"] = n_total
    overall["overall_verdict"] = "PASS" if n_pass == n_total else "FAIL"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(overall, fh, indent=2)
    print(f"\n[summary] {n_pass}/{n_total} scenarios passed -> {overall['overall_verdict']}")
    print(f"[summary] report written to {args.out}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
