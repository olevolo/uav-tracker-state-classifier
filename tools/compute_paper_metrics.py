"""Compute paper-specific metrics for the CSC UAV tracking paper.

Implements (per CLAUDE.md §Metrics):

  FCR   — False Confirmed Rate = N_false_confirmed / N_total
  FCD   — False Confirmed Duration = mean length of contiguous FC segments
  TTFC  — Time to False Confirmation = t_first_false_confirmed - t_last_confirmed
  UUR   — Unsafe Update Rate (proxy: fraction of frames where FC flag is set and
           template-update was NOT frozen; exact UUR requires tracker-side hook)
  Recovery@K (K=30 default) — per CLAUDE.md
  State-Conditioned AUC — AUC conditioned on each CSC derived state
  State Transition Matrix — N x N count matrix over derived states

Inputs
------
  --predictions_dir   : baseline predictions (<seq>.txt, one bbox per line)
  --telemetry_dir     : baseline telemetry (<seq>.jsonl)
  --states_dir        : CSC passive run states (<seq>.jsonl from run_with_csc.py)
  --labels_dir        : GT scene-state labels from build_scene_state_labels.py
                        (may be flat labels.jsonl or labels_per_sequence/<seq>.jsonl)
  --tracking_metrics_dir : evaluate_tracking_results.py output (summary.json)
  --confidence_calib  : JSON file with percentile calibration for confidence
  --output_dir        : where to write paper_metrics.csv, QUALITY_REPORT.md,
                        state_transition_matrix.csv, state_conditioned_auc.csv
  --recovery_k        : K for Recovery@K (default 30)

Outputs
-------
  paper_metrics.csv           — one row per sequence + aggregate row
  QUALITY_REPORT.md           — markdown table ready to paste into paper
  state_transition_matrix.csv — N x N CSV (counts), rows=from, cols=to
  state_conditioned_auc.csv   — AUC per state

Usage
-----
  python tools/compute_paper_metrics.py \\
      --tracker ortrack --dataset uav123 --split test \\
      --predictions_dir outputs/baselines/ortrack/uav123/test/predictions \\
      --telemetry_dir   outputs/baselines/ortrack/uav123/test/telemetry \\
      --states_dir      outputs/eval/ortrack/uav123/test/passive/states \\
      --labels_dir      outputs/eval/ortrack/uav123/test/labels \\
      --tracking_metrics_dir outputs/eval/ortrack/uav123/test/tracking_metrics \\
      --confidence_calib outputs/calibration/ortrack_lasot_confidence.json \\
      --output_dir      outputs/eval/ortrack/uav123/test/paper_metrics
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger("compute_paper_metrics")

# ---------------------------------------------------------------------------
# State name constants (aligned with csc_lib label_schema.py DerivedState)
# ---------------------------------------------------------------------------
STATE_NAMES = {
    0: "CORRECT_CONFIRMED",
    1: "CORRECT_UNCERTAIN",
    2: "LOST_AWARE",
    3: "FALSE_CONFIRMED",
}
N_STATES = 4
FALSE_CONFIRMED_IDX = 3
CORRECT_CONFIRMED_IDX = 0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_predictions(path: Path) -> np.ndarray:
    """Return (T, 4) float64 bbox array (xywh)."""
    rows: list[list[float]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                rows.append([0.0, 0.0, 0.0, 0.0])
                continue
            parts = [p for p in line.replace("\t", ",").split(",") if p]
            try:
                vals = [float(p) for p in parts[:4]]
            except ValueError:
                vals = [0.0, 0.0, 0.0, 0.0]
            rows.append((vals + [0.0] * 4)[:4])
    return np.asarray(rows, dtype=np.float64)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _load_confidence_calib(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _high_conf_threshold(calib: Optional[dict], default: float = 0.65) -> float:
    """Extract the q95 percentile confidence as the 'high confidence' threshold."""
    if calib is None:
        return default
    for key in ("q95", "p95", "quantile_95"):
        if key in calib:
            return float(calib[key])
    # Try nested structure: calib["confidence"]["q95"]
    nested = calib.get("confidence") or calib.get("features", {}).get("confidence")
    if isinstance(nested, dict):
        for key in ("q95", "p95", "quantile_95"):
            if key in nested:
                return float(nested[key])
    return default


def _collect_gt_labels(labels_dir: Path) -> dict[str, list[dict]]:
    """Load per-sequence GT labels from build_scene_state_labels output.

    Supports two layouts:
      - labels_dir/labels_per_sequence/<seq>.jsonl  (preferred)
      - labels_dir/labels.jsonl (flat, with 'sequence' key per row)
    Returns dict[seq_name] -> sorted list of row dicts.
    """
    per_seq_dir = labels_dir / "labels_per_sequence"
    groups: dict[str, list[dict]] = defaultdict(list)

    if per_seq_dir.exists():
        for p in sorted(per_seq_dir.glob("*.jsonl")):
            rows = _load_jsonl(p)
            seq = p.stem
            groups[seq].extend(rows)
    else:
        flat = labels_dir / "labels.jsonl"
        if flat.exists():
            for r in _load_jsonl(flat):
                seq = r.get("sequence", "unknown")
                groups[seq].append(r)
        else:
            # Fall back: any jsonl in labels_dir
            for p in sorted(labels_dir.glob("*.jsonl")):
                rows = _load_jsonl(p)
                if rows:
                    seq = rows[0].get("sequence", p.stem)
                    groups[seq].extend(rows)

    for seq in groups:
        groups[seq].sort(key=lambda r: r.get("frame_idx", 0))
    return dict(groups)


def _collect_csc_states(states_dir: Path) -> dict[str, list[dict]]:
    """Load per-sequence CSC prediction files from run_with_csc.py."""
    groups: dict[str, list[dict]] = {}
    for p in sorted(states_dir.glob("*.jsonl")):
        rows = _load_jsonl(p)
        if rows:
            rows.sort(key=lambda r: r.get("frame_idx", 0))
            groups[p.stem] = rows
    return groups


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def _iou_xywh(b1: np.ndarray, b2: np.ndarray) -> float:
    """IoU of two xywh boxes (single pair)."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = ix * iy
    union = w1 * h1 + w2 * h2 - inter
    return float(inter / union) if union > 0 else 0.0


def _iou_batch(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Vectorised IoU for aligned (T,4) arrays."""
    x1 = np.maximum(preds[:, 0], gts[:, 0])
    y1 = np.maximum(preds[:, 1], gts[:, 1])
    x2 = np.minimum(preds[:, 0] + preds[:, 2], gts[:, 0] + gts[:, 2])
    y2 = np.minimum(preds[:, 1] + preds[:, 3], gts[:, 1] + gts[:, 3])
    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)
    inter = iw * ih
    union = preds[:, 2] * preds[:, 3] + gts[:, 2] * gts[:, 3] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float64)


def compute_fcr(derived_states: np.ndarray) -> float:
    """FCR = N_false_confirmed / N_total (per CLAUDE.md §Metrics)."""
    n = len(derived_states)
    if n == 0:
        return 0.0
    return float((derived_states == FALSE_CONFIRMED_IDX).sum()) / n


def compute_fcd(derived_states: np.ndarray) -> float:
    """FCD = mean length of contiguous false_confirmed segments."""
    lengths: list[int] = []
    in_fc = False
    run = 0
    for s in derived_states:
        if s == FALSE_CONFIRMED_IDX:
            run += 1
            in_fc = True
        else:
            if in_fc:
                lengths.append(run)
                run = 0
                in_fc = False
    if in_fc:
        lengths.append(run)
    return float(np.mean(lengths)) if lengths else 0.0


def compute_ttfc(derived_states: np.ndarray) -> Optional[float]:
    """TTFC = t_first_false_confirmed - t_last_confirmed (frames).

    Returns None if there are no FC frames.
    Per CLAUDE.md definition; measures how quickly FC follows confirmed tracking.
    """
    fc_frames = np.where(derived_states == FALSE_CONFIRMED_IDX)[0]
    confirmed_frames = np.where(derived_states == CORRECT_CONFIRMED_IDX)[0]
    if len(fc_frames) == 0:
        return None
    t_fc = int(fc_frames[0])
    # Last confirmed frame strictly before the first FC frame
    conf_before = confirmed_frames[confirmed_frames < t_fc]
    if len(conf_before) == 0:
        return None
    t_conf = int(conf_before[-1])
    return float(t_fc - t_conf)


def compute_recovery_at_k(
    derived_states: np.ndarray,
    k: int = 30,
) -> float:
    """Recovery@K = recovered_FC_episodes_within_K_frames / total_FC_episodes.

    An FC episode is "recovered" if within K frames after the episode ends the
    state returns to CORRECT_CONFIRMED.
    Per CLAUDE.md §Metrics definition.
    """
    n = len(derived_states)
    # Extract contiguous FC episodes
    fc_episodes: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if derived_states[i] == FALSE_CONFIRMED_IDX:
            start = i
            while i < n and derived_states[i] == FALSE_CONFIRMED_IDX:
                i += 1
            fc_episodes.append((start, i - 1))
        else:
            i += 1

    if not fc_episodes:
        return 0.0

    recovered = 0
    for _, end in fc_episodes:
        lo = end + 1
        hi = min(n, end + 1 + k)
        if lo >= hi:
            continue
        window = derived_states[lo:hi]
        if np.any(window == CORRECT_CONFIRMED_IDX):
            recovered += 1
    return recovered / len(fc_episodes)


def compute_state_transition_matrix(
    derived_states: np.ndarray,
) -> np.ndarray:
    """N x N transition count matrix (rows=from, cols=to)."""
    mat = np.zeros((N_STATES, N_STATES), dtype=np.int64)
    for t in range(len(derived_states) - 1):
        frm = int(derived_states[t])
        to = int(derived_states[t + 1])
        if 0 <= frm < N_STATES and 0 <= to < N_STATES:
            mat[frm, to] += 1
    return mat


def compute_state_conditioned_auc(
    ious: np.ndarray,
    derived_states: np.ndarray,
    n_thresholds: int = 21,
) -> dict[str, float]:
    """AUC computed separately for frames in each derived state."""
    result: dict[str, float] = {}
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    for idx, name in STATE_NAMES.items():
        mask = derived_states == idx
        if mask.sum() == 0:
            result[name] = float("nan")
            continue
        seg = ious[mask]
        # Success curve: at each threshold t, fraction of frames with IoU >= t
        rates = np.array([(seg >= t).mean() for t in thresholds])
        auc = float(np.trapz(rates, thresholds))
        result[name] = round(auc, 6)
    return result


def compute_uur_proxy(states_rows: list[dict]) -> float:
    """Proxy for Unsafe Update Rate.

    UUR_proxy = frames where false_confirmed_flag=True AND
                should_skip_template_update=False divided by all frames.

    NOTE: True UUR requires knowing WHEN the tracker actually updated its
    template.  This proxy uses the CSC's own recommendation as a stand-in.
    It underestimates UUR when the tracker has no update hook.
    """
    n = 0
    unsafe = 0
    for row in states_rows:
        if row.get("init"):
            continue
        n += 1
        fc = bool(row.get("false_confirmed_flag", False))
        skip_update = bool(row.get("should_skip_template_update", False))
        if fc and not skip_update:
            unsafe += 1
    return unsafe / n if n > 0 else 0.0


# ---------------------------------------------------------------------------
# Sequence-level computation
# ---------------------------------------------------------------------------


def _extract_state_array_from_csc(
    csc_rows: list[dict],
    n_frames: int,
) -> np.ndarray:
    """Build per-frame derived_state array from CSC predictions.

    Fills missing frames with CORRECT_CONFIRMED (conservative default).
    """
    arr = np.full(n_frames, CORRECT_CONFIRMED_IDX, dtype=np.int64)
    for row in csc_rows:
        if row.get("init"):
            continue
        t = int(row.get("frame_idx", -1))
        if 0 <= t < n_frames:
            arr[t] = int(row.get("derived_state", CORRECT_CONFIRMED_IDX))
    return arr


def _extract_state_array_from_labels(
    label_rows: list[dict],
    n_frames: int,
) -> np.ndarray:
    """Build per-frame derived_state array from GT labels."""
    arr = np.full(n_frames, CORRECT_CONFIRMED_IDX, dtype=np.int64)
    for row in label_rows:
        t = int(row.get("frame_idx", -1))
        if 0 <= t < n_frames:
            arr[t] = int(row.get("derived_state", CORRECT_CONFIRMED_IDX))
    return arr


def compute_sequence_metrics(
    seq_name: str,
    preds: np.ndarray,
    gt_label_rows: list[dict],
    csc_rows: list[dict],
    calib_conf_threshold: float,
    recovery_k: int,
) -> dict:
    """Compute all paper metrics for one sequence."""
    n = len(preds)
    if n == 0:
        return {}

    # Build GT IoU array (requires GT bboxes from label rows)
    gt_bboxes = np.zeros((n, 4), dtype=np.float64)
    for row in gt_label_rows:
        t = int(row.get("frame_idx", -1))
        if 0 <= t < n:
            bb = row.get("gt_bbox") or row.get("bbox")
            if bb and len(bb) >= 4:
                gt_bboxes[t] = [float(v) for v in bb[:4]]

    ious = _iou_batch(preds, gt_bboxes)

    # Derived state from CSC predictions
    csc_states = _extract_state_array_from_csc(csc_rows, n)

    # Paper metrics
    fcr = compute_fcr(csc_states)
    fcd = compute_fcd(csc_states)
    ttfc = compute_ttfc(csc_states)
    recovery = compute_recovery_at_k(csc_states, k=recovery_k)
    uur_proxy = compute_uur_proxy(csc_rows)

    # State-conditioned AUC
    sc_auc = compute_state_conditioned_auc(ious, csc_states)

    # Transition matrix (returned separately for aggregation)
    trans_mat = compute_state_transition_matrix(csc_states)

    # Global AUC (for reference)
    thresholds = np.linspace(0.0, 1.0, 21)
    rates = np.array([(ious >= t).mean() for t in thresholds])
    global_auc = float(np.trapz(rates, thresholds))

    # N_fc frames and total for aggregate
    n_fc = int((csc_states == FALSE_CONFIRMED_IDX).sum())

    row: dict = {
        "sequence": seq_name,
        "n_frames": n,
        "n_fc_frames": n_fc,
        "fcr": round(fcr, 6),
        "fcd": round(fcd, 4),
        "ttfc": round(ttfc, 2) if ttfc is not None else None,
        f"recovery_at_{recovery_k}": round(recovery, 4),
        "uur_proxy": round(uur_proxy, 6),
        "auc_global": round(global_auc, 6),
    }
    for state_name, auc_val in sc_auc.items():
        row[f"auc_{state_name}"] = round(auc_val, 6) if not math.isnan(auc_val) else None

    return row, trans_mat


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute paper-specific CSC metrics.")
    p.add_argument("--tracker", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument(
        "--predictions_dir",
        required=True,
        help="Baseline predictions dir with <seq>.txt files.",
    )
    p.add_argument(
        "--telemetry_dir",
        default=None,
        help="Baseline telemetry dir (optional, for confidence).",
    )
    p.add_argument(
        "--states_dir",
        required=True,
        help="CSC passive run states dir (<seq>.jsonl from run_with_csc.py).",
    )
    p.add_argument(
        "--labels_dir",
        required=True,
        help="GT state labels from build_scene_state_labels.py.",
    )
    p.add_argument(
        "--tracking_metrics_dir",
        default=None,
        help="evaluate_tracking_results.py output dir (for including AUC in report).",
    )
    p.add_argument(
        "--confidence_calib",
        default=None,
        help="JSON calibration file for confidence thresholds.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Where to write outputs.",
    )
    p.add_argument("--recovery_k", type=int, default=30)
    p.add_argument("--max_sequences", type=int, default=None)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = Path(args.predictions_dir)
    states_dir = Path(args.states_dir)
    labels_dir = Path(args.labels_dir)

    if not pred_dir.exists():
        log.error("predictions_dir not found: %s", pred_dir)
        return 1
    if not states_dir.exists():
        log.error("states_dir not found: %s", states_dir)
        return 1

    # Load calibration
    calib = _load_confidence_calib(
        Path(args.confidence_calib) if args.confidence_calib else None
    )
    conf_threshold = _high_conf_threshold(calib)
    log.info("High-confidence threshold (q95): %.4f", conf_threshold)

    # Load GT labels
    log.info("Loading GT labels from %s", labels_dir)
    gt_label_map = _collect_gt_labels(labels_dir) if labels_dir.exists() else {}
    log.info("  %d sequences with GT labels", len(gt_label_map))

    # Load CSC states
    log.info("Loading CSC states from %s", states_dir)
    csc_states_map = _collect_csc_states(states_dir)
    log.info("  %d sequences with CSC states", len(csc_states_map))

    # Find prediction files
    pred_files = sorted(pred_dir.glob("*.txt"))
    if args.max_sequences:
        pred_files = pred_files[: args.max_sequences]
    log.info("Processing %d sequences", len(pred_files))

    # Accumulate per-sequence results
    seq_rows: list[dict] = []
    agg_trans = np.zeros((N_STATES, N_STATES), dtype=np.int64)
    total_n = 0
    total_fc = 0
    fcds: list[float] = []
    ttfcs: list[float] = []
    recovery_vals: list[float] = []
    uur_vals: list[float] = []
    sc_auc_accum: dict[str, list[float]] = defaultdict(list)

    for pred_path in pred_files:
        seq_name = pred_path.stem
        preds = _read_predictions(pred_path)
        n = len(preds)
        if n == 0:
            continue

        gt_rows = gt_label_map.get(seq_name, [])
        csc_rows = csc_states_map.get(seq_name, [])

        if not csc_rows:
            log.warning("No CSC states for sequence %s — skipping", seq_name)
            continue

        result = compute_sequence_metrics(
            seq_name=seq_name,
            preds=preds,
            gt_label_rows=gt_rows,
            csc_rows=csc_rows,
            calib_conf_threshold=conf_threshold,
            recovery_k=args.recovery_k,
        )
        if not result:
            continue
        row, trans_mat = result

        seq_rows.append(row)
        agg_trans += trans_mat
        total_n += row.get("n_frames", 0)
        total_fc += row.get("n_fc_frames", 0)
        if row["fcd"] > 0:
            fcds.append(row["fcd"])
        if row.get("ttfc") is not None:
            ttfcs.append(row["ttfc"])
        recovery_vals.append(row.get(f"recovery_at_{args.recovery_k}", 0.0))
        uur_vals.append(row.get("uur_proxy", 0.0))
        for state_name in STATE_NAMES.values():
            v = row.get(f"auc_{state_name}")
            if v is not None and not math.isnan(v):
                sc_auc_accum[state_name].append(v)

    if not seq_rows:
        log.error("No sequences processed.")
        return 1

    # ---------------------------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------------------------
    fcr_overall = total_fc / total_n if total_n > 0 else 0.0
    fcd_mean = float(np.mean(fcds)) if fcds else 0.0
    ttfc_mean = float(np.mean(ttfcs)) if ttfcs else None
    recovery_mean = float(np.mean(recovery_vals))
    uur_mean = float(np.mean(uur_vals))
    sc_auc_mean = {
        name: float(np.mean(vals)) if vals else float("nan")
        for name, vals in sc_auc_accum.items()
    }

    # Add aggregate row
    agg_row: dict = {
        "sequence": "__aggregate__",
        "n_frames": total_n,
        "n_fc_frames": total_fc,
        "fcr": round(fcr_overall, 6),
        "fcd": round(fcd_mean, 4),
        "ttfc": round(ttfc_mean, 2) if ttfc_mean is not None else None,
        f"recovery_at_{args.recovery_k}": round(recovery_mean, 4),
        "uur_proxy": round(uur_mean, 6),
        "auc_global": None,  # filled from tracking_metrics_dir if available
    }
    for state_name in STATE_NAMES.values():
        v = sc_auc_mean.get(state_name, float("nan"))
        agg_row[f"auc_{state_name}"] = round(v, 6) if not math.isnan(v) else None

    # Try loading global AUC from evaluate_tracking_results output
    global_auc: Optional[float] = None
    if args.tracking_metrics_dir:
        summary_path = Path(args.tracking_metrics_dir) / "summary.json"
        if summary_path.exists():
            try:
                with open(summary_path) as fh:
                    summary = json.load(fh)
                global_auc = summary.get("success_auc") or summary.get("auc")
                agg_row["auc_global"] = global_auc
            except Exception as exc:
                log.warning("Failed to load summary.json: %s", exc)

    seq_rows.append(agg_row)

    # ---------------------------------------------------------------------------
    # Write paper_metrics.csv
    # ---------------------------------------------------------------------------
    csv_path = out_dir / "paper_metrics.csv"
    fieldnames = list(seq_rows[0].keys()) if seq_rows else []
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in seq_rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    log.info("Wrote %s", csv_path)

    # ---------------------------------------------------------------------------
    # Write state transition matrix
    # ---------------------------------------------------------------------------
    trans_path = out_dir / "state_transition_matrix.csv"
    with open(trans_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["from \\ to"] + [STATE_NAMES[i] for i in range(N_STATES)]
        writer.writerow(header)
        for i in range(N_STATES):
            row_data = [STATE_NAMES[i]] + [int(agg_trans[i, j]) for j in range(N_STATES)]
            writer.writerow(row_data)
    log.info("Wrote %s", trans_path)

    # ---------------------------------------------------------------------------
    # Write state_conditioned_auc.csv
    # ---------------------------------------------------------------------------
    sc_auc_path = out_dir / "state_conditioned_auc.csv"
    with open(sc_auc_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["state", "mean_auc", "n_sequences"])
        for state_name in STATE_NAMES.values():
            vals = sc_auc_accum.get(state_name, [])
            mean_v = float(np.mean(vals)) if vals else float("nan")
            writer.writerow([state_name, f"{mean_v:.6f}" if not math.isnan(mean_v) else "nan", len(vals)])
    log.info("Wrote %s", sc_auc_path)

    # ---------------------------------------------------------------------------
    # Write QUALITY_REPORT.md
    # ---------------------------------------------------------------------------
    report_path = out_dir / "QUALITY_REPORT.md"
    n_seqs = len(seq_rows) - 1  # exclude aggregate row

    # State distribution
    state_counts: dict[str, int] = defaultdict(int)
    for row in seq_rows[:-1]:
        n = row.get("n_frames", 0)
        states_arr_approx = np.full(1, 0)  # placeholder
        # Best we can do here: use the aggregated n_fc to estimate
    state_dist_note = f"{total_fc}/{total_n} FC frames = {fcr_overall:.2%}"

    with open(report_path, "w") as fh:
        fh.write(f"# CSC Paper Metrics — {args.tracker.upper()} on UAV123\n\n")
        fh.write(f"**Tracker:** {args.tracker}  \n")
        fh.write(f"**Dataset:** {args.dataset} ({args.split})  \n")
        fh.write(f"**Sequences evaluated:** {n_seqs}  \n")
        fh.write(f"**Total frames:** {total_n}  \n")
        fh.write("\n---\n\n")
        fh.write("## Main Paper Table (aggregate over 123 sequences)\n\n")
        fh.write("| Metric | Value |\n")
        fh.write("|--------|-------|\n")
        fh.write(f"| AUC (global) | {global_auc:.4f} |\n" if global_auc else "| AUC (global) | — |\n")
        fh.write(f"| FCR (False Confirmed Rate) | {fcr_overall:.4f} |\n")
        fh.write(f"| FCD (mean duration, frames) | {fcd_mean:.2f} |\n")
        ttfc_str = f"{ttfc_mean:.2f}" if ttfc_mean is not None else "N/A"
        fh.write(f"| TTFC (mean, frames) | {ttfc_str} |\n")
        fh.write(f"| Recovery@{args.recovery_k} | {recovery_mean:.4f} |\n")
        fh.write(f"| UUR proxy | {uur_mean:.6f} |\n")
        fh.write("\n---\n\n")
        fh.write("## State-Conditioned AUC\n\n")
        fh.write("| State | Mean AUC | N sequences |\n")
        fh.write("|-------|----------|-------------|\n")
        for state_name in STATE_NAMES.values():
            vals = sc_auc_accum.get(state_name, [])
            mean_v = float(np.mean(vals)) if vals else float("nan")
            mean_str = f"{mean_v:.4f}" if not math.isnan(mean_v) else "N/A"
            fh.write(f"| {state_name} | {mean_str} | {len(vals)} |\n")
        fh.write("\n---\n\n")
        fh.write("## State Transition Matrix (counts)\n\n")
        header_cells = ["from \\ to"] + [STATE_NAMES[i] for i in range(N_STATES)]
        fh.write("| " + " | ".join(header_cells) + " |\n")
        fh.write("|" + "|".join(["---"] * len(header_cells)) + "|\n")
        for i in range(N_STATES):
            row_vals = [str(int(agg_trans[i, j])) for j in range(N_STATES)]
            fh.write(f"| {STATE_NAMES[i]} | " + " | ".join(row_vals) + " |\n")
        fh.write("\n---\n\n")
        fh.write("## Notes\n\n")
        fh.write(
            "- **FCR** = N_false_confirmed_frames / N_total_frames "
            "(using CSC derived_state from passive run)\n"
        )
        fh.write(
            "- **FCD** = mean length (frames) of contiguous FALSE_CONFIRMED segments\n"
        )
        fh.write(
            "- **TTFC** = mean (t_first_FC - t_last_CONFIRMED) "
            "per sequence where FC occurs\n"
        )
        fh.write(
            f"- **Recovery@{args.recovery_k}** = fraction of FC episodes where "
            f"state returns to CORRECT_CONFIRMED within {args.recovery_k} frames\n"
        )
        fh.write(
            "- **UUR proxy** = fraction of frames where CSC predicted FALSE_CONFIRMED "
            "but did NOT recommend skipping template update; "
            "true UUR requires tracker-side update hook\n"
        )
        fh.write(
            "- State-conditioned AUC: AUC computed only over frames in that state; "
            "IoU is computed vs GT bboxes from label rows\n"
        )
        fh.write(
            "- All results are from passive CSC inference — "
            "no control actions applied to the tracker\n"
        )
    log.info("Wrote %s", report_path)

    # Summary to stdout
    log.info(
        "=== AGGREGATE: tracker=%s, dataset=%s, n_seq=%d, n_frames=%d ===",
        args.tracker, args.dataset, n_seqs, total_n,
    )
    log.info("  FCR=%.4f  FCD=%.2f  TTFC=%s  Recovery@%d=%.4f  UUR_proxy=%.6f",
             fcr_overall, fcd_mean,
             f"{ttfc_mean:.2f}" if ttfc_mean is not None else "N/A",
             args.recovery_k, recovery_mean, uur_mean)
    log.info("  State-Conditioned AUC: %s",
             {k: f"{v:.4f}" if not math.isnan(v) else "N/A"
              for k, v in sc_auc_mean.items()})

    return 0


if __name__ == "__main__":
    sys.exit(main())
