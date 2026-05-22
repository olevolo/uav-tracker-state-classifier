"""diagnose_candidate_features.py — Per-feature AUC diagnostic for candidate events.

Loads a V5 candidate events NPZ and computes per-feature AUROC separating
correct candidates (candidate_correct_iou03=1) from wrong ones (=0).

Gate for scorer v2.1 training: dist_from_last AUC > 0.70 on car7/truck1 events.

Usage::

    PYTHONPATH=src:saltr/src python saltr/src/salt_r/diagnose_candidate_features.py \\
        --events saltr/data/candidate_events_v5_labeled.npz \\
        --sequences car7 truck1 uav7 \\
        --output saltr/results/candidate_feature_diag_v5.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_GATE_SEQUENCES: dict[str, list[str]] = {
    "uav123":      ["car7", "truck1"],
    "dtb70":       ["Girl2", "StreetBasketball1", "Animal1"],
    "visdrone_sot": [],   # no established hard sequences yet; skip gate
}

_FEATURE_NAMES = [
    "bbox_x_norm",
    "bbox_y_norm",
    "bbox_w_norm",
    "bbox_h_norm",
    "detector_score",
    "score_map_score",
    "geometry_area_ratio",
    "frame_area_ratio",
    "cosine_sim",
    "dist_from_last",
]


def _compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via Mann-Whitney U (no sklearn dependency)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import mannwhitneyu
    u, _ = mannwhitneyu(pos, neg, alternative="greater")
    return float(u) / (len(pos) * len(neg))


def _build_features(ev: dict) -> np.ndarray:
    fw = float(ev.get("frame_w") or 1)
    fh = float(ev.get("frame_h") or 1)
    bb = ev.get("candidate_bbox", [0, 0, 0, 0])
    return np.array([
        float(bb[0]) / max(fw, 1),
        float(bb[1]) / max(fh, 1),
        float(bb[2]) / max(fw, 1),
        float(bb[3]) / max(fh, 1),
        float(ev.get("detector_score") or 0.0),
        float(ev.get("score_map_score") or 0.0),
        float(ev.get("geometry_area_ratio", 1.0)),
        float(ev.get("frame_area_ratio", 0.0)),
        float(ev.get("cosine_sim", 0.0)),
        float(ev.get("dist_from_last", 0.0)),
    ], dtype=np.float32)


def diagnose(events_path: str, sequences: list[str] | None, output_path: str | None,
             gate_sequences: list[str] | None = None) -> dict:
    data = np.load(events_path, allow_pickle=True)
    raw_events = data["events"]

    all_results: dict = {}

    def _run_subset(name: str, evs: list[dict]) -> dict:
        labeled = [e for e in evs if e.get("candidate_correct_iou03") is not None
                   and e.get("candidate_iou") is not None]
        if not labeled:
            return {"n": 0, "n_positive": 0, "feature_auc": {}}
        labels = np.array([int(e["candidate_correct_iou03"]) for e in labeled])
        feats = np.stack([_build_features(e) for e in labeled])
        n_pos = int(labels.sum())
        aucs = {}
        for i, name_f in enumerate(_FEATURE_NAMES):
            aucs[name_f] = _compute_auc(feats[:, i], labels)
        # Report best direction (AUC or 1-AUC, whichever is higher)
        best_aucs = {}
        for name_f in _FEATURE_NAMES:
            auc_val = aucs[name_f]
            best_aucs[name_f] = max(auc_val, 1.0 - auc_val) if not np.isnan(auc_val) else float("nan")
        return {"n": len(labeled), "n_positive": n_pos, "positive_rate": n_pos / max(len(labeled), 1),
                "feature_auc": best_aucs}

    # All accepted events
    all_accepted = [dict(e) for e in raw_events if dict(e).get("accepted", False)]
    all_results["all_accepted"] = _run_subset("all_accepted", all_accepted)

    # Per-sequence breakdown (filtered)
    seq_filter = set(sequences) if sequences else None
    per_seq: dict = {}
    by_seq: dict[str, list] = {}
    for e in raw_events:
        e = dict(e)
        if not e.get("accepted", False):
            continue
        sid = e.get("seq_id", "")
        if seq_filter and sid not in seq_filter:
            continue
        by_seq.setdefault(sid, []).append(e)
    for sid, evs in sorted(by_seq.items()):
        per_seq[sid] = _run_subset(sid, evs)
    all_results["per_sequence"] = per_seq

    # Gate check — use explicitly provided sequences or dataset default
    gate_feature = "dist_from_last"
    gate_threshold = 0.70
    gate_seqs = gate_sequences if gate_sequences is not None else ["car7", "truck1"]
    if not gate_seqs:
        all_results["gate"] = {"feature": gate_feature, "threshold": gate_threshold,
                               "sequences": [], "pass": True, "notes": ["no gate sequences defined for this dataset"]}
    else:
        gate_pass = True
        gate_notes = []
        for seq in gate_seqs:
            if seq in per_seq:
                auc = per_seq[seq].get("feature_auc", {}).get(gate_feature, float("nan"))
                if np.isnan(auc) or auc < gate_threshold:
                    gate_pass = False
                    gate_notes.append(f"{seq}: {gate_feature} AUC={auc:.3f} < {gate_threshold}")
            else:
                gate_notes.append(f"{seq}: not in dataset (run with --sequences)")
        all_results["gate"] = {
            "feature": gate_feature,
            "threshold": gate_threshold,
            "sequences": gate_seqs,
            "pass": gate_pass and len(gate_notes) == 0,
            "notes": gate_notes,
        }

    # Print summary
    print(f"\n=== Candidate Feature Diagnostic ===")
    print(f"Source: {events_path}")
    acc = all_results["all_accepted"]
    print(f"All accepted: n={acc['n']}  n_positive={acc['n_positive']}  "
          f"positive_rate={acc.get('positive_rate', 0):.3f}")
    print(f"\nPer-feature AUC (all accepted, best direction):")
    for fname, auc_val in sorted(acc["feature_auc"].items(), key=lambda x: -x[1] if not np.isnan(x[1]) else -1):
        bar = "█" * int(auc_val * 20) if not np.isnan(auc_val) else ""
        print(f"  {fname:30s}  {auc_val:.3f}  {bar}")
    print(f"\nPer-sequence (filtered):")
    for sid, res in per_seq.items():
        aucs = res.get("feature_auc", {})
        dist_auc = aucs.get("dist_from_last", float("nan"))
        print(f"  {sid:15s}  n={res['n']:4d}  pos={res['n_positive']:4d}  "
              f"dist_from_last_AUC={dist_auc:.3f}")
    gate = all_results["gate"]
    print(f"\nGate ({gate['feature']} AUC > {gate['threshold']} on {gate['sequences']}): "
          f"{'PASS' if gate['pass'] else 'FAIL'}")
    for note in gate["notes"]:
        print(f"  ! {note}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved → {output_path}")

    return all_results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", required=True, help="Path to candidate_events_v5_<dataset>.npz")
    ap.add_argument("--dataset", choices=list(_GATE_SEQUENCES), default=None,
                    help="Dataset name — sets default gate sequences. If omitted, gate uses car7/truck1.")
    ap.add_argument("--sequences", nargs="+", default=None,
                    help="Sequences to include in per-seq breakdown (default: all)")
    ap.add_argument("--gate-sequences", nargs="+", default=None,
                    help="Sequences to use for AUC gate check. Overrides --dataset default.")
    ap.add_argument("--output", default=None, help="Path to save JSON results")
    args = ap.parse_args()

    # Resolve gate sequences: explicit > dataset default > uav123 fallback
    gate_seqs = args.gate_sequences
    if gate_seqs is None and args.dataset is not None:
        gate_seqs = _GATE_SEQUENCES.get(args.dataset, [])

    diagnose(args.events, args.sequences, args.output, gate_sequences=gate_seqs)


if __name__ == "__main__":
    main()
