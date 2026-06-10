"""Standalone CSC eval — composite version.

Computes per-axis metrics (localisation + confidence) plus the
derived 4-class metrics (CORRECT_CONFIRMED / CORRECT_UNCERTAIN /
LOST_AWARE / FALSE_CONFIRMED) using only telemetry-derived features
(GT IoU is intentionally unavailable at inference time).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from csc_lib.csc.config import CSCTrainConfig
from csc_lib.csc.dataset import _group_by_sequence
from csc_lib.csc.features import FEATURE_DIM, _State, build_runtime_feature
from csc_lib.csc.labeling.label_schema import (
    AUX_FLAGS,
    DERIVED_NAMES,
    LOCALIZATION_NAMES,
    NUM_DERIVED_STATES,
    NUM_LOCALIZATION_STATES,
    ConfidenceState,
    DerivedState,
    LocalizationState,
    derive_state,
)
from csc_lib.csc.model import build_model
from csc_lib.eval.custom_metrics.scene_state_metrics import (
    average_detection_delay,
    confusion_matrix,
    early_warning_recall,
    failure_auprc,
    failure_auroc,
    false_alarms_per_1000,
    macro_f1,
    per_state_prf,
    states_to_failure,
)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_labels_dir(labels_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(Path(labels_dir).rglob("labels.jsonl")):
        rows.extend(_load_jsonl(path))
    return rows


def _causal_inference_for_sequence(
    rows: list[dict],
    *,
    model,
    feature_cfg,
    image_size: tuple[int, int],
    device: str,
) -> dict[str, np.ndarray]:
    window = deque(maxlen=feature_cfg.window_size)
    state = _State()
    loc_pred: list[int] = []
    conf_pred: list[int] = []
    derived_pred: list[int] = []
    risks: list[float] = []
    aux_pred_arr: list[np.ndarray] = []
    latencies: list[float] = []

    model.eval()
    for r in rows:
        pred = tuple(r["pred_bbox"]) if r.get("pred_bbox") else None
        feat = build_runtime_feature(
            confidence=r.get("confidence"),
            apce=r.get("apce"),
            psr=r.get("psr"),
            pred_bbox=pred,
            image_size=image_size,
            state=state,
        )
        np.clip(feat, -feature_cfg.clip_value, feature_cfg.clip_value, out=feat)
        window.append(feat)
        if len(window) < feature_cfg.window_size:
            pad = feature_cfg.window_size - len(window)
            arr = np.stack([window[0]] * pad + list(window), axis=0)
        else:
            arr = np.stack(list(window), axis=0)
        x = torch.from_numpy(arr).unsqueeze(0).to(device)
        import time

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.predict(x)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        l_idx = int(out["predicted_localization"][0, -1].cpu().item())
        c_idx = int(out["predicted_confidence"][0, -1].cpu().item())
        d = int(derive_state(LocalizationState(l_idx), ConfidenceState(c_idx)))
        risk = float(out["risk_score"][0, -1].cpu().item())
        loc_pred.append(l_idx)
        conf_pred.append(c_idx)
        derived_pred.append(d)
        risks.append(risk)
        aux_pred_arr.append(out["aux_probs"][0, -1].cpu().numpy())

    return {
        "loc_pred": np.asarray(loc_pred, dtype=np.int64),
        "conf_pred": np.asarray(conf_pred, dtype=np.int64),
        "derived_pred": np.asarray(derived_pred, dtype=np.int64),
        "risks": np.asarray(risks, dtype=np.float64),
        "aux_pred": np.stack(aux_pred_arr, axis=0) if aux_pred_arr else np.zeros((0, len(AUX_FLAGS))),
        "latencies_ms": np.asarray(latencies, dtype=np.float64),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone CSC evaluation (composite).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--image_size", default="1280x720")
    p.add_argument("--risk_threshold", type=float, default=0.5)
    p.add_argument("--early_warning_k", type=int, default=10)
    p.add_argument("--save_predictions", action="store_true")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("eval_csc")
    args = parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    img_w, img_h = (int(x) for x in args.image_size.lower().split("x"))

    blob = torch.load(args.checkpoint, map_location=args.device)
    cfg = CSCTrainConfig.from_dict(blob["config"])
    cfg.model.feature_dim = FEATURE_DIM
    model = build_model(cfg.model).to(args.device)
    model.load_state_dict(blob["state_dict"])
    log.info("loaded CSC checkpoint: %s", args.checkpoint)

    rows = _load_labels_dir(Path(args.labels_dir))
    if not rows:
        raise SystemExit(f"no labels.jsonl rows in {args.labels_dir}")
    groups = _group_by_sequence(rows)
    log.info("evaluating CSC on %d sequences (%d frames)", len(groups), len(rows))

    all_loc_p, all_loc_t = [], []
    all_conf_p, all_conf_t = [], []
    all_der_p, all_der_t = [], []
    all_risk, all_truef = [], []
    per_seq_csv = ["dataset,sequence,n_frames,loc_macro_f1,derived_macro_f1,failure_auroc,failure_auprc,early_warning@k,avg_detect_delay,fa_per_1000,mean_latency_ms"]
    pred_jsonl = open(out_root / "csc_predictions.jsonl", "w") if args.save_predictions else None

    try:
        for (dataset_name, sequence_name), seq_rows in sorted(groups.items()):
            result = _causal_inference_for_sequence(
                seq_rows,
                model=model,
                feature_cfg=cfg.feature,
                image_size=(img_w, img_h),
                device=args.device,
            )
            loc_t = np.array([r.get("localization_state", 0) for r in seq_rows], dtype=np.int64)
            conf_t = np.array([r.get("confidence_state", 0) for r in seq_rows], dtype=np.int64)
            der_t = np.array([r.get("derived_state", 0) for r in seq_rows], dtype=np.int64)
            truef = states_to_failure(der_t)
            risk = result["risks"]

            all_loc_p.append(result["loc_pred"])
            all_loc_t.append(loc_t)
            all_conf_p.append(result["conf_pred"])
            all_conf_t.append(conf_t)
            all_der_p.append(result["derived_pred"])
            all_der_t.append(der_t)
            all_risk.append(risk)
            all_truef.append(truef)

            loc_f1 = macro_f1(loc_t, result["loc_pred"], n_states=NUM_LOCALIZATION_STATES, state_names=LOCALIZATION_NAMES)
            der_f1 = macro_f1(der_t, result["derived_pred"], n_states=NUM_DERIVED_STATES, state_names=DERIVED_NAMES)
            f_auroc = failure_auroc(truef, risk)
            f_auprc = failure_auprc(truef, risk)
            ewr = early_warning_recall(truef, risk, args.risk_threshold, k=args.early_warning_k)
            add = average_detection_delay(truef, risk, args.risk_threshold)
            fap = false_alarms_per_1000(truef, risk, args.risk_threshold)
            mean_lat = float(np.mean(result["latencies_ms"])) if result["latencies_ms"].size else 0.0

            per_seq_csv.append(",".join(str(x) for x in (
                dataset_name, sequence_name, len(seq_rows),
                f"{loc_f1:.4f}",
                f"{der_f1:.4f}",
                f"{f_auroc:.4f}",
                f"{f_auprc:.4f}",
                f"{ewr:.4f}",
                f"{add:.2f}" if not np.isnan(add) else "nan",
                f"{fap:.2f}",
                f"{mean_lat:.3f}",
            )))

            if pred_jsonl is not None:
                for t, r in enumerate(seq_rows):
                    pred_jsonl.write(json.dumps({
                        "dataset": dataset_name,
                        "sequence": sequence_name,
                        "frame_idx": int(r.get("frame_idx", t)),
                        "true_localization": int(loc_t[t]),
                        "pred_localization": int(result["loc_pred"][t]),
                        "true_confidence": int(conf_t[t]),
                        "pred_confidence": int(result["conf_pred"][t]),
                        "true_derived": int(der_t[t]),
                        "pred_derived": int(result["derived_pred"][t]),
                        "risk": float(risk[t]),
                        "aux_probs": {name: float(result["aux_pred"][t, j]) for j, name in enumerate(AUX_FLAGS)},
                    }) + "\n")
    finally:
        if pred_jsonl is not None:
            pred_jsonl.close()

    loc_p = np.concatenate(all_loc_p) if all_loc_p else np.zeros(0, dtype=np.int64)
    loc_t = np.concatenate(all_loc_t) if all_loc_t else np.zeros(0, dtype=np.int64)
    conf_p = np.concatenate(all_conf_p) if all_conf_p else np.zeros(0, dtype=np.int64)
    conf_t = np.concatenate(all_conf_t) if all_conf_t else np.zeros(0, dtype=np.int64)
    der_p = np.concatenate(all_der_p) if all_der_p else np.zeros(0, dtype=np.int64)
    der_t = np.concatenate(all_der_t) if all_der_t else np.zeros(0, dtype=np.int64)
    risk = np.concatenate(all_risk) if all_risk else np.zeros(0)
    truef = np.concatenate(all_truef) if all_truef else np.zeros(0, dtype=np.int8)

    summary = {
        "n_sequences": len(groups),
        "n_frames": int(loc_t.size),
        "localization_macro_f1": macro_f1(loc_t, loc_p, n_states=NUM_LOCALIZATION_STATES, state_names=LOCALIZATION_NAMES),
        "derived_macro_f1": macro_f1(der_t, der_p, n_states=NUM_DERIVED_STATES, state_names=DERIVED_NAMES),
        "false_confirmed_recall": float(per_state_prf(der_t, der_p, n_states=NUM_DERIVED_STATES, state_names=DERIVED_NAMES).get("FALSE_CONFIRMED", {}).get("recall", 0.0)),
        "failure_auroc": failure_auroc(truef, risk),
        "failure_auprc": failure_auprc(truef, risk),
        "early_warning_recall_k": float(early_warning_recall(truef, risk, args.risk_threshold, k=args.early_warning_k)),
        "early_warning_k": args.early_warning_k,
        "avg_detection_delay_frames": (
            float(average_detection_delay(truef, risk, args.risk_threshold))
            if not np.isnan(average_detection_delay(truef, risk, args.risk_threshold))
            else None
        ),
        "false_alarms_per_1000": float(false_alarms_per_1000(truef, risk, args.risk_threshold)),
        "risk_threshold": args.risk_threshold,
        "support_per_localization": {LOCALIZATION_NAMES[i]: int((loc_t == i).sum()) for i in range(NUM_LOCALIZATION_STATES)},
        "support_per_derived": {DERIVED_NAMES[i]: int((der_t == i).sum()) for i in range(NUM_DERIVED_STATES)},
    }
    loc_prf = per_state_prf(loc_t, loc_p, n_states=NUM_LOCALIZATION_STATES, state_names=LOCALIZATION_NAMES)
    der_prf = per_state_prf(der_t, der_p, n_states=NUM_DERIVED_STATES, state_names=DERIVED_NAMES)
    (out_root / "csc_metrics_summary.json").write_text(json.dumps(
        {**summary, "localization_per_state": loc_prf, "derived_per_state": der_prf}, indent=2,
    ))

    # Per-state CSV
    lines = ["axis,state,precision,recall,f1,support"]
    for name, v in loc_prf.items():
        lines.append(f"localization,{name},{v['precision']:.4f},{v['recall']:.4f},{v['f1']:.4f},{v['support']}")
    for name, v in der_prf.items():
        lines.append(f"derived,{name},{v['precision']:.4f},{v['recall']:.4f},{v['f1']:.4f},{v['support']}")
    (out_root / "csc_per_state.csv").write_text("\n".join(lines) + "\n")

    # Confusion matrices
    cm_loc = confusion_matrix(loc_t, loc_p, n_states=NUM_LOCALIZATION_STATES)
    cm_loc_lines = ["," + ",".join(LOCALIZATION_NAMES)]
    for i, name in enumerate(LOCALIZATION_NAMES):
        cm_loc_lines.append(name + "," + ",".join(str(int(v)) for v in cm_loc[i]))
    (out_root / "csc_confusion_localization.csv").write_text("\n".join(cm_loc_lines) + "\n")

    cm_der = confusion_matrix(der_t, der_p, n_states=NUM_DERIVED_STATES)
    cm_der_lines = ["," + ",".join(DERIVED_NAMES)]
    for i, name in enumerate(DERIVED_NAMES):
        cm_der_lines.append(name + "," + ",".join(str(int(v)) for v in cm_der[i]))
    (out_root / "csc_confusion_derived.csv").write_text("\n".join(cm_der_lines) + "\n")

    (out_root / "csc_per_sequence.csv").write_text("\n".join(per_seq_csv) + "\n")

    log.info(
        "CSC overall | locF1=%.3f derivedF1=%.3f | failure AUROC=%.3f AUPRC=%.3f | EW@%d=%.3f | ADD=%s | FA/1000=%.1f",
        summary["localization_macro_f1"], summary["derived_macro_f1"],
        summary["failure_auroc"], summary["failure_auprc"],
        args.early_warning_k, summary["early_warning_recall_k"],
        f"{summary['avg_detection_delay_frames']:.2f}" if summary["avg_detection_delay_frames"] is not None else "n/a",
        summary["false_alarms_per_1000"],
    )

    gates = {
        "loc_macro_f1>=0.55": summary["localization_macro_f1"] >= 0.55,
        "derived_macro_f1>=0.55": summary["derived_macro_f1"] >= 0.55,
        "failure_auroc>=0.80": summary["failure_auroc"] >= 0.80,
        "failure_auprc>=0.50": summary["failure_auprc"] >= 0.50,
        "false_alarms_per_1000<=30": summary["false_alarms_per_1000"] <= 30,
    }
    if "LOST" in loc_prf and loc_prf["LOST"]["support"] > 0:
        gates["LOST_recall>=0.75"] = loc_prf["LOST"]["recall"] >= 0.75
    if "UNCERTAIN" in loc_prf and loc_prf["UNCERTAIN"]["support"] > 0:
        gates["UNCERTAIN_recall>=0.60"] = loc_prf["UNCERTAIN"]["recall"] >= 0.60
    (out_root / "stage1_gate_check.json").write_text(json.dumps(gates, indent=2))
    log.info("Stage-1 gates: %s", gates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
