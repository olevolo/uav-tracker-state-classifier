#!/usr/bin/env python
"""UAV123 OOD eval — FALSE_CONFIRMED behavior of frozen CSC diagnosis models.

WHAT / WHY
----------
UAV123 is the project's TRUE out-of-distribution FINAL TEST set (it is NEVER in
CSC training and thresholds are NEVER tuned on it). The payoff of the factorized /
conditional 2-tower challenger is whether its anti-shortcut structure REDUCES
false FALSE_CONFIRMED (FC) firings on this OOD set vs the V3-prod model — that is
"Gate 4" of the controlled rematch.

This script SCORES frozen models on UAV123 and reports FC false-positive behavior:

  * false-FC rate  = fraction of FC-PREDICTED frames whose GT is NOT FC
                     (= 1 - FC precision; the OOD safety number)
  * FC precision / recall / F1
  * FC firings per 1000 frames
  * per-sequence + per-dataset breakdown

at TWO operating points, both FROZEN (no UAV123 tuning):
  (1) 4-way argmax  (the model's own decision rule)
  (2) the model's STORED calibrated FC threshold (V3: from its val_metrics; the
      factorized models: their checkpoint ``fc_thresh.thr``). Read-only — we do
      not search a threshold on UAV123.

MODELS
------
* V3-prod (always): the frozen 16-dim v2-feature causal CSCTCN. Reuses the EXACT
  V3 feature pipeline from tools/v3_fc_heldout_repro.py (CSCDataset ->
  build_sequence_features_v2) + the harness V3 dedup-per-frame extraction from
  tools/eval_factorized_compare.py — features are NOT reimplemented here.
* conditional / symmetric factorized (when ready, via --cond_ckpt / --fact_ckpt):
  these need the v4 41-dim ``feat_*`` shards for UAV123. UAV123 v4 shards do NOT
  exist yet (they require running the v4 builder on UAV123 RAW telemetry with the
  TRAINING calibrators). If --uav123_shards is given AND the file exists, we score
  them via eval_factorized_compare.predict_factorized using the checkpoint's
  STORED temperatures (calibrated in-domain — NOT on UAV123). Otherwise we print
  the precise blocker + the exact build command and score V3 only. No fabrication.

DATA (located, not fabricated)
------------------------------
* GT-derived labels : outputs/eval_v3fix/sglatrack/uav123/test/labels_v3
  (single labels.jsonl, 123 seqs / 112,578 frames; derived_state ∈ {CC,CU,LA,FC},
  pred_bbox / gt_bbox / iou / confidence|apce|psr in the SAME percentile-normalized
  scale as the V3 training labels v3fix_combined — so directly consumable by the
  V3 v2 pipeline). GT FC support = 654 frames across 21 sequences.
* raw telemetry (for the v4 builder, when building shards):
  outputs/eval_v3fix/sglatrack/uav123/test/passive_r2/telemetry/{seq}.jsonl

Run:
  # V3-prod UAV123 FC behavior (no factorized shards needed):
  .venv/bin/python tools/eval_uav123_ood.py

  # add the conditional challenger once its checkpoint + UAV123 v4 shards exist:
  .venv/bin/python tools/eval_uav123_ood.py \
      --uav123_shards outputs/csc_labels_v4/uav123_shards.jsonl \
      --cond_ckpt outputs/csc_training_v4/cond_fact_seed0/checkpoint_best.pth
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "tools"):
    sys.path.insert(0, str(_p))

# Derived class codes (single source of truth pinned to label_schema).
CC, CU, LA, FC = 0, 1, 2, 3
CLASS_NAMES = ["CC", "CU", "LA", "FC"]

DEFAULT_V3_RUN = PROJECT_ROOT / "outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2"
DEFAULT_UAV123_LABELS = PROJECT_ROOT / "outputs/eval_v3fix/sglatrack/uav123/test/labels_v3"
DEFAULT_UAV123_TELEMETRY = PROJECT_ROOT / "outputs/eval_v3fix/sglatrack/uav123/test/passive_r2/telemetry"
DEFAULT_OUT = PROJECT_ROOT / "outputs/csc_training_v4/uav123_ood_fc_metrics.json"


def _import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ===========================================================================
# FC behavior metrics (no GT-IoU at decision time; GT only for scoring)
# ===========================================================================
def fc_behavior(y_true: np.ndarray, pred_fc: np.ndarray) -> dict:
    """FC false-positive behavior from a boolean FC-prediction mask + GT.

    false_fc_rate = FP / (TP+FP) = 1 - precision  (the OOD safety number).
    """
    is_fc = y_true == FC
    tp = int(np.sum(pred_fc & is_fc))
    fp = int(np.sum(pred_fc & ~is_fc))
    fn = int(np.sum(~pred_fc & is_fc))
    n = int(y_true.size)
    fired = int(np.sum(pred_fc))
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    false_fc_rate = (fp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    return {
        "n_frames": n,
        "fc_support": int(is_fc.sum()),
        "fc_fired": fired,
        "fc_tp": tp, "fc_fp": fp, "fc_fn": fn,
        "fc_precision": prec,
        "fc_recall": rec,
        "fc_f1": f1,
        "false_fc_rate": false_fc_rate,          # = 1 - precision
        "fc_firings_per_1k": 1000.0 * fired / n if n else float("nan"),
        "false_fc_firings_per_1k": 1000.0 * fp / n if n else float("nan"),
    }


def fc_behavior_per_dataset(y_true, pred_fc, datasets) -> dict:
    out = {}
    for ds in sorted(set(datasets.tolist())):
        m = datasets == ds
        out[ds] = fc_behavior(y_true[m], pred_fc[m])
    return out


def fc_behavior_per_sequence(y_true, pred_fc, seq_ids) -> dict:
    """Per-sequence FC firings — surfaces which OOD sequences the model
    false-fires on (the operationally important failure list)."""
    rows = []
    for sid in sorted(set(seq_ids.tolist())):
        m = seq_ids == sid
        b = fc_behavior(y_true[m], pred_fc[m])
        if b["fc_fired"] > 0 or b["fc_support"] > 0:
            rows.append({"sequence": sid, **{k: b[k] for k in
                        ("n_frames", "fc_support", "fc_fired", "fc_tp", "fc_fp",
                         "fc_precision", "false_fc_rate")}})
    return {"sequences": rows}


# ===========================================================================
# V3-prod UAV123 predictions (REUSE v3_fc_heldout_repro + eval_factorized_compare)
# ===========================================================================
def predict_v3_uav123(v3_run_dir: Path, uav123_labels_dir: Path, device: str = "cpu") -> dict:
    """Dedup per-frame V3 4-way probs over ALL UAV123 sequences (no split).

    Reuses the V3 config/model/CSCDataset machinery (feature_version v2, window
    32) and the harness's full-softmax window runner. The dedup-per-frame causal
    rule matches tools/v3_fc_heldout_repro.py EXACTLY (window k's last step is
    positional frame k+W-1; first W-1 frames from window 0).

    Returns {preds:{(ds,seq,pos):(4,)}, gt:{(ds,seq,pos):int}, window, n_seq}.
    """
    import torch  # noqa: F401

    repro = _import_module(PROJECT_ROOT / "tools" / "v3_fc_heldout_repro.py", "v3_fc_heldout_repro")
    efc = _import_module(PROJECT_ROOT / "tools" / "eval_factorized_compare.py", "eval_factorized_compare")

    # Point the repro module's hardcoded paths at the prod run.
    repro.RUN_DIR = Path(v3_run_dir)
    repro.CKPT = Path(v3_run_dir) / "checkpoint_best.pth"
    repro.CONFIG = Path(v3_run_dir) / "config_resolved.yaml"
    repro.LABELS_DIR = Path(uav123_labels_dir)

    cfg = repro.load_config()
    assert cfg.feature.feature_version == "v2", cfg.feature.feature_version
    W = cfg.feature.window_size
    model = repro.load_model(cfg)

    rows = repro.load_labels_dir(repro.LABELS_DIR)
    # SAFETY: this MUST be UAV123 (final-test) and ONLY UAV123.
    ds_names = sorted({r.get("dataset") for r in rows})
    assert ds_names == ["uav123"], f"expected only uav123 labels, got {ds_names}"
    groups = repro._group_by_sequence(rows)  # sorted by frame_idx, stable

    ds = repro.CSCDataset(groups, cfg.feature, image_size=repro.IMAGE_SIZE)
    probs_win = efc._v3_predict_full(model, ds, device=device)  # full (W,4) per window

    preds: dict[tuple, np.ndarray] = {}
    gt: dict[tuple, int] = {}
    wi = 0
    for (dataset, sequence), srows in groups.items():
        T = len(srows)
        n_win = (T - W) + 1 if T >= W else 0
        if n_win == 0:
            continue
        seq_probs = probs_win[wi:wi + n_win]
        for pos in range(T):
            if pos <= W - 1:
                p = seq_probs[0][pos]
            else:
                k = pos - (W - 1)
                p = seq_probs[k][W - 1]
            key = (dataset, sequence, pos)
            preds[key] = np.asarray(p, dtype=np.float64)
            gt[key] = int(srows[pos].get("derived_state", 0))
        wi += n_win
    return {"preds": preds, "gt": gt, "window": W, "n_seq": len(groups)}


def read_v3_fc_threshold(v3_run_dir: Path) -> Optional[float]:
    """V3's FROZEN calibrated FC threshold from its val_metrics.json (read-only).

    Tries a few known key layouts; returns None if not present (then argmax-only).
    """
    vm_path = Path(v3_run_dir) / "val_metrics.json"
    if not vm_path.exists():
        return None
    vm = json.loads(vm_path.read_text())
    for path in (("fc_thresh", "thr"), ("false_confirmed_thr",), ("fc_threshold",)):
        cur = vm
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and isinstance(cur, (int, float)):
            return float(cur)
    return None


# ===========================================================================
# Reporting
# ===========================================================================
def _fmt(x, p=4):
    if x is None:
        return "   --"
    if isinstance(x, float):
        return "   n/a" if x != x else f"{x:.{p}f}"
    return str(x)


def print_fc_table(name: str, op_argmax: dict, op_thr: Optional[dict], thr: Optional[float]) -> None:
    print("\n" + "=" * 78)
    print(f"UAV123 OOD — FALSE_CONFIRMED behavior :: {name}")
    print("=" * 78)
    cols = [("4-way argmax", op_argmax)]
    if op_thr is not None:
        cols.append((f"P(FC)>={thr:.3f} (frozen)", op_thr))
    hdr = f"  {'metric':<28}" + "".join(f"{c[0]:>26}" for c in cols)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for lbl, key in (
        ("n_frames", "n_frames"),
        ("FC support (GT)", "fc_support"),
        ("FC fired", "fc_fired"),
        ("FC TP", "fc_tp"),
        ("FC FP (false-fire)", "fc_fp"),
        ("FC precision", "fc_precision"),
        ("FC recall", "fc_recall"),
        ("FC F1", "fc_f1"),
        ("false-FC rate (1-prec)", "false_fc_rate"),
        ("FC firings / 1k", "fc_firings_per_1k"),
        ("false-FC firings / 1k", "false_fc_firings_per_1k"),
    ):
        line = f"  {lbl:<28}"
        for _, op in cols:
            line += f"{_fmt(op[key]):>26}"
        print(line)


def print_per_dataset_fc(name: str, per_ds: dict) -> None:
    print(f"\n  [{name}] per-dataset FC (argmax):")
    for ds, b in per_ds.items():
        print(f"    {ds:<12} false-FC-rate={_fmt(b['false_fc_rate'])} "
              f"prec={_fmt(b['fc_precision'])} rec={_fmt(b['fc_recall'])} "
              f"fired={b['fc_fired']} fp={b['fc_fp']} (n={b['n_frames']}, GT-FC={b['fc_support']})")


def print_per_seq_fc(name: str, per_seq: dict, top: int = 12) -> None:
    rows = sorted(per_seq["sequences"], key=lambda r: (-r["fc_fp"], -r["fc_fired"]))
    print(f"\n  [{name}] top sequences by false-FC firings (fp):")
    print(f"    {'sequence':<14}{'fired':>7}{'tp':>5}{'fp':>6}{'GT-FC':>7}{'prec':>8}{'falseFC':>9}")
    for r in rows[:top]:
        print(f"    {r['sequence']:<14}{r['fc_fired']:>7}{r['fc_tp']:>5}{r['fc_fp']:>6}"
              f"{r['fc_support']:>7}{_fmt(r['fc_precision']):>8}{_fmt(r['false_fc_rate']):>9}")


def print_gate4(v3_arg: dict, cond_results: dict) -> dict:
    """Gate 4: does the conditional model NOTABLY reduce UAV123 false-FC vs V3?"""
    sep = "=" * 78
    print("\n" + sep)
    print("GATE 4 — UAV123 false-FC reduction (conditional vs V3-prod)")
    print(sep)
    verdict = {"v3_false_fc_rate": v3_arg["false_fc_rate"],
               "v3_false_fc_firings_per_1k": v3_arg["false_fc_firings_per_1k"],
               "challengers": {}}
    if not cond_results:
        print("  PENDING — no conditional/factorized challenger scored on UAV123.")
        print("  (V3-prod UAV123 FC behavior IS reported above; re-run with")
        print("   --uav123_shards + --cond_ckpt once both exist. No fabrication.)")
        verdict["recommendation"] = "PENDING_CONDITIONAL_ON_UAV123"
        print(sep)
        return verdict
    print(f"  V3-prod (argmax): false-FC rate={_fmt(v3_arg['false_fc_rate'])}  "
          f"false-FC firings/1k={_fmt(v3_arg['false_fc_firings_per_1k'])}  "
          f"FC F1={_fmt(v3_arg['fc_f1'])}\n")
    for nm, op in cond_results.items():
        d_rate = (op["false_fc_rate"] - v3_arg["false_fc_rate"]) if (
            op["false_fc_rate"] == op["false_fc_rate"]
            and v3_arg["false_fc_rate"] == v3_arg["false_fc_rate"]) else float("nan")
        d_fire = op["false_fc_firings_per_1k"] - v3_arg["false_fc_firings_per_1k"]
        reduced = (d_fire < 0) or (d_rate < 0)
        verdict["challengers"][nm] = {
            "false_fc_rate": op["false_fc_rate"],
            "false_fc_firings_per_1k": op["false_fc_firings_per_1k"],
            "fc_f1": op["fc_f1"],
            "d_false_fc_rate_vs_v3": d_rate,
            "d_false_fc_firings_per_1k_vs_v3": d_fire,
            "reduced_false_fc": bool(reduced),
        }
        print(f"  {nm:<22} false-FC rate={_fmt(op['false_fc_rate'])} (Δ={_fmt(d_rate, 4)})  "
              f"false-FC firings/1k={_fmt(op['false_fc_firings_per_1k'])} (Δ={d_fire:+.3f})  "
              f"FC F1={_fmt(op['fc_f1'])}  -> {'REDUCED' if reduced else 'NOT REDUCED'}")
    verdict["recommendation"] = "SEE_PER_CHALLENGER"
    print(sep)
    return verdict


# ===========================================================================
# Factorized / conditional scoring on UAV123 (needs v4 shards)
# ===========================================================================
def score_factorized_on_uav123(ckpt: Path, uav123_shards: Path, window: int, device: str):
    """Score a (symmetric OR conditional) factorized checkpoint on UAV123 v4 shards.

    Reuses eval_factorized_compare.predict_factorized (schema-tolerant loader +
    the checkpoint's STORED calibrated temperatures — never recalibrated on
    UAV123). Returns (preds{(ds,seq,pos):(4,)}, gt{(ds,seq,pos):int}, meta).
    """
    efc = _import_module(PROJECT_ROOT / "tools" / "eval_factorized_compare.py", "eval_factorized_compare")
    from csc_lib.csc.v4.features_v4 import FEATURE_NAMES_V4

    # Load UAV123 shards grouped by sequence (val_keys=all uav123 seqs).
    by_seq: dict[tuple, list[dict]] = {}
    import collections
    tmp = collections.defaultdict(list)
    gt: dict[tuple, int] = {}
    with open(uav123_shards) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            ds = d.get("dataset")
            if "uav123" not in str(ds).lower():
                raise RuntimeError(f"non-uav123 row in {uav123_shards} (dataset={ds}).")
            tmp[(ds, d.get("sequence"))].append(d)
    for k in tmp:
        tmp[k].sort(key=lambda r: r["frame_idx"])
    by_seq = dict(tmp)
    for (dataset, sequence), srows in by_seq.items():
        for pos, r in enumerate(srows):
            gt[(dataset, sequence, pos)] = int(r["derived"])

    preds, meta = efc.predict_factorized(
        ckpt, by_seq, FEATURE_NAMES_V4, window, device=device, use_ckpt_temps=True
    )
    return preds, gt, meta


# ===========================================================================
# Main
# ===========================================================================
def _jsonsafe(o):
    if isinstance(o, dict):
        return {k: _jsonsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonsafe(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    return o


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v3_run_dir", default=str(DEFAULT_V3_RUN))
    ap.add_argument("--uav123_labels_dir", default=str(DEFAULT_UAV123_LABELS),
                    help="dir holding the UAV123 GT-derived labels.jsonl (V3 v2 input)")
    ap.add_argument("--uav123_shards", default=None,
                    help="UAV123 v4 41-dim feat shards (jsonl) for the factorized/conditional models")
    ap.add_argument("--cond_ckpt", action="append", default=None,
                    help="conditional CondFactorizedCSC checkpoint (repeatable)")
    ap.add_argument("--fact_ckpt", action="append", default=None,
                    help="symmetric FactorizedCSC checkpoint (repeatable)")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip_v3", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    print("=" * 78)
    print("UAV123 OOD FC-BEHAVIOR EVAL  (FINAL-TEST-ONLY — no tuning/calibration on UAV123)")
    print("=" * 78)
    print(f"  labels    : {args.uav123_labels_dir}")
    print(f"  v3_run    : {args.v3_run_dir}")
    print(f"  shards    : {args.uav123_shards or '(none — factorized/conditional deferred)'}")

    payload: dict = {"dataset": "uav123", "final_test_only": True, "models": {}}

    # ---- V3-prod ----
    v3_arg = None
    if not args.skip_v3:
        print("\n>>> V3-prod: dedup per-frame prediction over all UAV123 seqs ...")
        v3 = predict_v3_uav123(Path(args.v3_run_dir), Path(args.uav123_labels_dir), device=args.device)
        keys = sorted(v3["preds"].keys())
        y_true = np.array([v3["gt"][k] for k in keys], dtype=np.int64)
        probs = np.stack([v3["preds"][k] for k in keys], axis=0)
        datasets = np.array([k[0] for k in keys])
        seqids = np.array([k[1] for k in keys])
        print(f"    scored {v3['n_seq']} seqs, {len(keys):,} dedup frames, window={v3['window']}")

        y_pred = probs.argmax(axis=1)
        pred_fc_arg = y_pred == FC
        v3_arg = fc_behavior(y_true, pred_fc_arg)

        thr = read_v3_fc_threshold(Path(args.v3_run_dir))
        v3_thr = None
        if thr is not None:
            v3_thr = fc_behavior(y_true, probs[:, FC] >= thr)
        print_fc_table("V3-prod", v3_arg, v3_thr, thr)
        per_ds = fc_behavior_per_dataset(y_true, pred_fc_arg, datasets)
        per_seq = fc_behavior_per_sequence(y_true, pred_fc_arg, seqids)
        print_per_dataset_fc("V3-prod", per_ds)
        print_per_seq_fc("V3-prod", per_seq)
        payload["models"]["V3-prod"] = {
            "argmax": v3_arg, "frozen_thr": v3_thr, "frozen_fc_threshold": thr,
            "per_dataset": per_ds, "per_sequence_false_fc": per_seq,
        }

    # ---- factorized / conditional (needs UAV123 v4 shards) ----
    cond_results: dict = {}
    fact_specs = []
    for cp in (args.cond_ckpt or []):
        fact_specs.append((f"cond_{Path(cp).parent.name or Path(cp).stem}", Path(cp)))
    for cp in (args.fact_ckpt or []):
        fact_specs.append((f"fact_{Path(cp).parent.name or Path(cp).stem}", Path(cp)))

    if fact_specs:
        shards = Path(args.uav123_shards) if args.uav123_shards else None
        if shards is None or not shards.exists():
            print("\n" + "=" * 78)
            print("FACTORIZED / CONDITIONAL ON UAV123 — BLOCKED (missing UAV123 v4 shards)")
            print("=" * 78)
            print(f"  --uav123_shards not provided or not found: {shards}")
            print("  These models read the v4 41-dim feat_* shards (NOT the v2 16-dim labels).")
            print("  UAV123 v4 shards must be built from UAV123 RAW telemetry + GT labels with")
            print("  the EXISTING (training) calibrators — DO NOT refit calibrators on UAV123.")
            print(_uav123_shard_build_command())
            payload["factorized_blocker"] = "missing_uav123_v4_shards"
        else:
            for nm, ckpt in fact_specs:
                if not ckpt.exists():
                    print(f"\n>>> {nm}: SKIP — checkpoint not found: {ckpt}")
                    continue
                print(f"\n>>> {nm}: scoring on UAV123 v4 shards {shards} ...")
                try:
                    preds, gt, meta = score_factorized_on_uav123(ckpt, shards, args.window, args.device)
                    keys = sorted(set(preds) & set(gt))
                    yt = np.array([gt[k] for k in keys], dtype=np.int64)
                    pr = np.stack([preds[k] for k in keys], axis=0)
                    dsa = np.array([k[0] for k in keys])
                    sqa = np.array([k[1] for k in keys])
                    yp = pr.argmax(axis=1)
                    op_arg = fc_behavior(yt, yp == FC)
                    # frozen FC threshold from the checkpoint (fc_thresh.thr), read-only.
                    import torch
                    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
                    thr = (ck.get("val", {}) or {}).get("fc_thresh", {}).get("thr") \
                        if isinstance(ck.get("val"), dict) else ck.get("fc_thresh", {}).get("thr")
                    op_thr = fc_behavior(yt, pr[:, FC] >= float(thr)) if thr is not None else None
                    print(f"    type={meta['model_type']} frames={len(keys):,}")
                    print_fc_table(nm, op_arg, op_thr, thr)
                    print_per_dataset_fc(nm, fc_behavior_per_dataset(yt, yp == FC, dsa))
                    print_per_seq_fc(nm, fc_behavior_per_sequence(yt, yp == FC, sqa))
                    cond_results[nm] = op_arg
                    payload["models"][nm] = {
                        "model_type": meta["model_type"], "argmax": op_arg,
                        "frozen_thr": op_thr, "frozen_fc_threshold": thr,
                    }
                except Exception as exc:
                    print(f"    {nm}: BLOCKED — {exc}")
                    payload.setdefault("factorized_errors", {})[nm] = str(exc)
    else:
        print("\n[info] no --cond_ckpt / --fact_ckpt given. Conditional checkpoints "
              "(outputs/csc_training_v4/cond_fact_seed*) are still TRAINING.")
        print("       V3-prod UAV123 FC behavior reported above. To add the conditional model:")
        print(_uav123_shard_build_command())

    # ---- Gate 4 ----
    if v3_arg is not None:
        payload["gate4"] = print_gate4(v3_arg, cond_results)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonsafe(payload), indent=2))
    print(f"\nsaved -> {out}")
    return 0


def _uav123_shard_build_command() -> str:
    return """
  # 1) build UAV123 v4 shards (RAW telemetry + GT labels, REUSE training calibrators):
  #    raw telemetry : outputs/eval_v3fix/sglatrack/uav123/test/passive_r2/telemetry/{seq}.jsonl
  #    GT labels     : outputs/eval_v3fix/sglatrack/uav123/test/labels_v3/uav123/test/labels.jsonl
  #    NOTE: tools/v4_build_labels.py currently FITS calibrators in pass 1; for a
  #    FINAL-TEST set it must LOAD the in-domain calibrators
  #    (outputs/csc_labels_v4/v4_feature_calibrators.json) instead of refitting on
  #    UAV123. Add a --load_calibrators hook (owned by the v4 builder) before this.
  .venv/bin/python tools/v4_build_labels.py \\
      --labels_jsonl outputs/eval_v3fix/sglatrack/uav123/test/labels_v3/uav123/test/labels.jsonl \\
      --telemetry_root outputs/eval_v3fix/sglatrack/uav123/test/passive_r2 \\
      --datasets uav123 \\
      --load_calibrators outputs/csc_labels_v4/v4_feature_calibrators.json \\
      --out outputs/csc_labels_v4/uav123_shards.jsonl
  # 2) score the conditional model on UAV123 (final-test-only; checkpoint's stored temps):
  .venv/bin/python tools/eval_uav123_ood.py \\
      --uav123_shards outputs/csc_labels_v4/uav123_shards.jsonl \\
      --cond_ckpt outputs/csc_training_v4/cond_fact_seed0/checkpoint_best.pth
""".rstrip()


if __name__ == "__main__":
    raise SystemExit(main())
