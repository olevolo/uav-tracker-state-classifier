#!/usr/bin/env python
"""Controlled challenger study: V3-prod vs two factorized 2-tower CSC models.

WHAT THIS IS
------------
A rigorous, **deduplicated-per-frame** comparison harness that scores three
4-way (CC/CU/LA/FC) diagnostic models on the *exact same* held-out frames and
prints a decisive side-by-side table + an explicit Go/No-Go gate verdict:

  * ``V3-prod``            : the frozen production model
                             (outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2),
                             a 16-dim v2-feature causal CSCTCN with a 4-way derived head.
  * ``factorized_small``   : a 2-tower FactorizedCSC (off_target axis + confirmed axis),
                             plain per-axis loss.
  * ``factorized_composed``: the same 2-tower architecture trained with an extra
                             composed-posterior loss term (lambda_composed > 0).

The two factorized models are produced by a sibling agent
(``csc_lib/csc/v4/model_v4_factorized.FactorizedCSC`` +
``tools/train_csc_v4_factorized.py``). This harness is **parameterized by
checkpoint paths** and runs the full 3-way comparison once they exist; until
then it runs V3 + self-tests the factorized path on synthetic data (and says so
— it never fabricates factorized numbers).

DEDUP / ALIGNMENT CONTRACT (verified — see module docstring of the report)
--------------------------------------------------------------------------
* SPLIT = V3's exact held-out split: ``val_sequences`` (58 seqs) from
  ``<v3_run>/split_info.json``.
* Eval frames = every frame of those 58 sequences (dedup per-frame: ONE causal
  prediction per positional frame slot, matching tools/v3_fc_heldout_repro.py).
* The V3 label dir (v3fix_combined) and the v4 shards
  (outputs/csc_labels_v4/train_shards.jsonl) hold the *same* per-sequence frames
  (measured: identical per-seq counts, 60,154 val frames; positional GT
  agreement 99.4%; the FALSE_CONFIRMED set is IDENTICAL — 734/734 positional
  match). Frames are aligned **positionally within each sequence** after sorting
  by ``frame_idx`` (NOTE: 4 lasot ``drone-*`` sequences have *duplicated*
  frame_idx values in both sources, so frame_idx is not a unique key — position
  is). The single authoritative GT for the headline table is the **v4 shard
  ``derived``** label (what the factorized models were trained against); V3 is
  additionally cross-checked against its own GT.

SHARED FACTORIZED CONTRACT (must match the checkpoints)
-------------------------------------------------------
Checkpoint dict (``torch.load``) carries: ``state_dict``, ``geom_features``,
``resp_features`` (lists of FEATURE_NAMES_V4 ``_pct`` names defining the two
disjoint towers), ``t_off``, ``t_conf`` (calibration temperatures),
``hidden``/``levels``/``kernel`` (TCN config), ``lambda_composed``. We build geom
& resp causal windows (len ``window``, left-pad, last-step) from the shard
``feat_*`` columns, run
``model.forward(geom, resp, last_step_only=True) -> (off_logit, conf_logit)``,
then compose with the stored temperatures into ``{p_cc,p_cu,p_la,p_fc}`` (using
the model's own ``compose()`` if present, else the documented product rule).

METRICS (all DEDUP per-frame, held-out val) for EACH of the 3 models
--------------------------------------------------------------------
* FC: precision / recall / F1 (4-way argmax AND at a calibrated threshold),
  FC-vs-CC, FC-vs-LA, FC-vs-ALL AUROC + AUPRC.
* CC / CU / LA per-class F1 + derived macro-F1.
* PER-DATASET macro-F1 + FC-F1 (an in-pool OOD-ish signal; the cheap LODO proxy).
* PER-SEQUENCE mean FC-F1 over the val sequences.

LEAVE-ONE-DATASET-OUT (--lodo): documents the exact retrain+eval command per
fold (needs the sibling trainer) and, as a cheap proxy now, reports the
per-dataset breakdown. UAV123 (true OOD final test) is DEFERRED and printed as a
PENDING command (it needs v4 full-telemetry features extracted on UAV123 first —
NOT fabricated here).

Run:
  # full 3-way comparison (once the factorized checkpoints exist):
  .venv/bin/python tools/eval_factorized_compare.py \
      --fact_small_ckpt   outputs/csc_training_v4/csc_v4_fact_small/checkpoint_best.pth \
      --fact_composed_ckpt outputs/csc_training_v4/csc_v4_fact_composed/checkpoint_best.pth

  # V3-only + harness self-test (no factorized checkpoints needed):
  .venv/bin/python tools/eval_factorized_compare.py

  # pure synthetic self-test (no datasets, no checkpoints):
  .venv/bin/python tools/eval_factorized_compare.py --selftest
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# --- import path: csc_lib lives at the repo ROOT; a 'src' tree also exists.
# Mirror tools/v3_fc_heldout_repro.py + tools/v4_factorized_proof.py headers.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "tools"):
    sys.path.insert(0, str(_p))

# Derived class codes (single source of truth: label_schema / v4types; pinned here).
CC, CU, LA, FC = 0, 1, 2, 3
CLASS_NAMES = ["CC", "CU", "LA", "FC"]

# Default artifact locations.
DEFAULT_V3_RUN = PROJECT_ROOT / "outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2"
DEFAULT_SHARDS = PROJECT_ROOT / "outputs/csc_labels_v4/train_shards.jsonl"
DEFAULT_V3_LABELS = PROJECT_ROOT / "outputs/csc_labels/sglatrack/v3fix_combined"
DEFAULT_FACT_SMALL = PROJECT_ROOT / "outputs/csc_training_v4/csc_v4_fact_small/checkpoint_best.pth"
DEFAULT_FACT_COMPOSED = PROJECT_ROOT / "outputs/csc_training_v4/csc_v4_fact_composed/checkpoint_best.pth"
DEFAULT_OUT = PROJECT_ROOT / "outputs/csc_training_v4/factorized_compare_metrics.json"


# ===========================================================================
# Pure metric helpers (no GT-IoU, all per-frame on dedup arrays)
# ===========================================================================
def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 0.0


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Per-class F1 for the 4 derived classes (matches sklearn macro/None)."""
    out: dict[str, float] = {}
    for c, nm in enumerate(CLASS_NAMES):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        out[nm] = _f1(tp, fp, fn)
    return out


def macro_f1_from_perclass(pc: dict[str, float]) -> float:
    return float(np.mean([pc[c] for c in CLASS_NAMES]))


def prf_for_class(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> tuple[float, float, float, int]:
    tp = int(np.sum((y_pred == c) & (y_true == c)))
    fp = int(np.sum((y_pred == c) & (y_true != c)))
    fn = int(np.sum((y_pred != c) & (y_true == c)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = _f1(tp, fp, fn)
    support = int(np.sum(y_true == c))
    return prec, rec, f1, support


def _auroc(y_bin: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via rank statistic (no sklearn dependency); nan if degenerate."""
    y = np.asarray(y_bin, dtype=bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    sorted_scores = score[order]
    # average ranks for ties
    ranks_sorted = np.arange(1, len(score) + 1, dtype=np.float64)
    i = 0
    n = len(score)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks_sorted[i : j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks[order] = ranks_sorted
    sum_pos = ranks[y].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _auprc(y_bin: np.ndarray, score: np.ndarray) -> float:
    """Average precision (area under PR curve), step interpolation; nan if no pos."""
    y = np.asarray(y_bin, dtype=bool)
    P = int(y.sum())
    if P == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    recall = tp / P
    precision = tp / np.maximum(tp + fp, 1)
    # AP = sum over thresholds of (R_k - R_{k-1}) * P_k
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    ap = float(np.sum((recall - rec_prev) * precision))
    return ap


def _prec_recall_f1_at_threshold(
    y_true: np.ndarray, p_fc: np.ndarray, thr: float
) -> tuple[float, float, float]:
    pred_fc = p_fc >= thr
    is_fc = y_true == FC
    tp = int(np.sum(pred_fc & is_fc))
    fp = int(np.sum(pred_fc & ~is_fc))
    fn = int(np.sum(~pred_fc & is_fc))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = _f1(tp, fp, fn)
    return prec, rec, f1


def _best_f1_threshold(y_true: np.ndarray, p_fc: np.ndarray) -> tuple[float, float, float, float]:
    """Sweep P(FC) thresholds, return (best_thr, prec, rec, f1) maximizing FC F1.

    This is the per-model "calibrated threshold" FC operating point (the 4-way
    argmax is reported separately). Thresholds are the unique scores on FC frames
    plus a small grid, keeping it cheap and deterministic.
    """
    is_fc = y_true == FC
    if not is_fc.any():
        return float("nan"), float("nan"), float("nan"), float("nan")
    # candidate thresholds: descending unique scores (sklearn-style PR sweep)
    cand = np.unique(p_fc)
    if cand.size > 2000:  # subsample for speed; keep extremes
        idx = np.linspace(0, cand.size - 1, 2000).astype(int)
        cand = cand[idx]
    best = (0.5, 0.0, 0.0, -1.0)
    for thr in cand:
        prec, rec, f1 = _prec_recall_f1_at_threshold(y_true, p_fc, float(thr))
        if f1 > best[3]:
            best = (float(thr), prec, rec, f1)
    return best  # (thr, prec, rec, f1)


def compute_model_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Full per-frame metric bundle for ONE model.

    Parameters
    ----------
    y_true : (N,) int   authoritative derived GT (0=CC..3=FC).
    probs  : (N,4) float per-frame [P(CC),P(CU),P(LA),P(FC)] (rows sum~1).
    """
    y_pred = probs.argmax(axis=1)
    p_fc = probs[:, FC]
    is_fc = y_true == FC
    is_cc = y_true == CC
    is_la = y_true == LA

    pc = per_class_f1(y_true, y_pred)
    macro = macro_f1_from_perclass(pc)

    # FC argmax operating point
    fc_prec_arg, fc_rec_arg, fc_f1_arg, fc_support = prf_for_class(y_true, y_pred, FC)
    # FC calibrated-threshold operating point
    thr, fc_prec_thr, fc_rec_thr, fc_f1_thr = _best_f1_threshold(y_true, p_fc)

    # FC ranking metrics
    fc_vs_all_auroc = _auroc(is_fc, p_fc)
    fc_vs_all_auprc = _auprc(is_fc, p_fc)
    m_fc_cc = is_fc | is_cc
    m_fc_la = is_fc | is_la
    fc_vs_cc_auroc = _auroc(is_fc[m_fc_cc], p_fc[m_fc_cc]) if m_fc_cc.any() else float("nan")
    fc_vs_cc_auprc = _auprc(is_fc[m_fc_cc], p_fc[m_fc_cc]) if m_fc_cc.any() else float("nan")
    fc_vs_la_auroc = _auroc(is_fc[m_fc_la], p_fc[m_fc_la]) if m_fc_la.any() else float("nan")
    fc_vs_la_auprc = _auprc(is_fc[m_fc_la], p_fc[m_fc_la]) if m_fc_la.any() else float("nan")

    return {
        "n_frames": int(y_true.size),
        "class_support": {nm: int(np.sum(y_true == c)) for c, nm in enumerate(CLASS_NAMES)},
        "per_class_f1": pc,
        "macro_f1": macro,
        "fc_argmax": {"precision": fc_prec_arg, "recall": fc_rec_arg, "f1": fc_f1_arg, "support": fc_support},
        "fc_calibrated": {"threshold": thr, "precision": fc_prec_thr, "recall": fc_rec_thr, "f1": fc_f1_thr},
        "fc_vs_cc": {"auroc": fc_vs_cc_auroc, "auprc": fc_vs_cc_auprc},
        "fc_vs_la": {"auroc": fc_vs_la_auroc, "auprc": fc_vs_la_auprc},
        "fc_vs_all": {"auroc": fc_vs_all_auroc, "auprc": fc_vs_all_auprc},
    }


def compute_per_dataset(
    y_true: np.ndarray, probs: np.ndarray, datasets: np.ndarray
) -> dict[str, dict]:
    """Per-dataset macro-F1, FC-F1 (4-way argmax), FC precision/recall, and the
    FC-vs-CC / FC-vs-LA / FC-vs-ALL AUROCs. The cheap (in-pool) LODO proxy."""
    out: dict[str, dict] = {}
    y_pred = probs.argmax(axis=1)
    p_fc = probs[:, FC]
    for ds in sorted(set(datasets.tolist())):
        m = datasets == ds
        yt, yp, pf = y_true[m], y_pred[m], p_fc[m]
        pc = per_class_f1(yt, yp)
        fc_prec, fc_rec, fc_f1, fc_sup = prf_for_class(yt, yp, FC)
        is_fc, is_cc, is_la = (yt == FC), (yt == CC), (yt == LA)
        m_fc_cc, m_fc_la = is_fc | is_cc, is_fc | is_la
        out[ds] = {
            "n_frames": int(m.sum()),
            "n_fc": int(is_fc.sum()),
            "macro_f1": macro_f1_from_perclass(pc),
            "fc_f1": pc["FC"],
            "fc_precision": fc_prec,
            "fc_recall": fc_rec,
            "fc_vs_cc_auroc": _auroc(is_fc[m_fc_cc], pf[m_fc_cc]) if m_fc_cc.any() else float("nan"),
            "fc_vs_la_auroc": _auroc(is_fc[m_fc_la], pf[m_fc_la]) if m_fc_la.any() else float("nan"),
            "fc_vs_all_auroc": _auroc(is_fc, pf),
        }
    return out


def compute_per_sequence_fc_f1(
    y_true: np.ndarray, probs: np.ndarray, seq_ids: np.ndarray
) -> dict:
    """Mean FC-F1 over the val sequences (a sequence with no FC frames and no FC
    prediction contributes F1=nan and is excluded from the mean; one with FC GT
    but zero TP contributes 0)."""
    y_pred = probs.argmax(axis=1)
    per_seq: list[float] = []
    n_with_fc = 0
    for sid in sorted(set(seq_ids.tolist())):
        m = seq_ids == sid
        yt, yp = y_true[m], y_pred[m]
        has_fc_gt = bool(np.any(yt == FC))
        has_fc_pred = bool(np.any(yp == FC))
        if not has_fc_gt and not has_fc_pred:
            continue  # FC-F1 undefined (no positives, no predicted positives)
        tp = int(np.sum((yp == FC) & (yt == FC)))
        fp = int(np.sum((yp == FC) & (yt != FC)))
        fn = int(np.sum((yp != FC) & (yt == FC)))
        per_seq.append(_f1(tp, fp, fn))
        if has_fc_gt:
            n_with_fc += 1
    mean = float(np.mean(per_seq)) if per_seq else float("nan")
    return {"mean_fc_f1": mean, "n_seq_scored": len(per_seq), "n_seq_with_fc_gt": n_with_fc}


# ===========================================================================
# V3-prod predictions (REUSE tools/v3_fc_heldout_repro.py)
# ===========================================================================
def _import_v3_repro():
    """Import tools/v3_fc_heldout_repro.py as a module (reuse its loaders)."""
    path = PROJECT_ROOT / "tools" / "v3_fc_heldout_repro.py"
    spec = importlib.util.spec_from_file_location("v3_fc_heldout_repro", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _v3_predict_full(model, ds, device: str = "cpu"):
    """Run V3 over a CSCDataset window-by-window, return per-window FULL 4-way probs.

    Mirrors v3_fc_heldout_repro._predict_dataset but keeps the full softmax (we
    need P(CC/CU/LA) for the LA/CU/CC F1 and the FC-vs-CC / FC-vs-LA slices, which
    the repro discards). One (T,4) array per window, in dataset (= insertion)
    order.
    """
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=256, shuffle=False)
    probs_win: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            out = model(x)
            p = torch.softmax(out.derived_logits, dim=-1).cpu().numpy()  # (B,T,4)
            for b in range(p.shape[0]):
                probs_win.append(p[b])
    return probs_win


def predict_v3(v3_run_dir: Path, v3_labels_dir: Path, device: str = "cpu") -> dict:
    """Return V3 dedup per-frame predictions keyed by (dataset, seq, pos).

    Uses v3_fc_heldout_repro's config/model/dataset machinery and replicates its
    dedup-per-frame causal extraction (window k's LAST step is positional frame
    k+W-1; the first W-1 frames are taken from window 0). Also asserts the split
    reproduced from labels == split_info.json.

    Returns dict with:
      preds:   {(ds,seq,pos): np.ndarray shape (4,)}   per-frame 4-way probs
      gt_v3:   {(ds,seq,pos): int}                      V3's own derived GT
      val_keys: list[(ds,seq)]                          the 58 val seqs
      window:  int
    """
    import torch  # noqa: F401

    repro = _import_v3_repro()
    # Honor caller-provided run/label dirs by patching the module globals the
    # repro reads (it hardcodes RUN_DIR/LABELS_DIR at import).
    repro.RUN_DIR = Path(v3_run_dir)
    repro.CKPT = Path(v3_run_dir) / "checkpoint_best.pth"
    repro.CONFIG = Path(v3_run_dir) / "config_resolved.yaml"
    repro.SPLIT_INFO = Path(v3_run_dir) / "split_info.json"
    repro.VAL_METRICS = Path(v3_run_dir) / "val_metrics.json"
    repro.LABELS_DIR = Path(v3_labels_dir)

    cfg = repro.load_config()
    assert cfg.feature.feature_version == "v2", cfg.feature.feature_version
    W = cfg.feature.window_size
    model = repro.load_model(cfg)

    rows = repro.load_labels_dir(repro.LABELS_DIR)
    groups = repro._group_by_sequence(rows)  # sorted by frame_idx, stable
    train_keys, val_keys = repro.split_sequences_stratified(
        list(groups.keys()), groups, cfg.val_fraction
    )
    si = json.loads(repro.SPLIT_INFO.read_text())
    si_val = set(tuple(x) for x in si["val_sequences"])
    assert set(val_keys) == si_val, "V3 val split does NOT match split_info.json!"

    val_rows = {k: groups[k] for k in val_keys}
    val_ds = repro.CSCDataset(val_rows, cfg.feature, image_size=repro.IMAGE_SIZE)
    probs_win = _v3_predict_full(model, val_ds, device=device)

    # Dedup per-frame (positional), EXACTLY as v3_fc_heldout_repro does.
    preds: dict[tuple, np.ndarray] = {}
    gt_v3: dict[tuple, int] = {}
    wi = 0
    for (dataset, sequence), srows in val_rows.items():
        T = len(srows)
        n_win = (T - W) + 1 if T >= W else 0
        if n_win == 0:
            # sequence shorter than the window — fall back to the single padded
            # window the dataset would NOT have created; skip (matches repro,
            # which also yields 0 windows). These frames are simply unscored by
            # V3; they are excluded from the common eval set downstream.
            continue
        seq_probs = probs_win[wi : wi + n_win]  # each (W,4)
        for pos in range(T):
            if pos <= W - 1:
                p = seq_probs[0][pos]          # step `pos` of window 0
            else:
                k = pos - (W - 1)              # window whose LAST step is frame `pos`
                p = seq_probs[k][W - 1]
            key = (dataset, sequence, pos)
            preds[key] = np.asarray(p, dtype=np.float64)
            gt_v3[key] = int(srows[pos].get("derived_state", 0))
        wi += n_win

    return {"preds": preds, "gt_v3": gt_v3, "val_keys": list(val_keys), "window": W}


# ===========================================================================
# Factorized model predictions (build geom/resp windows from shards -> compose)
# ===========================================================================
def _detect_model_type(ck: dict) -> str:
    """Detect factorized model type from the checkpoint schema (no fixed assumption).

    * ``conditional`` — has 3 axis temps / 3 confirmed heads:
      ``t_conf_on`` & ``t_conf_off`` keys present (top-level OR as state_dict
      buffers) and/or ``head_conf_on``/``head_conf_off`` (or ``conf_on_logit``)
      weights in the state_dict. Uses CondFactorizedCSC.compose(off,on,off_c).
    * ``symmetric`` — single ``confirmed`` head: a single ``t_conf`` and a
      ``resp_tower``/``conf_logit`` head. Uses FactorizedCSC.compose(off,conf).

    Detection is on KEYS PRESENT (not on a hardcoded schema) so it is tolerant of
    extra/renamed bookkeeping fields.
    """
    sd = ck.get("state_dict", {})
    sd_keys = list(sd.keys())
    has_cond = (
        ("t_conf_on" in ck) or ("t_conf_off" in ck)
        or ("t_conf_on" in sd) or ("t_conf_off" in sd)
        or any(("conf_on" in k) or ("conf_off" in k) for k in sd_keys)
        or any(("head_conf_on" in k) or ("head_conf_off" in k) for k in sd_keys)
    )
    return "conditional" if has_cond else "symmetric"


def _coerce_int(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _levels_to_int(levels, n_blocks_from_sd: Optional[int]) -> int:
    """Both factorized models take ``levels: int`` (number of residual blocks).

    The trainer stores ``levels`` as that int. Older/looser checkpoints may store
    a dilation LIST (e.g. ``[1,2,4,8]``) — in that case len(list) is the count.
    If absent, fall back to the block count recovered from the state_dict.
    """
    if isinstance(levels, int):
        return levels
    if isinstance(levels, (list, tuple)):
        return len(levels)
    if n_blocks_from_sd is not None:
        return n_blocks_from_sd
    return 3


def _n_blocks_from_state_dict(sd: dict, tower_prefix: str) -> Optional[int]:
    """Recover the residual-block count for a tower from its state_dict keys."""
    import re

    idxs = set()
    for k in sd:
        m = re.search(rf"{re.escape(tower_prefix)}\.blocks\.(\d+)\.", k)
        if m:
            idxs.add(int(m.group(1)))
    return (max(idxs) + 1) if idxs else None


def _load_factorized(ckpt_path: Path, device: str = "cpu", use_ckpt_temps: bool = True):
    """Load a factorized checkpoint (symmetric OR conditional). Returns (model, meta).

    SCHEMA-TOLERANT: the model type is detected from which temp keys / heads are
    present (``_detect_model_type``), NOT from a fixed schema. ``levels`` is
    coerced to an int whether the checkpoint stored an int (the trainer does) or a
    dilation list. Temperatures default to the values stored in the checkpoint
    (calibrated on the in-domain calib subset). ``use_ckpt_temps=False`` resets
    them to 1.0 (raw, uncalibrated) — never recalibrated on the eval set here.

    meta carries: model_type, geom_features, resp_features, the axis temps
    (t_off + t_conf OR t_off/t_conf_on/t_conf_off), lambda_composed, the TCN
    config (hidden/levels/kernel), and n_params.
    """
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "state_dict" not in ck:
        raise RuntimeError(f"{ckpt_path}: checkpoint is not a dict with 'state_dict'.")
    sd = ck["state_dict"]

    geom_features = ck.get("geom_features")
    resp_features = ck.get("resp_features")
    if not geom_features or not resp_features:
        raise RuntimeError(
            f"{ckpt_path}: checkpoint missing 'geom_features'/'resp_features' "
            "(the disjoint tower feature names). Cannot align shard columns."
        )

    model_type = _detect_model_type(ck)
    lambda_composed = ck.get("lambda_composed", None)
    hidden = _coerce_int(ck.get("hidden"), 32)
    kernel = _coerce_int(ck.get("kernel"), 3)
    # The geom tower's block prefix differs by model class (geom_tower vs geom_enc).
    n_blocks = (
        _n_blocks_from_state_dict(sd, "geom_tower")
        or _n_blocks_from_state_dict(sd, "geom_enc")
    )
    levels = _levels_to_int(ck.get("levels"), n_blocks)
    geom_dim = _coerce_int(ck.get("geom_dim"), len(geom_features))
    resp_dim = _coerce_int(ck.get("resp_dim"), len(resp_features))

    # Import the right model class.
    if model_type == "conditional":
        try:
            from csc_lib.csc.v4.model_v4_cond_factorized import CondFactorizedCSC as _ModelCls  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on sibling agent
            raise RuntimeError(
                "csc_lib.csc.v4.model_v4_cond_factorized.CondFactorizedCSC is not "
                f"importable ({exc}); cannot load the conditional checkpoint."
            ) from exc
    else:
        try:
            from csc_lib.csc.v4.model_v4_factorized import FactorizedCSC as _ModelCls  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on sibling agent
            raise RuntimeError(
                "csc_lib.csc.v4.model_v4_factorized.FactorizedCSC is not importable "
                f"({exc}); cannot load the symmetric checkpoint."
            ) from exc

    # Both classes share the (geom_dim, resp_dim, hidden, levels, kernel) signature.
    model = None
    ctor_errors = []
    for kwargs in (
        dict(geom_dim=geom_dim, resp_dim=resp_dim, hidden=hidden, levels=levels, kernel=kernel),
        dict(geom_dim=geom_dim, resp_dim=resp_dim, hidden=hidden, levels=levels),
        dict(geom_dim=geom_dim, resp_dim=resp_dim),
    ):
        try:
            model = _ModelCls(**kwargs)  # type: ignore[arg-type]
            break
        except Exception as e:  # try next signature
            ctor_errors.append(f"{list(kwargs.keys())} -> {e}")
            model = None
    if model is None:
        raise RuntimeError(
            f"{ckpt_path}: could not construct {_ModelCls.__name__} with any known "
            f"signature. Tried: {ctor_errors}"
        )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    # The temperature buffers live in the state_dict; loading them sets the
    # calibrated temps on the model. Tolerate ONLY the temp buffers being absent
    # (older ckpts) — any missing/unexpected LEARNED weight is a real mismatch.
    real_missing = [k for k in missing if not k.startswith(("t_off", "t_conf"))]
    real_unexpected = [k for k in unexpected if not k.startswith(("t_off", "t_conf"))]
    if real_missing or real_unexpected:
        raise RuntimeError(
            f"{ckpt_path.name}: state_dict mismatch on LEARNED weights "
            f"missing={real_missing[:6]} unexpected={real_unexpected[:6]} "
            f"(model_type={model_type}, geom_dim={geom_dim}, resp_dim={resp_dim}, "
            f"hidden={hidden}, levels={levels})."
        )

    # Resolve the calibrated temperatures (prefer top-level scalars, then buffers).
    def _temp(name: str) -> float:
        if name in ck:
            return float(ck[name])
        if name in sd:
            return float(sd[name])
        buf = getattr(model, name, None)
        return float(buf) if buf is not None else 1.0

    if model_type == "conditional":
        t_off = _temp("t_off")
        t_conf_on = _temp("t_conf_on")
        t_conf_off = _temp("t_conf_off")
        if not use_ckpt_temps:
            t_off = t_conf_on = t_conf_off = 1.0
        if hasattr(model, "set_temperatures"):
            model.set_temperatures(t_off, t_conf_on, t_conf_off)
        temps = {"t_off": t_off, "t_conf_on": t_conf_on, "t_conf_off": t_conf_off}
    else:
        t_off = _temp("t_off")
        t_conf = _temp("t_conf")
        if not use_ckpt_temps:
            t_off = t_conf = 1.0
        if hasattr(model, "set_temperatures"):
            model.set_temperatures(t_off, t_conf)
        temps = {"t_off": t_off, "t_conf": t_conf}

    model.eval()
    meta = {
        "ckpt": str(ckpt_path),
        "model_type": model_type,
        "geom_features": list(geom_features),
        "resp_features": list(resp_features),
        "lambda_composed": lambda_composed,
        "hidden": hidden,
        "levels": levels,
        "kernel": kernel,
        "use_ckpt_temps": bool(use_ckpt_temps),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        **temps,
    }
    return model, meta


def _to_arr(x) -> np.ndarray:
    return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, dtype=np.float64).reshape(-1)


def _compose_probs_symmetric(model, off_logit, conf_logit, t_off: float, t_conf: float) -> np.ndarray:
    """Symmetric FactorizedCSC: (off, conf) -> (N,4) [P(CC),P(CU),P(LA),P(FC)].

    Prefer the model's own ``compose(off, conf)`` (uses its stored temp buffers —
    the contractual source of truth, already set from the checkpoint). Falls back
    to the documented temperature-scaled product rule with the passed temps::

        P(off)=sigmoid(off/t_off)  P(conf)=sigmoid(conf/t_conf)
        P(FC)=P(off)P(conf)  P(LA)=P(off)(1-P(conf))
        P(CC)=(1-P(off))P(conf)  P(CU)=(1-P(off))(1-P(conf))
    """
    import torch

    off_t = torch.as_tensor(off_logit, dtype=torch.float32).reshape(-1)
    conf_t = torch.as_tensor(conf_logit, dtype=torch.float32).reshape(-1)
    compose_fn = getattr(model, "compose", None)
    if callable(compose_fn):
        out = None
        try:
            out = compose_fn(off_t, conf_t)
        except Exception:
            out = None
        if isinstance(out, dict) and {"p_cc", "p_cu", "p_la", "p_fc"} <= set(out):
            return np.stack(
                [_to_arr(out["p_cc"]), _to_arr(out["p_cu"]), _to_arr(out["p_la"]), _to_arr(out["p_fc"])],
                axis=1,
            )
        if out is not None and not isinstance(out, dict):
            arr = np.asarray(out.detach().cpu().numpy() if hasattr(out, "detach") else out, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 4:
                return arr
    # Fallback: documented product rule.
    p_off = torch.sigmoid(off_t / max(t_off, 1e-6)).numpy().astype(np.float64)
    p_conf = torch.sigmoid(conf_t / max(t_conf, 1e-6)).numpy().astype(np.float64)
    return np.stack(
        [(1.0 - p_off) * p_conf, (1.0 - p_off) * (1.0 - p_conf), p_off * (1.0 - p_conf), p_off * p_conf],
        axis=1,
    )


def _compose_probs_conditional(model, off_logit, conf_on_logit, conf_off_logit,
                               t_off: float, t_conf_on: float, t_conf_off: float) -> np.ndarray:
    """Conditional CondFactorizedCSC: (off, conf_on, conf_off) -> (N,4).

    The two confidence axes are CONDITIONAL (CC-vs-CU uses conf_on; FC-vs-LA uses
    conf_off)::

        P(CC)=(1-p_off)*p_on   P(CU)=(1-p_off)*(1-p_on)
        P(FC)=p_off*p_off_c    P(LA)=p_off*(1-p_off_c)

    Prefers the model's own ``compose(off, conf_on, conf_off)`` (its temp buffers
    are already set from the checkpoint); else applies the rule with passed temps.
    """
    import torch

    off_t = torch.as_tensor(off_logit, dtype=torch.float32).reshape(-1)
    on_t = torch.as_tensor(conf_on_logit, dtype=torch.float32).reshape(-1)
    offc_t = torch.as_tensor(conf_off_logit, dtype=torch.float32).reshape(-1)
    compose_fn = getattr(model, "compose", None)
    if callable(compose_fn):
        out = None
        try:
            out = compose_fn(off_t, on_t, offc_t)
        except Exception:
            out = None
        if isinstance(out, dict) and {"p_cc", "p_cu", "p_la", "p_fc"} <= set(out):
            return np.stack(
                [_to_arr(out["p_cc"]), _to_arr(out["p_cu"]), _to_arr(out["p_la"]), _to_arr(out["p_fc"])],
                axis=1,
            )
    # Fallback: documented conditional rule.
    p_off = torch.sigmoid(off_t / max(t_off, 1e-6)).numpy().astype(np.float64)
    p_on = torch.sigmoid(on_t / max(t_conf_on, 1e-6)).numpy().astype(np.float64)
    p_offc = torch.sigmoid(offc_t / max(t_conf_off, 1e-6)).numpy().astype(np.float64)
    return np.stack(
        [(1.0 - p_off) * p_on, (1.0 - p_off) * (1.0 - p_on), p_off * (1.0 - p_offc), p_off * p_offc],
        axis=1,
    )


def _build_causal_windows(feat_seq: np.ndarray, window: int) -> np.ndarray:
    """(T, F) per-sequence features -> (T, window, F) causal left-padded windows.

    Window ending at position t covers [t-window+1 .. t], left-padded with the
    first frame (edge-pad) for t < window-1 — matching the runtime "last-step"
    causal contract (the model only reads its last step). One window per frame,
    so we get a dedup per-frame prediction directly.
    """
    T, Fdim = feat_seq.shape
    out = np.empty((T, window, Fdim), dtype=np.float32)
    for t in range(T):
        lo = t - window + 1
        if lo >= 0:
            out[t] = feat_seq[lo : t + 1]
        else:
            pad = np.repeat(feat_seq[0:1], -lo, axis=0)  # edge-pad with first frame
            out[t] = np.concatenate([pad, feat_seq[0 : t + 1]], axis=0)
    return out


def predict_factorized(
    ckpt_path: Path,
    shard_rows_by_seq: dict[tuple, list[dict]],
    feature_names_v4: tuple[str, ...],
    window: int,
    device: str = "cpu",
    batch_windows: int = 4096,
    use_ckpt_temps: bool = True,
) -> tuple[dict, dict]:
    """Return factorized dedup per-frame predictions keyed by (dataset,seq,pos).

    Handles BOTH the symmetric (off, conf) and the conditional (off, conf_on,
    conf_off) factorized models — the type is detected from the checkpoint by
    ``_load_factorized``. Builds the two disjoint towers' causal windows from the
    shard ``feat_*`` columns (checkpoint geom_features/resp_features -> column
    indices via FEATURE_NAMES_V4), runs forward(last_step_only=True), composes
    with the checkpoint's stored (calibrated) temperatures.

    Returns (preds, meta) where preds = {(ds,seq,pos): np.ndarray (4,)}.
    """
    import torch

    model, meta = _load_factorized(ckpt_path, device=device, use_ckpt_temps=use_ckpt_temps)
    model_type = meta["model_type"]
    name2col = {n: i for i, n in enumerate(feature_names_v4)}
    miss_g = [n for n in meta["geom_features"] if n not in name2col]
    miss_r = [n for n in meta["resp_features"] if n not in name2col]
    if miss_g or miss_r:
        raise RuntimeError(
            f"{ckpt_path.name}: tower features not in FEATURE_NAMES_V4 "
            f"(geom missing {miss_g}, resp missing {miss_r})."
        )
    geom_cols = [name2col[n] for n in meta["geom_features"]]
    resp_cols = [name2col[n] for n in meta["resp_features"]]

    preds: dict[tuple, np.ndarray] = {}
    model.to(device)
    with torch.no_grad():
        for (dataset, sequence), srows in shard_rows_by_seq.items():
            T = len(srows)
            if T == 0:
                continue
            # Per-frame feature rows for the two towers.
            geom_seq = np.array([[r[f"feat_{c}"] for c in geom_cols] for r in srows], dtype=np.float32)
            resp_seq = np.array([[r[f"feat_{c}"] for c in resp_cols] for r in srows], dtype=np.float32)
            geom_w = _build_causal_windows(geom_seq, window)  # (T,W,Gd)
            resp_w = _build_causal_windows(resp_seq, window)   # (T,W,Rd)
            # Forward in chunks of windows (one window per frame).
            off_all = np.empty(T, dtype=np.float64)
            a_all = np.empty(T, dtype=np.float64)   # conf (sym) OR conf_on (cond)
            b_all = np.empty(T, dtype=np.float64)   # unused (sym)  OR conf_off (cond)
            for s in range(0, T, batch_windows):
                e = min(T, s + batch_windows)
                g = torch.from_numpy(geom_w[s:e]).to(device)
                rr = torch.from_numpy(resp_w[s:e]).to(device)
                logits = _forward_factorized(model, g, rr, model_type)
                off_all[s:e] = logits[0].reshape(-1).detach().cpu().numpy()
                a_all[s:e] = logits[1].reshape(-1).detach().cpu().numpy()
                if model_type == "conditional":
                    b_all[s:e] = logits[2].reshape(-1).detach().cpu().numpy()
            if model_type == "conditional":
                probs = _compose_probs_conditional(
                    model, off_all, a_all, b_all,
                    meta["t_off"], meta["t_conf_on"], meta["t_conf_off"],
                )
            else:
                probs = _compose_probs_symmetric(model, off_all, a_all, meta["t_off"], meta["t_conf"])
            for pos in range(T):
                preds[(dataset, sequence, pos)] = probs[pos]
    return preds, meta


def _forward_factorized(model, geom: "torch.Tensor", resp: "torch.Tensor", model_type: str):
    """Run forward(geom, resp, last_step_only=True) and return axis logits.

    * symmetric  -> (off_logit, conf_logit)
    * conditional-> (off_logit, conf_on_logit, conf_off_logit)

    Robust to the model returning a dict (the actual contract) or a tuple/list.
    """
    out = model.forward(geom, resp, last_step_only=True)
    if isinstance(out, dict):
        off = out.get("off_logit", out.get("off"))
        if model_type == "conditional":
            conf_on = out.get("conf_on_logit")
            conf_off = out.get("conf_off_logit")
            if off is not None and conf_on is not None and conf_off is not None:
                return off, conf_on, conf_off
        else:
            conf = out.get("conf_logit", out.get("conf"))
            if off is not None and conf is not None:
                return off, conf
    if isinstance(out, (tuple, list)):
        if model_type == "conditional" and len(out) >= 3:
            return out[0], out[1], out[2]
        if len(out) >= 2:
            return out[0], out[1]
    raise RuntimeError(
        f"Unexpected forward output for model_type={model_type}: type={type(out)} "
        f"keys={list(out.keys()) if isinstance(out, dict) else 'n/a'}"
    )


# ===========================================================================
# Shard loading + GT
# ===========================================================================
def load_shard_val(shards_path: Path, val_keys: set[tuple]) -> dict[tuple, list[dict]]:
    """Load shard rows for the val sequences, grouped by (dataset,seq), sorted by
    frame_idx (STABLE — matching dataset._group_by_sequence so positions align
    with V3). UAV123 safety guard included."""
    by_seq: dict[tuple, list[dict]] = defaultdict(list)
    with open(shards_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            ds = d.get("dataset")
            if ds and "uav123" in str(ds).lower() and "uavtrack" not in str(ds).lower():
                raise RuntimeError(f"UAV123 present in train shards ({ds}); refusing (final-test-only).")
            k = (ds, d.get("sequence"))
            if k in val_keys:
                by_seq[k].append(d)
    for k in by_seq:
        by_seq[k].sort(key=lambda r: r["frame_idx"])  # stable sort, matches V3
    return dict(by_seq)


# ===========================================================================
# Reporting
# ===========================================================================
def _fmt(x, w: int = 9, p: int = 4) -> str:
    if x is None:
        return f"{'--':>{w}}"
    if isinstance(x, float):
        if x != x:
            return f"{'n/a':>{w}}"
        return f"{x:>{w}.{p}f}"
    return f"{x:>{w}}"


def print_comparison_table(results: dict[str, dict], present_models: list[str]) -> None:
    """results[model] = compute_model_metrics(...) bundle. Columns = models."""
    cols = present_models
    header = f"{'metric':<34}" + "".join(f"{m:>22}" for m in cols)
    sep = "=" * len(header)
    print("\n" + sep)
    print("DEDUP PER-FRAME COMPARISON  (held-out val, 58 V3 val seqs, GT = v4 shard derived)")
    print(sep)
    print(header)
    print("-" * len(header))

    def row(label, getter):
        line = f"{label:<34}"
        for m in cols:
            try:
                v = getter(results[m])
            except (KeyError, TypeError):
                v = None
            line += f"{_fmt(v, 22):>22}" if isinstance(v, str) else f"{_fmt(v, 22):>22}"
        print(line)

    print("-- 4-way derived (argmax) --")
    row("CC F1", lambda r: r["per_class_f1"]["CC"])
    row("CU F1", lambda r: r["per_class_f1"]["CU"])
    row("LA F1", lambda r: r["per_class_f1"]["LA"])
    row("FC F1 (argmax)", lambda r: r["fc_argmax"]["f1"])
    row("macro-F1", lambda r: r["macro_f1"])
    print("-- FALSE_CONFIRMED operating points --")
    row("FC precision (argmax)", lambda r: r["fc_argmax"]["precision"])
    row("FC recall (argmax)", lambda r: r["fc_argmax"]["recall"])
    row("FC F1 (calibrated thr)", lambda r: r["fc_calibrated"]["f1"])
    row("FC precision (cal thr)", lambda r: r["fc_calibrated"]["precision"])
    row("FC recall (cal thr)", lambda r: r["fc_calibrated"]["recall"])
    row("FC cal threshold", lambda r: r["fc_calibrated"]["threshold"])
    print("-- FC ranking (AUROC / AUPRC) --")
    row("FC-vs-CC AUROC", lambda r: r["fc_vs_cc"]["auroc"])
    row("FC-vs-CC AUPRC", lambda r: r["fc_vs_cc"]["auprc"])
    row("FC-vs-LA AUROC", lambda r: r["fc_vs_la"]["auroc"])
    row("FC-vs-LA AUPRC", lambda r: r["fc_vs_la"]["auprc"])
    row("FC-vs-ALL AUROC", lambda r: r["fc_vs_all"]["auroc"])
    row("FC-vs-ALL AUPRC", lambda r: r["fc_vs_all"]["auprc"])
    print("-- support --")
    row("n_frames", lambda r: r["n_frames"])
    row("FC support", lambda r: r["fc_argmax"]["support"])
    print(sep)


def print_per_dataset(per_ds: dict[str, dict[str, dict]], present_models: list[str]) -> None:
    """per_ds[model][dataset] = {macro_f1, fc_f1, fc_vs_cc_auroc, fc_vs_la_auroc, ...}."""
    all_ds = sorted({d for m in present_models for d in per_ds.get(m, {})})
    print("\n" + "=" * 78)
    print("PER-DATASET BREAKDOWN  (in-pool OOD-ish signal; cheap LODO proxy)")
    print("=" * 78)
    for metric_key, metric_lbl in (
        ("macro_f1", "macro-F1"),
        ("fc_f1", "FC-F1 (argmax)"),
        ("fc_vs_cc_auroc", "FC-vs-CC AUROC"),
        ("fc_vs_la_auroc", "FC-vs-LA AUROC"),
    ):
        print(f"\n  [{metric_lbl}]   (n_frames / n_FC in parens, from GT)")
        hdr = f"  {'dataset':<16}" + "".join(f"{m:>22}" for m in present_models)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for ds in all_ds:
            line = f"  {ds:<16}"
            n_info = ""
            for m in present_models:
                cell = per_ds.get(m, {}).get(ds)
                if cell is None or cell.get(metric_key) is None:
                    line += f"{'--':>22}"
                else:
                    val = cell[metric_key]
                    line += f"{'n/a':>22}" if (isinstance(val, float) and val != val) else f"{val:>22.4f}"
                    n_info = f"({cell['n_frames']}/{cell['n_fc']})"
            print(line + f"   {n_info}")


def print_go_no_go(
    results: dict[str, dict],
    per_seq: dict[str, dict],
    present_models: list[str],
    fact_present: list[str],
) -> dict:
    """Explicit Go/No-Go verdict block with the numbers. Returns a machine dict."""
    v3 = results.get("V3-prod")
    sep = "=" * 78
    print("\n" + sep)
    print("GO / NO-GO GATE ASSESSMENT")
    print(sep)

    verdict = {"gates": {}, "recommendation": None}

    if v3 is None:
        print("  V3-prod metrics unavailable — cannot assess gates.")
        verdict["recommendation"] = "V3 metrics unavailable"
        print(sep)
        return verdict

    v3_fc_f1 = v3["fc_argmax"]["f1"]
    v3_cc_f1 = v3["per_class_f1"]["CC"]
    v3_la_f1 = v3["per_class_f1"]["LA"]
    v3_macro = v3["macro_f1"]
    print(f"  Reference V3-prod: FC-F1(argmax)={v3_fc_f1:.4f}  CC-F1={v3_cc_f1:.4f}  "
          f"LA-F1={v3_la_f1:.4f}  macro-F1={v3_macro:.4f}")
    print(f"  (sanity vs known V3 dedup: FC-F1~0.83 / CC-F1~0.94 / LA-F1~0.90 / macro~0.69)\n")

    if not fact_present:
        print("  GATE (1) in-domain FC-F1 >= ~0.78          : PENDING — no factorized checkpoint")
        print("  GATE (2) CC & LA macro regress <= 1-2 pts  : PENDING — no factorized checkpoint")
        print("  GATE (3) composed-loss helps vs small      : PENDING — needs both factorized ckpts")
        print("  GATE (4) UAV123 false-FC reduction (OOD)   : DEFERRED — needs v4 features on UAV123")
        print("\n  VERDICT: factorized checkpoints NOT READY. V3-prod remains production.")
        print("           Re-run this harness with --fact_small_ckpt / --fact_composed_ckpt once")
        print("           the sibling trainer (tools/train_csc_v4_factorized.py) has produced them.")
        verdict["recommendation"] = "PENDING_FACTORIZED_CHECKPOINTS"
        print(sep)
        return verdict

    # ---- Gate (1): in-domain FC-F1 >= ~0.78 (both argmax & calibrated) ----
    GATE_FC = 0.78
    g1 = {}
    print(f"  GATE (1) in-domain FC-F1 >= ~{GATE_FC:.2f} ?")
    for m in fact_present:
        fa = results[m]["fc_argmax"]["f1"]
        fc = results[m]["fc_calibrated"]["f1"]
        best = max(fa, fc)
        ok = best >= GATE_FC
        g1[m] = {"fc_f1_argmax": fa, "fc_f1_calibrated": fc, "pass": bool(ok)}
        print(f"      {m:<22} argmax={fa:.4f}  calibrated={fc:.4f}  -> {'PASS' if ok else 'FAIL'}")
    verdict["gates"]["g1_fc_f1"] = g1

    # ---- Gate (2): CC & LA (and macro) regress <= ~2 points vs V3 ----
    TOL = 0.02
    g2 = {}
    print(f"\n  GATE (2) CC & LA per-class F1 and macro-F1 regress <= ~2 pts vs V3 ?")
    for m in fact_present:
        cc = results[m]["per_class_f1"]["CC"]
        la = results[m]["per_class_f1"]["LA"]
        mac = results[m]["macro_f1"]
        d_cc = cc - v3_cc_f1
        d_la = la - v3_la_f1
        d_mac = mac - v3_macro
        ok = (d_cc >= -TOL) and (d_la >= -TOL)
        g2[m] = {
            "cc_f1": cc, "d_cc": d_cc, "la_f1": la, "d_la": d_la,
            "macro_f1": mac, "d_macro": d_mac, "pass": bool(ok),
        }
        print(f"      {m:<22} ΔCC={d_cc:+.4f}  ΔLA={d_la:+.4f}  Δmacro={d_mac:+.4f}  "
              f"-> {'PASS' if ok else 'FAIL'}")
    verdict["gates"]["g2_cc_la_regression"] = g2

    # ---- Gate (3): composed-loss helps vs factorized_small ----
    g3 = None
    print(f"\n  GATE (3) composed-loss helps vs factorized_small ?")
    if "factorized_small" in fact_present and "factorized_composed" in fact_present:
        s = results["factorized_small"]
        c = results["factorized_composed"]
        d_fc_argmax = c["fc_argmax"]["f1"] - s["fc_argmax"]["f1"]
        d_fc_cal = c["fc_calibrated"]["f1"] - s["fc_calibrated"]["f1"]
        d_fc_auroc = c["fc_vs_all"]["auroc"] - s["fc_vs_all"]["auroc"]
        d_macro = c["macro_f1"] - s["macro_f1"]
        helped = (d_fc_cal > 0) or (d_fc_argmax > 0) or (d_fc_auroc > 0)
        g3 = {
            "d_fc_f1_argmax": d_fc_argmax, "d_fc_f1_calibrated": d_fc_cal,
            "d_fc_vs_all_auroc": d_fc_auroc, "d_macro_f1": d_macro, "helped": bool(helped),
        }
        print(f"      composed - small:  ΔFC-F1(argmax)={d_fc_argmax:+.4f}  "
              f"ΔFC-F1(cal)={d_fc_cal:+.4f}  ΔFC-vs-ALL-AUROC={d_fc_auroc:+.4f}  "
              f"Δmacro={d_macro:+.4f}  -> {'HELPED' if helped else 'NO HELP'}")
    else:
        print("      PENDING — needs BOTH factorized_small and factorized_composed checkpoints.")
    verdict["gates"]["g3_composed_helps"] = g3

    # ---- Gate (4): UAV123 false-FC reduction — DEFERRED ----
    print(f"\n  GATE (4) UAV123 false-FC reduction (true OOD) : DEFERRED / PENDING")
    print("      Requires extracting v4 full-telemetry features on UAV123 first (not available).")
    verdict["gates"]["g4_uav123_false_fc"] = "PENDING_UAV123_FEATURE_EXTRACTION"

    # ---- Overall recommendation ----
    print("\n  " + "-" * 74)
    in_domain_pass = {}
    for m in fact_present:
        in_domain_pass[m] = bool(g1[m]["pass"] and g2[m]["pass"])
    any_challenger_passes = any(in_domain_pass.values())
    best_challenger = None
    if any_challenger_passes:
        # prefer composed if it both passes and helped; else any passing
        cand = [m for m in fact_present if in_domain_pass[m]]
        if "factorized_composed" in cand and (g3 and g3["helped"]):
            best_challenger = "factorized_composed"
        else:
            best_challenger = cand[0]

    if any_challenger_passes:
        print(f"  VERDICT: CHALLENGER MEETS IN-DOMAIN GATES -> proceed to UAV123 OOD test.")
        print(f"           Recommended challenger: {best_challenger}")
        print(f"           (in-domain gates 1+2 PASS: {[m for m in fact_present if in_domain_pass[m]]})")
        print(f"           V3-prod stays production until the UAV123 OOD false-FC reduction (gate 4)")
        print(f"           is measured and confirms the reduction.")
        verdict["recommendation"] = f"PROCEED_TO_UAV123:{best_challenger}"
    else:
        print(f"  VERDICT: NO challenger meets the in-domain gates.")
        print(f"           V3-prod REMAINS production. Report factorization as a MOTIVATED ABLATION")
        print(f"           (the 2x2 axis structure / anti-shortcut FC-vs-CC story) rather than the")
        print(f"           production model. Do NOT proceed to the UAV123 OOD test on this evidence.")
        verdict["recommendation"] = "V3_REMAINS_PRODUCTION_FACTORIZATION_AS_ABLATION"
    verdict["in_domain_pass"] = in_domain_pass
    print(sep)
    return verdict


def print_next_commands(args, fact_present: list[str]) -> None:
    """Print the exact LODO + UAV123-extraction commands to run next."""
    sep = "=" * 78
    print("\n" + sep)
    print("NEXT COMMANDS (LODO folds + UAV123 OOD extraction)")
    print(sep)
    print("""
LEAVE-ONE-DATASET-OUT (LODO) — true held-out-dataset generalization.
Each fold RETRAINS the factorized model EXCLUDING one dataset, then evaluates on
the held-out dataset's frames. NOT run here (needs the sibling trainer + a retrain
per fold). The 5 datasets in the train pool are: dtb70 got10k lasot uavdt_sot
visdrone_sot. Per fold (example: hold out lasot):

  # 1) retrain excluding 'lasot' (sibling trainer; --exclude_dataset is the LODO hook)
  .venv/bin/python tools/train_csc_v4_factorized.py \\
      --shards outputs/csc_labels_v4/train_shards.jsonl \\
      --exclude_dataset lasot \\
      --out_dir outputs/csc_training_v4/lodo_lasot_small \\
      --seed 42
  # (repeat with the composed-loss flag, e.g. --lambda_composed 1.0, for the composed variant)

  # 2) evaluate that fold ON the held-out dataset only:
  .venv/bin/python tools/eval_factorized_compare.py \\
      --fact_small_ckpt outputs/csc_training_v4/lodo_lasot_small/checkpoint_best.pth \\
      --eval_only_dataset lasot --lodo

Repeat for: dtb70, got10k, uavdt_sot, visdrone_sot. Aggregate the held-out-dataset
FC-F1 / macro-F1 across the 5 folds = the LODO generalization number.

UAV123 (TRUE OOD final test) — DEFERRED, requires feature extraction FIRST:

  # A) extract v4 full-telemetry features on UAV123 (NOT yet available; the v4
  #    feature pipeline must run on UAV123 telemetry to produce UAV123 shards):
  .venv/bin/python tools/v4_build_labels.py \\
      --dataset uav123 --telemetry_root <uav123 telemetry> \\
      --out outputs/csc_labels_v4/uav123_shards.jsonl
  # B) then score the frozen models on UAV123 (final-test-only; DO NOT tune on it):
  .venv/bin/python tools/eval_factorized_compare.py \\
      --shards outputs/csc_labels_v4/uav123_shards.jsonl \\
      --eval_only_dataset uav123 \\
      --fact_small_ckpt   outputs/csc_training_v4/csc_v4_fact_small/checkpoint_best.pth \\
      --fact_composed_ckpt outputs/csc_training_v4/csc_v4_fact_composed/checkpoint_best.pth
  #    (Gate 4 — UAV123 false-FC reduction — is decided ONLY from this run.)
""".rstrip())
    print(sep)


# ===========================================================================
# Self-test (synthetic; no datasets, no checkpoints)
# ===========================================================================
def _selftest_loader_roundtrip() -> None:
    """In-memory round-trip of BOTH model classes through the schema-tolerant
    loader: build -> set temps -> torch.save a trainer-style ckpt dict -> reload
    via _load_factorized -> assert type detection + temps + that the loaded
    model's forward/compose reproduces the source model exactly.
    """
    import tempfile

    import torch

    from csc_lib.csc.v4.model_v4_cond_factorized import CondFactorizedCSC
    from csc_lib.csc.v4.model_v4_factorized import FactorizedCSC

    GD, RD, W = 7, 10, 32
    geom_features = [f"g{i}_pct" for i in range(GD)]
    resp_features = [f"r{i}_pct" for i in range(RD)]
    torch.manual_seed(0)
    geom = torch.randn(3, W, GD)
    resp = torch.randn(3, W, RD)

    with tempfile.TemporaryDirectory() as td:
        # --- symmetric ---
        ms = FactorizedCSC(geom_dim=GD, resp_dim=RD, hidden=32, levels=3, kernel=3)
        ms.set_temperatures(1.9, 1.3)
        ms.eval()
        ck_s = {
            "state_dict": ms.state_dict(), "geom_features": geom_features,
            "resp_features": resp_features, "geom_dim": GD, "resp_dim": RD,
            "hidden": 32, "levels": 3, "kernel": 3, "t_off": 1.9, "t_conf": 1.3,
            "lambda_composed": 0.0,
        }
        p_s = Path(td) / "sym.pth"; torch.save(ck_s, p_s)
        m2, meta = _load_factorized(p_s)
        assert meta["model_type"] == "symmetric", meta["model_type"]
        assert abs(meta["t_off"] - 1.9) < 1e-5 and abs(meta["t_conf"] - 1.3) < 1e-5
        with torch.no_grad():
            o = _forward_factorized(m2, geom, resp, "symmetric")
            ref = ms.forward(geom, resp, last_step_only=True)
            assert torch.allclose(o[0], ref["off_logit"], atol=1e-5)
            assert torch.allclose(o[1], ref["conf_logit"], atol=1e-5)
            pr = _compose_probs_symmetric(m2, o[0].numpy(), o[1].numpy(), meta["t_off"], meta["t_conf"])
        assert np.allclose(pr.sum(1), 1.0, atol=1e-5)

        # --- conditional (3 temps / 3 heads) ---
        mc = CondFactorizedCSC(geom_dim=GD, resp_dim=RD, hidden=32, levels=3, kernel=3)
        mc.set_temperatures(1.5, 0.9, 1.2)
        mc.eval()
        ck_c = {
            "state_dict": mc.state_dict(), "geom_features": geom_features,
            "resp_features": resp_features, "geom_dim": GD, "resp_dim": RD,
            "hidden": 32, "levels": 3, "kernel": 3,
            "t_off": 1.5, "t_conf_on": 0.9, "t_conf_off": 1.2,
        }
        p_c = Path(td) / "cond.pth"; torch.save(ck_c, p_c)
        m3, metac = _load_factorized(p_c)
        assert metac["model_type"] == "conditional", metac["model_type"]
        assert abs(metac["t_conf_on"] - 0.9) < 1e-5 and abs(metac["t_conf_off"] - 1.2) < 1e-5
        with torch.no_grad():
            oc = _forward_factorized(m3, geom, resp, "conditional")
            refc = mc.forward(geom, resp, last_step_only=True)
            assert torch.allclose(oc[0], refc["off_logit"], atol=1e-5)
            assert torch.allclose(oc[1], refc["conf_on_logit"], atol=1e-5)
            assert torch.allclose(oc[2], refc["conf_off_logit"], atol=1e-5)
            prc = _compose_probs_conditional(
                m3, oc[0].numpy(), oc[1].numpy(), oc[2].numpy(),
                metac["t_off"], metac["t_conf_on"], metac["t_conf_off"],
            )
        assert np.allclose(prc.sum(1), 1.0, atol=1e-5)
    print("[selftest] loader round-trip OK: symmetric + conditional detected, "
          "temps restored, forward/compose reproduced")


def run_selftest() -> int:
    """Exercise every metric + the compose/window builders on synthetic data."""
    print("=" * 70)
    print("SELF-TEST  (synthetic; no datasets / no checkpoints)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    N = 5000
    # synthetic GT with a ~1% FC base rate (matching reality)
    y_true = rng.choice([CC, CU, LA, FC], size=N, p=[0.74, 0.04, 0.21, 0.01])

    # A "good" model: probs correlated with GT.
    def make_probs(skill: float) -> np.ndarray:
        logits = rng.normal(size=(N, 4)) * (1.0 - skill)
        logits[np.arange(N), y_true] += skill * 4.0
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    probs_good = make_probs(0.9)
    probs_meh = make_probs(0.4)

    m_good = compute_model_metrics(y_true, probs_good)
    m_meh = compute_model_metrics(y_true, probs_meh)
    assert m_good["macro_f1"] > m_meh["macro_f1"], "skill ordering violated"
    assert 0.0 <= m_good["fc_vs_all"]["auroc"] <= 1.0
    assert m_good["fc_vs_all"]["auroc"] > 0.7, m_good["fc_vs_all"]["auroc"]
    # AUROC sanity: perfect separation -> ~1.0
    perfect = np.zeros(N); perfect[y_true == FC] = 1.0
    assert abs(_auroc(y_true == FC, perfect + rng.normal(scale=1e-6, size=N)) - 1.0) < 1e-3
    # threshold sweep returns a valid F1
    assert 0.0 <= m_good["fc_calibrated"]["f1"] <= 1.0
    print(f"[selftest] metrics OK: good macro={m_good['macro_f1']:.3f} "
          f"FC-F1={m_good['fc_argmax']['f1']:.3f} FC-vs-ALL-AUROC={m_good['fc_vs_all']['auroc']:.3f}")
    print(f"[selftest]            meh  macro={m_meh['macro_f1']:.3f} "
          f"FC-F1={m_meh['fc_argmax']['f1']:.3f}")

    # per-dataset + per-seq
    datasets = rng.choice(["dtb70", "lasot", "got10k"], size=N)
    seq_ids = rng.integers(0, 30, size=N)
    pd = compute_per_dataset(y_true, probs_good, datasets)
    assert set(pd) == {"dtb70", "lasot", "got10k"}, pd.keys()
    ps = compute_per_sequence_fc_f1(y_true, probs_good, seq_ids)
    assert ps["n_seq_scored"] >= 1
    print(f"[selftest] per-dataset keys={sorted(pd)}  per-seq mean FC-F1={ps['mean_fc_f1']:.3f}")

    # compose product rule (no model.compose) — uses a tiny stub object.
    class _Stub:  # no compose() -> exercises the fallback product rule
        pass

    off = rng.normal(size=64); conf = rng.normal(size=64)
    p = _compose_probs_symmetric(_Stub(), off, conf, t_off=1.3, t_conf=0.8)
    assert p.shape == (64, 4)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6), "composed posterior must sum to 1"
    # equivalence check of the product rule by hand
    import torch
    p_off = torch.sigmoid(torch.tensor(off) / 1.3).numpy()
    p_conf = torch.sigmoid(torch.tensor(conf) / 0.8).numpy()
    assert np.allclose(p[:, FC], p_off * p_conf, atol=1e-6)
    assert np.allclose(p[:, CC], (1 - p_off) * p_conf, atol=1e-6)
    print(f"[selftest] symmetric compose product-rule OK (sums to 1, FC=off*conf verified)")

    # CONDITIONAL compose product rule (3 logits, conditional confidence axes).
    on = rng.normal(size=64); off_c = rng.normal(size=64)
    pc = _compose_probs_conditional(_Stub(), off, on, off_c, t_off=1.5, t_conf_on=0.9, t_conf_off=1.2)
    assert pc.shape == (64, 4)
    assert np.allclose(pc.sum(axis=1), 1.0, atol=1e-6), "conditional posterior must sum to 1"
    p_offv = torch.sigmoid(torch.tensor(off) / 1.5).numpy()
    p_on = torch.sigmoid(torch.tensor(on) / 0.9).numpy()
    p_offc = torch.sigmoid(torch.tensor(off_c) / 1.2).numpy()
    assert np.allclose(pc[:, FC], p_offv * p_offc, atol=1e-6), "FC must use conf_OFF"
    assert np.allclose(pc[:, CC], (1 - p_offv) * p_on, atol=1e-6), "CC must use conf_ON"
    print(f"[selftest] conditional compose product-rule OK (sums to 1, FC=off*conf_off, CC=(1-off)*conf_on)")

    # both real model classes round-trip through _load_factorized via an in-memory
    # checkpoint (verifies the schema-tolerant loader + type detection + temps).
    try:
        _selftest_loader_roundtrip()
    except Exception as exc:  # importable only when the sibling models exist
        print(f"[selftest] loader round-trip SKIPPED ({type(exc).__name__}: {exc})")

    # causal window builder
    feat = rng.normal(size=(10, 3)).astype(np.float32)
    w = _build_causal_windows(feat, window=4)
    assert w.shape == (10, 4, 3), w.shape
    # last step of window t must equal feat[t]
    for t in range(10):
        assert np.allclose(w[t, -1], feat[t]), t
    # early window is edge-padded with feat[0]
    assert np.allclose(w[0, 0], feat[0]) and np.allclose(w[0, -1], feat[0])
    print(f"[selftest] causal window builder OK: (T,W,F)={w.shape}, last-step==frame, edge-pad OK")

    print("\n[selftest] ALL CHECKS PASSED")
    print("=" * 70)
    return 0


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v3_run_dir", default=str(DEFAULT_V3_RUN))
    ap.add_argument("--v3_labels_dir", default=str(DEFAULT_V3_LABELS))
    ap.add_argument("--shards", default=str(DEFAULT_SHARDS))
    ap.add_argument("--fact_small_ckpt", default=str(DEFAULT_FACT_SMALL))
    ap.add_argument("--fact_composed_ckpt", default=str(DEFAULT_FACT_COMPOSED))
    ap.add_argument("--cond_ckpt", action="append", default=None,
                    help="conditional CondFactorizedCSC checkpoint path (repeatable; "
                         "auto-discovers outputs/csc_training_v4/cond_fact_seed{0,1,2} if omitted)")
    ap.add_argument("--no_ckpt_temps", action="store_true",
                    help="do NOT use the checkpoint's stored (calibrated) temperatures; "
                         "use raw temps=1.0 instead (never recalibrates on eval-val)")
    ap.add_argument("--window", type=int, default=32, help="causal window length for the factorized towers")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--lodo", action="store_true",
                    help="print LODO documentation prominently (does NOT run folds)")
    ap.add_argument("--eval_only_dataset", default=None,
                    help="restrict eval to a single dataset (used by LODO / UAV123 runs)")
    ap.add_argument("--skip_v3", action="store_true", help="skip the V3-prod baseline")
    ap.add_argument("--selftest", action="store_true", help="run synthetic self-test only and exit")
    args = ap.parse_args()

    if args.selftest:
        return run_selftest()

    # Always self-test the harness internals first (cheap, catches regressions).
    print(">>> harness internal self-test ...")
    run_selftest()

    from csc_lib.csc.v4.features_v4 import FEATURE_NAMES_V4  # noqa: E402

    # ----------------------------------------------------------------------
    # 1) V3-prod predictions (dedup per-frame) -> defines the candidate frame set
    # ----------------------------------------------------------------------
    v3_pred: Optional[dict] = None
    val_keys: list[tuple] = []
    if not args.skip_v3:
        print("\n>>> V3-prod: loading + dedup per-frame prediction (reusing v3_fc_heldout_repro) ...")
        v3_pred = predict_v3(Path(args.v3_run_dir), Path(args.v3_labels_dir), device=args.device)
        val_keys = v3_pred["val_keys"]
        print(f"    V3 val seqs={len(val_keys)}  dedup frames={len(v3_pred['preds']):,}  "
              f"window={v3_pred['window']}")
    else:
        # Derive val_keys from split_info directly.
        si = json.loads((Path(args.v3_run_dir) / "split_info.json").read_text())
        val_keys = [tuple(x) for x in si["val_sequences"]]
        print(f"\n>>> --skip_v3: val seqs from split_info = {len(val_keys)}")

    val_keys_set = set(val_keys)

    # ----------------------------------------------------------------------
    # 2) Load shard val rows -> authoritative GT (v4 derived) + factorized input
    # ----------------------------------------------------------------------
    print(f"\n>>> loading v4 shards for val seqs: {args.shards}")
    shard_by_seq = load_shard_val(Path(args.shards), val_keys_set)
    n_shard_frames = sum(len(v) for v in shard_by_seq.values())
    print(f"    shard val seqs={len(shard_by_seq)}  frames={n_shard_frames:,}")
    if args.eval_only_dataset:
        keep = {k for k in shard_by_seq if k[0] == args.eval_only_dataset}
        shard_by_seq = {k: shard_by_seq[k] for k in keep}
        print(f"    --eval_only_dataset={args.eval_only_dataset}: restricted to "
              f"{len(shard_by_seq)} seqs / {sum(len(v) for v in shard_by_seq.values())} frames")

    # Authoritative GT + dataset/seq id per (ds,seq,pos).
    gt: dict[tuple, int] = {}
    ds_of: dict[tuple, str] = {}
    seq_of: dict[tuple, str] = {}
    for (dataset, sequence), srows in shard_by_seq.items():
        for pos, r in enumerate(srows):
            key = (dataset, sequence, pos)
            gt[key] = int(r["derived"])
            ds_of[key] = dataset
            seq_of[key] = f"{dataset}/{sequence}"

    # ----------------------------------------------------------------------
    # 3) Factorized predictions (if checkpoints present + model importable)
    # ----------------------------------------------------------------------
    fact_specs = [
        ("factorized_small", Path(args.fact_small_ckpt)),
        ("factorized_composed", Path(args.fact_composed_ckpt)),
    ]
    # Conditional challenger checkpoint(s): --cond_ckpt PATH (repeatable). Named
    # cond_<stem> (e.g. cond_fact_seed0) so multiple seeds show as separate cols.
    cond_ckpts = list(args.cond_ckpt or [])
    if not cond_ckpts:
        # auto-discover the 3 default seed dirs if they exist (still TRAINING now)
        for seed in (0, 1, 2):
            d = PROJECT_ROOT / f"outputs/csc_training_v4/cond_fact_seed{seed}" / "checkpoint_best.pth"
            if d.exists():
                cond_ckpts.append(str(d))
    cond_names = []
    for cp in cond_ckpts:
        cp = Path(cp)
        nm = f"cond_{cp.parent.name}" if cp.parent.name else f"cond_{cp.stem}"
        fact_specs.append((nm, cp))
        cond_names.append(nm)

    fact_preds: dict[str, dict] = {}
    fact_meta: dict[str, dict] = {}
    fact_blockers: dict[str, str] = {}
    for name, ckpt in fact_specs:
        if not ckpt.exists():
            msg = f"checkpoint not found: {ckpt}"
            print(f"\n>>> {name}: SKIP — {msg}")
            fact_blockers[name] = msg
            continue
        print(f"\n>>> {name}: loading {ckpt} + building geom/resp windows + composing ...")
        try:
            preds, meta = predict_factorized(
                ckpt, shard_by_seq, FEATURE_NAMES_V4, args.window,
                device=args.device, use_ckpt_temps=not args.no_ckpt_temps,
            )
            fact_preds[name] = preds
            fact_meta[name] = meta
            if meta["model_type"] == "conditional":
                tstr = (f"t_off={meta['t_off']:.3f} t_conf_on={meta['t_conf_on']:.3f} "
                        f"t_conf_off={meta['t_conf_off']:.3f}")
            else:
                tstr = f"t_off={meta['t_off']:.3f} t_conf={meta['t_conf']:.3f}"
            print(f"    {name}: type={meta['model_type']} params={meta['n_params']:,}  {tstr}  "
                  f"lambda_composed={meta['lambda_composed']}  use_ckpt_temps={meta['use_ckpt_temps']}  "
                  f"geom_dim={len(meta['geom_features'])} resp_dim={len(meta['resp_features'])}")
        except Exception as exc:
            print(f"    {name}: BLOCKED — {exc}")
            fact_blockers[name] = str(exc)

    # ----------------------------------------------------------------------
    # 4) Common dedup frame set (intersection over models that produced preds)
    # ----------------------------------------------------------------------
    pred_sources: dict[str, dict] = {}
    if v3_pred is not None:
        pred_sources["V3-prod"] = v3_pred["preds"]
    for name in fact_preds:
        pred_sources[name] = fact_preds[name]

    if not pred_sources:
        print("\nFATAL: no model produced predictions (no V3, no factorized). Nothing to compare.")
        return 2

    common_keys = set(gt.keys())
    for src in pred_sources.values():
        common_keys &= set(src.keys())
    common_keys &= set(gt.keys())
    common = sorted(common_keys)
    if not common:
        print("\nFATAL: empty common frame set across models — alignment failed.")
        return 2
    print(f"\n>>> common dedup frame set across [{', '.join(pred_sources)}]: {len(common):,} frames")

    y_true = np.array([gt[k] for k in common], dtype=np.int64)
    datasets_arr = np.array([ds_of[k] for k in common])
    seqids_arr = np.array([seq_of[k] for k in common])

    present_models = list(pred_sources.keys())
    # All factorized challengers (symmetric + conditional) that produced preds, in
    # a stable order: small, composed, then any conditional seed(s).
    fact_present = [m for m in ("factorized_small", "factorized_composed") if m in fact_preds]
    fact_present += [m for m in fact_preds if m.startswith("cond_")]

    # ----------------------------------------------------------------------
    # 5) Metrics for each model on the common frame set
    # ----------------------------------------------------------------------
    results: dict[str, dict] = {}
    per_dataset: dict[str, dict] = {}
    per_seq: dict[str, dict] = {}
    for m in present_models:
        src = pred_sources[m]
        probs = np.stack([np.asarray(src[k], dtype=np.float64) for k in common], axis=0)  # (N,4)
        # numerical guard: renormalize (V3 softmax already sums to 1; composed should too)
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / np.where(row_sums > 0, row_sums, 1.0)
        results[m] = compute_model_metrics(y_true, probs)
        per_dataset[m] = compute_per_dataset(y_true, probs, datasets_arr)
        per_seq[m] = compute_per_sequence_fc_f1(y_true, probs, seqids_arr)

    # V3 cross-check vs its OWN GT (label-source sanity), only when not restricted.
    v3_self = None
    if v3_pred is not None and not args.eval_only_dataset:
        v3_gt_self = v3_pred["gt_v3"]
        ck = [k for k in common if k in v3_gt_self]
        if ck:
            yt = np.array([v3_gt_self[k] for k in ck])
            pp = np.stack([np.asarray(v3_pred["preds"][k]) for k in ck], axis=0)
            v3_self = compute_model_metrics(yt, pp)

    # ----------------------------------------------------------------------
    # 6) Reports
    # ----------------------------------------------------------------------
    print_comparison_table(results, present_models)

    if v3_self is not None:
        print("\n[V3 cross-check vs its OWN v3fix GT (label-source sanity, not the headline)]")
        print(f"    FC-F1(argmax)={v3_self['fc_argmax']['f1']:.4f}  "
              f"CC-F1={v3_self['per_class_f1']['CC']:.4f}  "
              f"LA-F1={v3_self['per_class_f1']['LA']:.4f}  "
              f"macro-F1={v3_self['macro_f1']:.4f}  "
              f"(headline uses v4-shard GT; labels agree ~99.4%)")

    print("\n-- per-sequence mean FC-F1 (over val sequences) --")
    for m in present_models:
        ps = per_seq[m]
        print(f"    {m:<22} mean FC-F1={ps['mean_fc_f1']:.4f}  "
              f"(scored {ps['n_seq_scored']} seqs, {ps['n_seq_with_fc_gt']} with FC GT)")

    print_per_dataset(per_dataset, present_models)

    verdict = print_go_no_go(results, per_seq, present_models, fact_present)

    if args.lodo or fact_present:
        print_next_commands(args, fact_present)
    else:
        # still surface the commands even in the V3-only/blocked case
        print_next_commands(args, fact_present)

    # ----------------------------------------------------------------------
    # 7) Persist machine-readable metrics
    # ----------------------------------------------------------------------
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

    payload = {
        "eval_frames": len(common),
        "val_sequences": len(val_keys),
        "gt_source": "v4_shard_derived",
        "eval_only_dataset": args.eval_only_dataset,
        "window": args.window,
        "present_models": present_models,
        "factorized_present": fact_present,
        "factorized_blockers": fact_blockers,
        "factorized_meta": fact_meta,
        "results": results,
        "v3_self_crosscheck": v3_self,
        "per_dataset": per_dataset,
        "per_sequence_fc_f1": per_seq,
        "go_no_go": verdict,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonsafe(payload), indent=2))
    print(f"\nsaved metrics -> {out_path}")

    if fact_blockers and not fact_present:
        print("\nBLOCKER: factorized checkpoints not ready — ran V3 + self-tested the harness only.")
        for n, m in fact_blockers.items():
            print(f"    {n}: {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
