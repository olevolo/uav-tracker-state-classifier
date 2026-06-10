"""Integrity reproduction of V3-prod FALSE_CONFIRMED eval on a TRUE held-out split.

Question under audit
--------------------
The frozen V3-prod model
  outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2
reports (val_metrics.json) derived FALSE_CONFIRMED F1=0.785 / recall=0.970 /
precision=0.659 over ``n_eval=1,867,392`` frames, while split_info.json shows
only 58 val sequences / 58,356 val *windows*.  Suspicion: in-sample inflation.

What this script does (REUSES the exact training pipeline — no reimplementation)
--------------------------------------------------------------------------------
* Loads the V3 checkpoint + builds CSCTCN via build_model(cfg) from the resolved
  config (feature_version v2, 16-dim, window 32, tcn, n_states 4, forecast heads).
* Loads labels with load_labels_dir, groups with _group_by_sequence, reproduces
  the held-out split with split_sequences_stratified (stratified_split=True) and
  ASSERTS it equals split_info.json's val_sequences.
* Builds features with CSCDataset (which dispatches to build_sequence_features_v2)
  — same windows, same feature builder, same image_size as training.
* Computes the derived 4-way FALSE_CONFIRMED metrics + FC-vs-ALL AUROC/AUPRC
  three ways:
    (A) WINDOW-FLATTENED held-out val  — EXACTLY as tools/train_csc.py::_evaluate
        (every frame counted in all overlapping windows; reproduces n_eval).
    (B) DEDUP per-frame held-out val   — one prediction per unique sequence-frame
        (each frame scored from the window where it is the LAST/causal step).
        This is the honest generalizing number.
    (C) WINDOW-FLATTENED full data     — train+val, same as (A) over all seqs,
        to test whether 0.785 is reproduced by the FULL dataset.

Metrics use the SAME helpers train_csc.py used: per_state_prf / macro_f1 /
failure_auroc / failure_auprc from scene_state_metrics.

Run:
  .venv/bin/python tools/v3_fc_heldout_repro.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from csc_lib.csc.config import CSCTrainConfig  # noqa: E402
from csc_lib.csc.dataset import (  # noqa: E402
    CSCDataset,
    _group_by_sequence,
    load_labels_dir,
    split_sequences_stratified,
)
from csc_lib.csc.labeling.label_schema import (  # noqa: E402
    DERIVED_NAMES,
    NUM_DERIVED_STATES,
    DerivedState,
)
from csc_lib.csc.model import build_model  # noqa: E402
from csc_lib.eval.custom_metrics.scene_state_metrics import (  # noqa: E402
    failure_auprc,
    failure_auroc,
    macro_f1,
    per_state_prf,
)

RUN_DIR = PROJECT_ROOT / "outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2"
CKPT = RUN_DIR / "checkpoint_best.pth"
CONFIG = RUN_DIR / "config_resolved.yaml"
SPLIT_INFO = RUN_DIR / "split_info.json"
VAL_METRICS = RUN_DIR / "val_metrics.json"
LABELS_DIR = PROJECT_ROOT / "outputs/csc_labels/sglatrack/v3fix_combined"

FC = int(DerivedState.FALSE_CONFIRMED)  # 3
LA = int(DerivedState.LOST_AWARE)       # 2
IMAGE_SIZE = (1280, 720)                # CSCDataset training default


# ---------------------------------------------------------------------------
def load_config() -> CSCTrainConfig:
    import yaml

    with open(CONFIG) as fh:
        d = yaml.safe_load(fh)
    return CSCTrainConfig.from_dict(d)


def load_model(cfg: CSCTrainConfig) -> torch.nn.Module:
    model = build_model(cfg.model)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=True)
    assert not missing and not unexpected
    model.eval()
    return model


def _predict_dataset(model: torch.nn.Module, ds: CSCDataset, device: str = "cpu"):
    """Run model over a dataset window-by-window.

    Returns three parallel python lists, ONE ENTRY PER WINDOW:
        der_pred_win[i]  : (T,) argmax of 4-way derived softmax for window i
        der_true_win[i]  : (T,) ground-truth derived label for window i
        risk_win[i]      : (T,) P(LOST_AWARE)+P(FALSE_CONFIRMED) for window i
        fcprob_win[i]    : (T,) P(FALSE_CONFIRMED) from the 4-way softmax
    plus per-window (dataset, sequence, frame_end) for dedup bookkeeping.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=256, shuffle=False)
    der_pred_win: list[np.ndarray] = []
    der_true_win: list[np.ndarray] = []
    risk_win: list[np.ndarray] = []
    fcprob_win: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(device)
            der_y = batch["derived"]
            out = model(x)
            der_probs = torch.softmax(out.derived_logits, dim=-1)  # (B,T,4)
            der_pred = der_probs.argmax(dim=-1).cpu().numpy()       # (B,T)
            risk = (der_probs[..., LA] + der_probs[..., FC]).cpu().numpy()
            fcp = der_probs[..., FC].cpu().numpy()
            for b in range(der_pred.shape[0]):
                der_pred_win.append(der_pred[b])
                der_true_win.append(der_y[b].numpy())
                risk_win.append(risk[b])
                fcprob_win.append(fcp[b])
    return der_pred_win, der_true_win, risk_win, fcprob_win


def _fc_block(dt: np.ndarray, dp: np.ndarray, fcprob: np.ndarray) -> dict:
    """Compute the same numbers val_metrics.json reports, for FC + macro."""
    per = per_state_prf(dt, dp, n_states=NUM_DERIVED_STATES, state_names=list(DERIVED_NAMES))
    fc = per["FALSE_CONFIRMED"]
    # FC-vs-ALL one-vs-rest AUROC / AUPRC using P(FC) as score.
    y_fc = (dt == FC).astype(np.int8)
    if 0 < int(y_fc.sum()) < y_fc.size:
        auroc = failure_auroc(y_fc, fcprob)
        auprc = failure_auprc(y_fc, fcprob)
    else:
        auroc = auprc = float("nan")
    return {
        "fc_precision": fc["precision"],
        "fc_recall": fc["recall"],
        "fc_f1": fc["f1"],
        "fc_support": fc["support"],
        "fc_vs_all_auroc": auroc,
        "fc_vs_all_auprc": auprc,
        "derived_macro_f1": macro_f1(dt, dp, n_states=NUM_DERIVED_STATES, state_names=list(DERIVED_NAMES)),
        "n_eval": int(dp.size),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    cfg = load_config()
    print(f"[cfg] feature_version={cfg.feature.feature_version} "
          f"window={cfg.feature.window_size} feat_dim={cfg.model.feature_dim} "
          f"kind={cfg.model.kind} n_states={cfg.model.n_states} "
          f"forecast={cfg.model.enable_forecast_heads} stratified={cfg.stratified_split}")
    assert cfg.feature.feature_version == "v2"
    assert cfg.feature.window_size == 32
    assert cfg.model.kind == "tcn"

    model = load_model(cfg)
    print(f"[model] loaded {type(model).__name__}  params={model.num_params:,}")

    rows = load_labels_dir(LABELS_DIR)
    groups = _group_by_sequence(rows)
    print(f"[data] {len(rows):,} rows  {len(groups)} sequences")

    # Reproduce the held-out split and ASSERT identity with split_info.json.
    train_keys, val_keys = split_sequences_stratified(
        list(groups.keys()), groups, cfg.val_fraction
    )
    si = json.loads(SPLIT_INFO.read_text())
    si_val = set(tuple(x) for x in si["val_sequences"])
    si_train = set(tuple(x) for x in si["train_sequences"])
    assert set(val_keys) == si_val, "VAL split does NOT match split_info.json!"
    assert set(train_keys) == si_train, "TRAIN split does NOT match split_info.json!"
    print(f"[split] reproduced EXACTLY: train={len(train_keys)} val={len(val_keys)} "
          f"(matches split_info.json)")

    # ---- Build datasets (same builder as training) ----
    val_rows = {k: groups[k] for k in val_keys}
    all_rows = dict(groups)
    val_ds = CSCDataset(val_rows, cfg.feature, image_size=IMAGE_SIZE)
    all_ds = CSCDataset(all_rows, cfg.feature, image_size=IMAGE_SIZE)
    print(f"[windows] val_windows={len(val_ds):,} "
          f"(val_windows*W={len(val_ds)*cfg.feature.window_size:,})  "
          f"all_windows={len(all_ds):,}")

    # =====================================================================
    # (A) WINDOW-FLATTENED held-out val — replicate train_csc.py::_evaluate
    # =====================================================================
    dpw, dtw, rkw, fcpw = _predict_dataset(model, val_ds)
    dp_A = np.concatenate(dpw)
    dt_A = np.concatenate(dtw)
    fcp_A = np.concatenate(fcpw)
    rk_A = np.concatenate(rkw)
    block_A = _fc_block(dt_A, dp_A, fcp_A)
    # also failure-AUROC the way val_metrics did (risk = P(LA)+P(FC), pos = state>=LA)
    tf_A = (dt_A >= LA).astype(np.int8)
    block_A["failure_auroc_paperstyle"] = failure_auroc(tf_A, rk_A)
    block_A["failure_auprc_paperstyle"] = failure_auprc(tf_A, rk_A)

    # =====================================================================
    # (B) DEDUP per-frame held-out val — one causal prediction per frame.
    # Each unique (dataset, sequence, frame_idx) appears in up to W windows;
    # the causal/honest prediction is from the window where that frame is the
    # LAST step (matches runtime last_step_only behaviour).  The first W-1
    # frames of each sequence never get a full-window causal prediction, so
    # for those we take their prediction from the first (earliest-end) window
    # in which they appear — i.e. position == frame_idx within window 0.
    # =====================================================================
    # Rebuild per (dataset,sequence): the window builder strides by 1 with
    # end in [W, T]; window k (0-indexed) covers frames [k, k+W-1] and its
    # LAST step is global frame (k+W-1).  So:
    #   - global frame f >= W-1  -> last step of window (f-(W-1))
    #   - global frame f <  W-1  -> step f of window 0 (its only appearance
    #                               before any later window starts; still
    #                               causal: window 0 covers [0,W-1]).
    W = cfg.feature.window_size
    # Map each val sequence to its window index range inside val_ds.windows.
    # CSCDataset preserves per-sequence contiguity in insertion order.
    dedup_pred: list[int] = []
    dedup_true: list[int] = []
    dedup_fcp: list[float] = []
    wi = 0  # running window index into the flat lists dpw/dtw/...
    for (dataset, sequence), srows in val_rows.items():
        T = len(srows)
        n_win = max(0, (T - W) // 1 + 1) if T >= W else 0
        if n_win == 0:
            wi += 0
            continue
        seq_pred = dpw[wi:wi + n_win]
        seq_true = dtw[wi:wi + n_win]
        seq_fcp = fcpw[wi:wi + n_win]
        # Ground-truth derived per global frame (from labels, authoritative).
        gt_frame = np.array([int(r.get("derived_state", 0)) for r in srows], dtype=np.int64)
        for f in range(T):
            if f <= W - 1:
                # appears as step f of window 0
                p = int(seq_pred[0][f])
                fc = float(seq_fcp[0][f])
            else:
                k = f - (W - 1)          # window whose LAST step is frame f
                p = int(seq_pred[k][W - 1])
                fc = float(seq_fcp[k][W - 1])
            dedup_pred.append(p)
            dedup_true.append(int(gt_frame[f]))
            dedup_fcp.append(fc)
        wi += n_win
    dp_B = np.asarray(dedup_pred, dtype=np.int64)
    dt_B = np.asarray(dedup_true, dtype=np.int64)
    fcp_B = np.asarray(dedup_fcp, dtype=np.float64)
    block_B = _fc_block(dt_B, dp_B, fcp_B)

    # =====================================================================
    # (C) WINDOW-FLATTENED full data (train+val) — same recipe as (A).
    # =====================================================================
    dpw2, dtw2, rkw2, fcpw2 = _predict_dataset(model, all_ds)
    dp_C = np.concatenate(dpw2)
    dt_C = np.concatenate(dtw2)
    fcp_C = np.concatenate(fcpw2)
    block_C = _fc_block(dt_C, dp_C, fcp_C)

    # ---- Reported numbers from val_metrics.json (for column 1) ----
    vm = json.loads(VAL_METRICS.read_text())
    fc_rep = vm["derived_per_state"]["FALSE_CONFIRMED"]
    rep = {
        "fc_precision": fc_rep["precision"],
        "fc_recall": fc_rep["recall"],
        "fc_f1": fc_rep["f1"],
        "fc_support": fc_rep["support"],
        "fc_vs_all_auroc": float("nan"),  # not reported as 1-vs-rest in val_metrics
        "fc_vs_all_auprc": float("nan"),
        "derived_macro_f1": vm["derived_macro_f1"],
        "n_eval": vm["n_eval"],
    }

    # ---- Pretty 3-column table ----
    def fmt(x):
        if isinstance(x, float):
            if x != x:  # nan
                return "    n/a"
            return f"{x:7.4f}"
        return f"{x:>7}"

    rows_order = [
        ("FC precision", "fc_precision"),
        ("FC recall", "fc_recall"),
        ("FC F1", "fc_f1"),
        ("FC support", "fc_support"),
        ("FC-vs-ALL AUROC", "fc_vs_all_auroc"),
        ("FC-vs-ALL AUPRC", "fc_vs_all_auprc"),
        ("derived macro-F1", "derived_macro_f1"),
        ("n_eval", "n_eval"),
    ]
    print("\n" + "=" * 84)
    print("V3-prod FALSE_CONFIRMED (derived 4-way head, argmax) — INTEGRITY REPRODUCTION")
    print("=" * 84)
    hdr = f"{'metric':<20}{'reported':>11}{'heldout-val(A win)':>20}{'heldout-val(B dedup)':>22}{'full(C win)':>13}"
    print(hdr)
    print("-" * len(hdr))
    for label, key in rows_order:
        print(f"{label:<20}{fmt(rep[key]):>11}{fmt(block_A[key]):>20}{fmt(block_B[key]):>22}{fmt(block_C[key]):>13}")

    print("\n[context] forecast head false_confirmed_next_10 (from val_metrics.json, "
          "held-out, window-flattened):")
    print(f"          AUROC={vm.get('false_confirmed_next_10_auroc')}  "
          f"AUPRC={vm.get('false_confirmed_next_10_auprc')}  "
          f"n={vm.get('false_confirmed_next_10_n')}  "
          f"pos_rate={vm.get('false_confirmed_next_10_pos_rate')}")

    # ---- Verdict diagnostics ----
    print("\n" + "=" * 84)
    print("VERDICT DIAGNOSTICS")
    print("=" * 84)
    print(f"  n_eval reported          : {rep['n_eval']:,}")
    print(f"  val_windows * window(32) : {len(val_ds) * W:,}")
    print(f"  -> reported n_eval == val_windows*32 ? "
          f"{rep['n_eval'] == len(val_ds) * W}")
    print(f"  (A) window-flattened HELD-OUT-VAL FC-F1 = {block_A['fc_f1']:.4f}  "
          f"(reproduces reported {rep['fc_f1']:.4f}? "
          f"{abs(block_A['fc_f1'] - rep['fc_f1']) < 0.01})")
    print(f"  (B) DEDUP per-frame HELD-OUT-VAL FC-F1   = {block_B['fc_f1']:.4f}  "
          f"<-- REAL generalizing held-out number")
    print(f"  (C) window-flattened FULL-DATA FC-F1     = {block_C['fc_f1']:.4f}")
    print(f"      FC support: A(win-val)={block_A['fc_support']:,}  "
          f"B(dedup-val)={block_B['fc_support']:,}  C(win-full)={block_C['fc_support']:,}")


if __name__ == "__main__":
    main()
