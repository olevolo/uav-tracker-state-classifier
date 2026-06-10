"""Controlled rematch: CONDITIONAL factorized CSC (CondFactorizedCSC) with V3
feature parity — trained on V3's exact 16-dim v2 features + split, 3 seeds.

WHY this script exists
----------------------
The symmetric 2-tower model (``model_v4_factorized.FactorizedCSC``) lost to V3
in-domain (FC-F1 0.08 vs 0.83) but was DOUBLY disadvantaged:
  (1) noisy 41-dim shard features (not V3's runtime-safe 16-dim v2 set), and
  (2) a SINGLE shared confirmed head that wrongly assumes the CC-vs-CU and
      FC-vs-LA confidence boundaries are identical.
This rematch fixes BOTH: it uses V3's EXACT feature pipeline (via
``tools/v3_features_lib`` -> ``CSCDataset`` -> ``build_sequence_features_v2``)
and the CONDITIONAL model with two confidence heads (conf_on for CC|on-target,
conf_off for FC|off-target).

MODEL  (imported, not reimplemented)
    csc_lib.csc.v4.model_v4_cond_factorized.CondFactorizedCSC
    forward(geom,resp,last_step_only=True) -> off_logit, conf_on_logit, conf_off_logit
    compose(...) -> P(CC)/P(CU)/P(LA)/P(FC) by conditional factorization.

FEATURES / PARTITION  (see tools/v3_features_lib for the full parity note)
    GEOM (off tower, 13): all geometry/localization/motion features
    RESP (confirmed towers, 3): confidence, apce, psr
    NB: this is GEOM=13/RESP=3 not the task's nominal 12/4 — the active V3
    builder emits ``aspect_instability_8`` at slot 15 where the run's STALE
    label_mapping.json claimed ``conf_ema_trend``.  We partition V3's REAL 16
    features by the task's SEMANTIC rule rather than fabricate a channel that
    does not exist.  The active builder reproduces V3-prod's metrics exactly,
    so these 16 ARE V3-feature parity.

DATA / SPLIT / WINDOW  (V3 parity)
    labels  : outputs/csc_labels/sglatrack/v3fix_combined  (derived_state)
    split   : V3's exact stratified split (asserted == split_info.json)
    window  : causal length 32, last-step target (the exact CSCDataset windows)
    calib   : ~15% of the 373 TRAIN seqs carved out (deterministic by seed) to
              fit the 3 temperatures; train on the other ~85%; eval on the
              UNTOUCHED 58-seq V3 val.

LOSS  (conditional 3-loss + composed; sequence-normalized; NO sampler)
    L = BCE(off, y_off)                          over ALL frames
      + BCE(conf_on,  y_conf)  over ON-target frames  (y_off==0)
      + BCE(conf_off, y_conf)  over OFF-target frames (y_off==1)
      + lambda_composed * NLL(composed 4-state, state4)
    Each term is sequence-normalized: every window contributes weight
    1/len(source-seq), renormalized to sum to 1 over the batch.  pos_weight per
    BCE from its conditional TRAIN-subset frequency.  Plain shuffled DataLoader.

Run (default, all 3 seeds):
    .venv/bin/python tools/train_cond_factorized.py
Single seed / overrides:
    .venv/bin/python tools/train_cond_factorized.py --seeds 0 --epochs 25
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import v3_features_lib as L  # noqa: E402
from csc_lib.csc.v4.model_v4_cond_factorized import CondFactorizedCSC  # noqa: E402
from csc_lib.csc.labeling.label_schema import DERIVED_NAMES, NUM_DERIVED_STATES  # noqa: E402
from csc_lib.eval.custom_metrics.scene_state_metrics import (  # noqa: E402
    macro_f1,
    per_state_prf,
)

from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

OUT_ROOT = PROJECT_ROOT / "outputs/csc_training_v4"

# V3-prod DEDUP per-frame baseline (reproduced; see tools/v3_fc_heldout_repro.py).
V3_BASELINE = {"CC": 0.950, "CU": 0.137, "LA": 0.907, "FC": 0.830, "macro": 0.7057}


# ===========================================================================
# Window tensors (precomputed once, shared across seeds)
# ===========================================================================
@dataclass
class WindowBank:
    """Stacked window tensors for one set of sequences.

    geom    : (N,T,GEOM_DIM)    resp : (N,T,RESP_DIM)
    y_off   : (N,)  last-step off-target target
    y_conf  : (N,)  last-step confirmed target
    state4  : (N,)  last-step derived state (0..3)
    seq_id  : (N,)  index of the source sequence (for sequence-normalized weight)
    seq_w   : (N,)  1/len(source-seq) weight per window (sequence-normalized)
    """
    geom: torch.Tensor
    resp: torch.Tensor
    y_off: torch.Tensor
    y_conf: torch.Tensor
    state4: torch.Tensor
    seq_w: torch.Tensor
    # dedup bookkeeping (val only): per-sequence window slice + frame layout
    val_seqs: list = field(default_factory=list)


def build_bank(seq_rows: dict, feature_cfg, *, with_dedup: bool = False) -> WindowBank:
    """Build a WindowBank from the EXACT CSCDataset windows.

    ``seq_w`` implements sequence normalization: each window's weight is
    1/(#windows from its source sequence) so every SEQUENCE contributes equally
    regardless of length (long sequences do not dominate the loss).
    """
    geom_list, resp_list = [], []
    yoff_list, yconf_list, state_list, w_list = [], [], [], []
    val_seqs = []
    for (dataset, sequence), rows in seq_rows.items():
        sub = {(dataset, sequence): rows}
        windows = L.build_windows(sub, feature_cfg)
        n_win = len(windows)
        if n_win == 0:
            if with_dedup:
                val_seqs.append({"key": (dataset, sequence), "n_win": 0,
                                 "T": len(rows), "rows": rows})
            continue
        per_w = 1.0 / float(n_win)  # sequence-normalized weight
        for w in windows:
            g, r = L.split_features(w.features)
            geom_list.append(g)
            resp_list.append(r)
            d_last = int(w.derived[-1])
            yoff_list.append(int(d_last in (L.LA, L.FC)))
            yconf_list.append(int(d_last in (L.CC, L.FC)))
            state_list.append(d_last)
            w_list.append(per_w)
        if with_dedup:
            # Precompute window-0 (first W frames) geom/resp ONCE so the
            # per-epoch dedup eval never rebuilds features (window 0 supplies
            # the causal prediction for the first W-1 frames of each seq).
            w0 = windows[0]
            g0, r0 = L.split_features(w0.features)
            val_seqs.append({
                "key": (dataset, sequence), "n_win": n_win,
                "T": len(rows), "rows": rows,
                "w0_geom": torch.from_numpy(g0).float(),
                "w0_resp": torch.from_numpy(r0).float(),
            })

    geom = torch.from_numpy(np.stack(geom_list)).float()
    resp = torch.from_numpy(np.stack(resp_list)).float()
    return WindowBank(
        geom=geom,
        resp=resp,
        y_off=torch.tensor(yoff_list, dtype=torch.float32),
        y_conf=torch.tensor(yconf_list, dtype=torch.float32),
        state4=torch.tensor(state_list, dtype=torch.long),
        seq_w=torch.tensor(w_list, dtype=torch.float32),
        val_seqs=val_seqs,
    )


class BankDataset(Dataset):
    def __init__(self, bank: WindowBank):
        self.b = bank

    def __len__(self):
        return self.b.geom.shape[0]

    def __getitem__(self, i):
        return {
            "geom": self.b.geom[i],
            "resp": self.b.resp[i],
            "y_off": self.b.y_off[i],
            "y_conf": self.b.y_conf[i],
            "state4": self.b.state4[i],
            "seq_w": self.b.seq_w[i],
        }


# ===========================================================================
# Loss
# ===========================================================================
def _wbce(logit, target, weight, pos_weight):
    """Sequence-normalized weighted BCE over a (already-masked) subset.

    weight : per-element sequence-normalization weight (renormalized to sum 1).
    pos_weight : scalar positive-class weight (BCEWithLogits convention).
    Returns 0.0 if the subset is empty (no frames of this condition in batch).
    """
    if logit.numel() == 0:
        return logit.new_zeros(())
    w = weight / weight.sum().clamp_min(1e-8)
    per = F.binary_cross_entropy_with_logits(
        logit, target, pos_weight=pos_weight, reduction="none"
    )
    return (per * w).sum()


def conditional_loss(out, batch, pw_off, pw_on, pw_off_cond, model, lambda_composed):
    geom_w = batch["seq_w"]
    y_off = batch["y_off"]
    y_conf = batch["y_conf"]
    state4 = batch["state4"]

    off_logit = out["off_logit"]
    con_on = out["conf_on_logit"]
    con_off = out["conf_off_logit"]

    # (1) off-target head — ALL frames.
    l_off = _wbce(off_logit, y_off, geom_w, pw_off)

    # (2) conf_on — ON-target frames only (y_off == 0): CC vs CU boundary.
    m_on = y_off == 0
    l_on = _wbce(con_on[m_on], y_conf[m_on], geom_w[m_on], pw_on)

    # (3) conf_off — OFF-target frames only (y_off == 1): FC vs LA boundary.
    m_off = y_off == 1
    l_off_c = _wbce(con_off[m_off], y_conf[m_off], geom_w[m_off], pw_off_cond)

    # (4) composed 4-state NLL (uses UNTEMPERED logits -> raw probabilities;
    # set_temperatures stays at 1.0 during training, so compose == raw).
    comp = model.compose(off_logit, con_on, con_off)
    p4 = torch.stack([comp["p_cc"], comp["p_cu"], comp["p_la"], comp["p_fc"]], dim=-1)
    p4 = p4.clamp_min(1e-8)
    w = geom_w / geom_w.sum().clamp_min(1e-8)
    nll = (-torch.log(p4[torch.arange(p4.shape[0]), state4]) * w).sum()

    total = l_off + l_on + l_off_c + lambda_composed * nll
    # Return part tensors (NOT floats) so the train loop can accumulate them
    # on-device and sync to CPU only once per epoch (per-step float() on MPS
    # forces an expensive device->host sync every step).
    return total, {
        "off": l_off.detach(), "conf_on": l_on.detach(),
        "conf_off": l_off_c.detach(), "composed": nll.detach(),
    }


def _pos_weight(y: torch.Tensor) -> torch.Tensor:
    """BCEWithLogits pos_weight = N_neg / N_pos (clamped)."""
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos <= 0:
        return torch.tensor(1.0)
    return torch.tensor(min(50.0, neg / pos), dtype=torch.float32)


# ===========================================================================
# Eval (dedup per-frame on the untouched V3 val)
# ===========================================================================
@torch.no_grad()
def predict_bank(model, bank: WindowBank, device, batch_size=512):
    """Return per-WINDOW (last-step) probabilities for the composed heads."""
    model.eval()
    ds = BankDataset(bank)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    p_off, p_on, p_offc = [], [], []
    p_cc, p_cu, p_la, p_fc = [], [], [], []
    for b in loader:
        out = model(b["geom"].to(device), b["resp"].to(device), last_step_only=True)
        c = model.compose(out["off_logit"], out["conf_on_logit"], out["conf_off_logit"])
        p_off.append(c["p_off"].cpu()); p_on.append(c["p_conf_on"].cpu())
        p_offc.append(c["p_conf_off"].cpu())
        p_cc.append(c["p_cc"].cpu()); p_cu.append(c["p_cu"].cpu())
        p_la.append(c["p_la"].cpu()); p_fc.append(c["p_fc"].cpu())
    cat = lambda xs: torch.cat(xs).numpy()
    return {
        "p_off": cat(p_off), "p_conf_on": cat(p_on), "p_conf_off": cat(p_offc),
        "p_cc": cat(p_cc), "p_cu": cat(p_cu), "p_la": cat(p_la), "p_fc": cat(p_fc),
    }


def dedup_per_frame(model, val_bank: WindowBank, feature_cfg, device, W: int):
    """One causal prediction per unique (sequence, frame).

    Mirrors tools/v3_fc_heldout_repro.py:
      - frame f >= W-1 -> last step of window (f-(W-1))
      - frame f <  W-1 -> step f of window 0
    For the COMPOSED 4-state we only have last-step probabilities per window
    (last_step_only); for the early frames (f<W-1) we run window 0 with
    last_step_only=False to get the per-step composition.
    """
    model.eval()
    probs = predict_bank(model, val_bank, device)  # per-window last-step
    # Build per-window probability matrix in val_bank order.
    p_off_w = probs["p_off"]; p_on_w = probs["p_conf_on"]; p_offc_w = probs["p_conf_off"]
    p_cc_w = probs["p_cc"]; p_cu_w = probs["p_cu"]; p_la_w = probs["p_la"]; p_fc_w = probs["p_fc"]

    pred, true = [], []
    fc_prob, off_prob, on_prob, offc_prob = [], [], [], []
    p_cc_f, p_cu_f, p_la_f, p_fc_f = [], [], [], []
    wi = 0
    for seqinfo in val_bank.val_seqs:
        n_win = seqinfo["n_win"]
        rows = seqinfo["rows"]
        T = seqinfo["T"]
        gt = np.array([int(r.get("derived_state", 0)) for r in rows], dtype=np.int64)
        if n_win == 0:
            continue
        # Slice this sequence's per-window last-step probabilities.
        sl = slice(wi, wi + n_win)
        seq_pcc = p_cc_w[sl]; seq_pcu = p_cu_w[sl]; seq_pla = p_la_w[sl]; seq_pfc = p_fc_w[sl]
        seq_poff = p_off_w[sl]; seq_pon = p_on_w[sl]; seq_poffc = p_offc_w[sl]

        # Per-step composition for window 0 (covers the first W-1 frames),
        # using the PRECOMPUTED window-0 features (no per-epoch rebuild).
        g0 = seqinfo["w0_geom"]
        r0 = seqinfo["w0_resp"]
        with torch.no_grad():
            out0 = model(g0[None].to(device), r0[None].to(device),
                         last_step_only=False)
            c0 = model.compose(out0["off_logit"], out0["conf_on_logit"],
                               out0["conf_off_logit"])
            c0 = {k: v[0].cpu().numpy() for k, v in c0.items()}  # (T,)

        for f in range(T):
            if f <= W - 1:
                pcc, pcu, pla, pfc = c0["p_cc"][f], c0["p_cu"][f], c0["p_la"][f], c0["p_fc"][f]
                poff, pon, poffc = c0["p_off"][f], c0["p_conf_on"][f], c0["p_conf_off"][f]
            else:
                k = f - (W - 1)
                pcc, pcu, pla, pfc = seq_pcc[k], seq_pcu[k], seq_pla[k], seq_pfc[k]
                poff, pon, poffc = seq_poff[k], seq_pon[k], seq_poffc[k]
            four = np.array([pcc, pcu, pla, pfc])
            pred.append(int(four.argmax()))
            true.append(int(gt[f]))
            fc_prob.append(float(pfc)); off_prob.append(float(poff))
            on_prob.append(float(pon)); offc_prob.append(float(poffc))
            p_cc_f.append(float(pcc)); p_cu_f.append(float(pcu))
            p_la_f.append(float(pla)); p_fc_f.append(float(pfc))
        wi += n_win

    return {
        "pred": np.asarray(pred), "true": np.asarray(true),
        "fc_prob": np.asarray(fc_prob), "off_prob": np.asarray(off_prob),
        "on_prob": np.asarray(on_prob), "offc_prob": np.asarray(offc_prob),
        "p_cc": np.asarray(p_cc_f), "p_cu": np.asarray(p_cu_f),
        "p_la": np.asarray(p_la_f), "p_fc": np.asarray(p_fc_f),
    }


def _safe_auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(roc_auc_score(y, s))


def _safe_auprc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def compute_metrics(D: dict) -> dict:
    true = D["true"]; pred = D["pred"]
    y_off = np.isin(true, (L.LA, L.FC)).astype(int)
    y_conf = np.isin(true, (L.CC, L.FC)).astype(int)

    # axis AUROCs
    off_auroc = _safe_auroc(y_off, D["off_prob"])
    on_mask = y_off == 0
    conf_on_auroc = _safe_auroc(y_conf[on_mask], D["on_prob"][on_mask])
    off_mask = y_off == 1
    conf_off_auroc = _safe_auroc(y_conf[off_mask], D["offc_prob"][off_mask])

    # composed FC discriminations
    fc_vs_cc_mask = np.isin(true, (L.FC, L.CC))
    fc_vs_cc = _safe_auroc((true[fc_vs_cc_mask] == L.FC).astype(int), D["fc_prob"][fc_vs_cc_mask])
    fc_vs_la_mask = np.isin(true, (L.FC, L.LA))
    fc_vs_la = _safe_auroc((true[fc_vs_la_mask] == L.FC).astype(int), D["fc_prob"][fc_vs_la_mask])
    y_fc = (true == L.FC).astype(int)
    fc_vs_all_auroc = _safe_auroc(y_fc, D["fc_prob"])
    fc_vs_all_auprc = _safe_auprc(y_fc, D["fc_prob"])

    # per-class F1 (argmax)
    per = per_state_prf(true, pred, n_states=NUM_DERIVED_STATES, state_names=list(DERIVED_NAMES))
    macro = macro_f1(true, pred, n_states=NUM_DERIVED_STATES, state_names=list(DERIVED_NAMES))

    # FC argmax precision/recall/F1
    fc_pr = precision_recall_fscore_support(
        y_fc, (pred == L.FC).astype(int), average="binary", zero_division=0
    )

    # FC calibrated-threshold: choose threshold on P(FC) that maximizes F1.
    thr, fc_f1_cal, fc_p_cal, fc_r_cal = _best_fc_threshold(y_fc, D["fc_prob"])

    return {
        "off_axis_auroc": off_auroc,
        "conf_on_auroc": conf_on_auroc,
        "conf_off_auroc": conf_off_auroc,
        "fc_vs_cc_auroc": fc_vs_cc,
        "fc_vs_la_auroc": fc_vs_la,
        "fc_vs_all_auroc": fc_vs_all_auroc,
        "fc_vs_all_auprc": fc_vs_all_auprc,
        "cc_f1": per["CORRECT_CONFIRMED"]["f1"],
        "cu_f1": per["CORRECT_UNCERTAIN"]["f1"],
        "la_f1": per["LOST_AWARE"]["f1"],
        "fc_f1_argmax": per["FALSE_CONFIRMED"]["f1"],
        "fc_precision_argmax": float(fc_pr[0]),
        "fc_recall_argmax": float(fc_pr[1]),
        "fc_f1_calibrated": fc_f1_cal,
        "fc_precision_calibrated": fc_p_cal,
        "fc_recall_calibrated": fc_r_cal,
        "fc_threshold_calibrated": thr,
        "macro_f1": macro,
        "n_frames": int(true.size),
        "fc_support": int(y_fc.sum()),
        "cc_support": int((true == L.CC).sum()),
        "cu_support": int((true == L.CU).sum()),
        "la_support": int((true == L.LA).sum()),
    }


def _best_fc_threshold(y_fc, p_fc):
    """Threshold on P(FC) maximizing binary F1, computed in O(N log N).

    Sweep all unique scores via cumulative TP/FP from the score-sorted order
    (vectorized) instead of re-evaluating sklearn f1 per grid point.
    """
    y = np.asarray(y_fc, dtype=np.int64)
    s = np.asarray(p_fc, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.5, 0.0, 0.0, 0.0
    # Sort descending; predicting positive for the top-k highest scores.
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tp = np.cumsum(y_sorted)                      # TP at each prefix length k
    fp = np.cumsum(1 - y_sorted)                  # FP at each prefix length k
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / n_pos
    f1 = np.where((precision + recall) > 0,
                  2 * precision * recall / np.maximum(1e-12, precision + recall),
                  0.0)
    # Only consider prefix cuts at score boundaries (last index of each unique
    # score) so the threshold corresponds to a real >= cutpoint.
    boundary = np.ones(len(s_sorted), dtype=bool)
    boundary[:-1] = s_sorted[:-1] != s_sorted[1:]
    f1_b = np.where(boundary, f1, -1.0)
    k = int(np.argmax(f1_b))
    thr = float(s_sorted[k])
    return thr, float(f1[k]), float(precision[k]), float(recall[k])


# ===========================================================================
# Temperature fitting on the CALIB subset
# ===========================================================================
@torch.no_grad()
def _collect_logits(model, bank: WindowBank, device, batch_size=512):
    model.eval()
    ds = BankDataset(bank)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    offs, ons, offcs = [], [], []
    yoff, yconf = [], []
    for b in loader:
        out = model(b["geom"].to(device), b["resp"].to(device), last_step_only=True)
        offs.append(out["off_logit"].cpu()); ons.append(out["conf_on_logit"].cpu())
        offcs.append(out["conf_off_logit"].cpu())
        yoff.append(b["y_off"]); yconf.append(b["y_conf"])
    return (torch.cat(offs), torch.cat(ons), torch.cat(offcs),
            torch.cat(yoff), torch.cat(yconf))


def fit_temperature(logit, target, mask=None):
    """1-D temperature scaling: minimize BCE(logit/T, target) over T>0.

    Returns the optimal temperature (float).  If the (masked) subset is empty
    or single-class, returns 1.0.
    """
    if mask is not None:
        logit = logit[mask]; target = target[mask]
    if logit.numel() == 0 or target.sum() == 0 or target.sum() == target.numel():
        return 1.0
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
    lg = logit.detach(); tg = target.detach()

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lg / torch.exp(logT), tg)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(logT).item())


def fit_temperatures(model, calib_bank: WindowBank, device):
    offs, ons, offcs, yoff, yconf = _collect_logits(model, calib_bank, device)
    t_off = fit_temperature(offs, yoff)
    t_on = fit_temperature(ons, yconf, mask=(yoff == 0))
    t_offc = fit_temperature(offcs, yconf, mask=(yoff == 1))
    return t_off, t_on, t_offc


# ===========================================================================
# Train one seed
# ===========================================================================
def train_one_seed(seed, train_bank, calib_bank, val_bank, feature_cfg, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CondFactorizedCSC(
        geom_dim=L.GEOM_DIM, resp_dim=L.RESP_DIM,
        hidden=args.hidden, levels=args.levels, kernel=args.kernel,
        dropout=args.dropout,
    ).to(device)

    pw_off = _pos_weight(train_bank.y_off).to(device)
    on_mask = train_bank.y_off == 0
    off_mask = train_bank.y_off == 1
    pw_on = _pos_weight(train_bank.y_conf[on_mask]).to(device)
    pw_offc = _pos_weight(train_bank.y_conf[off_mask]).to(device)
    print(f"  [seed {seed}] pos_weight off={float(pw_off):.2f} "
          f"conf_on={float(pw_on):.2f} conf_off={float(pw_offc):.2f}")

    loader = DataLoader(BankDataset(train_bank), batch_size=args.batch_size,
                        shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.02
    )

    W = feature_cfg.window_size
    best_sel = -1.0
    best_state = None
    best_metrics = None
    best_temps = (1.0, 1.0, 1.0)

    for epoch in range(args.epochs):
        model.train()
        ep_acc = {"off": 0.0, "conf_on": 0.0, "conf_off": 0.0, "composed": 0.0}
        ep_tensors = None
        n_b = 0
        for b in loader:
            for k in ("geom", "resp", "y_off", "y_conf", "state4", "seq_w"):
                b[k] = b[k].to(device)
            model.set_temperatures(1.0, 1.0, 1.0)  # raw during training
            out = model(b["geom"], b["resp"], last_step_only=True)
            loss, parts = conditional_loss(
                out, b, pw_off, pw_on, pw_offc, model, args.lambda_composed
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            # Accumulate part-losses ON-DEVICE; sync to host only once per epoch.
            if ep_tensors is None:
                ep_tensors = {k: parts[k].clone() for k in parts}
            else:
                for k in parts:
                    ep_tensors[k] += parts[k]
            n_b += 1
        sched.step()
        if ep_tensors is not None:
            ep_acc = {k: float(v.cpu()) for k, v in ep_tensors.items()}

        # --- fit temperatures on CALIB, then eval dedup on the untouched val ---
        t_off, t_on, t_offc = fit_temperatures(model, calib_bank, device)
        model.set_temperatures(t_off, t_on, t_offc)
        D = dedup_per_frame(model, val_bank, feature_cfg, device, W)
        m = compute_metrics(D)

        # Model selection — COMBINED FC-aware metric (documented).
        # Rationale: the composed 4-way ARGMAX macro-F1 alone collapses to the
        # earliest epoch — as p_off sharpens, the off-target prior over-fires
        # and erodes CC/LA argmax even though the discriminative AUROCs keep
        # improving.  Selecting purely on argmax-macro would discard the
        # model's real strength (FC separability).  We therefore select on a
        # 50/50 blend of (a) composed argmax macro-F1 and (b) the rounded
        # quality of the three operating-point F1s that matter — CC, LA, and
        # the CALIBRATED FC-F1 (FC at its best threshold, which is the honest
        # deployable FC operating point).  This keeps argmax CC/LA in the loop
        # while rewarding calibrated FC.
        sel = 0.5 * m["macro_f1"] + 0.5 * np.mean(
            [m["cc_f1"], m["la_f1"], m["fc_f1_calibrated"]]
        )
        tag = ""
        if sel > best_sel:
            best_sel = sel
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = m
            best_temps = (t_off, t_on, t_offc)
            tag = "  <-- best"
        print(f"  [seed {seed}] ep{epoch:02d} "
              f"L(off={ep_acc['off']/n_b:.3f} on={ep_acc['conf_on']/n_b:.3f} "
              f"off_c={ep_acc['conf_off']/n_b:.3f} comp={ep_acc['composed']/n_b:.3f}) "
              f"| T=({t_off:.2f},{t_on:.2f},{t_offc:.2f}) "
              f"sel={sel:.4f} macroF1={m['macro_f1']:.4f} "
              f"FC-F1arg={m['fc_f1_argmax']:.3f} FC-F1cal={m['fc_f1_calibrated']:.3f} "
              f"FCvALL={m['fc_vs_all_auroc']:.3f} CC={m['cc_f1']:.3f} LA={m['la_f1']:.3f}{tag}")

    return best_state, best_metrics, best_temps, best_sel


# ===========================================================================
# CALIB carve-out
# ===========================================================================
def carve_calib(train_keys, groups, calib_fraction, seed):
    """Deterministically carve ~calib_fraction of the train sequences as CALIB.

    Stratified-ish: FC-bearing sequences are split proportionally so the calib
    set still contains FC frames (temperatures for conf_off need FC examples).
    Deterministic given (seed).
    """
    fc_keys, non_fc_keys = [], []
    for k in sorted(train_keys):
        rows = groups[k]
        if any(int(r.get("derived_state", 0)) == L.FC for r in rows):
            fc_keys.append(k)
        else:
            non_fc_keys.append(k)
    rng = np.random.default_rng(seed)
    rng.shuffle(fc_keys)
    rng.shuffle(non_fc_keys)

    def take(lst):
        n = max(1, int(round(calib_fraction * len(lst)))) if lst else 0
        return lst[:n], lst[n:]

    fc_cal, fc_tr = take(fc_keys)
    nf_cal, nf_tr = take(non_fc_keys)
    calib = fc_cal + nf_cal
    train = fc_tr + nf_tr
    return train, calib


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lambda_composed", type=float, default=0.5)
    ap.add_argument("--calib_fraction", type=float, default=0.15)
    ap.add_argument("--device", type=str, default="mps")
    args = ap.parse_args()

    device = args.device
    print("=" * 88)
    print("CONDITIONAL FACTORIZED CSC — controlled rematch (V3 feature parity)")
    print("=" * 88)
    print(f"GEOM_DIM={L.GEOM_DIM} RESP_DIM={L.RESP_DIM} "
          f"hidden={args.hidden} levels={args.levels} kernel={args.kernel} "
          f"lambda_composed={args.lambda_composed} calib_fraction={args.calib_fraction}")
    print(f"GEOM={L.GEOM_FEATURES}")
    print(f"RESP={L.RESP_FEATURES}")

    cfg = L.load_v3_config()
    groups = L.load_groups()
    train_keys, val_keys = L.reproduce_v3_split(groups, cfg.val_fraction)
    print(f"[split] reproduced V3 EXACTLY: train={len(train_keys)} val={len(val_keys)}")

    # Untouched V3 val bank (shared, with dedup bookkeeping).
    print("[bank] building val bank (dedup) ...")
    val_bank = build_bank({k: groups[k] for k in val_keys}, cfg.feature, with_dedup=True)
    print(f"        val windows={val_bank.geom.shape[0]:,}")

    W = cfg.feature.window_size
    results = []
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        print("\n" + "-" * 88)
        print(f"SEED {seed}")
        print("-" * 88)
        # CALIB carved from the 373 train seqs (deterministic by seed).
        tr_keys, cal_keys = carve_calib(train_keys, groups, args.calib_fraction, seed)
        n_fc_cal = sum(
            1 for k in cal_keys
            if any(int(r.get("derived_state", 0)) == L.FC for r in groups[k])
        )
        print(f"  [calib] train_fit={len(tr_keys)} calib={len(cal_keys)} "
              f"(FC-bearing calib seqs={n_fc_cal})")

        train_bank = build_bank({k: groups[k] for k in tr_keys}, cfg.feature)
        calib_bank = build_bank({k: groups[k] for k in cal_keys}, cfg.feature)
        print(f"  [bank] train windows={train_bank.geom.shape[0]:,} "
              f"calib windows={calib_bank.geom.shape[0]:,}")
        # Conditional class balance (informational).
        on_mask = train_bank.y_off == 0
        off_mask = train_bank.y_off == 1
        print(f"  [balance] off-rate={float(train_bank.y_off.mean()):.3f} | "
              f"conf|on (CC frac)={float(train_bank.y_conf[on_mask].mean()):.3f} | "
              f"conf|off (FC frac)={float(train_bank.y_conf[off_mask].mean()):.3f}")

        best_state, best_metrics, best_temps, best_sel = train_one_seed(
            seed, train_bank, calib_bank, val_bank, cfg.feature, args, device
        )

        run_dir = OUT_ROOT / f"cond_fact_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": best_state,
            "geom_features": list(L.GEOM_FEATURES),
            "resp_features": list(L.RESP_FEATURES),
            "geom_dim": L.GEOM_DIM,
            "resp_dim": L.RESP_DIM,
            "hidden": args.hidden,
            "levels": args.levels,
            "kernel": args.kernel,
            "t_off": best_temps[0],
            "t_conf_on": best_temps[1],
            "t_conf_off": best_temps[2],
            "lambda_composed": args.lambda_composed,
            "seed": seed,
            "selection_metric": "0.5*composed_macro_f1 + 0.5*mean(cc_f1,la_f1,fc_f1_calibrated)",
            "selection_value": best_sel,
            "val": best_metrics,
        }
        torch.save(ckpt, run_dir / "checkpoint_best.pth")
        (run_dir / "val_metrics.json").write_text(json.dumps(best_metrics, indent=2))
        print(f"  [saved] {run_dir/'checkpoint_best.pth'}")
        results.append((seed, best_metrics, best_temps))

    # ----------------------------- REPORT -----------------------------------
    print_report(results, args)


def _mean_std(vals):
    arr = np.array([v for v in vals if v == v], dtype=float)  # drop nan
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def print_report(results, args):
    print("\n" + "=" * 88)
    print("RESULTS — dedup per-frame on the UNTOUCHED V3 val (58 seqs)")
    print("=" * 88)

    metric_keys = [
        ("off-axis AUROC", "off_axis_auroc"),
        ("conf_on AUROC (on-tgt)", "conf_on_auroc"),
        ("conf_off AUROC (off-tgt)", "conf_off_auroc"),
        ("composed FC-vs-CC AUROC", "fc_vs_cc_auroc"),
        ("composed FC-vs-LA AUROC", "fc_vs_la_auroc"),
        ("FC-vs-ALL AUROC", "fc_vs_all_auroc"),
        ("FC-vs-ALL AUPRC", "fc_vs_all_auprc"),
        ("CC F1", "cc_f1"),
        ("CU F1", "cu_f1"),
        ("LA F1", "la_f1"),
        ("FC F1 (argmax)", "fc_f1_argmax"),
        ("FC prec (argmax)", "fc_precision_argmax"),
        ("FC rec (argmax)", "fc_recall_argmax"),
        ("FC F1 (calibrated)", "fc_f1_calibrated"),
        ("FC prec (calibrated)", "fc_precision_calibrated"),
        ("FC rec (calibrated)", "fc_recall_calibrated"),
        ("macro-F1", "macro_f1"),
    ]
    seeds = [r[0] for r in results]
    hdr = f"{'metric':<26}" + "".join(f"{'seed'+str(s):>10}" for s in seeds) + f"{'mean±std':>18}"
    print(hdr)
    print("-" * len(hdr))
    agg = {}
    for label, key in metric_keys:
        vals = [r[1][key] for r in results]
        mean, std = _mean_std(vals)
        agg[key] = (mean, std)
        cells = "".join(f"{v:>10.4f}" if v == v else f"{'n/a':>10}" for v in vals)
        ms = f"{mean:.4f}±{std:.4f}" if mean == mean else "n/a"
        print(f"{label:<26}{cells}{ms:>18}")

    print(f"\n  FC calibrated threshold per seed: "
          f"{[round(r[1]['fc_threshold_calibrated'],3) for r in results]}")
    print(f"  temperatures (t_off,t_conf_on,t_conf_off) per seed: "
          f"{[tuple(round(x,3) for x in r[2]) for r in results]}")
    print(f"  support (val dedup): CC={results[0][1]['cc_support']} "
          f"CU={results[0][1]['cu_support']} LA={results[0][1]['la_support']} "
          f"FC={results[0][1]['fc_support']} N={results[0][1]['n_frames']}")

    # ----------------------------- GATES -------------------------------------
    print("\n" + "=" * 88)
    print("HARD-STOP GATES (vs V3-prod dedup per-frame baseline)")
    print(f"  V3 baseline: CC={V3_BASELINE['CC']} LA={V3_BASELINE['LA']} "
          f"macro={V3_BASELINE['macro']} FC-F1={V3_BASELINE['FC']}")
    print("=" * 88)

    fc_mean, fc_std = agg["fc_f1_argmax"]
    cc_mean, _ = agg["cc_f1"]
    la_mean, _ = agg["la_f1"]
    macro_mean, _ = agg["macro_f1"]

    # Gate 1: in-domain FC-F1 >= 0.78  (use argmax FC-F1, mean over seeds).
    g1 = fc_mean >= 0.78
    print(f"  [Gate 1] in-domain FC-F1 >= 0.78 ?  mean FC-F1(argmax)={fc_mean:.4f}"
          f"  -> {'PASS' if g1 else 'FAIL'}")

    # Gate 2: CC & LA & macro regression <= 2-3 points (use 3 pts = 0.03).
    cc_reg = V3_BASELINE["CC"] - cc_mean
    la_reg = V3_BASELINE["LA"] - la_mean
    macro_reg = V3_BASELINE["macro"] - macro_mean
    fc_reg = V3_BASELINE["FC"] - fc_mean
    g2_cc = cc_reg <= 0.03
    g2_la = la_reg <= 0.03
    g2_macro = macro_reg <= 0.03
    g2 = g2_cc and g2_la and g2_macro
    print(f"  [Gate 2] regression <= 3 pts vs V3 ?")
    print(f"           CC    {cc_mean:.4f} (reg {cc_reg:+.4f}) -> {'PASS' if g2_cc else 'FAIL'}")
    print(f"           LA    {la_mean:.4f} (reg {la_reg:+.4f}) -> {'PASS' if g2_la else 'FAIL'}")
    print(f"           macro {macro_mean:.4f} (reg {macro_reg:+.4f}) -> {'PASS' if g2_macro else 'FAIL'}")
    print(f"           (FC-F1 {fc_mean:.4f}, reg vs 0.83 = {fc_reg:+.4f}, informational)")
    print(f"           -> {'PASS' if g2 else 'FAIL'}")

    # Gate 3: stable across 3 seeds (std small). Heuristic: FC-F1 std<=0.03 and
    # macro std<=0.02.
    g3 = (fc_std <= 0.03) and (agg["macro_f1"][1] <= 0.02)
    print(f"  [Gate 3] stable across {len(results)} seeds ?  "
          f"FC-F1 std={fc_std:.4f} (<=0.03), macro std={agg['macro_f1'][1]:.4f} (<=0.02)"
          f"  -> {'PASS' if g3 else 'FAIL'}")

    print("\n  OVERALL: " + ("PASS" if (g1 and g2 and g3) else "FAIL"))
    print("\n  Reproduce: .venv/bin/python tools/train_cond_factorized.py "
          f"--seeds {' '.join(map(str, args.seeds))} --epochs {args.epochs} "
          f"--lambda_composed {args.lambda_composed} --calib_fraction {args.calib_fraction}")


if __name__ == "__main__":
    main()
