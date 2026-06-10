"""Train the factorized 2-tower CSC-v4 challenger (model_v4_factorized.FactorizedCSC).

Controlled challenger to the frozen V3-prod diagnosis model. Two strictly
isolated causal-TCN towers — geometry (off_target axis) and response (confirmed
axis) — are composed into the 4-state distribution ``{CC, CU, LA, FC}`` where
``FC = off × conf``. The factorization structurally prevents the joint-head
shortcut that collapses FC-vs-CC.

SAFETY: trains only on TRAIN-set shards (lasot/got10k/dtb70/uavdt_sot/
visdrone_sot). UAV123 is never read here.

Data contract (outputs/csc_labels_v4/train_shards.jsonl)
--------------------------------------------------------
Each row: ``feat_0..feat_40`` (names = FEATURE_NAMES_V4), ``derived``
(0=CC,1=CU,2=LA,3=FC), ``dataset``, ``sequence``, ``frame_idx``, ``iou``.

  GEOM (off_target tower, 7):
    log_w_ratio_to_init_pct, log_h_ratio_to_init_pct, log_area_ratio_to_init_pct,
    aspect_ratio_pct, velocity_pct, acceleration_pct, area_ratio_pct
  RESP (confirmed tower, 10):
    apce_pct, psr_pct, response_entropy_pct, response_entropy_pct_delta,
    sm_local_peak_margin_pct, sm_local_top2_ratio_pct,
    sm_local_top2_ratio_pct_delta, sm_n_secondary_pct, confidence_pct,
    conf_ema_trend_pct

Labels per frame: ``y_off = derived in {2,3}``; ``y_conf = derived in {0,3}``;
``state4 = derived``.

Windows: causal, length 32, left-pad with the first frame, last-step target;
per (dataset, sequence) ordered by frame_idx.

Split: V3's EXACT split (outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/
split_info.json) -> val = val_sequences, train = the rest.

Loss
----
``L = BCE(off, y_off, pos_weight) + BCE(conf, y_conf, pos_weight)
     + lambda_composed * CE_composed`` where ``CE_composed = -log(p_state4)``
using ``compose()`` at temperature 1. Every term is SEQUENCE-NORMALIZED: each
frame is weighted by ``1/len(its sequence)`` (renormalized so each sequence
contributes ~equal total loss), de-concentrating long FC-heavy sequences like
drone-13. A plain shuffled DataLoader is used — NO WeightedRandomSampler.

Calibration: after training, fit ``t_off`` / ``t_conf`` independently on val by
1-D search minimizing each axis's (sequence-normalized) BCE.

Run
---
    python -m tools.train_csc_v4_factorized --name csc_v4_fact_small    --lambda_composed 0.0
    python -m tools.train_csc_v4_factorized --name csc_v4_fact_composed --lambda_composed 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "src")
from csc_lib.csc.v4.features_v4 import FEATURE_NAMES_V4  # noqa: E402
from csc_lib.csc.v4.model_v4_factorized import FactorizedCSC  # noqa: E402

try:
    from sklearn.metrics import (  # noqa: E402
        average_precision_score,
        f1_score,
        roc_auc_score,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"sklearn required for metrics: {exc}")

# --- shared contract constants ---------------------------------------------
GEOM = [
    "log_w_ratio_to_init_pct", "log_h_ratio_to_init_pct", "log_area_ratio_to_init_pct",
    "aspect_ratio_pct", "velocity_pct", "acceleration_pct", "area_ratio_pct",
]
RESP = [
    "apce_pct", "psr_pct", "response_entropy_pct", "response_entropy_pct_delta",
    "sm_local_peak_margin_pct", "sm_local_top2_ratio_pct",
    "sm_local_top2_ratio_pct_delta", "sm_n_secondary_pct", "confidence_pct",
    "conf_ema_trend_pct",
]

REPO = Path(__file__).resolve().parent.parent
SHARD = REPO / "outputs" / "csc_labels_v4" / "train_shards.jsonl"
SPLIT_INFO = REPO / "outputs" / "csc_training" / "sglatrack_r3_fcw3_w32_tcn32_stage2" / "split_info.json"
WINDOW = 32

# CC, CU, LA, FC
STATE_NAMES = ["CC", "CU", "LA", "FC"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _resolve_groups() -> tuple[list[int], list[int]]:
    """Map GEOM/RESP names -> shard feat columns; assert strict isolation."""
    geom_idx = [FEATURE_NAMES_V4.index(n) for n in GEOM]
    resp_idx = [FEATURE_NAMES_V4.index(n) for n in RESP]
    assert set(geom_idx).isdisjoint(resp_idx), "GEOM/RESP overlap — isolation broken!"
    return geom_idx, resp_idx


def load_sequences(geom_idx: list[int], resp_idx: list[int]) -> dict[tuple[str, str], dict]:
    """Read the shard once; group rows per (dataset, sequence), ordered by frame_idx.

    Returns ``{(dataset, sequence): {"geom": (L,Gd), "resp": (L,Rd),
    "y_off": (L,), "y_conf": (L,), "state4": (L,)}}``.
    """
    if not SHARD.exists():
        raise SystemExit(f"shard not found: {SHARD}")
    feat_keys = [f"feat_{i}" for i in range(len(FEATURE_NAMES_V4))]
    buckets: dict[tuple[str, str], list[tuple[int, list[float], int]]] = {}
    n = 0
    with open(SHARD) as f:
        for line in f:
            r = json.loads(line)
            key = (r["dataset"], r["sequence"])
            feats = [r[k] for k in feat_keys]
            buckets.setdefault(key, []).append((int(r["frame_idx"]), feats, int(r["derived"])))
            n += 1
    print(f"[data] read {n:,} rows over {len(buckets)} sequences")

    seqs: dict[tuple[str, str], dict] = {}
    for key, rows in buckets.items():
        rows.sort(key=lambda x: x[0])  # order by frame_idx
        feats = np.asarray([fr[1] for fr in rows], dtype=np.float32)  # (L, F)
        derived = np.asarray([fr[2] for fr in rows], dtype=np.int64)  # (L,)
        seqs[key] = {
            "geom": feats[:, geom_idx],                       # (L, Gd)
            "resp": feats[:, resp_idx],                       # (L, Rd)
            "y_off": np.isin(derived, [2, 3]).astype(np.float32),   # LA or FC
            "y_conf": np.isin(derived, [0, 3]).astype(np.float32),  # CC or FC
            "state4": derived,                                # 0..3
        }
    return seqs


class WindowDataset(Dataset):
    """Causal length-32 windows (left-pad with first frame), last-step target.

    Each sample carries ``seq_weight = 1/len(sequence)`` so that, after global
    renormalization in the trainer, every sequence contributes ~equal total
    loss regardless of its frame count.
    """

    def __init__(self, seqs: dict, keys: list, window: int = WINDOW) -> None:
        self.window = window
        self.geom: list[np.ndarray] = []
        self.resp: list[np.ndarray] = []
        self.y_off: list[float] = []
        self.y_conf: list[float] = []
        self.state4: list[int] = []
        self.seq_weight: list[float] = []
        self.meta: list[tuple] = []  # (dataset, sequence, frame_idx) for dedup
        for key in keys:
            s = seqs[key]
            L = s["geom"].shape[0]
            w = 1.0 / float(L)
            for t in range(L):
                lo = t - window + 1
                if lo < 0:
                    pad = -lo
                    geom_w = np.concatenate(
                        [np.repeat(s["geom"][:1], pad, axis=0), s["geom"][: t + 1]], axis=0
                    )
                    resp_w = np.concatenate(
                        [np.repeat(s["resp"][:1], pad, axis=0), s["resp"][: t + 1]], axis=0
                    )
                else:
                    geom_w = s["geom"][lo : t + 1]
                    resp_w = s["resp"][lo : t + 1]
                self.geom.append(geom_w)
                self.resp.append(resp_w)
                self.y_off.append(float(s["y_off"][t]))
                self.y_conf.append(float(s["y_conf"][t]))
                self.state4.append(int(s["state4"][t]))
                self.seq_weight.append(w)
                self.meta.append((key[0], key[1], t))

    def __len__(self) -> int:
        return len(self.geom)

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.geom[i]),       # (T, Gd)
            torch.from_numpy(self.resp[i]),       # (T, Rd)
            torch.tensor(self.y_off[i], dtype=torch.float32),
            torch.tensor(self.y_conf[i], dtype=torch.float32),
            torch.tensor(self.state4[i], dtype=torch.long),
            torch.tensor(self.seq_weight[i], dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Training / eval helpers
# ---------------------------------------------------------------------------


def composed_nll(model: FactorizedCSC, off_logit: torch.Tensor, conf_logit: torch.Tensor,
                 state4: torch.Tensor) -> torch.Tensor:
    """Per-sample -log(p_state4) using compose() at temperature 1.

    Calibration temperatures are deliberately bypassed during training (the
    loss should optimize the raw logits, not the post-hoc calibration). Returns
    a (B,) tensor of per-sample NLL.
    """
    p_off = torch.sigmoid(off_logit).clamp(1e-6, 1 - 1e-6)
    p_conf = torch.sigmoid(conf_logit).clamp(1e-6, 1 - 1e-6)
    p_cc = (1 - p_off) * p_conf
    p_cu = (1 - p_off) * (1 - p_conf)
    p_la = p_off * (1 - p_conf)
    p_fc = p_off * p_conf
    probs = torch.stack([p_cc, p_cu, p_la, p_fc], dim=-1)  # (B, 4)
    p_true = probs.gather(-1, state4.view(-1, 1)).squeeze(-1).clamp_min(1e-12)
    return -torch.log(p_true)


@torch.no_grad()
def collect_val_logits(model: FactorizedCSC, loader: DataLoader, device: str):
    """Run val once; return raw logits, labels, seq-weights and dedup meta keys."""
    model.eval()
    offs, confs, yoff, yconf, st4, sw = [], [], [], [], [], []
    for geom, resp, y_off, y_conf, state4, w in loader:
        out = model.forward(geom.to(device), resp.to(device), last_step_only=True)
        offs.append(out["off_logit"].cpu())
        confs.append(out["conf_logit"].cpu())
        yoff.append(y_off)
        yconf.append(y_conf)
        st4.append(state4)
        sw.append(w)
    return (
        torch.cat(offs).numpy(),
        torch.cat(confs).numpy(),
        torch.cat(yoff).numpy(),
        torch.cat(yconf).numpy(),
        torch.cat(st4).numpy(),
        torch.cat(sw).numpy(),
    )


def fit_temperature(logits: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    """1-D search for the temperature minimizing (weighted) binary cross-entropy."""
    z = torch.from_numpy(logits.astype(np.float64))
    y = torch.from_numpy(labels.astype(np.float64))
    w = torch.from_numpy(weights.astype(np.float64))
    w = w / w.sum()
    best_t, best_loss = 1.0, float("inf")
    for t in np.linspace(0.25, 5.0, 96):
        p = torch.sigmoid(z / t).clamp(1e-7, 1 - 1e-7)
        bce = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))
        loss = float((w * bce).sum())
        if loss < best_loss:
            best_loss, best_t = loss, float(t)
    return best_t


def _safe_auroc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def dedup_last_step(meta_keys: list[tuple], *arrays: np.ndarray):
    """Keep one prediction per unique (dataset, sequence, frame_idx) meta key.

    The dataset is already one-window-per-frame, but dedup is applied for safety
    so each unique frame contributes exactly once to the reported metrics.
    """
    seen: dict[tuple, int] = {}
    keep: list[int] = []
    for i, k in enumerate(meta_keys):
        if k not in seen:
            seen[k] = i
            keep.append(i)
    idx = np.asarray(keep, dtype=np.int64)
    return tuple(a[idx] for a in arrays)


def evaluate(model: FactorizedCSC, val_ds: WindowDataset, loader: DataLoader, device: str) -> dict:
    """Full held-out report (dedup per-frame). Uses calibrated temperatures."""
    off_l, conf_l, y_off, y_conf, state4, sw = collect_val_logits(model, loader, device)

    # dedup per unique frame (val DataLoader is NOT shuffled -> meta order matches)
    off_l, conf_l, y_off, y_conf, state4 = dedup_last_step(
        val_ds.meta, off_l, conf_l, y_off, y_conf, state4
    )

    t_off = float(model.t_off.cpu())
    t_conf = float(model.t_conf.cpu())
    p_off = 1.0 / (1.0 + np.exp(-off_l / t_off))
    p_conf = 1.0 / (1.0 + np.exp(-conf_l / t_conf))
    p_cc = (1 - p_off) * p_conf
    p_cu = (1 - p_off) * (1 - p_conf)
    p_la = p_off * (1 - p_conf)
    p_fc = p_off * p_conf
    probs4 = np.stack([p_cc, p_cu, p_la, p_fc], axis=1)  # (N,4)
    argmax4 = probs4.argmax(axis=1)

    is_cc = (state4 == 0); is_cu = (state4 == 1); is_la = (state4 == 2); is_fc = (state4 == 3)

    # --- axis AUROCs ---
    off_auroc = _safe_auroc(y_off, p_off)                 # on vs off-target
    conf_auroc = _safe_auroc(y_conf, p_conf)              # conf vs unconf

    # --- composed FC AUROCs (FC probability as the score) ---
    # FC vs CC: restrict to confirmed states {CC, FC}
    m_fc_cc = is_fc | is_cc
    fc_vs_cc = _safe_auroc(is_fc[m_fc_cc].astype(int), p_fc[m_fc_cc])
    # FC vs LA: restrict to off-target states {LA, FC}
    m_fc_la = is_fc | is_la
    fc_vs_la = _safe_auroc(is_fc[m_fc_la].astype(int), p_fc[m_fc_la])
    # FC vs ALL
    fc_vs_all = _safe_auroc(is_fc.astype(int), p_fc)
    fc_auprc = (float(average_precision_score(is_fc.astype(int), p_fc))
                if is_fc.sum() > 0 else float("nan"))

    # --- 4-way argmax F1s ---
    f1_argmax = {
        STATE_NAMES[c]: float(f1_score((state4 == c).astype(int), (argmax4 == c).astype(int),
                                       zero_division=0))
        for c in range(4)
    }
    macro_f1_argmax = float(np.mean(list(f1_argmax.values())))

    # FC precision/recall/F1 at 4-way argmax
    fc_pred_argmax = (argmax4 == 3)
    tp = int((fc_pred_argmax & is_fc).sum()); fp = int((fc_pred_argmax & ~is_fc).sum())
    fn = int((~fc_pred_argmax & is_fc).sum())
    fc_prec_a = tp / (tp + fp) if (tp + fp) else 0.0
    fc_rec_a = tp / (tp + fn) if (tp + fn) else 0.0
    fc_f1_a = (2 * fc_prec_a * fc_rec_a / (fc_prec_a + fc_rec_a)) if (fc_prec_a + fc_rec_a) else 0.0

    # --- FC at a calibrated threshold on p_fc (max-F1 sweep over p_fc) ---
    best = {"thr": 0.5, "f1": 0.0, "prec": 0.0, "rec": 0.0}
    if is_fc.sum() > 0:
        order = np.argsort(-p_fc)
        y_sorted = is_fc[order].astype(int)
        cum_tp = np.cumsum(y_sorted)
        cum_fp = np.cumsum(1 - y_sorted)
        total_pos = int(is_fc.sum())
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1)
        rec = cum_tp / total_pos
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
        bi = int(np.argmax(f1))
        best = {
            "thr": float(p_fc[order][bi]),
            "f1": float(f1[bi]),
            "prec": float(prec[bi]),
            "rec": float(rec[bi]),
        }

    return {
        "n_val_frames": int(len(state4)),
        "class_counts": {STATE_NAMES[c]: int((state4 == c).sum()) for c in range(4)},
        "off_axis_auroc": off_auroc,
        "conf_axis_auroc": conf_auroc,
        "fc_vs_cc_auroc": fc_vs_cc,
        "fc_vs_la_auroc": fc_vs_la,
        "fc_vs_all_auroc": fc_vs_all,
        "fc_vs_all_auprc": fc_auprc,
        "fc_argmax": {"precision": fc_prec_a, "recall": fc_rec_a, "f1": fc_f1_a},
        "fc_thresh": best,
        "per_class_f1_argmax": f1_argmax,
        "macro_f1_argmax": macro_f1_argmax,
    }


# ---------------------------------------------------------------------------
# Train one variant
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "mps" if (args.device == "mps" and torch.backends.mps.is_available()) else "cpu"
    print(f"[train] variant={args.name} lambda_composed={args.lambda_composed} device={device}")

    geom_idx, resp_idx = _resolve_groups()
    print(f"[train] GEOM cols={geom_idx}  RESP cols={resp_idx}")
    seqs = load_sequences(geom_idx, resp_idx)

    # V3's EXACT split: val = val_sequences, train = the rest
    split = json.load(open(SPLIT_INFO))
    val_keys = [tuple(x) for x in split["val_sequences"] if tuple(x) in seqs]
    val_set = set(val_keys)
    train_keys = [k for k in seqs if k not in val_set]
    print(f"[train] split (V3): {len(train_keys)} train / {len(val_keys)} val sequences")
    assert len(val_keys) == len(split["val_sequences"]), "missing val sequences in shard"

    train_ds = WindowDataset(seqs, train_keys)
    val_ds = WindowDataset(seqs, val_keys)
    print(f"[train] windows: {len(train_ds):,} train / {len(val_ds):,} val")

    # SEQUENCE-NORMALIZED weighting: each frame weighted 1/len(seq), then the
    # whole train set renormalized so the mean weight is 1 (so the effective
    # number of contributing samples ~ number of sequences, and lr stays sane).
    tr_w = np.asarray(train_ds.seq_weight, dtype=np.float64)
    tr_w = tr_w / tr_w.mean()  # mean weight -> 1
    # patch the per-sample weights in the dataset so the loader returns them
    train_ds.seq_weight = tr_w.astype(np.float32).tolist()

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, drop_last=False)

    # pos_weight per axis (on the training frames) = neg/pos
    yoff_tr = np.asarray(train_ds.y_off)
    yconf_tr = np.asarray(train_ds.y_conf)
    off_pos = max(yoff_tr.sum(), 1.0); off_neg = len(yoff_tr) - yoff_tr.sum()
    conf_pos = max(yconf_tr.sum(), 1.0); conf_neg = len(yconf_tr) - yconf_tr.sum()
    pw_off = torch.tensor(off_neg / off_pos, dtype=torch.float32, device=device)
    pw_conf = torch.tensor(conf_neg / conf_pos, dtype=torch.float32, device=device)
    print(f"[train] pos_weight off={float(pw_off):.3f} (pos={int(off_pos)}) "
          f"conf={float(pw_conf):.3f} (pos={int(conf_pos)})")

    model = FactorizedCSC(
        geom_dim=len(GEOM), resp_dim=len(RESP),
        hidden=args.hidden, levels=args.levels, kernel=args.kernel, dropout=args.dropout,
    ).to(device)
    print(f"[train] FactorizedCSC params={model.num_params:,}")

    bce_off = nn.BCEWithLogitsLoss(pos_weight=pw_off, reduction="none")
    bce_conf = nn.BCEWithLogitsLoss(pos_weight=pw_conf, reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_macro = -1.0
    best_state = None
    best_metrics: dict = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        tot, tot_off, tot_conf, tot_comp, wsum = 0.0, 0.0, 0.0, 0.0, 0.0
        for geom, resp, y_off, y_conf, state4, w in train_loader:
            geom = geom.to(device); resp = resp.to(device)
            y_off = y_off.to(device); y_conf = y_conf.to(device)
            state4 = state4.to(device); w = w.to(device)

            out = model.forward(geom, resp, last_step_only=True)
            off_logit = out["off_logit"]; conf_logit = out["conf_logit"]

            l_off = (w * bce_off(off_logit, y_off)).sum() / w.sum()
            l_conf = (w * bce_conf(conf_logit, y_conf)).sum() / w.sum()
            if args.lambda_composed > 0:
                l_comp = (w * composed_nll(model, off_logit, conf_logit, state4)).sum() / w.sum()
            else:
                l_comp = torch.zeros((), device=device)
            loss = l_off + l_conf + args.lambda_composed * l_comp

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            bw = float(w.sum())
            tot += float(loss.detach()) * bw; tot_off += float(l_off.detach()) * bw
            tot_conf += float(l_conf.detach()) * bw
            tot_comp += float(l_comp.detach()) * bw; wsum += bw

        # fit temperatures on val each epoch (independent per axis) before report
        off_l, conf_l, y_off_v, y_conf_v, state4_v, sw_v = collect_val_logits(model, val_loader, device)
        # dedup before fitting/reporting
        off_l, conf_l, y_off_v, y_conf_v, state4_v, sw_v = dedup_last_step(
            val_ds.meta, off_l, conf_l, y_off_v, y_conf_v, state4_v, sw_v
        )
        t_off = fit_temperature(off_l, y_off_v, sw_v)
        t_conf = fit_temperature(conf_l, y_conf_v, sw_v)
        model.set_temperatures(t_off, t_conf)

        metrics = evaluate(model, val_ds, val_loader, device)
        dt = time.perf_counter() - t0
        print(
            f"[ep {epoch:02d}] loss={tot/wsum:.4f} (off={tot_off/wsum:.4f} "
            f"conf={tot_conf/wsum:.4f} comp={tot_comp/wsum:.4f}) | "
            f"off_AUC={metrics['off_axis_auroc']:.3f} conf_AUC={metrics['conf_axis_auroc']:.3f} "
            f"FCvsCC={metrics['fc_vs_cc_auroc']:.3f} FCvsALL={metrics['fc_vs_all_auroc']:.3f} "
            f"FC-F1(thr)={metrics['fc_thresh']['f1']:.3f} macroF1={metrics['macro_f1_argmax']:.3f} "
            f"t_off={t_off:.2f} t_conf={t_conf:.2f} ({dt:.1f}s)"
        )

        # model selection: macro-F1 (4-way argmax) on held-out val
        if metrics["macro_f1_argmax"] > best_macro:
            best_macro = metrics["macro_f1_argmax"]
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics["t_off"] = t_off
            best_metrics["t_conf"] = t_conf

    # restore best and write checkpoint
    assert best_state is not None
    model.load_state_dict(best_state)
    out_dir = REPO / "outputs" / "csc_training_v4" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": best_state,
        "geom_features": GEOM,
        "resp_features": RESP,
        "geom_dim": len(GEOM),
        "resp_dim": len(RESP),
        "hidden": args.hidden,
        "levels": args.levels,
        "kernel": args.kernel,
        "t_off": best_metrics["t_off"],
        "t_conf": best_metrics["t_conf"],
        "lambda_composed": args.lambda_composed,
        "val": best_metrics,
    }
    torch.save(ckpt, out_dir / "checkpoint_best.pth")
    json.dump(best_metrics, open(out_dir / "val_metrics.json", "w"), indent=2)
    print(f"[train] saved -> {out_dir/'checkpoint_best.pth'} (best macroF1={best_macro:.3f})")

    _print_report(args.name, args.lambda_composed, best_metrics)
    return best_metrics


def _print_report(name: str, lam: float, m: dict) -> None:
    print("\n" + "=" * 72)
    print(f"HELD-OUT VAL REPORT — {name} (lambda_composed={lam})")
    print("=" * 72)
    print(f"val frames (dedup): {m['n_val_frames']:,}  class counts: {m['class_counts']}")
    print(f"  temperatures: t_off={m['t_off']:.3f}  t_conf={m['t_conf']:.3f}")
    print("-- axis AUROCs --")
    print(f"  off-axis  (on vs off-target): {m['off_axis_auroc']:.4f}")
    print(f"  conf-axis (conf vs unconf)  : {m['conf_axis_auroc']:.4f}")
    print("-- composed FC AUROCs --")
    print(f"  FC vs CC : {m['fc_vs_cc_auroc']:.4f}")
    print(f"  FC vs LA : {m['fc_vs_la_auroc']:.4f}")
    print(f"  FC vs ALL: {m['fc_vs_all_auroc']:.4f}  (AUPRC {m['fc_vs_all_auprc']:.4f})")
    print("-- FC detection --")
    print(f"  4-way argmax : P={m['fc_argmax']['precision']:.3f} "
          f"R={m['fc_argmax']['recall']:.3f} F1={m['fc_argmax']['f1']:.3f}")
    print(f"  calib thresh : P={m['fc_thresh']['prec']:.3f} "
          f"R={m['fc_thresh']['rec']:.3f} F1={m['fc_thresh']['f1']:.3f} (thr p_fc={m['fc_thresh']['thr']:.4f})")
    print("-- per-class F1 (4-way argmax) --")
    pf = m["per_class_f1_argmax"]
    print(f"  CC={pf['CC']:.3f}  CU={pf['CU']:.3f}  LA={pf['LA']:.3f}  FC={pf['FC']:.3f}  "
          f"| macro-F1={m['macro_f1_argmax']:.3f}")
    print("=" * 72 + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Train factorized 2-tower CSC-v4 challenger")
    p.add_argument("--name", required=True, help="output subdir under outputs/csc_training_v4/")
    p.add_argument("--lambda_composed", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--kernel", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
