#!/usr/bin/env python
"""CSC-v4 multi-head trainer (GPU/MPS).

Trains `csc_lib.csc.v4.model_v4.CSCv4` on the shards from tools/v4_build_labels.py.
Heads + losses:
  derived(4)            weighted CE  (state_weights, FC up-weighted)
  fc_subtype(3)         CE
  la_subtype(6)         CE
  hazard_1/3/10         BCE (pos-weighted)
  do_not_act            BCE
  template_update_safe  BCE
  action_utility(7)     MSE on ΔIoU, MASKED to in_action_window frames
Causal: target = label of the window's LAST frame (matches CSCv4.predict last_step_only).
Train set only (NEVER UAV123). Default device = mps (Apple GPU); falls back to cpu.
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "salrtd" / "src", PROJECT_ROOT / "src", PROJECT_ROOT):
    sys.path.insert(0, str(_p))

from csc_lib.csc.v4.model_v4 import CSCv4
from csc_lib.csc.v4.v4types import ACTION_NAMES, DerivedStateV4, FCSubtype, LASubtype

N_FEAT = 16
WINDOW = 32
ACT_KEYS = [f"act_{a}" for a in ACTION_NAMES]


def _device(pref: str) -> str:
    if pref == "mps" and torch.backends.mps.is_available():
        return "mps"
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ShardSet(torch.utils.data.Dataset):
    """Per-frame causal windows (last-step target) from the v4 shard jsonl."""

    def __init__(self, rows: list[dict], feat_dim: int = N_FEAT, fc_cap: int = 0):
        self.feat_dim = feat_dim
        byseq: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            byseq[(r["dataset"], r["sequence"])].append(r)
        self.feats: list[np.ndarray] = []      # per-seq (n,F)
        self.labels: list[dict] = []           # per-seq dict of arrays
        self.index: list[tuple[int, int]] = [] # (seq_i, t)
        for si, (_, rs) in enumerate(sorted(byseq.items())):
            rs.sort(key=lambda r: int(r["frame_idx"]))
            n = len(rs)
            fv = np.array([[float(r[f"feat_{i}"]) for i in range(feat_dim)] for r in rs], np.float32)
            self.feats.append(fv)
            der = np.array([r["derived"] for r in rs], np.int64)
            # FC-source cap (de-concentration): cap each sequence's FC contribution to the
            # derived loss at `fc_cap` effective frames. Stops the model MEMORIZING the few
            # FC-heavy sequences (drone-13 = 33% of all FC, train AUROC 0.99 / val 0.53)
            # instead of learning the generalizable geometry rule. ONLY FC frames are
            # down-weighted; CC/CU/LA stay 1.0 so the other classes/heads are unaffected.
            n_fc = int((der == int(DerivedStateV4.FC)).sum())
            w_fc = (float(fc_cap) / n_fc) if (fc_cap and n_fc > fc_cap) else 1.0
            dweight = np.where(der == int(DerivedStateV4.FC), w_fc, 1.0).astype(np.float32)
            self.labels.append({
                "derived": der,
                "fc_subtype": np.array([r["fc_subtype"] for r in rs], np.int64),
                "la_subtype": np.array([r["la_subtype"] for r in rs], np.int64),
                "hazard": np.array([[r["hazard_1"], r["hazard_3"], r["hazard_10"]] for r in rs], np.float32),
                "action": np.array([[r[k] for k in ACT_KEYS] for r in rs], np.float32),
                # per-action trust mask; 1.0 = utility label trustworthy, 0.0 = ignore in loss.
                # default 1.0 when column absent (backward-compat with old shards).
                "act_trust": np.array([[r.get(f"act_trust_{a}", 1.0) for a in ACTION_NAMES] for r in rs], np.float32),
                "do_not_act": np.array([r["do_not_act"] for r in rs], np.float32),
                "tus": np.array([r["template_update_safe"] for r in rs], np.float32),
                "in_win": np.array([r["in_action_window"] for r in rs], np.float32),
                "dweight": dweight,
            })
            self.index += [(si, t) for t in range(n)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, t = self.index[i]
        fv = self.feats[si]
        lo = max(0, t - WINDOW + 1)
        win = fv[lo:t + 1]
        if win.shape[0] < WINDOW:                       # left-pad (causal)
            win = np.concatenate([np.repeat(win[:1], WINDOW - win.shape[0], 0), win], 0)
        lb = self.labels[si]
        return (torch.from_numpy(win),
                int(lb["derived"][t]), int(lb["fc_subtype"][t]), int(lb["la_subtype"][t]),
                torch.from_numpy(lb["hazard"][t]), torch.from_numpy(lb["action"][t]),
                float(lb["do_not_act"][t]), float(lb["tus"][t]), float(lb["in_win"][t]),
                torch.from_numpy(lb["act_trust"][t]), float(lb["dweight"][t]))


def _macro_f1(y_true, y_pred, n_cls):
    f1s = []
    for c in range(n_cls):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)), f1s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="outputs/csc_labels_v4/train_shards.jsonl")
    ap.add_argument("--out_dir", default="outputs/csc_training_v4/csc_v4_r1")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--state_weights", nargs=4, type=float, default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--w_derived", type=float, default=1.0)
    ap.add_argument("--w_fcsub", type=float, default=0.5)
    ap.add_argument("--w_lasub", type=float, default=0.5)
    ap.add_argument("--w_hazard", type=float, default=0.5)
    ap.add_argument("--w_action", type=float, default=1.0)
    ap.add_argument("--w_dna", type=float, default=0.3)
    ap.add_argument("--w_tus", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--balanced_sampler", type=int, default=1,
                    help="1=class-balanced oversampling per batch (V3 fc_source_balanced equiv; "
                         "fixes rare-FC never-learned: plain shuffle + 3x weight left FC recall=0); 0=plain shuffle")
    ap.add_argument("--sampler_fc_boost", type=float, default=1.0,
                    help="extra multiplier on FC sample weight beyond inverse-frequency "
                         "(1.0 = pure inverse-freq ~25%% FC/batch)")
    ap.add_argument("--fc_source_cap", type=int, default=200,
                    help="de-concentration: cap each sequence's FC contribution to the derived "
                         "loss at this many effective frames (down-weight only; 0=off). Stops "
                         "memorizing FC-heavy sequences (drone-13 = 33%% of FC). NOT a sampler.")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = _device(args.device)
    print(f"device={dev} (requested {args.device})", file=sys.stderr)

    rows = []
    with open(args.shards) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
            if args.limit and len(rows) >= args.limit:
                break
    print(f"loaded {len(rows)} shard rows", file=sys.stderr)
    feat_dim = sum(1 for k in rows[0] if k.startswith("feat_")) if rows else N_FEAT
    print(f"feature_dim inferred from shard = {feat_dim}", file=sys.stderr)
    # stratified GROUP split by sequence (no leakage across the val boundary).
    # FC is rare + concentrated, so a random split can land ~0 FC frames in val
    # and the FC head degrades invisibly. Bucket sequences by (dataset, has_FC,
    # has_LA) and assign val_fraction of each bucket to val so FC- and LA-bearing
    # sequences appear in BOTH train and val.
    rng = np.random.default_rng(args.seed)
    seq_rows: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        seq_rows[(r["dataset"], r["sequence"])].append(r)
    buckets: dict[tuple, list[tuple]] = defaultdict(list)
    for seq_key, rs in seq_rows.items():
        ders = [r["derived"] for r in rs]
        has_fc = any(d == int(DerivedStateV4.FC) for d in ders)
        has_la = any(d == int(DerivedStateV4.LA) for d in ders)
        buckets[(seq_key[0], has_fc, has_la)].append(seq_key)
    val_seqs: set = set()
    for bkey in sorted(buckets):
        bseqs = sorted(buckets[bkey])
        rng.shuffle(bseqs)
        nb = len(bseqs)
        k = int(round(nb * args.val_fraction))
        # buckets with >=2 seqs contribute >=1 to each side when possible
        if nb >= 2:
            k = min(max(k, 1), nb - 1)
        val_seqs.update(bseqs[:k])
    tr_rows = [r for r in rows if (r["dataset"], r["sequence"]) not in val_seqs]
    va_rows = [r for r in rows if (r["dataset"], r["sequence"]) in val_seqs]
    tr, va = ShardSet(tr_rows, feat_dim, fc_cap=args.fc_source_cap), ShardSet(va_rows, feat_dim)
    print(f"train frames {len(tr)} / val frames {len(va)} ; val seqs {len(val_seqs)}", file=sys.stderr)
    dc = Counter(r["derived"] for r in tr_rows)
    print(f"train derived dist: {{{', '.join(f'{DerivedStateV4(k).name}:{dc[k]}' for k in sorted(dc))}}}", file=sys.stderr)
    vc = Counter(r["derived"] for r in va_rows)
    print(f"val derived dist:   {{{', '.join(f'{DerivedStateV4(k).name}:{vc[k]}' for k in sorted(vc))}}}", file=sys.stderr)
    print(f"PROOF FC present: train FC frames {dc.get(int(DerivedStateV4.FC), 0)} / "
          f"val FC frames {vc.get(int(DerivedStateV4.FC), 0)}", file=sys.stderr)

    pin = (dev == "cpu")
    if args.balanced_sampler:
        # FIX: V4's plain shuffle + a 3x CE weight left FC (~1.6% of frames) NEVER
        # predicted (val FC recall 0.000 for 5+ epochs). V3-prod used a class-balanced
        # sampler (fc_source_balanced, ~28% FC/batch) and reached FC recall ~0.88 at
        # epoch 1 EVEN FROM SCRATCH (stage1 log). Mirror it: inverse-frequency
        # oversampling so every derived class is ~equally represented per batch.
        cls = np.array([tr.labels[si]["derived"][t] for (si, t) in tr.index], dtype=np.int64)
        freq = np.bincount(cls, minlength=4).astype(np.float64)
        inv = 1.0 / np.clip(freq, 1.0, None)
        inv[int(DerivedStateV4.FC)] *= args.sampler_fc_boost
        wts = inv[cls]
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(wts, dtype=torch.double), num_samples=len(tr), replacement=True)
        tl = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, sampler=sampler,
                                         drop_last=True, num_workers=0)
        p = wts / wts.sum()
        exp = np.bincount(cls, weights=p, minlength=4)
        print(f"balanced sampler ON: expected per-batch class mix "
              f"CC/CU/LA/FC = {exp[0]:.2f}/{exp[1]:.2f}/{exp[2]:.2f}/{exp[3]:.2f}", file=sys.stderr)
    else:
        tl = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                                         drop_last=True, num_workers=0)
    vl = torch.utils.data.DataLoader(va, batch_size=512, shuffle=False, num_workers=0)

    model = CSCv4(feature_dim=feat_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    sw = torch.tensor(args.state_weights, dtype=torch.float32, device=dev)
    # hazard pos-weights from train freq
    hz = np.stack([r and [r["hazard_1"], r["hazard_3"], r["hazard_10"]] for r in tr_rows]).astype(np.float32)
    hz_pos = hz.mean(0).clip(1e-3, 1 - 1e-3)
    hz_pw = torch.tensor((1 - hz_pos) / hz_pos, dtype=torch.float32, device=dev)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "train_log.jsonl", "w")
    best = -1.0

    def run_epoch(loader, train: bool):
        model.train(train)
        agg = defaultdict(float); nb = 0
        yt_d, yp_d, yt_l, yp_l, yt_f, yp_f = [], [], [], [], [], []
        for batch in loader:
            win, d, fcs, las, hzt, act, dna, tus, inw, act_trust, dw = batch
            win = win.float().to(dev); d = d.to(dev); fcs = fcs.to(dev); las = las.to(dev)
            hzt = hzt.float().to(dev); act = act.float().to(dev)
            dna = dna.float().to(dev); tus = tus.float().to(dev); inw = inw.float().to(dev)
            act_trust = act_trust.float().to(dev)   # (B,7) float32 (MPS needs float32, not float64)
            dw = dw.float().to(dev)                  # (B,) FC-source-cap per-sample weight (de-concentration)
            with torch.set_grad_enabled(train):
                out = model.forward(win, last_step_only=True)   # dict of (B,1,dim)
                g = lambda k: out[k][:, 0]                       # (B,dim)
                # derived CE: per-class weight (sw) AND per-sample FC-source-cap weight (dw).
                l_d = (Fn.cross_entropy(g("derived"), d, weight=sw, reduction="none") * dw).sum() / dw.sum().clamp_min(1.0)
                l_fc = Fn.cross_entropy(g("fc_subtype"), fcs)
                l_la = Fn.cross_entropy(g("la_subtype"), las)
                l_hz = Fn.binary_cross_entropy_with_logits(g("hazard"), hzt, pos_weight=hz_pw)
                l_dna = Fn.binary_cross_entropy_with_logits(g("do_not_act")[:, 0], dna)
                l_tus = Fn.binary_cross_entropy_with_logits(g("template_update_safe")[:, 0], tus)
                # action MSE masked to action-window frames AND per-action trust
                m = inw.unsqueeze(1) * act_trust  # (B,7); drop untrusted (action,frame) entries
                denom = m.sum().clamp_min(1.0)
                l_act = (((g("action_utility") - act) ** 2) * m).sum() / denom
                loss = (args.w_derived * l_d + args.w_fcsub * l_fc + args.w_lasub * l_la
                        + args.w_hazard * l_hz + args.w_action * l_act
                        + args.w_dna * l_dna + args.w_tus * l_tus)
                if train:
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            agg["loss"] += float(loss.detach()); agg["d"] += float(l_d.detach()); agg["act"] += float(l_act.detach()); nb += 1
            if not train:
                yt_d.append(d.cpu().numpy()); yp_d.append(g("derived").argmax(1).cpu().numpy())
                yt_l.append(las.cpu().numpy()); yp_l.append(g("la_subtype").argmax(1).cpu().numpy())
                yt_f.append(fcs.cpu().numpy()); yp_f.append(g("fc_subtype").argmax(1).cpu().numpy())
        res = {k: v / max(nb, 1) for k, v in agg.items()}
        if not train and yt_d:
            yt_d_all = np.concatenate(yt_d); yp_d_all = np.concatenate(yp_d)
            res["derived_mF1"], f1s_d = _macro_f1(yt_d_all, yp_d_all, 4)
            res["la_mF1"], _ = _macro_f1(np.concatenate(yt_l), np.concatenate(yp_l), len(LASubtype))
            res["fc_mF1"], _ = _macro_f1(np.concatenate(yt_f), np.concatenate(yp_f), len(FCSubtype))
            # FC = derived class 3 (safety-critical); surface F1, recall, support explicitly
            fc_c = int(DerivedStateV4.FC)
            res["derived_fc_f1"] = float(f1s_d[fc_c])
            fc_tp = int(((yp_d_all == fc_c) & (yt_d_all == fc_c)).sum())
            fc_support = int((yt_d_all == fc_c).sum())
            res["derived_fc_recall"] = fc_tp / fc_support if fc_support else 0.0
            res["derived_fc_support"] = fc_support
        return res

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_r = run_epoch(tl, True)
        va_r = run_epoch(vl, False)
        sched.step()
        sel = (va_r.get("derived_mF1", 0.0) + 0.5 * va_r.get("la_mF1", 0.0)
               + 0.5 * va_r.get("derived_fc_f1", 0.0))   # FC-aware (F1, not recall, so all-FC can't game it)
        rec = {"epoch": ep, "train": tr_r, "val": va_r, "sel": sel, "secs": round(time.time() - t0, 1),
               "lr": opt.param_groups[0]["lr"]}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"ep {ep:2d}/{args.epochs} | loss {tr_r['loss']:.3f} | val derivedF1 {va_r.get('derived_mF1',0):.3f} "
              f"laF1 {va_r.get('la_mF1',0):.3f} fcF1 {va_r.get('fc_mF1',0):.3f} "
              f"FC[f1 {va_r.get('derived_fc_f1',0):.3f} rec {va_r.get('derived_fc_recall',0):.3f} "
              f"n {va_r.get('derived_fc_support',0)}] actMSE {va_r.get('act',0):.4f} "
              f"| sel {sel:.4f} | {rec['secs']}s")
        if sel > best:
            best = sel
            torch.save({"state_dict": model.state_dict(), "feature_dim": feat_dim,
                        "val": va_r, "args": vars(args)}, out_dir / "checkpoint_best.pth")
            print(f"   -> new best {best:.4f} saved")
    torch.save({"state_dict": model.state_dict(), "feature_dim": feat_dim, "args": vars(args)},
               out_dir / "checkpoint_last.pth")
    log.close()
    print(f"done; best sel={best:.4f} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
