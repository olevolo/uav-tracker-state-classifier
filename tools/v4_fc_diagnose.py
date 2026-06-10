#!/usr/bin/env python
"""V4 FC diagnostic: is FC learnable-but-argmax-hidden, or genuinely not learned?

Loads a trained CSCv4 checkpoint, rebuilds the SAME stratified train/val split as
train_csc_v4 (seed/val_fraction), runs the derived head, and reports per split:
  - argmax FC recall/precision/F1 (the metric that showed ~0)
  - P(FC) AUROC / AUPRC (discriminability, threshold-free)
  - P(FC) separation FC-vs-rest + a threshold sweep
Decides: high train AUROC + low val AUROC => generalization / label-feature mismatch.
High AUROC + low argmax-recall => DECISION-RULE problem (threshold/binary head fixes
it). Low AUROC on both train and val => genuine optimization / representation problem.
Offline; train-set val split only (NEVER UAV123).
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT, PROJECT_ROOT / "tools"):
    sys.path.insert(0, str(_p))

from csc_lib.csc.v4.model_v4 import CSCv4
from csc_lib.csc.v4.v4types import DerivedStateV4
import train_csc_v4 as T

NAMES = {0: "CC", 1: "CU", 2: "LA", 3: "FC"}
FC = int(DerivedStateV4.FC)


def _split_rows(rows: list[dict], seed: int, val_fraction: float) -> tuple[list[dict], list[dict], set, dict]:
    """Replicate the exact grouped stratified split from train_csc_v4.py."""
    rng = np.random.default_rng(seed)
    seq_rows: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        seq_rows[(r["dataset"], r["sequence"])].append(r)
    buckets: dict[tuple, list[tuple]] = defaultdict(list)
    for sk, rs in seq_rows.items():
        ders = [r["derived"] for r in rs]
        buckets[(sk[0], any(d == FC for d in ders), any(d == int(DerivedStateV4.LA) for d in ders))].append(sk)
    val_seqs: set = set()
    for bk in sorted(buckets):
        bs = sorted(buckets[bk]); rng.shuffle(bs); nb = len(bs)
        k = int(round(nb * val_fraction))
        if nb >= 2:
            k = min(max(k, 1), nb - 1)
        val_seqs.update(bs[:k])
    tr_rows = [r for r in rows if (r["dataset"], r["sequence"]) not in val_seqs]
    va_rows = [r for r in rows if (r["dataset"], r["sequence"]) in val_seqs]
    return tr_rows, va_rows, val_seqs, seq_rows


def _subset_train_all_fc(ds: T.ShardSet, max_frames: int, seed: int):
    """Sample train for speed while keeping every FC frame."""
    if max_frames <= 0 or len(ds) <= max_frames:
        return ds, f"full={len(ds)}"
    y = np.array([int(ds.labels[si]["derived"][t]) for (si, t) in ds.index], dtype=np.int64)
    fc_idx = np.flatnonzero(y == FC)
    non_idx = np.flatnonzero(y != FC)
    rng = np.random.default_rng(seed)
    remaining = max(0, max_frames - len(fc_idx))
    keep_non = rng.choice(non_idx, size=min(remaining, len(non_idx)), replace=False)
    keep = np.concatenate([fc_idx, keep_non])
    rng.shuffle(keep)
    return torch.utils.data.Subset(ds, keep.tolist()), f"stratified_sample={len(keep)} all_fc={len(fc_idx)}"


def _forward(model: CSCv4, ds, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    P, Y = [], []
    with torch.no_grad():
        for batch in loader:
            win = batch[0].float().to(device)
            out = model.forward(win, last_step_only=True)
            P.append(torch.softmax(out["derived"][:, 0], dim=1).cpu().numpy())
            Y.append(np.asarray(batch[1]))
    return np.concatenate(P), np.concatenate(Y)


def _report_split(name: str, note: str, P: np.ndarray, Y: np.ndarray) -> dict:
    pfc = P[:, FC]
    isfc = (Y == FC).astype(int)
    argmax = P.argmax(1)

    # ---- argmax FC (the metric that showed ~0) ----
    tp = int(((argmax == FC) & (Y == FC)).sum())
    fp = int(((argmax == FC) & (Y != FC)).sum())
    fn = int(((argmax != FC) & (Y == FC)).sum())
    rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\n================ {name.upper()} ({note}) ================")
    print("=== ARGMAX FC (what the training metric reported) ===")
    print(f"FC support={int(isfc.sum())}  argmax: recall={rec:.3f} precision={prec:.3f} f1={f1:.3f} "
          f"(predicted FC on {int((argmax == FC).sum())} frames)")
    dest = Counter(argmax[Y == FC].tolist())
    print("  true-FC frames argmax-classified as:", {NAMES[k]: v for k, v in sorted(dest.items())})

    # ---- threshold-free P(FC): the real test ----
    auroc = roc_auc_score(isfc, pfc) if isfc.sum() and isfc.sum() < len(isfc) else float("nan")
    auprc = average_precision_score(isfc, pfc) if isfc.sum() else float("nan")
    print("\n=== THRESHOLD-FREE P(FC) (the decisive test) ===")
    print(f"FC AUROC = {auroc:.4f}   FC AUPRC = {auprc:.4f}   (base rate {isfc.mean():.4f})")
    if isfc.sum():
        print(f"P(FC) median: FC-frames={np.median(pfc[isfc == 1]):.4f}  non-FC={np.median(pfc[isfc == 0]):.4f}")
        for q in (50, 75, 90, 95, 99):
            print(f"  P(FC) p{q:>2}: FC={np.percentile(pfc[isfc == 1], q):.4f}  non-FC={np.percentile(pfc[isfc == 0], q):.4f}")

    print("\n=== threshold sweep on P(FC) ===")
    print("  thr   recall  prec    #pred")
    for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        pred = pfc >= thr
        tp_ = int((pred & (isfc == 1)).sum()); fp_ = int((pred & (isfc == 0)).sum())
        r_ = tp_ / max(int(isfc.sum()), 1); p_ = tp_ / max(tp_ + fp_, 1)
        print(f"  {thr:.1f}  {r_:.3f}   {p_:.3f}   {int(pred.sum())}")

    return {
        "name": name,
        "argmax_recall": rec,
        "argmax_precision": prec,
        "argmax_f1": f1,
        "auroc": float(auroc),
        "auprc": float(auprc),
        "base_rate": float(isfc.mean()),
        "fc_support": int(isfc.sum()),
    }


def _print_top_fc_sequences(label: str, rows: list[dict], top_k: int) -> None:
    seq = defaultdict(lambda: [0, 0])
    for r in rows:
        key = (r["dataset"], r["sequence"])
        seq[key][0] += 1
        seq[key][1] += int(r["derived"] == FC)
    ranked = [(fc, n, ds, name) for (ds, name), (n, fc) in seq.items() if fc]
    ranked.sort(reverse=True)
    print(f"\n=== TOP {label.upper()} FC-BEARING SEQUENCES ===")
    for fc, n, ds, name in ranked[:top_k]:
        print(f"  {fc:5d} / {n:5d}  {ds:12s}  {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/csc_training_v4/csc_v4_r1/checkpoint_best.pth")
    ap.add_argument("--shards", default="outputs/csc_labels_v4/train_shards.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--train_max_frames", type=int, default=100000,
                    help="Train diagnostic cap; keeps all train FC frames and samples non-FC frames.")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--top_sequences", type=int, default=15)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    feat_dim = int(ck.get("feature_dim", 25))
    print(f"checkpoint feat_dim={feat_dim} balanced_sampler={ck.get('args',{}).get('balanced_sampler')}", file=sys.stderr)

    rows = [json.loads(l) for l in open(args.shards) if l.strip()]
    shard_feat_dim = sum(1 for k in rows[0] if k.startswith("feat_")) if rows else 0
    if shard_feat_dim != feat_dim:
        raise SystemExit(
            f"ERROR: checkpoint feature_dim={feat_dim} but shard has {shard_feat_dim} feature columns. "
            "Rebuild/retrain a matching checkpoint; refusing to silently slice features."
        )
    tr_rows, va_rows, val_seqs, _ = _split_rows(rows, args.seed, args.val_fraction)
    print(f"split rows: train={len(tr_rows)} val={len(va_rows)} val_seqs={len(val_seqs)}", file=sys.stderr)

    model = CSCv4(feature_dim=feat_dim).to(args.device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    reports = []
    if args.split in ("train", "both"):
        tr = T.ShardSet(tr_rows, feat_dim)
        tr_eval, note = _subset_train_all_fc(tr, args.train_max_frames, args.seed)
        P, Y = _forward(model, tr_eval, args.device, args.batch_size)
        reports.append(_report_split("train", note, P, Y))
        _print_top_fc_sequences("train", tr_rows, args.top_sequences)

    if args.split in ("val", "both"):
        va = T.ShardSet(va_rows, feat_dim)
        P, Y = _forward(model, va, args.device, args.batch_size)
        reports.append(_report_split("val", f"full={len(va)}", P, Y))
        _print_top_fc_sequences("val", va_rows, args.top_sequences)

    print("\n=== VERDICT ===")
    by_name = {r["name"]: r for r in reports}
    if "train" in by_name and "val" in by_name:
        train_auc, val_auc = by_name["train"]["auroc"], by_name["val"]["auroc"]
        if np.isfinite(train_auc) and np.isfinite(val_auc) and train_auc >= 0.85 and val_auc < 0.70:
            print(f"Train FC AUROC {train_auc:.3f} HIGH but val {val_auc:.3f} LOW -> GENERALIZATION / LABEL-FEATURE MISMATCH. "
                  "Do not tune thresholds; audit FC labels/domains and move FC to an identity/candidate verifier.")
        elif np.isfinite(train_auc) and train_auc < 0.70:
            print(f"Train FC AUROC {train_auc:.3f} LOW -> OPTIMIZATION/REPRESENTATION problem even in-sample.")
        else:
            print("Train/val gap is not catastrophic; inspect threshold and precision targets.")
        return 0

    auroc = reports[0]["auroc"] if reports else float("nan")
    if not np.isfinite(auroc):
        print("FC AUROC undefined (no FC in selected split).")
    elif auroc >= 0.85:
        print(f"FC AUROC {auroc:.3f} HIGH -> FC IS learnable; argmax hides it -> DECISION-RULE problem "
              "(calibrated threshold / binary FC head fixes it, no oversampling needed).")
    elif auroc >= 0.70:
        print(f"FC AUROC {auroc:.3f} MODERATE -> partially learned; threshold helps but head/loss redesign "
              "needed for strong FC.")
    else:
        print(f"FC AUROC {auroc:.3f} LOW -> genuine LEARNING problem; threshold alone won't save it "
              "(need head decoupling + asymmetric/logit-adjusted loss + hard negatives).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
