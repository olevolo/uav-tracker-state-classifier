#!/usr/bin/env python3
"""Rank trained CSC models on HELD-OUT VALIDATION metrics → pick the best one.

DATA-INTEGRITY (CLAUDE.md): ranking uses each run's own validation split
(LaSOT / GOT-10k / DTB70 / VisDrone / UAVDT). It NEVER touches UAV123 — that is
final-test-only. The model that wins this ranking is the one that later gets the
final UAV123 eval (reported, not tuned).

Per run it reads:
  - train_log.jsonl  -> best epoch = argmax(selection_score); its val metrics +
                        lr + convergence diagnostic. Works WHILE training (the
                        log is rewritten per epoch).
  - val_metrics.json -> authoritative per-class FC precision/recall/F1 +
                        macro-F1 from the banked best checkpoint (only after the
                        run is DONE).

Composite model score (FC-weighted but multi-class aware — see
csc-multi-class-strategy: do NOT collapse to an FC-only detector):

    score = 0.35*macro_F1 + 0.30*FC_F1 + 0.20*failure_AUPRC + 0.15*FC_recall

Convergence check (the whole point of the cosine re-run): best_epoch <= 3 on a
30-epoch budget is flagged EARLY-PEAK (suspect under-converged / LR too high).

Usage:
    .venv/bin/python tools/compare_csc_models.py            # the 3 v3fix candidates
    .venv/bin/python tools/compare_csc_models.py --runs DIR1 DIR2 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "outputs" / "csc_training"

# (label, output-dir-name)
DEFAULT_RUNS = [
    ("R1  (V1, cosine, FCw4.0)",      "sglatrack_v3fix_tcn16_stage1"),
    ("R2  (V2, cosine, FCw4.0)",      "sglatrack_run2_scalectx_tcn16_stage1"),
    ("R1.5 (V1, cosine, FCw3.0)",     "sglatrack_v3fix_r15_tcn16_stage1"),
]


def _safe(x, default=float("nan")):
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def read_train_log(run_dir: Path) -> dict | None:
    """Return the best epoch row (argmax selection_score) + totals, or None."""
    p = run_dir / "train_log.jsonl"
    if not p.exists():
        return None
    rows = []
    for line in p.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return None
    scored = [r for r in rows if r.get("selection_score") is not None]
    if not scored:
        return None
    best = max(scored, key=lambda r: r["selection_score"])
    return {
        "best_epoch": best.get("epoch"),
        "epochs_logged": max(r.get("epoch", 0) for r in rows),
        "best_sel": _safe(best.get("selection_score")),
        "best_macro_f1": _safe(best.get("val_derived_f1")),
        "best_failure_auprc": _safe(best.get("val_failure_auprc")),
        "best_fc_recall": _safe(best.get("val_fc_recall")),
        "best_lr": _safe(best.get("lr")),
        "last_epoch": rows[-1].get("epoch"),
    }


def read_val_metrics(run_dir: Path) -> dict | None:
    """Authoritative per-class metrics from the banked best ckpt (after DONE)."""
    p = run_dir / "val_metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    fc = (d.get("derived_per_state") or {}).get("FALSE_CONFIRMED", {})
    return {
        "macro_f1": _safe(d.get("derived_macro_f1")),
        "failure_auprc": _safe(d.get("failure_auprc")),
        "fc_f1": _safe(fc.get("f1")),
        "fc_precision": _safe(fc.get("precision")),
        "fc_recall": _safe(fc.get("recall")),
        "fc_support": fc.get("support"),
        "n_eval": d.get("n_eval"),
    }


def run_state(run_dir: Path) -> str:
    """DONE if a fresh val_metrics.json exists AND training_done marker; else status."""
    if not run_dir.exists():
        return "MISSING"
    has_val = (run_dir / "val_metrics.json").exists()
    has_log = (run_dir / "train_log.jsonl").exists()
    if not has_log:
        return "NOT_STARTED"
    # val_metrics.json is (re)written only at the very end of a run
    val_mt = (run_dir / "val_metrics.json").stat().st_mtime if has_val else 0
    log_mt = (run_dir / "train_log.jsonl").stat().st_mtime
    if has_val and val_mt >= log_mt:
        return "DONE"
    return "TRAINING"


def composite(macro_f1, fc_f1, failure_auprc, fc_recall) -> float:
    vals = [macro_f1, fc_f1, failure_auprc, fc_recall]
    if any(v != v for v in vals):  # NaN guard
        return float("nan")
    return 0.35 * macro_f1 + 0.30 * fc_f1 + 0.20 * failure_auprc + 0.15 * fc_recall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Run dir names under outputs/csc_training (default: 3 v3fix candidates)")
    args = ap.parse_args()

    if args.runs:
        runs = [(name, name) for name in args.runs]
    else:
        runs = DEFAULT_RUNS

    print("=" * 100)
    print("CSC MODEL COMPARISON — held-out VAL only (NOT UAV123; UAV123 is final-test-only)")
    print("=" * 100)

    table = []
    for label, dirname in runs:
        run_dir = TRAIN_DIR / dirname
        state = run_state(run_dir)
        tl = read_train_log(run_dir)
        vm = read_val_metrics(run_dir)

        # Prefer authoritative val_metrics (DONE); fall back to best-epoch log row.
        if vm is not None and state == "DONE":
            macro_f1, fc_f1 = vm["macro_f1"], vm["fc_f1"]
            failure_auprc, fc_recall = vm["failure_auprc"], vm["fc_recall"]
            fc_prec = vm["fc_precision"]
            src = "val_metrics"
        elif tl is not None:
            macro_f1 = tl["best_macro_f1"]
            failure_auprc, fc_recall = tl["best_failure_auprc"], tl["best_fc_recall"]
            fc_f1, fc_prec = float("nan"), float("nan")  # not in per-epoch log
            src = "train_log(best-ep, FC_F1 pending DONE)"
        else:
            table.append((label, state, None))
            continue

        score = composite(macro_f1, fc_f1, failure_auprc, fc_recall)
        conv = ""
        if tl and tl["best_epoch"] is not None:
            be, tot = tl["best_epoch"], tl["epochs_logged"]
            conv = f"best_ep={be}/{tot}"
            if be <= 3 and tot >= 6:
                conv += " ⚠EARLY-PEAK"
        table.append((label, state, dict(
            macro_f1=macro_f1, fc_f1=fc_f1, fc_prec=fc_prec, fc_recall=fc_recall,
            failure_auprc=failure_auprc, score=score, conv=conv, src=src,
        )))

    # Header
    print(f"\n{'model':<28}{'state':<10}{'macroF1':>8}{'FC_F1':>8}{'FC_P':>7}{'FC_R':>7}{'failAUPRC':>11}{'SCORE':>8}  notes")
    print("-" * 100)
    ranked = []
    for label, state, m in table:
        if m is None:
            print(f"{label:<28}{state:<10}{'—':>8}{'—':>8}{'—':>7}{'—':>7}{'—':>11}{'—':>8}  (no data)")
            continue
        def g(x): return f"{x:.3f}" if x == x else "  -  "
        print(f"{label:<28}{state:<10}{g(m['macro_f1']):>8}{g(m['fc_f1']):>8}{g(m['fc_prec']):>7}"
              f"{g(m['fc_recall']):>7}{g(m['failure_auprc']):>11}{g(m['score']):>8}  {m['conv']} [{m['src']}]")
        if m["score"] == m["score"]:
            ranked.append((m["score"], label, state, m["src"]))

    print("-" * 100)
    if ranked:
        ranked.sort(reverse=True)
        print("\nRANKING (by composite score; ⓘ = still TRAINING/partial, not final):")
        for i, (score, label, state, src) in enumerate(ranked, 1):
            flag = " ⓘ" if "DONE" not in state else ""
            print(f"  {i}. {label:<28} score={score:.4f}  [{state}]{flag}")
        best_label = ranked[0][1]
        all_done = all("DONE" in r[2] for r in ranked) and len(ranked) == len(runs)
        print(f"\n  ➤ Current leader: {best_label}")
        if not all_done:
            print("    (NOT final — some runs still training / FC_F1 pending. Re-run after all DONE.)")
        else:
            print("    ✓ All runs DONE — this is the model to take to final UAV123 eval.")
    else:
        print("\nNo completed/in-progress runs with metrics yet.")


if __name__ == "__main__":
    main()
