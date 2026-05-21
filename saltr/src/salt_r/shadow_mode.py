"""shadow_mode.py — SALT-RD Stage 1 Shadow Mode observer.

Runs SALT-RD alongside pre-recorded tracker telemetry. TSA/SALTRunner output is
treated as read-only ground truth — nothing is modified. SALT-RD observes and
logs what it WOULD have recommended per frame.

Per-frame output
----------------
- p_fc, p_ifd10, p_ifd20, p_recoverable, p_needs_full_compute
- learned_state: TRUSTED_TRACKING | LOW_EVIDENCE_TRACKING | FALSE_CONFIRMED_RISK |
                 PROACTIVE_DYNAMIC_RISK | REACQUIRE_NEEDED
- proposed_action: allow | verify | block (template_update)
                   none | abstain | run (recovery)
- tsa_likely_action: inferred from confirmed_streak / low_conf_streak features
- disagree_template: SALT-RD says block, TSA would allow
- disagree_reinit: SALT-RD says abstain, TSA would reinit

Exit gates before Stage 2 (Advisory/Veto):
  - wrir = 0 (vacuously true in shadow)
  - no leakage: this script never writes to training data
  - calibration from val only; diagnostic never used for threshold tuning

Usage
-----
    python -m salt_r.shadow_mode \\
        --npz saltr/data/salt_rd_v2_labels.npz \\
        --checkpoint saltr/checkpoints/production/saltrd_best.pt \\
        --memory-sidecar saltr/data/salt_rd_memory_sidecar.npz \\
        --split val \\
        --output saltr/results/shadow_mode_val.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Learned-state classifier
# ---------------------------------------------------------------------------

def classify_learned_state(probs: dict[str, float]) -> str:
    """Map SALT-RD head probabilities to a named learned state.

    Priority order mirrors the Phase-5 state machine in the HANDOFF.
    """
    p_fc  = probs.get("false_confirmed", 0.0)
    p_rec = probs.get("recoverable", 0.0)
    p_i10 = probs.get("imminent_failure_dynamic_10", probs.get("failure_in_10", 0.0))
    p_i20 = probs.get("imminent_failure_dynamic_20", probs.get("failure_in_20", 0.0))

    if p_fc >= 0.60:
        return "FALSE_CONFIRMED_RISK"
    if p_rec >= 0.60:
        return "REACQUIRE_NEEDED"
    if max(p_i10, p_i20) >= 0.50:
        return "PROACTIVE_DYNAMIC_RISK"
    if p_fc >= 0.30:
        return "LOW_EVIDENCE_TRACKING"
    return "TRUSTED_TRACKING"


# ---------------------------------------------------------------------------
# Policy thresholds (runner-up config from policy sweep)
# ---------------------------------------------------------------------------

_FC_BLOCK    = 0.60   # block template update
_FC_VERIFY   = 0.40   # verify (but allow)
_FC_ABSTAIN  = 0.70   # abstain from reinit
_IFD_FULL    = 0.50   # force full compute

# Advisory/veto mode — same block threshold as shadow, validated by msu/wrir gates
_FC_BLOCK_ADVISORY   = 0.60
_FC_VERIFY_ADVISORY  = 0.40
_FC_ABSTAIN_ADVISORY = 0.70


def _apply_shadow_policy(
    probs: dict[str, float],
    fc_block: float = _FC_BLOCK,
    fc_verify: float = _FC_VERIFY,
    fc_abstain: float = _FC_ABSTAIN,
    ifd_full: float = _IFD_FULL,
) -> dict[str, str]:
    p_fc = probs.get("false_confirmed", 0.0)
    p_i10 = probs.get("imminent_failure_dynamic_10", probs.get("failure_in_10", 0.0))
    p_i20 = probs.get("imminent_failure_dynamic_20", probs.get("failure_in_20", 0.0))
    p_rec = probs.get("recoverable", 0.0)

    if p_fc >= fc_block:
        template = "block"
        recovery = "abstain"
    elif p_fc >= fc_verify:
        template = "verify"
        recovery = "none"
    else:
        template = "allow"
        recovery = "run" if p_rec >= 0.60 else "none"

    # Only relevant when fc_abstain < fc_block (non-default): escalate recovery to
    # "abstain" even when template is only "verify"/"allow". With default thresholds
    # (fc_abstain=0.70 > fc_block=0.60) this is a no-op — block already sets "abstain".
    if fc_abstain < fc_block and p_fc >= fc_abstain:
        recovery = "abstain"

    compute = "full" if (p_fc >= fc_block or max(p_i10, p_i20) >= ifd_full) else "normal"

    return {"template_update": template, "recovery": recovery, "compute": compute}


# ---------------------------------------------------------------------------
# TSA state inference from telemetry features
# ---------------------------------------------------------------------------

# Feature index constants (must match collect_features.FEATURE_NAMES)
_CONFIRMED_STREAK_IDX = 13  # confirmed_streak
_LOW_CONF_STREAK_IDX  = 14  # low_conf_streak


def _infer_tsa_action(feat: np.ndarray) -> dict[str, str]:
    """Infer what TSA is likely doing based on telemetry features."""
    confirmed_streak = float(feat[_CONFIRMED_STREAK_IDX])
    low_conf_streak  = float(feat[_LOW_CONF_STREAK_IDX])
    # TSA updates template when confirmed for several consecutive frames
    tsa_template = "allow" if confirmed_streak >= 3 else "hold"
    # TSA reinits when low_conf_streak is very high (approaching LOST)
    tsa_recovery = "reinit" if low_conf_streak >= 10 else "none"
    return {"template_update": tsa_template, "recovery": tsa_recovery}


# ---------------------------------------------------------------------------
# Per-frame analysis
# ---------------------------------------------------------------------------

def _analyse_frame(
    feat_37: np.ndarray,
    label_row: np.ndarray,
    label_names: list[str],
    model: Any,
    device: str,
    window_buf: list[np.ndarray],
    window_size: int,
    fc_block: float = _FC_BLOCK,
    fc_verify: float = _FC_VERIFY,
    fc_abstain: float = _FC_ABSTAIN,
) -> dict[str, Any] | None:
    """Return per-frame shadow entry, or None if window not yet full."""
    window_buf.append(feat_37.astype(np.float32))
    if len(window_buf) > window_size:
        window_buf.pop(0)
    if len(window_buf) < window_size:
        return None

    window = np.stack(window_buf, axis=0)  # (window_size, 37)
    probs: dict[str, float] = model.predict_single(window, device=device)

    learned_state = classify_learned_state(probs)
    action = _apply_shadow_policy(probs, fc_block=fc_block, fc_verify=fc_verify, fc_abstain=fc_abstain)
    tsa_action = _infer_tsa_action(feat_37[:28])  # telemetry-only features

    fc_idx  = label_names.index("false_confirmed")
    cor_idx = label_names.index("correct")
    label_fc  = int(label_row[fc_idx])
    label_cor = int(label_row[cor_idx])

    # Disagreements
    disagree_template = (
        action["template_update"] == "block"
        and tsa_action["template_update"] == "allow"
    )
    disagree_reinit = (
        action["recovery"] == "abstain"
        and tsa_action["recovery"] == "reinit"
    )
    # False alarm: SALT-RD blocks but tracker is actually correct
    false_alarm = (action["template_update"] == "block") and (label_cor == 1)
    # True block: SALT-RD blocks and tracker is false-confirmed
    true_block  = (action["template_update"] == "block") and (label_fc == 1)

    return {
        "p_fc":   round(probs.get("false_confirmed", 0.0), 4),
        "p_i10":  round(probs.get("imminent_failure_dynamic_10",
                                  probs.get("failure_in_10", 0.0)), 4),
        "p_i20":  round(probs.get("imminent_failure_dynamic_20",
                                  probs.get("failure_in_20", 0.0)), 4),
        "p_rec":  round(probs.get("recoverable", 0.0), 4),
        "learned_state": learned_state,
        "template_action": action["template_update"],
        "recovery_action": action["recovery"],
        "tsa_template":    tsa_action["template_update"],
        "tsa_reinit":      int(tsa_action["recovery"] == "reinit"),
        "label_fc":   label_fc,
        "label_cor":  label_cor,
        "disagree_template": int(disagree_template),
        "disagree_reinit":   int(disagree_reinit),
        "false_alarm":  int(false_alarm),
        "true_block":   int(true_block),
    }


# ---------------------------------------------------------------------------
# Per-sequence runner
# ---------------------------------------------------------------------------

def run_sequence(
    seq_key: str,
    features: np.ndarray,   # (T, 28)
    memory_feats: np.ndarray | None,  # (T, 9) or None
    labels: np.ndarray,     # (T, n_labels)
    label_names: list[str],
    model: Any,
    device: str,
    window_size: int,
    fc_block: float = _FC_BLOCK,
    fc_verify: float = _FC_VERIFY,
    fc_abstain: float = _FC_ABSTAIN,
) -> dict[str, Any]:
    T = len(features)
    mem_dim = memory_feats.shape[1] if memory_feats is not None else 0
    window_buf: list[np.ndarray] = []
    frames: list[dict[str, Any]] = []

    for t in range(T):
        if memory_feats is not None:
            feat_full = np.concatenate([features[t], memory_feats[t]])
        else:
            feat_full = features[t].copy()

        entry = _analyse_frame(
            feat_full, labels[t], label_names,
            model, device, window_buf, window_size,
            fc_block=fc_block, fc_verify=fc_verify, fc_abstain=fc_abstain,
        )
        if entry is not None:
            frames.append(entry)

    if not frames:
        return {"seq_key": seq_key, "n_frames": T, "n_active": 0, "stats": {}}

    # Aggregate stats
    n = len(frames)
    n_fc   = sum(f["label_fc"] for f in frames)
    n_cor  = sum(f["label_cor"] for f in frames)
    n_block = sum(1 for f in frames if f["template_action"] == "block")
    n_verify = sum(1 for f in frames if f["template_action"] == "verify")
    n_disagree_tmpl = sum(f["disagree_template"] for f in frames)
    n_disagree_reinit = sum(f["disagree_reinit"] for f in frames)
    n_false_alarm = sum(f["false_alarm"] for f in frames)
    n_true_block  = sum(f["true_block"] for f in frames)
    n_tsa_allow   = sum(1 for f in frames if f["tsa_template"] == "allow")
    n_tsa_reinit  = sum(f.get("tsa_reinit", 0) for f in frames)

    state_counts: dict[str, int] = defaultdict(int)
    for f in frames:
        state_counts[f["learned_state"]] += 1

    # Block rate conditioned on label
    block_given_fc  = n_true_block  / n_fc  if n_fc  > 0 else float("nan")
    block_given_cor = n_false_alarm / n_cor if n_cor > 0 else float("nan")

    return {
        "seq_key": seq_key,
        "n_frames": T,
        "n_active": n,
        "stats": {
            "n_fc":   n_fc,
            "n_correct": n_cor,
            "n_block": n_block,
            "n_verify": n_verify,
            "n_true_block":  n_true_block,
            "n_false_alarm": n_false_alarm,
            "n_disagree_template": n_disagree_tmpl,
            "n_disagree_reinit":   n_disagree_reinit,
            "n_tsa_allow": n_tsa_allow,
            "n_tsa_reinit": n_tsa_reinit,
            "block_rate_when_fc":      round(block_given_fc,  4) if not __import__("math").isnan(block_given_fc)  else None,
            "block_rate_when_correct": round(block_given_cor, 4) if not __import__("math").isnan(block_given_cor) else None,
            "learned_state_dist": dict(state_counts),
        },
        "frames": frames,  # full per-frame log
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz",        required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--memory-sidecar", default=None)
    parser.add_argument("--split",      default="val",
                        choices=["train", "val", "diagnostic"])
    parser.add_argument("--output",     required=True)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--no-frames",  action="store_true",
                        help="Omit per-frame log from JSON (smaller output)")
    parser.add_argument("--mode", choices=["shadow", "advisory", "stage3"], default="shadow",
                        help="shadow = observe only (Stage 1); advisory = conservative veto (Stage 2); "
                             "stage3 = primary learned controller reporting (Stage 3)")
    args = parser.parse_args(argv)

    # Select policy thresholds
    if args.mode in ("advisory", "stage3"):
        fc_block  = _FC_BLOCK_ADVISORY
        fc_verify = _FC_VERIFY_ADVISORY
        fc_abstain = _FC_ABSTAIN_ADVISORY
    else:
        fc_block  = _FC_BLOCK
        fc_verify = _FC_VERIFY
        fc_abstain = _FC_ABSTAIN

    import sys
    repo_root = Path(__file__).resolve().parents[4]
    for p in [str(repo_root / "src"), str(repo_root / "saltr" / "src")]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from salt_r.model import build_model

    print(f"[shadow_mode] Loading checkpoint: {args.checkpoint}", flush=True)
    model = build_model(args.checkpoint, device=args.device)
    model.eval()

    import torch
    ck = torch.load(args.checkpoint, map_location="cpu")
    window_size = int(ck.get("window_size", 20))
    feat_names  = list(ck.get("feature_names", []))
    if feat_names:
        assert len(feat_names) > _LOW_CONF_STREAK_IDX, "Feature names too short for streak indices"
        assert feat_names[_CONFIRMED_STREAK_IDX] == "confirmed_streak", (
            f"Expected feat_names[{_CONFIRMED_STREAK_IDX}]='confirmed_streak', got {feat_names[_CONFIRMED_STREAK_IDX]!r}"
        )
        assert feat_names[_LOW_CONF_STREAK_IDX] == "low_conf_streak", (
            f"Expected feat_names[{_LOW_CONF_STREAK_IDX}]='low_conf_streak', got {feat_names[_LOW_CONF_STREAK_IDX]!r}"
        )
    mem_dim     = int(ck.get("memory_dim", 0))
    label_names_ck = list(ck.get("label_names", []))
    print(f"[shadow_mode] window_size={window_size}  feat_dim={len(feat_names)}  memory_dim={mem_dim}", flush=True)

    print(f"[shadow_mode] Loading NPZ: {args.npz}", flush=True)
    npz = np.load(args.npz, allow_pickle=True)
    label_names = [str(x) for x in npz["label_names"].tolist()]

    sidecar = None
    if args.memory_sidecar:
        print(f"[shadow_mode] Loading sidecar: {args.memory_sidecar}", flush=True)
        sidecar = np.load(args.memory_sidecar, allow_pickle=True)

    # Select sequences for this split
    seq_keys = sorted(
        k.replace("split/", "")
        for k in npz.files
        if k.startswith("split/") and str(npz[k]) == args.split
    )
    print(f"[shadow_mode] Split={args.split}  sequences={len(seq_keys)}", flush=True)

    results = []
    for seq_key in seq_keys:
        feat_key  = f"features/{seq_key}"
        label_key = f"labels/{seq_key}"
        mem_key   = f"memory_features/{seq_key}"

        if feat_key not in npz.files or label_key not in npz.files:
            print(f"  [skip] {seq_key} — missing features/labels", flush=True)
            continue

        features = npz[feat_key].astype(np.float32)     # (T, 28)
        labels   = npz[label_key].astype(np.float32)    # (T, n_labels)
        T = len(features)

        memory_feats = None
        if sidecar is not None and mem_key in sidecar.files:
            memory_feats = sidecar[mem_key][:T].astype(np.float32)  # (T, 9)
        elif mem_dim > 0:
            memory_feats = np.zeros((T, mem_dim), dtype=np.float32)

        seq_result = run_sequence(
            seq_key=seq_key,
            features=features,
            memory_feats=memory_feats,
            labels=labels,
            label_names=label_names,
            model=model,
            device=args.device,
            window_size=window_size,
            fc_block=fc_block,
            fc_verify=fc_verify,
            fc_abstain=fc_abstain,
        )
        if args.no_frames:
            seq_result.pop("frames", None)
        results.append(seq_result)

        s = seq_result["stats"]
        print(
            f"  {seq_key:<42} "
            f"fc={s.get('n_fc',0):>4}  "
            f"block={s.get('n_block',0):>4}  "
            f"disagree={s.get('n_disagree_template',0):>4}  "
            f"FA={s.get('block_rate_when_correct') or 0:.2f}  "
            f"cov={s.get('block_rate_when_fc') or 0:.2f}",
            flush=True,
        )

    # Global aggregates
    agg: dict[str, Any] = defaultdict(int)
    state_global: dict[str, int] = defaultdict(int)
    for r in results:
        s = r["stats"]
        for k in ("n_fc","n_correct","n_block","n_verify",
                  "n_true_block","n_false_alarm",
                  "n_disagree_template","n_disagree_reinit",
                  "n_tsa_allow","n_tsa_reinit"):
            agg[k] += s.get(k) or 0  # guard None from old JSON versions
        agg["n_active"] += r.get("n_active", 0)
        for state, cnt in s.get("learned_state_dist", {}).items():
            state_global[state] += cnt

    total_fc  = agg["n_fc"]
    total_cor = agg["n_correct"]
    total_block = agg["n_block"]

    tp = agg["n_true_block"]
    fp = agg["n_false_alarm"]

    import math
    coverage   = tp / total_fc  if total_fc  > 0 else float("nan")
    false_alarm_rate = fp / total_cor if total_cor > 0 else float("nan")
    block_rate = total_block / agg["n_active"] if agg["n_active"] > 0 else float("nan")
    msu  = fp / agg["n_tsa_allow"] if agg["n_tsa_allow"] > 0 else float("nan")
    wrir = agg["n_disagree_reinit"] / agg["n_tsa_reinit"] if agg["n_tsa_reinit"] > 0 else 0.0

    print()
    print("=" * 70)
    print(f"  SALT-RD {args.mode.upper()} MODE  —  split={args.split}  "
          f"seqs={len(results)}  frames={agg['n_active']:,}")
    print(f"  thresholds: fc_block={fc_block}  fc_verify={fc_verify}  fc_abstain={fc_abstain}")
    print("=" * 70)
    print(f"  GT label stats:  fc={total_fc:,}  correct={total_cor:,}")
    print()
    print(f"  SALT-RD recommendations:")
    print(f"    block template update:  {total_block:>6,} frames  "
          f"({100*block_rate:.1f}% of active)")
    print(f"    verify (but allow):     {agg['n_verify']:>6,} frames")
    print(f"    disagree with TSA:      {agg['n_disagree_template']:>6,} frames")
    print()
    print(f"  Quality:")
    print(f"    coverage  (block|fc=1):      {coverage:.3f}  "
          f"({tp:,}/{total_fc:,} fc frames blocked)")
    print(f"    false alarm (block|correct): {false_alarm_rate:.3f}  "
          f"({fp:,}/{total_cor:,} correct frames blocked)")
    print(f"    msu (missed safe updates):   {msu:.3f}  "
          f"({fp:,}/{agg['n_tsa_allow']:,} of TSA-allow frames blocked)")
    print(f"    wrir (wrong reinit rate):    {wrir:.4f}  "
          f"({agg['n_disagree_reinit']:,}/{agg['n_tsa_reinit']:,} TSA-reinit blocked)")
    if args.mode == "advisory":
        print()
        print(f"  Stage 2 Advisory Gates:")
        wrir_pass = wrir == 0.0
        msu_pass  = not math.isnan(msu) and msu < 0.40
        print(f"    wrir == 0.0:   {'✅ PASS' if wrir_pass else '❌ FAIL'}  ({wrir:.4f})")
        print(f"    msu  < 0.40:   {'✅ PASS' if msu_pass  else '❌ FAIL'}  ({msu:.3f})")
        overall = "GO" if (wrir_pass and msu_pass) else "STOP"
        print(f"    → {overall}")
    if args.mode == "stage3":
        print()
        print(f"  Stage 3 Primary Learned Controller — Gates:")
        wrir_pass = wrir == 0.0
        msu_pass  = not math.isnan(msu) and msu < 0.40
        print(f"    wrir == 0.0:   {'✅ PASS' if wrir_pass else '❌ FAIL'}  ({wrir:.4f})")
        print(f"    msu  < 0.40:   {'✅ PASS' if msu_pass  else '❌ FAIL'}  ({msu:.3f})")
        overall = "GO" if (wrir_pass and msu_pass) else "STOP"
        print(f"    → Stage 3 deployment: {overall}")
    print()
    print(f"  Learned state distribution (over active frames):")
    for state in ["TRUSTED_TRACKING","LOW_EVIDENCE_TRACKING","FALSE_CONFIRMED_RISK",
                  "PROACTIVE_DYNAMIC_RISK","REACQUIRE_NEEDED"]:
        cnt = state_global.get(state, 0)
        pct = 100 * cnt / agg["n_active"] if agg["n_active"] > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {state:<30} {cnt:>6,}  ({pct:5.1f}%)  {bar}")
    print("=" * 70)

    # Save
    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "memory_sidecar": args.memory_sidecar,
        "split": args.split,
        "mode": args.mode,
        "n_sequences": len(results),
        "policy": {
            "fc_block": fc_block, "fc_verify": fc_verify,
            "fc_abstain": fc_abstain, "ifd_full": _IFD_FULL,
        },
        "aggregate": {
            **dict(agg),
            "state_distribution": dict(state_global),
            "coverage_block_when_fc": round(coverage, 4) if not math.isnan(coverage) else None,
            "false_alarm_rate_block_when_correct": round(false_alarm_rate, 4) if not math.isnan(false_alarm_rate) else None,
            "overall_block_rate": round(block_rate, 4) if not math.isnan(block_rate) else None,
            "msu": round(msu, 4) if not math.isnan(msu) else None,
            "wrir": round(wrir, 4),
        },
        "sequences": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[shadow_mode] Saved → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
