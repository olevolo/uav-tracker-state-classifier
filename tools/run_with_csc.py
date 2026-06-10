"""Run any tracker adapter with CSC inference (passive or control mode).

Output schema:
    outputs/csc_runs/<tracker>_<dataset>_<split>_<csc_model>/
        predictions/<seq>.txt
        telemetry/<seq>.jsonl
        states/<seq>.jsonl          ← per-frame CSC output
        metrics.json                ← runtime / FPS summary

``states/<seq>.jsonl`` row keys (one per non-init frame):
    frame_idx, localization_probs, confidence_probs,
    predicted_localization, predicted_confidence,
    derived_state, false_confirmed_flag,
    risk_score, aux_probs,
    should_freeze_template, should_expand_search,
    should_request_redetection, should_skip_template_update,
    tracker_latency_ms, csc_latency_ms

Control mode (--csc_mode control):
    Only template-update freeze is implemented for trackers that expose
    ``set_update_enabled(bool)``.  Search-expansion and redetection hints
    are logged but produce no behaviour change (TODO for Phase 3 / CSC §10).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))   sys.path.insert(0, str(PROJECT_ROOT))

# Limit PyTorch thread-pool to avoid tracker/CSC inter-thread contention.
# Hardware: M4 Pro (10 P-cores + 4 E-cores). Under typical dev load with
# concurrent training/baseline jobs (5+ processes), threads=2 keeps total
# OMP workers ≤ P-core count and avoids context-switch thrashing.
# Measured under load: threads=4 → 63.6 FPS, threads=6 → 42.4 FPS (-33%).
# For clean paper-quality benchmark (kill -STOP others), sweep [4,6,8,10]
# on a quiet machine — solo optimum is likely higher than 2.
import torch as _torch
_torch.set_num_threads(4)
_torch.set_num_interop_threads(1)

_TRACKER_NAMES = ["sglatrack", "ostrack", "ortrack", "ortrack_deit", "avtrack", "evptrack", "fartrack", "uetrack"]


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _load_dataset(name: str, split: str):
    import csc_uav_tracking  # noqa: F401
    from csc_uav_tracking.registry import DATASETS

    if name not in DATASETS.names():
        raise SystemExit(
            f"unknown dataset {name!r}. registered: {DATASETS.names()}"
        )
    if name == "got10k":
        return DATASETS.build(name, split=split)
    return DATASETS.build(name)


def _load_tracker(name: str, weights_path: Optional[str], device: str):
    """Import the named adapter (triggers @TRACKERS.register) then build."""
    _adapter_map = {
        "sglatrack":    "uav_tracker.trackers.sglatrack",
        "ostrack":      "uav_tracker.trackers.ostrack",
        "ortrack":      "uav_tracker.trackers.transformer.ortrack",
        "ortrack_deit": "uav_tracker.trackers.transformer.ortrack",
        "avtrack":      "uav_tracker.trackers.avtrack",
        "evptrack":     "uav_tracker.trackers.evptrack",
        "fartrack":     "uav_tracker.trackers.fartrack",
        "uetrack":      "uav_tracker.trackers.uetrack",
    }
    if name not in _adapter_map:
        raise SystemExit(
            f"unknown tracker {name!r}. supported: {_TRACKER_NAMES}"
        )
    try:
        import importlib
        importlib.import_module(_adapter_map[name])
        from uav_tracker.registry import TRACKERS
    except Exception as exc:
        raise SystemExit(
            f"Failed to import tracker adapter {name!r}: {exc!r}"
        )
    if name not in TRACKERS:
        raise SystemExit(
            f"{name!r} not in registry after import. Known: {TRACKERS.names()}"
        )
    kwargs: dict[str, Any] = {"device": device}
    if weights_path:
        kwargs["weights_path"] = weights_path
    return TRACKERS.build(name, **kwargs)


def _bbox_tuple(b) -> tuple[float, float, float, float]:
    if b is None:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(b.x), float(b.y), float(b.w), float(b.h))


def _telemetry_from_state(state) -> dict[str, Any]:
    """Best-effort extraction of confidence/APCE/PSR/score-map stats."""
    out: dict[str, Any] = {}
    if state is None:
        return out
    for attr in ("confidence", "apce", "psr", "response_entropy"):
        v = getattr(state, attr, None)
        if v is not None:
            try:
                out[attr] = float(v)
            except (TypeError, ValueError):
                pass
    raw = getattr(state, "raw", {}) or getattr(state, "aux", {}) or {}
    if isinstance(raw, dict) and "raw" in raw:
        raw = raw["raw"]
    for key in (
        "score", "score_max", "response_max", "response_mean", "response_std",
        "token_keep_ratio", "active_layers", "search_factor", "template_age",
    ):
        if key in raw and raw[key] is not None:
            try:
                out[key] = float(raw[key])
            except (TypeError, ValueError):
                pass
    sm = raw.get("score_map_stats") if isinstance(raw, dict) else None
    if isinstance(sm, dict):
        for k, v in sm.items():
            try:
                out[f"sm_{k}"] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def _try_set_update_enabled(tracker, enabled: bool) -> bool:
    """Apply template-update freeze if the tracker exposes the hook."""
    fn = getattr(tracker, "set_update_enabled", None)
    if callable(fn):
        try:
            fn(enabled)
            return True
        except Exception:
            return False
    # TODO (Phase 3): wire freeze into tracker adapters that carry template-update
    # logic internally (SGLATrack.try_update_template / OSTrack template refresh).
    return False


def _try_set_search_factor(tracker, factor: float) -> bool:
    """Widen/restore the search region if the tracker exposes the hook.

    SGLATrack.set_search_factor(f) sets the search-crop area factor for the next
    update() (default 4.0). Larger = wider search = the proper LOST recovery
    action. Returns True if applied. No-op (False) for trackers without the hook.
    """
    fn = getattr(tracker, "set_search_factor", None)
    if callable(fn):
        try:
            fn(factor)
            return True
        except Exception:
            return False
    return False


def _try_override_search_center(tracker, cx: float, cy: float, w: float, h: float) -> bool:
    """Relocate the NEXT-frame search center onto a re-detection candidate.

    SGLATrack.override_search_center(cx,cy,w,h) sets self._state so the next
    update() crops around (cx,cy) instead of the last (possibly wrong) box. This
    is the real re-localise action for a genuine loss / distractor capture, as
    opposed to merely widening the existing (mis-centred) crop. Returns True if
    applied; no-op (False) for trackers without the hook.
    """
    fn = getattr(tracker, "override_search_center", None)
    if callable(fn):
        try:
            fn(float(cx), float(cy), float(w), float(h))
            return True
        except Exception:
            return False
    return False


def _parse_float_csv(value: str) -> tuple[float, ...]:
    out: list[float] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            raise SystemExit(f"invalid comma-separated float value: {value!r}")
    return tuple(out)


def _try_sgla_redetect(
    tracker,
    frame: np.ndarray,
    args,
    *,
    last_good_center=None,
    last_good_size=None,
    cur_bbox=None,
    top_k: int = 1,
    frame_idx: int = -1,
) -> tuple[Optional[object], float]:
    """Run event-driven SGLATrack self-redetect, if the adapter exposes it.

    Returns the single best candidate dict (``top_k==1``, default) or a list of
    up to ``top_k`` spatially-distinct candidates (``top_k>1``, used by the FC
    challenge controller for cross-frame association)."""
    fn = getattr(tracker, "redetect", None)
    if not callable(fn):
        return None, 0.0

    from uav_tracker.types import BBox

    anchors: list[BBox] = []
    if last_good_center is not None and last_good_size is not None:
        _cx, _cy = last_good_center
        _w, _h = last_good_size
        anchors.append(BBox(x=float(_cx) - float(_w) / 2.0,
                            y=float(_cy) - float(_h) / 2.0,
                            w=float(_w), h=float(_h)))
    elif cur_bbox is not None:
        _x, _y, _w, _h = cur_bbox
        anchors.append(BBox(x=float(_x), y=float(_y), w=float(_w), h=float(_h)))

    t0 = time.perf_counter()
    try:
        result = fn(
            frame,
            factors=_parse_float_csv(args.sgla_redetect_factors),
            anchor_bboxes=anchors,
            include_current=bool(args.sgla_redetect_include_current),
            grid_size=int(args.sgla_redetect_grid),
            max_candidates=int(args.sgla_redetect_max_candidates),
            min_apce=float(args.sgla_redetect_min_apce),
            rank_by=getattr(args, "sgla_redetect_rank_by", "quality"),
            top_k=int(top_k),
            frame_idx=int(frame_idx),
        )
    except Exception:
        return None, (time.perf_counter() - t0) * 1000.0
    return result, (time.perf_counter() - t0) * 1000.0


def _best_relocate_candidate(candidates, min_ratio: float):
    """Pick the strongest SECONDARY score-map peak as a relocate target.

    candidates = SGLATrack score_map_stats["candidates"]: rank-0 is the chosen
    (top-1) peak, rank>=1 are spatially-distinct competing peaks (post-NMS). On a
    genuine loss the top-1 is usually background/distractor, so we bet on the
    strongest alternative peak whose score is a real fraction (>= min_ratio) of
    the top peak. Returns (cx, cy, w, h) in frame coords, or None.
    """
    if not candidates:
        return None
    for c in candidates:
        try:
            if int(c.get("rank", 0)) >= 1 and float(c.get("score_ratio", 0.0)) >= min_ratio:
                cx, cy = c["center"]
                bb = c["bbox"]
                return (float(cx), float(cy), float(bb[2]), float(bb[3]))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return None


def _combined_vote_count(tel: dict, args) -> int:
    """Count score-map/appearance signals voting 'true loss / true FC'.

    Shared by the LA combined gate and the FC precision gate. On UAV123 (123 seqs):
    within LA frames the 5-signal structural ensemble separates true-loss vs
    false-LA at AUROC ~0.95; within FC frames a vote>=2 fires on only ~0.4% of
    false-FC (tracker-fine) frames — a high-precision do-no-harm FC gate.
    """
    top2r   = tel.get("sm_local_top2_ratio")
    apce    = tel.get("apce")
    cosine  = tel.get("last_cosine_sim")
    entropy = tel.get("response_entropy")
    pmargin = tel.get("sm_local_peak_margin")
    votes = 0
    if top2r is not None and top2r >= args.gate_top2ratio:      votes += 1
    if apce  is not None and apce  <= args.gate_apce:           votes += 1
    if cosine is not None and cosine <= args.gate_cosine:       votes += 1
    if entropy is not None and entropy >= args.gate_entropy:    votes += 1
    if pmargin is not None and pmargin <= args.gate_peakmargin: votes += 1
    return votes


def _hard_la_gate(tel: dict, csc_pred, args) -> bool:
    """Genuine-loss (recoverable → act) vs FALSE-LA (target fine → hold) within a
    CSC-predicted LA frame, from runtime-only signals (no GT).

    Measured separability (tools/la_separability.py, UAV123): within the HARD
    tercile, true-loss vs false-LA separates at AUROC ~0.77-0.85 on score-map
    structure + appearance — NOT on raw confidence (degenerate) or displacement.
    Presets:
      combined   : K-of-N vote over the 5 strongest signals (robust; default)
      appearance : low search<->template cosine AND flat response (low APCE)
      response   : competing peaks (high top2_ratio, low peak_margin, high entropy)
      csc_head   : trust the model's own lost_aware_next_10 forecast head
    Returns True => treat as true-loss (apply recovery); False => hold.
    """
    top2r   = tel.get("sm_local_top2_ratio")
    apce    = tel.get("apce")
    cosine  = tel.get("last_cosine_sim")
    entropy = tel.get("response_entropy")
    pmargin = tel.get("sm_local_peak_margin")
    la10    = getattr(csc_pred, "lost_aware_next_10_prob", None)

    preset = args.gate_preset
    if preset == "csc_head":
        return la10 is not None and float(la10) >= args.gate_lostaware
    if preset == "cosine":
        # Strongest SINGLE discriminator measured on UAV123 LA frames
        # (true-loss vs false-LA AUROC 0.925, tools/la_gate_ensemble.py): a low
        # search<->template cosine means the tracker drifted onto wrong content
        # => recoverable true loss. Beats the diluted K-of-N combined vote (0.805),
        # whose apce/peak_margin/top2_ratio votes are near-random (~0.57) here.
        return cosine is not None and cosine <= args.gate_cosine
    if preset == "scoremap2":
        # 123-seq-validated gate (taxonomy sweep): a TRUE loss has a diffuse,
        # multi-peak score map. top2_ratio>=0.30 AND entropy>=4.0 suppresses the
        # false-LA guards (uav6 fires 2.8%, truck2/car13/boat8 0%, easy 0%) while
        # keeping ~64% true-loss recall (car12 83%, group3_2 88%). top2_ratio and
        # response_entropy are the two strongest cross-seq signals (AUROC 0.89/0.87).
        return ((top2r is not None and top2r >= args.gate_top2ratio)
                and (entropy is not None and entropy >= args.gate_entropy))
    if preset == "appearance":
        return ((cosine is not None and cosine <= args.gate_cosine)
                and (apce is not None and apce <= args.gate_apce))
    if preset == "response":
        return ((top2r is not None and top2r >= args.gate_top2ratio)
                and (entropy is not None and entropy >= args.gate_entropy)
                and (pmargin is not None and pmargin <= args.gate_peakmargin))
    # combined: K-of-N vote (each signal is one vote toward "true loss")
    return _combined_vote_count(tel, args) >= args.gate_vote_k


def _csc_model_name(checkpoint_path: str) -> str:
    """Derive a short run-tag from the checkpoint path stem."""
    return Path(checkpoint_path).stem


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run any tracker adapter with CSC.")
    p.add_argument(
        "--tracker",
        required=True,
        choices=_TRACKER_NAMES,
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=["lasot", "got10k", "uav123", "dtb70", "visdrone_sot", "uavdt_sot", "uavtrack112", "uav123_10fps"],
    )
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--csc_checkpoint", required=True, help="Path to CSC .pth checkpoint.")
    p.add_argument(
        "--csc_mode",
        choices=["passive", "control"],
        default="passive",
        help="passive = observe only; control = apply freeze hint.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--output_dir",
        default=None,
        help="Root output dir. Defaults to outputs/csc_runs.",
    )
    p.add_argument(
        "--run_tag",
        default=None,
        help="Override the auto-generated run tag (default: {tracker}_{dataset}_{split}_{ckpt_stem}).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Cap the number of sequences (smoke / debug).",
    )
    p.add_argument(
        "--include_sequences",
        nargs="*",
        default=None,
        help="If set, only run these sequence names. E.g. --include_sequences Yacht4 Animal1",
    )
    p.add_argument(
        "--skip_sequences",
        nargs="*",
        default=None,
        help="Skip these sequence names. E.g. --skip_sequences group3_4",
    )
    p.add_argument(
        "--weights_path",
        default=None,
        help="Override path to tracker weights checkpoint.",
    )
    p.add_argument(
        "--calibration_prefix",
        default=None,
        help=(
            "Explicit calibrator filename prefix in outputs/calibration/ "
            "(e.g. 'sglatrack_all_v2'). REQUIRED for V2 checkpoints: the CSC "
            "labels were built on percentile-calibrated APCE/PSR/confidence, so "
            "live inference must apply the SAME calibrator or raw telemetry "
            "(apce~111, psr~2869) collapses every prediction to CC. When unset, "
            "the prefix is inferred from the checkpoint path (often no match)."
        ),
    )
    p.add_argument(
        "--policy_freeze_fc_only",
        action="store_true",
        default=False,
        help="Fix: freeze template only on FC, NOT on LA. Prevents over-freeze on lost sequences.",
    )
    # ------------------------------------------------------------------ #
    # TODO (control levers NOT yet effective for SGLATrack — 2026-06-01): #
    #   * template-FREEZE is largely INERT: SGLATrack.try_update_template  #
    #     fires at most max_updates=5 times/sequence, >=100 frames apart,  #
    #     behind strict guards (apce>220, psr>2000, cosine>0.80). Blocking #
    #     <=5 rare updates barely changes the trajectory, so freeze on/off #
    #     has negligible effect on AUC/FCR. set_update_enabled() IS wired  #
    #     (capabilities.can_freeze_template=True) but the underlying update#
    #     is too rare to matter. Needs a real online-template path to be   #
    #     a meaningful lever.                                              #
    #   * WIDER-SEARCH is NOT wired: capabilities.can_widen_search=False   #
    #     (sglatrack.py: "_SEARCH_FACTOR" is a module const, needs to be   #
    #     an instance var with a setter).                                  #
    #   * RE-DETECTION / reinit is never invoked in the control loop.      #
    #   => The ONLY effective SGLATrack control lever today is the         #
    #      exit-router (force block 9 on risky states). Until the above    #
    #      are wired, do-no-harm holding (see --policy_hold_on_la) is the  #
    #      safest action on LOST frames.                                   #
    # ------------------------------------------------------------------ #
    p.add_argument(
        "--policy_hold_on_la",
        action="store_true",
        default=False,
        help=(
            "On LA (LOST_AWARE): DO NOTHING / hold. Suppress the exit-router "
            "block-9 override (the only live lever) and keep the template "
            "frozen. Rationale: the proper LOST actions (wider search / "
            "re-detection) are NOT wired yet, and block-9 thrashing on a "
            "lost frame creates a drift feedback loop (false LA -> block-9 -> "
            "worse -> more LA). Holding steady is least-harm. FC keeps its "
            "block-9 + freeze. Requires --csc_mode control."
        ),
    )
    p.add_argument(
        "--policy_lost_widen_search",
        action="store_true",
        default=False,
        help=(
            "REAL LOST action (the proper recovery lever). On LA (LOST_AWARE) "
            "progressively WIDEN the search region via SGLATrack.set_search_factor "
            "INSTEAD of forcing exit block-9. Factor grows base*(1+step*consec_LA) "
            "up to base*max, resets to base on recovery (CC), holds on CU/FC. "
            "Block-9 is reserved for FC (confidently-wrong → deeper compute helps). "
            "Rationale (measured): block-9 on a FALSE-LA frame thrashes the tracker "
            "into a drift loop (uav6 LA 15%%->40%%); a wider crop on a false-LA is "
            "near-harmless (target still inside the larger crop) yet genuinely "
            "recovers a truly-lost target. This is what lets EASY scenes stay ~0 "
            "ΔAUC while HARD scenes gain. Requires --csc_mode control + --exit_router."
        ),
    )
    p.add_argument(
        "--lost_widen_max",
        type=float,
        default=1.5,
        help="Max search-factor multiplier when fully lost (default 1.5 → base 4.0 → 6.0). "
             "Kept gentle: widening trades target resolution for area, so a large factor "
             "is destructive on the (common) false-LA frames.",
    )
    p.add_argument(
        "--lost_widen_step",
        type=float,
        default=0.15,
        help="Per-armed-LA-frame widen increment (default 0.15).",
    )
    p.add_argument(
        "--lost_arm_frames",
        type=int,
        default=5,
        help=(
            "Require N CONSECUTIVE LA frames before the wider-search arms (default 5). "
            "An isolated / short false-LA burst (typical on easy scenes) never reaches "
            "N, so the search stays at base — 'do nothing on easy'. Only a sustained "
            "loss arms the widen. NOTE (measured on UAV123): the CSC over-predicts LA on "
            "ambiguous-but-OK sequences (e.g. uav6: 98%% LA, 72-frame run, yet GT-fine), "
            "so even armed widen damages them — no causal runtime signal (confidence is "
            "degenerate ~0.017; displacement does not separate) distinguishes false-LA "
            "from true loss. Widen is therefore a measured-negative ablation; the safe "
            "LA policy is --policy_hold_on_la."
        ),
    )
    p.add_argument(
        "--policy_gated_redetect",
        action="store_true",
        default=False,
        help=(
            "GATED LA recovery — the fix for the false-LA wall. On LA, run a "
            "runtime HARD-true-LA gate (--gate_preset); ONLY on a genuine loss "
            "apply the recovery action (--redetect_action: widen / relocate / "
            "both), and only after --redetect_arm_frames of sustained gated-LA. A "
            "false-LA frame (gate shut) HOLDS — no widen, no relocate, no block-9 "
            "thrash (identical to --policy_hold_on_la there). This is what ungated "
            "widen lacked: the gate separates true-loss from false-LA at AUROC "
            "~0.85 (score-map structure + appearance, tools/la_separability.py), so "
            "the action fires only where it helps. Requires --csc_mode control. "
            "Mutually exclusive with --policy_hold_on_la / --policy_lost_widen_search."
        ),
    )
    p.add_argument("--gate_preset", default="combined",
                   choices=["combined", "appearance", "response", "csc_head", "cosine", "scoremap2"],
                   help="HARD-true-LA gate rule (see _hard_la_gate). Default 'combined' (K-of-N vote).")
    p.add_argument("--gate_vote_k", type=int, default=3,
                   help="combined gate: min signals (of 5) voting true-loss to fire (default 3).")
    p.add_argument("--gate_top2ratio", type=float, default=0.30,
                   help="Gate: sm_local_top2_ratio >= this => competing-peak vote (HARD median true 0.46 / false 0.13).")
    p.add_argument("--gate_apce", type=float, default=110.0,
                   help="Gate: apce <= this => flat-response vote (HARD median true 56 / false 159).")
    p.add_argument("--gate_cosine", type=float, default=0.85,
                   help="Gate: last_cosine_sim <= this => appearance-mismatch vote (HARD median true 0.81 / false 0.88).")
    p.add_argument("--gate_entropy", type=float, default=4.0,
                   help="Gate: response_entropy >= this => flat-map vote (HARD median true 4.32 / false 3.73).")
    p.add_argument("--gate_peakmargin", type=float, default=0.35,
                   help="Gate: sm_local_peak_margin <= this => no-dominant-peak vote (HARD median true 0.21 / false 0.53).")
    p.add_argument("--gate_lostaware", type=float, default=0.99,
                   help="csc_head gate: lost_aware_next_10_prob >= this => true-loss (HARD median true 0.994 / false 0.979).")
    # ----- forecast-driven control (use the V3 forecast heads in the LA trigger) -----
    p.add_argument("--gate_forecast_thresh", type=float, default=0.0,
                   help="Forecast-CONFIRMED firing (precision): if >0, the reactive LA gate ALSO "
                        "requires lost_aware_next_10_prob >= this (AND) before motion_bridge fires — "
                        "suppresses false-fires on borderline frames (forecast must agree). 0=off.")
    p.add_argument("--proactive_bridge", action="store_true", default=False,
                   help="Proactive (LEAD-time) LA recovery: pre-arm the motion_bridge when the forecast "
                        "heads warn (lost_aware_next_10 & failure_next_10 high) BEFORE the loss fully "
                        "establishes (state not yet LA), so consec_gate_la is already armed when the "
                        "real loss hits. Uses --proactive_la_thresh / --proactive_fail_thresh.")
    p.add_argument("--proactive_la_thresh", type=float, default=0.90,
                   help="proactive_bridge: min lost_aware_next_10_prob to pre-arm (default 0.90).")
    p.add_argument("--proactive_fail_thresh", type=float, default=0.90,
                   help="proactive_bridge: min failure_next_10_prob to pre-arm (default 0.90).")
    p.add_argument("--redetect_action", default="widen_relocate",
                   choices=[
                       "widen",
                       "relocate",
                       "widen_relocate",
                       "motion_bridge",
                       "bridge_relocate",
                       "sgla_redetect",
                       "bridge_sgla",
                   ],
                   help="Recovery action when the gate fires on LA: widen search / relocate to "
                        "secondary peak / both / motion_bridge (extrapolate last-good motion) / "
                        "sgla_redetect (extra SGLATrack wide-crop self re-detection).")
    p.add_argument("--redetect_arm_frames", type=int, default=3,
                   help="Require N consecutive GATED-LA frames before acting (default 3; avoids single-frame twitch).")
    p.add_argument("--relocate_min_ratio", type=float, default=0.30,
                   help="relocate: min score_ratio of a secondary peak to jump to it (default 0.30).")
    p.add_argument("--sgla_redetect_factors", default="8,12,16",
                   help="sgla_redetect: comma-separated wide-crop factors to try with the frozen SGLATrack template.")
    p.add_argument("--sgla_redetect_include_current", action="store_true", default=False,
                   help="sgla_redetect: also search around the current tracker state in addition to last-good/current hint.")
    p.add_argument("--sgla_redetect_grid", type=int, default=0,
                   help="sgla_redetect: optional NxN sparse grid anchors over the frame (0=off).")
    p.add_argument("--sgla_redetect_max_candidates", type=int, default=3,
                   help="sgla_redetect: max candidate peaks decoded per wide crop (default 3).")
    p.add_argument("--sgla_redetect_min_apce", type=float, default=0.0,
                   help="sgla_redetect: reject wide-crop candidates below this APCE (0=off).")
    p.add_argument("--sgla_redetect_rank_by", default="quality", choices=["quality", "identity"],
                   help="sgla_redetect: pick the best candidate by 'quality' (APCE*score_ratio — sharpest "
                        "peak, for LA recovery) or 'identity' (sim_to_init — most on-template, for the FC "
                        "challenge-and-switch controller, which must find the RIGHT object not a sharp distractor).")
    p.add_argument("--sgla_redetect_min_sim", type=float, default=0.0,
                   help="sgla_redetect VERIFIER: only APPLY a redetect candidate whose peak-local "
                        "embedding cosine to the FROZEN initial template (sim_to_init) >= this. 0=off. "
                        "This is the V4 verifier identity floor — rejects distractor jumps (bird1_1/car11) "
                        "where the wide-crop score map is sharp but the content is the wrong object.")
    p.add_argument("--bridge_max_frames", type=int, default=30,
                   help="motion_bridge: max frames to extrapolate the last-good motion before giving up (hold).")
    p.add_argument("--bridge_vel_ema", type=float, default=0.5,
                   help="motion_bridge: EMA factor for the velocity estimate from CONFIRMED frames (0=instant, 0.9=slow).")
    p.add_argument("--bridge_max_resid_ratio", type=float, default=1e9,
                   help="motion_bridge: only extrapolate when the pre-loss velocity is REGULAR — "
                        "EMA velocity-residual / target-scale <= this. Erratic targets (e.g. a darting "
                        "bird) have high residual → bridge is suppressed (hold) instead of chasing a "
                        "wrong constant-velocity guess. Default 1e9 = off (always bridge).")
    p.add_argument("--bridge_max_disp", type=float, default=0.0,
                   help="motion_bridge SAFETY CAP: bound the total extrapolated displacement to this "
                        "multiple of target scale from the last-good position. Stops a wrong velocity on a "
                        "gate-misfired EASY track from sending the search runaway (UAVTrack112 car6_2 "
                        "0.84->0.26). 0 = off. ~6-10 bounds damage while still allowing genuine recovery.")
    # ----- Alternative FALSE-CONFIRMED (FC) control actions -----------------
    # The production FC action is block-9 (deeper compute via the exit-router),
    # which costs AUC (the FC-control scaffold = -0.03 HARD on smoke) for little
    # gain (FC AUC ceiling ±0.004). These let us explore better FC actions and
    # measure each against passive on AUC + runtime-FCR/FCD. Router-independent.
    p.add_argument(
        "--policy_fc_control", action="store_true", default=False,
        help=("Apply an FC (false-confirmed) control action directly on FC frames "
              "(router-independent). Choose the action with --fc_action. Requires "
              "--csc_mode control. Mutually exclusive with --policy_gated_redetect."))
    p.add_argument("--fc_action", default="freeze_only",
                   choices=["block9", "freeze_only", "hold_lastgood", "widen", "relocate"],
                   help="FC action: block9 (current production, deeper compute) / freeze_only (just "
                        "freeze template, the do-no-harm baseline) / hold_lastgood (snap search back to "
                        "last CONFIRMED position — reject the confident-wrong box) / widen / relocate.")
    p.add_argument("--fc_streak_frames", type=int, default=2,
                   help="Require N consecutive FC frames before the FC action fires (default 2; FC is "
                        "~2x over-flagged so a 1-frame streak guard cuts false alarms).")
    p.add_argument("--fc_gate_vote_k", type=int, default=0,
                   help="High-precision true-FC gate: require >=K score-map votes (same signals as the "
                        "LA combined gate) before the FC action fires. 0=off (legacy: act on every FC "
                        "frame). On UAV123 ~86%% of FC predictions are false-FC (tracker fine); K=2 fires "
                        "on only ~0.4%% of those, preventing the hold_lastgood easy-scene regression.")
    # --- FC challenge-and-switch (MVP): FC triggers a READ-ONLY redetect that
    # only switches after temporal verification, with an abort window + rollback.
    # Unlike --policy_fc_control (immediate lever on FC), this NEVER moves the
    # incumbent until a candidate is stably better, so a false-FC costs only a
    # few extra forwards instead of a catastrophic switch. Mutually exclusive
    # with --policy_fc_control; composes with --policy_gated_redetect (LA).
    p.add_argument("--policy_fc_challenge", action="store_true", default=False,
                   help="FC challenge-and-switch controller: on an FC streak, freeze the template and run "
                        "a READ-ONLY redetect each frame; SWITCH onto a candidate only if it reappears near "
                        "the same place AND is stably better than the incumbent for --fc_challenge_confirm "
                        "frames, then hold an abort window with rollback. Requires --csc_mode control. "
                        "Mutually exclusive with --policy_fc_control.")
    p.add_argument("--fc_challenge_streak", type=int, default=3,
                   help="FC challenge: consecutive FC frames (high FC risk) before a challenge starts (default 3).")
    p.add_argument("--fc_challenge_confirm", type=int, default=3,
                   help="FC challenge: consecutive 'candidate stably better + reappears nearby' frames before a switch.")
    p.add_argument("--fc_challenge_max_frames", type=int, default=10,
                   help="FC challenge: abandon the challenge (declare false FC) after this many frames with no switch.")
    p.add_argument("--fc_challenge_abort_window", type=int, default=5,
                   help="FC challenge: after a switch, monitor the new track this many frames; rollback if it degrades.")
    p.add_argument("--fc_challenge_sim_margin", type=float, default=0.05,
                   help="FC challenge: RELATIVE identity margin — candidate sim_to_init must beat incumbent "
                        "initial_template_sim by at least this much to count as better (also the abort-window "
                        "identity-collapse threshold).")
    p.add_argument("--fc_challenge_apce_keep", type=float, default=0.6,
                   help="FC challenge: candidate APCE must be >= this fraction of incumbent APCE (response-competitive).")
    p.add_argument("--fc_challenge_reappear_radius", type=float, default=1.0,
                   help="FC challenge: candidate reappears 'near the same place' if within this * sqrt(w*h) of the running anchor.")
    p.add_argument("--fc_challenge_switch_mode", default="identity",
                   choices=["identity", "displacement", "association"],
                   help="FC challenge switch test: 'identity' (candidate must beat incumbent sim_to_init by "
                        "--fc_challenge_sim_margin — safe default, never fires on saturated-identity FC); "
                        "'displacement' (candidate genuinely relocated off the wrong incumbent + reappears); "
                        "'association' (track the redetect candidate nearest the last-good trajectory across "
                        "frames — the only signal that picks the real object over the confident distractor in "
                        "a true-FC; uses --fc_challenge_topk candidates + --fc_challenge_assoc_gate).")
    p.add_argument("--fc_challenge_min_switch_disp", type=float, default=0.5,
                   help="FC challenge displacement mode: candidate must be >= this * sqrt(w*h) from the current incumbent center.")
    p.add_argument("--fc_challenge_topk", type=int, default=6,
                   help="FC challenge association mode: number of spatially-distinct redetect candidates to track/associate.")
    p.add_argument("--fc_challenge_assoc_gate", type=float, default=2.0,
                   help="FC challenge association mode: a candidate associates to the tracked anchor if within this * sqrt(w*h).")
    # ---- policy_fc_recover (Phase 2 detect-verify-recover; csc_lib/csc/recover) ----
    # Same FC streak trigger as policy_fc_challenge but runs the V4-borrow stack:
    # PrototypeMemory (anchor + recent EMA + distractor) -> CandidateVerifier
    # (identity + distractor veto + peak margin) -> SPRT (adaptive temporal
    # confirmation). Optional event-triggered RT-DETRv2-S detector merges its
    # boxes with SGLATrack's internal multi-crop pyramid (the SGLA-only
    # candidate set has 0% oracle recall@IoU.5 on UAV123 hard FC frames).
    # Mutually exclusive with --policy_fc_control / --policy_fc_challenge.
    p.add_argument("--policy_fc_recover", action="store_true", default=False,
                   help="FC detect-verify-recover (csc_lib/csc/recover): SGLA + RT-DETR candidates -> "
                        "PrototypeMemory verifier (distractor veto, identity floor) -> SPRT switch + "
                        "abort window with rollback. Replaces --policy_fc_challenge with adaptive "
                        "temporal verification and an event-triggered external detector.")
    p.add_argument("--fc_recover_streak", type=int, default=2,
                   help="Consecutive FC frames before entering CHALLENGE (recover).")
    p.add_argument("--fc_recover_top_k", type=int, default=5,
                   help="Top-K spatially-distinct candidates from SGLATrack's internal redetect.")
    p.add_argument("--fc_recover_use_detector", action="store_true", default=False,
                   help="Add RT-DETRv2-S boxes to the candidate pool during a CHALLENGE (event-triggered).")
    p.add_argument("--fc_recover_detector_top_k", type=int, default=5,
                   help="Cap on RT-DETR boxes per CHALLENGE step (sorted by proximity to last-CC).")
    p.add_argument("--fc_recover_detector_conf", type=float, default=0.30,
                   help="Detector min confidence threshold.")
    p.add_argument("--fc_recover_sprt_alpha", type=float, default=0.05,
                   help="SPRT target false-switch probability (smaller -> harder to fire).")
    p.add_argument("--fc_recover_sprt_beta", type=float, default=0.10,
                   help="SPRT target miss probability.")
    p.add_argument("--fc_recover_sprt_budget", type=int, default=3,
                   help="SPRT max successful switches per sequence.")
    p.add_argument("--fc_recover_accept_margin", type=float, default=0.55,
                   help="CandidateVerifier accept margin (>= 0.55 = strict, lower = more lenient).")
    p.add_argument("--fc_recover_distractor_veto", type=float, default=0.85,
                   help="CandidateVerifier hard distractor cosine veto.")
    p.add_argument("--fc_recover_min_identity", type=float, default=0.35,
                   help="CandidateVerifier identity floor (sim_to_init or sim_to_recent).")
    p.add_argument("--fc_recover_challenge_max", type=int, default=15,
                   help="Hard cap on CHALLENGE phase frames (bounds redetect cost).")
    p.add_argument("--fc_recover_abort_window", type=int, default=5,
                   help="Frames to monitor after a switch before COMMIT.")
    p.add_argument("--fc_recover_rollback_streak", type=int, default=2,
                   help="Consecutive LA/FC frames inside abort window before rollback. "
                        "1 = legacy (rollback on first regression — caused 100%% rollback "
                        "in early smoke runs); 2+ tolerates settling frames after a switch.")
    p.add_argument("--fc_recover_motion_max_disp", type=float, default=0.0,
                   help="Verifier motion gate: hard reject candidates further than this "
                        "(pixels) from last-CC center. 0 = disabled (no spatial gate); "
                        "200-400 protects against catastrophic wrong-object switches.")
    p.add_argument("--fc_recover_motion_max_disp_per_frame", type=float, default=50.0,
                   help="Per-frame velocity cap added to the static motion gate. Total "
                        "allowed displacement = max(motion_max_disp, this * frames_since_cc).")
    p.add_argument("--gated_freeze", action="store_true", default=False,
                   help="With --policy_gated_redetect / --policy_fc_control: freeze the template ONLY on a "
                        "gated action frame (true-loss / true-FC), not on every raw CSC LA/FC hint. The raw "
                        "hint over-fires on false-LA/false-FC (easy-scene majority); freezing there blocks a "
                        "good update on a healthy track (the car6_5 -0.11 control-mode regression).")
    p.add_argument("--no_runner_template_update", action="store_true", default=False,
                   help="Control mode: do NOT have the runner explicitly call try_update_template on "
                        "non-frozen frames. Makes control template behaviour identical to PASSIVE (the "
                        "eval5_clamp baseline keeps the frame-0 template), so ΔAUC isolates the LA/FC "
                        "recovery levers and healthy easy scenes stay at ΔAUC~0 (fixes the car6_5 -0.11 "
                        "confound from runner-driven template drift).")
    p.add_argument("--recovery_update_window", type=int, default=-1,
                   help="Control mode: allow the runner template refresh ONLY for N frames after a gated "
                        "loss/FC action — the re-acquisition window that LOCKS IN motion_bridge recovery "
                        "(car9/car1_s need this; bridge re-centres, the refresh re-acquires). -1 = legacy "
                        "(refresh on every unfrozen frame, which drifts always-healthy tracks like car6_5). "
                        "With N>=0 an always-healthy track never refreshes == passive => ΔAUC~0 on easy.")
    p.add_argument(
        "--risk_gate_open_streak",
        type=int,
        default=1,
        help=(
            "Risk-gate: require N risky-state (LA/FC) frames in the trailing window "
            "before the gate opens on the reactive-state axis. Default 1 (any single "
            "frame, == prior behaviour). Set 2 to ignore isolated single-frame "
            "false-LA on easy scenes (keeps them at 0 intervention). The forecast / "
            "risk_score axes are unaffected (they remain sensitive early-warnings)."
        ),
    )
    p.add_argument(
        "--policy_tau_fc",
        type=float,
        default=0.0,
        help="Fix: min P(FC) to call FC (0=off). E.g. 0.75 reduces false-alarm exits.",
    )
    p.add_argument(
        "--policy_fc_streak",
        type=int,
        default=1,
        help="Fix: require N consecutive FC frames before freeze fires (1=immediate, 3=streak).",
    )
    p.add_argument(
        "--exit_router",
        action="store_true",
        default=False,
        help=(
            "SGLATrack only. Activate StateExitRouter: CSC-predicted state "
            "overrides SGLATrack's (collapsed) MLP exit-router per frame. "
            "No effect on other trackers. Requires --csc_mode control or "
            "passive (routing happens regardless of template-freeze control). "
            "Default OFF — preserves baseline behaviour."
        ),
    )
    p.add_argument(
        "--exit_router_hold",
        type=int,
        default=5,
        help="StateExitRouter min_hold_frames hysteresis (default 5).",
    )
    p.add_argument(
        "--exit_router_min_conf",
        type=float,
        default=0.0,
        help=(
            "StateExitRouter min CSC state confidence to trust a risky-state "
            "prediction. 0.0 = trust all (default). 0.6 = downgrade low-"
            "confidence risky states to uncertain."
        ),
    )
    p.add_argument(
        "--csc_advisor",
        action="store_true",
        default=False,
        help=(
            "Activate CSCAdvisor (Variant C): stateful template-update gating "
            "via CSC state + hysteresis. Works for any tracker. Requires "
            "--csc_mode control to take effect (in passive mode the decision "
            "is logged but the tracker is not affected). Default OFF."
        ),
    )
    p.add_argument(
        "--advisor_streak",
        type=int,
        default=5,
        help="CSCAdvisor: consecutive safe frames required before unblocking (Gate 3). Default 5.",
    )
    p.add_argument(
        "--advisor_cooldown",
        type=int,
        default=15,
        help="CSCAdvisor: min frames between template updates (Gate 5). Default 15.",
    )
    p.add_argument(
        "--advisor_max_hold",
        type=int,
        default=50,
        help="CSCAdvisor: max consecutive blocked frames before soft release. Default 50.",
    )
    p.add_argument(
        "--proactive_v3",
        action="store_true",
        default=False,
        help=(
            "Proactive control using Stage-2 V3 forecast heads: triggers exit routing "
            "when fc_n10_prob > --proactive_threshold, BEFORE FC establishes. "
            "Requires --exit_router and a checkpoint with forecast heads enabled."
        ),
    )
    p.add_argument(
        "--proactive_threshold",
        type=float,
        default=0.7,
        help="fc_n10_prob threshold for proactive exit routing. Default 0.7.",
    )
    p.add_argument(
        "--control_risk_gate",
        action="store_true",
        default=False,
        help=(
            "Risk-gate (causal): when ON, bypass ALL control on frames with no "
            "recent evidence of risk — behave exactly as passive. 'No recent "
            "risk' over the trailing --risk_gate_window = no LA/FC state AND "
            "forecast fc_n10/failure_n10 below --risk_gate_fc_thresh AND "
            "risk_score below --risk_gate_risk_thresh. Removes the easy-scene "
            "template-update drift cost without touching hard-scene gains "
            "(persistent risk keeps the gate open). Requires --csc_mode control. "
            "Default OFF (so ablation can compare gated vs ungated control)."
        ),
    )
    p.add_argument(
        "--risk_gate_window",
        type=int,
        default=10,
        help="Risk-gate: trailing window (frames) scanned for recent risk. Default 10.",
    )
    p.add_argument(
        "--risk_gate_fc_thresh",
        type=float,
        default=0.3,
        help=(
            "Risk-gate: forecast prob (fc_n10 / failure_n10) above which the "
            "window counts as risky and the gate opens. Conservative (low) by "
            "design — any forecast hint keeps control active. Default 0.3."
        ),
    )
    p.add_argument(
        "--risk_gate_risk_thresh",
        type=float,
        default=0.5,
        help="Risk-gate: risk_score above which the window counts as risky. Default 0.5.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    import torch
    from csc_lib.csc.inference import load_runtime
    from csc_lib.csc.fc_challenge import FCChallengeConfig, FCChallengeController

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger("run_with_csc")
    args = parse_args(argv)
    _set_seed(args.seed)

    csc_model_tag = _csc_model_name(args.csc_checkpoint)
    run_tag = args.run_tag if args.run_tag else f"{args.tracker}_{args.dataset}_{args.split}_{csc_model_tag}"

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "outputs" / "csc_runs")

    out_root = Path(output_dir) / run_tag
    pred_dir = out_root / "predictions"
    tel_dir = out_root / "telemetry"
    states_dir = out_root / "states"
    for d in (pred_dir, tel_dir, states_dir):
        d.mkdir(parents=True, exist_ok=True)

    log.info("loading dataset %s/%s", args.dataset, args.split)
    dataset = _load_dataset(args.dataset, args.split)

    log.info("loading tracker %s on %s", args.tracker, args.device)
    tracker = _load_tracker(args.tracker, args.weights_path, args.device)
    # Resolve tracker capabilities; fall back to safe defaults if adapter predates this API.
    from uav_tracker.trackers.capabilities import TrackerCapabilities
    caps: TrackerCapabilities = getattr(tracker, "capabilities", TrackerCapabilities())
    log.info(
        "tracker capabilities: freeze_template=%s  widen_search=%s  "
        "force_reinit=%s  reject_bbox=%s  reduce_pruning=%s",
        caps.can_freeze_template, caps.can_widen_search,
        caps.can_force_reinit, caps.can_reject_bbox, caps.can_reduce_pruning,
    )
    # Baseline search-region factor (SGLATrack default 4.0). The wider-search
    # control lever raises the tracker's instance _search_factor above this on
    # LOST and restores it on recovery; capturing the base keeps the policy
    # tracker-agnostic (other adapters fall back to 4.0 and ignore the hook).
    base_search_factor = float(getattr(tracker, "_search_factor", 4.0))
    if args.policy_lost_widen_search:
        if not caps.can_widen_search:
            log.warning(
                "--policy_lost_widen_search set but tracker %s reports "
                "can_widen_search=False; widen will be a no-op.",
                args.tracker,
            )
        log.info(
            "policy_lost_widen_search ON: base_search_factor=%.2f max=%.2f step=%.2f",
            base_search_factor, args.lost_widen_max, args.lost_widen_step,
        )
    if args.policy_gated_redetect:
        if args.csc_mode != "control":
            raise SystemExit("--policy_gated_redetect requires --csc_mode control")
        if args.policy_hold_on_la or args.policy_lost_widen_search:
            raise SystemExit(
                "--policy_gated_redetect is mutually exclusive with "
                "--policy_hold_on_la / --policy_lost_widen_search"
            )
        if not caps.can_widen_search and args.redetect_action in ("widen", "widen_relocate"):
            log.warning("--redetect_action=%s but tracker %s can_widen_search=False "
                        "→ widen is a no-op", args.redetect_action, args.tracker)
        if args.redetect_action in ("sgla_redetect", "bridge_sgla"):
            _parse_float_csv(args.sgla_redetect_factors)
            if not callable(getattr(tracker, "redetect", None)):
                log.warning("--redetect_action=%s but tracker %s has no redetect() hook "
                            "→ SGLA self-redetect is a no-op", args.redetect_action, args.tracker)
        log.info(
            "policy_gated_redetect ON: gate=%s action=%s arm=%d "
            "(base_search_factor=%.2f max=%.2f step=%.2f)",
            args.gate_preset, args.redetect_action, args.redetect_arm_frames,
            base_search_factor, args.lost_widen_max, args.lost_widen_step,
        )
    if args.policy_fc_control:
        if args.csc_mode != "control":
            raise SystemExit("--policy_fc_control requires --csc_mode control")
        if args.policy_gated_redetect and args.redetect_action in ("widen", "widen_relocate") \
                and args.fc_action in ("widen",):
            log.warning("both LA and FC actions use the WIDEN search-factor lever — they may "
                        "interact; prefer motion_bridge (LA) + hold_lastgood (FC), which use the "
                        "one-shot search-center lever on disjoint states (LA=2 / FC=3).")
        log.info("policy_fc_control ON: fc_action=%s streak=%d%s",
                 args.fc_action, args.fc_streak_frames,
                 " (combined with policy_gated_redetect: LA=2 + FC=3, disjoint states)"
                 if args.policy_gated_redetect else "")

    if args.policy_fc_challenge:
        if args.csc_mode != "control":
            raise SystemExit("--policy_fc_challenge requires --csc_mode control")
        if args.policy_fc_control:
            raise SystemExit(
                "--policy_fc_challenge is mutually exclusive with --policy_fc_control "
                "(both act on FC; pick the challenge-and-switch controller OR the direct action)."
            )
        if not callable(getattr(tracker, "redetect", None)):
            log.warning("--policy_fc_challenge but tracker %s has no redetect() hook "
                        "→ challenge can never find a candidate (freeze-only behaviour).", args.tracker)
        log.info(
            "policy_fc_challenge ON: streak=%d confirm=%d max=%d abort=%d "
            "sim_margin=%.3f apce_keep=%.2f reappear_r=%.2f%s",
            args.fc_challenge_streak, args.fc_challenge_confirm, args.fc_challenge_max_frames,
            args.fc_challenge_abort_window, args.fc_challenge_sim_margin,
            args.fc_challenge_apce_keep, args.fc_challenge_reappear_radius,
            " (combined with policy_gated_redetect: LA=2 + FC=3, disjoint states)"
            if args.policy_gated_redetect else "",
        )

    if args.policy_fc_recover:
        if args.csc_mode != "control":
            raise SystemExit("--policy_fc_recover requires --csc_mode control")
        if args.policy_fc_control or args.policy_fc_challenge:
            raise SystemExit(
                "--policy_fc_recover is mutually exclusive with "
                "--policy_fc_control / --policy_fc_challenge "
                "(all three act on FC; pick the recover controller for the V4-borrow stack)."
            )
        if not callable(getattr(tracker, "redetect", None)):
            log.warning("--policy_fc_recover but tracker %s has no redetect() hook "
                        "→ no SGLA-internal candidate source; relying on RT-DETR if enabled.",
                        args.tracker)
        log.info(
            "policy_fc_recover ON: streak=%d topK=%d sprt(α=%.2f β=%.2f budget=%d) "
            "verifier(accept=%.2f distr_veto=%.2f min_id=%.2f) detector=%s",
            args.fc_recover_streak, args.fc_recover_top_k,
            args.fc_recover_sprt_alpha, args.fc_recover_sprt_beta,
            args.fc_recover_sprt_budget, args.fc_recover_accept_margin,
            args.fc_recover_distractor_veto, args.fc_recover_min_identity,
            "rtdetrv2_s" if args.fc_recover_use_detector else "off",
        )

    log.info("loading CSC from %s", args.csc_checkpoint)
    # Auto-detect calibration directory: prefer the project-level
    # outputs/calibration/ directory, which is the standard location for
    # per-tracker calibrator JSON files.  Fall back to the checkpoint's parent
    # directory as a convenience for self-contained run bundles.
    # Calibrators are loaded lazily — missing files are silently skipped.
    _ckpt_path = Path(args.csc_checkpoint)
    _default_cal_dir = PROJECT_ROOT / "outputs" / "calibration"
    _sibling_cal_dir = _ckpt_path.parent
    _cal_dir: Optional[Path] = None
    if _default_cal_dir.is_dir():
        _cal_dir = _default_cal_dir
    elif _sibling_cal_dir.is_dir():
        _cal_dir = _sibling_cal_dir
    csc = load_runtime(
        _ckpt_path,
        device=args.device,
        calibration_dir=_cal_dir,
        tracker_name=args.calibration_prefix,
    )

    # Apply policy fixes if specified
    if args.policy_freeze_fc_only:
        csc.policy.freeze_on_la = False
        log.info("Policy fix: freeze_on_la=False (FC only, not LA)")
    if args.policy_tau_fc > 0:
        csc.policy.tau_fc = args.policy_tau_fc
        log.info("Policy fix: tau_fc=%.2f (confidence gate)", args.policy_tau_fc)
    if args.policy_fc_streak > 1:
        csc.policy.fc_streak_required = args.policy_fc_streak
        log.info("Policy fix: fc_streak_required=%d", args.policy_fc_streak)

    # --- StateExitRouter (SGLATrack only, default OFF) ---
    exit_router = None
    if args.exit_router:
        if args.tracker != "sglatrack":
            log.warning(
                "--exit_router is only supported for sglatrack; "
                "ignoring (tracker=%s)",
                args.tracker,
            )
        else:
            from uav_tracker.trackers.exit_router import StateExitRouter
            exit_router = StateExitRouter(
                min_hold_frames=args.exit_router_hold,
                min_state_confidence=args.exit_router_min_conf,
            )
            log.info(
                "StateExitRouter enabled (min_hold=%d, min_conf=%.2f)",
                args.exit_router_hold,
                args.exit_router_min_conf,
            )

    # --- CSCAdvisor (Variant C, default OFF) ---
    csc_advisor = None
    if args.csc_advisor:
        from uav_tracker.trackers.csc_advisor import CSCAdvisor
        csc_advisor = CSCAdvisor(
            streak_required=args.advisor_streak,
            cooldown_frames=args.advisor_cooldown,
            max_hold_frames=args.advisor_max_hold,
        )
        log.info(
            "CSCAdvisor enabled (streak=%d, cooldown=%d, max_hold=%d, mode=%s)",
            args.advisor_streak,
            args.advisor_cooldown,
            args.advisor_max_hold,
            args.csc_mode,
        )

    sequences = list(dataset)
    if args.max_sequences:
        sequences = sequences[: args.max_sequences]
    if args.include_sequences is not None:
        include_set = set(args.include_sequences)
        sequences = [s for s in sequences if s.name in include_set]
        log.info("--include_sequences: kept %d sequences", len(sequences))
    if args.skip_sequences is not None:
        skip_set = set(args.skip_sequences)
        sequences = [s for s in sequences if s.name not in skip_set]
        log.info("--skip_sequences: skipping %s, kept %d", args.skip_sequences, len(sequences))

    # Resume: skip sequences whose state file already exists and is non-empty
    done_seqs = {p.stem for p in states_dir.glob("*.jsonl") if p.stat().st_size > 0}
    if done_seqs:
        sequences = [s for s in sequences if s.name not in done_seqs]
        log.info("resume: %d sequences already done, %d remaining", len(done_seqs), len(sequences))

    n_seq = len(sequences)
    total_frames = 0

    # Per-stage latency accumulators (ms, across all frames of all sequences)
    tracker_latencies_ms: list[float] = []
    csc_latencies_ms: list[float] = []
    total_latencies_ms: list[float] = []

    seq_records: list[dict] = []
    update_freezes = 0
    search_widen_frames = 0  # frames where --policy_lost_widen_search applied factor > base
    gated_la_frames = 0       # frames where the HARD-true-LA gate FIRED on an LA frame
    gated_relocate_frames = 0 # frames where a relocate (override_search_center) was applied
    sgla_redetect_calls = 0   # extra SGLATrack self-redetect invocations
    sgla_redetect_hits = 0    # calls that returned an accepted candidate
    sgla_redetect_ms_total = 0.0
    fc_fired_frames = 0       # frames where --policy_fc_control fired an FC action
    # --- FC challenge-and-switch aggregate counters (--policy_fc_challenge) ---
    fc_challenge_starts = 0
    fc_challenge_switches = 0
    fc_challenge_commits = 0
    fc_challenge_rollbacks = 0
    fc_challenge_aborts = 0
    fc_challenge_redetect_calls = 0
    fc_challenge_frames = 0   # frames where a challenge/abort window was active
    exit_router_switches_total = 0
    advisor_blocks_total = 0
    risk_gate_closed_frames = 0  # frames where --control_risk_gate bypassed control

    # FC challenge-and-switch controller (one instance, reset per sequence).
    fc_challenge_ctrl: Optional[FCChallengeController] = None
    if args.policy_fc_challenge:
        fc_challenge_ctrl = FCChallengeController(
            config=FCChallengeConfig(
                confirm_frames=int(args.fc_challenge_confirm),
                challenge_max_frames=int(args.fc_challenge_max_frames),
                abort_window=int(args.fc_challenge_abort_window),
                sim_margin=float(args.fc_challenge_sim_margin),
                apce_keep_ratio=float(args.fc_challenge_apce_keep),
                reappear_radius=float(args.fc_challenge_reappear_radius),
                switch_mode=str(args.fc_challenge_switch_mode),
                min_switch_disp=float(args.fc_challenge_min_switch_disp),
                assoc_gate=float(args.fc_challenge_assoc_gate),
            )
        )

    # FC detect-verify-recover controller (--policy_fc_recover).
    # Constructed once and reset per sequence. The candidate generator merges
    # SGLA-internal redetect with optional event-triggered RT-DETRv2-S.
    fc_recover_ctrl = None
    fc_recover_generator = None
    fc_recover_starts = 0
    fc_recover_switches = 0
    fc_recover_commits = 0
    fc_recover_rollbacks = 0
    fc_recover_aborts = 0
    fc_recover_redetect_calls = 0
    fc_recover_frames = 0
    fc_recover_distractor_seeds = 0
    fc_recover_verified_total = 0
    fc_recover_detector_used = 0
    if args.policy_fc_recover:
        from csc_lib.csc.recover import (
            FCRecoverConfig,
            FCRecoverController,
            CandidateGeneratorConfig,
            MultiSourceCandidateGenerator,
        )
        fc_recover_ctrl = FCRecoverController(
            config=FCRecoverConfig(
                fc_streak_required=int(args.fc_recover_streak),
                redetect_top_k=int(args.fc_recover_top_k),
                sprt_alpha=float(args.fc_recover_sprt_alpha),
                sprt_beta=float(args.fc_recover_sprt_beta),
                sprt_false_alert_budget=int(args.fc_recover_sprt_budget),
                verifier_accept_margin=float(args.fc_recover_accept_margin),
                verifier_distractor_veto=float(args.fc_recover_distractor_veto),
                verifier_min_identity=float(args.fc_recover_min_identity),
                challenge_max_frames=int(args.fc_recover_challenge_max),
                abort_window=int(args.fc_recover_abort_window),
                rollback_la_fc_streak=int(args.fc_recover_rollback_streak),
                motion_max_disp=float(args.fc_recover_motion_max_disp),
                motion_max_disp_per_frame=float(args.fc_recover_motion_max_disp_per_frame),
            )
        )
        # Optional external detector (RT-DETRv2-S). Loads lazily on first detect()
        # call; failure to load downgrades to SGLA-only without aborting the run.
        _detector = None
        if args.fc_recover_use_detector:
            try:
                import uav_tracker.detectors.rtdetr  # noqa: F401  registers detector
                from uav_tracker.registry import DETECTORS
                _detector = DETECTORS.build(
                    "rtdetrv2_s",
                    device=args.device,
                    conf_threshold=float(args.fc_recover_detector_conf),
                )
                log.info("fc_recover: RT-DETRv2-S detector wired (conf>=%.2f)",
                         float(args.fc_recover_detector_conf))
            except Exception as exc:
                log.warning("fc_recover: failed to load RT-DETRv2-S (%s); SGLA-only", exc)
                _detector = None
        fc_recover_generator = MultiSourceCandidateGenerator(
            detector=_detector,
            config=CandidateGeneratorConfig(
                sgla_top_k=int(args.fc_recover_top_k),
                use_external_detector=(_detector is not None),
                detector_top_k=int(args.fc_recover_detector_top_k),
                detector_conf_threshold=float(args.fc_recover_detector_conf),
            ),
        )

    # Resume correctness: seed aggregate counters from already-completed state
    # files so a resumed (0-new-frame) run still reports correct totals instead
    # of overwriting metrics.json with zeros. The per-frame "risk_gate_open"
    # field is written into every state row when --control_risk_gate is on, so
    # the closed-frame count is exactly reconstructable.
    for _done_stem in done_seqs:
        _sf = states_dir / f"{_done_stem}.jsonl"
        try:
            _rows = [json.loads(_l) for _l in _sf.read_text().splitlines() if _l.strip()]
        except Exception:
            continue
        total_frames += len(_rows)
        if args.control_risk_gate:
            risk_gate_closed_frames += sum(
                1 for _r in _rows if not _r.get("risk_gate_open", True)
            )

    for i, seq in enumerate(sequences):
        log.info("[%d/%d] %s", i + 1, n_seq, seq.name)

        # Determine image size from first frame for CSC reset
        frames_iter = iter(seq.frames)
        try:
            first_frame = next(frames_iter)
        except StopIteration:
            log.warning("sequence %s is empty, skipping", seq.name)
            continue
        h, w = first_frame.shape[:2]
        csc.reset(image_size=(int(w), int(h)))

        # Reset exit router for each new sequence so hysteresis state doesn't
        # carry over from previous sequences.
        if exit_router is not None:
            exit_router.reset()

        if csc_advisor is not None:
            csc_advisor.reset()

        if fc_challenge_ctrl is not None:
            fc_challenge_ctrl.reset()
        if fc_recover_ctrl is not None:
            fc_recover_ctrl.reset()

        bboxes: list[tuple[float, float, float, float]] = []
        telemetry: list[dict] = []
        states_rows: list[dict] = []
        seq_tracker_ms: list[float] = []
        seq_csc_ms: list[float] = []
        # Trailing per-frame risk signals for the --control_risk_gate.  Reset per
        # sequence so risk from one sequence never leaks into the next.
        risk_window: list[dict] = []

        # --- Init frame 0 ---
        t0 = time.perf_counter()
        tracker.init(first_frame, seq.init_bbox)
        init_ms = (time.perf_counter() - t0) * 1000.0

        bboxes.append(_bbox_tuple(seq.init_bbox))
        telemetry.append({"frame_idx": 0, "init": True, "latency_ms": init_ms})
        states_rows.append({"frame_idx": 0, "init": True})

        # --- Subsequent frames ---
        frame_idx = 1
        # Force-layer idx to apply at the NEXT frame's update().  Holds the FINAL
        # routing decision: the reactive router idx, OR the proactive_v3 override
        # when it fires.  BUGFIX: previously the loop re-read exit_router.current_idx
        # here, but the proactive_v3 branch only mutates a LOCAL exit_force_idx and
        # never the router's current_idx — so the proactive override was logged but
        # never applied to the tracker (verified: applied force == pure-reactive
        # reconstruction, 100% match on bike1/uav4/duck1_2/S0501/bird1_3).  Tracking
        # the decision explicitly makes proactive_v3 actually reach the tracker
        # (causal: decided at end of frame t, applied at t+1).
        next_force_idx = -1  # -1 = let SGLATrack MLP decide (== passive default)
        # Wider-search lever state (per sequence). consec_la counts the current
        # contiguous LA run; next_search_factor is the factor to apply at the
        # NEXT update() (causal). Both reset per sequence so a lost episode in
        # one sequence never widens the start of the next.
        consec_la = 0
        next_search_factor = base_search_factor
        # Gated re-detection per-sequence state (reset so a loss in one sequence
        # never relocates/widens the start of the next). consec_gate_la counts the
        # current contiguous GATED-LA (true-loss) run; next_search_center holds a
        # one-shot relocate target to apply at the next update().
        consec_gate_la = 0
        next_search_center = None
        # Motion-bridge state (--redetect_action motion_bridge): velocity estimated
        # from CONFIRMED frames + last trusted center/size, used to extrapolate the
        # target's position during a gated loss. Reset per sequence.
        bridge_vel = (0.0, 0.0)
        last_good_center = None
        last_good_size = None
        frames_since_loss = 0
        vel_resid = 1e9  # EMA of velocity prediction-error magnitude (regularity); 1e9 = unmeasured
        # FC-control per-sequence state (--policy_fc_control): last CONFIRMED
        # position (for the hold_lastgood action) + consecutive-FC streak counter.
        last_cc_center = None
        last_cc_size = None
        consec_fc = 0
        recovery_update_frames = 0  # >0 => post-loss re-acquisition window; allow runner template refresh
        for frame in frames_iter:
            # --- StateExitRouter: apply the previous frame's FINAL force decision
            # BEFORE calling tracker.update().  On frame 1 next_force_idx is -1
            # (MLP decides), same as the tracker default.
            if exit_router is not None or args.policy_fc_control:
                # _force_layer_idx is read on every update() call.  Direct
                # attribute set is safe — no method encapsulation needed here.
                # (policy_fc_control applies its block9 action router-independently.)
                tracker._force_layer_idx = next_force_idx  # type: ignore[attr-defined]

            # Wider-search lever: apply the previous frame's search-factor decision
            # BEFORE update() (causal — same as _force_layer_idx). Default base
            # factor == passive behaviour; raised on LA when the policy is on.
            if args.policy_lost_widen_search:
                if (_try_set_search_factor(tracker, next_search_factor)
                        and next_search_factor > base_search_factor + 1e-6):
                    search_widen_frames += 1

            # Gated re-detection: apply the previous frame's recovery decision
            # BEFORE update() (causal). Always restore the search factor (base ==
            # passive) so a widen un-arms on recovery; relocate is a one-shot
            # override of the search center onto a re-detection candidate.
            # (Also serves --policy_fc_control: same search-factor/center levers.)
            if args.policy_gated_redetect or args.policy_fc_control or args.policy_fc_challenge or args.policy_fc_recover:
                if (_try_set_search_factor(tracker, next_search_factor)
                        and next_search_factor > base_search_factor + 1e-6):
                    search_widen_frames += 1
                if next_search_center is not None:
                    if _try_override_search_center(tracker, *next_search_center):
                        gated_relocate_frames += 1
                    next_search_center = None  # one-shot

            t_track = time.perf_counter()
            state = tracker.update(frame)
            track_ms = (time.perf_counter() - t_track) * 1000.0

            tel = _telemetry_from_state(state)
            tel["frame_idx"] = frame_idx
            tel["latency_ms"] = track_ms
            # Identity / appearance drift signals (SGLATrack only; other adapters
            # return the default 1.0 so CSC sees a safe constant instead of None).
            tel["initial_template_sim"] = float(getattr(tracker, "_initial_template_sim", 1.0))
            tel["last_cosine_sim"]       = float(getattr(tracker, "_last_cosine_sim", 1.0))
            tel["appearance_drift"]      = float(1.0 - tel["initial_template_sim"])
            # Capture the block that SGLATrack actually used (patched backbone
            # returns this in aux["selected_block"]; -1 if not available).
            _applied_force = getattr(tracker, "_force_layer_idx", -1)
            _selected_block = None
            _aux = getattr(state, "aux", None)
            if isinstance(_aux, dict):
                _sb = _aux.get("selected_block")
                if _sb is not None:
                    _selected_block = int(_sb)
            telemetry.append(tel)

            bbox = _bbox_tuple(getattr(state, "bbox", None))
            bboxes.append(bbox)

            # Score-map candidate peaks (frame-space bboxes, post-NMS) for the
            # relocate action. _telemetry_from_state drops these (it keeps only
            # scalar sm_* fields), so read them straight off the TrackState.
            cur_candidates = []
            _sms = getattr(state, "score_map_stats", None)
            if isinstance(_sms, dict) and isinstance(_sms.get("candidates"), list):
                cur_candidates = _sms["candidates"]
            gate_fired_this_frame = False
            fc_fired_this_frame = False
            sgla_redetect_this_frame = False
            sgla_redetect_hit_this_frame = False
            sgla_redetect_result = None
            sgla_redetect_ms = 0.0
            # FC challenge-and-switch per-frame decision (set in its control block).
            fc_challenge_decision = None
            fc_challenge_active_this_frame = False
            fc_challenge_committed_this_frame = False
            # FC recover (V4-borrow) per-frame decision (set below if armed).
            fc_recover_decision = None
            fc_recover_active_this_frame = False
            fc_recover_committed_this_frame = False

            with torch.no_grad():
                csc_pred = csc.step(
                    confidence=tel.get("confidence"),
                    apce=tel.get("apce"),
                    psr=tel.get("psr"),
                    pred_bbox=bbox,
                    extra=tel,  # V3 builder reads response_entropy + sm_* from here (ignored by V1/V2)
                )

            # --- Risk-gate (causal): decide whether control is allowed on this
            # frame.  When --control_risk_gate is OFF, gate_open stays True and
            # behaviour is identical to before.  When ON, the gate CLOSES (control
            # fully bypassed = passive) only when the trailing window shows NO
            # recent evidence of risk on EVERY axis: no LA/FC state, forecast
            # fc_n10/failure_n10 below threshold, and risk_score below threshold.
            # Uses only current + past frames (no look-ahead).  This removes the
            # easy-scene template-update drift cost (boat2/boat6) while keeping
            # hard-scene gains (persistent risk keeps the gate open).
            gate_open = True
            if args.control_risk_gate and args.csc_mode == "control":
                _fc10 = csc_pred.false_confirmed_next_10_prob
                _fail10 = csc_pred.failure_next_10_prob
                risk_window.append({
                    "state": int(csc_pred.derived_state),
                    "fc10": float(_fc10) if _fc10 is not None else 0.0,
                    "fail10": float(_fail10) if _fail10 is not None else 0.0,
                    "risk": float(csc_pred.risk_score),
                })
                if len(risk_window) > args.risk_gate_window:
                    del risk_window[0]
                # Reactive-state axis: require >= risk_gate_open_streak risky
                # (LA/FC) frames in the trailing window before it counts as risk.
                # Default 1 == prior "any single frame" behaviour; 2 ignores an
                # isolated single-frame false-LA on an easy scene (stays 0-control).
                _n_failure_state = sum(1 for r in risk_window if r["state"] in (2, 3))
                _has_failure_state = _n_failure_state >= args.risk_gate_open_streak
                _no_recent_risk = (
                    not _has_failure_state
                    and max(r["fc10"] for r in risk_window) < args.risk_gate_fc_thresh
                    and max(r["fail10"] for r in risk_window) < args.risk_gate_fc_thresh
                    and max(r["risk"] for r in risk_window) < args.risk_gate_risk_thresh
                )
                gate_open = not _no_recent_risk
                if not gate_open:
                    risk_gate_closed_frames += 1

            # --- StateExitRouter: advance router with this frame's CSC state.
            # The returned force_idx will be applied at the NEXT frame's
            # tracker.update() call (causal — no look-ahead).
            exit_force_idx: Optional[int] = None
            exit_state_str: Optional[str] = None
            proactive_fired = False
            if exit_router is not None:
                from uav_tracker.trackers.exit_router import (
                    DERIVED_INT_TO_ROUTER_STATE,
                    _DERIVED_FALLBACK_STATE,
                )
                exit_state_str = DERIVED_INT_TO_ROUTER_STATE.get(
                    int(csc_pred.derived_state), _DERIVED_FALLBACK_STATE
                )
                # Use the argmax of localization_probs as state confidence proxy
                # (not a true per-class confidence, but a reasonable signal).
                _loc_conf = float(
                    csc_pred.localization_probs[csc_pred.predicted_localization]
                )
                exit_force_idx = exit_router.step(
                    exit_state_str, state_confidence=_loc_conf
                )
                # Proactive V3: if fc_n10_prob > threshold and current state is not yet FC,
                # override to force block 9 pre-emptively (acts before failure establishes).
                # Suppressed when the risk-gate is closed (no recent risk → passive).
                if (gate_open
                        and args.proactive_v3
                        and csc_pred.false_confirmed_next_10_prob is not None
                        and float(csc_pred.false_confirmed_next_10_prob) >= args.proactive_threshold
                        and int(csc_pred.derived_state) != 3):  # not already FC
                    exit_force_idx = 3  # force block 9 (deeper, more robust)
                    proactive_fired = True
                # --- policy_hold_on_la: DO NOTHING on LA (hold). The proper LOST
                # actions (wider search / re-detect) are NOT wired (see TODO at
                # --policy_hold_on_la), and block-9 thrashing on a lost frame
                # drives a drift feedback loop (false LA -> block-9 -> worse ->
                # more LA, e.g. uav6 LA 15%->40% under default control). So on LA
                # suppress the block override (keep the MLP default) — the
                # template still freezes via freeze_on_la. Final word for LA.
                if args.policy_hold_on_la and int(csc_pred.derived_state) == 2:
                    exit_force_idx = -1
                    proactive_fired = False
                # --- policy_lost_widen_search: the REAL LOST action. On LA, the
                # recovery lever is a WIDER search region (set next frame), NOT
                # block-9 (which thrashes a false-LA into a drift loop). Block-9
                # stays the FC action (router already maps FC->block9). Progressive:
                # the factor grows with the contiguous LA run and resets on recovery
                # (CC). Gated by the risk-gate (closed => passive base factor).
                if args.policy_lost_widen_search:
                    _ds = int(csc_pred.derived_state)
                    if gate_open and _ds == 2:            # LA
                        consec_la += 1
                        if consec_la >= args.lost_arm_frames:   # only after sustained loss
                            _steps = consec_la - args.lost_arm_frames + 1
                            _mult = min(args.lost_widen_max,
                                        1.0 + args.lost_widen_step * _steps)
                            next_search_factor = base_search_factor * _mult
                            exit_force_idx = -1           # widen handles LA, not block-9
                            proactive_fired = False
                        # else: not yet armed -> hold (do nothing on a short false-LA)
                    elif gate_open and _ds == 3:          # FC: keep block-9 via router
                        pass                              #     (hold current search factor)
                    else:                                 # CC / CU / gate-closed -> recovered
                        consec_la = 0
                        next_search_factor = base_search_factor
                    # CU(1)/FC(3) with gate open: hold the current widen (FC still
                    # gets its block-9 via the router; CU rides out the episode).
                # Persist the FINAL routing decision (reactive idx or proactive
                # override) so the NEXT frame's update() actually applies it.
                # When the risk-gate is closed, bypass routing entirely (-1 =
                # passive MLP default) — the router still advanced above so its
                # hysteresis stays coherent for when the gate reopens.
                if gate_open:
                    next_force_idx = exit_force_idx if exit_force_idx is not None else -1
                else:
                    next_force_idx = -1

            # --- policy_gated_redetect: GATED LA recovery (false-LA wall fix).
            # Router-independent. On LA the HARD-true-LA gate decides genuine-loss
            # vs false-LA from runtime signals; only a genuine loss (and only after
            # redetect_arm_frames of SUSTAINED gated-LA) triggers the action
            # (widen / relocate / both). A false-LA frame holds: base factor, no
            # relocate, and (router on) no block-9 thrash — i.e. == policy_hold_on_la.
            # --- LA + FC control decisions. policy_gated_redetect acts on LA (state
            # 2); policy_fc_control acts on FC (state 3) — DISJOINT states, so they
            # COMPOSE. The search levers are reset to their per-frame default ONCE
            # here, BEFORE either branch, so neither policy's "not my state" path
            # clobbers the other's action (the widen factor is recomputed from the LA
            # streak each frame; the relocate/bridge/hold center is one-shot).
            if (args.policy_gated_redetect or args.policy_fc_control or args.policy_fc_challenge or args.policy_fc_recover) and args.csc_mode == "control":
                next_search_factor = base_search_factor
                next_search_center = None
                # fc_control owns the force-idx lever only when there is no router
                # (with a router, the router block above already set next_force_idx).
                if args.policy_fc_control and exit_router is None:
                    next_force_idx = -1

            if args.policy_gated_redetect and args.csc_mode == "control":
                _ds = int(csc_pred.derived_state)
                # Motion-bridge bookkeeping: on CONFIRMED frames trust the position
                # and refresh the velocity model (EMA of frame-to-frame center delta),
                # so a later gated loss can be bridged by extrapolating last-good motion.
                if args.redetect_action in ("motion_bridge", "bridge_relocate", "sgla_redetect", "bridge_sgla"):
                    _bx, _by, _bw, _bh = bbox
                    _cx_now, _cy_now = _bx + _bw / 2.0, _by + _bh / 2.0
                    if _ds == 0:  # CC: trust position, refresh velocity + regularity
                        if last_good_center is not None:
                            _ivx = _cx_now - last_good_center[0]
                            _ivy = _cy_now - last_good_center[1]
                            _resid = ((_ivx - bridge_vel[0]) ** 2 + (_ivy - bridge_vel[1]) ** 2) ** 0.5
                            _a = args.bridge_vel_ema
                            vel_resid = _resid if vel_resid >= 1e8 else (_a * vel_resid + (1 - _a) * _resid)
                            bridge_vel = (_a * bridge_vel[0] + (1 - _a) * _ivx,
                                          _a * bridge_vel[1] + (1 - _a) * _ivy)
                        last_good_center = (_cx_now, _cy_now)
                        last_good_size = (_bw, _bh)
                        frames_since_loss = 0
                # Forecast-driven firing: lost_aware_next_10 / failure_next_10 heads.
                _la10 = csc_pred.lost_aware_next_10_prob
                _fail10 = csc_pred.failure_next_10_prob
                _la10f = float(_la10) if _la10 is not None else 0.0
                _fail10f = float(_fail10) if _fail10 is not None else 0.0
                _forecast_ok = (args.gate_forecast_thresh <= 0.0) or (_la10f >= args.gate_forecast_thresh)
                # reactive, forecast-CONFIRMED true loss (precision filter when thresh>0):
                _reactive_fire = (_ds == 2 and _hard_la_gate(tel, csc_pred, args) and _forecast_ok)
                # PROACTIVE pre-arm on a forecast warning before the loss establishes:
                _proactive_fire = (args.proactive_bridge and _ds != 2
                                   and _la10f >= args.proactive_la_thresh
                                   and _fail10f >= args.proactive_fail_thresh)
                if gate_open and (_reactive_fire or _proactive_fire):
                    gate_fired_this_frame = True
                    consec_gate_la += 1
                    gated_la_frames += 1
                    frames_since_loss += 1
                    if consec_gate_la >= args.redetect_arm_frames:
                        _steps = consec_gate_la - args.redetect_arm_frames + 1
                        if args.redetect_action in ("widen", "widen_relocate"):
                            next_search_factor = base_search_factor * min(
                                args.lost_widen_max, 1.0 + args.lost_widen_step * _steps)
                        if args.redetect_action in ("relocate", "widen_relocate"):
                            _cand = _best_relocate_candidate(cur_candidates, args.relocate_min_ratio)
                            if _cand is not None:
                                next_search_center = _cand
                        _do_bridge = (args.redetect_action == "motion_bridge")
                        if args.redetect_action == "bridge_relocate":
                            # Hybrid: prefer a STRONG secondary score-map peak (target
                            # reappeared after an abrupt move, e.g. person19_3 +0.27);
                            # fall back to motion extrapolation when no confident peak
                            # (smooth loss, e.g. group3_2 +0.49). A high --relocate_min_ratio
                            # keeps easy scenes safe (relocate to a weak peak drifts, person9).
                            _cand = _best_relocate_candidate(cur_candidates, args.relocate_min_ratio)
                            if _cand is not None:
                                next_search_center = _cand
                            else:
                                _do_bridge = True
                        if args.redetect_action in ("sgla_redetect", "bridge_sgla"):
                            sgla_redetect_this_frame = True
                            sgla_redetect_calls += 1
                            sgla_redetect_result, sgla_redetect_ms = _try_sgla_redetect(
                                tracker,
                                frame,
                                args,
                                last_good_center=last_good_center,
                                last_good_size=last_good_size,
                                cur_bbox=bbox,
                                frame_idx=frame_idx,
                            )
                            sgla_redetect_ms_total += float(sgla_redetect_ms)
                            track_ms += float(sgla_redetect_ms)
                            tel["latency_ms"] = float(track_ms)
                            if sgla_redetect_result is not None:
                                # V4 verifier gate: reject low-identity (distractor) jumps.
                                _verified = True
                                if args.sgla_redetect_min_sim > 0.0:
                                    _sim = float(sgla_redetect_result.get("sim_to_init", float("nan")))
                                    _verified = bool(np.isfinite(_sim) and _sim >= args.sgla_redetect_min_sim)
                                if _verified:
                                    _cx, _cy = sgla_redetect_result["center"]
                                    _bw, _bh = sgla_redetect_result["bbox"][2], sgla_redetect_result["bbox"][3]
                                    next_search_center = (float(_cx), float(_cy), float(_bw), float(_bh))
                                    sgla_redetect_hit_this_frame = True
                                    sgla_redetect_hits += 1
                                elif args.redetect_action == "bridge_sgla":
                                    _do_bridge = True   # rejected candidate -> fall back to motion_bridge
                            elif args.redetect_action == "bridge_sgla":
                                _do_bridge = True
                        if (_do_bridge
                                and last_good_center is not None
                                and frames_since_loss <= args.bridge_max_frames):
                            _w, _h = last_good_size if last_good_size else (bbox[2], bbox[3])
                            _scale = max(1.0, (_w * _h) ** 0.5)
                            # Regularity gate: only extrapolate when pre-loss motion was
                            # smooth (low velocity residual); erratic targets → hold.
                            if vel_resid / _scale <= args.bridge_max_resid_ratio:
                                _dx = bridge_vel[0] * frames_since_loss
                                _dy = bridge_vel[1] * frames_since_loss
                                # Safety cap: bound the total extrapolated displacement to
                                # bridge_max_disp * scale from the last-good position. A wrong
                                # velocity on a (gate-misfired) easy track would otherwise send
                                # the search runaway and lose a healthy target (UAVTrack112 car6_2
                                # 0.84->0.26). 0 = off.
                                if args.bridge_max_disp > 0:
                                    _mag = (_dx * _dx + _dy * _dy) ** 0.5
                                    _cap = args.bridge_max_disp * _scale
                                    if _mag > _cap and _mag > 1e-6:
                                        _r = _cap / _mag
                                        _dx *= _r; _dy *= _r
                                next_search_center = (last_good_center[0] + _dx,
                                                      last_good_center[1] + _dy, _w, _h)
                    if exit_router is not None:
                        next_force_idx = -1   # don't thrash block-9 on a lost frame
                else:
                    # not a gated true-loss: hold (search defaults already reset above).
                    if _ds == 2:
                        consec_gate_la = max(0, consec_gate_la - 1)  # decay, not hard reset
                        frames_since_loss += 1
                        if exit_router is not None:
                            next_force_idx = -1   # still LA: hold, no block-9 thrash
                    else:
                        consec_gate_la = 0

            if args.policy_fc_control and args.csc_mode == "control":
                _ds = int(csc_pred.derived_state)
                _bx, _by, _bw, _bh = bbox
                if _ds == 0:  # CC: remember last trusted position
                    last_cc_center = (_bx + _bw / 2.0, _by + _bh / 2.0)
                    last_cc_size = (_bw, _bh)
                if gate_open and _ds == 3:  # FC
                    consec_fc += 1
                    _fc_gate_ok = (args.fc_gate_vote_k <= 0
                                   or _combined_vote_count(tel, args) >= args.fc_gate_vote_k)
                    if consec_fc >= args.fc_streak_frames and _fc_gate_ok:
                        fc_fired_this_frame = True
                        fc_fired_frames += 1
                        act = args.fc_action
                        if act == "block9":
                            next_force_idx = 3   # force block 9 (deeper compute)
                        elif act == "widen":
                            next_search_factor = base_search_factor * args.lost_widen_max
                        elif act == "relocate":
                            _cand = _best_relocate_candidate(cur_candidates, args.relocate_min_ratio)
                            if _cand is not None:
                                next_search_center = _cand
                        elif act == "hold_lastgood":
                            if last_cc_center is not None:
                                _w, _h = last_cc_size if last_cc_size else (_bw, _bh)
                                next_search_center = (last_cc_center[0], last_cc_center[1], _w, _h)
                        # freeze_only: no search/compute change; freeze applied below
                else:
                    consec_fc = 0

            # --- policy_fc_challenge: FC challenge-and-switch (read-only redetect,
            # temporal verification, switch + abort window with rollback). Disjoint
            # from LA (state 2), so it COMPOSES with policy_gated_redetect. Unlike
            # policy_fc_control, the incumbent is NEVER moved until a candidate is
            # confirmed stably better — a false FC costs only extra forwards.
            if (args.policy_fc_challenge and args.csc_mode == "control"
                    and fc_challenge_ctrl is not None):
                _ds = int(csc_pred.derived_state)
                _bx, _by, _bw, _bh = bbox
                if _ds == 0:  # CC: remember last trusted position (rollback fallback)
                    last_cc_center = (_bx + _bw / 2.0, _by + _bh / 2.0)
                    last_cc_size = (_bw, _bh)
                # High-FC-risk trigger: streak + (optional) high-precision vote gate.
                # Only a trigger STARTS a challenge; the controller owns the rest.
                if gate_open and _ds == 3:  # FC
                    consec_fc += 1
                else:
                    consec_fc = 0
                _fc_gate_ok = (args.fc_gate_vote_k <= 0
                               or _combined_vote_count(tel, args) >= args.fc_gate_vote_k)
                _fc_trigger = bool(
                    gate_open and _ds == 3
                    and consec_fc >= args.fc_challenge_streak and _fc_gate_ok
                )

                def _challenge_redetect():
                    _topk = (int(args.fc_challenge_topk)
                             if args.fc_challenge_switch_mode == "association" else 1)
                    _res, _ms = _try_sgla_redetect(
                        tracker, frame, args,
                        last_good_center=last_cc_center,
                        last_good_size=last_cc_size,
                        cur_bbox=bbox,
                        top_k=_topk,
                        frame_idx=frame_idx,
                    )
                    return _res, _ms

                fc_challenge_decision = fc_challenge_ctrl.step(
                    derived_state=_ds,
                    fc_trigger=_fc_trigger,
                    bbox=bbox,
                    initial_template_sim=float(tel.get("initial_template_sim", 1.0)),
                    incumbent_apce=float(tel.get("apce", 0.0)),
                    incumbent_center=last_cc_center,
                    incumbent_size=last_cc_size,
                    redetect_fn=_challenge_redetect,
                )
                _dec = fc_challenge_decision

                # Re-detect accounting (read-only forwards run inside step()).
                if _dec.ran_redetect:
                    sgla_redetect_this_frame = True
                    sgla_redetect_calls += 1
                    fc_challenge_redetect_calls += 1
                    sgla_redetect_ms = float(_dec.redetect_ms)
                    sgla_redetect_ms_total += float(_dec.redetect_ms)
                    track_ms += float(_dec.redetect_ms)
                    tel["latency_ms"] = float(track_ms)

                # Apply the one-shot search-center lever: a verified SWITCH or a ROLLBACK.
                if _dec.switch_center is not None:
                    next_search_center = _dec.switch_center
                    sgla_redetect_hit_this_frame = True
                    sgla_redetect_hits += 1
                    fc_challenge_switches += 1
                elif _dec.rollback_center is not None:
                    next_search_center = _dec.rollback_center
                    fc_challenge_rollbacks += 1

                # Freeze during the whole challenge/abort episode; open the recovery
                # window only AFTER a commit so the template can lock in the switch.
                fc_challenge_active_this_frame = bool(_dec.freeze_template)
                fc_challenge_committed_this_frame = bool(_dec.committed)
                # Aggregate counters (decision flags — controller counters reset per seq).
                if _dec.started:
                    fc_challenge_starts += 1
                if _dec.committed:
                    fc_challenge_commits += 1
                if _dec.aborted:
                    fc_challenge_aborts += 1
                if fc_challenge_ctrl.active or _dec.committed or _dec.aborted or _dec.switch_center is not None:
                    fc_challenge_frames += 1

            # --- policy_fc_recover: detect-verify-recover (csc_lib/csc/recover) ---
            # Same FC-streak trigger as policy_fc_challenge but routes the candidate
            # set through PrototypeMemory (anchor + recent + distractor) ->
            # CandidateVerifier -> SPRT (adaptive temporal confirmation). Optional
            # event-triggered RT-DETRv2-S boxes augment SGLATrack's internal pool
            # (Phase 0 oracle showed SGLA-internal recall@IoU.5 = 0% on hard FC).
            if (args.policy_fc_recover and args.csc_mode == "control"
                    and fc_recover_ctrl is not None):
                _ds = int(csc_pred.derived_state)
                _bx, _by, _bw, _bh = bbox

                # Maintain memory: seed anchor on first call, recent prototype on
                # confidence-validated CC frames, last-CC bbox always.
                _anchor_emb = getattr(tracker, "_initial_template_embedding", None)
                if _anchor_emb is not None:
                    fc_recover_ctrl.maybe_seed_anchor(_anchor_emb)
                _peak_emb = getattr(tracker, "_last_search_peak_local", None)
                if _ds == 0:
                    last_cc_center = (_bx + _bw / 2.0, _by + _bh / 2.0)
                    last_cc_size = (_bw, _bh)
                    if _peak_emb is not None and float(tel.get("confidence", 0.0)) >= \
                            fc_recover_ctrl.config.update_recent_min_confidence:
                        fc_recover_ctrl.note_cc(
                            _peak_emb, frame_idx=frame_idx,
                            bbox=(_bx, _by, _bw, _bh),
                        )

                # FC-streak gate (mirrors fc_challenge gating; precision vote optional).
                if gate_open and _ds == 3:
                    consec_fc += 1
                else:
                    consec_fc = 0
                _fc_gate_ok = (args.fc_gate_vote_k <= 0
                               or _combined_vote_count(tel, args) >= args.fc_gate_vote_k)
                _fc_trigger = bool(
                    gate_open and _ds == 3
                    and consec_fc >= args.fc_recover_streak and _fc_gate_ok
                )

                # Build the candidate generator callback (closure over tracker/frame).
                _last_cc_bbox_obj = None
                if last_cc_center is not None and last_cc_size is not None:
                    from uav_tracker.types import BBox as _BBox
                    _last_cc_bbox_obj = _BBox(
                        x=last_cc_center[0] - last_cc_size[0] / 2.0,
                        y=last_cc_center[1] - last_cc_size[1] / 2.0,
                        w=last_cc_size[0], h=last_cc_size[1],
                    )

                def _recover_redetect():
                    if fc_recover_generator is None:
                        return [], 0.0
                    return fc_recover_generator(
                        tracker, frame,
                        last_cc_bbox=_last_cc_bbox_obj,
                        frame_idx=frame_idx,
                    )

                fc_recover_decision = fc_recover_ctrl.step(
                    derived_state=_ds,
                    fc_trigger=_fc_trigger,
                    incumbent_bbox=bbox,
                    incumbent_emb=_peak_emb,
                    redetect_fn=_recover_redetect,
                    frame_idx=frame_idx,
                )
                _dec = fc_recover_decision

                # Re-detect accounting (read-only forwards run inside step()).
                if _dec.ran_redetect:
                    sgla_redetect_this_frame = True
                    sgla_redetect_calls += 1
                    fc_recover_redetect_calls += 1
                    sgla_redetect_ms = float(_dec.redetect_ms)
                    sgla_redetect_ms_total += float(_dec.redetect_ms)
                    track_ms += float(_dec.redetect_ms)
                    tel["latency_ms"] = float(track_ms)

                # Apply the one-shot search-center lever: a verified SWITCH or a ROLLBACK.
                if _dec.switch_center is not None:
                    next_search_center = _dec.switch_center
                    sgla_redetect_hit_this_frame = True
                    sgla_redetect_hits += 1
                    fc_recover_switches += 1
                elif _dec.rollback_center is not None:
                    next_search_center = _dec.rollback_center
                    fc_recover_rollbacks += 1

                fc_recover_active_this_frame = bool(_dec.freeze_template)
                fc_recover_committed_this_frame = bool(_dec.committed)
                if _dec.started:
                    fc_recover_starts += 1
                if _dec.committed:
                    fc_recover_commits += 1
                if _dec.aborted:
                    fc_recover_aborts += 1
                if _dec.distractor_seeded:
                    fc_recover_distractor_seeds += 1
                fc_recover_verified_total += int(_dec.n_verified)
                if fc_recover_ctrl.active or _dec.committed or _dec.aborted \
                        or _dec.switch_center is not None:
                    fc_recover_frames += 1

            # --- CSCAdvisor (Variant C): stateful template-update gating ---
            advisor_decision = None
            advisor_state_str: Optional[str] = None
            if csc_advisor is not None:
                from uav_tracker.trackers.csc_advisor import (
                    DERIVED_INT_TO_ADVISOR_STATE,
                    _DERIVED_FALLBACK_STATE,
                )
                # Reuse exit_state_str if exit_router already computed it,
                # otherwise derive from derived_state.
                advisor_state_str = exit_state_str or DERIVED_INT_TO_ADVISOR_STATE.get(
                    int(csc_pred.derived_state), _DERIVED_FALLBACK_STATE
                )
                advisor_decision = csc_advisor.step(advisor_state_str, frame_idx)

            # --- Control hooks (Phase 1: template-update freeze only) ---
            # Gated by the risk-gate: when closed, skip the entire block so no
            # freeze toggle and no try_update_template fire — the template stays
            # as-is (frame-0 on easy-from-start scenes) == passive behaviour.
            # advisor.step()/exit_router.step() already ran above so their
            # hysteresis stays coherent; only the APPLICATION is suppressed.
            # Recovery-window template refresh: after a gated loss/FC action, allow the
            # runner to refresh the template for a few frames (re-acquisition lock-in —
            # the mechanism behind the car9/car1_s recovery). Otherwise keep it frozen
            # == PASSIVE so an always-healthy track never drifts (car6_5). window<0 =
            # legacy (always refresh when unfrozen).
            if args.recovery_update_window >= 0:
                if gate_fired_this_frame or fc_fired_this_frame or fc_challenge_committed_this_frame or fc_recover_committed_this_frame:
                    recovery_update_frames = args.recovery_update_window
                elif recovery_update_frames > 0:
                    recovery_update_frames -= 1

            if args.csc_mode == "control" and gate_open:
                # CSCAdvisor takes priority when enabled (stateful hysteresis);
                # fall back to stateless CSC hint when advisor is off.
                if (args.gated_freeze and (args.policy_gated_redetect or args.policy_fc_control)) \
                        or args.policy_fc_challenge or args.policy_fc_recover:
                    # Gated freeze: freeze the template ONLY on a gated true-loss /
                    # true-FC ACTION frame — NOT on every raw CSC LA/FC hint. The raw
                    # hint over-fires on false-LA/false-FC (the easy-scene majority),
                    # and freezing there blocks a beneficial update on a healthy track
                    # (car6_5 -0.11). When we don't act, we don't freeze == passive.
                    # policy_fc_challenge / policy_fc_recover: the controller owns the
                    # freeze (active during the whole challenge/abort episode,
                    # released on commit).
                    should_freeze = (
                        bool(gate_fired_this_frame)
                        or bool(fc_fired_this_frame)
                        or bool(fc_challenge_active_this_frame)
                        or bool(fc_recover_active_this_frame)
                    )
                else:
                    should_freeze = (
                        advisor_decision.blocked
                        if advisor_decision is not None
                        else bool(csc_pred.should_skip_template_update)
                    )
                # Proactive V3 consistency: a forecast FC-warning must ALSO block
                # the template update (CLAUDE.md control policy: false_confirmed →
                # block update).  Without this, proactive forces block 9 but still
                # lets the template drift onto a distractor — the likely mechanism
                # behind the duck1_2 / S1602 control FCR regressions.
                if proactive_fired:
                    should_freeze = True
                # FC control: every FC action also freezes the template (don't learn
                # appearance from a confidently-wrong frame).
                if fc_fired_this_frame:
                    should_freeze = True
                # FC challenge: freeze throughout the challenge/abort episode; on a
                # COMMIT release the freeze so the verified switch can lock in (the
                # recovery window opened above governs the runner refresh). Same
                # logic for FC recover (V4-borrow stack).
                if fc_challenge_active_this_frame or fc_recover_active_this_frame:
                    should_freeze = True
                elif fc_challenge_committed_this_frame or fc_recover_committed_this_frame:
                    should_freeze = False
                if caps.can_freeze_template:
                    if should_freeze:
                        if _try_set_update_enabled(tracker, False):
                            update_freezes += 1
                    else:
                        # Unfreeze so that try_update_template (if available) can fire.
                        _try_set_update_enabled(tracker, True)
                        # SGLATrack: call try_update_template explicitly — tracker.update()
                        # does not call it internally; the runner is responsible.
                        # --no_runner_template_update skips this so control mode matches
                        # PASSIVE template behaviour on non-action frames (the eval5_clamp
                        # baseline never refreshes the template); the runner-driven refresh
                        # is a confound that drifts healthy easy tracks (car6_5 -0.11).
                        if args.recovery_update_window >= 0:
                            # Recovery-window mode: keep healthy-frame runner updates off,
                            # but allow a short post-action refresh to lock in re-acquisition.
                            _allow_runner_update = recovery_update_frames > 0
                        else:
                            # Legacy mode: --no_runner_template_update disables all explicit
                            # runner-driven template refreshes.
                            _allow_runner_update = not args.no_runner_template_update
                        _try_update_fn = (
                            getattr(tracker, "try_update_template", None)
                            if _allow_runner_update
                            else None
                        )
                        if callable(_try_update_fn):
                            _apce = tel.get("apce", 0.0)
                            _psr = tel.get("psr", 0.0)
                            _cosine_sim = getattr(tracker, "_last_cosine_sim", 1.0)
                            _try_update_fn(
                                frame,
                                getattr(state, "bbox", None),
                                float(_apce),
                                float(_psr),
                                frame_idx,
                                float(_cosine_sim),
                            )
                # TODO: wire should_expand_search once caps.can_widen_search adapters exist.
                # TODO: wire should_request_redetection once detector integration is ready.

            row: dict = {
                "frame_idx": frame_idx,
                "localization_probs": csc_pred.localization_probs.tolist(),
                "confidence_probs": csc_pred.confidence_probs.tolist(),
                "derived_probs": csc_pred.derived_probs.tolist(),
                "predicted_localization": int(csc_pred.predicted_localization),
                "predicted_confidence": int(csc_pred.predicted_confidence),
                "derived_state": int(csc_pred.derived_state),
                "false_confirmed_flag": bool(csc_pred.false_confirmed_flag),
                "risk_score": float(csc_pred.risk_score),
                "aux_probs": csc_pred.aux_probs,
                "should_freeze_template": bool(csc_pred.should_freeze_template),
                "should_expand_search": bool(csc_pred.should_expand_search),
                "should_request_redetection": bool(
                    csc_pred.should_request_redetection
                ),
                "should_skip_template_update": bool(
                    csc_pred.should_skip_template_update
                ),
                # V3 proactive forecast heads (None for V2 ckpts without forecast).
                # Serialised at top level so diagnosis can see early-warning signal
                # even when the per-frame classified state is CC.
                "failure_next_10_prob": (
                    float(csc_pred.failure_next_10_prob)
                    if csc_pred.failure_next_10_prob is not None else None
                ),
                "false_confirmed_next_10_prob": (
                    float(csc_pred.false_confirmed_next_10_prob)
                    if csc_pred.false_confirmed_next_10_prob is not None else None
                ),
                "lost_aware_next_10_prob": (
                    float(csc_pred.lost_aware_next_10_prob)
                    if csc_pred.lost_aware_next_10_prob is not None else None
                ),
                "tracker_latency_ms": float(track_ms),
                "csc_latency_ms": float(csc_pred.latency_ms),
            }
            # Risk-gate telemetry (only populated when --control_risk_gate is on)
            if args.control_risk_gate:
                row["risk_gate_open"] = bool(gate_open)
            # Exit router telemetry (only populated when --exit_router is active)
            if exit_router is not None:
                row["exit_router_state"] = exit_state_str
                row["exit_router_force_idx"] = exit_force_idx
                row["exit_router_applied_force"] = _applied_force
                row["exit_router_selected_block"] = _selected_block
            # Wider-search telemetry (only when --policy_lost_widen_search is on)
            if args.policy_lost_widen_search:
                row["search_factor"] = float(
                    getattr(tracker, "_search_factor", base_search_factor)
                )
                row["consec_la"] = int(consec_la)
            # Gated re-detection telemetry (only when --policy_gated_redetect is on)
            if args.policy_gated_redetect:
                row["gate_fired"] = bool(gate_fired_this_frame)
                row["consec_gate_la"] = int(consec_gate_la)
                row["search_factor"] = float(
                    getattr(tracker, "_search_factor", base_search_factor)
                )
                row["relocate_armed"] = bool(next_search_center is not None)
                row["sgla_redetect_called"] = bool(sgla_redetect_this_frame)
                row["sgla_redetect_hit"] = bool(sgla_redetect_hit_this_frame)
                row["sgla_redetect_latency_ms"] = float(sgla_redetect_ms)
                if sgla_redetect_result is not None:
                    row["sgla_redetect_bbox"] = sgla_redetect_result.get("bbox")
                    row["sgla_redetect_factor"] = float(sgla_redetect_result.get("factor", 0.0))
                    row["sgla_redetect_apce"] = float(sgla_redetect_result.get("apce", 0.0))
                    row["sgla_redetect_quality"] = float(sgla_redetect_result.get("quality", 0.0))
                    row["sgla_redetect_sim_to_init"] = float(sgla_redetect_result.get("sim_to_init", float("nan")))
                    row["sgla_redetect_anchor"] = sgla_redetect_result.get("anchor")
            # FC-control telemetry (only when --policy_fc_control is on)
            if args.policy_fc_control:
                row["fc_fired"] = bool(fc_fired_this_frame)
                row["consec_fc"] = int(consec_fc)
                row["search_factor"] = float(
                    getattr(tracker, "_search_factor", base_search_factor)
                )
            # FC challenge-and-switch telemetry (only when --policy_fc_challenge is on)
            if args.policy_fc_challenge:
                row["fc_challenge_phase"] = (
                    fc_challenge_decision.phase if fc_challenge_decision is not None else "idle"
                )
                row["fc_challenge_reason"] = (
                    fc_challenge_decision.reason if fc_challenge_decision is not None else ""
                )
                row["fc_challenge_active"] = bool(fc_challenge_active_this_frame)
                row["consec_fc"] = int(consec_fc)
                if fc_challenge_decision is not None:
                    _fcd = fc_challenge_decision
                    row["fc_challenge_started"] = bool(_fcd.started)
                    row["fc_challenge_ran_redetect"] = bool(_fcd.ran_redetect)
                    row["fc_challenge_stable_frames"] = int(_fcd.stable_frames)
                    row["fc_challenge_switched"] = bool(_fcd.switch_center is not None)
                    row["fc_challenge_rolled_back"] = bool(_fcd.rollback_center is not None)
                    row["fc_challenge_committed"] = bool(_fcd.committed)
                    row["fc_challenge_aborted"] = bool(_fcd.aborted)
                    if _fcd.cand_center is not None:
                        row["fc_challenge_cand_center"] = [
                            float(_fcd.cand_center[0]), float(_fcd.cand_center[1])
                        ]
                    row["fc_challenge_cand_sim"] = float(_fcd.cand_evidence)
                    row["fc_challenge_incumbent_sim"] = float(_fcd.incumbent_evidence)
            # FC recover (V4-borrow) telemetry (only when --policy_fc_recover is on).
            if args.policy_fc_recover:
                row["fc_recover_phase"] = (
                    fc_recover_decision.phase if fc_recover_decision is not None else "idle"
                )
                row["fc_recover_reason"] = (
                    fc_recover_decision.reason if fc_recover_decision is not None else ""
                )
                row["fc_recover_active"] = bool(fc_recover_active_this_frame)
                row["consec_fc"] = int(consec_fc)
                if fc_recover_decision is not None:
                    _fcr = fc_recover_decision
                    row["fc_recover_started"] = bool(_fcr.started)
                    row["fc_recover_ran_redetect"] = bool(_fcr.ran_redetect)
                    row["fc_recover_n_candidates"] = int(_fcr.n_candidates)
                    row["fc_recover_n_verified"] = int(_fcr.n_verified)
                    row["fc_recover_switched"] = bool(_fcr.switch_center is not None)
                    row["fc_recover_rolled_back"] = bool(_fcr.rollback_center is not None)
                    row["fc_recover_committed"] = bool(_fcr.committed)
                    row["fc_recover_aborted"] = bool(_fcr.aborted)
                    row["fc_recover_distractor_seeded"] = bool(_fcr.distractor_seeded)
                    row["fc_recover_sprt_evidence"] = float(_fcr.sprt_evidence)
                    row["fc_recover_sprt_decision"] = str(_fcr.sprt_decision)
                    row["fc_recover_cand_score"] = float(_fcr.cand_score)
                    row["fc_recover_cand_sim_init"] = float(_fcr.cand_sim_init)
                    row["fc_recover_cand_sim_recent"] = float(_fcr.cand_sim_recent)
                    row["fc_recover_cand_sim_distractor"] = float(_fcr.cand_sim_distractor)
            # CSCAdvisor telemetry (only populated when --csc_advisor is active)
            if advisor_decision is not None:
                row["advisor_state"] = advisor_state_str
                row["advisor_blocked"] = advisor_decision.blocked
                row["advisor_reason"] = advisor_decision.reason
                row["advisor_streak"] = advisor_decision.trusted_streak
                row["advisor_consec_blocked"] = advisor_decision.consecutive_blocked
            states_rows.append(row)

            seq_tracker_ms.append(track_ms)
            seq_csc_ms.append(float(csc_pred.latency_ms))
            tracker_latencies_ms.append(track_ms)
            csc_latencies_ms.append(float(csc_pred.latency_ms))
            total_latencies_ms.append(track_ms + float(csc_pred.latency_ms))
            frame_idx += 1

        # --- Save sequence outputs ---
        with open(pred_dir / f"{seq.name}.txt", "w") as fh:
            for b in bboxes:
                fh.write(",".join(f"{v:.4f}" for v in b) + "\n")
        with open(tel_dir / f"{seq.name}.jsonl", "w") as fh:
            for r in telemetry:
                fh.write(json.dumps(r) + "\n")
        with open(states_dir / f"{seq.name}.jsonl", "w") as fh:
            for r in states_rows:
                fh.write(json.dumps(r) + "\n")

        n_frames = len(bboxes)
        total_frames += n_frames
        seq_total_ms = float(sum(seq_tracker_ms) + sum(seq_csc_ms))
        seq_rec: dict = {
            "sequence": seq.name,
            "n_frames": n_frames,
            "time_ms": seq_total_ms,
            "fps": (
                n_frames / (seq_total_ms / 1000.0) if seq_total_ms > 0 else 0.0
            ),
        }
        if exit_router is not None:
            seq_rec["exit_router_stats"] = exit_router.stats_dict()
            exit_router_switches_total += exit_router.stats.n_switched
        if csc_advisor is not None:
            seq_rec["advisor_stats"] = csc_advisor.stats_dict()
            advisor_blocks_total += csc_advisor.stats.n_blocked
        seq_records.append(seq_rec)

    # --- Runtime metrics ---
    from csc_lib.eval.custom_metrics.runtime_metrics import latency_summary

    tracker_stats = latency_summary(np.array(tracker_latencies_ms))
    csc_stats = latency_summary(np.array(csc_latencies_ms))
    total_stats = latency_summary(np.array(total_latencies_ms))

    metrics = {
        "tracker": args.tracker,
        "dataset": args.dataset,
        "split": args.split,
        "csc_mode": args.csc_mode,
        "csc_checkpoint": args.csc_checkpoint,
        "weights_path": args.weights_path,
        "device": args.device,
        "git_commit": _git_commit(),
        "datetime": datetime.now(timezone.utc).isoformat(),
        "n_sequences": len(seq_records),
        "n_frames": total_frames,
        "mean_tracker_fps": tracker_stats["mean_fps"],
        "mean_csc_fps": csc_stats["mean_fps"],
        "mean_total_fps": total_stats["mean_fps"],
        "control_template_update_freezes": update_freezes,
        "control_search_widen_frames": search_widen_frames,
        "control_gated_la_frames": gated_la_frames,
        "control_gated_relocate_frames": gated_relocate_frames,
        "control_sgla_redetect_calls": sgla_redetect_calls,
        "control_sgla_redetect_hits": sgla_redetect_hits,
        "control_sgla_redetect_hit_rate": (
            sgla_redetect_hits / sgla_redetect_calls if sgla_redetect_calls > 0 else 0.0
        ),
        "control_sgla_redetect_mean_ms": (
            sgla_redetect_ms_total / sgla_redetect_calls if sgla_redetect_calls > 0 else 0.0
        ),
        "control_fc_fired_frames": fc_fired_frames,
        # FC challenge-and-switch summary (--policy_fc_challenge)
        "control_fc_challenge_starts": fc_challenge_starts,
        "control_fc_challenge_switches": fc_challenge_switches,
        "control_fc_challenge_commits": fc_challenge_commits,
        "control_fc_challenge_rollbacks": fc_challenge_rollbacks,
        "control_fc_challenge_aborts": fc_challenge_aborts,
        "control_fc_challenge_redetect_calls": fc_challenge_redetect_calls,
        "control_fc_challenge_active_frames": fc_challenge_frames,
        "control_fc_recover_starts": fc_recover_starts,
        "control_fc_recover_switches": fc_recover_switches,
        "control_fc_recover_commits": fc_recover_commits,
        "control_fc_recover_rollbacks": fc_recover_rollbacks,
        "control_fc_recover_aborts": fc_recover_aborts,
        "control_fc_recover_redetect_calls": fc_recover_redetect_calls,
        "control_fc_recover_active_frames": fc_recover_frames,
        "control_fc_recover_distractor_seeds": fc_recover_distractor_seeds,
        "control_fc_recover_verified_total": fc_recover_verified_total,
        "tracker_latency": {
            k: tracker_stats[k]
            for k in ("mean_ms", "p95_ms", "p99_ms", "median_ms", "p90_ms")
        },
        "csc_latency": {
            k: csc_stats[k]
            for k in ("mean_ms", "p95_ms", "p99_ms", "median_ms", "p90_ms")
        },
        "total_latency": {
            k: total_stats[k]
            for k in ("mean_ms", "p95_ms", "p99_ms", "median_ms", "p90_ms")
        },
        "sequences": seq_records,
    }

    if exit_router is not None:
        metrics["exit_router_enabled"] = True
        metrics["exit_router_min_hold"] = args.exit_router_hold
        metrics["exit_router_min_conf"] = args.exit_router_min_conf
        metrics["exit_router_switches_total"] = exit_router_switches_total
        metrics["exit_router_policy"] = exit_router.policy_summary()
    else:
        metrics["exit_router_enabled"] = False

    if args.control_risk_gate:
        metrics["control_risk_gate_enabled"] = True
        metrics["risk_gate_window"] = args.risk_gate_window
        metrics["risk_gate_fc_thresh"] = args.risk_gate_fc_thresh
        metrics["risk_gate_risk_thresh"] = args.risk_gate_risk_thresh
        metrics["risk_gate_closed_frames"] = risk_gate_closed_frames
        metrics["risk_gate_closed_frac"] = (
            risk_gate_closed_frames / total_frames if total_frames > 0 else 0.0
        )
    else:
        metrics["control_risk_gate_enabled"] = False

    metrics["policy_lost_widen_search"] = bool(args.policy_lost_widen_search)
    if args.policy_lost_widen_search:
        metrics["lost_widen_max"] = args.lost_widen_max
        metrics["lost_widen_step"] = args.lost_widen_step
        metrics["base_search_factor"] = base_search_factor
        metrics["risk_gate_open_streak"] = args.risk_gate_open_streak

    if csc_advisor is not None:
        metrics["csc_advisor_enabled"] = True
        metrics["advisor_streak_required"] = args.advisor_streak
        metrics["advisor_cooldown_frames"] = args.advisor_cooldown
        metrics["advisor_max_hold_frames"] = args.advisor_max_hold
        metrics["advisor_blocks_total"] = advisor_blocks_total
        metrics["advisor_block_policy"] = sorted(csc_advisor._block_states)
    else:
        metrics["csc_advisor_enabled"] = False

    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(
        "done: %d seq, %d frames — tracker %.1f fps, csc %.1f fps, total %.1f fps"
        " (freezes=%d, advisor_blocks=%d, exit_switches=%d, widen=%d) -> %s",
        len(seq_records),
        total_frames,
        metrics["mean_tracker_fps"],
        metrics["mean_csc_fps"],
        metrics["mean_total_fps"],
        update_freezes,
        advisor_blocks_total,
        exit_router_switches_total,
        search_widen_frames,
        out_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
