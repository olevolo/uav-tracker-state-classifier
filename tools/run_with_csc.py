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

sys.path.insert(0, str(PROJECT_ROOT / "salrtd" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))   # src/ wins over salrtd/src/
sys.path.insert(0, str(PROJECT_ROOT))

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
    exit_router_switches_total = 0
    advisor_blocks_total = 0
    risk_gate_closed_frames = 0  # frames where --control_risk_gate bypassed control

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
        for frame in frames_iter:
            # --- StateExitRouter: apply the previous frame's FINAL force decision
            # BEFORE calling tracker.update().  On frame 1 next_force_idx is -1
            # (MLP decides), same as the tracker default.
            if exit_router is not None:
                # _force_layer_idx is read on every update() call.  Direct
                # attribute set is safe — no method encapsulation needed here.
                tracker._force_layer_idx = next_force_idx  # type: ignore[attr-defined]

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
                _has_failure_state = any(r["state"] in (2, 3) for r in risk_window)
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
                # Persist the FINAL routing decision (reactive idx or proactive
                # override) so the NEXT frame's update() actually applies it.
                # When the risk-gate is closed, bypass routing entirely (-1 =
                # passive MLP default) — the router still advanced above so its
                # hysteresis stays coherent for when the gate reopens.
                if gate_open:
                    next_force_idx = exit_force_idx if exit_force_idx is not None else -1
                else:
                    next_force_idx = -1

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
            if args.csc_mode == "control" and gate_open:
                # CSCAdvisor takes priority when enabled (stateful hysteresis);
                # fall back to stateless CSC hint when advisor is off.
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
                if caps.can_freeze_template:
                    if should_freeze:
                        if _try_set_update_enabled(tracker, False):
                            update_freezes += 1
                    else:
                        # Unfreeze so that try_update_template (if available) can fire.
                        _try_set_update_enabled(tracker, True)
                        # SGLATrack: call try_update_template explicitly — tracker.update()
                        # does not call it internally; the runner is responsible.
                        _try_update_fn = getattr(tracker, "try_update_template", None)
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
        " (freezes=%d, advisor_blocks=%d, exit_switches=%d) -> %s",
        len(seq_records),
        total_frames,
        metrics["mean_tracker_fps"],
        metrics["mean_csc_fps"],
        metrics["mean_total_fps"],
        update_freezes,
        advisor_blocks_total,
        exit_router_switches_total,
        out_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
