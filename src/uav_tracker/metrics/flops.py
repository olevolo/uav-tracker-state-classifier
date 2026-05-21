"""FLOPs accounting helper (PLAN §11 Phase 1 exit demo).

Aggregates per-frame tier FLOPs + detector FLOPs + signal FLOPs to the
per-sequence GFLOPs/frame reported in Table 2.

Returns a structured dict with mean, tracker, detector, and saltrd breakdown.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Default YOLO detector FLOPs (GFLOPs per call) — matches yolo.py _YOLOV8N_FLOPS
_YOLOV8N_GFLOPS = 8.7


def flops_per_frame(
    telemetry: Iterable[Any],
    tier_flops: dict[str, float],
    signal_flops_per_frame: float = 0.001,
) -> dict[str, float]:
    """Aggregate GFLOPs/frame from a telemetry stream.

    Parameters
    ----------
    telemetry:
        Iterable of per-frame dicts. Each dict may have:
        - ``"tier"``: tier name (str) — looked up in *tier_flops*
        - ``"used_detector"``: bool — whether a detector call was made this frame
        - ``"saltrd_action_compute"``: str — SALTRD action (e.g. "FULL", "LIGHT", "SKIP")
    tier_flops:
        Tier name → GFLOPs/frame (static estimate). E.g. ``{"full": 0.9, "light": 0.6}``.
    signal_flops_per_frame:
        Per-frame overhead for risk model forward pass (default 0.001 GFLOPs).

    Returns
    -------
    dict with keys:
    - ``"mean_gflops"``: total mean GFLOPs/frame (tracker + detector + saltrd)
    - ``"tracker_gflops"``: tracker compute GFLOPs/frame
    - ``"detector_gflops"``: detector (YOLO) GFLOPs/frame
    - ``"saltrd_gflops"``: SALT-RD signal model GFLOPs/frame
    """
    frames = list(telemetry)
    n = len(frames)

    if n == 0:
        return {
            "mean_gflops": 0.0,
            "tracker_gflops": 0.0,
            "detector_gflops": 0.0,
            "saltrd_gflops": 0.0,
        }

    tracker_total = 0.0
    detector_total = 0.0
    saltrd_total = 0.0

    for frame in frames:
        if isinstance(frame, dict):
            tier = frame.get("tier", None)
            used_detector = frame.get("used_detector", False)
            saltrd_action = frame.get("saltrd_action_compute", None)
        else:
            # Support TelemetryEntry-like objects with attributes
            tier = getattr(frame, "tier", None)
            used_detector = getattr(frame, "used_detector", False)
            saltrd_action = getattr(frame, "saltrd_action_compute", None)

        # Tracker FLOPs from tier
        if tier is not None and tier in tier_flops:
            tracker_total += tier_flops[tier]
        elif tier_flops:
            # fallback: use the max tier value if tier name is unknown
            tracker_total += max(tier_flops.values())

        # Detector FLOPs (YOLO call)
        if used_detector:
            detector_total += _YOLOV8N_GFLOPS

        # SALT-RD signal model FLOPs — active unless action is explicitly SKIP/NONE
        _skip_actions = {"SKIP", "NONE", "skip", "none"}
        if saltrd_action is None or saltrd_action not in _skip_actions:
            saltrd_total += signal_flops_per_frame

    tracker_gflops = tracker_total / n
    detector_gflops = detector_total / n
    saltrd_gflops = saltrd_total / n
    mean_gflops = tracker_gflops + detector_gflops + saltrd_gflops

    return {
        "mean_gflops": mean_gflops,
        "tracker_gflops": tracker_gflops,
        "detector_gflops": detector_gflops,
        "saltrd_gflops": saltrd_gflops,
    }


def measure_tracker_gflops(
    tracker_class: type,
    input_shape: tuple[int, ...] = (1, 3, 256, 256),
) -> float:
    """Attempt thop profile; fallback to tracker.flops_per_update() / 1e9.

    Parameters
    ----------
    tracker_class:
        A tracker class (e.g. SGLATracker) — instantiated with no args.
    input_shape:
        Dummy input shape for thop profiling (default: (1, 3, 256, 256)).

    Returns
    -------
    float
        GFLOPs per forward pass / update call.
    """
    # First, try to get the static estimate from the class or a fresh instance
    try:
        instance = tracker_class()
        if hasattr(instance, "flops_per_update"):
            static_gflops = instance.flops_per_update() / 1e9
        else:
            static_gflops = None
    except Exception as exc:
        logger.debug("Could not instantiate %s for flops estimate: %s", tracker_class, exc)
        static_gflops = None

    # Try thop profiling (requires the model to be loaded and thop installed)
    try:
        import torch
        import thop

        # Attempt to get an underlying nn.Module from the tracker instance
        model = getattr(instance, "_model", None) if "instance" in dir() else None
        if model is None:
            raise AttributeError("No _model attribute found")

        dummy_input = torch.zeros(*input_shape)
        macs, _ = thop.profile(model, inputs=(dummy_input,), verbose=False)
        gflops = float(macs) * 2 / 1e9  # MACs → FLOPs
        logger.debug("thop profile: %.4f GFLOPs for %s", gflops, tracker_class.__name__)
        return gflops
    except Exception as exc:
        logger.debug(
            "thop profiling failed for %s (%s); using static estimate",
            tracker_class.__name__ if hasattr(tracker_class, "__name__") else tracker_class,
            exc,
        )

    # Fallback to static estimate
    if static_gflops is not None:
        return static_gflops

    # Last resort: return 0.9 (SGLATrack DeiT-tiny baseline)
    logger.warning(
        "Could not determine GFLOPs for %s; returning 0.9 GFLOPs (SGLATrack default)",
        tracker_class,
    )
    return 0.9
