"""Smoke test for the OSTrack tracker adapter.

Runs OSTrack on the LaSOT ``bicycle-1`` sequence (or whichever sequence the
LaSOT registry yields first) for 50 frames on CPU and reports throughput +
telemetry summaries. Use ``UAV_WEIGHTS_ROOT`` to override the default weights
location (``~/uav-tracker-weights``).
"""
from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

# Ensure src/ is importable before any project imports.
_REPO = Path(__file__).resolve().parents[1]
for sub in ("src"):
    p = _REPO / sub
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

from uav_tracker.registry import TRACKERS
from uav_tracker.trackers import ostrack as _ostrack_module  # noqa: F401  (registers)
from csc_uav_tracking.registry import DATASETS
import csc_uav_tracking.datasets.lasot  # noqa: F401  (registers "lasot")


N_FRAMES = 50
PREFERRED_SEQ = "bicycle-1"


def _pick_sequence(ds):
    """Return the preferred sequence if present, else the first one."""
    fallback = None
    for seq in ds:
        if seq.name == PREFERRED_SEQ:
            return seq
        if fallback is None:
            fallback = seq
            # If we can't quickly find the preferred sequence we return the
            # first one — but keep iterating (cheap) in case bicycle-1 shows
            # up later, since LaSOT iteration is alphabetical and bicycle is
            # near the start.
    return fallback


def main() -> int:
    print("[ostrack-smoke] building tracker (cpu)…", flush=True)
    tracker = TRACKERS.build("ostrack", device="cpu")

    print("[ostrack-smoke] loading LaSOT…", flush=True)
    ds = DATASETS.build("lasot")
    seq = _pick_sequence(ds)
    if seq is None:
        print("[ostrack-smoke] FAIL — no LaSOT sequences found", flush=True)
        return 1
    print(f"[ostrack-smoke] using sequence: {seq.name} ({len(seq.ground_truth)} gt frames)",
          flush=True)

    init_bbox = seq.init_bbox
    print(f"[ostrack-smoke] init bbox: x={init_bbox.x} y={init_bbox.y} "
          f"w={init_bbox.w} h={init_bbox.h}",
          flush=True)

    bboxes: list = []
    confs: list[float] = []
    apces: list[float] = []
    psrs: list[float] = []
    entropies: list[float] = []
    durations: list[float] = []

    n_seen = 0
    for frame in seq.frames:
        if n_seen == 0:
            tracker.init(frame, init_bbox)
            bboxes.append(init_bbox)
            confs.append(1.0)
            apces.append(float("nan"))
            psrs.append(float("nan"))
            entropies.append(float("nan"))
            durations.append(0.0)
        else:
            t0 = time.perf_counter()
            state = tracker.update(frame)
            dt = time.perf_counter() - t0
            durations.append(dt)
            bboxes.append(state.bbox)
            confs.append(state.confidence)
            apces.append(state.apce)
            psrs.append(state.psr)
            entropies.append(state.response_entropy)
            assert state.confidence is not None and math.isfinite(state.confidence), (
                f"non-finite confidence at frame {n_seen}: {state.confidence!r}"
            )
        n_seen += 1
        if n_seen >= N_FRAMES:
            break

    if n_seen < N_FRAMES:
        print(f"[ostrack-smoke] FAIL — only got {n_seen}/{N_FRAMES} frames", flush=True)
        return 1

    assert len(bboxes) == N_FRAMES, f"expected {N_FRAMES} bboxes, got {len(bboxes)}"

    # Update durations exclude the init step (which has dt=0). Mean FPS over
    # the 49 update calls.
    update_durations = durations[1:]
    mean_update_s = statistics.mean(update_durations)
    mean_fps = 1.0 / mean_update_s if mean_update_s > 0 else float("inf")

    # Telemetry means over update frames (exclude the synthetic init entry).
    update_confs = confs[1:]
    update_apces = [v for v in apces[1:] if math.isfinite(v)]
    update_psrs = [v for v in psrs[1:] if math.isfinite(v)]
    update_entropies = [v for v in entropies[1:] if math.isfinite(v)]

    print(f"[ostrack-smoke] PASS — {N_FRAMES} bboxes, all confidences finite",
          flush=True)
    print(f"[ostrack-smoke] mean FPS:        {mean_fps:.2f}  "
          f"(mean update {mean_update_s * 1000:.1f} ms over {len(update_durations)} calls)",
          flush=True)
    print(f"[ostrack-smoke] mean confidence: {statistics.mean(update_confs):.4f}",
          flush=True)
    print(f"[ostrack-smoke] mean APCE:       {statistics.mean(update_apces):.2f}",
          flush=True)
    print(f"[ostrack-smoke] mean PSR:        {statistics.mean(update_psrs):.2f}",
          flush=True)
    print(f"[ostrack-smoke] mean entropy:    {statistics.mean(update_entropies):.3f}",
          flush=True)
    print(f"[ostrack-smoke] is_stub_mode:    {tracker.is_stub_mode}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
