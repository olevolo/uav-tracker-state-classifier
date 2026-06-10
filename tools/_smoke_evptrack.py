"""EVPTrack adapter smoke — 1 init + 1 track, timestamped, CUDA-aware.

Run with::

    perl -e 'alarm 180; exec @ARGV' .venv/bin/python -u tools/_smoke_evptrack.py

If something hangs the alarm wins after 180 s.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    log(f"[1/8] python={sys.version.split()[0]}  CUDA={torch.cuda.is_available()}  threads={torch.get_num_threads()}")

    _REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_REPO / "src"))

    log("[2/8] importing uav_tracker registry")
    from uav_tracker.registry import TRACKERS
    from uav_tracker.types import BBox

    log("[3/8] importing evptrack adapter (registers tracker)")
    from uav_tracker.trackers import evptrack  # noqa: F401

    log("[4/8] building tracker (cpu) — this loads cfg + builds model")
    t0 = time.perf_counter()
    tracker = TRACKERS.build("evptrack", device="cpu")
    log(f"[4/8] build done in {time.perf_counter() - t0:.2f}s; "
        f"is_stub_mode={getattr(tracker, 'is_stub_mode', None)}")

    # Locate one LaSOT sequence and read frame 0
    log("[5/8] locating a LaSOT sequence")
    lasot = Path(os.environ.get("UAV_DATA_ROOT", str(Path.home() / "uav-tracker-data"))) / "LaSOT"
    if not lasot.is_dir():
        raise FileNotFoundError(f"LaSOT root not found at {lasot}")
    seq_dir = None
    for cat in sorted(lasot.iterdir()):
        if not cat.is_dir():
            continue
        for s in sorted(cat.iterdir()):
            if (s / "img").is_dir() and (s / "groundtruth.txt").exists():
                seq_dir = s
                break
        if seq_dir is not None:
            break
    if seq_dir is None:
        raise FileNotFoundError("no LaSOT sequence with img/ + groundtruth.txt")
    log(f"[5/8] sequence: {seq_dir.name}")

    f0_path = sorted((seq_dir / "img").glob("*.jpg"))[0]
    f1_path = sorted((seq_dir / "img").glob("*.jpg"))[1]
    with open(seq_dir / "groundtruth.txt") as fh:
        first = fh.readline().strip()
    x, y, w, h = (float(v) for v in first.split(","))
    init_bbox = BBox(x=x, y=y, w=w, h=h)

    f0 = cv2.imread(str(f0_path))
    f1 = cv2.imread(str(f1_path))
    if f0 is None or f1 is None:
        raise RuntimeError("failed to read frames")
    log(f"[5/8] frames loaded: shape={f0.shape}  init_bbox={init_bbox}")

    log("[6/8] tracker.init() — first heavy call, may take 30+s on CPU")
    t0 = time.perf_counter()
    with torch.no_grad():
        tracker.init(f0, init_bbox)
    log(f"[6/8] init done in {time.perf_counter() - t0:.2f}s")

    log("[7/8] tracker.update() — single frame")
    t0 = time.perf_counter()
    with torch.no_grad():
        state = tracker.update(f1)
    dt = time.perf_counter() - t0
    log(f"[7/8] update done in {dt:.2f}s ({1.0/dt:.2f} FPS)")

    # Assertions
    log("[8/8] running assertions")
    assert state.bbox is not None and state.bbox.w > 0 and state.bbox.h > 0
    for v in (state.bbox.x, state.bbox.y, state.bbox.w, state.bbox.h):
        assert math.isfinite(v)
    if state.confidence is not None:
        assert math.isfinite(float(state.confidence))
    if state.apce is not None:
        assert math.isfinite(float(state.apce))
    if state.psr is not None:
        assert math.isfinite(float(state.psr))

    print("=" * 64, flush=True)
    print(f"EVPTrack smoke PASS  "
          f"({'DRY-RUN' if getattr(tracker, 'is_stub_mode', False) else 'REAL WEIGHTS'})", flush=True)
    print(f"  bbox       : x={state.bbox.x:.1f} y={state.bbox.y:.1f} "
          f"w={state.bbox.w:.1f} h={state.bbox.h:.1f}", flush=True)
    print(f"  confidence : {state.confidence}", flush=True)
    print(f"  APCE       : {state.apce}", flush=True)
    print(f"  PSR        : {state.psr}", flush=True)
    print(f"  update_ms  : {dt*1000:.1f}", flush=True)
    print("=" * 64, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
