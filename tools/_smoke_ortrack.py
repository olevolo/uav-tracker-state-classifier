"""Smoke test for the ORTrack adapter.

Loads the first available LaSOT sequence, runs init + 49 updates with
the ORTrack DeiT-tiny distilled (D-DeiT) tracker on CPU, and asserts
basic telemetry shape and mean stats.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Inject src/ paths so this script works without `pip install -e .`
ROOT = Path(__file__).resolve().parents[1]
for sub in ("src"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import math

from uav_tracker.registry import TRACKERS
from uav_tracker.trackers import ortrack as _ortrack_register  # noqa: F401  triggers registration
from csc_uav_tracking.registry import DATASETS
from csc_uav_tracking.datasets import lasot as _lasot_register  # noqa: F401  triggers registration


def _isfinite(x: float) -> bool:
    return x is not None and math.isfinite(x)


def main() -> int:
    assert "ortrack" in TRACKERS, f"ortrack not in TRACKERS: {TRACKERS.names()}"
    print(f"[smoke] TRACKERS contains ortrack -> OK ({TRACKERS.names()})")

    # Build dataset — try bicycle-1 first, fall back to first available sequence
    ds = DATASETS.build("lasot", categories=["bicycle"], max_frames=60)
    seq = None
    for s in ds:
        if s.name == "bicycle-1":
            seq = s
            break
    if seq is None:
        ds = DATASETS.build("lasot", max_frames=60)
        for s in ds:
            seq = s
            break
    assert seq is not None, "No LaSOT sequence available"
    print(f"[smoke] LaSOT sequence: {seq.name} ({seq.category})")

    # Build tracker on CPU
    tracker = TRACKERS.build("ortrack", device="cpu")
    print(f"[smoke] tracker built: {type(tracker).__name__}")

    # Run init + 49 updates
    frames_iter = iter(seq.frames)
    init_frame = next(frames_iter)
    init_bbox = seq.init_bbox
    tracker.init(init_frame, init_bbox)
    print(f"[smoke] post-init stub={getattr(tracker, 'is_stub_mode', '?')}")

    bboxes = [init_bbox]
    confidences = []
    apces = []
    psrs = []
    occ_scores = []
    fps_samples = []

    n_updates = 49
    for i in range(n_updates):
        frame = next(frames_iter)
        t0 = time.perf_counter()
        st = tracker.update(frame)
        dt = time.perf_counter() - t0
        fps_samples.append(1.0 / max(dt, 1e-9))
        bboxes.append(st.bbox)
        confidences.append(st.confidence)
        apces.append(st.apce)
        psrs.append(st.psr)
        raw = (st.aux or {}).get("raw", {})
        if "occlusion_score" in raw:
            occ_scores.append(raw["occlusion_score"])

    assert len(bboxes) == 50, f"expected 50 bboxes, got {len(bboxes)}"
    for i, c in enumerate(confidences):
        assert _isfinite(c), f"non-finite confidence at update {i}: {c}"
    assert all(_isfinite(a) for a in apces), "non-finite apce"
    assert all(_isfinite(p) for p in psrs), "non-finite psr"

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    print(f"[smoke] mean FPS:        {_mean(fps_samples):.2f}")
    print(f"[smoke] mean confidence: {_mean(confidences):.4f}")
    print(f"[smoke] mean APCE:       {_mean(apces):.2f}")
    print(f"[smoke] mean PSR:        {_mean(psrs):.2f}")
    if occ_scores:
        print(f"[smoke] mean occlusion:  {_mean(occ_scores):.4f}  "
              f"(min {min(occ_scores):.4f}, max {max(occ_scores):.4f})")
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
