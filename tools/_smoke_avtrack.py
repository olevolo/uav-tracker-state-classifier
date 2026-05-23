"""Smoke test for AVTrack adapter.

Runs the AVTrack-DeiT tracker for 50 frames on the first available UAV123
sequence (LaSOT bicycle-1 not present locally). Verifies:
  - Build / load succeeds.
  - Init + 49 update calls return finite bboxes.
  - With weights: ``confidence > 0`` (sanity).
  - Without weights: shapes/finiteness only (dry-run).
  - Reports mean FPS, confidence, APCE, PSR, active_layers.

Run:
    .venv/bin/python tools/_smoke_avtrack.py
"""
from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add the salrtd src dir so ``import uav_tracker`` works without install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SALRTD_SRC = _REPO_ROOT / "salrtd" / "src"
if str(_SALRTD_SRC) not in sys.path:
    sys.path.insert(0, str(_SALRTD_SRC))


# ---------------------------------------------------------------------------
# Dataset discovery — try LaSOT bicycle-1 first, then UAV123 first sequence,
# else synthesise a tiny RGB sequence so the smoke test is self-contained.
# ---------------------------------------------------------------------------

def _try_lasot_bicycle1() -> tuple[list[np.ndarray], tuple[float, float, float, float]] | None:
    """Look for a LaSOT bicycle-1 sequence under common roots."""
    candidates = [
        Path("/Users/voleksiuk/uav-tracker-data/LaSOT/bicycle/bicycle-1"),
        Path("/Users/voleksiuk/uav-tracker-data/lasot/bicycle/bicycle-1"),
        Path("/Users/voleksiuk/uav-tracker-data/LaSOTBenchmark/bicycle/bicycle-1"),
    ]
    for root in candidates:
        if not root.exists():
            continue
        img_dir = root / "img"
        gt = root / "groundtruth.txt"
        if not img_dir.is_dir() or not gt.exists():
            continue
        files = sorted(img_dir.glob("*.jpg"))[:50]
        if len(files) < 50:
            continue
        with open(gt) as fh:
            first = fh.readline().strip()
        parts = first.replace(",", " ").split()
        bbox = tuple(float(p) for p in parts[:4])
        frames = [cv2.imread(str(f)) for f in files]
        if any(f is None for f in frames):
            continue
        print(f"[dataset] LaSOT bicycle-1 found at {root}")
        return frames, bbox  # type: ignore[return-value]
    return None


def _try_uav123_first() -> tuple[list[np.ndarray], tuple[float, float, float, float], str] | None:
    """Pick the lexically-first UAV123 sequence and return its first 50 frames."""
    data_seq_root = Path("/Users/voleksiuk/uav-tracker-data/uav123/UAV123/data_seq/UAV123")
    anno_root = Path("/Users/voleksiuk/uav-tracker-data/uav123/UAV123/anno/UAV123")
    if not data_seq_root.is_dir() or not anno_root.is_dir():
        return None
    seq_dirs = sorted(p for p in data_seq_root.iterdir() if p.is_dir())
    for seq_dir in seq_dirs:
        anno = anno_root / f"{seq_dir.name}.txt"
        if not anno.exists():
            continue
        files = sorted(seq_dir.glob("*.jpg"))[:50]
        if len(files) < 50:
            continue
        with open(anno) as fh:
            first = fh.readline().strip()
        parts = first.replace(",", " ").split()
        try:
            bbox = tuple(float(p) for p in parts[:4])
        except ValueError:
            continue
        if any(math.isnan(v) for v in bbox):
            continue
        frames = [cv2.imread(str(f)) for f in files]
        if any(f is None for f in frames):
            continue
        print(f"[dataset] UAV123 first sequence: {seq_dir.name}")
        return frames, bbox, seq_dir.name  # type: ignore[return-value]
    return None


def _synthetic_sequence() -> tuple[list[np.ndarray], tuple[float, float, float, float], str]:
    """Last-resort fallback: 50-frame moving disk on a textured background."""
    rng = np.random.default_rng(0)
    H, W = 360, 640
    bg = (rng.normal(128, 32, (H, W, 3)).clip(0, 255)).astype(np.uint8)
    frames: list[np.ndarray] = []
    cx0, cy0 = 200.0, 180.0
    R = 20
    for k in range(50):
        f = bg.copy()
        cv2.circle(f, (int(cx0 + 1.5 * k), int(cy0 + 0.6 * k)), R, (220, 30, 30), -1)
        frames.append(f)
    bbox = (cx0 - R, cy0 - R, 2 * R, 2 * R)
    return frames, bbox, "synthetic-disk"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Side-effect import: registers ``avtrack`` in TRACKERS.
    from uav_tracker.trackers import avtrack  # noqa: F401
    from uav_tracker.registry import TRACKERS
    from uav_tracker.types import BBox

    assert "avtrack" in TRACKERS, (
        f"avtrack not registered. Known: {TRACKERS.names()}"
    )

    # --- Dataset (try LaSOT, then UAV123, else synthetic) ---
    seq_name = "LaSOT/bicycle-1"
    lasot = _try_lasot_bicycle1()
    if lasot is not None:
        frames, bbox = lasot
    else:
        uav = _try_uav123_first()
        if uav is not None:
            frames, bbox, seq_name = uav
            seq_name = f"UAV123/{seq_name}"
        else:
            frames, bbox, seq_name = _synthetic_sequence()
            seq_name = f"synthetic/{seq_name}"

    assert len(frames) == 50, f"need 50 frames, got {len(frames)}"
    print(f"[smoke] sequence={seq_name}  init_bbox={bbox}")

    # --- Build tracker ---
    tracker = TRACKERS.build("avtrack", device="cpu")
    print(f"[smoke] tracker built: {type(tracker).__name__}")

    # --- Init on frame 0 ---
    init_bbox = BBox(*bbox)
    t0 = time.perf_counter()
    tracker.init(frames[0], init_bbox)

    is_stub = bool(getattr(tracker, "is_stub_mode", False))
    if is_stub:
        print("[smoke] MISSING_WEIGHTS — dry-run mode")

    # --- Run 49 updates ---
    bboxes: list[BBox] = [init_bbox]
    confs: list[float] = [1.0]
    apces: list[float] = [0.0]
    psrs: list[float] = [0.0]
    entropies: list[float] = [0.0]
    active_layers_list: list[int] = []
    token_keep_list: list[float] = []

    for k in range(1, 50):
        st = tracker.update(frames[k])
        bboxes.append(st.bbox)
        confs.append(float(st.confidence))
        apces.append(float(st.apce))
        psrs.append(float(st.psr))
        entropies.append(float(st.response_entropy))

        raw = (st.aux or {}).get("raw", {})
        if "active_layers" in raw:
            active_layers_list.append(int(raw["active_layers"]))
        if "token_keep_ratio" in raw:
            token_keep_list.append(float(raw["token_keep_ratio"]))
    elapsed = time.perf_counter() - t0
    fps = len(frames) / max(elapsed, 1e-6)

    # --- Assertions: shape & finiteness ---
    assert len(bboxes) == 50, f"bboxes len {len(bboxes)}"
    for i, b in enumerate(bboxes):
        for v, name in ((b.x, "x"), (b.y, "y"), (b.w, "w"), (b.h, "h")):
            assert math.isfinite(v), f"bbox[{i}].{name} not finite ({v})"
        assert b.w > 0 and b.h > 0, f"bbox[{i}] non-positive size ({b.w}, {b.h})"

    for i, c in enumerate(confs):
        assert math.isfinite(c), f"conf[{i}] not finite ({c})"
        assert 0.0 <= c <= 1.0, f"conf[{i}] out of [0,1]: {c}"

    for i, a in enumerate(apces):
        assert math.isfinite(a), f"apce[{i}] not finite"
    for i, p in enumerate(psrs):
        assert math.isfinite(p), f"psr[{i}] not finite"

    if not is_stub:
        # With real weights, we expect at least one frame's confidence > 0.
        assert any(c > 0 for c in confs[1:]), \
            f"all confidences zero with real weights — confs={confs[:5]}..."

    # We expect a healthy adaptive layer signal (10 gated blocks plus 2 always-on).
    assert active_layers_list, "no active_layers telemetry produced"
    assert all(2 <= n <= 12 for n in active_layers_list), \
        f"active_layers out of [2,12]: {active_layers_list[:5]}"
    assert token_keep_list and all(0.0 <= r <= 1.0 for r in token_keep_list)

    # --- Report ---
    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    print()
    print("=" * 60)
    print(f" AVTrack smoke — sequence={seq_name}, frames=50")
    print("-" * 60)
    print(f"  mode          : {'DRY-RUN (random init)' if is_stub else 'WEIGHTED'}")
    print(f"  mean FPS      : {fps:7.2f}")
    print(f"  mean conf     : {_mean(confs[1:]):7.4f}")
    print(f"  mean APCE     : {_mean(apces[1:]):7.2f}")
    print(f"  mean PSR      : {_mean(psrs[1:]):7.2f}")
    print(f"  mean entropy  : {_mean(entropies[1:]):7.4f}")
    print(f"  mean active L : {_mean([float(n) for n in active_layers_list]):7.2f}"
          f" (range {min(active_layers_list)}..{max(active_layers_list)} of 12)")
    print(f"  token keep    : {_mean(token_keep_list):7.4f}")
    print("=" * 60)
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
