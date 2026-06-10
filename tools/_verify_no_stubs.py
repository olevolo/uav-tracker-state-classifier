"""Verify all 5 trackers load real weights — one subprocess per tracker.

Each tracker adapter injects its own external `papers/code/<tracker>/` lib into
sys.path, and those packages all use the top-level name ``lib``. So **two
different trackers cannot coexist in one Python process**. Verification therefore
runs one subprocess per tracker.

Run::
    perl -e 'alarm 900; exec @ARGV' .venv/bin/python -u tools/_verify_no_stubs.py

Exits 0 only if every tracker subprocess exits 0 with finite, non-zero confidence.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
PY = str(_REPO / ".venv" / "bin" / "python")

_CHILD = r"""
import json, math, os, sys, time
from pathlib import Path
import cv2, torch

repo = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path(os.getcwd())
sys.path.insert(0, str(repo / "src"))

name = sys.argv[1]
__import__(f"uav_tracker.trackers.{name}")
from uav_tracker.registry import TRACKERS
from uav_tracker.types import BBox

lasot = Path(os.environ.get("UAV_DATA_ROOT", str(Path.home() / "uav-tracker-data"))) / "LaSOT"
seq = None
for cat in sorted(lasot.iterdir()):
    if cat.is_dir():
        for s in sorted(cat.iterdir()):
            if (s / "img").is_dir() and (s / "groundtruth.txt").exists():
                seq = s
                break
    if seq is not None:
        break
frames = sorted((seq / "img").glob("*.jpg"))
f0, f1 = cv2.imread(str(frames[0])), cv2.imread(str(frames[1]))
with open(seq / "groundtruth.txt") as fh:
    x, y, w, h = (float(v) for v in fh.readline().split(","))

t0 = time.perf_counter()
tracker = TRACKERS.build(name, device="cpu")
build_t = time.perf_counter() - t0
t0 = time.perf_counter()
with torch.no_grad():
    tracker.init(f0, BBox(x=x, y=y, w=w, h=h))
init_t = time.perf_counter() - t0
t0 = time.perf_counter()
with torch.no_grad():
    state = tracker.update(f1)
upd_t = time.perf_counter() - t0
stub = bool(getattr(tracker, "is_stub_mode", False))
conf = float(state.confidence) if state.confidence is not None else float("nan")
apce = float(state.apce) if state.apce is not None else float("nan")
psr = float(state.psr) if state.psr is not None else float("nan")
ok = (not stub) and math.isfinite(conf) and abs(conf) > 0.0
print(json.dumps({
    "name": name, "ok": ok, "stub": stub,
    "build_s": build_t, "init_s": init_t, "update_s": upd_t,
    "confidence": conf, "apce": apce, "psr": psr,
    "bbox": [state.bbox.x, state.bbox.y, state.bbox.w, state.bbox.h] if state.bbox else None,
}))
"""


def main() -> int:
    print(f"[{time.strftime('%H:%M:%S')}] running 5 subprocesses (one per tracker)…", flush=True)
    rows = []
    for name in ("sglatrack", "ostrack", "ortrack", "avtrack", "evptrack"):
        print(f"[{time.strftime('%H:%M:%S')}] --- {name} subprocess ---", flush=True)
        p = subprocess.run(
            [PY, "-u", "-c", _CHILD, name],
            cwd=str(_REPO),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if p.returncode != 0:
            print(f"  FAILED rc={p.returncode}", flush=True)
            print("  stderr:", p.stderr[-800:], flush=True)
            rows.append({"name": name, "ok": False, "stub": None, "error": p.stderr[-200:]})
            continue
        try:
            j = json.loads(p.stdout.strip().split("\n")[-1])
        except Exception as exc:
            print(f"  parse failure: {exc}\nstdout: {p.stdout[-400:]}", flush=True)
            rows.append({"name": name, "ok": False, "stub": None, "error": str(exc)})
            continue
        rows.append(j)
        print(f"  stub={j['stub']} build={j['build_s']:.2f}s init={j['init_s']:.2f}s "
              f"upd={j['update_s']:.2f}s conf={j['confidence']:.4f} ok={j['ok']}", flush=True)

    print("\n" + "=" * 110, flush=True)
    print(f"{'tracker':12} {'OK':4} {'stub':6} {'build':>8} {'init':>8} {'upd':>8} "
          f"{'conf':>10} {'apce':>10} {'psr':>12}", flush=True)
    print("-" * 110, flush=True)
    for j in rows:
        if "error" in j:
            print(f"{j['name']:12} NO   ?      ----  ----  ----  ERROR: {j['error'][:60]}", flush=True)
            continue
        print(f"{j['name']:12} {('YES' if j['ok'] else 'NO '):4} {str(j['stub']):6} "
              f"{j['build_s']:7.2f}s {j['init_s']:7.2f}s {j['update_s']:7.2f}s "
              f"{j['confidence']:10.4f} {j['apce']:10.2f} {j['psr']:12.2f}", flush=True)
    print("=" * 110, flush=True)

    failed = [r["name"] for r in rows if not r.get("ok", False)]
    if failed:
        print(f"FAIL — not in real-weights mode: {failed}", flush=True)
        return 1
    print("PASS — all 5 trackers loaded real weights", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
