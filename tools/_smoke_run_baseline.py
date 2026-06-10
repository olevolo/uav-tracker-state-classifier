"""Smoke test for run_baseline.py (tracker-agnostic baseline runner).

Calls run_baseline.main() with:
  --tracker sglatrack --dataset got10k --split val --max_sequences 1
  --device cpu --output_dir outputs/_smoke_baseline

PASS criteria:
  1. manifest.json written with expected keys
  2. At least 1 prediction file under predictions/
  3. Prediction file has at least 1 non-empty line
  4. Telemetry file present with at least 1 row

Run:
    perl -e 'alarm 240; exec @ARGV' .venv/bin/python -u tools/_smoke_run_baseline.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

print(f"CUDA=N/A (cpu smoke)", flush=True)

_REPO = Path(__file__).resolve().parent.parent
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import run_baseline  # noqa: E402


def _ts(step: str, idx: int, total: int) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{idx}/{total}] {ts} {step}", flush=True)


def main() -> int:
    N = 4

    _ts("importing run_baseline", 1, N)
    # Already imported above; just confirm the function is callable
    assert callable(run_baseline.main), "run_baseline.main not callable"

    out_dir = str(_REPO / "outputs" / "_smoke_baseline")

    _ts("calling run_baseline.main (sglatrack/got10k/val, 1 sequence, cpu)", 2, N)
    t0 = time.perf_counter()
    ret = run_baseline.main(
        [
            "--tracker", "sglatrack",
            "--dataset", "got10k",
            "--split", "val",
            "--max_sequences", "1",
            "--device", "cpu",
            "--output_dir", out_dir,
        ]
    )
    elapsed = time.perf_counter() - t0
    print(f"  run_baseline returned {ret} in {elapsed:.1f}s", flush=True)
    assert ret == 0, f"run_baseline.main returned {ret}"

    _ts("verifying outputs", 3, N)
    out_root = Path(out_dir) / "got10k" / "val"
    manifest_path = out_root / "manifest.json"
    assert manifest_path.exists(), f"manifest.json not found at {manifest_path}"

    with open(manifest_path) as fh:
        manifest = json.load(fh)

    required_keys = {
        "tracker", "dataset", "split", "device",
        "git_commit", "weights_path", "seed", "datetime",
        "n_sequences", "n_frames", "total_time_s", "mean_fps", "sequences",
    }
    missing_keys = required_keys - set(manifest.keys())
    assert not missing_keys, f"manifest missing keys: {missing_keys}"
    assert manifest["tracker"] == "sglatrack"
    assert manifest["dataset"] == "got10k"
    assert manifest["n_sequences"] >= 1, "expected at least 1 sequence in manifest"
    print(
        f"  manifest OK: {manifest['n_sequences']} seq, "
        f"{manifest['n_frames']} frames, {manifest['mean_fps']:.1f} fps",
        flush=True,
    )

    pred_dir = out_root / "predictions"
    pred_files = list(pred_dir.glob("*.txt"))
    assert len(pred_files) >= 1, f"no prediction .txt files in {pred_dir}"
    first_pred = pred_files[0]
    lines = [l.strip() for l in first_pred.read_text().splitlines() if l.strip()]
    assert len(lines) >= 1, f"prediction file {first_pred.name} is empty"
    print(f"  predictions OK: {len(pred_files)} file(s), first has {len(lines)} lines", flush=True)

    tel_dir = out_root / "telemetry"
    tel_files = list(tel_dir.glob("*.jsonl"))
    assert len(tel_files) >= 1, f"no telemetry .jsonl files in {tel_dir}"
    first_tel_rows = [l for l in tel_files[0].read_text().splitlines() if l.strip()]
    assert len(first_tel_rows) >= 1, "telemetry file is empty"
    # validate first row is parseable JSON with frame_idx
    row0 = json.loads(first_tel_rows[0])
    assert "frame_idx" in row0 or row0.get("init"), f"unexpected telemetry row: {row0}"
    print(f"  telemetry OK: {len(tel_files)} file(s), first has {len(first_tel_rows)} rows", flush=True)

    _ts("PASS", 4, N)
    print("\n===== _smoke_run_baseline PASS =====", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
