"""Smoke test for run_with_csc.py (tracker-agnostic CSC runner).

Calls run_with_csc.main() with:
  --tracker sglatrack --dataset got10k --split val
  --csc_checkpoint outputs/csc_training/csc_gru_got10k_v2/checkpoint_best.pth
  --csc_mode passive --max_sequences 1 --device cpu
  --output_dir outputs/_smoke_csc

PASS criteria:
  1. states/<seq>.jsonl written with at least 1 row (the init row + ≥1 track row)
  2. metrics.json is valid JSON with n_sequences >= 1 and mean_total_fps > 0

Run:
    perl -e 'alarm 240; exec @ARGV' .venv/bin/python -u tools/_smoke_run_with_csc.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

print("CUDA=N/A (cpu smoke)", flush=True)

_REPO = Path(__file__).resolve().parent.parent
_TOOLS = _REPO / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import run_with_csc  # noqa: E402

# Default checkpoint — can be overridden via env CSC_CHECKPOINT.
# csc_gru_v2 is the newest compatible checkpoint (head_loc / head_conf split).
_DEFAULT_CSC_CKPT = str(
    _REPO / "outputs" / "csc_training" / "csc_gru_v2" / "checkpoint_best.pth"
)


def _ts(step: str, idx: int, total: int) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{idx}/{total}] {ts} {step}", flush=True)


def main() -> int:
    import os
    N = 4

    _ts("importing run_with_csc", 1, N)
    assert callable(run_with_csc.main), "run_with_csc.main not callable"

    csc_ckpt = os.environ.get("CSC_CHECKPOINT", _DEFAULT_CSC_CKPT)
    if not Path(csc_ckpt).exists():
        print(
            f"  SKIP: CSC checkpoint not found at {csc_ckpt}\n"
            f"  Set CSC_CHECKPOINT env var to a valid path, or generate via train_csc.py.\n"
            "  ===== _smoke_run_with_csc SKIP =====",
            flush=True,
        )
        return 0  # treat as skip, not failure, when checkpoint absent

    out_dir = str(_REPO / "outputs" / "_smoke_csc")

    _ts(f"calling run_with_csc.main (sglatrack/got10k/val, 1 seq, cpu, passive)", 2, N)
    t0 = time.perf_counter()
    ret = run_with_csc.main(
        [
            "--tracker", "sglatrack",
            "--dataset", "got10k",
            "--split", "val",
            "--csc_checkpoint", csc_ckpt,
            "--csc_mode", "passive",
            "--max_sequences", "1",
            "--device", "cpu",
            "--output_dir", out_dir,
        ]
    )
    elapsed = time.perf_counter() - t0
    print(f"  run_with_csc returned {ret} in {elapsed:.1f}s", flush=True)
    assert ret == 0, f"run_with_csc.main returned {ret}"

    _ts("verifying outputs", 3, N)
    # Derive the run_tag the same way run_with_csc does
    from pathlib import Path as P
    csc_model_tag = P(csc_ckpt).stem
    run_tag = f"sglatrack_got10k_val_{csc_model_tag}"
    out_root = P(out_dir) / run_tag

    # states/<seq>.jsonl
    states_dir = out_root / "states"
    state_files = list(states_dir.glob("*.jsonl"))
    assert len(state_files) >= 1, f"no states .jsonl files in {states_dir}"
    first_state = state_files[0]
    state_lines = [l for l in first_state.read_text().splitlines() if l.strip()]
    assert len(state_lines) >= 1, f"states file {first_state.name} is empty"
    # Verify that non-init rows have the required keys
    non_init = [json.loads(l) for l in state_lines if not json.loads(l).get("init")]
    if non_init:
        row = non_init[0]
        for key in ("frame_idx", "risk_score", "derived_state", "should_skip_template_update"):
            assert key in row, f"states row missing key {key!r}: {row}"
    print(
        f"  states OK: {len(state_files)} file(s), first has {len(state_lines)} rows "
        f"({len(non_init)} non-init)",
        flush=True,
    )

    # metrics.json
    metrics_path = out_root / "metrics.json"
    assert metrics_path.exists(), f"metrics.json not found at {metrics_path}"
    with open(metrics_path) as fh:
        metrics = json.load(fh)
    assert metrics.get("n_sequences", 0) >= 1, "metrics: n_sequences < 1"
    assert metrics.get("mean_total_fps", 0) > 0, "metrics: mean_total_fps == 0"
    for key in ("mean_tracker_fps", "mean_csc_fps", "mean_total_fps",
                "tracker_latency", "csc_latency", "total_latency"):
        assert key in metrics, f"metrics.json missing key {key!r}"
    print(
        f"  metrics OK: {metrics['n_sequences']} seq, "
        f"tracker={metrics['mean_tracker_fps']:.1f} fps, "
        f"csc={metrics['mean_csc_fps']:.1f} fps, "
        f"total={metrics['mean_total_fps']:.1f} fps",
        flush=True,
    )

    _ts("PASS", 4, N)
    print("\n===== _smoke_run_with_csc PASS =====", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
