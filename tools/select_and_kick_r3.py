#!/usr/bin/env python3
"""Select R3 variant (R2-style or R25-style) and kick R3_S1 training.

Reads fc_n10_threshold_aware.json (must include r2 + r25 ckpts).
Picks winner by composite score: R@FPR≤3% + P@R50 (both critical for control).
Writes marker file logs/v3fix_full/r3_winner.txt with "fcw4" or "fcw3".
Kicks training using the matching config.

Idempotent — exits if R3_S1 already started (sentinel: logs/v3fix_full/r3_kicked.done).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "v3fix_full"
SENTINEL = LOG / "r3_kicked.done"
WINNER_MARKER = LOG / "r3_winner.txt"

CONFIGS = {
    "r2": {
        "label": "R2-style (V2 + FCw=4.0)",
        "variant": "fcw4",
        "stage1": "configs/csc/csc_tcn32_v2_fcw4_w32_stage1.yaml",
        "stage2": "configs/csc/csc_tcn32_v2_fcw4_w32_stage2.yaml",
        "ckpt_name": "sglatrack_run2_scalectx_tcn16_stage2",
    },
    "r25": {
        "label": "R25-style (V2 + FCw=3.0)",
        "variant": "fcw3",
        "stage1": "configs/csc/csc_tcn32_v2_fcw3_w32_stage1.yaml",
        "stage2": "configs/csc/csc_tcn32_v2_fcw3_w32_stage2.yaml",
        "ckpt_name": "sglatrack_r25_fcw3_tcn16_stage2",
    },
}


def score(entry: dict) -> float:
    """Composite control-mode score: R@FPR≤3% + P@R50."""
    return entry.get("recall_at_fpr_03", 0.0) + entry.get("precision_at_recall_50", 0.0)


def main() -> int:
    if SENTINEL.exists():
        print(f"R3 already kicked: {SENTINEL.read_text().strip()}")
        return 0

    thr_path = LOG / "fc_n10_threshold_aware.json"
    if not thr_path.exists():
        print(f"ERR: {thr_path} missing — run forecast_threshold_metrics first")
        return 1
    entries = json.loads(thr_path.read_text())
    by_name = {Path(e["ckpt"]).parent.name: e for e in entries}

    scores = {}
    for tag, cfg in CONFIGS.items():
        e = by_name.get(cfg["ckpt_name"])
        if e is None:
            print(f"WARN: no entry for {tag} ({cfg['ckpt_name']}) — cannot compare")
            return 2
        s = score(e)
        scores[tag] = (s, e["recall_at_fpr_03"], e["precision_at_recall_50"])
        print(f"  {tag} ({cfg['label']}): score={s:.4f}  R@FPR≤3%={e['recall_at_fpr_03']:.3f}  P@R50={e['precision_at_recall_50']:.3f}")

    winner = max(scores, key=lambda k: scores[k][0])
    print(f"\n→ WINNER: {winner} ({CONFIGS[winner]['label']})")

    WINNER_MARKER.write_text(CONFIGS[winner]["variant"])
    print(f"  marker: {WINNER_MARKER}")

    cfg_path = CONFIGS[winner]["stage1"]
    log_file = LOG / "r3_stage1.log"
    cmd = (
        f"cd {ROOT} && nohup .venv/bin/python tools/train_csc.py "
        f"--config {cfg_path} > {log_file} 2>&1 &"
    )
    print(f"  kicking: {cmd}")
    rc = os.system(cmd)
    if rc == 0:
        SENTINEL.write_text(f"winner={winner} variant={CONFIGS[winner]['variant']}")
        print(f"  ✓ R3_S1 launched ({log_file})")
        return 0
    print(f"  ✗ kickoff failed rc={rc}")
    return 3


if __name__ == "__main__":
    sys.exit(main())
