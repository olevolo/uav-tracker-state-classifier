#!/usr/bin/env python
"""Smoke test for --control_risk_gate.

Runs SGLATrack + CSC (R2) on one EASY sequence (boat2 — known to lose AUC
under ungated control purely from template-update drift, with ZERO CSC
intervention) in three modes and one HARD sequence (uav4) gated, then compares
AUC / template-freezes / gate-closed-fraction.

Expected:
  boat2 passive          : AUC baseline (static frame-0 template)
  boat2 control          : AUC drops (template updates from frame ~101)
  boat2 control+gate     : gate CLOSES on most frames -> AUC ~= passive
  uav4  control+gate     : gate mostly OPEN (persistent risk) -> control acts

Diagnosis-only offline benchmark on UAV123 (final-test set; used here only as a
mechanical smoke of the gate code path, NOT for tuning).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
R2_CKPT = ROOT / "outputs/csc_training/sglatrack_run2_scalectx_tcn16_stage2/checkpoint_best.pth"
CALIB = "sglatrack_all_v2"
OUT = ROOT / "outputs/_smoke_risk_gate"
GT_DIR = Path("~/uav-tracker-data/uav123/UAV123/anno/UAV123").expanduser()


def iou_xywh(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    px, py, pw, ph = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    gx, gy, gw, gh = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    ix1 = np.maximum(px, gx); iy1 = np.maximum(py, gy)
    ix2 = np.minimum(px + pw, gx + gw); iy2 = np.minimum(py + ph, gy + gh)
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = pw * ph + gw * gh - inter
    return np.where(union > 0, inter / union, 0.0)


def success_auc(ious: np.ndarray) -> float:
    th = np.linspace(0, 1, 21)
    return float(np.mean([(ious >= t).mean() for t in th]))


def read_preds(path: Path) -> np.ndarray:
    rows = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if ln:
            rows.append([float(x) for x in ln.split(",")])
    return np.array(rows)


def read_gt(name: str) -> np.ndarray:
    txt = (GT_DIR / f"{name}.txt").read_text()
    rows = [[float(x) for x in ln.replace(",", " ").split()]
            for ln in txt.splitlines() if ln.strip()]
    return np.array(rows)


def run(seq: str, mode: str, tag: str, gate: bool) -> Path:
    cmd = [str(PYTHON), "-u", str(ROOT / "tools/run_with_csc.py"),
           "--tracker", "sglatrack", "--dataset", "uav123", "--split", "test",
           "--csc_checkpoint", str(R2_CKPT), "--csc_mode", mode,
           "--calibration_prefix", CALIB, "--device", "cpu",
           "--output_dir", str(OUT), "--run_tag", tag,
           "--include_sequences", seq]
    if mode == "control":
        cmd += ["--exit_router", "--proactive_v3", "--proactive_threshold", "0.7"]
    if gate:
        cmd += ["--control_risk_gate"]
    print(f"  RUN {tag}", flush=True)
    subprocess.run(cmd, check=True)
    return OUT / tag


def summarize(run_dir: Path, seq: str) -> dict:
    preds = read_preds(run_dir / "predictions" / f"{seq}.txt")
    gt = read_gt(seq)
    n = min(len(preds), len(gt))
    ious = iou_xywh(preds[1:n], gt[1:n])  # skip init frame
    m = json.loads((run_dir / "metrics.json").read_text())
    return {
        "auc": success_auc(ious),
        "freezes": m.get("control_template_update_freezes", 0),
        "gate_closed_frac": m.get("risk_gate_closed_frac", None),
        "n": n,
    }


def main() -> int:
    configs = [
        ("boat2", "passive", "boat2_passive", False),
        ("boat2", "control", "boat2_control", False),
        ("boat2", "control", "boat2_control_gate", True),
        ("uav4", "control", "uav4_control", False),
        ("uav4", "control", "uav4_control_gate", True),
    ]
    results = {}
    for seq, mode, tag, gate in configs:
        d = run(seq, mode, tag, gate)
        results[tag] = (seq, summarize(d, seq))

    print("\n=== RISK-GATE SMOKE ===")
    print(f"{'config':22s} {'AUC':>7s} {'freezes':>8s} {'gate_closed':>12s}")
    for tag, (seq, r) in results.items():
        gc = r["gate_closed_frac"]
        gc_s = f"{gc:.3f}" if gc is not None else "  off"
        print(f"{tag:22s} {r['auc']:7.4f} {r['freezes']:8d} {gc_s:>12s}")

    # Assertions on the expected mechanism (boat2 easy):
    a_p = results["boat2_passive"][1]["auc"]
    a_c = results["boat2_control"][1]["auc"]
    a_g = results["boat2_control_gate"][1]["auc"]
    gc_easy = results["boat2_control_gate"][1]["gate_closed_frac"]
    gc_hard = results["uav4_control_gate"][1]["gate_closed_frac"]
    print("\n--- interpretation ---")
    print(f"boat2 passive AUC      = {a_p:.4f}")
    print(f"boat2 control AUC      = {a_c:.4f}  (delta vs passive = {a_c - a_p:+.4f})")
    print(f"boat2 control+gate AUC = {a_g:.4f}  (delta vs passive = {a_g - a_p:+.4f})")
    print(f"gate-closed frac: easy(boat2)={gc_easy:.3f}  hard(uav4)={gc_hard:.3f}")
    ok_recover = abs(a_g - a_p) <= abs(a_c - a_p) + 1e-9
    ok_asym = gc_easy > gc_hard
    print(f"\nGATE RECOVERS EASY AUC toward passive: {ok_recover}  "
          f"(|gate-passive| {abs(a_g-a_p):.4f} <= |control-passive| {abs(a_c-a_p):.4f})")
    print(f"GATE CLOSES MORE ON EASY THAN HARD:     {ok_asym}  "
          f"({gc_easy:.3f} > {gc_hard:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
