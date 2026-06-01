#!/usr/bin/env python3
"""Consolidate the full LIVE matrix into ONE results document.

Reads, per (tracker, dataset, calibration-variant) cell under outputs/eval/:
  metrics.json                               (run_with_csc: fps/overhead)
  tracking_metrics/metrics_summary.json      (AUC, Precision@20, runtime)
  paper_metrics/QUALITY_REPORT.md            (FCR, FCD, TTFC, Recovery@30, UUR)
  paper_metrics/state_conditioned_auc.csv    (SC-AUC per state)
  paper_metrics/confusion_3state_<ds>.csv    (FC recall/precision, 3-/4-state mF1)

Writes RESULTS.md:
  §1 Passive diagnosis — 4 trackers x 4 datasets (uniform aerial_v2)
  §2 Calibration both-ways — sglatrack aerial_v2 vs all_v2 x 4 datasets
  §3 Control mode — sglatrack x 4 datasets (passive vs control)  [if present]
Measured numbers only; missing cells render as "—" (never fabricated).
"""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "outputs/eval"
TRACKERS = ["sglatrack", "avtrack", "ortrack", "ostrack"]
DATASETS = ["uav123", "uav123_10fps", "uavtrack112", "dtb70"]
DS_NOTE = {
    "uav123": "final-test · CLEAN",
    "uav123_10fps": "final-test · CLEAN",
    "uavtrack112": "held-out · CLEAN (aerial)",
    "dtb70": "IN-SAMPLE / circular",
}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def f(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def read_tracking(run: Path) -> dict:
    p = run / "tracking_metrics" / "metrics_summary.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    macro = d.get("macro", {})
    rt = d.get("runtime", {}) or {}
    out = {"auc": macro.get("auc"), "pr20": macro.get("precision_20"),
           "n_seq": d.get("n_sequences"), "n_frames": d.get("n_frames")}
    # FPS from runtime mean latency if available
    for k in ("mean_ms", "mean_latency_ms", "p50_ms", "median_ms"):
        v = rt.get(k)
        if v:
            out["fps"] = 1000.0 / v
            break
    return out


def read_runfps(run: Path) -> float | None:
    p = run / "metrics.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    for k in ("total_fps", "fps_total", "fps", "tracker_fps"):
        if isinstance(d.get(k), (int, float)):
            return float(d[k])
    return None


_QR = {
    "fcr": r"FCR.*?\|\s*([0-9.]+)",
    "fcd": r"FCD.*?\|\s*([0-9.]+)",
    "ttfc": r"TTFC.*?\|\s*([0-9.]+)",
    "rec30": r"Recovery@30\s*\|\s*([0-9.]+)",
    "uur": r"UUR.*?\|\s*([0-9.]+)",
}


def read_quality(run: Path) -> dict:
    p = run / "paper_metrics" / "QUALITY_REPORT.md"
    if not p.exists():
        return {}
    t = p.read_text()
    return {k: _num(m.group(1)) for k, rx in _QR.items() if (m := re.search(rx, t))}


def read_scauc(run: Path) -> dict:
    p = run / "paper_metrics" / "state_conditioned_auc.csv"
    if not p.exists():
        return {}
    out = {}
    for row in csv.DictReader(p.open()):
        out[row["state"]] = _num(row.get("mean_auc"))
    return {"scauc_fc": out.get("FALSE_CONFIRMED"), "scauc_cc": out.get("CORRECT_CONFIRMED")}


def read_confusion3(run: Path, ds: str) -> dict:
    p = run / "paper_metrics" / f"confusion_3state_{ds}.csv"
    if not p.exists():
        return {}
    rows, foot = [], ""
    for line in p.read_text().splitlines():
        if line.startswith("#"):
            foot = line
        elif line.strip():
            rows.append(line)
    out = {}
    try:
        rd = list(csv.DictReader(rows))
        col_fc = sum(_num(r["pred_FC"]) or 0 for r in rd)
        for r in rd:
            if r["gt_state"] == "FC":
                out["fc_recall"] = _num(r["recall"])
                tp = _num(r["pred_FC"]) or 0
                out["fc_prec"] = (tp / col_fc) if col_fc else None
    except Exception:
        pass
    if (m := re.search(r"3state_macro_f1=([0-9.]+)", foot)):
        out["mf1_3"] = _num(m.group(1))
    if (m := re.search(r"4state_macro_f1=([0-9.]+)", foot)):
        out["mf1_4"] = _num(m.group(1))
    return out


def gather(t: str, ds: str, run_tag: str) -> dict | None:
    run = EVAL / t / ds / "test" / run_tag
    if not (run / "states").exists() and not (run / "paper_metrics").exists():
        return None
    d = {"tracker": t, "dataset": ds, "run_tag": run_tag}
    d.update(read_tracking(run))
    if d.get("fps") is None and (rf := read_runfps(run)) is not None:
        d["fps"] = rf
    d.update(read_quality(run))
    d.update(read_scauc(run))
    d.update(read_confusion3(run, ds))
    return d


def row(d: dict) -> str:
    return ("| " + " | ".join([
        d["tracker"], f(d.get("auc")), f(d.get("pr20")), f(d.get("fps"), 1),
        f(d.get("fcr")), f(d.get("fcd"), 2), f(d.get("rec30")),
        f(d.get("scauc_fc")), f(d.get("mf1_3")), f(d.get("fc_recall")), f(d.get("fc_prec")),
    ]) + " |")

HDR = ("| Tracker | AUC | Pr@20 | FPS | FCR | FCD | Rec@30 | SC-AUC(FC) | 3-mF1 | FC-rec | FC-prec |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|")


def main() -> int:
    out = ["# CSC V3-Fix — Consolidated Results (measured)",
           "",
           "Model: **R3-fcw3** (`sglatrack_r3_fcw3_w32_tcn32_stage2`, v2/16-dim, window=32, FCw=3.0) — "
           "validation-selected, reused unchanged on every set. Passive diagnosis (CSC observes only).",
           "States: CC=CORRECT_CONFIRMED, CU=CORRECT_UNCERTAIN, LA=LOST_AWARE, FC=FALSE_CONFIRMED. "
           "3-mF1 collapses CU→CC (LA/FC F1 invariant).",
           "",
           "Calibration: **uniform `*_aerial_v2`** (GOT10k+DTB70+VisDrone, n≈69k, identical recipe per tracker). "
           "Trackers dropped: FARTrack, EVPTrack, UETrack (integration-blocked).",
           "",
           "## 1. Passive diagnosis — 4 trackers × 4 datasets",
           ""]
    for ds in DATASETS:
        out += [f"### {ds}  ·  _{DS_NOTE[ds]}_", "", HDR]
        for t in TRACKERS:
            d = gather(t, ds, f"{t}_r3_passive")
            out.append(row(d) if d else f"| {t} | — | — | — | — | — | — | — | — | — | — |")
        out.append("")

    out += ["## 2. Calibration both-ways — sglatrack: aerial_v2 vs all_v2", "",
            "Same LIVE tracker output; only the per-tracker confidence calibrator differs. "
            "`all_v2` (n=754k) additionally includes LaSOT/UAVDT/UAVTrack112 → contaminates UAVTrack112 as held-out.",
            "",
            "| Dataset | Calib | FCR | FCD | FC-rec | FC-prec | SC-AUC(FC) | 3-mF1 |",
            "|---|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        for tag, lab in [(f"sglatrack_r3_passive", "aerial_v2"), ("sglatrack_r3_all_v2", "all_v2")]:
            d = gather("sglatrack", ds, tag) or {}
            out.append("| " + " | ".join([
                ds, lab, f(d.get("fcr")), f(d.get("fcd"), 2), f(d.get("fc_recall")),
                f(d.get("fc_prec")), f(d.get("scauc_fc")), f(d.get("mf1_3"))]) + " |")
    out.append("")

    # §3 control (passive vs control) — only if control runs exist
    out += ["## 3. Control mode — sglatrack × 4 datasets (passive vs control)", ""]
    ctrl_rows = []
    for ds in DATASETS:
        passive = gather("sglatrack", ds, "sglatrack_r3_passive") or {}
        ctrl = None
        for ctag in (f"sglatrack_r3_control", "control_v3fix_tcn16", "control_v3_proactive"):
            ctrl = gather("sglatrack", ds, ctag)
            if ctrl:
                break
        if ctrl:
            ctrl_rows.append((ds, passive, ctrl))
    if ctrl_rows:
        out += ["| Dataset | Mode | AUC | Pr@20 | FCR | FCD | Rec@30 |", "|---|---|---|---|---|---|---|"]
        for ds, p, c in ctrl_rows:
            for lab, d in [("passive", p), ("control", c)]:
                out.append("| " + " | ".join([ds, lab, f(d.get("auc")), f(d.get("pr20")),
                            f(d.get("fcr")), f(d.get("fcd"), 2), f(d.get("rec30"))]) + " |")
    else:
        out.append("_Control-mode runs pending — table will populate once `csc_mode=control` cells complete._")
    out.append("")

    dest = ROOT / "RESULTS.md"
    dest.write_text("\n".join(out) + "\n")
    print(f"wrote {dest} ({len(out)} lines)")
    # quick coverage summary to stdout
    done = sum(1 for t in TRACKERS for ds in DATASETS if gather(t, ds, f"{t}_r3_passive"))
    print(f"passive cells with data: {done}/{len(TRACKERS)*len(DATASETS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
