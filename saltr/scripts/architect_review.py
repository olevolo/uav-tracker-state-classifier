#!/usr/bin/env python3
"""architect_review.py — automated SALT-RD architect review pass.

Runs every 10 minutes via Claude Code cron job. Checks project health gates
and appends a timestamped review section to codex.md.

Gates checked:
  1. New git commits since last review
  2. Running processes (collection / training / bench)
  3. Candidate artifact validity (seq_id, dist_from_last, frame dims)
  4. Active blockers (BUG-29, Track A, Track B)
"""
from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]  # project root
CODEX = REPO / "codex.md"
SUPER_PLAN = REPO / "SUPER_PLAN.md"
ORACLE_NPZ = REPO / "saltr/results/reinit_oracle_dataset.npz"
TRAIN_POLICY = REPO / "saltr/src/salt_r/train_policy.py"


DATASETS = ["uav123", "dtb70", "visdrone_sot"]


def _candidate_artifacts() -> list[Path]:
    """Return all candidate event artifacts — legacy combined, per-dataset, and smoke."""
    data_dir = REPO / "saltr/data"
    artifacts: list[Path] = []
    # Per-dataset V5+ artifacts (canonical going forward)
    for ds in DATASETS:
        for v in range(5, 10):
            artifacts.append(data_dir / f"candidate_events_v{v}_{ds}.npz")
    # Legacy combined artifacts (pre-per-dataset rule)
    for name in [
        "candidate_events_v5_labeled.npz",
        "candidate_events_v4_labeled.npz",
        "candidate_events_v3_labeled.npz",
        "candidate_events_v2_labeled.npz",
        "candidate_events_labeled.npz",
    ]:
        artifacts.append(data_dir / name)
    # Smoke artifacts
    artifacts.extend(sorted(data_dir.glob("candidate_events_smoke_v*.npz"), reverse=True))
    # Deduplicate preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for path in artifacts:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def _run(cmd: str, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            cmd, shell=True, cwd=str(cwd or REPO),
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def _git_log() -> str:
    return _run("git log --oneline -5")


def _running_processes() -> list[str]:
    checks = [
        ("build_candidate_dataset", "collecting candidates"),
        ("train_candidate_scorer",  "training scorer"),
        ("train_policy",            "training policy"),
        ("fast_bench",              "benchmarking"),
        ("rollout_policy",          "rolling out policy"),
    ]
    active = []
    plist = _run("pgrep -fla python") or _run("ps aux | grep python | grep -v grep")
    for keyword, label in checks:
        if keyword in plist:
            active.append(label)
    return active


def _validate_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        import numpy as np
        data = np.load(str(path), allow_pickle=True)

        # Actual schema: data["events"] is an object array of dicts,
        # data["stats"] is a 1-element object array with aggregate stats.
        if "events" not in data:
            return {"exists": True, "error": "missing 'events' key", "keys": list(data.keys())}

        events = data["events"]
        n = len(events)
        if n == 0:
            return {"exists": True, "n_events": 0, "gates_pass": False}

        seq_nonblank = float(sum(
            1 for e in events if isinstance(e, dict) and e.get("seq_id", "").strip()
        )) / n
        frame_dims_nonzero = float(sum(
            1 for e in events if isinstance(e, dict) and e.get("frame_h", 0) > 0 and e.get("frame_w", 0) > 0
        )) / n
        dist_nonzero = float(sum(
            1 for e in events if isinstance(e, dict) and abs(e.get("dist_from_last", 0.0)) > 1e-6
        )) / n

        # Legacy positive rate from label_good_candidate. This target is known to be
        # too strict for V4 because it requires future_iou_gain > 0 even when
        # candidate_iou is already high. Prefer candidate_correct_iou03 once V5 lands.
        labeled = [e for e in events if isinstance(e, dict) and e.get("label_good_candidate") is not None]
        pos_rate = (
            float(sum(1 for e in labeled if e.get("label_good_candidate", 0) == 1)) / max(len(labeled), 1)
            if labeled else 0.0
        )

        correct03_labeled = [
            e for e in events if isinstance(e, dict) and e.get("candidate_correct_iou03") is not None
        ]
        correct03_rate = (
            float(sum(1 for e in correct03_labeled if e.get("candidate_correct_iou03", 0) == 1))
            / max(len(correct03_labeled), 1)
            if correct03_labeled
            else 0.0
        )

        # Source breakdown
        detector_events = sum(1 for e in events if isinstance(e, dict) and e.get("source") == "detector")
        scoremap_events = n - detector_events

        # dist_from_last non-zero rate among DETECTOR events specifically
        det_events_list = [e for e in events if isinstance(e, dict) and e.get("source") == "detector"]
        det_dist_nz = (
            float(sum(1 for e in det_events_list if abs(e.get("dist_from_last", 0.0)) > 1e-6))
            / max(len(det_events_list), 1)
        ) if det_events_list else 0.0

        mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        return {
            "exists": True,
            "n_events": n,
            "n_detector": detector_events,
            "n_scoremap": scoremap_events,
            "mtime": mtime,
            "seq_nonblank_rate": seq_nonblank,
            "frame_dims_nonzero_rate": frame_dims_nonzero,
            "dist_from_last_nonzero_rate": dist_nonzero,
            "det_dist_nonzero_rate": det_dist_nz,
            "positive_rate": pos_rate,
            "candidate_correct_iou03_rate": correct03_rate,
            "has_candidate_correct_iou03": bool(correct03_labeled),
            # Gates: seq_id, frame_dims always; dist_from_last gated on DETECTOR events
            # (score-map dist is always ~0 by construction — score-map IS at tracker center)
            "gates_pass": seq_nonblank > 0.5 and frame_dims_nonzero > 0.5 and det_dist_nz > 0.3,
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}


def _best_artifact_summary() -> tuple[str, bool]:
    lines = []
    any_pass = False
    for artifact in _candidate_artifacts():
        info = _validate_artifact(artifact)
        name = artifact.name
        if not info["exists"]:
            continue
        if "error" in info:
            lines.append(f"- `{name}` — ERROR: {info['error']}")
            continue
        gate = info.get("gates_pass", False)
        any_pass = any_pass or gate
        flag = "PASS" if gate else "FAIL"
        lines.append(
            f"- `{name}` ({info['mtime']}, n={info['n_events']}, "
            f"det={info.get('n_detector',0)}/sm={info.get('n_scoremap',0)}) — "
            f"seq_nonblank={info['seq_nonblank_rate']:.2f} "
            f"frame_dims={info['frame_dims_nonzero_rate']:.2f} "
            f"det_dist_nz={info.get('det_dist_nonzero_rate',0):.2f} "
            f"legacy_pos={info['positive_rate']:.3f} "
            f"correct_iou03={info.get('candidate_correct_iou03_rate',0):.3f} [{flag}]"
        )
    if not lines:
        lines.append("- No candidate artifact found. Collection required.")
    return "\n".join(lines), any_pass


def _new_commits_since_last_review() -> str:
    # Find the last review timestamp from codex.md
    if not CODEX.exists():
        return _run("git log --oneline -3")
    text = CODEX.read_text(errors="replace")
    # Find the most recent "Review Pass" timestamp
    import re
    matches = re.findall(r"Review Pass.*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2})", text)
    if not matches:
        return _run("git log --oneline -3")
    last_ts = matches[-1]
    # Get commits after that time
    commits = _run(f'git log --oneline --after="{last_ts}" --format="%h %s"')
    return commits or "(no new commits since last review)"


def _assess_blockers() -> list[str]:
    """Return list of active blocker statements based on artifact state."""
    blockers = []
    train_policy_uses_legacy_candidate_label = (
        TRAIN_POLICY.exists()
        and 'ev.get("label_good_candidate"' in TRAIN_POLICY.read_text(errors="replace")
    )

    # BUG-29: check if dist_from_last is wired (artifact gate)
    best_valid = False
    valid_infos = []
    for a in _candidate_artifacts():
        info = _validate_artifact(a)
        if info.get("gates_pass"):
            best_valid = True
            valid_infos.append((a, info))
    if not best_valid:
        blockers.append(
            "BUG-29 ACTIVE: No valid candidate artifact (seq_id+frame_dims+dist_from_last gates). "
            "Re-run smoke collection on car7/truck1/uav7 and validate artifact before scorer training."
        )
    else:
        # Prefer the most informative artifact: V5 correctness labels if present, then
        # newest mtime. V4 smoke only proves plumbing; V5 proves trainable labels.
        valid_infos.sort(
            key=lambda pair: (
                bool(pair[1].get("has_candidate_correct_iou03", False)),
                pair[0].stat().st_mtime if pair[0].exists() else 0.0,
            ),
            reverse=True,
        )
        artifact, info = valid_infos[0]
        blockers.append(f"BUG-29 RESOLVED for artifact plumbing: `{artifact.name}` passes seq/frame/det-distance gates.")
        if not info.get("has_candidate_correct_iou03", False):
            blockers.append(
                "Candidate scorer training still BLOCKED: artifact uses legacy label_good_candidate "
                "(candidate_iou > 0.3 AND future_iou_gain > 0). Patch V5 labels to "
                "candidate_correct_iou03/iou05 before training."
            )
        elif info.get("candidate_correct_iou03_rate", 0.0) <= 0.05:
            blockers.append(
                "Candidate scorer training still BLOCKED: candidate_correct_iou03_rate <= 0.05. "
                "This falsifies current candidate generation on the smoke subset."
            )
        elif train_policy_uses_legacy_candidate_label:
            blockers.append(
                "V5 artifact label gate PASSED, but scorer training still BLOCKED: "
                "train_policy.py reads legacy label_good_candidate instead of candidate_correct_iou03."
            )
        else:
            blockers.append(
                "V5 label gate PASSED: proceed to full V5 collection and feature diagnostics before scorer training."
            )

    # Per-dataset V5 artifact status (permanent rule: train per dataset)
    data_dir = REPO / "saltr/data"
    missing_datasets = []
    for ds in DATASETS:
        ds_artifact = data_dir / f"candidate_events_v5_{ds}.npz"
        if not ds_artifact.exists():
            missing_datasets.append(ds)
    if missing_datasets:
        blockers.append(
            f"PER-DATASET COLLECTION PENDING: missing V5 artifacts for {missing_datasets}. "
            "Run build_candidate_dataset.py --dataset <ds> separately for each dataset."
        )
    else:
        blockers.append("PER-DATASET V5 artifacts: all 3 datasets present (uav123/dtb70/visdrone_sot).")

    # Check if oracle NPZ exists for collection
    if not ORACLE_NPZ.exists():
        blockers.append(
            "Oracle NPZ missing at saltr/results/reinit_oracle_dataset.npz. "
            "Cannot run build_candidate_dataset.py without it."
        )

    return blockers


def _format_review(ts: str) -> str:
    git_log = _git_log()
    new_commits = _new_commits_since_last_review()
    processes = _running_processes()
    artifact_summary, artifact_pass = _best_artifact_summary()
    blockers = _assess_blockers()

    proc_line = ", ".join(processes) if processes else "none running"
    blocker_lines = "\n".join(f"- {b}" for b in blockers)

    return f"""
## Review Pass - {ts}

Files/commits reviewed: git log, artifact mtimes, process list, codex.md.

**Recent commits:**
{new_commits}

**Running processes:** {proc_line}

**Candidate artifact gates (per-dataset — uav123 / dtb70 / visdrone_sot):**
{artifact_summary}

**Active blockers:**
{blocker_lines}

**Architecture decision (permanent):**
- Train and benchmark **per dataset** — never combine uav123+dtb70+visdrone in one artifact.
- Scorer training blocked until per-dataset V5 artifact passes gates AND correct_iou03_rate > 0.10.
- No threshold sweeps. No AUROC-only progress.

**Next 3 concrete actions:**
1. Wait for UAV123 V5 → rename to `candidate_events_v5_uav123.npz`, then run DTB70 + VisDrone collections.
2. Train scorer v2.1 per dataset: `train_candidate_scorer.py --dataset candidate_events_v5_<ds>.npz --output checkpoints/candidate_scorer_v21_<ds>/`
3. Regression gate per dataset (car7/truck1/bike2 for uav123) before full benchmark.

---
"""


def main() -> None:
    tz_offset = "+03:00"  # EEST
    ts = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M EEST")

    review_text = _format_review(ts)

    # Append to codex.md
    if CODEX.exists():
        with CODEX.open("a") as f:
            f.write(review_text)
        print(f"[architect_review] Appended review to {CODEX}", file=sys.stderr)
    else:
        print(f"[architect_review] codex.md not found at {CODEX}", file=sys.stderr)
        print(review_text)

    # Append abbreviated status line to ARCHITECT.md
    architect_md = REPO / "ARCHITECT.md"
    if architect_md.exists():
        _, any_pass = _best_artifact_summary()
        processes = _running_processes()
        proc_line = ", ".join(processes) if processes else "none"
        status = "GATE_PASS" if any_pass else "GATES_FAIL"
        with architect_md.open("a") as f:
            f.write(f"\n<!-- cron {ts} | {status} | procs: {proc_line} -->\n")
        print(f"[architect_review] Updated status in {architect_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
