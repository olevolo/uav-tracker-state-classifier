#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_post_v2_eval.sh — Post-CSC-v2-training full evaluation orchestrator
#
# Purpose:
#   Wait for CSC v2 training (sglatrack_v2_tcn16) to complete, then run
#   passive diagnosis on 5 trackers and SGLATrack control variants A/B/C/AC
#   on UAV123 test set, and assemble FINAL_RESULTS_V2.md.
#
# Usage:
#   nohup bash tools/run_post_v2_eval.sh > logs/post_v2_eval.log 2>&1 &
#
# Safety gate (CLAUDE.md §Research Constraints):
#   CSC_NOT_TRAINED_ON_UAV123=1 is exported internally throughout.
#   The v2 checkpoint was trained on LaSOT + GOT-10k + UAVDT + VisDrone only.
#
# Trackers: sglatrack  ortrack  ostrack  avtrack  evptrack
#
# Outputs:
#   outputs/eval_v2/<tracker>/uav123/test/    — passive eval per tracker
#   outputs/eval_v2/sglatrack/uav123/test/    — control variants A/B/C/AC
#   outputs/results_v2/training_summary.json  — best-epoch metrics
#   outputs/results_v2/FINAL_RESULTS_V2.md   — consolidated paper table
#   logs/v2_eval_<tracker>.log               — per-tracker passive logs
#   logs/v2_control_A.log                    — SGLATrack control-A log
#   logs/v2_control_B.log                    — SGLATrack control-B log
#   logs/v2_control_C.log                    — SGLATrack control-C log
#   logs/v2_control_AC.log                   — SGLATrack control-AC log
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

V2_CKPT="${PROJECT_ROOT}/outputs/csc_training/sglatrack_v2_tcn16/checkpoint_best.pth"
V2_LOG="${PROJECT_ROOT}/logs/sglatrack_v2_pipeline.log"
RESULTS_DIR="${PROJECT_ROOT}/outputs/results_v2"

# All 5 trackers for passive eval
TRACKERS=(sglatrack ortrack ostrack avtrack evptrack)

# Safety: never train on UAV123 — required by run_uav123_final_eval.sh
export CSC_NOT_TRAINED_ON_UAV123=1

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TS()   { date '+%H:%M:%S'; }
log()  { echo "[$(TS)] $*"; }
warn() { echo "[$(TS)] WARN: $*" >&2; }
fail() { echo "[$(TS)] FATAL: $*" >&2; exit 1; }

# Count prediction files in a directory (only .txt files)
count_preds() {
    local dir="$1"
    ls "${dir}/"*.txt 2>/dev/null | wc -l | tr -d ' '
}

# ---------------------------------------------------------------------------
# Step 0: Create required directories
# ---------------------------------------------------------------------------
mkdir -p "${RESULTS_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"

log "=== run_post_v2_eval.sh START ==="
log "Project root : ${PROJECT_ROOT}"
log "V2 checkpoint: ${V2_CKPT}"
log "Results dir  : ${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# Step 1: Wait for CSC v2 training to complete
# ---------------------------------------------------------------------------
log ""
log "=== Step 1: Wait for CSC v2 training ==="

TRAINING_DONE=0

# If checkpoint already exists AND the done marker is in the log, skip waiting
if [[ -f "${V2_CKPT}" ]]; then
    if grep -q "=== CSC v2 DONE for sglatrack ===" "${V2_LOG}" 2>/dev/null; then
        log "  Training already completed. Checkpoint: ${V2_CKPT}"
        TRAINING_DONE=1
    fi
fi

if [[ "${TRAINING_DONE}" -eq 0 ]]; then
    log "  Polling ${V2_LOG} every 60s for '=== CSC v2 DONE for sglatrack ==='"
    log "  Also watching for checkpoint: ${V2_CKPT}"

    while true; do
        # Check for explicit done marker in log
        if [[ -f "${V2_LOG}" ]] && grep -q "=== CSC v2 DONE for sglatrack ===" "${V2_LOG}" 2>/dev/null; then
            log "  Done marker found in log."
            TRAINING_DONE=1
            break
        fi

        # Fallback: if the checkpoint exists and training process is not running,
        # and at least one epoch has been logged, consider training done.
        # This handles the case where the log file format differs.
        if [[ -f "${V2_CKPT}" ]]; then
            # Check if PID 84509 (or any successor) is still running
            if ! kill -0 84509 2>/dev/null; then
                log "  Checkpoint exists and PID 84509 not running — assuming training complete."
                TRAINING_DONE=1
                break
            fi
        fi

        sleep 60
        log "  Still waiting... (checkpoint exists: $([[ -f "${V2_CKPT}" ]] && echo yes || echo no))"
    done
fi

# Assert checkpoint exists before proceeding
if [[ ! -f "${V2_CKPT}" ]]; then
    fail "Checkpoint not found after training completed: ${V2_CKPT}"
fi
log "  Checkpoint verified: ${V2_CKPT}"

# ---------------------------------------------------------------------------
# Step 2: Extract + log training metrics
# ---------------------------------------------------------------------------
log ""
log "=== Step 2: Extract training metrics ==="

TRAINING_SUMMARY="${RESULTS_DIR}/training_summary.json"

if [[ -f "${TRAINING_SUMMARY}" ]]; then
    log "  SKIP: ${TRAINING_SUMMARY} already exists (delete to rerun)"
else
    TRAIN_LOG_JSONL="${PROJECT_ROOT}/outputs/csc_training/sglatrack_v2_tcn16/train_log.jsonl"

    "$PYTHON" - <<'PYEOF'
import json
import os
import sys
from pathlib import Path

project_root = Path(os.environ.get("PROJECT_ROOT", "."))
train_log = project_root / "outputs/csc_training/sglatrack_v2_tcn16/train_log.jsonl"
v2_log = project_root / "logs/sglatrack_v2_pipeline.log"
out_file = project_root / "outputs/results_v2/training_summary.json"

summary = {
    "checkpoint": "outputs/csc_training/sglatrack_v2_tcn16/checkpoint_best.pth",
    "best_epoch": None,
    "best_derivedF1": None,
    "best_FC_recall": None,
    "best_AUROC": None,
    "best_AUPRC": None,
    "total_epochs": None,
    "source": "train_log.jsonl",
}

epochs = []
if train_log.exists():
    for line in train_log.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # Handle truncated lines (rare corruption)
        if not line.startswith("{"):
            line = "{" + line if "epoch" in line else None
            if line is None:
                continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        epochs.append(row)

if epochs:
    summary["total_epochs"] = len(epochs)
    # Selection score = 0.5 * derivedF1 + 0.3 * FC_recall + 0.2 * AUROC
    # (matches train_csc.py logic)
    def score(r):
        f1 = r.get("val_derived_f1", 0.0)
        fc = r.get("val_fc_recall", 0.0)
        auc = r.get("val_failure_auroc", 0.0)
        return 0.5 * f1 + 0.3 * fc + 0.2 * auc

    best = max(epochs, key=score)
    summary["best_epoch"] = best.get("epoch")
    summary["best_derivedF1"] = round(best.get("val_derived_f1", 0.0), 4)
    summary["best_FC_recall"] = round(best.get("val_fc_recall", 0.0), 4)
    summary["best_AUROC"] = round(best.get("val_failure_auroc", 0.0), 4)
    summary["best_AUPRC"] = round(best.get("val_failure_auprc", 0.0), 4)
    summary["best_locF1"] = round(best.get("val_loc_f1", 0.0), 4)
    summary["best_confF1"] = round(best.get("val_conf_f1", 0.0), 4)
    summary["best_selection_score"] = round(score(best), 4)
    summary["all_epochs"] = [
        {
            "epoch": r.get("epoch"),
            "derivedF1": round(r.get("val_derived_f1", 0.0), 4),
            "FC_recall": round(r.get("val_fc_recall", 0.0), 4),
            "AUROC": round(r.get("val_failure_auroc", 0.0), 4),
            "AUPRC": round(r.get("val_failure_auprc", 0.0), 4),
            "selection_score": round(score(r), 4),
        }
        for r in epochs
    ]
else:
    # Fallback: parse from pipeline log
    summary["source"] = "sglatrack_v2_pipeline.log (fallback)"
    if v2_log.exists():
        import re
        pattern = re.compile(
            r"ep\s+(\d+)/\d+\s+\|.*?derivedF1=([\d.]+).*?AUROC=([\d.]+).*?AUPRC=([\d.]+).*?FC_recall=([\d.]+)"
        )
        best_score = -1.0
        for line in v2_log.read_text().splitlines():
            m = pattern.search(line)
            if not m:
                continue
            ep = int(m.group(1))
            f1 = float(m.group(2))
            auc = float(m.group(3))
            auprc = float(m.group(4))
            fc = float(m.group(5))
            s = 0.5 * f1 + 0.3 * fc + 0.2 * auc
            if s > best_score:
                best_score = s
                summary["best_epoch"] = ep
                summary["best_derivedF1"] = round(f1, 4)
                summary["best_FC_recall"] = round(fc, 4)
                summary["best_AUROC"] = round(auc, 4)
                summary["best_AUPRC"] = round(auprc, 4)
                summary["best_selection_score"] = round(s, 4)

out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PYEOF
    log "  Training summary written to ${TRAINING_SUMMARY}"
fi

# ---------------------------------------------------------------------------
# Step 3: Diagnosis gates — check if training met quality targets
# ---------------------------------------------------------------------------
log ""
log "=== Step 3: Diagnosis gates ==="

GATES_PASSED=1
GATE_STATUS=0

if [[ -f "${TRAINING_SUMMARY}" ]]; then
    "$PYTHON" - <<'PYEOF' || GATE_STATUS=$?
import json, sys, os
from pathlib import Path

project_root = Path(os.environ.get("PROJECT_ROOT", "."))
summary_path = project_root / "outputs/results_v2/training_summary.json"

with open(summary_path) as f:
    s = json.load(f)

passed = True

f1 = s.get("best_derivedF1") or 0.0
fc = s.get("best_FC_recall") or 0.0
auc = s.get("best_AUROC") or 0.0

print(f"  Best epoch    : {s.get('best_epoch')} / {s.get('total_epochs')}")
print(f"  derivedF1     : {f1:.4f}  (target >= 0.60)")
print(f"  FC_recall     : {fc:.4f}  (target >= 0.50)")
print(f"  AUROC         : {auc:.4f}  (target >= 0.92)")
print(f"  selection_score: {s.get('best_selection_score', '?')}")

if f1 < 0.60:
    print(f"  WARN: F1 below target ({f1:.4f} < 0.60) — check label distribution", flush=True)
    passed = False
if fc < 0.50:
    print(f"  WARN: FC_recall low ({fc:.4f} < 0.50) — check aerial hard negative count", flush=True)
    passed = False
if auc < 0.92:
    print(f"  WARN: AUROC low ({auc:.4f} < 0.92) — check calibration consistency", flush=True)
    passed = False

if passed:
    print("  All gates PASSED.")
else:
    print("  Some gates WARN — audit recommended.")
    sys.exit(1)  # triggers label audit below
PYEOF
else
    warn "Training summary missing — skipping gate checks"
    GATE_STATUS=0
fi

if [[ "${GATE_STATUS}" -ne 0 ]]; then
    GATES_PASSED=0
    warn "Running label distribution audit..."
    "$PYTHON" "${PROJECT_ROOT}/tools/audit_label_distribution.py" \
        --labels_dir "${PROJECT_ROOT}/outputs/csc_labels/sglatrack/v2_combined" \
        --output "${RESULTS_DIR}/label_audit.json" 2>/dev/null || true
    log "  Label audit written to ${RESULTS_DIR}/label_audit.json (if tool succeeded)"
fi

if [[ "${GATES_PASSED}" -eq 1 ]]; then
    log "  All diagnosis gates PASSED — proceeding."
else
    log "  Diagnosis gates WARNED — continuing (audit logged, not aborting)."
fi

# ---------------------------------------------------------------------------
# Step 4: Wait for all 5 tracker UAV123 baselines (123 predictions each)
# ---------------------------------------------------------------------------
log ""
log "=== Step 4: Wait for all tracker UAV123 baselines ==="

# bash 3.x compat: derive path from tracker name (no associative arrays)
baseline_dir() { echo "${PROJECT_ROOT}/outputs/baselines/${1}/uav123/test/predictions"; }

all_baselines_ready() {
    local all_ok=1
    for tracker in "${TRACKERS[@]}"; do
        local dir; dir="$(baseline_dir "$tracker")"
        local n; n=$(count_preds "$dir")
        if [[ "$n" -lt 123 ]]; then
            all_ok=0
            log "  Waiting: ${tracker} has ${n}/123 predictions in ${dir}"
        fi
    done
    echo "$all_ok"
}

# Poll until all baselines are ready
while true; do
    if [[ "$(all_baselines_ready)" -eq 1 ]]; then
        log "  All 5 tracker baselines have 123 prediction files."
        break
    fi
    log "  Not all baselines ready — sleeping 60s..."
    sleep 60
done

# Print final counts
for tracker in "${TRACKERS[@]}"; do
    n=$(count_preds "$(baseline_dir "$tracker")")
    log "  ${tracker}: ${n} predictions"
done

# ---------------------------------------------------------------------------
# Step 5: Run passive eval for all 5 trackers with v2 CSC
# ---------------------------------------------------------------------------
log ""
log "=== Step 5: Passive eval — all 5 trackers with v2 CSC ==="

for tracker in "${TRACKERS[@]}"; do
    log ""
    log "  [${tracker}] Starting passive eval..."
    PASSIVE_OUT="${PROJECT_ROOT}/outputs/eval_v2/${tracker}/uav123/test"
    PASSIVE_METRICS="${PASSIVE_OUT}/passive/metrics.json"
    EVAL_LOG="${PROJECT_ROOT}/logs/v2_eval_${tracker}.log"

    # Determine the run_tag that run_with_csc.py will produce so we can check skip
    # run_tag = <tracker>_uav123_test_<csc_model_tag>
    # csc_model_tag is derived from checkpoint dir name by run_with_csc._csc_model_name
    # For sglatrack_v2_tcn16: tag will be "sglatrack_v2_tcn16" or similar
    # The skip guard checks for passive/metrics.json (moved from run_tag dir)
    if [[ -f "${PASSIVE_METRICS}" ]]; then
        log "  [${tracker}] SKIP: ${PASSIVE_METRICS} exists"
        continue
    fi

    mkdir -p "${PASSIVE_OUT}"
    mkdir -p "$(dirname "${EVAL_LOG}")"

    log "  [${tracker}] Running run_with_csc.py (log: ${EVAL_LOG})"
    CSC_NOT_TRAINED_ON_UAV123=1 \
    "${PYTHON}" -u "${PROJECT_ROOT}/tools/run_with_csc.py" \
        --tracker "${tracker}" \
        --dataset uav123 \
        --split test \
        --csc_checkpoint "${V2_CKPT}" \
        --csc_mode passive \
        --output_dir "${PASSIVE_OUT}" \
        --device cpu \
        > "${EVAL_LOG}" 2>&1

    # run_with_csc writes to output_dir/<run_tag>/; move to passive/
    # Find the newest metrics.json that is not already in passive/
    RUN_TAG_DIR=$(ls -td "${PASSIVE_OUT}"/*/metrics.json 2>/dev/null \
        | grep -v '/passive/' | head -1 \
        | xargs -I{} dirname {} 2>/dev/null || true)
    if [[ -n "${RUN_TAG_DIR}" ]]; then
        log "  [${tracker}] Moving ${RUN_TAG_DIR} → ${PASSIVE_OUT}/passive"
        rm -rf "${PASSIVE_OUT}/passive"
        mv "${RUN_TAG_DIR}" "${PASSIVE_OUT}/passive"
        log "  [${tracker}] Passive eval complete."
    else
        warn "[${tracker}] Could not find run_tag directory — check ${EVAL_LOG}"
    fi
done

# ---------------------------------------------------------------------------
# Step 6: Run tracking + paper metrics for each tracker
# ---------------------------------------------------------------------------
log ""
log "=== Step 6: Tracking metrics + paper metrics for all 5 trackers ==="

for tracker in "${TRACKERS[@]}"; do
    log ""
    log "  [${tracker}] Running metrics pipeline..."
    EVAL_V2_ROOT="${PROJECT_ROOT}/outputs/eval_v2/${tracker}/uav123/test"
    PASSIVE_DIR="${EVAL_V2_ROOT}/passive"
    TRACKING_EVAL_DIR="${EVAL_V2_ROOT}/tracking_metrics"
    EPISODE_EVAL_DIR="${EVAL_V2_ROOT}/episode_metrics"
    PAPER_METRICS_DIR="${EVAL_V2_ROOT}/paper_metrics"
    LABELS_DIR="${EVAL_V2_ROOT}/labels"

    # Determine calibration files for this tracker
    # Priority: sglatrack uses sglatrack_v2; others use <tracker>_aerial_v2
    if [[ "${tracker}" == "sglatrack" ]]; then
        CALIB_TAG="sglatrack_v2"
    else
        CALIB_TAG="${tracker}_aerial_v2"
    fi
    CALIB_CONF="${PROJECT_ROOT}/outputs/calibration/${CALIB_TAG}_confidence.json"
    BASELINE_PREDS="${PROJECT_ROOT}/outputs/baselines/${tracker}/uav123/test/predictions"
    BASELINE_TEL="${PROJECT_ROOT}/outputs/baselines/${tracker}/uav123/test/telemetry"

    # Verify calibrator exists
    if [[ ! -f "${CALIB_CONF}" ]]; then
        warn "[${tracker}] Calibrator not found: ${CALIB_CONF} — skipping metrics for this tracker"
        continue
    fi

    # Verify passive eval succeeded
    if [[ ! -d "${PASSIVE_DIR}/states" ]]; then
        warn "[${tracker}] Passive states dir missing: ${PASSIVE_DIR}/states — skipping metrics"
        continue
    fi

    # --- 6a. GT label generation (eval-only, never used for training) ---
    LABELS_LEAF="${LABELS_DIR}/uav123/test"
    if [[ -d "${LABELS_LEAF}/labels_per_sequence" ]] && \
       [[ -n "$(ls "${LABELS_LEAF}/labels_per_sequence/"*.jsonl 2>/dev/null | head -1)" ]]; then
        log "  [${tracker}] SKIP labels: already exist"
    elif [[ -f "${LABELS_LEAF}/labels.jsonl" ]]; then
        log "  [${tracker}] SKIP labels: labels.jsonl exists"
    else
        log "  [${tracker}] Generating GT state labels..."
        mkdir -p "${LABELS_DIR}"
        CSC_NOT_TRAINED_ON_UAV123=1 \
        "${PYTHON}" -u "${PROJECT_ROOT}/tools/build_scene_state_labels.py" \
            --tracker "${tracker}" \
            --dataset uav123 \
            --split test \
            --baseline_dir "${PROJECT_ROOT}/outputs/baselines/${tracker}" \
            --calibration_dir "${PROJECT_ROOT}/outputs/calibration" \
            --calibrator_tag "${CALIB_TAG}" \
            --output_dir "${LABELS_DIR}" || warn "[${tracker}] Label generation failed — continuing"
        log "  [${tracker}] Labels done."
    fi

    # --- 6b. Standard tracking metrics ---
    if [[ -f "${TRACKING_EVAL_DIR}/summary.json" ]]; then
        log "  [${tracker}] SKIP tracking metrics: summary.json exists"
    else
        log "  [${tracker}] Computing AUC / Precision / FPS..."
        mkdir -p "${TRACKING_EVAL_DIR}"
        "${PYTHON}" -u "${PROJECT_ROOT}/tools/evaluate_tracking_results.py" \
            --dataset uav123 \
            --split test \
            --pred_dir "${BASELINE_PREDS}" \
            --telemetry_dir "${BASELINE_TEL}" \
            --output_dir "${TRACKING_EVAL_DIR}" || warn "[${tracker}] Tracking metrics failed — continuing"
        log "  [${tracker}] Tracking metrics done."
    fi

    # --- 6c. Episode-level CSC metrics ---
    if [[ -f "${EPISODE_EVAL_DIR}/episode_summary.json" ]]; then
        log "  [${tracker}] SKIP episode metrics: episode_summary.json exists"
    else
        log "  [${tracker}] Computing episode metrics..."
        mkdir -p "${EPISODE_EVAL_DIR}"
        "${PYTHON}" -u "${PROJECT_ROOT}/tools/evaluate_csc_episodes.py" \
            --labels "${LABELS_DIR}/uav123/test" \
            --predictions "${PASSIVE_DIR}/states" \
            --out "${EPISODE_EVAL_DIR}" || warn "[${tracker}] Episode metrics failed — continuing"
        log "  [${tracker}] Episode metrics done."
    fi

    # --- 6d. Paper metrics ---
    if [[ -f "${PAPER_METRICS_DIR}/paper_metrics.csv" ]]; then
        log "  [${tracker}] SKIP paper metrics: paper_metrics.csv exists"
    else
        log "  [${tracker}] Computing paper metrics (FCR/FCD/TTFC/Recovery@30)..."
        mkdir -p "${PAPER_METRICS_DIR}"
        "${PYTHON}" -u "${PROJECT_ROOT}/tools/compute_paper_metrics.py" \
            --tracker "${tracker}" \
            --dataset uav123 \
            --split test \
            --predictions_dir "${BASELINE_PREDS}" \
            --telemetry_dir "${BASELINE_TEL}" \
            --states_dir "${PASSIVE_DIR}/states" \
            --labels_dir "${LABELS_DIR}/uav123/test" \
            --tracking_metrics_dir "${TRACKING_EVAL_DIR}" \
            --output_dir "${PAPER_METRICS_DIR}" \
            --confidence_calib "${CALIB_CONF}" \
            --recovery_k 30 || warn "[${tracker}] Paper metrics failed — continuing"
        log "  [${tracker}] Paper metrics done."
    fi

    log "  [${tracker}] All metrics complete."
done

# ---------------------------------------------------------------------------
# Step 7: SGLATrack control variants (A, B, C, AC)
# ---------------------------------------------------------------------------
log ""
log "=== Step 7: SGLATrack control variants ==="

SGLA_PASSIVE_DIR="${PROJECT_ROOT}/outputs/eval_v2/sglatrack/uav123/test"

# Variant A — StateExitRouter (layer adaptation only)
CTRL_A_DIR="${SGLA_PASSIVE_DIR}/control_A"
CTRL_A_LOG="${PROJECT_ROOT}/logs/v2_control_A.log"
if [[ -f "${CTRL_A_DIR}/metrics.json" ]]; then
    log "  [control_A] SKIP: metrics.json exists"
else
    log "  [control_A] Running StateExitRouter variant..."
    mkdir -p "${CTRL_A_DIR}"
    CSC_NOT_TRAINED_ON_UAV123=1 \
    "${PYTHON}" -u "${PROJECT_ROOT}/tools/run_with_csc.py" \
        --tracker sglatrack \
        --dataset uav123 \
        --split test \
        --csc_checkpoint "${V2_CKPT}" \
        --csc_mode control \
        --exit_router \
        --output_dir "${CTRL_A_DIR}" \
        --device cpu \
        > "${CTRL_A_LOG}" 2>&1 || warn "control_A failed — check ${CTRL_A_LOG}"

    # Move run_tag subdirectory up to ctrl_A dir
    RUN_TAG_DIR=$(ls -td "${CTRL_A_DIR}"/*/metrics.json 2>/dev/null \
        | head -1 | xargs -I{} dirname {} 2>/dev/null || true)
    if [[ -n "${RUN_TAG_DIR}" ]]; then
        log "  [control_A] Renaming run_tag dir to flat control_A"
        find "${CTRL_A_DIR}" -maxdepth 1 -mindepth 1 -type d | while read -r d; do
            if [[ -f "${d}/metrics.json" ]]; then
                for item in predictions telemetry states metrics.json; do
                    [[ -e "${d}/${item}" ]] && mv "${d}/${item}" "${CTRL_A_DIR}/" || true
                done
                rmdir "${d}" 2>/dev/null || true
            fi
        done
    fi
    log "  [control_A] Done."
fi

# Variant B — basic control (freeze only, no exit_router, no advisor)
CTRL_B_DIR="${SGLA_PASSIVE_DIR}/control_B"
CTRL_B_LOG="${PROJECT_ROOT}/logs/v2_control_B.log"
if [[ -f "${CTRL_B_DIR}/metrics.json" ]]; then
    log "  [control_B] SKIP: metrics.json exists"
else
    log "  [control_B] Running basic control (freeze only)..."
    mkdir -p "${CTRL_B_DIR}"
    CSC_NOT_TRAINED_ON_UAV123=1 \
    "${PYTHON}" -u "${PROJECT_ROOT}/tools/run_with_csc.py" \
        --tracker sglatrack \
        --dataset uav123 \
        --split test \
        --csc_checkpoint "${V2_CKPT}" \
        --csc_mode control \
        --output_dir "${CTRL_B_DIR}" \
        --device cpu \
        > "${CTRL_B_LOG}" 2>&1 || warn "control_B failed — check ${CTRL_B_LOG}"

    # Move run_tag subdirectory up
    find "${CTRL_B_DIR}" -maxdepth 1 -mindepth 1 -type d | while read -r d; do
        if [[ -f "${d}/metrics.json" ]]; then
            for item in predictions telemetry states metrics.json; do
                [[ -e "${d}/${item}" ]] && mv "${d}/${item}" "${CTRL_B_DIR}/" || true
            done
            rmdir "${d}" 2>/dev/null || true
        fi
    done
    log "  [control_B] Done."
fi

# Variant C — CSCAdvisor (full stateful gating)
CTRL_C_DIR="${SGLA_PASSIVE_DIR}/control_C"
CTRL_C_LOG="${PROJECT_ROOT}/logs/v2_control_C.log"
if [[ -f "${CTRL_C_DIR}/metrics.json" ]]; then
    log "  [control_C] SKIP: metrics.json exists"
else
    log "  [control_C] Running CSCAdvisor (full stateful gating)..."
    mkdir -p "${CTRL_C_DIR}"
    CSC_NOT_TRAINED_ON_UAV123=1 \
    "${PYTHON}" -u "${PROJECT_ROOT}/tools/run_with_csc.py" \
        --tracker sglatrack \
        --dataset uav123 \
        --split test \
        --csc_checkpoint "${V2_CKPT}" \
        --csc_mode control \
        --csc_advisor \
        --output_dir "${CTRL_C_DIR}" \
        --device cpu \
        > "${CTRL_C_LOG}" 2>&1 || warn "control_C failed — check ${CTRL_C_LOG}"

    # Move run_tag subdirectory up
    find "${CTRL_C_DIR}" -maxdepth 1 -mindepth 1 -type d | while read -r d; do
        if [[ -f "${d}/metrics.json" ]]; then
            for item in predictions telemetry states metrics.json; do
                [[ -e "${d}/${item}" ]] && mv "${d}/${item}" "${CTRL_C_DIR}/" || true
            done
            rmdir "${d}" 2>/dev/null || true
        fi
    done
    log "  [control_C] Done."
fi

# Variant AC — exit_router + csc_advisor combined
CTRL_AC_DIR="${SGLA_PASSIVE_DIR}/control_AC"
CTRL_AC_LOG="${PROJECT_ROOT}/logs/v2_control_AC.log"
if [[ -f "${CTRL_AC_DIR}/metrics.json" ]]; then
    log "  [control_AC] SKIP: metrics.json exists"
else
    log "  [control_AC] Running exit_router + CSCAdvisor combined..."
    mkdir -p "${CTRL_AC_DIR}"
    CSC_NOT_TRAINED_ON_UAV123=1 \
    "${PYTHON}" -u "${PROJECT_ROOT}/tools/run_with_csc.py" \
        --tracker sglatrack \
        --dataset uav123 \
        --split test \
        --csc_checkpoint "${V2_CKPT}" \
        --csc_mode control \
        --exit_router \
        --csc_advisor \
        --output_dir "${CTRL_AC_DIR}" \
        --device cpu \
        > "${CTRL_AC_LOG}" 2>&1 || warn "control_AC failed — check ${CTRL_AC_LOG}"

    # Move run_tag subdirectory up
    find "${CTRL_AC_DIR}" -maxdepth 1 -mindepth 1 -type d | while read -r d; do
        if [[ -f "${d}/metrics.json" ]]; then
            for item in predictions telemetry states metrics.json; do
                [[ -e "${d}/${item}" ]] && mv "${d}/${item}" "${CTRL_AC_DIR}/" || true
            done
            rmdir "${d}" 2>/dev/null || true
        fi
    done
    log "  [control_AC] Done."
fi

# ---------------------------------------------------------------------------
# Step 8: Collect all results and write FINAL_RESULTS_V2.md
# ---------------------------------------------------------------------------
log ""
log "=== Step 8: Assemble FINAL_RESULTS_V2.md ==="

FINAL_MD="${RESULTS_DIR}/FINAL_RESULTS_V2.md"

"$PYTHON" - <<'PYEOF'
import json
import csv
import os
from pathlib import Path

project_root = Path(os.environ.get("PROJECT_ROOT", "."))
results_dir = project_root / "outputs/results_v2"
eval_v2_root = project_root / "outputs/eval_v2"
summary_path = results_dir / "training_summary.json"
out_path = results_dir / "FINAL_RESULTS_V2.md"

trackers = ["sglatrack", "ortrack", "ostrack", "avtrack", "evptrack"]

lines = []
lines.append("# FINAL_RESULTS_V2 — CSC v2 Evaluation on UAV123")
lines.append("")
lines.append(f"*Generated: {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}*")
lines.append("")

# --- Training summary ---
lines.append("---")
lines.append("")
lines.append("## Training Summary (sglatrack_v2_tcn16)")
lines.append("")
if summary_path.exists():
    s = json.loads(summary_path.read_text())
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total epochs | {s.get('total_epochs', '?')} |")
    lines.append(f"| Best epoch | {s.get('best_epoch', '?')} |")
    lines.append(f"| derivedF1 | {s.get('best_derivedF1', '?')} |")
    lines.append(f"| FC_recall | {s.get('best_FC_recall', '?')} |")
    lines.append(f"| AUROC | {s.get('best_AUROC', '?')} |")
    lines.append(f"| AUPRC | {s.get('best_AUPRC', '?')} |")
    lines.append(f"| Selection score | {s.get('best_selection_score', '?')} |")
    lines.append(f"| Checkpoint | `{s.get('checkpoint', '?')}` |")
else:
    lines.append("*training_summary.json not found*")
lines.append("")

# --- 5-tracker passive diagnosis table ---
lines.append("---")
lines.append("")
lines.append("## 5-Tracker Passive Diagnosis Table (UAV123)")
lines.append("")
lines.append("| Tracker | AUC | Precision@20 | FPS | FCR | FCD | Lost Rate | Recovery@30 |")
lines.append("|---------|-----|-------------|-----|-----|-----|-----------|-------------|")

for tracker in trackers:
    eval_dir = eval_v2_root / tracker / "uav123/test"
    tracking_json = eval_dir / "tracking_metrics/summary.json"
    paper_csv = eval_dir / "paper_metrics/paper_metrics.csv"

    auc = prec = fps = fcr = fcd = lost_rate = rec30 = "—"

    if tracking_json.exists():
        try:
            tm = json.loads(tracking_json.read_text())
            auc = f"{tm.get('success_auc', tm.get('AUC', '?')):.4f}" if isinstance(tm.get('success_auc', tm.get('AUC')), float) else str(tm.get('success_auc', tm.get('AUC', '?')))
            prec = f"{tm.get('precision_at_20', tm.get('Precision@20', '?')):.4f}" if isinstance(tm.get('precision_at_20', tm.get('Precision@20')), float) else str(tm.get('precision_at_20', tm.get('Precision@20', '?')))
            fps = f"{tm.get('fps', '?'):.1f}" if isinstance(tm.get('fps'), float) else str(tm.get('fps', '?'))
        except Exception:
            pass

    if paper_csv.exists():
        try:
            with open(paper_csv) as f:
                rows = list(csv.DictReader(f))
            # Look for aggregate row (last row or row with sequence == "aggregate")
            agg = None
            for row in rows:
                seq = row.get("sequence", row.get("tracker", ""))
                if "aggregate" in str(seq).lower() or "mean" in str(seq).lower():
                    agg = row
                    break
            if agg is None and rows:
                agg = rows[-1]
            if agg:
                def _fmt(key, fmt=".4f"):
                    v = agg.get(key)
                    if v is None:
                        return "—"
                    try:
                        return format(float(v), fmt)
                    except Exception:
                        return str(v)
                fcr = _fmt("fcr")
                fcd = _fmt("fcd", ".1f")
                lost_rate = _fmt("lost_rate")
                rec30 = _fmt("recovery_at_30")
        except Exception:
            pass

    lines.append(f"| {tracker} | {auc} | {prec} | {fps} | {fcr} | {fcd} | {lost_rate} | {rec30} |")

lines.append("")

# --- SGLATrack control comparison table ---
lines.append("---")
lines.append("")
lines.append("## SGLATrack Control Variant Comparison (UAV123)")
lines.append("")
lines.append("| Variant | Description | AUC | FCR | FCD | Recovery@30 |")
lines.append("|---------|-------------|-----|-----|-----|-------------|")

control_variants = [
    ("passive",    "Passive diagnosis only",              eval_v2_root / "sglatrack/uav123/test/passive"),
    ("control_A",  "StateExitRouter (layer adaptation)",  eval_v2_root / "sglatrack/uav123/test/control_A"),
    ("control_B",  "Freeze only (basic control)",         eval_v2_root / "sglatrack/uav123/test/control_B"),
    ("control_C",  "CSCAdvisor (full stateful gating)",   eval_v2_root / "sglatrack/uav123/test/control_C"),
    ("control_AC", "ExitRouter + CSCAdvisor combined",    eval_v2_root / "sglatrack/uav123/test/control_AC"),
]

# Passive tracking metrics from the standard eval dir (step 6 above)
sgla_tracking = eval_v2_root / "sglatrack/uav123/test/tracking_metrics/summary.json"
sgla_paper = eval_v2_root / "sglatrack/uav123/test/paper_metrics/paper_metrics.csv"

for variant_id, desc, variant_dir in control_variants:
    auc = prec = fcr = fcd = rec30 = "—"

    # For passive, use the step-6 metrics computed above
    if variant_id == "passive":
        if sgla_tracking.exists():
            try:
                tm = json.loads(sgla_tracking.read_text())
                auc = f"{tm.get('success_auc', tm.get('AUC', '?')):.4f}" if isinstance(tm.get('success_auc', tm.get('AUC')), float) else "—"
            except Exception:
                pass
        if sgla_paper.exists():
            try:
                with open(sgla_paper) as f:
                    rows = list(csv.DictReader(f))
                agg = next((r for r in rows if "aggregate" in str(r.get("sequence","")).lower()), rows[-1] if rows else None)
                if agg:
                    def _fmt(key, fmt=".4f"):
                        v = agg.get(key)
                        if v is None: return "—"
                        try: return format(float(v), fmt)
                        except: return str(v)
                    fcr = _fmt("fcr")
                    fcd = _fmt("fcd", ".1f")
                    rec30 = _fmt("recovery_at_30")
            except Exception:
                pass
    else:
        metrics_json = variant_dir / "metrics.json"
        if metrics_json.exists():
            try:
                m = json.loads(metrics_json.read_text())
                auc = f"{m.get('success_auc', m.get('AUC', '—')):.4f}" if isinstance(m.get('success_auc', m.get('AUC')), float) else "—"
            except Exception:
                pass
        # If paper_metrics exist for control variant, use them
        ctrl_paper = variant_dir / "paper_metrics/paper_metrics.csv"
        if ctrl_paper.exists():
            try:
                with open(ctrl_paper) as f:
                    rows = list(csv.DictReader(f))
                agg = next((r for r in rows if "aggregate" in str(r.get("sequence","")).lower()), rows[-1] if rows else None)
                if agg:
                    def _fmt_c(key, fmt=".4f"):
                        v = agg.get(key)
                        if v is None: return "—"
                        try: return format(float(v), fmt)
                        except: return str(v)
                    fcr = _fmt_c("fcr")
                    fcd = _fmt_c("fcd", ".1f")
                    rec30 = _fmt_c("recovery_at_30")
            except Exception:
                pass

    lines.append(f"| {variant_id} | {desc} | {auc} | {fcr} | {fcd} | {rec30} |")

lines.append("")
lines.append("---")
lines.append("")
lines.append("## Notes")
lines.append("")
lines.append("- All UAV123 evaluations use `CSC_NOT_TRAINED_ON_UAV123=1` (CLAUDE.md safety gate)")
lines.append("- CSC v2 checkpoint: `outputs/csc_training/sglatrack_v2_tcn16/checkpoint_best.pth`")
lines.append("- Calibrators: sglatrack uses `sglatrack_v2`; others use `<tracker>_aerial_v2`")
lines.append("- Control variants A/B/C/AC apply to SGLATrack only (other trackers: passive only)")
lines.append("- `—` indicates metric not yet computed or eval step incomplete")
lines.append("")

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(lines) + "\n")
print(f"Written: {out_path}")
PYEOF

log ""
log "========================================"
log " run_post_v2_eval.sh DONE"
log "========================================"
log ""
log "  Training summary : ${RESULTS_DIR}/training_summary.json"
log "  Final results MD : ${RESULTS_DIR}/FINAL_RESULTS_V2.md"
log ""
log "  Per-tracker eval dirs:"
for tracker in "${TRACKERS[@]}"; do
    log "    ${tracker}: ${PROJECT_ROOT}/outputs/eval_v2/${tracker}/uav123/test/"
done
log ""
log "  SGLATrack control logs:"
log "    A : ${PROJECT_ROOT}/logs/v2_control_A.log"
log "    B : ${PROJECT_ROOT}/logs/v2_control_B.log"
log "    C : ${PROJECT_ROOT}/logs/v2_control_C.log"
log "    AC: ${PROJECT_ROOT}/logs/v2_control_AC.log"
