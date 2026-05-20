#!/usr/bin/env bash
# run_phase1.sh — SALT-RD Phase 1 full pipeline
#
# Usage: bash saltr/run_phase1.sh [--max-frames N] [--datasets d1 d2 d3]
# Steps:
#   1. Collect NPZ from frozen tracker on UAV123/VisDrone/DTB70
#   2. Train SALTRD GRU model
#   3. Eval: AUROC/AUPRC/ECE/NT2F + GO/NO-GO gate
#
# Must be run from project root with PYTHONPATH=src set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NPZ=""   # set after schema selection; override with --npz
CHECKPOINT_DIR="${PROJECT_ROOT}/saltr/checkpoints"
RESULTS_DIR="${PROJECT_ROOT}/saltr/results"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PYTHONPATH="src:saltr/src"
DATASETS="${DATASETS:-uav123 visdrone_sot dtb70}"
MAX_FRAMES="${MAX_FRAMES:-}"
EPOCHS="${EPOCHS:-50}"
LABEL_SCHEMA="${LABEL_SCHEMA:-v0}"   # v0 (7 heads) or v1 (9 heads with split dynamic labels)
CALIBRATE="${CALIBRATE:-}"           # set to "1" to enable --calibrate-heads for reliability heads
FORCE_RECOMPUTE_LABELS=""            # set via --force-recompute-labels to regenerate even if NPZ exists

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-frames) MAX_FRAMES="$2"; shift 2 ;;
        --datasets)
            shift
            DATASETS=""
            while [[ $# -gt 0 && "$1" != --* ]]; do
                DATASETS="${DATASETS:+$DATASETS }$1"
                shift
            done
            ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --npz) NPZ="$2"; shift 2 ;;
        --label-schema) LABEL_SCHEMA="$2"; shift 2 ;;
        --calibrate) CALIBRATE="1"; shift ;;
        --force-recompute-labels) FORCE_RECOMPUTE_LABELS="1"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cd "${PROJECT_ROOT}"

# Select NPZ and checkpoint prefix based on label schema
if [[ "${LABEL_SCHEMA}" == "v2" ]]; then
    NPZ="${NPZ:-${PROJECT_ROOT}/saltr/data/salt_rd_v2_labels.npz}"
    CHECKPOINT_DIR="${PROJECT_ROOT}/saltr/checkpoints/v2"
elif [[ "${LABEL_SCHEMA}" == "v1" ]]; then
    NPZ="${NPZ:-${PROJECT_ROOT}/saltr/data/salt_rd_v1_labels.npz}"
    CHECKPOINT_DIR="${PROJECT_ROOT}/saltr/checkpoints/v1"
else
    NPZ="${NPZ:-${PROJECT_ROOT}/saltr/data/salt_rd_v0.npz}"
fi

# Optional --calibrate-heads flag for eval step
CALIBRATE_ARGS=""
if [[ -n "${CALIBRATE}" ]]; then
    CALIBRATE_ARGS="--calibrate-heads"
fi

# Schema tag for output artifact filenames — prevents v0/v1/v2 results mixing
SCHEMA_TAG="${LABEL_SCHEMA}"

# Step 1: Collect / generate labels
if [[ "${LABEL_SCHEMA}" == "v2" ]]; then
    V1_NPZ="${PROJECT_ROOT}/saltr/data/salt_rd_v1_labels.npz"
    V0_NPZ="${PROJECT_ROOT}/saltr/data/salt_rd_v0.npz"
    if [[ -n "${FORCE_RECOMPUTE_LABELS}" && -f "${NPZ}" ]]; then
        echo "=== Step 1: --force-recompute-labels: removing ${NPZ} ==="
        rm "${NPZ}"
    fi
    if [[ -f "${NPZ}" ]]; then
        echo "=== Step 1: Skipping (${NPZ} exists) ==="
    elif [[ -f "${V1_NPZ}" ]]; then
        echo "=== Step 1: Generating v2 labels from ${V1_NPZ} ==="
        PYTHONPATH="${PYTHONPATH}" "${PYTHON}" -c "
from salt_r.collect_features import recompute_labels_v2
recompute_labels_v2('${V1_NPZ}', '${NPZ}')
"
    elif [[ -f "${V0_NPZ}" ]]; then
        echo "=== Step 1: Generating v1 then v2 labels from ${V0_NPZ} ==="
        PYTHONPATH="${PYTHONPATH}" "${PYTHON}" -c "
from salt_r.collect_features import recompute_labels_v1, recompute_labels_v2
recompute_labels_v1('${V0_NPZ}', '${V1_NPZ}')
recompute_labels_v2('${V1_NPZ}', '${NPZ}')
"
    else
        echo "ERROR: --label-schema v2 requires ${NPZ}, ${V1_NPZ}, or ${V0_NPZ}."
        echo "  Run bash saltr/run_phase1.sh first (v0 collection) before training v2."
        exit 1
    fi
elif [[ "${LABEL_SCHEMA}" == "v1" ]]; then
    V0_NPZ="${PROJECT_ROOT}/saltr/data/salt_rd_v0.npz"
    if [[ -n "${FORCE_RECOMPUTE_LABELS}" && -f "${NPZ}" ]]; then
        echo "=== Step 1: --force-recompute-labels: removing ${NPZ} ==="
        rm "${NPZ}"
    fi
    if [[ -f "${NPZ}" ]]; then
        echo "=== Step 1: Skipping (${NPZ} exists) ==="
    elif [[ -f "${V0_NPZ}" ]]; then
        echo "=== Step 1: Generating v1 labels from ${V0_NPZ} ==="
        PYTHONPATH="${PYTHONPATH}" "${PYTHON}" -c "
from salt_r.collect_features import recompute_labels_v1
recompute_labels_v1('${V0_NPZ}', '${NPZ}')
"
    else
        echo "ERROR: --label-schema v1 requires either ${NPZ} or ${V0_NPZ}."
        echo "  Run bash saltr/run_phase1.sh first (v0 collection) before training v1."
        exit 1
    fi
else
    if [[ -n "${FORCE_RECOMPUTE_LABELS}" && -f "${NPZ}" ]]; then
        echo "=== Step 1: --force-recompute-labels: removing ${NPZ} ==="
        rm "${NPZ}"
    fi
    if [[ ! -f "${NPZ}" ]]; then
        echo "=== Step 1: Collect features ==="
        FRAME_CAP_ARG=""
        if [[ -n "${MAX_FRAMES}" ]]; then
            FRAME_CAP_ARG="--max-frames ${MAX_FRAMES}"
        fi
        PYTHONPATH="${PYTHONPATH}" "${PYTHON}" saltr/src/salt_r/collect_features.py \
            --output "${NPZ}" \
            --datasets ${DATASETS} \
            ${FRAME_CAP_ARG}
    else
        echo "=== Step 1: Skipping (${NPZ} exists) ==="
    fi
fi

# Step 2: Train
echo ""
echo "=== Step 2: Train (schema=${LABEL_SCHEMA}) ==="
mkdir -p "${CHECKPOINT_DIR}"
PYTHONPATH="${PYTHONPATH}" "${PYTHON}" saltr/src/salt_r/train.py \
    --npz "${NPZ}" \
    --output "${CHECKPOINT_DIR}" \
    --epochs "${EPOCHS}" \
    --label-schema "${LABEL_SCHEMA}"

# Step 3: Eval (val split, with predictions export — combined step)
CHECKPOINT="${CHECKPOINT_DIR}/saltrd_best.pt"
echo ""
echo "=== Step 3: Eval (val split, schema=${SCHEMA_TAG}${CALIBRATE:+, calibrated}) ==="
mkdir -p "${RESULTS_DIR}"
PYTHONPATH="${PYTHONPATH}" "${PYTHON}" saltr/src/salt_r/eval.py \
    --npz "${NPZ}" \
    --checkpoint "${CHECKPOINT}" \
    --split val \
    --output "${RESULTS_DIR}/eval_val_${SCHEMA_TAG}.json" \
    --predictions-output "${RESULTS_DIR}/preds_val_${SCHEMA_TAG}.json" \
    ${CALIBRATE_ARGS}

# Step 4: Policy offline replay
PREDS_JSON="${RESULTS_DIR}/preds_val_${SCHEMA_TAG}.json"
echo ""
echo "=== Step 4: Policy offline replay ==="
if [[ -f "${PREDS_JSON}" ]]; then
    if [[ "${LABEL_SCHEMA}" == "v2" ]]; then
        # v2 schema: use Phase 6 policy_sweep (v2-aware: ifd10/20, memory, e-process)
        # legacy salt_r.policy does NOT use these signals; use run_phase6.sh instead.
        echo "NOTE: label-schema=v2 → using policy_sweep.py (Phase 6). For memory/e-process, run run_phase6.sh."
        PYTHONPATH="${PYTHONPATH}" "${PYTHON}" -m salt_r.policy_sweep \
            --preds "${PREDS_JSON}" \
            --labels "${NPZ}" \
            --output "${RESULTS_DIR}/policy_val_${SCHEMA_TAG}.json"
    else
        # v0/v1 schema: legacy policy replay (ifd5 only, no ifd10/20/memory/e-process)
        echo "NOTE: salt_r.policy is legacy (v0/v1 only). For v2 experiments, use run_phase6.sh."
        PYTHONPATH="${PYTHONPATH}" "${PYTHON}" -m salt_r.policy \
            --probs-json "${PREDS_JSON}" \
            --npz "${NPZ}" \
            --output "${RESULTS_DIR}/policy_val_${SCHEMA_TAG}.json"
    fi
else
    echo "No predictions JSON found; skipping policy replay."
fi

echo ""
echo "=== Step 3b: Eval (diagnostic split) ==="
PYTHONPATH="${PYTHONPATH}" "${PYTHON}" saltr/src/salt_r/eval.py \
    --npz "${NPZ}" \
    --checkpoint "${CHECKPOINT}" \
    --split diagnostic \
    --output "${RESULTS_DIR}/eval_diagnostic_${SCHEMA_TAG}.json" 2>/dev/null || echo "(no diagnostic sequences in NPZ)"

echo ""
echo "Pipeline complete. Check GO/NO-GO above."
echo "Checkpoint: ${CHECKPOINT}"
echo "Results: ${RESULTS_DIR}/eval_val_${SCHEMA_TAG}.json"
