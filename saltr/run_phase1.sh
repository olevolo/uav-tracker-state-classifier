#!/usr/bin/env bash
# run_phase1.sh — SALT-RD Phase 1 full pipeline
#
# Usage: bash saltr/run_phase1.sh [--max-frames N] [--datasets d1 d2 d3]
#
# Steps:
#   1. Collect NPZ from frozen tracker on UAV123/VisDrone/DTB70
#   2. Train SALTRD GRU model
#   3. Eval: AUROC/AUPRC/ECE/NT2F + GO/NO-GO gate
#
# Must be run from project root with PYTHONPATH=src set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NPZ="${PROJECT_ROOT}/saltr/data/salt_rd_v0.npz"
CHECKPOINT_DIR="${PROJECT_ROOT}/saltr/checkpoints"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PYTHONPATH="src:saltr/src"
DATASETS="${DATASETS:-uav123 visdrone_sot dtb70}"
MAX_FRAMES="${MAX_FRAMES:-}"
EPOCHS="${EPOCHS:-50}"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-frames) MAX_FRAMES="$2"; shift 2 ;;
        --datasets) DATASETS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --npz) NPZ="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cd "${PROJECT_ROOT}"

# Step 1: Collect
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

# Step 2: Train
echo ""
echo "=== Step 2: Train ==="
mkdir -p "${CHECKPOINT_DIR}"
PYTHONPATH="${PYTHONPATH}" "${PYTHON}" saltr/src/salt_r/train.py \
    --npz "${NPZ}" \
    --output "${CHECKPOINT_DIR}" \
    --epochs "${EPOCHS}"

# Step 3: Eval
CHECKPOINT="${CHECKPOINT_DIR}/saltrd_best.pt"
echo ""
echo "=== Step 3: Eval (val split) ==="
PYTHONPATH="${PYTHONPATH}" "${PYTHON}" saltr/src/salt_r/eval.py \
    --npz "${NPZ}" \
    --checkpoint "${CHECKPOINT}" \
    --split val \
    --output "${CHECKPOINT_DIR}/eval_val.json"

echo ""
echo "=== Step 3b: Eval (diagnostic split) ==="
PYTHONPATH="${PYTHONPATH}" "${PYTHON}" saltr/src/salt_r/eval.py \
    --npz "${NPZ}" \
    --checkpoint "${CHECKPOINT}" \
    --split diagnostic \
    --output "${CHECKPOINT_DIR}/eval_diagnostic.json" 2>/dev/null || echo "(no diagnostic sequences in NPZ)"

echo ""
echo "Pipeline complete. Check GO/NO-GO above."
echo "Checkpoint: ${CHECKPOINT}"
echo "Eval results: ${CHECKPOINT_DIR}/eval_val.json"
