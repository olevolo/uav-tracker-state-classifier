#!/usr/bin/env bash
# Phase 6: Policy sweep evaluation for SALT-RD v2+ models.
# Runs policy_sweep.py which uses the v2-aware policy (ifd10/20, e-process, memory).
# Replaces the old salt_r.policy path in run_phase1.sh for v2+ experiments.

set -euo pipefail
PYTHONPATH="${PYTHONPATH:-src:saltr/src}"
export PYTHONPATH

PREDS="${1:-saltr/results/preds_val_v2_retrained.json}"
LABELS="${2:-saltr/data/salt_rd_v2_labels.npz}"
EPROCESS="${3:-}"
MEMORY="${4:-}"
OUTPUT="${5:-saltr/results/policy_sweep_v2.json}"

echo "[phase6] Preds:    $PREDS"
echo "[phase6] Labels:   $LABELS"
echo "[phase6] E-process: ${EPROCESS:-none}"
echo "[phase6] Memory:    ${MEMORY:-none}"
echo "[phase6] Output:    $OUTPUT"

CMD=".venv/bin/python -m salt_r.policy_sweep --preds $PREDS --labels $LABELS --output $OUTPUT"
if [ -n "$EPROCESS" ]; then CMD="$CMD --eprocess $EPROCESS"; fi
if [ -n "$MEMORY" ]; then CMD="$CMD --memory $MEMORY"; fi

echo "[phase6] Running: $CMD"
eval "$CMD"
echo "[phase6] Done."
