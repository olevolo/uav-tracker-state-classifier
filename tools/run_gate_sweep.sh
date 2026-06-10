#!/bin/bash
# Gate threshold sweep for AVTrack and ORTrack combo_mb.
# Tests gate_lostaware in {0.95, 0.97, 0.99} to find sweet spot:
# protect EASY (no regression) while preserving HARD gains.
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/csc_prod/checkpoint_best.pth
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4

run(){ local TRACKER=$1 CALIB=$2 THRESH=$3
  local ROOT="outputs/eval_gate_sweep/${TRACKER}/uav123/test"
  local TAG="combo_mb_t${THRESH/./}"
  local LOG="outputs/_logs/gate_sweep_${TRACKER}_${THRESH/./}.log"
  mkdir -p "$(dirname "$LOG")"
  [[ -d "$ROOT/$TAG/predictions" ]] && \
    echo "[SKIP] $TRACKER thresh=$THRESH already done" && return 0
  echo "[START] $TRACKER thresh=$THRESH → $TAG"
  $PY -u tools/run_with_csc.py \
    --tracker "$TRACKER" --dataset uav123 --split test \
    --csc_checkpoint "$CKPT" --calibration_prefix "${CALIB}" --device cpu \
    --output_dir "$ROOT" --run_tag "$TAG" \
    --csc_mode control \
    --policy_gated_redetect \
      --gate_preset csc_head --gate_lostaware "$THRESH" \
      --redetect_action motion_bridge --redetect_arm_frames 3 \
      --bridge_max_frames 30 --bridge_vel_ema 0.5 \
      --gated_freeze --no_runner_template_update \
    --policy_fc_control \
      --fc_action hold_lastgood --fc_streak_frames 2 --fc_gate_vote_k 2 \
      --no_runner_template_update \
    --control_risk_gate >> "$LOG" 2>&1
  echo "[DONE] $TRACKER thresh=$THRESH  exit=$?"
}
for THRESH in 0.95 0.97 0.99; do
  run avtrack avtrack_aerial_v2 "$THRESH" &
  run ortrack ortrack_aerial_v2 "$THRESH" &
done
wait
echo "ALL SWEEP DONE"
