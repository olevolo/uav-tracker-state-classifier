#!/bin/bash
# FULL UAV123 (123 seqs) — combined LA+FC control, the recommended final config:
#   LA: motion_bridge (gated, combined K5, arm3) — recover smooth-motion losses
#   FC: hold_lastgood (snap back to last confirmed) — cut false-confirmed duration
#   + --control_risk_gate to suppress easy-scene template-update overhead.
# clean (NO exit_router/proactive block-9 scaffold). EXPLORATORY on UAV123
# (final threshold calibration belongs on VisDrone/UAVDT, never DTB70-leaked).
# Resumable (run_with_csc skips non-empty state files).
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval7_gated/csc/sglatrack/uav123/test
PASSIVE=outputs/eval5_clamp/csc/sglatrack/uav123/test/passive
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
LOG=outputs/_logs/run_full_combo.log; : > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
RUN="$ROOT/full_combo"
say "================ FULL combo (123 seqs) START ================"
$PY -u tools/run_with_csc.py --tracker sglatrack --dataset uav123 --split test \
  --csc_checkpoint "$CKPT" --calibration_prefix sglatrack_aerial_v2 --device cpu \
  --output_dir "$ROOT" --run_tag full_combo --control_risk_gate \
  --csc_mode control \
  --policy_gated_redetect --redetect_action motion_bridge --gate_preset combined --gate_vote_k 5 --redetect_arm_frames 3 --bridge_max_frames 30 \
  --policy_fc_control --fc_action hold_lastgood --fc_streak_frames 2 >>"$LOG" 2>&1 \
  || { say "FAIL run"; exit 1; }
say "--- combo metrics ---"
$PY -c "import json; m=json.load(open('$RUN/metrics.json')); print({k:m[k] for k in m if 'gated' in k or 'fc_fired' in k or 'widen' in k or 'risk_gate' in k or k in ('mean_total_fps','total_frames')})" | tee -a "$LOG"
say "--- FULL UAV123 ΔAUC + FCR/FCD vs passive (123 seqs) ---"
$PY -u tools/la_smoke.py --run_dir "$RUN" --baseline_dir "$PASSIVE" --fc_metrics --tag full_combo 2>/dev/null \
  | tail -8 | tee -a "$LOG"
say "================ DONE ================"
