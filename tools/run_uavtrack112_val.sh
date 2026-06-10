#!/bin/bash
# ============================================================================
# VALIDATION-set generalization (UAVTrack112, NOT UAV123) — confirm the scoremap2
# gate + motion_bridge + gated FC, with the SAME FROZEN thresholds chosen on UAV123,
# transfers to a held-out dataset. Generates a FRESH passive (same tracker/calib) so
# the ΔAUC is consistent, then the frozen control config, then the tercile compare.
# SAFETY: offline SOT benchmark, control-only INFERENCE, no training.
# ============================================================================
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval8_cosine/csc/sglatrack/uavtrack112/test
CALIB=sglatrack_all_v2
COMBO="--policy_gated_redetect --gate_preset scoremap2 --gate_top2ratio 0.30 --gate_entropy 4.0 \
 --redetect_action motion_bridge --redetect_arm_frames 3 --bridge_max_frames 30 --bridge_vel_ema 0.5 \
 --policy_fc_control --fc_action hold_lastgood --fc_streak_frames 2 --fc_gate_vote_k 2 \
 --gated_freeze --no_runner_template_update"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
mkdir -p outputs/_logs "$ROOT"
LOG=outputs/_logs/run_uavtrack112_val.log; : > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say ">>> UAVTrack112 PASSIVE baseline (fresh, same tracker+calib)"
rm -rf "$ROOT/val_passive"
$PY -u tools/run_with_csc.py --tracker sglatrack --dataset uavtrack112 --split test \
  --csc_checkpoint "$CKPT" --calibration_prefix "$CALIB" --device cpu \
  --csc_mode passive --output_dir "$ROOT" --run_tag val_passive >>"$LOG" 2>&1 || { say "FAIL passive"; exit 1; }

say ">>> UAVTrack112 CONTROL (frozen bridge config, UAV123-frozen thresholds)"
rm -rf "$ROOT/val_frozen"
$PY -u tools/run_with_csc.py --tracker sglatrack --dataset uavtrack112 --split test \
  --csc_checkpoint "$CKPT" --calibration_prefix "$CALIB" --device cpu \
  --csc_mode control --output_dir "$ROOT" --run_tag val_frozen $COMBO >>"$LOG" 2>&1 || { say "FAIL control"; exit 1; }

say "--- UAVTrack112 tercile compare (control vs fresh passive) ---"
$PY -u tools/agg_full.py --dataset uavtrack112 --baseline "$ROOT/val_passive" \
  --run_dir "$ROOT/val_frozen" --tag uavtrack112_val 2>/dev/null | tee -a "$LOG"
say "===== UAVTrack112 VAL DONE ====="; echo "UAVTRACK112_VAL_DONE" >> "$LOG"
