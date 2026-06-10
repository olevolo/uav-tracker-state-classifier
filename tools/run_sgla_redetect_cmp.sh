#!/bin/bash
# SGLATrack self-redetect vs motion_bridge comparison (UAV123 hard set), per Codex's MVP.
# Same gate (scoremap2 0.45/4.3) + frozen + LA-only; only the ACTION changes.
# Key tests: does sgla_redetect recover motion_bridge's FAILURES (car11/bird1_1 erratic;
# uav2 extreme loss) WITHOUT wrecking guards/easy (uav6/car6_2 — the relocate-catastrophe risk)?
# ΔAUC vs eval5_clamp passive. EXPLORATORY offline SOT.
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval9_sgla/csc/sglatrack/uav123/test
SEQS="group3_2 car1_s person14_1 car11 bird1_1 uav2 uav6 car6_2"
GATE="--csc_mode control --policy_gated_redetect --gate_preset scoremap2 --gate_top2ratio 0.45 --gate_entropy 4.3 --redetect_arm_frames 3 --gated_freeze --no_runner_template_update"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
mkdir -p outputs/_logs "$ROOT"
LOG=outputs/_logs/run_sgla_cmp.log; : > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
run_cfg(){ local TAG="$1"; shift; local RUN="$ROOT/$TAG"; rm -rf "$RUN"
  say ">>> [$TAG] $*"
  $PY -u tools/run_with_csc.py --tracker sglatrack --dataset uav123 --split test \
    --csc_checkpoint "$CKPT" --calibration_prefix sglatrack_all_v2 --device cpu \
    --output_dir "$ROOT" --run_tag "$TAG" --include_sequences $SEQS \
    $GATE "$@" >>"$LOG" 2>&1 || { say "  FAIL $TAG"; return 1; }
  say "--- [$TAG] ΔAUC ---"
  $PY -u tools/la_smoke.py --run_dir "$RUN" --seqs $SEQS --tag "$TAG" 2>/dev/null | tee -a "$LOG"; }
say "===== sgla_redetect comparison ($SEQS) ====="
run_cfg mb         --redetect_action motion_bridge --bridge_max_frames 30 --bridge_vel_ema 0.5
run_cfg sgla       --redetect_action sgla_redetect --sgla_redetect_factors 8,12,16 --sgla_redetect_grid 0 --sgla_redetect_min_apce 0
run_cfg bridgesgla --redetect_action bridge_sgla --sgla_redetect_factors 8,12,16 --bridge_max_frames 30 --bridge_vel_ema 0.5
say "===== DONE ====="; echo "SGLA_CMP_DONE" >> "$LOG"
