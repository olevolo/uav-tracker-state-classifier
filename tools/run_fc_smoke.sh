#!/bin/bash
# FC-control alternatives on high-FC SGLATrack seqs (UAV123, EXPLORATORY).
# Router-independent --policy_fc_control: block9 (current) vs freeze_only (do-no-harm)
# vs hold_lastgood (snap to last confirmed) vs widen. Measure AUC + FCR + FCD vs passive.
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval7_gated/csc/sglatrack/uav123/test
SEQS="car6_4 wakeboard10 car17 boat8 car7 car2_s car5"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
LOG=outputs/_logs/run_fc_smoke.log; : > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
run_fc(){ local ACT="$1"; local TAG="fc_$ACT"; local RUN="$ROOT/$TAG"; rm -rf "$RUN"
  say ">>> [$TAG]"
  $PY -u tools/run_with_csc.py --tracker sglatrack --dataset uav123 --split test \
    --csc_checkpoint "$CKPT" --calibration_prefix sglatrack_aerial_v2 --device cpu \
    --output_dir "$ROOT" --run_tag "$TAG" --include_sequences $SEQS \
    --csc_mode control --policy_fc_control --fc_action "$ACT" --fc_streak_frames 2 >>"$LOG" 2>&1 \
    || { say "  FAIL $TAG"; return 1; }
  say "--- [$TAG] AUC+FC vs passive ---"
  $PY -u tools/la_smoke.py --run_dir "$RUN" --seqs $SEQS --tag "$TAG" --fc_metrics 2>/dev/null | tee -a "$LOG"; }
say "============ FC-control smoke ============"
run_fc freeze_only
run_fc block9
run_fc hold_lastgood
run_fc widen
say "============ DONE ============"
