#!/bin/bash
# LA-vs-FC AUC attribution on the top-HARD winner seqs. Runs the recommended config
# but with ONLY LA control (policy_gated_redetect) then ONLY FC control
# (policy_fc_control), so per-seq ΔAUC isolates each lever's contribution to AUC.
# Usage: bash run_la_fc_ablation.sh <dataset> <baseline_dir> seq1 seq2 ...
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
DS="$1"; BASE="$2"; shift 2; SEQS="$*"
ROOT=outputs/eval8_cosine/csc/sglatrack/$DS/test
BASEFLAGS="--gate_preset scoremap2 --gate_top2ratio 0.45 --gate_entropy 4.3 \
 --redetect_action motion_bridge --redetect_arm_frames 3 --bridge_max_frames 30 --bridge_vel_ema 0.5 \
 --fc_action hold_lastgood --fc_streak_frames 2 --fc_gate_vote_k 2 --gated_freeze --no_runner_template_update"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
run(){ local TAG="$1"; shift; rm -rf "$ROOT/$TAG"
  $PY -u tools/run_with_csc.py --tracker sglatrack --dataset "$DS" --split test \
    --csc_checkpoint "$CKPT" --calibration_prefix sglatrack_all_v2 --device cpu \
    --output_dir "$ROOT" --run_tag "$TAG" --include_sequences $SEQS \
    --csc_mode control "$@" $BASEFLAGS > "outputs/_logs/run_${DS}_${TAG}.log" 2>&1 || echo "FAIL $TAG"; }
run abl_la_only --policy_gated_redetect
run abl_fc_only --policy_fc_control
echo "===== $DS  LA-only (policy_gated_redetect) ====="
$PY tools/agg_full.py --dataset "$DS" --baseline "$BASE" --run_dir "$ROOT/abl_la_only" --tag la 2>/dev/null | sed -n '/TOP WINNERS/,/TOP LOSERS/p'
echo "===== $DS  FC-only (policy_fc_control) ====="
$PY tools/agg_full.py --dataset "$DS" --baseline "$BASE" --run_dir "$ROOT/abl_fc_only" --tag fc 2>/dev/null | sed -n '/TOP WINNERS/,/TOP LOSERS/p'
