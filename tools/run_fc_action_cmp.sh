#!/bin/bash
# FC-action comparison (gated vote>=2, frozen). Tests the user's point: widen /
# freeze_only (no hard snap) vs hold_lastgood (hard snap → teleport-on-misfire risk:
# group1 -0.466, introduces FC on uav3/car1_3). Seqs = FC-heavy + the problem seqs.
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval8_cosine/csc/sglatrack/uav123/test
SEQS="car6_4 wakeboard10 car17 boat8 car7 car2_s car5 uav3 car1_3"
COMMON="--csc_mode control --policy_fc_control --fc_streak_frames 2 --fc_gate_vote_k 2 --gated_freeze --no_runner_template_update"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
run(){ local A="$1"; local TAG="fccmp_$A"; rm -rf "$ROOT/$TAG"
  $PY -u tools/run_with_csc.py --tracker sglatrack --dataset uav123 --split test \
    --csc_checkpoint "$CKPT" --calibration_prefix sglatrack_all_v2 --device cpu \
    --output_dir "$ROOT" --run_tag "$TAG" --include_sequences $SEQS \
    $COMMON --fc_action "$A" --lost_widen_max 1.5 > "outputs/_logs/run_$TAG.log" 2>&1 || { echo "FAIL $A"; return; }
  echo "================ fc_action=$A ================"
  $PY tools/la_smoke.py --run_dir "$ROOT/$TAG" --seqs $SEQS --tag "$A" --fc_metrics 2>/dev/null \
    | grep -E 'seq +gtfail|car6_4|wakeboard10|car17|boat8|car7|car2_s|car5|uav3|car1_3|mean ΔAUC|mean FCR|mean FCD'
}
for A in freeze_only widen hold_lastgood; do run "$A"; done
echo "FC_CMP_DONE"
