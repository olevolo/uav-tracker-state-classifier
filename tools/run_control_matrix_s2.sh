#!/bin/bash
# CONTROL MATRIX (Stage 2) — redetect-based control on the same hard scenes:
#   LA: --policy_gated_redetect --gate_preset csc_head + sgla_redetect (now wired on AV/OR)
#   FC: --policy_fc_challenge --fc_challenge_switch_mode association (the FC improvement:
#       track the redetect candidate nearest the last-good trajectory; switch+abort/rollback)
# Waits for Stage 1 to finish first (avoid CPU contention). Baseline = *_r3_passive.
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval_control_matrix
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
mkdir -p "$ROOT" outputs/_logs
S1LOG=outputs/_logs/control_matrix.log
LOG=outputs/_logs/control_matrix_s2.log; : > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Wait for Stage 1 done marker (max ~90 min).
say "waiting for Stage 1 to finish..."
for i in $(seq 1 180); do grep -q CONTROL_MATRIX_S1_DONE "$S1LOG" 2>/dev/null && break; sleep 30; done
rm -rf "$ROOT/_smoke_or" 2>/dev/null
say "Stage 1 done — starting Stage 2"

cal_for(){ case "$1" in sglatrack) echo sglatrack_all_v2;; avtrack) echo avtrack_aerial_v2;; ortrack) echo ortrack_aerial_v2;; esac; }
hard_for(){
 case "$1/$2" in
  sglatrack/uav123)       echo "car9 car1_s person19_3 group2_1 group3_2 bird1_3 person1_s uav1_3 car1_3 car12";;
  sglatrack/uav123_10fps) echo "car9 car1_s person19_3 group3_2 uav1_3 person1_s group2_1 person14_1 car1_3 car3_s";;
  avtrack/uav123)         echo "person19_3 group3_2 person16 car9 uav1_3 bird1_2 bird1_3 person1_s car1_3 car12";;
  avtrack/uav123_10fps)   echo "person19_3 uav1_3 bird1_3 car1_3 group2_1 person16 bird1_2 car12 uav1_1 car17";;
  ortrack/uav123)         echo "person19_3 group3_2 person10 bird1_3 bird1_2 person1_s uav1_3 car7 car12 person14_1";;
  ortrack/uav123_10fps)   echo "person19_3 uav1_3 person19_2 group3_2 person1_s person10 bird1_3 group2_1 wakeboard6 bird1_2";;
 esac
}
run(){ local trk="$1" ds="$2" tag="$3"; shift 3
  local cal=$(cal_for "$trk"); local seqs=$(hard_for "$trk" "$ds"); local out="$ROOT/$trk/$ds"
  rm -rf "$out/$tag"; mkdir -p "$out"
  say ">>> [$trk/$ds/$tag] $*"
  $PY -u tools/run_with_csc.py --tracker "$trk" --dataset "$ds" --split test \
    --csc_checkpoint "$CKPT" --calibration_prefix "$cal" --device cpu \
    --output_dir "$out" --run_tag "$tag" --include_sequences $seqs \
    --csc_mode control --gated_freeze "$@" >>"$LOG" 2>&1 || { say "  FAIL $trk/$ds/$tag"; return 0; }
  say "  done $trk/$ds/$tag"; }

say "===== CONTROL MATRIX Stage 2 (LA sgla_redetect + FC challenge association) ====="
for trk in sglatrack avtrack ortrack; do
 for ds in uav123 uav123_10fps; do
   run "$trk" "$ds" la_sgla --recovery_update_window 0 \
     --policy_gated_redetect --gate_preset csc_head --gate_lostaware 0.90 \
     --redetect_action sgla_redetect --redetect_arm_frames 3 \
     --sgla_redetect_factors 8,12,16 --sgla_redetect_grid 0 --sgla_redetect_max_candidates 3
   run "$trk" "$ds" fc_chal --recovery_update_window 5 \
     --policy_fc_challenge --fc_challenge_switch_mode association --fc_challenge_streak 2 \
     --fc_challenge_confirm 3 --fc_challenge_abort_window 5 --fc_challenge_topk 6 \
     --fc_challenge_assoc_gate 2.0 --sgla_redetect_factors 8,12,16 --sgla_redetect_grid 2
 done
done
say "===== Stage 2 DONE ====="
echo "CONTROL_MATRIX_S2_DONE" >> "$LOG"