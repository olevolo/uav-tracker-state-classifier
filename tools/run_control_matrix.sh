#!/bin/bash
# CONTROL MATRIX (Stage 1) — LA + FC control on the top-hard scenes of each
# (tracker, dataset), tracker-agnostic levers that work via override_search_center
# (now wired on AVTrack/ORTrack too). Baseline = existing *_r3_passive runs.
# EXPLORATORY offline SOT. Levers:
#   LA: --policy_gated_redetect --gate_preset csc_head (forecast head, agnostic) + motion_bridge
#   FC: --policy_fc_control --fc_action hold_lastgood (snap search to last-confirmed)
# Both: --gated_freeze --recovery_update_window 0 (clean: frozen template except post-action).
set -u
cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
CKPT=outputs/csc_training/sglatrack_r3_fcw3_w32_tcn32_stage2/checkpoint_best.pth
ROOT=outputs/eval_control_matrix
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
mkdir -p "$ROOT" outputs/_logs
LOG=outputs/_logs/control_matrix.log; : > "$LOG"
say(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

cal_for(){ case "$1" in sglatrack) echo sglatrack_all_v2;; avtrack) echo avtrack_aerial_v2;; ortrack) echo ortrack_aerial_v2;; esac; }

# HARD10 sets per tracker/dataset (top FC+LA frames, from passive states)
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
  local cal=$(cal_for "$trk"); local seqs=$(hard_for "$trk" "$ds")
  local out="$ROOT/$trk/$ds"; rm -rf "$out/$tag"; mkdir -p "$out"
  say ">>> [$trk/$ds/$tag] seqs=$(echo $seqs|wc -w|tr -d ' ')  $*"
  $PY -u tools/run_with_csc.py --tracker "$trk" --dataset "$ds" --split test \
    --csc_checkpoint "$CKPT" --calibration_prefix "$cal" --device cpu \
    --output_dir "$out" --run_tag "$tag" --include_sequences $seqs \
    --csc_mode control --gated_freeze --recovery_update_window 0 "$@" \
    >>"$LOG" 2>&1 || { say "  FAIL $trk/$ds/$tag"; return 0; }
  say "  done $trk/$ds/$tag"; }

say "===== CONTROL MATRIX Stage 1 (LA motion_bridge + FC hold_lastgood) ====="
for trk in sglatrack avtrack ortrack; do
 for ds in uav123 uav123_10fps; do
   run "$trk" "$ds" la_mb \
     --policy_gated_redetect --gate_preset csc_head --gate_lostaware 0.90 \
     --redetect_action motion_bridge --redetect_arm_frames 3 --bridge_max_frames 30 --bridge_vel_ema 0.5
   run "$trk" "$ds" fc_hold \
     --policy_fc_control --fc_action hold_lastgood --fc_streak_frames 2 --fc_gate_vote_k 0
 done
done
say "===== Stage 1 DONE ====="
echo "CONTROL_MATRIX_S1_DONE" >> "$LOG"