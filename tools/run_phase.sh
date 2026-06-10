#!/bin/bash
# Single-phase runner — call as: bash run_phase.sh <tracker> <phase> [dataset] [seq1 seq2 ...]
# Runs ONE mode on uav123 (or specified dataset). Resumable via state file mechanism.
# dataset default: uav123. If sequences listed after dataset, only those are run.
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
TRACKER="$1"; PHASE="$2"
DATASET="${3:-uav123}"; shift 3 2>/dev/null || shift 2
SEQS="$*"

PY=.venv/bin/python
CKPT=outputs/csc_training/csc_prod/checkpoint_best.pth
FC="--fc_action hold_lastgood --fc_streak_frames 2 --fc_gate_vote_k 2 --no_runner_template_update"

case "$TRACKER" in
  sglatrack)
    CALIB=sglatrack_aerial_v2
    # LA: scoremap2+sgla_redetect+min_sim  /  LA_MB: scoremap2+motion_bridge (like full_combo)
    GATE_RD="--gate_preset scoremap2 --gate_top2ratio 0.45 --gate_entropy 4.3 \
      --redetect_action sgla_redetect --sgla_redetect_factors 8,12,16 \
      --sgla_redetect_min_sim 0.6 --sgla_redetect_min_apce 0 \
      --gated_freeze --no_runner_template_update"
    GATE_MB="--gate_preset combined --gate_vote_k 5 \
      --redetect_action motion_bridge --redetect_arm_frames 3 \
      --bridge_max_frames 30 --bridge_vel_ema 0.5 \
      --gated_freeze --no_runner_template_update"
    EVAL_RD=eval11_sgla
    EVAL_MB=eval13_sgla   # motion_bridge variant (same as eval7 full_combo but clean dir)
    ;;
  avtrack)
    CALIB=avtrack_aerial_v2
    GATE_RD="--gate_preset csc_head --gate_lostaware 0.90 \
      --redetect_action sgla_redetect --sgla_redetect_factors 8,12,16 \
      --sgla_redetect_min_sim 0.6 --sgla_redetect_min_apce 0 \
      --gated_freeze --no_runner_template_update"
    GATE_MB="--gate_preset csc_head --gate_lostaware 0.90 \
      --redetect_action motion_bridge --redetect_arm_frames 3 \
      --bridge_max_frames 30 --bridge_vel_ema 0.5 \
      --gated_freeze --no_runner_template_update"
    EVAL_RD=eval12_avtrack
    EVAL_MB=eval13_avtrack
    ;;
  ortrack)
    CALIB=ortrack_aerial_v2
    GATE_RD="--gate_preset csc_head --gate_lostaware 0.95 \
      --redetect_action sgla_redetect --sgla_redetect_factors 8,12,16 \
      --sgla_redetect_min_sim 0.6 --sgla_redetect_min_apce 0 \
      --gated_freeze --no_runner_template_update"
    GATE_MB="--gate_preset csc_head --gate_lostaware 0.90 \
      --redetect_action motion_bridge --redetect_arm_frames 3 \
      --bridge_max_frames 30 --bridge_vel_ema 0.5 \
      --gated_freeze --no_runner_template_update"
    EVAL_RD=eval12_ortrack
    EVAL_MB=eval13_ortrack
    ;;
  *) echo "Unknown tracker: $TRACKER"; exit 1 ;;
esac

# Route phase to correct eval dir + gate variant
case "$PHASE" in
  passive)   GATE=""; EVAL_DIR="$EVAL_RD"; FLAGS="--csc_mode passive" ;;
  la_only)   GATE="$GATE_RD"; EVAL_DIR="$EVAL_RD"; FLAGS="--csc_mode control --policy_gated_redetect $GATE_RD" ;;
  fc_only)   GATE=""; EVAL_DIR="$EVAL_RD"; FLAGS="--csc_mode control --policy_fc_control $FC" ;;
  combo)     GATE="$GATE_RD"; EVAL_DIR="$EVAL_RD"; FLAGS="--csc_mode control --policy_gated_redetect $GATE_RD --policy_fc_control $FC" ;;
  # motion_bridge variants (balanced, risk_gate protected)
  la_mb)     GATE=""; EVAL_DIR="$EVAL_MB"; FLAGS="--csc_mode control --policy_gated_redetect $GATE_MB" ;;
  fc_mb)     GATE=""; EVAL_DIR="$EVAL_MB"; FLAGS="--csc_mode control --policy_fc_control $FC" ;;
  combo_mb)  GATE=""; EVAL_DIR="$EVAL_MB"; FLAGS="--csc_mode control --policy_gated_redetect $GATE_MB --policy_fc_control $FC --control_risk_gate" ;;
  *) echo "Unknown phase: $PHASE"; exit 1 ;;
esac

[[ "$DATASET" == "uav123" ]] && DS_SUFFIX="" || DS_SUFFIX="_${DATASET}"
ROOT=outputs/${EVAL_DIR}${DS_SUFFIX}/csc/${TRACKER}/${DATASET}/test
LOG=outputs/_logs/run_${TRACKER}_${DATASET}_${PHASE}.log
mkdir -p "$(dirname "$LOG")"
say(){ echo "[$(date '+%H:%M:%S')] [$TRACKER/$DATASET/$PHASE] $*" | tee -a "$LOG"; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4

SEQ_FLAG=""; [[ -n "$SEQS" ]] && SEQ_FLAG="--include_sequences $SEQS"

say "START $TRACKER/$DATASET/$PHASE"
$PY -u tools/run_with_csc.py \
  --tracker "$TRACKER" --dataset "$DATASET" --split test \
  --csc_checkpoint "$CKPT" --calibration_prefix "$CALIB" --device cpu \
  --output_dir "$ROOT" --run_tag "$PHASE" \
  $FLAGS $SEQ_FLAG >> "$LOG" 2>&1
STATUS=$?
say "EXIT $STATUS"
exit $STATUS
