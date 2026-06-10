#!/bin/bash
# R4 UAV123 passive eval — fire ONCE the R4 stage2 checkpoint exists and training
# is no longer running. Re-runs the SGLATrack tracker live (CPU-heavy) so it must
# only run after R4 training has freed the CPU. Safe to re-run (resumes).
cd /Users/voleksiuk/projects/uav-tracker-detector
CKPT=outputs/csc_training/sglatrack_r4_v3feat_tcn16_stage2/checkpoint_best.pth
RUN=outputs/eval/sglatrack/uav123/test/sglatrack_r4_passive

if [ ! -f "$CKPT" ]; then
  echo "R4 stage2 checkpoint not present yet: $CKPT — skipping eval."
  exit 1
fi

echo "=== R4 UAV123 passive eval START $(date) ==="
# Live passive diagnosis. --calibration_prefix sglatrack_all_v2 normalizes
# confidence/apce/psr; the 7 V3 fields ride in RAW telemetry via step(extra=tel).
.venv/bin/python -u tools/run_with_csc.py \
  --tracker sglatrack --dataset uav123 --split test \
  --csc_checkpoint "$CKPT" --csc_mode passive --device cpu \
  --output_dir outputs/eval/sglatrack/uav123/test --run_tag sglatrack_r4_passive \
  --calibration_prefix sglatrack_all_v2
RC=$?
if [ $RC -ne 0 ]; then echo "!!! R4 eval run_with_csc failed rc=$RC $(date)"; exit $RC; fi

echo "=== R4 paper_metrics $(date) ==="
.venv/bin/python -u tools/compute_paper_metrics.py \
  --tracker sglatrack --dataset uav123 --split test \
  --predictions_dir "$RUN/predictions" --states_dir "$RUN/states" --telemetry_dir "$RUN/telemetry" \
  --labels_dir outputs/eval/sglatrack/uav123/test/labels_v3/uav123/test/labels_per_sequence \
  --confidence_calib outputs/calibration/sglatrack_all_v2_confidence.json \
  --output_dir "$RUN/paper_metrics" --recovery_k 30

echo "=== R4 EVAL DONE $(date) — compare paper_metrics vs R2/R3; update FINAL_REPORT.md §4 ==="
