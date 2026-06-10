#!/bin/bash
# V4 re-extraction: re-run SGLATrack passive saving FULL telemetry (response + APPEARANCE
# fields last_cosine_sim/initial_template_sim/appearance_drift, which the old baselines lack)
# over the TRAIN set -> outputs/baselines_v4/. CPU (measured 76 vs 30 FPS MPS for this per-frame
# small-ViT workload; GPU is for TRAINING, not extraction). Usage: bash run_v4_reextract.sh <group>
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
PY=.venv/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
OUT=outputs/baselines_v4
mkdir -p outputs/_logs "$OUT"
GROUP="${1:-rest}"
run(){ local ds=$1; local sp=$2; local LG=outputs/_logs/reextract_${ds}_${sp}.log
  echo "[$(date '+%H:%M:%S')] >>> $ds/$sp"
  $PY tools/run_baseline.py --tracker sglatrack --dataset "$ds" --split "$sp" \
    --device cpu --output_dir "$OUT" > "$LG" 2>&1 && echo "[$(date '+%H:%M:%S')] DONE $ds/$sp ($(grep -oE 'mean FPS=[0-9.]+' "$LG"|tail -1))" || echo "FAIL $ds/$sp"; }
if [ "$GROUP" = lasot ]; then
  run lasot train
  echo "REEXTRACT_LASOT_DONE"
else
  run got10k val
  run dtb70 test
  run visdrone_sot test
  run uavdt_sot test
  echo "REEXTRACT_REST_DONE"
fi
