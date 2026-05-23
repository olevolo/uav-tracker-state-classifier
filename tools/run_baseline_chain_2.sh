#!/usr/bin/env bash
# Second-wave baseline chain: remaining LaSOT runs + post-pipeline for all.
# Run AFTER run_baseline_chain_1.sh completes (check: ls outputs/baselines/ostrack/uav123/test/manifest.json).
# Usage: nohup bash tools/run_baseline_chain_2.sh > /tmp/csc_chain2.log 2>&1 &
set -e
cd "$(dirname "$0")/.."

echo "[$(date '+%H:%M:%S')] === Chain 2 start ==="

# Post-baseline pipeline for trackers from chain 1 (if not already done)
for t in sglatrack ortrack; do
  if [ -f "outputs/baselines/$t/lasot/train/manifest.json" ]; then
    echo "[$(date '+%H:%M:%S')] Running post-pipeline for $t..." >&2
    bash tools/run_post_baseline_pipeline.sh "$t"
  else
    echo "[$(date '+%H:%M:%S')] WARN: $t/lasot baseline not found, skipping post-pipeline" >&2
  fi
done

# LaSOT baselines for slow trackers (better on GPU, but queueing for CPU fallback)
echo "[$(date '+%H:%M:%S')] === AVTrack on LaSOT ===" && \
make baseline TRACKER=avtrack DATASET=lasot SPLIT=train DEVICE=cpu && \
bash tools/run_post_baseline_pipeline.sh avtrack && \

echo "[$(date '+%H:%M:%S')] === EVPTrack on LaSOT ===" && \
make baseline TRACKER=evptrack DATASET=lasot SPLIT=train DEVICE=cpu && \
bash tools/run_post_baseline_pipeline.sh evptrack && \

echo "[$(date '+%H:%M:%S')] === OSTrack on LaSOT ===" && \
make baseline TRACKER=ostrack DATASET=lasot SPLIT=train DEVICE=cpu && \
bash tools/run_post_baseline_pipeline.sh ostrack && \

echo "[$(date '+%H:%M:%S')] === Chain 2 DONE ==="
echo ""
echo "Next steps:"
echo "  1. Run CSC passive on UAV123 for each tracker: make with-csc TRACKER=X ..."
echo "  2. Compute paper Table 4: make diagnose + gate for each CSC model"
echo "  3. Final: run_paper_tables.py"
