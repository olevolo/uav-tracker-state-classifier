#!/bin/bash
# Bridge DURATION sweep (FROZEN, BRC=0.5). Hypothesis: a shorter motion-bridge
# limits how far a wrong extrapolation drifts on abrupt/erratic losses
# (car11 -0.29, bird1_1/bird1_3/person19_3) while still catching fast smooth
# recoveries (group3_2 +0.49, person14_1 +0.30). Guard uav6, easy car6_2 ~0.
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
SEQS="group3_2 person14_1 uav3 bird1_1 bird1_3 car11 person19_3 uav6 car6_2"
for BMF in 6 12; do
  TPL="--no_runner_template_update" SEQS="$SEQS" BRC="0.5" BMF="$BMF" \
    bash tools/run_full_combo_uav123.sh "frozen_bmf${BMF}"
done
echo "BMF_SWEEP_DONE"
