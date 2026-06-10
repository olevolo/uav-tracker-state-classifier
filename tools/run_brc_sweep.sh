#!/bin/bash
# Bridge motion-smoothness cap sweep (FROZEN CSC levers). Find a --bridge_max_resid_ratio
# that gates OUT erratic-motion bridges (bird1_1/bird1_3/car11/person19_3 -0.13..-0.29,
# where velocity extrapolation goes wrong) while KEEPING smooth-motion wins
# (group3_2 +0.49, person14_1 +0.30). Guard uav6, easy car6_2 must stay ~0.
set -u; cd /Users/voleksiuk/projects/uav-tracker-detector
SEQS="group3_2 person14_1 uav3 bird1_1 bird1_3 car11 person19_3 uav6 car6_2"
for BRC in 0.3 0.5 1.0; do
  TPL="--no_runner_template_update" SEQS="$SEQS" BRC="$BRC" \
    bash tools/run_full_combo_uav123.sh "frozen_brc${BRC}"
done
echo "BRC_SWEEP_DONE"
