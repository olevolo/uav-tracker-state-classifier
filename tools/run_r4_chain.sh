#!/bin/bash
# R4 full training chain: stage1 (4-state classifier, v3 features) -> stage2 (forecast heads).
cd /Users/voleksiuk/projects/uav-tracker-detector
S1OUT=outputs/csc_training/sglatrack_r4_v3feat_tcn16_stage1
S2OUT=outputs/csc_training/sglatrack_r4_v3feat_tcn16_stage2

echo "=== R4 STAGE1 START $(date) ==="
.venv/bin/python -u tools/train_csc.py --config configs/csc/csc_tcn16_r4_v3feat_stage1.yaml
S1=$?
if [ $S1 -ne 0 ]; then echo "!!! R4 STAGE1 FAILED rc=$S1 $(date)"; exit $S1; fi
if [ ! -f "$S1OUT/checkpoint_best.pth" ]; then echo "!!! R4 STAGE1 produced no checkpoint_best.pth"; exit 2; fi
echo "=== R4 STAGE1 DONE $(date) — best ckpt present ==="

echo "=== R4 STAGE2 START $(date) ==="
.venv/bin/python -u tools/train_csc.py --config configs/csc/csc_tcn16_r4_v3feat_stage2.yaml
S2=$?
if [ $S2 -ne 0 ]; then echo "!!! R4 STAGE2 FAILED rc=$S2 $(date)"; exit $S2; fi
echo "=== R4 CHAIN DONE $(date) — stage2 ckpt: $S2OUT/checkpoint_best.pth ==="
