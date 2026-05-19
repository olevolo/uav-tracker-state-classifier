# UAV Tracker — SALT v3

SALT (State-Adaptive Lightweight Tracker) wraps **SGLATrack** (DeiT-tiny) with a Target State Assessor that switches between full-compute and CE-pruned inference, and triggers YOLO-based recovery on target loss.

```
Frame → SGLATracker (DeiT-tiny, 0.9 GFLOPs trunk / 1.27 GFLOPs full)
              │
              ▼
    TargetStateAssessor (APCE + PSR + VelocityDrift)
              │
         ┌────┴────────────────────────────┐
     CONFIRMED                         OCCLUDED / LOST
    (CE kr=0.50)                       (full compute)
         │                                  │
         │                           YOLO26m VisDrone
         │                           4-guard recovery
         └────────────────────────────────┘
```

## Benchmark Results

### UAV123 (6-sequence fast bench)

| Method | MEAN AUC | FPS |
|--------|:--------:|:---:|
| SGLATrack baseline | 0.541 | ~78 |
| **SALT v3 (CE kr=0.50)** | **0.616** | **56** |
| Δ | **+0.075** | — |

Key wins: uav2 AUC 0.136→0.507 (+0.371 via recovery), truck1 0.721→0.778.

### VisDrone-SOT test-dev (35 sequences)

| Method | AUC | Pr@20 | FPS |
|--------|:---:|:-----:|:---:|
| SGLATrack | 0.672 | 0.859 | 78 |
| SALT v3 | 0.673 | 0.858 | 40 |

## Architecture

| Component | Description |
|-----------|-------------|
| SGLATrack (DeiT-tiny) | Primary tracker; 0.9 GFLOPs trunk, 1.27 GFLOPs full |
| CE token pruning (kr=0.50) | Active on CONFIRMED state; prunes background search tokens |
| APCECalibrator | Adaptive APCE thresholds: LOST<20, CONFIRMED≥80 |
| VelocityDriftMonitor | Detects false-CONFIRMED (drift+PSR decay → DISTRACTOR_RISK) |
| Supervised MLP head | 92.5% val_acc TSA classifier trained on all 123 UAV123 seqs |
| YOLO26m VisDrone | Recovery detector (55% mAP@0.5); 4-guard pipeline |
| CosineAppearanceMemory | 32×32 reference embedding for recovery gating |

## Quick Start

```bash
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt

# Benchmark on UAV123
PYTHONPATH=src .venv/bin/python scripts/fast_bench.py --mode salt --dataset uav123

# Benchmark on VisDrone SOT
PYTHONPATH=src .venv/bin/python scripts/fast_bench.py --mode salt --dataset visdrone_sot

# Compare pruning strategies
PYTHONPATH=src .venv/bin/python scripts/fast_bench.py --mode salt --dataset uav123 --pruning ce
PYTHONPATH=src .venv/bin/python scripts/fast_bench.py --mode salt --dataset uav123 --pruning ctem

# Train supervised TSA head
PYTHONPATH=src .venv/bin/python scripts/train_tsa_classifier.py --mode sglatrack
```

## TSA State Machine

| State | APCE | Compute routing | Recovery |
|-------|:----:|:---------------:|:--------:|
| CONFIRMED | ≥ 80 | CE kr=0.50 | — |
| OCCLUDED | 20–80 | full depth | escalation after 25f |
| LOST | < 20 | full depth | YOLO26m after 5 LOST |
| DISTRACTOR_RISK | CONFIRMED + VelocityDrift | full depth | — |

## CE Token Pruning

CE scoring uses Q·K^T cross-attention to score search tokens. At kr=0.50 on CONFIRMED frames:
- uav2: +0.015, bike2: +0.005, car7: +0.042 (vs no pruning)
- Net FLOP reduction: ~0.9% (overhead exceeds savings above kr=0.57; CTEM always cheaper to score but collapses small-target sequences)

## Layout

```
src/uav_tracker/
├── salt_runner.py                  # Main SALT pipeline
├── trackers/sglatrack.py           # SGLATrack + CE routing
├── ml/tsa/                         # Target State Assessor
│   ├── target_state_assessor.py    # APCECalibrator, VelocityDriftMonitor, head
│   └── velocity_drift.py           # VelocityDriftMonitor
├── ml/appearance_memory/           # CosineAppearanceMemory (32×32)
├── detectors/visdrone_yolo26m.py   # YOLO26m VisDrone recovery detector
└── datasets/
    ├── uav123.py                   # UAV123 (123 seqs)
    ├── visdrone_sot.py             # VisDrone-SOT test-dev (35 seqs)
    └── dtb70.py                    # DTB70 (70 seqs, awaiting download)
scripts/
├── fast_bench.py                   # Fast benchmark (6 seqs ~2-3 min)
├── train_tsa_classifier.py         # Supervised TSA head training
└── run_comparison_benchmark.py     # Full comparison benchmark
configs/experiments/
├── salt.yaml                       # Active SALT config
└── v2_full_ml.yaml                 # Legacy ML pipeline config
```

## Datasets

See [DATASETS.md](DATASETS.md) for dataset paths, env vars, and benchmark commands.

## Citation

```bibtex
@article{oleksiuk2026entropyguided,
  author  = {Oleksiuk, V. and Velhosh, S.},
  title   = {Entropy-Guided Tracker Switching Method for Unmanned Aerial Vehicle Real-Time Tracking},
  journal = {Electronics and Information Technologies},
  volume  = {33},
  year    = {2026}
}
```
