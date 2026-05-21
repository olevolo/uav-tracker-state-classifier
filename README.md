# UAV Tracker — SALT v3 + SALT-RD

SALT (State-Adaptive Lightweight Tracker) wraps **SGLATrack** (DeiT-tiny) with a
compute-routing layer that switches between CE-pruned and full-depth inference, and
triggers YOLO-based recovery on target loss.

SALT-RD is the learned trust controller (no TSA) that gates template updates and
reinit decisions alongside the tracker.

```
Frame → SGLATracker (DeiT-tiny, 0.9 GFLOPs trunk / 1.27 GFLOPs full)
              │
              ▼
    SALT-RD controller (no TSA)
    EvidenceExtractor → SALTRDController (GRU)
    28-dim telemetry → p_false_confirmed, p_failure, p_recoverable
              │
         ┌────┴────────────────────────────┐
    TRUSTED / LOW_EVIDENCE             FALSE_CONFIRMED_RISK
    (CE kr=0.50 on CONFIRMED)          (block template update)
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
| CE token pruning (kr=0.50) | Active on CONFIRMED frames; prunes background search tokens |
| EvidenceExtractor | Extracts 28-dim telemetry from SGLATrack per frame |
| SALTRDController (GRU) | Learned trust controller; no TSA; gates template updates and reinit |
| APCECalibrator | Adaptive APCE thresholds: LOST<20, CONFIRMED≥80 |
| VelocityDriftMonitor | Detects false-CONFIRMED via drift+PSR decay |
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

# Train supervised head
PYTHONPATH=src .venv/bin/python scripts/train_tsa_classifier.py --mode sglatrack
```

## SALT-RD Controller

Production checkpoint: `saltr/checkpoints/production_no_flow/saltrd_best.pt`

| Property | Value |
|----------|-------|
| Val fc AUROC | 0.885 |
| Diag fc AUROC | 0.598 |
| Stage gate | wrir=0, msu=0.081, coverage=65.7% |

Oracle audit: reinit policy +0.083 hard AUC → BUILD policy confirmed.

See `saltr/README_PROD.md` for commands and `RESULTS.md` for full benchmark results.

## CE Token Pruning

CE scoring uses Q·K^T cross-attention to score search tokens. At kr=0.50 on CONFIRMED frames:
- uav2: +0.015, bike2: +0.005, car7: +0.042 (vs no pruning)
- Net FLOP reduction: ~0.9% (overhead exceeds savings above kr=0.57; CTEM always cheaper to score but collapses small-target sequences)

## Layout

```
src/uav_tracker/
├── salt_runner.py                  # Main SALT pipeline
├── trackers/sglatrack.py           # SGLATrack + CE routing
├── ml/tsa/                         # APCE calibration, VelocityDriftMonitor (frozen baseline)
│   ├── target_state_assessor.py    # APCECalibrator, VelocityDriftMonitor
│   └── velocity_drift.py           # VelocityDriftMonitor
├── ml/appearance_memory/           # CosineAppearanceMemory (32×32)
├── detectors/visdrone_yolo26m.py   # YOLO26m VisDrone recovery detector
└── datasets/
    ├── uav123.py                   # UAV123 (123 seqs)
    ├── visdrone_sot.py             # VisDrone-SOT test-dev (35 seqs)
    └── dtb70.py                    # DTB70 (70 seqs)
saltr/
├── src/salt_r/                     # SALT-RD implementation
├── checkpoints/production_no_flow/ # Production checkpoint (val fc AUROC 0.885)
└── README_PROD.md                  # SALT-RD production reference
scripts/
├── fast_bench.py                   # Fast benchmark (6 seqs ~2-3 min)
├── train_tsa_classifier.py         # Supervised head training
└── run_comparison_benchmark.py     # Full comparison benchmark
configs/experiments/
└── salt.yaml                       # Active SALT config
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
