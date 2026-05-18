# UAV Tracker — ML-Driven Detection & Scene Classification

A modular UAV object tracker that uses a **learned scene classifier** to route each frame to the appropriate tracker tier. Replaces hand-tuned entropy thresholds with neural scene classification.

```
Frame → SceneClassifier (MobileNetV3) → Scene Class
                                              │
                       ┌──────────────────────┼──────────────────────┐
                     CLEAR              CHALLENGING             RECOVERY
                       │                     │                     │
                 KCF Henriques          OSTrack-256            YOLOv8n
                   (tier 0)               (tier 2)             (tier 3)
```

## Scene Classes

| Class | Condition | Default Tier |
|-------|-----------|--------------|
| `CLEAR` | Smooth motion, confidence > 0.8 | 0 — KCF |
| `MODERATE` | Mild entropy / H̄ 0.3–0.5 | 1 — MobileTrack |
| `CHALLENGING` | Fast motion / cluttered background | 2 — OSTrack-256 |
| `RISK_LOSS` | Target about to be lost (IoU dropping) | 2 + pre-arm detection |
| `RECOVERY` | Target lost, reacquiring | 3 — YOLOv8n |
| `LOW_RES` | Small target (bbox < 400 px²) | 2 — small crop |

## ML Modules

| Module | Type | Description |
|--------|------|-------------|
| `MobileNetV3TinyClassifier` | Supervised CNN | 6-class scene classifier, every 5 frames |
| `MLPDifficultyPredictor` | Supervised regression | Predicts IoU drop over next 10 frames |
| `CosineAppearanceMemory` | Self-learning | Appearance templates with exponential forgetting |
| `OnlineLSTMMotionPredictor` | Self-learning | Motion pattern learning via online SGD |
| `DefaultModelWarmer` | Infrastructure | Pre-loads + JIT-warms all tiers before frame 0 |

## Quick Start

```bash
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt

uav-tracker list-plugins
uav-tracker evaluate --config configs/experiments/v2_full_ml.yaml --dataset synthetic
```

## Training the Scene Classifier

```bash
# 1. Generate per-frame difficulty labels (runs KCF simulation on UAV123)
python scripts/generate_ml_labels.py \
  --dataset-root $UAV_DATA_ROOT/uav123 \
  --output $UAV_DATA_ROOT/uav123_labels.npz

# 2. Train (30 epochs, AdamW, cosine LR, mixed precision)
python scripts/train_scene_classifier.py \
  --config configs/training/scene_classifier_uav123.yaml
```

## Benchmark (Stub Backbones)

15 UAV123 sequences, 500 frames. Deep trackers use random-init stubs — load real weights for production numbers.

| Method | AUC | Pr@20 | FPS |
|--------|-----|-------|-----|
| KCF Henriques (tier 0) | 0.293 | 0.422 | 219 |
| OSTrack-256 stub | 0.219 | 0.281 | 160 |
| STARK-S50 stub | 0.203 | 0.249 | 123 |
| hybrid_v2_ml (full pipeline) | 0.284 | 0.440 | 75 |

Expected with real OSTrack-256 weights: AUC ~0.65+ (see `docs/adr/0013-heavy-tracker-tier2.md`).

## Layout

```
src/uav_tracker/
├── ml/                       # All ML modules
│   ├── scene_classifier/     # CNN + feature extractor + online adaptor
│   ├── difficulty_predictor/ # MLP regression
│   ├── appearance_memory/    # Cosine similarity store
│   ├── motion_predictor/     # Online LSTM
│   └── warmer/               # Cold-start elimination
├── trackers/transformer/     # OSTrack-256, STARK-S50 stubs
├── trackers/kcf_*.py         # KCF baselines (tier 0)
├── schedulers/ml_scene_scheduler.py  # Scene class → tier routing
├── training/                 # Label generator, augmentation
└── datasets/uav123_ml.py     # Train/val/test splits (98/12/8)
```

## ADRs

0009 scene taxonomy · 0010 ML protocols · 0011 UAV123 split · 0012 label generation · 0013 heavy trackers · 0014 model warmer · 0015 online adaptation bounds


A hybrid UAV object tracker that switches between a lightweight KCF+Kalman tracker and a heavier Siamese deep tracker (SiamFC by default, MobileTrack variant optional). The switch is driven by the Shannon entropy of the target's recent motion-orientation distribution after camera ego-motion is removed: low entropy → predictable motion → KCF suffices; high entropy → engage the deep tracker before KCF drifts. A hysteresis state machine with confirmation and cooldown windows prevents flapping.

## Why

The paper introduces a principled, cheap switching signal (residual-flow orientation entropy) that preserves most of the deep tracker's accuracy at a fraction of the compute. This repo is a faithful re-implementation plus a deliberate set of improvements where the paper is weak:

- Circular-statistics alternative to Shannon entropy for small-sample stability.
- Adaptive thresholds as a plugin (paper thresholds are dataset-calibrated).
- Three-level global-motion fallback (RANSAC → LMedS → reuse-prior) — strict improvement, on by default.
- Optional third tier: a YOLO-based detector for recovery from complete target loss (paper flags this as future work).
- Calibrated per-frame confidence + explicit `locked`/`uncertain`/`lost` status.

Every paper deviation ships as its own registered plugin and is compared empirically in ablations; nothing is removed.

## Architecture

Modular by design: `Tracker`, `Detector`, `SwitchSignal`, `Scheduler` are Protocols backed by registries. Adding a new tracker is one file + `@TRACKERS.register(...)` + YAML — no edits to the runner, scheduler, or evaluator. See `PLAN.md` §3 (ASCII architecture diagram) and §4 (plugin spec) for the full picture.

## Development

Three-agent workflow (Architect / Engineer / DevOps). See `AGENTS.md` for scope, governance, and git rules; `PLAN.md` for the full plan, phases, experiments, and fidelity contract; `docs/adr/` for decision records.

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
