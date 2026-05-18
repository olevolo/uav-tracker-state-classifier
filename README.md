# UAV Entropy-Guided Tracker

![ci](https://img.shields.io/badge/ci-pending-lightgrey) ![license](https://img.shields.io/badge/license-pending-lightgrey)

Research codebase implementing Oleksiuk & Velhosh (2026), *"Entropy-Guided Tracker Switching Method for Unmanned Aerial Vehicle Real-Time Tracking"* (*Electronics and Information Technologies*, vol. 33).

## Quickstart

```bash
git clone <repo-url> uav-entropy-tracker && cd uav-entropy-tracker
make setup
uav-tracker doctor
uav-tracker list-plugins
uav-tracker evaluate --config configs/experiments/paper_entropy_hybrid.yaml --limit 5
```

## What

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
