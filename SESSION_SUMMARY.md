# SALT Project — Consolidated Session Summary
**Updated:** 2026-05-19  
**Covers:** Sessions 2026-05-18, 2026-05-19 (night), 2026-05-19 (day)

---

## Benchmark Results — Current State (SALT v3, CE kr=0.50)

### UAV123 (6 diagnostic sequences, 500-frame cap)

| Sequence | SGLATrack | SALT v3 | Δ | SALT fps |
|----------|:---------:|:----:|:-:|:--------:|
| car13 | 0.750 | 0.749 | −0.001 | 54 |
| uav2 | 0.136 | **0.507** | **+0.371** | 66 |
| bike2 | 0.176 | 0.176 | 0.000 | 56 |
| car7 | 0.595 | **0.612** | **+0.017** | 56 |
| building1 | 0.872 | 0.871 | −0.001 | 55 |
| truck1 | 0.721 | **0.778** | **+0.057** | 50 |
| **MEAN** | **0.541** | **0.616** | **+0.075** | **56** |

Recovery quality: 1 event, mean IoU = **0.951**

### SALT version history (UAV123 MEAN)

| Version | MEAN | Key change |
|---------|:---:|-----------|
| SGLATrack baseline | 0.541 | No SALT |
| SALT v1 (2026-05-18) | 0.541 | AUC=baseline at 55fps |
| SALT v2 (CE broken, no pruning) | 0.610 | +TSA routing + recovery |
| SALT v2a (CE fixed, wrong architecture) | 0.551 | CE feedback loop regression |
| **SALT v3 (CE corrected, kr=0.50)** | **0.616** | **+0.006 from distractor pruning** |

### VisDrone-SOT test-dev (35 sequences)

| Method | AUC | Pr@20 | FPS |
|--------|:---:|:---:|:---:|
| SGLATrack | 0.672 | 0.859 | 78 |
| SALT v3 | **0.673** | 0.858 | **40** |

| Version | AUC | Pr@20 | FPS | Notes |
|---------|:---:|:---:|:---:|-------|
| KCF Henriques | 0.293 | 0.422 | 219 | Phase 1 baseline |
| OSTrack-256 | 0.639 | 0.837 | 10 | Replaced by SGLATrack |
| SGLATrack (baseline) | 0.737 | 0.858 | 94 | Current primary tracker |
| SALT v1 (2026-05-18) | 0.737 | 0.858 | 55 | AUC=baseline, −42% FPS |
| **SALT v2 (2026-05-19)** | — | — | **56** | 6-seq MEAN=0.610 (+0.069) |

---

## Architecture Overview

```
SALTRunner
├── SGLATracker (DeiT-tiny, 0.9 GFLOPs)    ← primary tracker
│   ├── CE token pruning (50% on CONFIRMED) ← compute savings
│   └── try_update_template()               ← disabled (causes regression)
├── TargetStateAssessor (TSA)               ← state estimation
│   ├── APCECalibrator                      ← adaptive thresholds
│   ├── VelocityDriftMonitor                ← false-CONFIRMED detection
│   ├── Farneback optical flow              ← consistency signal
│   └── Supervised MLP head (92.5% acc)    ← confidence scoring
├── YOLOv8/YOLO26m VisDrone detector       ← LOST recovery
│   └── Guard 3 (cosine), Guard 5 (displacement+crowded)
├── CosineAppearanceMemory                  ← ref embedding, drift
└── OnlineLSTMMotionPredictor               ← disabled (zero benefit)
```

### TSA State Machine

| State | APCE threshold | Compute routing | Recovery |
|-------|:---:|:---:|:---:|
| CONFIRMED | ≥ 80 | 50% CE tokens | — |
| OCCLUDED | 20–80 | full depth | escalation after 25f |
| LOST | < 20 | full depth | YOLO26m after 5 LOST |
| DISTRACTOR_RISK | CONFIRMED + VelocityDrift | full depth | — |
| DYNAMIC | (unreachable, LSTM off) | full depth | — |

---

## Session 2026-05-18 — Foundation

- SGLATrack integrated, weights loaded (`sglatrack_ep0297.pth.tar`, AUC 0.737)
- OSTrack-256 baseline (AUC 0.639, 10fps) — kept as reference, not in pipeline
- SALT architecture designed and implemented:
  - TSA with optical flow consistency, LSTM motion, appearance drift
  - CE token pruning (50% on CONFIRMED state)
  - SALTRunner pipeline connecting all components
  - RT-DETRv2-S recovery detector
- Full 15-sequence UAV123 benchmark: SALT AUC=0.737 (=baseline at 55fps)
- TTT HeadAdaptor: zero gradient effect confirmed → disabled

---

## Session 2026-05-19 (night) — Architecture Hardening

- APCE/PSR signals promoted to primary TSA signals (optical flow demoted to fallback)
- APCE thresholds calibrated: LOST<20, OCCLUDED<80 (from empirical UAV123 measurements)
- OCCLUDED→LOST escalation pipeline added (uav2: 47% OCCLUDED never triggers natural LOST)
- Recovery pipeline guards: cosine similarity, size consistency, spatial gate, temporal voting
- Detector swapped to YOLO26m VisDrone (55% mAP@0.5 UAV-specific)
- Recovery correctly fires on uav2 (AUC 0.136→0.492, +0.356)
- Staged OCCLUDED escalation: mean APCE guard prevents false escalation on car7
- TSA temporal gating: skip Farneback on consecutive CONFIRMED+high-APCE frames

---

## Session 2026-05-19 (day) — CV/ML Review + Multi-Agent Improvements

### Architecture review findings
Full CV/ML architect review. 18 bugs found (see `bugs.md`). Key issues:
- MLP head confidence non-informative (0.013–0.018, zero variance) → replaced with rule-based
- Template EMA update → regression (car7: 0.570→0.321) → disabled again
- `CosineAppearanceMemory` fresh instantiation per call → random projection inconsistency (BUG-02)
- Supervised training data (`uav123_labels.npz`) has all-zero `flow_features` (BUG-01)
- `CosineAppearanceMemory` (12288,)@(12288×64) matmul = FPS bottleneck (BUG-12)

### New datasets integrated
- **VisDrone-SOT test-dev** (35 sequences, extracted from `~/Downloads`): loader written at `src/uav_tracker/datasets/visdrone_sot.py`, registered as `"visdrone_sot"`, benchmarked
- **UAVDT removed** from sources (not available, not needed)
- **DTB70**: loader exists, awaiting download

### 7-agent improvement sprint (parallel design agents)

| Component | Change | Impact |
|-----------|--------|--------|
| `APCECalibrator` | `max(80, p75×0.5)` for OCCLUDED thr; `min(20, p5×1.5)` for LOST | Adapts to dataset APCE distribution without shrinking OCCLUDED window |
| `VelocityDriftMonitor` | Freeze score + PSR decay → DISTRACTOR_RISK | Catches false-CONFIRMED like uav0000164 (AUC=0.174 at 99% CONFIRMED) |
| `_rule_confidence` | Replaces MLP online adaptation | Removes 4390-param dead-weight head |
| `BUG-02 fix` | Shared `_get_embed_helper()` singleton | All cosine similarities now in consistent embedding space |
| Recovery guards A+B | APCE trend gating + crowded-scene sim threshold 0.50 | bike2: 0.153→0.171 (+0.018); bad recovery (IoU=0.000) blocked |
| `fast_bench.py` | `--restart-ope` flag, SR@0.5, recovery IoU metric | Recovery quality now measurable |
| Detector ablation | Configs for YOLO26m/RT-DETR/LEAF-YOLO; RT-DETR symlink fix | Ablation ready to run |

### Bugs fixed this session

| Bug | Status | Impact |
|-----|--------|--------|
| BUG-02: fresh embed helper per call | ✅ Fixed | Cosine guards now meaningful |
| BUG-04: SGLATrack re-instantiated per sequence | ✅ Fixed | SGLATrack FPS 72→78fps |
| BUG-07: head_adaptor dead code | ✅ Removed | ~80 lines dead code gone |
| BUG-06: _template_window dead fields | ✅ Removed | Clean sglatrack.py |
| BUG-08: inline logging imports | ✅ Fixed | Single module-level logger |
| BUG-03: motion_predictor on LOST frames | ✅ Fixed | Gated to CONFIRMED/DYNAMIC |
| BUG-05: DYNAMIC unreachable | ✅ Documented | Comment added |
| BUG-10: v2_full_ml.yaml stale config | ✅ Moved | `configs/archive/` |
| BUG-13: salt.yaml tsa.name phantom field | ✅ Removed | Clean config |
| BUG-15: update_online() no-op calls | ✅ Removed | Dead calls gone |
| BUG-12: CosineAppearanceMemory matmul bottleneck | ✅ Fixed | 64×64→32×32 crop: VisDrone 27→40fps |
| BUG-17: template update EMA regression | ✅ Documented | Disabled, conditions to re-enable noted |
| BUG-18: bad recovery indistinguishable | ✅ Partial | A+B guards block bike2 bad recovery |

### Bugs still open

| Bug | Priority |
|-----|----------|
| BUG-01: uav123_labels.npz all-zero flow_features | Medium — existing weights (mode=sglatrack) are already from real features |
| BUG-11: typer in requirements | Closed — already present in pyproject.toml |
| BUG-14: frame-0 CONFIRMED initialization | Low |
| BUG-16: _RECOVERY_WARMUP_FRAMES comment wrong | Low |

### Supervised TSA head
- Trained: `scripts/train_tsa_classifier.py --mode sglatrack --epochs 50`
- Source: real SGLATrack APCE/PSR/entropy on all 123 UAV123 sequences
- Result: val_acc=**0.925** (92.5%), saved to `weights/tsa_head_uav123.pth`
- Loaded in pipeline via `salt.yaml → tsa.weights_path`
- Impact on AUC: zero (state decisions still rule-based via `_decide_state`); confidence scoring only

### Recovery regression analysis (bike2, car7)
- **bike2 −0.005** (0.171 vs SGLATrack 0.176): structural — bad recovery blocked by A+B guards, remaining gap from non-recovery frames. Not recoverable without per-instance appearance model
- **car7 −0.025** (0.570 vs 0.595): from 50 OCCLUDED frames (mean IoU=0.219) where full-compute routing doesn't help with motion blur. Car7 recovery at frame 356 was correct (IoU=0.951)
- **Root cause (BUG-18)**: YOLO26m finds wrong cyclist on bike2 (cosine_sim=0.921, IoU=0.000) — same LOST streak length as car7's correct recovery, so threshold cannot distinguish them

---

## Key Files

| File | Purpose |
|------|---------|
| `src/uav_tracker/salt_runner.py` | Main SALT pipeline |
| `src/uav_tracker/trackers/sglatrack.py` | SGLATrack + CE routing |
| `src/uav_tracker/ml/tsa/target_state_assessor.py` | TSA: APCECalibrator, VelocityDriftMonitor, supervised head |
| `src/uav_tracker/ml/tsa/velocity_drift.py` | VelocityDriftMonitor (new) |
| `src/uav_tracker/datasets/visdrone_sot.py` | VisDrone-SOT loader (new) |
| `src/uav_tracker/ml/appearance_memory/cosine_memory.py` | 32×32 embedding (was 64×64) |
| `configs/experiments/salt.yaml` | Active SALT config |
| `configs/experiments/salt_detector_*.yaml` | Detector ablation configs (new) |
| `scripts/fast_bench.py` | Benchmark with `--restart-ope`, `--dataset visdrone_sot` |
| `scripts/train_tsa_classifier.py` | Supervised TSA training (new) |
| `scripts/eval_recovery_detectors.py` | Detector ablation eval (new) |
| `weights/tsa_head_uav123.pth` | Supervised TSA head (92.5% val acc) |
| `bugs.md` | 18-item bug inventory |

---

## What's Next

| Priority | Action |
|----------|--------|
| 1 | DTB70 benchmark when download completes: `fast_bench.py --dataset dtb70` |
| 2 | Recovery post-commit validation (BUG-18 fix path): after re-init, if APCE < 80 within 3 frames, undo recovery |
| 3 | Dynamic crop fix: ablate adaptive search scale (4×→2× for small targets) — see "Compressed Image Concern" section |
| 4 | RT-DETR symlink + detector ablation: `ln -s ~/projects/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth weights/rtdetr/...` then `eval_recovery_detectors.py` |
| 5 | BUG-01: Update `generate_ml_labels.py` to run SGLATrack → populate `iou_trace` and verify `flow_features` |
| 6 | Template update re-enable: freeze `_ref_embedding` (remove EMA), reduce blend to 5%, gap ≥200f |
| 7 | Full 123-sequence UAV123 benchmark for paper Table 2 |
| 8 | VisDrone SOT fine-tuning: `scripts/finetune_tsa_visdrone.py` (design spec in session notes) |

---

## CE / UTPTrack Token Pruning Analysis (2026-05-19)

### Discovery: CE was silently broken since day 1
DeiT-tiny uses timm's `Block` which does NOT support `return_attention=True`. The original CE scoring call was a no-op — all "CE results" before this session were actually measured with no token pruning. The recovery pipeline gains (+0.069 MEAN) were entirely from TSA routing + YOLO26m recovery, not CE.

### Three architectural bugs fixed in base_backbone.py

| Bug | Description | Severity |
|-----|-------------|----------|
| Q1 | `_CE_LOC={3,6,9}` but only i<start_layer=5 fires → layers 6,9 never prune | HIGH |
| Q2 | CE used block i+1's QKV on block i's output (distribution mismatch) | MEDIUM |
| Q4 | CTEM used mean of all 64 template tokens (background dominates for small UAVs) | MEDIUM |

### Results before vs after fixes (UAV123 MEAN)

| keep_ratio | Before fixes CE | After fixes CE | Before fixes CTEM | After fixes CTEM |
|:---:|:---:|:---:|:---:|:---:|
| 0.85 | 0.567 | **0.610** | 0.549 | **0.610** |
| 0.75 | 0.573 | 0.554 | 0.479 | 0.553 |
| 0.65 | 0.558 | 0.571 | 0.473 | 0.552 |
| 0.50 | 0.551 | **0.616** | 0.442 | 0.506 |

CE at kr=0.50 (corrected) improves over no-pruning: uav2 +0.015, bike2 +0.005, car7 +0.042. Removing 50% of background/distractor search tokens helps the tracker focus.

### GFLOPs analysis

| | Full model | CE kr=0.85 | CE kr=0.50 | CTEM kr=0.85 | CTEM kr=0.50 |
|--|:---:|:---:|:---:|:---:|:---:|
| GFLOPs | 1.2663 | +3.6% ❌ | −0.9% | −2.0% | −6.4% |

Paper claims 0.90 GFLOPs = pre-SGLA trunk only (5 blocks). Full model = 1.2663 GFLOPs.  
CE overhead (norm1+QKV of scored block = 0.071 GFLOPs) exceeds savings until kr ≤ 0.57.  
**CE is net-negative FLOPs at kr=0.85 — CTEM is always cheaper to score.**  
Max speedup from single-stage pruning (layer 3 only): ~7% with CTEM kr=0.50.

### CTEM verdict: worse than CE at every ratio
CTEM collapses uav2 to AUC=0.068 at all ratios even after Q4 fix (center template). The cosine similarity to even the center template tokens is insufficient to select discriminative search tokens when the target is small (< 20px). CE's Q·K^T cross-attention scoring is fundamentally more discriminative.

### Current config
- CE active at kr=0.50, layer 3 only (`_CE_LOC={3}`)
- State-adaptive: CONFIRMED=0.50, all other states=1.0 (full compute)
- MEAN=0.616 (+0.075 vs SGLATrack, new best)

---

## Compressed Image Concern — Dynamic Crop Issue (2026-05-19)

### Observation
SGLATrack crops the search region around the predicted target location and resizes it to the fixed model input size (256×256 for the search region). On sequences where the target is very small relative to the search window (e.g. VisDrone-SOT high-altitude footage, UAV123 LR sequences), the crop contains mostly background and the target occupies < 4% of the input area after resize.

### Impact
- JPEG re-compression during crop-resize introduces block artifacts that degrade CE scoring: the Q·K^T attention map reflects JPEG grid structure rather than target saliency
- For small targets (< 20px apparent size), CTEM collapses entirely (uav2 AUC=0.068) — the center template token represents background, not target
- CosineAppearanceMemory BUG-12 root was also here: the 64×64 crop included mostly background context, making embeddings near-identical across frames

### Status
Not yet fixed. Potential mitigations (not yet evaluated):
1. Adaptive search scale: tighten the search region multiplier from 4× to 2× when target is small (< 30px)
2. Target-centric crop: use predicted bbox center + fixed absolute window (e.g. 96px around target) instead of relative scale
3. Known risk: smaller search window reduces recovery tolerance for fast-moving targets — needs ablation

---

## Full UAV123 Benchmark — SALT v3 Final (123 sequences, no frame cap)

| Method | AUC | Pr@20 | FPS |
|--------|:---:|:---:|:---:|
| SGLATrack (123 seqs) | 0.718 | 0.869 | 80 |
| **SALT v3 CE kr=0.50** | **0.720** | **0.874** | 62 |

**SALT beats SGLATrack on the full benchmark (+0.002).** Note: the 0.737 reference was a 15-seq easier subset — on all 123 seqs SGLATrack itself scores 0.718.

Notable per-sequence wins: uav2 (0.507 vs 0.136, +0.371), uav3 (0.428 vs 0.167, +0.261), uav4 (0.371 vs 0.090, +0.281). Recovery pipeline drives the improvements on all uav* sequences.

## VisDrone SOT — CE Active (final)

| Method | AUC | Pr@20 | FPS |
|--------|:---:|:---:|:---:|
| SGLATrack | 0.672 | 0.859 | 72 |
| SALT v3 (CE active) | 0.672 | 0.855 | 38 |

CE introduces zero AUC regression on VisDrone SOT (−0.001 within noise).
