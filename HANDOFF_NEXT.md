# HANDOFF — SALT-RD: Candidate-Aware Verification Track

**Date:** 2026-05-21  
**Sessions covered:** Sessions 1–5 (2026-05-21) — proxy memory, SGLA memory, DINOv2 ROI pilot, SGLATrack top-K candidate mining
**Owner:** Staff CV/AI/ML review track  
**Test count:** 214 SALT-RD unit tests passing (`tests/unit/test_saltr_*.py`)
**Worktree:** modified `HANDOFF_NEXT.md`, `src/uav_tracker/trackers/sglatrack.py`, `saltr/src/salt_r/candidate_mining_pilot.py`, tests, plus generated pilot JSONs

---

## Current State (one sentence)

Phase 4B/4C is **scientifically complete**: proxy memory proves false-confirmed is learnable (diag fc AUROC 0.598→0.796), but real SGLATrack embeddings, first-pass DINOv2 ROI identity, and SGLATrack score-map top-K candidates all fail the hard diagnostic gate; next work should move to **external/teacher candidate verification** (CoTracker3/TAPIR point consistency + candidate-aware DINO/DAM-style memory), not more generic single-bbox embeddings.

---

## Most Recent Commits

```
0595315 docs: merge NIGHT_SUMMARY_0521 + HANDOFF_NEXT → single canonical handoff
29ddf72 fix(saltr): P0 bootstrap + P1 fail-fast/preds-length + P2 provenance — 207 tests
71ad804 fix(saltr): P0 #2 — extractor uses SALTRunner.run() for trajectory alignment
36a110c fix(saltr): P0/P1 extractor+OOF blocking issues — 205 tests pass
e94853d feat(saltr): make_oof_predictions.py — 5-fold OOF train preds, no leakage
24c0df8 feat(saltr): sgla_memory_extractor.py — causal RAM sidecar from real backbone embeddings
4eebd37 fix(tracker): _reset_embedding_cache — close init/early-return lifecycle gap
7fe62e4 fix(tracker+saltr): P2/P3 embedding hook + sidecar width guards
1645189 feat(tracker): SGLATrack per-frame embedding export hook — 3 views, zero overhead
499af75 feat(saltr): memory ablation infra + 3 ablation runs — pos_only drives diagnostic gain
```

---

## Critical Numbers

### Baseline table (strict comparisons — use v2_retrained for all future gates)

| Checkpoint | Input | Val fc AUROC | Val fc AUPRC | Diag fc AUROC | Diag fc AUPRC |
|---|---|---:|---:|---:|---:|
| v2_corrected | 28-dim | 0.885 | 0.338 | 0.548 | 0.248 |
| **v2_retrained (strict baseline)** | **28-dim** | **0.885** | **0.338** | **0.598** | **0.281** |
| v2.1 proxy memory | 37-dim | 0.857 | 0.243 | 0.796 | 0.518 |
| v2.2 SGLA target | 32-dim | ≥ 0.870 | ≥ 0.300 | > 0.774 | — |
| v2.2 SGLA score_weighted | 32-dim | 0.858 | 0.260 | 0.584 | — |
| v2.2 SGLA peak_local | 32-dim | 0.858 | — | 0.584 | — |

**Historical note:** v2_corrected diagnostic 0.548 vs v2_retrained 0.598 — use retrained as strict baseline.

### Ablation results (false_confirmed AUROC, diagnostic / val)

| Ablation | dims | Diag fc AUROC | Val fc AUROC | Val fc AUPRC | Decision |
|---|---|---:|---:|---:|---|
| telemetry only | 28 | 0.598 | 0.885 | 0.338 | strict baseline |
| margin only | 29 | 0.602 | 0.881 | 0.345 | proxy margin = noise |
| **pos only (RAM)** | **32** | **0.774** | **0.852** | **0.195** | **key driver — 89% of full gain** |
| neg only (DRM) | 32 | 0.496 | 0.872 | 0.312 | harmful without positive context |
| full memory 9-dim | 37 | 0.796 | 0.857 | 0.243 | best diag, worst val AUPRC |

**Decision:** Build real SGLATrack sidecar with **pos-only RAM** (4 features). Negative memory not validated — do not include in v2.2.

### Policy sweep results (49 val sequences, 30 892 frames)

| Config | macro_tcr | wrir | recall | density/1kf |
|---|---:|---:|---:|---:|
| No-memory (fc=0.4, reinit=0.8) | 0.0361 | 0.000 | 0.953 | 532 |
| **Memory runner-up (fc=0.4–0.9, reinit=0.4, mem=0.00)** | **0.0055** | **0.000** | **0.977** | **677** |
| Memory best (fc=0.4, reinit=0.4, mem=+0.20) | 0.0031 | 0.000 | 1.000 | 889 |

**Use runner-up config** for policy deployment — best config blocks 67% of safe template updates (msu=0.671).

### E-process (formal, ifd10 risk, α=0.10, ε=0.50)

| Metric | Value | Status |
|---|---|---|
| Median lead time | 12.0f | ✅ |
| Event recall | 3.1% | ❌ analysis only |
| FA/1000f | 0.21 | ✅ |

**E-process: analysis/monitoring tool only. Not suitable as runtime gate.**

### New literature/code screening (architectural decision)

| Direction | Source | Local/URL | Verdict for current SALT-RD track |
|---|---|---|---|
| Target-candidate association | KeepTrack: "Learning Target Candidate Association to Keep Track of What Not to Track" | https://arxiv.org/abs/2103.16556 | **Most relevant conceptually**: false-confirmed requires reasoning over target+distractors, not unary bbox confidence. Motivated Phase 4C top-K candidate pilot. |
| Point-guided update safety | PTDT: Point tracking-guided dynamic tokens for long-term small object tracking | `papers/ptdt.pdf` | **Next implementation target**: use point consistency as teacher for candidate verification. |
| Distractor-aware memory | DAM4SAM CVPR 2025 | `papers/36_DAM4SAM_Distractor_Aware_Memory_SAM2_CVPR2025.pdf`, https://github.com/jovanavidenovic/DAM4SAM | Good memory principle, but full SAM2 memory is too heavy. Use lightweight target-vs-distractor margin only. |
| Offline point teacher | CoTracker3 | `papers/CoTracker3_2025_Simpler_and_Better_Point_Tracking_by_Pseudo_Labeling_Real_Videos.pdf`, https://github.com/facebookresearch/co-tracker | Best immediate teacher candidate for false-confirmed: compare tracked point cloud with SGLA bbox/candidates. |
| UAV tracker replacement | ORTrack CVPR 2025 | `papers/code/ORTrack/README.md` | Good AUC/backbone candidate later; not the quickest fix for SALT-RD false-confirmed. |
| Token pruning / FPS | UTPTrack CVPR 2026 | `papers/code/UTPTrack/README.md` | Strong compute competitor; not a false-confirmed solution. Use later for FPS/AUC track. |
| Observer-follower UAV system | SDG-Track | `papers/code/SDG-Track/README.md` | Useful if we need external candidate generator; heavier architecture than current trust-controller path. |

### Candidate mining pilot (SGLATrack score-map alternatives)

Implementation:
- `src/uav_tracker/trackers/sglatrack.py` now exports top-K local candidate boxes in `score_map_stats["candidates"]`.
- `saltr/src/salt_r/candidate_mining_pilot.py` runs SALTRunner and evaluates oracle top-K recall against offline GT.
- Result artifact: `saltr/results/candidate_mining_pilot_diagnostic.json`.

Diagnostic false-confirmed oracle recall:

| Scope | n fc frames | mean K | top-3 IoU≥0.3 | top-5 IoU≥0.3 | best IoU mean | Decision |
|---|---:|---:|---:|---:|---:|---|
| all diagnostic | 289 | 4.61 | 0.266 | 0.298 | 0.216 | **KILL as sole source** |
| DTB70 hard | 183 | 4.38 | 0.421 | 0.470 | 0.340 | borderline, mostly Sheep1 |
| UAV123 bike2 | 106 | 5.00 | 0.000 | 0.000 | 0.000 | **KILL** |

Per sequence:

| Sequence | n fc | top-5 IoU≥0.3 | best IoU mean | Note |
|---|---:|---:|---:|---|
| `dtb70/Gull2` | 29 | 0.000 | 0.027 | target absent from score-map alternatives |
| `dtb70/Sheep1` | 82 | 1.000 | 0.686 | candidate-aware verifier can work here |
| `dtb70/StreetBasketball1` | 72 | 0.056 | 0.073 | target mostly absent from top-K |
| `uav123/bike2` | 106 | 0.000 | 0.000 | target absent from top-K despite K=5 |

Interpretation:
- SGLATrack score-map candidates are **not enough** for general false-confirmed recovery. They help on Sheep1, but fail on bike2/Gull2/StreetBasketball1.
- This explains why DINO/SGLA single-bbox memory failed: often the correct target is not just mis-ranked; it is missing from SGLATrack's local response candidates.
- Next candidate source must be external or teacher-guided: CoTracker3/TAPIR point consistency, optical-flow/RAFT cycle consistency, or triggered detector/SAM/TAM candidate generator.

---

## Phase Plan

### ✅ Phase 0 — Baseline fixed
- NIGHT_SUMMARY_0521.md committed
- Strict baseline = v2_retrained (not v2_corrected)
- All ablations complete

### ✅ Phase 1 — Proxy memory ablations
- 4 ablation runs: margin/pos/neg/full
- pos_only is key driver (89% of diagnostic gain)
- neg_only is actively harmful alone

### ✅ Phase 2 — Real SGLATrack embeddings (KILL)

**✅ Pre-flight code fix — done:**

`sgla_memory_extractor.py` now sets `interrupted=True` on `KeyboardInterrupt` and raises `RuntimeError` without saving unless `--allow-partial` or `--smoke-test` is set. Canonical sidecar can no longer be silently truncated by Ctrl+C.

**✅ Step 1: OOF predictions — done**

Output: `saltr/results/preds_all_v2_oof_teacher.json` — 228 seqs (175 OOF train, 49 val teacher, 4 diag teacher). Exact coverage confirmed by `_validate_merged()`.

**✅ Step 2: Smoke extraction (3 sequences) — passed**

All gates green: `n_ram_updates~1` on all 3 seqs, `mean_pos_mean_sim` 0.90–0.92, 0 skipped. Bootstrap fired at t=1 as expected.

**✅ Step 3: Full extraction (228 seqs) — done**

QA passed: 228 keys, n_sequences=228, active_pos_sequences=228, no bad arrays.

**✅ Step 4: Train v2.2 score_weighted — done (KILL)**

| Checkpoint | Diag fc AUROC | Val fc AUROC | Val fc AUPRC |
|---|---:|---:|---:|
| v2_retrained (baseline) | 0.598 | 0.885 | 0.338 |
| v2.1 proxy memory | 0.796 | 0.857 | 0.243 |
| **v2.2 score_weighted** | **0.584** | **0.858** | **0.260** |
| Gate (мін) | >0.774 | ≥0.870 | ≥0.300 |

score_weighted гірший за baseline на diag (0.584 < 0.598) — embeddings не несуть корисного сигналу. Per fallback plan.

**✅ Step 5: peak_local extraction + training — done (KILL)**

peak_local is effectively identical to score_weighted:

| Checkpoint | Diag fc AUROC | Val fc AUROC |
|---|---:|---:|
| v2_retrained baseline | 0.598 | 0.885 |
| v2.1 proxy memory | 0.796 | 0.857 |
| v2.2 score_weighted | 0.584 | 0.858 |
| **v2.2 peak_local** | **0.584** | **0.858** |

Decision: do **not** spend more time on SGLA `global` unless needed for completeness. The views are numerically different, but all come from localization-oriented DeiT-tiny search tokens and do not carry reliable identity signal for hard false-confirmed cases.

### ✅ Phase 3 — Per-dataset evaluation guard

`saltr/src/salt_r/eval.py` now writes and prints `per_dataset_head_metrics`. This is mandatory for future claims because pooled val is UAV123-heavy and can hide VisDrone/DTB70 regressions.

Smoke result on diagnostic baseline:

| Dataset | Seqs | Frames | fc base | fc AUROC | fc AUPRC |
|---|---:|---:|---:|---:|---:|
| dtb70 | 3 | 869 | 21.1% | 0.582 | 0.302 |
| uav123 | 1 | 553 | 19.2% | 0.637 | 0.229 |

### ✅ Phase 4 — DINOv2 ROI identity pilot (first-pass KILL on diagnostic)

Implementation file: `saltr/src/salt_r/dino_identity_pilot.py`

What it does:
- uses `bbox_pred/{seq}` for current tracker crop;
- uses frame-0 `bbox_gt/{seq}` as trusted initial target crop;
- extracts frozen DINOv2 embeddings with official Meta Torch Hub model (`facebookresearch/dinov2`, default `dinov2_vits14`);
- preprocessing: square ROI crop with context, resize to 224 or 518, bicubic, ImageNet mean/std;
- supports `--embedding-mode cls` and `--embedding-mode patch_mean`;
- computes single-feature false-confirmed AUROC/AUPRC for `1 - dino_init_sim`, `1 - dino_mem_max_sim`, `dino_delta_prev`, `dino_update_age_norm`;
- uses OOF/teacher preds only for causal memory update gates.

Official implementation sources checked:
- Meta DINOv2 repo / Torch Hub: https://github.com/facebookresearch/dinov2
- DINOv2 model card: https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
- DINOv2 hub backbones: https://raw.githubusercontent.com/facebookresearch/dinov2/main/dinov2/hub/backbones.py
- DINOv2 transforms reference: https://raw.githubusercontent.com/facebookresearch/dinov2/main/dinov2/data/transforms.py

Pilot selection:
- 4 diagnostic hard sequences: `dtb70/Gull2`, `dtb70/Sheep1`, `dtb70/StreetBasketball1`, `uav123/bike2`
- 8 dataset-balanced val sequences
- frame stride 5 plus all false-confirmed positives

Pilot gate:
- diagnostic AUROC(`1 - dino_init_sim`) ≥ 0.65
- val AUROC(`1 - dino_init_sim`) ≥ 0.60

Results:

| Variant | Diag AUROC | Val AUROC | Overall AUROC | Gate |
|---|---:|---:|---:|---|
| CLS, context 2.0, 224 | 0.514 | 0.862 | 0.802 | FAIL |
| CLS, context 1.2, 224 | 0.563 | 0.896 | 0.743 | FAIL |
| CLS, context 3.0, 224 | 0.518 | 0.897 | 0.797 | FAIL |
| CLS, context 1.2, 518 | 0.547 | 0.864 | 0.674 | FAIL |
| patch_mean, context 1.2, 224 | 0.560 | 0.901 | 0.777 | FAIL |

Interpretation:
- DINOv2 has a useful identity signal on val and on `uav123/bike2`.
- It fails the hard diagnostic gate because DTB70 hard cases (`Gull2`, partly `Animal2`-like organic scenes) are not solved by generic initial-crop similarity.
- Do **not** build a full DINO sidecar yet. First-pass DINO confirms the issue is not just SGLA tokens, but generic crop identity also struggles on the hardest organic/distractor cases.

Commands:

```bash
# Dry-run selection
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.dino_identity_pilot \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --preds saltr/results/preds_all_v2_oof_teacher.json \
  --output saltr/results/dino_identity_pilot_dryrun.json \
  --dry-run --max-val-seqs 8

# Best tested first-pass variant (still fails diagnostic gate)
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.dino_identity_pilot \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --preds saltr/results/preds_all_v2_oof_teacher.json \
  --output saltr/results/dino_identity_pilot_patch_ctx1p2.json \
  --max-val-seqs 8 --frame-stride 5 \
  --context-scale 1.2 --embedding-mode patch_mean \
  --device auto --batch-size 32
```

### ✅ Phase 4C — Candidate-aware SGLATrack score-map pilot (KILL as sole source)

Motivation:
- KeepTrack-style target-candidate association suggests that false-confirmed tracking should be diagnosed by reasoning over **target plus distractors**, not only by comparing the current top-1 bbox to template.
- Our previous DINO/SGLA memory pilots were unary: `current bbox vs memory`. They could not ask whether a better target candidate exists nearby.

What was implemented:
- `src/uav_tracker/trackers/sglatrack.py`
  - `_select_candidate_peak_indices()`: greedy NMS over the 16×16 post-Hann score map.
  - `_extract_candidate_diagnostics()`: decodes up to 5 local candidate peaks into frame-space bboxes via `size_map` and `offset_map`.
  - `score_map_stats` now includes:
    - `local_top1`, `local_top2`, `local_peak_margin`, `local_top2_ratio`
    - `n_secondary`
    - `candidates`: list of `{rank, row, col, score, score_ratio, bbox, center}`
- `saltr/src/salt_r/candidate_mining_pilot.py`
  - runs SALTRunner over selected NPZ sequences;
  - compares top-K candidate bboxes against offline GT;
  - reports oracle top-3/top-5 recall on false-confirmed frames.
- `tests/unit/test_saltr_candidate_mining.py`
  - protects NMS, grid→frame bbox mapping, and oracle-hit metrics.

Diagnostic command:

```bash
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.candidate_mining_pilot \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --config-path configs/prod/salt.yaml \
  --output saltr/results/candidate_mining_pilot_diagnostic.json \
  --splits diagnostic \
  --include-frame-records
```

Gate:
- GO if false-confirmed `oracle_top3_recall_iou03 >= 0.50` on diagnostic.
- KILL if top-5 recall stays below 0.60 or if key sequences have zero target-in-candidates.

Result:
- Overall diagnostic top-3 IoU≥0.3 recall = 0.266; top-5 = 0.298.
- `Sheep1` is strongly recoverable from SGLATrack candidates (top-5 = 1.000).
- `bike2`, `Gull2`, and `StreetBasketball1` are not: correct target is essentially absent from top-K.

Decision:
- **KILL SGLATrack top-K candidates as the only candidate source.**
- Keep the hook: it is still useful telemetry and will support candidate-aware sidecars when an external candidate source exists.
- Next: external/teacher candidate verification.

Best next architecture:
1. Use CoTracker3/TAPIR/RAFT to generate point-consistency target hypothesis from the initial bbox.
2. Compare SGLATrack top-1 and any external/top-K candidates against the point cloud.
3. Add DINOv2 only as target-vs-candidate margin, not unary init similarity.
4. Distill teacher features into SALT-RD; runtime can start as conservative veto only.

### ✅ Phase 5A — Point-teacher pilot (LK/Farneback passes gate, CoTracker3 DROPPED)

**CoTracker3 KILLED — crashed laptop RAM (8–16 GB peak), marginal gain (+0.019 AUROC overall vs LK).**

Pilot run on 4 diagnostic seqs + 5 extended seqs. Methods: CoTracker3, Farneback, LK.

Gate: `diagnostic all fc AUROC ≥ 0.65` on `pt_inside_pred_ratio`.

| Method | pt_inside_pred_ratio (diag) | gate? | RAM |
|---|---:|---|---|
| CoTracker3 | 0.748 | ✅ | ~8–16 GB → crash |
| **Farneback** | **0.739** | **✅** | ~50 MB |
| LK | 0.729 | ✅ | ~20 MB |

**Per-seq best features (CoTracker3 vs LK, diagnostic):**

| Seq | CT3 best / AUROC | LK best / AUROC | Note |
|---|---|---|---|
| Gull2 | pt_survival_since_init / 0.928 | pt_median_motion / 0.920 | CT3 unique on survival; LK motion = equiv |
| Sheep1 | pt_inside_pred_ratio / 0.922 | pt_bbox_center_disagree / 0.807 | CT3 +0.17 on inside; LK motion features ok |
| StreetBasketball1 | pt_bbox_center_disagree / 0.708 | pt_bbox_center_disagree / 0.731 | identical, LK slightly better |
| bike2 | pt_cluster_area_ratio / 0.815 | pt_cluster_area_ratio / 0.808 | all methods identical |

**Decision rules:**
- StreetBasketball1 (fc_rate=74%) and Gull2 are **hard outlier scenes** — tested separately, not in main gate.
- Core gate seqs: bike2 + Sheep1 → LK sufficient, no need for CoTracker3.
- LK is runtime-compatible candidate (lightweight enough for online use later).

**Artifacts:**
- `saltr/results/point_teacher_pilot_diagnostic.json` — 4 diag seqs, all 3 methods
- `saltr/results/point_teacher_pilot_extended.json` — 5 seqs (bike2/Sheep1/group2_1/uav3/uav0000074)
- `saltr/results/point_teacher_pilot_bike2_cotracker3.json` — bike2 CT3 full-seq
- `saltr/results/point_teacher_pilot_bike2_cotracker3_windows.json` — bike2 CT3 windowed

### ⏳ Phase 5B — Point feature sidecar + train v2.3 (next)

Working name: **SALT-RD v2.3 with LK point features**.

Infrastructure: `saltr/src/salt_r/point_sidecar_extractor.py` — **implemented and tested (230 tests)**.

Design:
- Sliding-window re-seeding every `stride=15` frames, track `window=25` frames with LK.
- Seed from **PRED bbox** (causal — no GT oracle).
- Latest-seed wins per frame (each frame covered by most recent window).
- Outputs `point_features/{seq}` float32 (T, 13) per sequence.
- Fail-fast on any skip (canonical mode); `--smoke-test N` / `--allow-partial` for debug.

Feature set for v2.3 (selected from pilot AUROC):

| Feature | Diag AUROC (LK) | Best on |
|---|---:|---|
| `pt_inside_pred_ratio` | 0.729 | Sheep1 (0.748), bike2 (0.751) |
| `pt_cluster_area_ratio` | 0.563 | bike2 (0.808), Gull2 (0.905 Farr) |
| `pt_bbox_center_disagreement` | 0.624 | StreetBasketball1 (0.731), Sheep1 |
| `pt_forward_backward_error` | 0.591 | Gull2 (0.919) |
| `pt_median_motion` | 0.592 | Gull2 (0.920) |

Train SALT-RD v2.3 with these 5 additional point features (28-dim telemetry + 5-dim point = 33-dim).

**Commands:**

```bash
# Step 1: Smoke extraction (2 sequences)
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.point_sidecar_extractor \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --output saltr/data/salt_rd_point_sidecar_smoke.npz \
  --smoke-test 2 --stride 15 --window 25

# Step 2: Full extraction (228 sequences, ~20–40 min on CPU)
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.point_sidecar_extractor \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --output saltr/data/salt_rd_point_sidecar_lk.npz \
  --method lk --stride 15 --window 25 --max-side 320

# Step 3: Train v2.3 (after wiring point sidecar into train.py)
# Feature names: pt_inside_pred_ratio,pt_cluster_area_ratio,pt_bbox_center_disagreement,pt_forward_backward_error,pt_median_motion
```

**Gate v2.3:**
- Diag fc AUROC > 0.774 (must beat pos_only proxy) AND val fc ≥ 0.870
- Per-dataset reporting required (no pooled-only claims)

**TODO before training:**
- Wire `point_sidecar_extractor.py` output into `train.py` / `eval.py`
  (same `--memory-sidecar` / `--memory-feature-names` pattern as SGLA extractor, or new `--point-sidecar` arg)

Phase 5C — candidate-aware DINO margin (if v2.3 point features pass gate):
- Only if pt_inside_pred_ratio alone is not enough on Sheep1 after sidecar extraction
- Use LK point-cloud bbox as candidate source, DINO margin as identity check
- Gate: diagnostic fc AUROC > 0.80, val fc ≥ 0.87

### ⏳ Phase 6 — Conservative policy integration

Runtime policy config (runner-up, validated conservative):
- `fc_t=0.60, reinit_t=0.70, mem_t=0.00`
- Block template update: `p_fc > 0.60 AND memory_margin < 0.00`
- Abstain recovery/reinit: `p_fc > 0.70`
- E-process: monitoring only

**Policy gate:** wrir=0, msu < 0.40, macro_tcr beats strict no-memory baseline (0.0361).

### ⏳ Phase 7 — SALT-RD as learned calibrated TSA successor

Thesis: **SALT-RD becomes what TSA wanted to be, but learned and calibrated.** TSA remains the rule-based safety scaffold during transition; SALT-RD gradually takes over state interpretation and action selection only after replay and runtime gates pass.

#### Stage 1 — Shadow Mode (observe only)

**✅ Implemented + run: `saltr/src/salt_r/shadow_mode.py`**

Results (v2.1 checkpoint + proxy memory sidecar):

**⚠️ Bug note:** `coverage_block_when_fc=0.0` in stored JSONs is an artifact of an old version that
didn't populate `n_true_block`/`n_false_alarm` at per-seq level. Fixed defensively in
`shadow_mode.py` (agg loop now guards `None`). Re-run to get correct JSON aggregate.

**Real numbers (recomputed from per-seq block_rate_when_fc):**

| Split | coverage_block\|fc | false_alarm_rate\|correct | overall_block_rate |
|---|---:|---:|---:|
| val (49 seq) | **0.452** | **0.059** ✅ | 8.6% |
| diagnostic (4 seq) | **0.628** | **0.317** ❌ | 27.6% |

Val: 45% of gt-fc frames would be blocked, only 5.9% false alarms on correct frames. Good.
Diagnostic FAR driven by Gull2 (FAR=0.91) and StreetBasketball1 (FAR=0.67) — hard outlier scenes.

Per-seq diagnostic:

| Seq | cov | FAR | Note |
|---|---:|---:|---|
| bike2 | 0.66 | 0.11 | ✅ acceptable |
| Sheep1 | 0.51 | 0.17 | ok |
| StreetBasketball1 | 0.72 | 0.67 | ❌ hard scene — test separately |
| Gull2 | 0.66 | 0.91 | ❌ hard scene — test separately |

**Commands:**

```bash
# Re-run shadow mode with fixed aggregate (produces correct n_true_block in JSON)
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.shadow_mode \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --checkpoint saltr/checkpoints/v2_1_memory/saltrd_best.pt \
  --memory-sidecar saltr/data/salt_rd_memory_sidecar.npz \
  --split val --output saltr/results/shadow_mode_val_v2_1_fixed.json

PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.shadow_mode \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --checkpoint saltr/checkpoints/v2_1_memory/saltrd_best.pt \
  --memory-sidecar saltr/data/salt_rd_memory_sidecar.npz \
  --split diagnostic --output saltr/results/shadow_mode_diagnostic_v2_1_fixed.json
```

**Exit gates for Stage 1 → Stage 2:**
- no leakage: train sidecar uses OOF predictions, val/diagnostic use teacher predictions
- calibration checked on val only; diagnostic is never used for threshold fitting
- event replay shows template corruption and wrong reinit reductions without AUC/regression surprises

**After v2.3 is trained:** re-run shadow mode with v2.3 checkpoint + point sidecar to see if
diagnostic FAR improves (expected: point features reduce false blocks on Gull2/StreetBasketball1).

#### Stage 2 — Advisory/Veto Mode (safe limited authority)

TSA still decides the classical state (`CONFIRMED`, `OCCLUDED`, `LOST`) and controls normal tracking. SALT-RD is allowed to **veto dangerous actions**, but not move the box or reinitialize the tracker by itself.

Allowed SALT-RD actions:
- block template update when `p_false_confirmed` is high or positive-memory similarity collapses
- block detector reinit when false-confirmed/recovery risk is high
- force full compute when `p_ifd10/p_ifd20` indicates proactive risk
- raise e-process or calibrated-risk alerts for monitoring

Forbidden in this stage:
- no direct bbox correction
- no direct detector winner selection
- no learned reinit without TSA/recovery confirmation
- no threshold tuning on diagnostic

Exit gates:
- `wrir = 0`
- template corruption rate improves versus strict v2 baseline
- wrong reinit rate improves versus strict v2 baseline
- safe template updates are not over-blocked (`msu < 0.40`)
- val fc AUROC/AUPRC do not regress beyond the v2.2 gate

#### Stage 3 — Primary Learned Controller (TSA compatibility layer)

SALT-RD becomes the primary state/action controller, while TSA remains as a fallback monitor and compatibility layer. Classical TSA states are no longer the main decision surface; they become one input among telemetry, memory, calibrated probabilities, and policy hysteresis.

Learned state mapping:
- `TRUSTED_TRACKING`: allow normal tracking, CE pruning, template update
- `LOW_EVIDENCE_TRACKING`: full compute, no template update, no reinit
- `FALSE_CONFIRMED_RISK`: freeze template, block reinit, request verification
- `PROACTIVE_DYNAMIC_RISK`: full compute, search verification, early warning
- `REACQUIRE_NEEDED`: detector/recovery allowed only through calibrated memory/recovery gates

Deployment rule:
TSA is not deleted until LODO generalization, replay policy, and a small runtime rollout all pass. If learned control regresses AUC or causes unsafe reinit, fall back to Stage 2 and keep SALT-RD as a trust/veto layer.

### ⏳ Phase 8 — If teacher candidate verifier weak, fallback ladder
1. Hard-case visual audit: render crop panels for `Gull2`, `Sheep1`, `StreetBasketball1`, `bike2` with GT/pred bbox, SGLA candidates, IoU, `p_fc`, DINO similarity.
2. Crop-upscaled point tracking: run CoTracker3/TAPIR on enlarged search crops instead of full frame for 5–20 px UAV targets.
3. RAFT/GMFlow forward-backward cycle teacher if point trackers fail on tiny targets.
4. Detector/SAM/TAM candidate generator:
   - SAMURAI/SAM2/EfficientTAM only as triggered offline/teacher fallback, not always-on edge runtime.
   - Use spatial hint from predicted target trajectory.
5. Contrastive distractor head:
   - ranking loss `sim(target_mem, correct_or_point_candidate) > sim(target_mem, top1_distractor) + margin`.
6. If all candidate teachers fail, narrow runtime claim to safe veto/template-update blocking and open separate AUC/domain adaptation track.

### ⏳ Phase 9 — Separate AUC/domain-adaptation track
LoRAT/fine-tune SGLATrack. Distinct from SALT-RD trust controller claim.

---

## SGLATrack Embedding Hook (confirmed)

**File:** `src/uav_tracker/trackers/sglatrack.py`  
**Backbone:** DeiT-tiny, `embed_dim=192`, joint template (64 tok) + search (256 tok)

3 views available after every `update()` / `update_with_state()`:

```python
# In hook (both update() and update_with_state()), post-Hann score map:
self._last_search_global          = search_tokens.mean(0)                          # (192,)
self._last_search_score_weighted  = (hann_weights * search_tokens).sum(0)          # (192,) softmax weighted
self._last_search_peak_local      = search_tokens[3×3_peak_neighborhood].mean(0)   # (192,)
self._last_template_embedding     = template_tokens.mean(0)                        # (192,)
```

**Reset lifecycle:**
- `_reset_embedding_cache()` called in `init()`, `update()` early-return, `update_with_state()` early-return
- All 4 attrs `None` until first successful forward

**Reading:** after `runner.run(seq)` yields each entry, `runner.tracker._last_search_score_weighted` has current frame's embedding.

**Bootstrap design (critical for diagnostic sequences):**
At frame t=1, `_last_template_embedding` is unconditionally added to RAM as `source="init_template"` — this ensures RAM has at least one anchor even when p_fc stays high throughout (hard false-confirmed cases where gated search updates never open).

---

## Infrastructure Built This Session

| Module | Purpose | Status |
|---|---|---|
| `saltr/src/salt_r/sgla_memory_extractor.py` | Causal pos-only RAM sidecar from real backbone embeddings | ✅ 207 tests |
| `saltr/src/salt_r/make_oof_predictions.py` | 5-fold OOF train predictions, no in-sample leakage | ✅ 207 tests |
| `--memory-feature-names` in train.py + eval.py | Subset sidecar features; any embedding width | ✅ |
| `memory_dim` from actual sidecar width | Not hardcoded to 9; handles 1/4/9/192/512-dim | ✅ |
| Eval fail-fast | ValueError if checkpoint expects memory but sidecar absent or >10% missing | ✅ |
| Provenance fields | `memory_sidecar_md5`, `memory_feature_names_used`, `n_sequences_with_memory` | ✅ |
| `_reset_embedding_cache()` | init/early-return lifecycle gaps closed | ✅ |
| Post-Hann score map in hook | score_weighted/peak_local match tracker localization peak | ✅ |
| Sidecar width validation | Consistent width across all sequences; checkpoint/sidecar match | ✅ |
| OOF validator split | `_validate_merged()` for OOF-only; `validate_final_merged()` for Phase 5 | ✅ |
| Extractor bootstrap | Template embedding at t=1 always added unconditionally | ✅ |
| Extractor fail-fast | RuntimeError on any skip (canonical mode); `--allow-partial` escape hatch | ✅ |
| Preds-length validation | `len(preds) != T` raises before loop; out-of-range defaults are blocking | ✅ |
| OOF provenance | `_build_meta()` stores real checkpoint path + MD5 | ✅ |
| Per-dataset eval | `per_dataset_head_metrics` in `eval.py`; printed fc table | ✅ 18 focused tests |
| DINOv2 ROI pilot | `dino_identity_pilot.py`; Torch Hub DINOv2, CLS/patch_mean modes, dry-run + pilot JSONs | ✅ 18 focused tests |
| SGLATrack top-K candidates | post-Hann local maxima + bbox decode in `score_map_stats["candidates"]` | ✅ 214 SALT-RD tests |
| Candidate mining pilot | `candidate_mining_pilot.py`; oracle top-K recall on false-confirmed frames | ✅ diagnostic KILL |
| Point-teacher pilot | `point_teacher_pilot.py`; LK/Farneback/CT3 on 4 diag + 5 val seqs; CT3 DROPPED | ✅ LK/Farr gate PASS |
| Shadow mode | `shadow_mode.py`; Stage 1 observe-only; val cov=45.2% FAR=5.9%; diag FAR high (Gull2/SB1) | ✅ run + bug fixed |
| Point sidecar extractor | `point_sidecar_extractor.py`; LK/Farneback sliding-window PRED-seeded, 228 seqs | ✅ 230 tests |

---

## Artifact Inventory (canonical state)

| Artifact | Path | Status |
|---|---|---|
| v2 labels (full-horizon) | `saltr/data/salt_rd_v2_labels.npz` | ✅ 14 labels, 228 seqs |
| proxy memory sidecar | `saltr/data/salt_rd_memory_sidecar.npz` | ✅ 228 seqs, 0 oracle |
| v2_corrected checkpoint | `saltr/checkpoints/v2_corrected/saltrd_best.pt` | ✅ 28-dim |
| v2_retrained checkpoint | `saltr/checkpoints/v2_corrected/saltrd_best.pt` | ✅ strict baseline |
| v2.1 proxy-memory checkpoint | `saltr/checkpoints/v2_1_memory/saltrd_best.pt` | ✅ 37-dim |
| preds all splits (retrained) | `saltr/results/preds_all_v2_retrained.json` | ✅ 228 seqs |
| ablation summary | `saltr/results/ablation_summary_0521.json` | ✅ |
| eval val v2 retrained | `saltr/results/eval_val_v2_retrained.json` | ✅ canonical |
| eval val v2.1 memory | `saltr/results/eval_val_v2_1_memory.json` | ✅ |
| eval diagnostic v2 retrained | `saltr/results/eval_diagnostic_v2_retrained.json` | ✅ canonical |
| eval diagnostic v2.1 memory | `saltr/results/eval_diagnostic_v2_1_memory.json` | ✅ |
| policy sweep no-mem | `saltr/results/policy_sweep_v2_baseline_nomem.json` | ✅ |
| policy sweep with memory | `saltr/results/policy_sweep_v2_retrained.json` | ✅ |
| e-process val | `saltr/results/eprocess_val_v2_retrained.json` | ✅ analysis only |
| OOF predictions | `saltr/results/preds_all_v2_oof_teacher.json` | ✅ 228 seqs, exact coverage |
| SGLA pos memory sidecar (score_weighted) | `saltr/data/salt_rd_sgla_pos_memory_sidecar.npz` | ✅ 228 seqs |
| SGLA pos memory sidecar (peak_local) | `saltr/data/salt_rd_sgla_pos_memory_sidecar_peak_local.npz` | ✅ 228 seqs |
| v2.2 checkpoint (score_weighted) | `saltr/checkpoints/v2_2_sgla_pos/saltrd_best.pt` | ✅ KILL — diag 0.584 |
| v2.2 checkpoint (peak_local) | `saltr/checkpoints/v2_2_sgla_peak_local/saltrd_best.pt` | ✅ KILL — diag 0.584 |
| DINO dry-run selection | `saltr/results/dino_identity_pilot_dryrun.json` | ✅ 4 diag + 8 balanced val |
| DINO pilot CLS ctx2.0 | `saltr/results/dino_identity_pilot.json` | ✅ FAIL diag 0.514 / val 0.862 |
| DINO pilot CLS ctx1.2 | `saltr/results/dino_identity_pilot_ctx1p2.json` | ✅ FAIL diag 0.563 / val 0.896 |
| DINO pilot CLS ctx3.0 | `saltr/results/dino_identity_pilot_ctx3p0.json` | ✅ FAIL diag 0.518 / val 0.897 |
| DINO pilot CLS ctx1.2 518 | `saltr/results/dino_identity_pilot_ctx1p2_518.json` | ✅ FAIL diag 0.547 / val 0.864 |
| DINO pilot patch_mean ctx1.2 | `saltr/results/dino_identity_pilot_patch_ctx1p2.json` | ✅ FAIL diag 0.560 / val 0.901 |
| Candidate mining diagnostic | `saltr/results/candidate_mining_pilot_diagnostic.json` | ✅ KILL overall top5@0.3=0.298 |

---

## Red Lines (permanent)

- **NO training on diagnostic sequences** (bike2, Gull2, Sheep1, StreetBasketball1, uav0000164)
- **NO calibration on train split**
- **NO comparing micro-tcr (new) with macro-tcr (old policy.py)** — different metrics, not equivalent
- **Strict baseline = v2_retrained** for all future gates — not v2_corrected
- **Proxy memory = "telemetry-proxy ablation"** in any writeup — not "DAM-style memory"
- **Memory: if improving val but NOT diagnostic → overfit**, do not claim real improvement
- **Negative memory: do NOT include in v2.2** — neg_only gave 0.496 AUROC (sub-random); only validated as part of full 9-dim with positive context
- **E-process: analysis/monitoring only** — 3.1% recall is not a runtime gate
- **Policy: use runner-up config** (mem_t=0.00), not best (mem_t=0.20 blocks 67% safe updates)
- **OOF required for final claim** — train sidecar using preds_all_v2_oof_teacher.json, not preds_train in-sample
- **Do not remove TSA directly** — transition through Shadow → Advisory/Veto → Primary Learned Controller
- **Do not build full DINO sidecar from first-pass CLS/patch_mean** — diagnostic gate failed; next DINO work must first prove a hard-case signal.
- **Per-dataset reporting required** for all future GO/KILL decisions — pooled val is not enough.
- **Do not treat SGLATrack top-K as sufficient candidate generator** — diagnostic top-5 IoU≥0.3 recall is 0.298 overall and 0.000 on bike2/Gull2.

---

## Fallback Decision Tree

```
Phase 2: Real SGLA embeddings
  │
  ├─ score_weighted weak → peak_local weak
  │
  ├─ SGLA views weak (diag fc < baseline) → DINOv2 ROI pilot
  │
  ├─ DINOv2 first-pass weak on diagnostic
  │     → render hard-case crop panels
  │     → bbox-centered / foreground patch pooling
  │     → contrastive distractor head
  │     → CoTracker3/TAPIR/RAFT cycle teacher
  │
  └─ Val recovers but diag stays below 0.774 → representation quality issue
        → add contrastive ranking loss: sim(pos_mem, crop) > sim(neg_mem, crop) + margin
        → OR add top-2 response peak margin / secondary peak crop similarity

Phase 4C: SGLATrack score-map candidates
  │
  ├─ Sheep1 works (top5@0.3 = 1.000)
  ├─ bike2/Gull2/StreetBasketball fail (target absent from top-K)
  └─ Keep hook as telemetry, but move to external/teacher candidate source

Phase 5: External/teacher candidate verifier
  │
  ├─ CoTracker3/TAPIR/RAFT point teacher gives fc signal
  │     → train v2.4 candidate sidecar
  │
  ├─ point teacher weak on tiny targets
  │     → crop-upscale + RAFT/GMFlow cycle consistency
  │
  └─ candidate still absent
        → triggered detector/SAM/TAM candidate generator

Phase 6: Controller plateaus
  └─ Open separate SALT-AUC track: LoRAT/fine-tune SGLATrack backbone
     SALT-RD claim remains: trust controller reduces template corruption / wrong reinit

Phase 7: TSA → SALT-RD transition
  │
  ├─ Stage 1 Shadow: SALT-RD observes only; TSA controls everything
  │
  ├─ Stage 2 Advisory/Veto: SALT-RD blocks risky updates/reinit/full-compute choices
  │                    but never moves bbox or reinitializes alone
  │
  └─ Stage 3 Primary Learned Controller: SALT-RD owns state/action policy,
       TSA remains fallback until LODO + replay + runtime rollout pass
```

---

## Research Trajectory

```
1. ✅ fc signal works (AUROC 0.885 val, 0.796 diag with proxy)
2. ✅ Memory direction validated (diag fc 0.598 → 0.796)
3. ✅ pos_only is key driver (0.774 diag AUROC, 89% of full gain)
4. ✅ Real SGLATrack embeddings tested and killed (score_weighted/peak_local diag 0.584)
5. ✅ DINOv2 first-pass ROI identity tested: val strong, hard diagnostic weak
6. ✅ SGLATrack candidate mining tested: Sheep1 positive, overall hard diagnostic KILL
7. ✅ CoTracker3 KILLED (memory crash, marginal +0.019 over LK); LK/Farneback pass gate 0.65
8. ✅ Shadow Mode Stage 1: val cov=45.2%/FAR=5.9%; diag FAR high on Gull2/SB1 (outlier scenes)
9. ⏳ LK point sidecar extraction (228 seq) + train SALT-RD v2.3 (33-dim telemetry+point)
10. ⏳ Re-run shadow mode with v2.3 to check diagnostic FAR improvement
11. ⏳ LODO generalization (mandatory before paper)
12. ⏳ Runtime rollout (conservative: block template only first)
13. ⏳ TSA-to-SALT-RD migration: Shadow → Advisory/Veto → Primary Learned Controller
```

---

## SALT Tracker State (from SESSION_SUMMARY.md)

For tracker architecture, benchmark results, and CE/UTPTrack analysis from sessions 2026-05-18/19, see `SESSION_SUMMARY.md`.

Current SALT v3: AUC 0.720 on full UAV123 (123 seqs), 0.672 VisDrone-SOT, CE kr=0.50, 62 fps.

---

## Starting Prompt for Next Session

```
Read HANDOFF_NEXT.md.

STATE:
- Tests green: `PYTHONPATH=src:saltr/src .venv/bin/python -m pytest tests/unit/test_saltr_*.py -q` → 230 passed
- Canonical handoff is HANDOFF_NEXT.md

Phase 5A COMPLETE:
- CoTracker3 KILLED — crashed laptop RAM, marginal +0.019 AUROC over LK
- LK passes gate: pt_inside_pred_ratio diag AUROC = 0.729 > 0.65
- Farneback also passes: 0.739 (slightly better)
- Per-seq: Gull2 best = pt_median_motion/fbe (LK 0.920), Sheep1 = pt_inside (LK 0.748), bike2 = pt_cluster_area (LK 0.808)
- StreetBasketball1 and Gull2 are hard outlier scenes — test separately, not in main gate

Phase 5B READY — point_sidecar_extractor.py implemented:
- saltr/src/salt_r/point_sidecar_extractor.py: LK/Farneback, PRED bbox seeding, sliding-window stride=15 window=25
- 16 unit tests pass; fail-fast on any skip; --smoke-test / --allow-partial flags
- OUTPUT NPZ schema: point_features/{seq} (T, 13), extractor_method, stride, window, source_npz_md5

Shadow Mode Stage 1 COMPLETE (v2.1 + proxy memory):
- saltr/results/shadow_mode_diagnostic_v2_1.json
- saltr/results/shadow_mode_val_v2_1.json
- BUG: stored aggregate coverage_block_when_fc=0.0 was wrong (old version missing n_true_block)
- REAL val: coverage=45.2%, FAR=5.9% ← good
- REAL diagnostic: coverage=62.8%, FAR=31.7% ← high FAR driven by Gull2/StreetBasketball1 (outlier)
- shadow_mode.py fixed: agg loop now guards None values

PENDING (two parallel tracks):

Track A — Phase 5B extraction + train v2.3:
1. Smoke: run point_sidecar_extractor --smoke-test 2, verify shape/nan fraction
2. Full extraction (228 seq, ~20-40 min): --method lk --stride 15 --window 25
3. Wire point sidecar into train.py (new --point-sidecar arg or extend --memory-sidecar)
4. Train v2.3 with 5 point features: pt_inside_pred_ratio,pt_cluster_area_ratio,
   pt_bbox_center_disagreement,pt_forward_backward_error,pt_median_motion
5. Eval gate: diag fc AUROC > 0.774 AND val fc ≥ 0.870

Track B — Shadow Mode continuation:
1. Re-run shadow mode with fixed code to get correct aggregate JSON
2. After v2.3 trained: re-run shadow mode with v2.3 checkpoint to check diagnostic FAR
3. Stage 2 advisory/veto prep: design --mode advisory flag for shadow_mode.py

RED LINES:
- Use v2_retrained as strict baseline (not v2_corrected) for all gate comparisons
- CoTracker3: do not re-introduce; LK/Farneback sufficient
- No full DINO sidecar until hard diagnostic pilot passes (separate from point sidecar)
- No SGLATrack-only candidate verifier; diagnostic oracle failed except Sheep1
- Neg memory: do not include in v2.2/v2.3
- E-process: monitoring only
- OOF preds required for final claim
- StreetBasketball1 and Gull2: hard outlier scenes, report separately, not in main gate
```
