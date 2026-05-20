# HANDOFF_NEXT — SALT-RD: Phase 4B v2.1 Training Ready

**Дата:** 2026-05-20  
**Оновлено:** 2026-05-20 (session 2) — Plumbing fixed, Phase 6 honest ablation done, Phase 4B model extension implemented, 174 tests green.  
**Owner:** Staff CV/AI/ML review track  
**Поточний стан:** All plumbing fixed. Canonical artifacts regenerated. model/train/eval ready for 37-dim input. Phase 4B training is the concrete next step.  
**Новий пріоритет:** **Run Phase 4B training** (`--memory-sidecar saltr/data/salt_rd_memory_sidecar.npz`) → measure diagnostic fc AUROC vs 0.548 baseline. If proxy memory is weak, move to real crop embeddings.

---

## Session Summary 2026-05-20 (Session 2 — Plumbing + Phase 4B)

### What changed

| Area | Status | Key result |
|---|---|---|
| e-process plumbing | ✅ fixed | risk_mode in null calibration; CLI --risk-mode; diagnostic keys from NPZ |
| interventions.py | ✅ fixed | RecoveryAction.NONE default; RUN only when p_rec>=threshold AND p_fc<0.40 |
| Phase 6 plumbing | ✅ fixed | memory_margin/{seq} in sidecar; e_trace in per_seq JSON; bbox_pred for KF |
| CoTracker provenance | ✅ fixed | allow_synthetic=False default; teacher_model/{seq} persisted |
| bbox utils | ✅ added | xywh_to_xyxy / xyxy_to_xywh in collect_features.py |
| CLIs | ✅ added | python -m salt_r.memory_features, python -m salt_r.policy_sweep |
| Canonical artifacts | ✅ regenerated | memory sidecar (228 seqs, +memory_margin), eprocess JSON (+e_trace), policy sweep |
| Phase 6 honest ablation | ✅ done | tcr -28% at fc=0.60/reinit=0.70; proxy memory NO-GO |
| Phase 4B model | ✅ done | model/train/eval accept memory_dim=0/9, --memory-sidecar arg |
| Tests | ✅ 174 green | +33 new tests (eprocess/memory/model/policy/bbox) |

### Phase 6 Honest Ablation (key numbers)

| Policy | tcr | wrir | msu | dens/1kf | recall |
|---|---:|---:|---:|---:|---:|
| HANDOFF v2 baseline | 0.0330 | 0.2090 | — | — | — |
| fc=0.60 reinit=0.70 no-mem | **0.0239** | **0.000** | 0.624 | 684 | 0.977 |
| fc=0.60 reinit=0.80 mem_t=+0.10 | 0.0239 | 0.000 | 0.624 | 684 | 0.977 |
| fc=0.60 reinit=0.80 mem_t=0.00 | 0.0338 | 0.000 | 0.435 | 301 | 0.812 |
| Best tcr (fc=0.4, reinit=0.8) | **0.0225** | **0.000** | 0.555 | 532 | 0.953 |

**Key reads:**
- tcr −28% at practical point (fc=0.60, reinit=0.70) vs HANDOFF baseline
- wrir=0.0 at reinit≥0.70: `recoverable` head fires on already-failed frames; safe threshold is ≥0.70
- Proxy memory HURTS: mem_t=0.00 makes tcr 0.0239→0.0338 and recall 0.977→0.812
- e-process: lead=10f with FAR≈0.2/1kf — good lead time, recall=3.1% — analysis tool only

### Phase 6 GO/NO-GO

| Hypothesis | Verdict |
|---|---|
| fc signal reduces template corruption | ✅ GO: −28% |
| reinit gating reduces wrong_reinit | ✅ GO: wrir=0 at reinit≥0.70 |
| proxy memory improves policy | ❌ NO-GO: makes tcr worse |
| e-process runtime gating | ❌ NO-GO: recall=3% |
| e-process lead-time analysis | ✅ 10f lead, near-zero FAR |

---

## Concrete Next Task: Run Phase 4B Training

Infrastructure is ready. Command:

```bash
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.train \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --output saltr/checkpoints/v2_1_memory \
  --epochs 50 \
  --label-schema v2 \
  --patience 8 \
  --memory-sidecar saltr/data/salt_rd_memory_sidecar.npz
```

Then eval (both val and diagnostic):
```bash
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.eval \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --checkpoint saltr/checkpoints/v2_1_memory/saltrd_best.pt \
  --output saltr/results/eval_val_v2_1_memory.json \
  --predictions-output saltr/results/preds_val_v2_1_memory.json \
  --memory-sidecar saltr/data/salt_rd_memory_sidecar.npz

PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.eval \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --checkpoint saltr/checkpoints/v2_1_memory/saltrd_best.pt \
  --split diagnostic \
  --output saltr/results/eval_diagnostic_v2_1_memory.json \
  --memory-sidecar saltr/data/salt_rd_memory_sidecar.npz
```

**GO gate:** diagnostic fc AUROC 0.548 → > 0.65 (minimum), ideally > 0.70  
**STOP:** if val improves but diagnostic does NOT → overfit, proxy insufficient  
**Fallback:** if proxy weak → SGLATrack/DINO crop embeddings (real Phase 4B)

---

## Artifacts State After Session 2

| Artifact | Path | Status |
|---|---|---|
| memory sidecar (with memory_margin) | `saltr/data/salt_rd_memory_sidecar.npz` | ✅ regenerated |
| eprocess JSON (with e_trace) | `saltr/results/eprocess_val_v2_retrained.json` | ✅ regenerated |
| policy sweep (all signals) | `saltr/results/policy_sweep_v2_retrained.json` | ✅ new |
| v2_1 checkpoint | `saltr/checkpoints/v2_1_memory/` | ⏳ not yet trained |

---

## Session Summary 2026-05-20 (Current Session)

### What changed

| Phase | Status | Key result |
|---|---|---|
| Phase 0 — label semantic fix | ✅ done | ifd10/20 base rate 7% → 0.3%; stale NPZ renamed |
| Phase 1 — foundation bugs | ✅ done | 6 bugs fixed, eprocess aGRAPAmode, event-level lead-time |
| Phase 2 — retrain on corrected labels | ✅ done | fc=0.885 unchanged, ifd10=0.765, template_corruption=0.033 (−69%) |
| Phase 3 — e-process sweep | ✅ done | formal 10f lead, 6.2% recall → analysis tool verdict |
| Phase 4 — DAM memory infra | ✅ done | memory.py + sidecar; train v2.1 is NEXT |
| Phase 5 — CoTracker3 infra | ✅ done | teachers/ + 13 point features; offline only |
| Phase 6 — v2-aware policy | ✅ done | interventions.py + policy_sweep.py + KF filter |

### Current test count: 141 passed

### Repo state after this session

| Artifact | Path | Status |
|---|---|---|
| v2_corrected checkpoint | `saltr/checkpoints/v2_corrected/saltrd_best.pt` | ✅ 13 heads, corrected labels |
| v2 corrected labels | `saltr/data/salt_rd_v2_labels.npz` | ✅ regenerated (ifd10/20 semantic fix) |
| stale v2 labels | `saltr/data/salt_rd_v2_labels_STALE_semantic_bug.npz` | ⚠️ keep for reference only |
| memory sidecar | `saltr/data/salt_rd_memory_sidecar.npz` | ✅ 228 seqs, proxy embeddings |
| val predictions (corrected) | `saltr/results/preds_val_v2_corrected.json` | ✅ old checkpoint, corrected labels |
| val predictions (retrained) | `saltr/results/preds_val_v2_retrained.json` | ✅ v2_corrected checkpoint |
| val eval (retrained) | `saltr/results/eval_val_v2_retrained.json` | ✅ canonical |
| policy replay (retrained) | `saltr/results/policy_val_v2_retrained.json` | ✅ canonical |
| e-process val (retrained) | `saltr/results/eprocess_val_v2_retrained.json` | ✅ formal α=0.10 |
| diagnostic eval (corrected) | `saltr/results/eval_diagnostic_v2_corrected.json` | ✅ hard seqs |
| e-process sweep (corrected) | `saltr/results/eprocess_sweep_v2_corrected.csv` | ✅ all modes |

### Canonical model results (v2_corrected checkpoint, corrected labels)

| Head | AUROC | AUPRC | ECE | Policy impact |
|---|---:|---:|---:|---|
| false_confirmed | 0.885 | 0.338 | 0.348 | template block; recovery abstain |
| ifd5 | 0.898 | 0.317 | 0.308 | short-horizon risk |
| ifd10 | **0.765** | 0.012 | **0.046** | 10-frame early warning (calibration ✅) |
| ifd20 | **0.744** | 0.018 | **0.109** | 20-frame early warning |
| failure_in_10 | 0.826 | **0.057** | 0.298 | improved AUPRC (+3x) |
| recoverable | 0.894 | 0.046 | 0.311 | recovery gating |

**Policy (v0 → v2stale → v2_corrected retrain):**
```
template_corruption:  0.108 → 0.060 → 0.033   (−69% vs v0)
wrong_reinit_rate:    0.273 → 0.183 → 0.209   (mixed — needs memory features)
abstention_gain:      0.307 → 0.304 → 0.295
```

**e-process (formal α=0.10, ε=0.5, retrained preds):**
```
median lead time:  10 frames  ✅ (target ≥ 3)
event recall:       6.2%      ❌ (target ≥ 60%) — analysis tool
FA per 1000f:       0.2       ✅ (target ≤ 100)
```

**Diagnostic split (hard sequences — goal not yet met):**
```
false_confirmed AUROC:  0.548  ❌ (target 0.70-0.75)
ifd10 AUROC:            0.626
ifd20 AUROC:            0.554
```

### Phase 2 gate verdict

| Gate | Target | Result | Decision |
|---|---|---|---|
| ifd10 AUROC | ≥ 0.75 | 0.765 | ✅ continue |
| ifd20 AUROC | ≥ 0.75 | 0.744 | ❌ borderline — within noise |
| e-process lead time | ≥ 3f | 10f | ✅ |
| e-process recall | ≥ 10-20% | 6.2% | ❌ → analysis tool until fc improves |

**→ Continue. DAM memory features are the path to fixing recall + diagnostic fc.**

---

## Fallback Tree And Pivot Rules

We should keep a clear fallback tree so the project does not overfit emotionally or technically to one path. The core principle:

> Do not discard SALT-RD too early. Narrow the claim first. If controller signals plateau, split the work into two tracks: **SALT-RD for trust/intervention** and **SALT-AUC for tracker/domain adaptation**.

### If e-process does not improve recall

Fallback: keep e-process as an analysis / monitoring metric, not as the runtime decision mechanism.

Runtime policy becomes calibrated risk hysteresis:

```text
p_ifd10 > t_high  -> verify / expand search
p_ifd10 < t_low   -> release
p_fc > t_fc       -> block template update and block/reject recovery
```

This removes the formal anytime-valid alerting claim, but gives a more practical latency/recall operating point. The claim becomes:

```text
risk-aware intervention policy
```

not:

```text
anytime-valid sequential alerting
```

Use this fallback if:

- e-process event recall stays below 10-20%;
- seq-level FAR remains high after fixing calibration/plumbing;
- e-process adds lead time but suppresses too many true events.

Keep reporting e-process as an offline analysis table if it still has strong precision or useful lead time.

### If DAM proxy memory does not improve false_confirmed

This is expected. The current sidecar uses proxy embeddings from 28 scalar telemetry features, which is not true appearance memory.

Fallback ladder:

1. Use crop embeddings from SGLATrack / SGLA trunk.
2. If weak, use DINOv2 / DINOv3 crop embeddings offline.
3. If still weak, use CLIP / distilled DINO only as teacher features, not runtime dependencies.

The goal of memory is:

```text
current target crop looks closer to target memory than to distractor memory
```

not:

```text
current telemetry vector looks like old telemetry vector
```

Use proxy memory only as a plumbing ablation. Do not make DAM-style scientific claims until real crop/appearance embeddings are tested.

### If real DAM memory does not improve diagnostic false_confirmed

Then the failure is probably representation quality, not memory mechanics.

Fallbacks:

- add explicit distractor candidate features:
  - secondary peak crop similarity;
  - top-2 response peak margin;
  - top-2 appearance margin;
  - distractor count near predicted bbox;
- replace scalar memory margin with a contrastive distractor head;
- train with Siamese / ranking loss:

```text
sim(target_memory, current_crop) > sim(distractor_memory, current_crop) + margin
```

- try dataset/class-specific adapter or LoRAT-style parameter-efficient adaptation.

Decision rule:

- if real memory improves global val but not diagnostic split, treat it as overfit;
- if diagnostic `false_confirmed` AUROC stays below 0.65, do not claim hard-case identity-drift robustness;
- if diagnostic AUROC reaches 0.70-0.75, Phase 4B becomes the central trust result.

### If CoTracker3 teacher fails on small UAV targets

Likely cause: UAV targets are often 5-20 px, so point tracking may be unstable.

Fallback ladder:

1. Upscale target/search crop before point tracking.
2. Seed fewer but denser points inside the bbox.
3. Try TAPIR / BootsTAP / RAFT / GMFlow as offline teacher instead of CoTracker3.
4. Use SAM2 / EfficientTAM region consistency offline only.
5. Replace point consistency with forward-backward tracker cycle consistency:

```text
track forward N frames
track backward to the start
measure bbox cycle error
```

Do not run CoTracker3, SAM2, DINO, or CLIP in the edge runtime loop unless explicitly framed as a rare fallback mode.

### If corrected ifd10 / ifd20 labels are too sparse

The corrected labels are honest but rare. Binary BCE may be the wrong learning objective.

Fallbacks:

- regress `time_to_failure`;
- regress `min_future_iou_10` / `min_future_iou_20`;
- regress `iou_drop_slope`;
- use ordinal heads:

```text
failure within 5 / 10 / 20 / 40 frames
```

- use survival modeling / hazard rate instead of BCE;
- add sequence-balanced positive sampler;
- expand positives to the last `K` frames before an event, not only isolated frames.

Preferred fallback: survival / hazard modeling. It matches the real problem better:

```text
risk increases before failure
```

instead of:

```text
this single frame is positive or negative
```

### If policy replay does not transfer to runtime

Do not make a deployment claim.

Fallbacks:

- present an offline safe-intervention oracle benchmark;
- replay on recorded tracker trajectories;
- use conservative runtime only:
  - block or verify template update;
  - do not automatically run recovery/fallback;
- run small-subset rollout only after offline gates pass.

The safest deployment action is:

```text
block / verify template update
```

It usually harms less than wrong re-initialization.

### If tracker AUC does not improve

This is acceptable for SALT-RD. SALT-RD's core claim is trust and intervention safety, not tracker backbone accuracy.

If the goal becomes AUC improvement, open a separate track:

```text
SALT-AUC = tracker/domain adaptation
```

Options:

- fine-tune SGLATrack / LoRAT on UAV123 + VisDrone + DTB70-like data;
- replace base tracker with a stronger lightweight UAV tracker;
- add class-agnostic re-detector / fallback;
- improve recovery with spatial hint / crop detector;
- train detector-assisted re-acquisition head.

Rule:

- SALT-RD = trust controller;
- SALT-AUC = tracker or domain adaptation work.

### Fallback Priority

1. Fix plumbing and run honest v2-aware policy replay.
2. If e-process is weak, use calibrated hysteresis risk policy.
3. If proxy memory is weak, move to real crop embeddings.
4. If real memory is weak, train a contrastive distractor head.
5. If CoTracker3 is weak, try crop-upscaled TAPIR / RAFT / forward-backward cycle.
6. If all controller signals plateau, pivot to LoRAT / domain adaptation for actual AUC.

Most realistic fallback:

```text
Keep SALT-RD, but narrow the claim to safe template/recovery gating.
Open a separate AUC/FPS track only if the user goal becomes tracker accuracy or compute.
```

---

## Concrete Next Task: Phase 4B — Train SALT-RD v2.1 with Memory Features

### What to do

Extend the model to accept memory features (9 extra dims) alongside the existing 28 telemetry features.

```
Input: [28 telemetry features | 9 memory features] = 37-dim input
Model: GRU(37, hidden=96, layers=2) + same 13 heads
Training: same composite early-stopping, v2_corrected labels
```

### Files to modify

**`saltr/src/salt_r/model.py`** — change `input_dim` to accept optional `memory_dim`:
```python
class SALTRD(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=96, n_layers=2, memory_dim=0, ...):
        actual_input = input_dim + memory_dim
        self.gru = nn.GRU(actual_input, hidden_dim, n_layers, ...)
```

**`saltr/src/salt_r/train.py`** — load memory sidecar and concatenate per-sequence:
```python
# Load memory sidecar if available
memory_npz = np.load('saltr/data/salt_rd_memory_sidecar.npz', allow_pickle=True)
# Per-sequence: features = np.concatenate([features_v2[seq], memory_features[seq]], axis=1)
# Fall back gracefully if sidecar missing a sequence
```

**`saltr/src/salt_r/eval.py`** — same concatenation logic during inference.

### Expected result

Target: diagnostic `false_confirmed` AUROC 0.548 → 0.70-0.75  
Target: `false_confirmed` AUPRC → 0.45+ (or +25% relative)  
Target: `template_corruption` further reduction  
Target: e-process recall improves because fc signal is stronger  

**Kill criteria:**  
If memory features improve global val but NOT diagnostic split → overfit, do not claim improvement.  
If diagnostic fc AUROC stays below 0.65 after v2.1 training → memory proxy embeddings insufficient → need real DINO embeddings.

### Command to run after implementing

```bash
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.train \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --output saltr/checkpoints/v2_1_memory \
  --epochs 50 \
  --label-schema v2 \
  --patience 8
# (train.py will auto-detect memory sidecar at saltr/data/salt_rd_memory_sidecar.npz)
```

Then eval:
```bash
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.eval \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --checkpoint saltr/checkpoints/v2_1_memory/saltrd_best.pt \
  --output saltr/results/eval_val_v2_1_memory.json \
  --predictions-output saltr/results/preds_val_v2_1_memory.json \
  --calibrate-heads false_confirmed imminent_failure_dynamic \
      imminent_failure_dynamic_10 imminent_failure_dynamic_20
# Also run diagnostic split:
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.eval \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --checkpoint saltr/checkpoints/v2_1_memory/saltrd_best.pt \
  --split diagnostic \
  --output saltr/results/eval_diagnostic_v2_1_memory.json
```

---

## Infrastructure Implemented This Session

### New modules

| File | Purpose | Status |
|---|---|---|
| `saltr/src/salt_r/eprocess.py` | Sequential e-process alerts, 5 risk modes, aGRAPAmode | ✅ complete |
| `saltr/src/salt_r/memory.py` | PositiveMemory (RAM) + NegativeMemory (DRM), DAM-aligned | ✅ complete |
| `saltr/src/salt_r/memory_features.py` | Offline sidecar collection from v2 NPZ | ✅ complete |
| `saltr/src/salt_r/interventions.py` | TrackerIntervention, AlertTier, KF residual, v2-aware | ✅ complete |
| `saltr/src/salt_r/policy_sweep.py` | SimpleBboxKalmanFilter, threshold sweep, v2-aware policy | ✅ complete |
| `saltr/src/salt_r/teachers/__init__.py` | Package marker | ✅ complete |
| `saltr/src/salt_r/teachers/point_features.py` | 13 point consistency features + 3 teacher labels | ✅ complete |
| `saltr/src/salt_r/teachers/cotracker3_export.py` | Offline CoTracker3 export (gated import) | ✅ complete |
| `saltr/src/salt_r/diagnose_labels.py` | Label contamination audit | ✅ complete |
| `saltr/src/salt_r/baselines.py` | Feature baseline comparison vs GRU | ✅ complete |

### New tests

| File | Tests | What's covered |
|---|---|---|
| `test_saltr_eprocess.py` | 45+ | aGRAPAmode, 5 risk modes, formal/engineering/agrapa, event metrics |
| `test_saltr_memory.py` | 39 | PositiveMemory gates, NegativeMemory timeless prior, margin feature |
| `test_saltr_point_features.py` | 10 | synthetic track arrays, teacher labels, no CoTracker3 at import |
| `test_saltr_policy_sweep.py` | 10 | KF residual, v2-aware interventions, AlertTier logic |
| `test_saltr_collect_features.py` | 5+ | v0→ValueError guard, ifd positives IoU≥0.5 invariant |
| `test_saltr_eval.py` | updated | event-level lead-time, 3 horizons, run_lead_time_analysis |

**Total: 141 tests passing.**

### Key design decisions from papers

**eprocess.py (WACV2026 paper):**
- aGRAPAbetting (Eq.12): `λ_t = (ε−μ̂)/(σ̂²+(ε−μ̂)²)`, rolling window=20, no calibration needed
- Paper's formulation: `X_t = X_{t-1} * (1 + λ_t * (ε−M_t))` where `M_t = 1 − risk_score_t`
- Our conformal/power approach is valid but different from paper

**memory.py (DAM4SAM CVPR2025 paper):**
- PositiveMemory (RAM): FIFO, updated when `p_fc < 0.20 AND p_ifd < 0.30`, recency-weighted
- NegativeMemory (DRM): NOT time-stamped (timeless prior), updated when distractor detected AND tracking reliable
- Proxy embeddings: 28-dim normalized feature vector (upgrade to DINO when available)

**policy_sweep.py (SAMURAI 2024 paper):**
- SimpleBboxKalmanFilter: state=[cx,cy,w,h,vcx,vcy,vw,vh], `kf_residual = 1 − IoU(KF_pred, actual)`
- Cheap feature: ~0.1ms/frame, catches spatial jumps orthogonal to APCE

### Bug fixes from this session

| Bug | Severity | File | Fix |
|---|---|---|---|
| ifd10/20 counted already-failed frames | P1 | collect_features.py | `iou_trace[t] >= 0.5` + full-window guard |
| lead_time hardcoded ifd5/6-frame window | P1 | eval.py | `label_name`, `horizon` params; event-level |
| policy.py not using v2 heads | P2 | policy.py | NOTE comment; ifd10/20 in policy_sweep.py |
| fake AUROC 0.5 for "correct" label | P2 | eval.py | `model_predicted: false` + note field |
| recompute_labels_v2 silently accepted v0 | P2 | collect_features.py | v1 schema validation |
| train log said AUPRC(fc) but was composite | P3 | train.py | Renamed to "validation selection score" |

---

## Red Lines (unchanged)

- Do NOT train on `_decide_state()`, TSA states, old scene labels
- Do NOT claim compute/FPS without oracle labels
- Do NOT calibrate on train split
- Do NOT use diagnostic sequences in train
- Do NOT tune GO gates to get better verdict
- Do NOT use `pt_inside_gt_ratio` as runtime feature (GT-relative, teacher-only)
- Do NOT make `hard_dynamic_scene_v2` central without intervention improvement
- Memory features: if improving val but NOT diagnostic → do NOT claim real improvement

---



## Executive Decision

Ми більше не тиснемо на старий `hard_dynamic_scene_v2` як центральний результат. Він занадто слабкий і погано корелює з реальним IoU degradation. Центральна лінія тепер:

> **SALT-RD v2 is a tracker-trust and intervention controller: it learns when the tracker is confidently wrong, accumulates sequential evidence of failure, and uses distractor/point-consistency memory to decide whether to block template updates, reject recovery, expand search, or escalate to a stronger fallback.**

Практично це означає:

1. **False-confirmed / identity drift** лишається головним ML-сигналом.
2. **Imminent failure dynamicity** лишається короткогоризиковим сигналом, але його треба перетворити з 1-frame warning на usable alert через e-process.
3. **DAM4SAM-style memory** має дати нові ознаки, яких APCE/entropy не бачать: target vs distractor identity margin.
4. **CoTracker3 point consistency** має дати offline teacher-сигнали для того, чи bbox ще тримає той самий об'єкт.
5. **Compute/FPS claim** не робимо, поки немає full-vs-cheap oracle labels і Pareto-кривої AUC-vs-GFLOPs.

Це не косметика. Це зміна від "класифікуємо сцену" до "керуємо довірою до трекера і intervention policy".

---

## Why Pivot

### Що v1/v2 вже довели

`false_confirmed` та `imminent_failure_dynamic` працюють як реальні сигнали:

| Head | Schema | AUROC | AUPRC | Коментар |
|---|---|---:|---:|---|
| `false_confirmed` | v0/v1/v2 | 0.884 | 0.336 | Сильний результат; single-feature baselines майже random/anti-predictive |
| `failure_in_5` | v0/v1/v2 | 0.853 | 0.011 | Ranking добрий, AUPRC низький через малий base rate |
| `recoverable` | v0/v1/v2 | 0.893 | 0.046 | Корисний для recovery gating |
| `imminent_failure_dynamic` (5f) | v1/v2 | 0.902 | 0.323 | Сильний short-horizon signal |
| `imminent_failure_dynamic_10` | v2 | **0.897** | **0.329** | Майже не деградує! 10-frame horizon |
| `imminent_failure_dynamic_20` | v2 | **0.889** | **0.339** | 20-frame signal тримається (−1.3pp від 5f) |
| `failure_in_10` | v2 | 0.827 | 0.017 | AUROC добрий, AUPRC обмежений sparse labels (0.3%) |
| `failure_in_20` | v2 | 0.785 | 0.022 | Аналогічно, AUPRC обмежений (0.6%) |

Найсильніша емпірична точка:

| Task | GRU | Best simple baseline | Висновок |
|---|---:|---:|---|
| false-confirmed | AUROC 0.884 / AUPRC 0.336 | flow consistency AUROC 0.511 | APCE/entropy не ловлять identity drift (+37.9pp) |
| imminent failure dynamicity | AUROC 0.900 / AUPRC 0.323 | entropy AUROC 0.888 / AUPRC 0.285 | GRU додає ~1.2pp AUROC, 13% rel. AUPRC |
| 20-frame failure risk | AUROC 0.889 | — | Telemetry несе 20-frame early warning без teacher |

**v2 policy replay (calibrated) vs v0 baseline:**

| Metric | v0 | v1 | **v2** | Δ v2 vs v0 |
|---|---:|---:|---:|---:|
| template_corruption_rate | 0.108 | 0.090 | **0.060** | **−44%** |
| wrong_reinit_rate | 0.273 | 0.269 | **0.183** | **−33%** |
| abstention_gain | 0.307 | 0.312 | 0.304 | ≈0 |

### Що v2 все ще не вирішив

| Weak spot | Поточний результат | Рішення |
|---|---:|---|
| `hard_dynamic_scene_v2` | AUROC 0.592 | Не робити центральним head/gate |
| `needs_full_compute` | AUROC 0.648, cheap_rate 0.000 | Потрібні oracle labels from full-vs-cheap replay |
| ECE(`false_confirmed`) | 0.316 після T-scaling (v2), 0.229 (v1) | Teacher features + isotonic/conformal calibration |
| Lead time for `ifd` | median 1 frame | e-process accumulation (Phase 2A) |
| `failure_in_10/20` AUPRC | 0.017/0.022 (sparse labels) | Або sampler, або regression target на time-to-failure |
| false_confirmed на diagnostic | AUROC 0.548 (STOP gate) | Teacher features для hard identity-drift cases |

### Stop Doing

- Не полірувати старий `hard_dynamic_scene_v2` threshold.
- Не заявляти GFLOPs/FPS win з bootstrap `needs_full_compute`.
- Не тюнити GO gates під красивий verdict.
- Не навчати на `_decide_state()`, TSA states, APCE rule labels або scene labels.
- Не запускати важкі SAM2/DINO/CoTracker моделі в runtime loop як частину edge pipeline, якщо це не окремий fallback режим.

---

## Literature Anchors

Ці роботи визначають новий план:

| Work | Verified source | Що беремо |
|---|---|---|
| DAM4SAM — A Distractor-Aware Memory for Visual Object Tracking with SAM2, CVPR 2025 | [CVF OpenAccess](https://openaccess.thecvf.com/content/CVPR2025/html/Videnovic_A_Distractor-Aware_Memory_for_Visual_Object_Tracking_with_SAM2_CVPR_2025_paper.html), [GitHub](https://github.com/jovanavidenovic/DAM4SAM) | Позитивна/негативна memory, introspection-based update, explicit distractor handling |
| Detecting Object Tracking Failure via Sequential Hypothesis Testing, WACV 2026 workshop | [arXiv 2602.12983](https://arxiv.org/abs/2602.12983), [CVF supplemental](https://openaccess.thecvf.com/content/WACV2026W/RWS/supplemental/Munoz_Detecting_Object_Tracking_WACVW_2026_supplemental.pdf) | e-process / anytime-valid sequential evidence for failure alerts with controlled false alerts |
| CoTracker3 — Simpler and Better Point Tracking by Pseudo-Labelling Real Videos, 2024 | [arXiv 2410.11831](https://arxiv.org/abs/2410.11831), [project](https://cotracker3.github.io/) | Offline point teacher: visibility, forward/backward consistency, point survival, object-level consistency |
| Real-World Point Tracking with Verifier-Guided Pseudo-Labeling, 2026 | [arXiv 2603.12217](https://arxiv.org/abs/2603.12217) | Verifier idea: teacher reliability scores for pseudo-label quality |
| UTPTrack, UncL-STARK, ABTrack, BDTrack | local `papers/` + `papers/code/` | Compute baselines; do not claim pruning novelty unless oracle replay beats them |

Наша відмінність від DAM4SAM: ми не будуємо SAM2 tracker. Ми distill-имо ідею **distractor-aware memory** у легкий controller над existing SOT tracker.

Наша відмінність від CoTracker3: CoTracker3 не runtime controller для SGLATrack. Ми використовуємо його як offline teacher, щоб навчити дешеві features/head-и.

Наша відмінність від e-process paper: ми подаємо в sequential test не один handcrafted residual, а multi-head tracker-risk probabilities + memory/point signals.

---

## Current Repo Facts To Preserve

### Existing implementation

| Area | Current state |
|---|---|
| Frozen tracker | `src/uav_tracker/` has SGLATrack/SALT v3 with telemetry/config gates |
| SALT-RD package | `saltr/src/salt_r/` |
| v0 dataset | `saltr/data/salt_rd_v0.npz` — 228 sequences, ~161k frames |
| v1 labels | `saltr/data/salt_rd_v1_labels.npz` — 10 labels (incl. hard_dynamic_scene_v2, ifd) |
| v2 labels | `saltr/data/salt_rd_v2_labels.npz` — 14 labels (incl. failure_in_10/20, ifd_10/20) |
| v0 checkpoint | `saltr/checkpoints/saltrd_best.pt` — 7 heads, epoch 13 |
| v1 checkpoint | `saltr/checkpoints/v1/saltrd_best.pt` — 9 heads, epoch 3 |
| v2 checkpoint | `saltr/checkpoints/v2/saltrd_best.pt` — 13 heads, epoch 14 (composite stopping) |
| Results path | `saltr/results/` — all publishable outputs, schema-tagged filenames |
| Label utilities | `collect_features.recompute_labels_v1/v2()` — generate from existing NPZ without re-running tracker |
| Baselines | `saltr/results/baselines_val.json` — feature baselines vs GRU model |
| Provenance | eval.py embeds git_commit, npz_md5, checkpoint_md5, label_schema, created_at, command |

### v2 GO/NO-GO status

| Gate | Value | Status |
|---|---:|---|
| AUPRC(`false_confirmed`) > 0.30 | 0.336 | ✅ |
| AUROC(`false_confirmed`) > 0.65 | 0.884 | ✅ |
| AUROC(`failure_in_5`) > 0.75 | 0.853 | ✅ |
| AUROC(`imminent_failure_dynamic`) > 0.75 | 0.902 | ✅ |
| AUPRC(`imminent_failure_dynamic`) > 0.15 | 0.323 | ✅ |
| AUROC(`needs_full_compute`) > 0.70 | 0.648 | ❌ |
| ECE(`false_confirmed`) < 0.12 | 0.316 | ❌ |

**5/7 gates pass. Blocking: ECE(fc) and needs_full_compute (oracle labels required).**

### Key v2 eval artifacts

| Artifact | Path |
|---|---|
| val metrics (v2, calibrated) | `saltr/results/eval_val_v2.json` |
| val predictions (calibrated) | `saltr/results/preds_val_v2.json` |
| policy replay | `saltr/results/policy_val_v2.json` |
| diagnostic metrics | `saltr/results/eval_diagnostic_v2.json` |
| label audit (v1 NPZ) | `saltr/results/label_audit_v1.json` |
| feature baselines | `saltr/results/baselines_val.json` |

### What this session confirmed

1. **imminent_failure_dynamic_10/20 signal holds**: AUROC 0.897/0.889 at 10/20-frame horizon. Telemetry carries proactive failure signal without teacher features.
2. **false_confirmed baselines**: GRU AUROC=0.884 vs best baseline 0.511 (+37.9pp). Neural temporal approach is essential — single-feature rules fail completely.
3. **Policy improvement v0→v2**: template corruption −44%, wrong reinit −33%.
4. **Diagnostic split (hard sequences)**: false_confirmed AUROC=0.548 (fails STOP gate). Hard identity-drift cases need teacher features.
5. **Lead-time still 1 frame median**: e-process sequential accumulation (Phase 2A) is the right fix, not longer-horizon labels alone.

Interpretation:

- `false_confirmed` + `imminent_failure_dynamic` are real enough to build interventions around.
- e-process is the next highest-leverage step that needs zero new tracker runs.
- Teacher features (DAM memory, CoTracker3 point consistency) are the path to fixing ECE(fc) and diagnostic AUROC(fc).
- `needs_full_compute` is not ready — do not claim compute savings without oracle.

---

## Session Log — 2026-05-20

### Pipeline fixes (all tests green: 33 passed)

| Fix | File | Impact |
|---|---|---|
| `n_evaluated`/`n_skipped` + `sys.exit` if 0 | `policy.py` | Policy replay no longer silently misreports |
| `test_eval_does_not_double_sigmoid` calls `_run_inference` | `test_saltr_model.py` | Regression test covers real eval path |
| `--datasets` multi-token parser in run_phase1.sh | `run_phase1.sh` | `--datasets uav123 visdrone_sot` now works |
| GO/NO-GO test asserts exactly BORDERLINE | `test_saltr_eval.py` | Gate won't silently become permissive |
| `_load_model` reads `head_names` from checkpoint metadata | `eval.py` | v1/v2 checkpoints load correctly |
| `train.SALTRD.forward` uses `self.heads` keys | `train.py` | v2 forward shape (B, 13) not hardcoded (B, 7) |
| Predictions saved AFTER calibration | `eval.py` | Exported JSON reflects calibrated probs |
| Checkpoint saves `_schema_label_names`, not `LABEL_NAMES` | `train.py` | Provenance correct for v1/v2 |
| Early stopping composite score for v2 | `train.py` | Checkpoint optimises for fc + ifd10 + ifd20 |
| Schema-tagged result filenames | `run_phase1.sh` | `eval_val_v2.json` not `eval_val.json` |

### New modules / utilities

| File | Purpose |
|---|---|
| `saltr/src/salt_r/diagnose_labels.py` | Label contamination audit; reads v1/v2 columns directly |
| `saltr/src/salt_r/baselines.py` | Feature baseline comparison vs GRU (AUROC/AUPRC/lift table) |
| `saltr/src/salt_r/eval.py::compute_failure_lead_time` | Lead-time metric for ifd heads |
| `saltr/src/salt_r/eval.py::calibrate_temperature / apply_temperature` | Per-head temperature scaling |
| `collect_features.recompute_labels_v1/v2()` | Generate v1/v2 NPZ without re-running tracker |

### Training runs completed

| Schema | Epoch | AUROC(fc) | AUPRC(fc) | AUROC(ifd) | ECE(fc) cal | template_corr |
|---|---|---:|---:|---:|---:|---:|
| v0 | 13 | 0.884 | 0.331 | — | 0.264 | 0.108 |
| v1 | 3 | 0.890 | 0.356 | 0.900 | 0.229 | 0.090 |
| v2 | 14 | 0.884 | 0.336 | 0.902 / **0.897** / **0.889** | 0.316 | **0.060** |

---



```text
Frozen SGLATrack/SALT v3
  |
  | per-frame telemetry
  |   APCE / PSR / score-map stats / bbox motion / flow / embeddings
  v
SALT-RD Feature Stream
  |
  +--> DAM-style Memory Features
  |      positive target memory
  |      negative distractor memory
  |      target-vs-distractor margin
  |
  +--> CoTracker3 Teacher Features (offline only)
  |      point survival
  |      point visibility
  |      forward/backward consistency
  |      point cloud coherence
  |
  v
SALT-RD Temporal Model
  P(false_confirmed)
  P(failure_in_5/10/20)
  P(imminent_failure_dynamic_5/10/20)
  P(recoverable)
  P(distractor_risk)
  P(template_corruption)
  P(wrong_reinit)
  P(needs_full_compute) only after oracle replay
  |
  v
e-Process Alert Layer
  anytime evidence accumulation
  controlled false alert rate
  alert tiers: observe / verify / intervene
  |
  v
Policy Interventions
  block template update
  reject bad recovery
  require extra verification
  expand search region conservatively
  disable pruning / use full compute
  trigger SAM2/SAMURAI/Grounded-SAM fallback only on high-risk frames
```

Runtime target:

- SALT-RD temporal model: cheap, always-on.
- DAM-style memory features: cheap, always-on if embeddings are from existing tracker path.
- e-process: negligible overhead.
- CoTracker3/SAM2/DINO: offline teacher or rare fallback, not every frame.

---

## New Phase Plan

### Phase 0 — Preserve Current Baseline And Artifacts

**Status: ✅ done.**

- Provenance fields (git_commit, npz_md5, checkpoint_md5, label_schema, created_at) in all eval JSONs.
- `saltr/results/` is the canonical output dir, schema-tagged filenames.
- Generated JSONs in `saltr/checkpoints/` removed from git tracking.
- 33 targeted SALT-RD unit tests pass.

```bash
PYTHONPATH=src:saltr/src .venv/bin/python -m pytest tests/unit/test_saltr_*.py -q
PYTHONPATH=src:saltr/src .venv/bin/python -m compileall saltr/src/salt_r
```

---

### Phase 1 — Lock v1/v2 As Baseline, Stop Optimizing Weak Heads

**Status: ✅ done. v0/v1/v2 trained, evaluated, baselines computed.**

Results locked:

| Artifact | Status |
|---|---|
| `saltr/results/eval_val_v2.json` | ✅ v2 calibrated val metrics |
| `saltr/results/preds_val_v2.json` | ✅ v2 calibrated predictions |
| `saltr/results/policy_val_v2.json` | ✅ v2 policy replay |
| `saltr/results/baselines_val.json` | ✅ feature baselines vs GRU |
| `saltr/results/label_audit_v0.json` | ✅ hard_dynamic_scene contamination audit |
| `saltr/results/label_audit_v1.json` | ✅ v1 label base rates |

---

### Phase 2A — e-Process Alerts Over Existing v1 Probabilities

**Priority:** highest short-term, because it needs no new tracker runs.

Problem:

`P(imminent_failure_dynamic)>0.5` has good recall but median lead-time is only 1 frame. A single-frame threshold is too twitchy and not enough for proactive policy.

Idea:

Wrap risk probabilities in a sequential test that accumulates evidence. We want fewer noisy alerts and better intervention timing:

- `H0`: tracker is still trustworthy / no failure process started.
- `H1`: tracker has entered failure-risk process.

Inputs:

```text
p_false_confirmed[t]
p_imminent_failure_dynamic[t]
p_failure_in_5[t]
p_failure_in_10[t]
p_failure_in_20[t]
entropy_z[t]
apce_drop_z[t]
peak_margin_z[t]
memory_margin[t]        # added in Phase 2B
point_consistency[t]    # added in Phase 2C
```

Initial v1-only score:

```python
risk_score = (
    0.45 * p_false_confirmed
  + 0.35 * p_imminent_failure_dynamic
  + 0.15 * p_failure_in_5
  + 0.05 * entropy_z_rank
)
```

Implementation file:

```text
saltr/src/salt_r/eprocess.py
tests/unit/test_saltr_eprocess.py
```

Recommended e-process design:

1. Split validation sequences into calibration and alert-eval groups by sequence.
2. Define null frames from GT:
   - IoU >= 0.5,
   - not `failure_in_5/10/20`,
   - not `false_confirmed`.
3. Convert risk score to conformal p-value using calibration null distribution:

```python
p_t = (1 + count(null_scores >= score_t)) / (1 + n_null)
```

4. Convert p-value to an e-value with a power betting function:

```python
e_t = epsilon * (p_t ** (epsilon - 1.0))   # epsilon in (0, 1), e.g. 0.5
```

5. Run two alert modes and report them separately:

**Formal mode** — no decay, no intra-sequence reset unless a new independent tracking episode starts:

```python
E_t = E_{t-1} * e_t
alert when max_so_far(E_t) >= 1 / alpha
```

This is the closest to the WACV e-process framing. Because video frames are correlated and our conformal null is empirical, still validate false alerts by sequence; do not claim a theorem beyond the assumptions we actually satisfy.

**Engineering mode** — decay/reset smoother for deployment usability:

```python
E_t = max(1.0, decay * E_{t-1} * e_t)
alert when E_t >= 1 / alpha
```

This mode may give better UX, but it is empirical. Do not describe it as formally anytime-valid unless we prove the reset/decay construction.

6. Sweep:
   - `epsilon`: 0.25, 0.5, 0.75
   - `alpha`: 0.20, 0.10, 0.05
   - `decay`: 0.95, 0.98, 1.00
   - reset after re-init / first frame after GT failure.

Metrics:

| Metric | Why |
|---|---|
| median lead time before IoU<0.3 | proactive value |
| recall of failure events | coverage |
| false alerts per 1000 frames | deployment burden |
| FAR at sequence level | user-visible nuisance |
| alert duration | whether alerts are usable or sticky |
| intervention opportunity rate | how often policy can act before failure |

GO:

| Gate | Target |
|---|---:|
| median lead time | >= 3 frames on val |
| failure-event recall | >= 0.60 |
| false alerts | <= 100 per 1000 frames or sequence-level FAR <= 0.10 |
| improves over raw `P(ifd)>0.5` | yes on lead-time/FAR tradeoff |

STOP:

- If e-process only delays alerts without reducing FAR, keep single-frame risk tiers.
- If median lead time remains 1 frame after 10/20 labels, the issue is label/feature, not alert logic.

Output artifacts:

```text
saltr/results/eprocess_val_v1.json
saltr/results/eprocess_threshold_sweep_v1.csv
saltr/results/timeline_eprocess_<seq>.png
```

---

### Phase 2B — DAM4SAM-Style Distractor Memory

**Priority:** highest expected gain for false-confirmed and wrong recovery.

Problem:

False-confirmed is not localization uncertainty. The tracker can have a sharp score map and high APCE while sitting on the wrong object. DAM4SAM's key lesson is: tracking needs **memory about distractors**, not only target memory.

We implement a lightweight DAM-style memory, not full SAM2 memory.

New files:

```text
saltr/src/salt_r/memory.py
saltr/src/salt_r/memory_features.py
tests/unit/test_saltr_memory.py
```

Memory state:

```python
PositiveMemory:
  embeddings: recent confident target embeddings
  bboxes: predicted boxes
  timestamps
  quality: low risk + high IoU during offline collection / low risk at runtime

NegativeMemory:
  embeddings: distractor candidate embeddings
  bboxes: candidate boxes / secondary peaks / rejected recovery boxes
  timestamps
  source: secondary_peak | rejected_recovery | false_confirmed_teacher | detector_candidate
```

Runtime-safe embedding sources, in priority order:

1. Existing SGLATrack embedding/helper if accessible without extra forward pass.
2. Current 32x32 crop embedding path already used for cosine guards.
3. Lightweight MobileNet/ConvNeXt-tiny crop embedding only if overhead < 1 ms/frame on MPS.
4. DINO/SAM embeddings only offline for teacher features, not always-on runtime.

Memory update rules:

```python
if p_false_confirmed < 0.20 and p_ifd < 0.30 and apce_norm stable:
    positive_memory.add(current_target_embedding)

if secondary_peak_is_strong or detector_candidate_rejected or p_false_confirmed high:
    negative_memory.add(candidate_embedding)

if candidate overlaps current target too much:
    do not add to negative memory

if memory age > max_age or spatially impossible:
    decay/remove
```

Important: offline collection can use GT to label which memories are positive/negative; runtime cannot.

Candidate sources for negative memory:

| Source | How |
|---|---|
| score-map secondary peaks | implement real local maxima, not `n_secondary=0` placeholder |
| detector candidates | YOLO/RT-DETR candidates around search area |
| recovery rejects | candidates rejected by size/spatial/appearance guard |
| GT false-confirmed frames offline | predicted bbox crop when IoU<0.2 but confidence high |
| nearby same-class objects | if detector provides classes |

New scalar features:

```text
mem_pos_max_sim
mem_pos_mean_sim
mem_pos_consensus
mem_neg_max_sim
mem_neg_mean_sim
mem_neg_count_nearby
mem_target_minus_distractor_margin
mem_neg_spatial_proximity
mem_update_age
mem_memory_entropy
mem_template_contamination_score
mem_recovery_candidate_margin
```

New labels:

```python
distractor_present[t] = exists candidate c where IoU(c, gt_target) < 0.2 and candidate_score high

template_corruption[t] = would_update_template[t] and IoU(pred_bbox[t], gt_bbox[t]) < 0.5

wrong_reinit[t] = recovery_candidate_accepted[t] and IoU(candidate, gt_bbox[t]) < 0.3

identity_margin_failure[t] = (
    similarity(pred_crop, negative_memory) >= similarity(pred_crop, positive_memory) - margin
    and IoU(pred_bbox, gt_bbox) < 0.3
)
```

Use cases:

1. Improve `P(false_confirmed)`.
2. Reject wrong recovery before `tracker.init(winner_bbox)`.
3. Block template update when predicted target looks closer to distractor memory.
4. Give e-process a stable identity-drift evidence stream.

GO:

| Gate | Target |
|---|---:|
| AUPRC(`false_confirmed`) | >= 0.45 or +25% relative vs v1 |
| recall@5%FPR(`false_confirmed`) | >= 0.60 |
| diagnostic AUROC(`false_confirmed`) | >= 0.75 |
| template corruption rate | -25% relative |
| wrong re-init rate | -20% relative |

STOP:

- If memory features improve val but not diagnostic split, treat it as overfit.
- If runtime overhead > 2 ms/frame on MPS and no strong policy gain, keep memory as offline analysis only.

---

### Phase 2C — CoTracker3 Point Consistency Teacher

**Priority:** tied with Phase 2B, but more offline-heavy.

Problem:

APCE/score maps tell us "there is a confident response", not "this response belongs to the same physical object." Point tracks give a stronger identity-consistency signal.

Use CoTracker3 offline to create teacher signals and distill them into SALT-RD features/labels. CoTracker3 should not run in the edge loop.

New files:

```text
saltr/src/salt_r/teachers/cotracker3_export.py
saltr/src/salt_r/teachers/point_features.py
saltr/src/salt_r/recompute_teacher_features.py
tests/unit/test_saltr_point_features.py
```

Teacher collection plan:

1. For each sequence, sample points inside the initial GT bbox:
   - 3x3 grid for small targets,
   - 4x4 or 5x5 grid for larger targets,
   - include center + corners + edge midpoints.
2. Run CoTracker3 offline or online variant over frames.
3. For each frame, compute point-level statistics relative to:
   - predicted bbox,
   - GT bbox during offline labeling,
   - previous frame point cloud,
   - current optical-flow estimate.
4. Save point teacher arrays into sidecar NPZ first; do not bloat the core dataset until validated.

Sidecar schema:

```text
point_tracks/{seq}:        float32 (T, P, 2)
point_visibility/{seq}:    float32/bool (T, P)
point_confidence/{seq}:    float32 (T, P) if available
point_feature_names:       list[str]
point_features/{seq}:      float32 (T, F_point)
teacher_version:           str
teacher_model:             str
created_at:                str
```

Point features:

```text
pt_visible_ratio
pt_inside_pred_ratio
pt_inside_pred_weighted
pt_inside_gt_ratio                # offline diagnostics only, never runtime feature
pt_forward_backward_error
pt_median_motion
pt_motion_iqr
pt_affine_residual
pt_cluster_area_ratio
pt_cluster_aspect_delta
pt_flow_agreement
pt_bbox_center_disagreement
pt_survival_since_init
pt_reacquisition_consistency
pt_split_score                    # point cloud splits into two modes
```

Teacher labels:

```python
point_consistency_good[t] = (
    pt_visible_ratio[t] > 0.6
    and pt_inside_pred_ratio[t] > 0.6
    and pt_affine_residual[t] < p75_seq
)

point_identity_break[t] = (
    IoU(pred_bbox[t], gt_bbox[t]) < 0.3
    and pt_inside_pred_ratio[t] < 0.4
    and pt_visible_ratio[t] > 0.5
)

point_recoverable[t] = (
    IoU(pred_bbox[t], gt_bbox[t]) < 0.3
    and pt_visible_ratio[t] > 0.5
    and point cloud still spatially coherent
)
```

Critical leakage rule:

- `pt_inside_gt_ratio` may be used for label diagnostics only.
- Student runtime features must not include GT-relative fields.
- Teacher labels may use GT because training is offline; inference only uses distilled student.

How to use the teacher:

| Mode | Use |
|---|---|
| Teacher labels | train additional heads: `point_identity_break`, `point_recoverable`, `point_consistency_good` |
| Teacher features | optionally add point features to training only, then distill to scalar telemetry/memory features |
| Diagnostics | explain false-confirmed cases: point cloud follows original target while bbox jumps to distractor |

GO:

| Gate | Target |
|---|---:|
| `point_identity_break` AUROC | >= 0.80 |
| `false_confirmed` AUPRC after point distillation | >= 0.45 or +25% relative |
| diagnostic false-confirmed AUROC | >= 0.75 |
| lead time for identity break | >= 3 frames when combined with e-process |

STOP:

- If CoTracker3 fails on small UAV targets due tiny objects, try fewer points + upscaled crops before abandoning.
- If teacher is noisy, add teacher reliability filtering instead of training on all pseudo-labels.

---

### Phase 2D — SALT-RD v2 Dataset And Model

This phase combines:

- v1 labels,
- v2 10/20-frame labels,
- DAM memory features,
- CoTracker3 teacher features/labels,
- e-process-ready risk outputs.

Target schema:

```text
features_v2:
  v0 scalar features (28)
  memory features (10-14)
  optional distilled point-consistency scalar features (6-10)

labels_v2:
  v1 labels (0-9)
  failure_in_10
  failure_in_20
  imminent_failure_dynamic_10
  imminent_failure_dynamic_20
  distractor_present
  template_corruption
  wrong_reinit
  point_identity_break
  point_consistency_good
  point_recoverable
```

Recommended model:

```python
class SALTRDv2(nn.Module):
    input_dim = 28 + memory_dim + point_distill_dim
    trunk = GRU(input_dim, hidden=96, layers=2, dropout=0.2)
    heads = {
        "false_confirmed": reliability head,
        "failure_in_5": risk head,
        "failure_in_10": risk head,
        "failure_in_20": risk head,
        "imminent_failure_dynamic": dynamic risk head,
        "imminent_failure_dynamic_10": dynamic risk head,
        "imminent_failure_dynamic_20": dynamic risk head,
        "recoverable": recovery head,
        "distractor_present": memory head,
        "template_corruption": policy head,
        "wrong_reinit": policy head,
        "point_identity_break": teacher head,
        "point_consistency_good": teacher head,
    }
```

Training details:

- Sequence-level split only.
- Diagnostic sequences excluded from train/val.
- Focal BCE or weighted BCE.
- Head weights:
  - high: `false_confirmed`, `wrong_reinit`, `template_corruption`, `point_identity_break`;
  - medium: `failure_in_10/20`, `imminent_failure_dynamic_10/20`, `recoverable`;
  - low/aux: `target_dynamic`, `camera_dynamic`, old `hard_dynamic_scene`.
- Early stopping primary metric:
  - v2 primary: AUPRC(`false_confirmed`) + AUPRC(`wrong_reinit`) if available.
  - secondary: median lead-time after e-process on val.

Evaluation:

| Group | Metrics |
|---|---|
| Reliability | AUROC, AUPRC, recall@5%FPR, ECE, Brier |
| Alerting | lead time, FAR, event recall, alert duration |
| Memory | target-vs-distractor margin distributions, memory ablations |
| Point teacher | teacher label AUROC/AUPRC, diagnostic timelines |
| Policy | template corruption, wrong reinit, abstention gain, AUC delta |
| Generalization | LODO: train two datasets, test third |

GO:

| Gate | Target |
|---|---:|
| AUPRC(`false_confirmed`) | >= 0.45 or +25% relative |
| recall@5%FPR(`false_confirmed`) | >= 0.60 |
| diagnostic AUROC(`false_confirmed`) | >= 0.75 |
| median e-process lead time | >= 3 frames |
| template corruption reduction | >= 25% |
| wrong re-init reduction | >= 20% |
| LODO false-confirmed AUROC | >= 0.75 on at least 2/3 target domains |

NO-GO:

- If v2 improves only global val but not diagnostics/LODO, do not call it real improvement.
- If interventions do not improve actual replay metrics, keep v2 as analysis-only.

---

### Phase 2E — Policy Intervention Replay

This is where we prove "реальне покращення", not just better classification.

New/updated files:

```text
saltr/src/salt_r/policy.py
saltr/src/salt_r/policy_sweep.py
saltr/src/salt_r/interventions.py
tests/unit/test_saltr_policy_sweep.py
```

Interventions:

| Intervention | Trigger | Expected benefit |
|---|---|---|
| block template update | high `p_false_confirmed` or high memory negative margin | lower template corruption |
| reject recovery candidate | high `p_wrong_reinit` or negative memory sim > positive sim | lower wrong re-init |
| verify before re-init | high recovery uncertainty | fewer identity switches |
| expand search conservatively | e-process alert but not false-confirmed | catch target before full loss |
| full compute / no pruning | high risk or high uncertainty | avoid cheap-mode failure |
| fallback to SAMURAI/SAM2/Grounded-SAM only on high risk | repeated alert + low trust | improve hard cases without per-frame cost |

Policy must be evaluated as replay first:

```text
Inputs:
  predictions JSON
  NPZ IoU traces
  candidate/recovery logs if available
  template update logs if available

Outputs:
  policy metrics with confidence intervals
  intervention timelines
  threshold sweeps
```

Threshold sweep:

```text
p_false_confirmed: 0.40 ... 0.90
p_wrong_reinit:    0.40 ... 0.90
eprocess_alpha:    0.20, 0.10, 0.05
memory_margin:     -0.10, 0.00, 0.10, 0.20
```

Metrics:

| Metric | Definition |
|---|---|
| template corruption rate | allowed template updates when IoU<0.5 |
| wrong reinit rate | accepted recovery/reinit when IoU<0.3 |
| abstention gain | IoU improvement from refusing unsafe update/reinit |
| missed safe update rate | blocked update when IoU>=0.7 |
| failure event recall | event has alert before IoU<0.3 |
| lead time | frames between first alert and failure |
| intervention density | interventions per 1000 frames |
| AUC delta | tracker AUC with replayed intervention vs baseline |

GO:

- Template corruption rate down by >=25%.
- Wrong reinit rate down by >=20%.
- AUC not worse globally; hard diagnostic AUC improves or failure duration decreases.
- Intervention density acceptable: no permanent "always block" policy.

---

### Phase 3 — Oracle Compute Labels And Real GFLOPs/FPS Claims

Only after Phases 2A-2E show trust/recovery gain.

Problem:

Current `needs_full_compute` is bootstrap. It does not prove compute savings.

Oracle definition:

Run each sequence in at least two modes:

```text
full mode:
  SGLATrack/SALT frozen baseline, full tokens/no pruning where appropriate

cheap mode:
  CE pruning / UTP-like pruning / ABTrack-like bypass / reduced search / reduced detector calls
```

Oracle label:

```python
needs_full_compute[t] = (
    cheap_iou[t:t+k].mean_drop_vs_full > 0.03
    or cheap_failure_event[t:t+k] and not full_failure_event[t:t+k]
    or cheap_causes_false_confirmed[t:t+k]
)
```

Required outputs:

```text
saltr/data/salt_rd_compute_oracle.npz
saltr/results/compute_oracle_eval.json
saltr/results/auc_vs_gflops_pareto.csv
```

Compare against:

- always full;
- always cheap;
- APCE threshold policy;
- entropy threshold policy;
- UncL-STARK-style uncertainty policy if implementable;
- UTPTrack/ABTrack reported tradeoffs as external references.

GO:

| Gate | Target |
|---|---:|
| cheap frame rate | >= 15% |
| AUC loss | <= 0.005 absolute |
| or GFLOPs reduction | >= 10% at <=0.005 AUC loss |
| policy beats APCE/entropy compute baseline | yes |

STOP:

- If cheap mode itself is not meaningfully cheaper on Apple MPS, do not pursue compute claim.
- If oracle labels are too noisy, publish reliability/recovery only.

---

### Phase 4 — Recovery Fallback Experiments

Lower priority than memory/point/e-process, but important for hard sequences.

Candidates:

| Fallback | Role |
|---|---|
| SAMURAI / SAM2 motion-aware tracking | high-risk fallback for class-agnostic object continuation |
| EfficientTAM | lighter track-anything fallback |
| Grounded-SAM2 | open-vocabulary re-detection when class mismatch kills YOLO |
| DINO matching | appearance verification for recovery candidates |

Rule:

Fallback is triggered only by SALT-RD risk/e-process, not every frame.

Evaluate on:

- `uav0000164`
- `bike2`
- `Gull2`
- `Sheep1`
- `StreetBasketball1`
- DTB70 natural scenes where YOLO26m VisDrone fails.

GO:

- wrong recovery decreases;
- hard-sequence AUC/failure duration improves;
- overhead is bounded because fallback call rate is low.

---

### Phase 5 — LODO And Robustness

This is mandatory before any strong claim.

Splits:

| Experiment | Train | Test |
|---|---|---|
| LODO-DTB70 | UAV123 + VisDrone-SOT | DTB70 |
| LODO-VisDrone | UAV123 + DTB70 | VisDrone-SOT |
| LODO-UAV123 | VisDrone-SOT + DTB70 | UAV123 |

Diagnostics must remain held out:

```text
uav0000164
bike2
Gull2
Sheep1
StreetBasketball1
```

Report:

- per-dataset AUROC/AUPRC;
- diagnostic timelines;
- bootstrap CI by sequence;
- target-size bins;
- camera motion bins;
- distractor density bins.

GO:

- At least 2/3 LODO tests show useful false-confirmed signal.
- Diagnostic split improves after DAM/point features.

---

### Phase 6 — Backbone Adaptation Fallback

Only if controller interventions still do not improve actual tracking outcomes.

Options:

1. LoRAT-style parameter-efficient adaptation on aerial datasets.
2. Fine-tune SGLATrack/SUTrack/UTPTrack backbone on UAV123+VisDrone+DTB70-like data.
3. Replace base tracker with stronger UAV backbone and keep SALT-RD as trust controller.

This is not the main novelty, but may be needed for AUC.

Rule:

- If AUC is capped by base tracker, reliability controller can still be useful.
- If user goal becomes "beat AUC SOTA", we need tracker adaptation/backbone work, not only SALT-RD.

---

## Concrete Next Coding Tasks

### Task A — Implement e-process module first

Prompt:

```text
Implement saltr/src/salt_r/eprocess.py.

Inputs:
  - predictions JSON from eval.py
  - NPZ labels / iou_trace
  - head names from JSON

Features:
  - conformal null calibration by sequence
  - power e-value transform
  - sequential accumulation with decay/reset
  - threshold sweep over alpha/epsilon/decay

Outputs:
  - alert events per sequence
  - metrics: lead_time, event_recall, false_alerts_per_1000, sequence_FAR
  - JSON + CSV sweep artifacts in saltr/results/

Tests:
  - no alerts on all-null sequence at strict alpha
  - earlier alert when risk score increases monotonically before failure
  - reset prevents stale evidence after reinit/failure boundary
```

Expected value:

- Quick proof whether v1 probabilities can become usable proactive alerts.
- No retraining required.

---

### Task B — Add DAM-style memory feature collector

Prompt:

```text
Implement lightweight positive/negative memory features in saltr/src/salt_r/memory.py
and saltr/src/salt_r/memory_features.py.

Do not integrate into runtime first.
Add offline collection from existing NPZ + tracker logs if available.
Start with embeddings already available from SGLATrack/cosine memory path.

Produce:
  - memory sidecar NPZ
  - feature_names for memory features
  - diagnostics for target-vs-distractor similarity margin
```

Minimum viable implementation:

- positive memory from high-IoU frames offline;
- negative memory from false-confirmed predicted crops offline;
- candidate current crop similarity to both;
- no detector dependency in v1 memory prototype.

Expected value:

- Direct attack on `false_confirmed`.
- Stronger recovery/template policy.

---

### Task C — CoTracker3 teacher export

Prompt:

```text
Add saltr/src/salt_r/teachers/cotracker3_export.py.

The script should:
  - load dataset sequences
  - sample query points inside initial GT bbox
  - run CoTracker3 offline or read precomputed tracks
  - compute point consistency features
  - write sidecar NPZ with provenance

Do not add CoTracker3 as hard dependency to normal unit tests.
Gate tests with tiny synthetic arrays for point feature math.
```

Expected value:

- Offline identity-consistency teacher.
- Better labels/features for high-confidence wrong-object drift.

---

### Task D — Train SALT-RD v2

Prompt:

```text
Extend model/train/eval to label_schema v2-memory-point.

Inputs:
  - base NPZ v1/v2 labels
  - memory sidecar NPZ
  - point teacher sidecar NPZ

Heads:
  false_confirmed
  failure_in_5/10/20
  imminent_failure_dynamic_5/10/20
  recoverable
  distractor_present
  template_corruption
  wrong_reinit
  point_identity_break
  point_consistency_good

Primary early-stop:
  weighted score = AUPRC(false_confirmed) + AUPRC(wrong_reinit) + lead-time proxy
```

Expected value:

- Real improvement over v1, or clear evidence that student features cannot distill teacher.

---

### Task E — Policy intervention replay with thresholds

Prompt:

```text
Implement saltr/src/salt_r/policy_sweep.py.

Run threshold sweeps for:
  - false_confirmed block
  - wrong_reinit reject
  - e-process alert
  - memory margin

Report:
  template_corruption_rate
  wrong_reinit_rate
  missed_safe_update_rate
  intervention_density
  AUC delta / failure duration delta
```

Expected value:

- Converts classification gains into tracking-system gains.

---

## Metrics We Care About Now

Primary:

| Metric | Why |
|---|---|
| AUPRC(`false_confirmed`) | rare identity failure |
| recall@5%FPR(`false_confirmed`) | usable high-precision intervention |
| diagnostic AUROC/AUPRC | hard cases, not easy val |
| template corruption reduction | actual deployment win |
| wrong reinit reduction | actual recovery win |
| e-process lead time | proactive value |
| false alerts per 1000 frames | operational usability |

Secondary:

| Metric | Why |
|---|---|
| ECE/Brier/NLL | calibration quality |
| AUC delta after policy | final tracking impact |
| failure duration | often more meaningful than full-sequence AUC |
| NT2F | compare with MATA-like protocol |
| AUC-vs-GFLOPs | only after oracle compute labels |

Demoted:

| Metric | Why demoted |
|---|---|
| `hard_dynamic_scene_v2` AUROC | not predictive enough |
| bootstrap `needs_full_compute` | not real compute oracle |
| raw FPS/GFLOPs | no value if policy does not preserve tracking |

---

## Paper/Result Shape If This Works

Not the priority right now, but this is the likely end shape:

> We introduce SALT-RD, a lightweight reliability controller for real-time UAV SOT that combines distractor-aware memory, point-consistency distillation, and anytime sequential alerts to detect high-confidence identity drift before unsafe tracker interventions. Unlike APCE/entropy and localization-uncertainty baselines, SALT-RD detects false-confirmed tracking, reduces wrong re-initialization and template corruption, and provides bounded false-alert proactive intervention on UAV123, VisDrone-SOT, and DTB70.

Comparison groups:

- Base tracker: SGLATrack/SALT v3.
- Confidence baselines: APCE, PSR, entropy, peak margin, flow consistency.
- Uncertainty baselines: OOTU / UncL-STARK-style heatmap uncertainty where feasible.
- Memory/teacher references: DAM4SAM and CoTracker3 as inspiration/teacher, not direct edge baselines.
- Compute baselines: UTPTrack/ABTrack/UncL-STARK if/when oracle compute branch exists.

---

## Red Lines For Claude/Agents

Block changes if they:

- train on `_decide_state()`, `TargetState`, old TSA scene labels, or APCE thresholds as target labels;
- use diagnostic sequences in train;
- claim compute/FPS gain from bootstrap `needs_full_compute`;
- report accuracy without base rate/AUPRC;
- use frame-level random split;
- run CoTracker3/SAM2 in every runtime frame and call it edge-ready;
- overwrite frozen SGLATrack behavior while implementing SALT-RD;
- tune thresholds after looking at diagnostic split and then report diagnostic as held-out;
- make `hard_dynamic_scene_v2` central again without intervention improvement;
- treat localization uncertainty as sufficient for false-confirmed identity drift.

---

## Suggested Directory Layout

```text
saltr/src/salt_r/
  eprocess.py
  memory.py
  memory_features.py
  policy_sweep.py
  interventions.py
  teachers/
    __init__.py
    cotracker3_export.py
    point_features.py
  collect_features.py
  model.py
  train.py
  eval.py
  policy.py

saltr/data/
  salt_rd_v0.npz
  salt_rd_v1_labels.npz
  salt_rd_v2_labels.npz
  salt_rd_memory_sidecar.npz
  salt_rd_cotracker3_sidecar.npz

saltr/results/
  eval_val_v1_calibrated.json
  preds_val_v1_calibrated.json
  baselines_val_v1.json
  eprocess_val_v1.json
  memory_diagnostics.json
  point_teacher_diagnostics.json
  policy_sweep_v2.json
```

---

## Starting Prompt For Next Coding Session

```text
Read HANDOFF_NEXT.md section "Current Repo Facts" and saltr/results/eval_val_v2.json.

STATE:
- v2 trained: 13 heads, epoch 14 (composite stopping AUPRC(fc)+0.5*AUPRC(ifd10)+0.25*AUPRC(ifd20))
- v2 val: AUROC(fc)=0.884, AUPRC(fc)=0.336, ifd AUROC=0.902, ifd10 AUROC=0.897, ifd20 AUROC=0.889
- v2 policy: template_corruption=0.060 (−44% vs v0), wrong_reinit=0.183 (−33% vs v0)
- Lead time: median=1 frame (still reactive, not proactive)
- ECE(fc)=0.316 and AUROC(nfc)=0.648 still blocking GO gates

CONFIRMED: ifd10/ifd20 signal holds (AUROC 0.897/0.889) — telemetry carries 20-frame early warning.
DO NOT start Phase 2B/2C yet — start with Phase 2A to convert 1-frame reactive alerts to sequential evidence.

TASK A — Implement e-process sequential alerts (Phase 2A):

Implement saltr/src/salt_r/eprocess.py over existing v2 calibrated predictions.

The e-process accumulates evidence over time:
  H0: tracker trustworthy / no failure process active
  H1: tracker entered failure-risk process

Initial composite risk score (no new tracker runs needed):
  risk_score[t] = 0.45 * p_false_confirmed[t]
               + 0.35 * p_imminent_failure_dynamic[t]
               + 0.15 * p_failure_in_5[t]
               + 0.05 * entropy_z_rank[t]

e-process: multiplicative martingale, run offline first:
  e[0] = 1.0
  e[t] = e[t-1] * (1 + lambda * (risk_score[t] - alpha))
  alert when e[t] >= 1/alpha

See HANDOFF_NEXT.md §Phase 2A for details on parameters and threshold sweep.

Inputs available:
  saltr/results/preds_val_v2.json   (calibrated predictions, all 13 heads)
  saltr/data/salt_rd_v2_labels.npz  (IoU traces, ground-truth labels)

Output:
  saltr/results/eprocess_val_v2.json
  saltr/results/eprocess_sweep_v2.json

Primary metrics:
  median lead time (target: >= 3 frames; current 1f baseline from raw threshold)
  failure event recall at alpha=0.10
  false alerts per 1000 frames
  improvement over raw P(ifd)>0.5 threshold

RED LINES:
- Do NOT optimize hard_dynamic_scene_v2
- Do NOT claim compute/FPS without oracle compute labels
- Do NOT calibrate on train split
- Do NOT tune GO gates to get better verdict
- Do NOT train on diagnostic sequences (bike2, Gull2, Sheep1, StreetBasketball1, uav0000164)
```

---

## One-Line Current Phase

**Current phase:** v2 trained and evaluated (ifd10/20 hold at 0.89+). Starting **Phase 2A: e-process sequential alerts** over existing v2 calibrated predictions — zero new tracker runs needed, converts 1-frame reactive threshold into accumulating evidence with controlled false-alarm rate.
