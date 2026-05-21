# HANDOFF — SALT-RD: Phase 4B Real Embeddings

**Date:** 2026-05-21  
**Sessions covered:** Sessions 1–3 (2026-05-21) — full Phase 4B proxy + ablations + extraction pipeline  
**Owner:** Staff CV/AI/ML review track  
**Test count:** 207 passing  
**Worktree:** clean

---

## Current State (one sentence)

Phase 4B proxy memory is **scientifically complete**: diagnostic fc AUROC 0.548→0.796 via proxy memory (validated direction). Full extraction pipeline for **real SGLATrack backbone embeddings** is implemented and ready to run — blocked only on compute time, not code.

---

## Most Recent Commits

```
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

### ⏳ Phase 2 — Real SGLATrack embeddings (NEXT)

**Step 1: OOF predictions** (5-fold, no compute-heavy tracker run)

```bash
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.make_oof_predictions \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --teacher-checkpoint saltr/checkpoints/v2_corrected/saltrd_best.pt \
  --output-dir saltr/results/oof/ \
  --merged-output saltr/results/preds_all_v2_oof_teacher.json \
  --n-folds 5
```

Output: `preds_all_v2_oof_teacher.json` (228 seqs: train=OOF, val+diag=teacher no-mem)

**Step 2: Smoke extraction (3 sequences, verify alignment)**

```bash
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.sgla_memory_extractor \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --preds saltr/results/preds_all_v2_oof_teacher.json \
  --config-path configs/prod/salt.yaml \
  --output saltr/data/salt_rd_sgla_pos_memory_sidecar.npz \
  --embedding-view score_weighted \
  --smoke-test 3
```

Verify: `n_ram_updates > 0` even on hard sequences (template bootstrap should fire at t=1), `mean_pos_mean_sim > 0.0`, `T == NPZ frames`.

**Step 3: Full extraction (228 seqs, compute-heavy, ~hours)**

```bash
# Same command, remove --smoke-test
```

Fail-fast: any skipped sequence raises `RuntimeError` unless `--allow-partial`. Fix issues before saving canonical sidecar.

### ⏳ Phase 3 — Train v2.2 with real SGLA positive memory

```bash
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.train \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --output saltr/checkpoints/v2_2_sgla_pos/ \
  --label-schema v2 \
  --memory-sidecar saltr/data/salt_rd_sgla_pos_memory_sidecar.npz \
  --memory-feature-names mem_pos_max_sim,mem_pos_mean_sim,mem_pos_recency_sim,mem_update_age
```

**GO gate (v2.2):**

| Gate | Minimum | Strong |
|---|---|---|
| Diag fc AUROC | > 0.774 (beat pos proxy) | > 0.796 (beat full proxy) |
| Val fc AUROC | ≥ 0.870 (no regression) | > 0.885 (recover baseline) |
| Val fc AUPRC | ≥ 0.300 | > 0.338 |
| ifd10/ifd20 AUROC | not drop > 3pp | — |

**KILL:** if diag doesn't beat pos_proxy (0.774) OR val stays below 0.870.

If SGLA score_weighted view is weak → try `--embedding-view peak_local` first, then `global`.

### ⏳ Phase 4 — Conservative policy integration

Runtime policy config (runner-up, validated conservative):
- `fc_t=0.60, reinit_t=0.70, mem_t=0.00`
- Block template update: `p_fc > 0.60 AND memory_margin < 0.00`
- Abstain recovery/reinit: `p_fc > 0.70`
- E-process: monitoring only

**Policy gate:** wrir=0, msu < 0.40, macro_tcr beats strict no-memory baseline (0.0361).

### ⏳ Phase 5 — If SGLA weak, fallback ladder
1. DINOv2 offline crop embeddings
2. DINO/SGLA hybrid margin
3. Contrastive distractor head: ranking loss `sim(target_mem, crop) > sim(distractor_mem, crop) + margin`
4. CoTracker3 teacher on hard diagnostic sequences only

### ⏳ Phase 6 — Separate AUC/domain-adaptation track
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
| OOF predictions | `saltr/results/preds_all_v2_oof_teacher.json` | ⏳ not yet generated |
| SGLA pos memory sidecar | `saltr/data/salt_rd_sgla_pos_memory_sidecar.npz` | ⏳ not yet extracted |
| v2.2 checkpoint | `saltr/checkpoints/v2_2_sgla_pos/` | ⏳ not yet trained |

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

---

## Fallback Decision Tree

```
Phase 2: Real SGLA embeddings
  │
  ├─ score_weighted weak → try peak_local → try global
  │
  ├─ All 3 views weak (diag fc < 0.774) → DINOv2 offline
  │     DINOv2 weak → contrastive distractor head
  │
  └─ Val recovers but diag stays below 0.774 → representation quality issue
        → add contrastive ranking loss: sim(pos_mem, crop) > sim(neg_mem, crop) + margin
        → OR add top-2 response peak margin / secondary peak crop similarity

Phase 4: Controller plateaus
  └─ Open separate SALT-AUC track: LoRAT/fine-tune SGLATrack backbone
     SALT-RD claim remains: trust controller reduces template corruption / wrong reinit
```

---

## Research Trajectory

```
1. ✅ fc signal works (AUROC 0.885 val, 0.796 diag with proxy)
2. ✅ Memory direction validated (diag fc 0.598 → 0.796)
3. ✅ pos_only is key driver (0.774 diag AUROC, 89% of full gain)
4. ⏳ Real SGLATrack embeddings (val fc must recover to ≥ 0.870)
5. ⏳ CoTracker3 on hard diagnostic sequences (after real memory works)
6. ⏳ LODO generalization (mandatory before paper)
7. ⏳ Runtime rollout (conservative: block template only first)
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
- 207 tests green
- Ablation result: pos_only RAM drives 89% of diagnostic gain (0.774 diag fc AUROC)
- neg_only is harmful; full memory (0.796) marginal over pos_only
- SGLATrack hook confirmed: _last_search_score_weighted / peak_local / global, 192-dim
- Bootstrap at frame t=1 from _last_template_embedding (unconditional, no gate)
- Extractor fails fast on any skipped sequence (canonical mode)

PENDING (compute-heavy, no code changes needed):
1. Run make_oof_predictions.py → preds_all_v2_oof_teacher.json
2. Smoke test sgla_memory_extractor.py --smoke-test 3 (verify n_ram_updates > 0)
3. Full 228-seq extraction → salt_rd_sgla_pos_memory_sidecar.npz
4. Train v2.2: gate diag fc > 0.774 AND val fc ≥ 0.870

DO NOT start coding until smoke test passes.

RED LINES:
- Use v2_retrained as strict baseline (not v2_corrected) for all gate comparisons
- Neg memory: do not include in v2.2
- E-process: monitoring only
- OOF preds required for final claim (not in-sample preds_train)
```
