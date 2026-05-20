# Night Summary — 2026-05-21

**Session:** Phase 4B gate-package — post-training eval, policy sweep, ablation roadmap  
**Model state:** v2.1 proxy-memory (37-dim, 9 memory dims) vs v2_corrected/v2_retrained baselines (28-dim, no memory)

**Baseline note:** the headline diagnostic gain `0.548 → 0.796` compares against `eval_diagnostic_v2_corrected.json`.
The stricter retrained no-memory baseline in `eval_diagnostic_v2_retrained.json` is `0.598 → 0.796` (+19.8pp).
Both comparisons pass the absolute `fc AUROC > 0.70` GO gate, but use the corrected label when quoting the +24.8pp number.

---

## 1. Eval Model — Memory vs No-Memory

### Val Split (49 sequences)

| Head | v2_retrained (baseline) | v2.1 memory | Δ | Status |
|---|---:|---:|---:|---|
| **false_confirmed** AUROC | 0.885 | 0.857 | −2.9pp | ❌ degraded |
| **false_confirmed** AUPRC | 0.338 | 0.243 | −28% | ❌ degraded |
| false_confirmed ECE | 0.348 | 0.270 | −7.8pp | ↑ calibration improved |
| ifd AUROC (5f) | 0.898 | 0.879 | −2pp | ↓ minor |
| ifd10 AUROC | 0.765 | 0.786 | +2pp | ↑ minor |
| ifd20 AUROC | 0.744 | 0.744 | flat | = |
| recoverable AUROC | 0.894 | 0.911 | +2pp | ↑ minor |
| ifd10 event recall @t=0.5 | **8.2%** | **64.7%** | **+56pp** | 🔥 |
| ifd20 event recall @t=0.5 | 35.3% | 85.9% | +51pp | 🔥 |

**Val verdict:** fc signal degraded with proxy memory. ifd recall improved dramatically — the memory features help trigger ifd heads earlier, but the fc head itself overfits the proxy representation.

---

### Diagnostic Split (4 hard sequences — held-out, never trained)

| Head | v2_corrected baseline | v2.1 memory | Δ | Gate |
|---|---:|---:|---:|---|
| **false_confirmed** AUROC | **0.548** | **0.796** | **+24.8pp** | ✅ >0.70 GO |
| **false_confirmed** AUPRC | **0.248** | **0.518** | **+108%** | ✅ strong |
| false_confirmed ECE | 0.394 | 0.331 | −6.3pp | ↑ |
| ifd10 AUROC | 0.626 | 0.514 | −11.2pp | ❌ degraded |
| ifd20 AUROC | 0.554 | 0.422 | −13.2pp | ❌ degraded |
| ifd (5f) AUROC | 0.660 | 0.706 | +4.6pp | ↑ |
| failure_in_10 AUROC | 0.686 | 0.694 | +0.8pp | = |

**Diagnostic verdict:** false_confirmed dramatically improved on hard identity-drift cases. The memory features carry a real signal for distractors. However ifd heads degraded — proxy embeddings confuse the multi-horizon temporal reasoning. Against the stricter retrained no-memory diagnostic baseline, fc AUROC improves `0.598 → 0.796` (+19.8pp), so the direction still validates.

---

### Gate Assessment

| Gate | Threshold | Result | Decision |
|---|---|---|---|
| Diagnostic fc AUROC minimum | > 0.65 | **0.796** | ✅ GO |
| Diagnostic fc AUROC strong | > 0.70 | **0.796** | ✅ GO |
| Val fc AUROC no-regression | ≥ 0.870 | **0.857** | ❌ degraded |
| Pattern: val-only improvement without diagnostic | KILL | Diagnostic improved, val degraded | ✅ not a KILL |

**Summary:** Memory direction **VALIDATED** on hard cases. Proxy embedding **INSUFFICIENT** for val — L2-normalized 28-scalar telemetry is too coarse for cosine similarity in RAM/DRM. The ifd degradation on diagnostic is a side-effect of proxy noise bleeding into temporal GRU state.

---

## 2. Policy Sweep

Both sweeps: 360 configs, 49 sequences, 30 892 frames.

### No-Memory Baseline (v2_corrected, 28-dim)

| Config | fc_t | reinit_t | macro_tcr | wrir | recall | density/1kf |
|---|---|---|---|---|---|---|
| Best constrained (wrir=0) | 0.4 | 0.8 | 0.0361 | 0.000 | 0.953 | 532 |
| fc=0.5 | 0.5 | 0.8 | 0.0701 | 0.000 | — | — |
| fc=0.6 | 0.6 | 0.8 | 0.1074 | 0.000 | — | — |

**Note:** mem_margin_threshold, eprocess_alpha have zero effect in no-mem file — all configs at same fc/reinit produce identical metrics.

### With Memory (v2.1, 37-dim)

| Config | fc_t | reinit_t | mem_t | macro_tcr | wrir | recall | density/1kf |
|---|---|---|---|---|---|---|---|
| **Best** | 0.4 | 0.4 | **+0.20** | **0.0031** | 0.000 | **1.000** | 889 |
| Runner-up | 0.4–0.9 | 0.4 | **+0.00** | **0.0055** | 0.000 | 0.977 | 677 |
| mem_t=+0.10 | 0.4 | 0.4 | **+0.10** | 0.0235 | 0.000 | 1.000 | 885 |
| mem_t=−0.10 | 0.4 | 0.4 | **−0.10** | 0.0069 | 0.256 | 0.965 | 636 |

**Delta at best constrained (wrir=0):**

| Metric | No-mem | With-mem | Delta |
|---|---:|---:|---|
| macro_tcr | 0.0361 | **0.0031** | **−91% (11.7×)** |
| wrir | 0.000 | 0.000 | = |
| recall | 0.953 | **1.000** | +4.7pp |
| density/1kf | 532 | 889 | +67% (more interventions) |

**Caution on best config:** mem_t=+0.20 produces missed_safe_update_rate=0.671 — the policy blocks 67% of safe template updates. The runner-up (mem_t=0.00, macro_tcr=0.0055) is more conservative and may be safer for deployment.

**eprocess_alpha has zero effect** on policy metrics — the e-process layer is not wired into the policy sweep's intervention trigger. E-process is parallel monitoring only.

---

### E-Process (formal mode, ifd10 risk, α=0.10, ε=0.50)

| Metric | Value | Target | Status |
|---|---|---|---|
| Median lead time | **12.0 f** | ≥ 3f | ✅ |
| Mean lead time | 12.0 f | | |
| Failure event recall | **3.1%** | ≥ 60% | ❌ |
| False alerts / 1000f | **0.21** | ≤ 100 | ✅ |
| Seq-level FAR | 0.167 | | |
| Total events | 64 | | |
| TP alerts | 2 | | |

Raw detector baselines (threshold=0.5 for comparison):

| Detector | FA/1000f | Recall |
|---|---:|---:|
| raw ifd5 | 234.2 | 79.7% |
| raw ifd10 | 9.5 | 6.3% |
| raw ifd20 | 25.8 | 21.9% |
| raw fc | 335.5 | 84.4% |
| **e-process (ifd10)** | **0.21** | **3.1%** |

E-process achieves near-zero FAR but recall collapses to 3.1%. **Verdict: analysis tool only.** Not suitable as primary runtime gate. Use calibrated risk hysteresis (p_ifd10 > threshold) instead.

---

## 3. Ablation Roadmap

The 9 memory features are:

| Group | Features | Dims |
|---|---|---|
| Positive memory (RAM) | mem_pos_max_sim, mem_pos_mean_sim, mem_pos_recency_sim, mem_update_age | 0,1,2,7 |
| Negative memory (DRM) | mem_neg_max_sim, mem_neg_mean_sim, mem_neg_count_nearby, mem_neg_size | 3,4,5,8 |
| Margin | mem_target_minus_distractor_margin | 6 |

Positive memory updates when: `p_fc < 0.20 AND p_ifd < 0.30 AND apce_norm > 0.4` (every 5f)  
Negative memory updates when: `secondary_peak_ratio > 0.65 AND p_fc < 0.25` OR `fc_proxy=True` when `p_fc > 0.40`

### Ablations Not Yet Run (needed to attribute effect)

| Ablation | Input dims | Purpose |
|---|---|---|
| telemetry only | 28 | baseline (already have) |
| telemetry + positive memory | 28 + 4 | does RAM alone help? |
| telemetry + negative memory | 28 + 4 | does DRM alone help? |
| telemetry + margin only | 28 + 1 | is the derived margin sufficient? |
| full memory 9-dim | 28 + 9 | current result (already have) |

**Current finding:** Full 9-dim improves diagnostic fc +25pp but degrades val fc −3pp. The margin (dim 6) is the most semantically clean signal — likely the most useful single feature. Recommend running margin-only ablation first before committing to 9-dim.

---

## 4. Gate Decision + Next Steps

### Gate: ✅ GO — Memory direction validated

```
diagnostic fc AUROC: 0.548 → 0.796 (+24.8pp vs corrected baseline; +19.8pp vs retrained baseline)  >0.70 strong GO
val fc AUROC:        0.885 → 0.857 (−2.9pp)   proxy insufficient
pattern:             diagnostic improved, val degraded → proxy overfit
```

The "val-only improvement" KILL condition does NOT fire. The correct interpretation is:
> Proxy memory identifies hard identity-drift cases in diagnostic, but the 28-scalar L2-norm is too coarse for general-purpose cosine similarity on val sequences.

### Runtime-Safe Policy (since fc diagnostic improved)

Do not aggressively deploy. Conservative integration only:

| Intervention | Trigger | Status |
|---|---|---|
| Block template update | p_fc > 0.60 AND mem_margin < 0.00 | ✅ safe to test |
| Recovery abstain | p_fc > 0.70 | ✅ safe to test |
| No aggressive detector reinit | p_fc > 0.80 | ✅ safe to test |

Use policy config: `fc_t=0.6, reinit_t=0.7, mem_t=0.00` (runner-up, macro_tcr=0.0055, wrir=0.0, recall=0.977, density=677/1kf).  
Do NOT use best config (mem_t=0.20) — missed_safe_update_rate=0.671 is too aggressive for deployment.

### Next: Phase 4B Real — Real Crop Embeddings

Since proxy is insufficient for val:

| Priority | Task |
|---|---|
| 1 | Extract per-frame crop embeddings from SGLATrack trunk (already computed in forward pass, zero extra cost) |
| 2 | Replace `_make_proxy_embedding()` in memory_features.py with real 256/512-dim backbone embedding |
| 3 | Retrain v2.2 with real memory sidecar |
| 4 | Gate: diagnostic fc > 0.796 (beat proxy) AND val fc ≥ 0.870 (no regression) |

If SGLATrack embeddings are weak → DINOv2 offline crop extraction.  
If real memory still doesn't beat proxy on diagnostic → representation quality issue → contrastive distractor head.

### Run ablations in parallel with real embedding extraction

Before investing in SGLATrack hook plumbing, run the 4 ablations above on proxy to understand which feature group drives the diagnostic gain. If margin-only (1 dim) achieves most of the +25pp, the hook needs only to export a similarity margin, not full embedding vectors.

---

## 5. Red Lines (standing)

- NO training on diagnostic sequences (bike2, Gull2, Sheep1, StreetBasketball1, uav0000164)  
- NO calibration on train split  
- Proxy memory label: "telemetry-proxy ablation" in any writeup — not "DAM-style memory"  
- Val degradation is NOT a KILL if diagnostic improved (different failure modes)  
- Do NOT compare micro-tcr (new, per-frame) with macro-tcr (old policy.py) as equivalent  
- Do NOT claim e-process for runtime gating while recall < 10%  
- ifd10/ifd20 recall numbers (64.7%/85.9%) are at threshold=0.5 with memory sidecar — not comparable to raw ifd baseline without sidecar

---

## Artifact Inventory (current canonical state)

| Artifact | Path | Status |
|---|---|---|
| v2 labels (fixed, full-horizon) | `saltr/data/salt_rd_v2_labels.npz` | ✅ canonical |
| memory sidecar (228 seqs, 0 oracle) | `saltr/data/salt_rd_memory_sidecar.npz` | ✅ canonical |
| v2_corrected checkpoint (28-dim) | `saltr/checkpoints/v2_corrected/saltrd_best.pt` | ✅ baseline |
| v2.1 memory checkpoint (37-dim) | `saltr/checkpoints/v2_1_memory/saltrd_best.pt` | ✅ current |
| preds all splits | `saltr/results/preds_all_v2_retrained.json` | ✅ |
| eval val v2 baseline | `saltr/results/eval_val_v2_retrained.json` | ✅ canonical |
| eval val v2.1 memory | `saltr/results/eval_val_v2_1_memory.json` | ✅ |
| eval diagnostic v2 corrected baseline | `saltr/results/eval_diagnostic_v2_corrected.json` | ✅ headline +24.8pp comparison |
| eval diagnostic v2 retrained baseline | `saltr/results/eval_diagnostic_v2_retrained.json` | ✅ stricter +19.8pp comparison |
| eval diagnostic v2.1 memory | `saltr/results/eval_diagnostic_v2_1_memory.json` | ✅ |
| policy sweep no-mem | `saltr/results/policy_sweep_v2_baseline_nomem.json` | ✅ canonical |
| policy sweep with memory | `saltr/results/policy_sweep_v2_retrained.json` | ✅ canonical |
| e-process val (ifd10 mode) | `saltr/results/eprocess_val_v2_retrained.json` | ✅ analysis only |

**Tests:** 183 passing.
