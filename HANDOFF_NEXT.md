# HANDOFF — SALT-RD: Reliability + Neural Scene Dynamicity for UAV SOT

**Дата:** 2026-05-19  
**Оновлено:** 2026-05-20 — pipeline complete, BORDERLINE verdict, next = calibration + LODO  
**Статус:** Phases 0→1a→1b→1c→2 DONE + tested. NPZ collected, model trained, eval done.  
**Читай також:** `THOUGHTS.md`, `papers/README.md`, `FROZEN.md`

---

## ✅ Реалізовано — Phase 0 (commit ecfcb0f)

| Завдання | Статус | Файл |
|---|---|---|
| `FROZEN.md` — policy freeze | ✅ done | `FROZEN.md` |
| Config gates в YAML | ✅ done | `configs/prod/salt.yaml` |
| `enable_ce` gate → runtime | ✅ done | `sglatrack.py` + `salt_runner.py` |
| `enable_velocity_drift` gate → runtime | ✅ done | `target_state_assessor.py` + `salt_runner.py` |
| `enable_dynamic` / `enable_salt_rd` в YAML | ✅ done | `configs/prod/salt.yaml` (motion_predictor.enabled вже був false) |
| `TrackState` extended: +5 fields | ✅ done | `types.py` |
| `score_map_stats` computation в SGLATracker | ✅ done | `sglatrack.py` (обидва update/update_with_state) |
| `saltr/` scaffold | ✅ done | `saltr/src/salt_r/` |
| `collect_features.py` skeleton + NPZ schema | ✅ done | `saltr/src/salt_r/collect_features.py` |
| `model.py` / `train.py` / `eval.py` stubs | ✅ done | `saltr/src/salt_r/` |
| 174 тести проходять | ✅ verified | — |

### Що реально gate-ується в runtime

| Gate | YAML ключ | Де читається | Ефект при `false` |
|---|---|---|---|
| `enable_ce` | top-level | `salt_runner.py:183` → `SGLATracker.__init__` | `ce_keep_rate = 1.0` (no pruning) |
| `enable_velocity_drift` | top-level | `salt_runner.py:193` → `TargetStateAssessor.__init__` | `is_drifted()` не викликається, DISTRACTOR_RISK недосяжний |
| `enable_dynamic` (alias) | `motion_predictor.enabled` | вже `false` у YAML — `lstm_pred` завжди `None` | DYNAMIC state недосяжний |
| `enable_salt_rd` | top-level | **не читається ще** — для Phase 2 |

### score_map_stats — поля в TrackState

```python
track_state.score_map_stats = {
    "top1":             float,   # highest response value (raw)
    "top2":             float,   # second highest
    "peak_margin":      float,   # top1 - top2 (ambiguity metric)
    "peak_width":       int,     # cells above 50% of peak
    "n_secondary":      int,     # 0 placeholder (v1: local-maxima detection)
    "peak_distance":    float,   # peak location distance from map center (cells)
    "heatmap_mass_topk":float,   # fraction of total mass in top-10 cells
}
```

---

## Рішення

Йдемо не в "ще один tracker" і не в rule-based SALT v3. Центральна ідея тепер:

> **SALT-RD — proactive tracking-risk dynamicity controller for real-time UAV single-object tracking.**

SGLATrack/SALT v3 лишається frozen baseline. Нова наукова робота — легка нейромережа, яка з внутрішніх сигналів трекера, motion/flow/appearance сигналів і offline teacher labels проактивно прогнозує:

- наскільки сцена tracking-dynamic саме для поточного target;
- tracking-risk до того, як AUC вже впав;
- false-confirmed: чи трекер впевнено сидить на неправильному об'єкті;
- recovery readiness: чи є сенс запускати або приймати re-acquisition;
- чи буде failure найближчим часом;
- чи треба витрачати повний compute;
- чи можна безпечно оновлювати template.

Тобто SALT-RD не обіцяє автоматично побити SOTA AUC backbone-ів. Він має дати deployment-важливу властивість: **проактивно оцінити tracking-risk і вирішити, коли довіряти трекеру, коли економити compute, коли заборонити template update, і коли робити re-acquisition.**

---

## Наукова позиція

**Thesis:** "Proactive Tracking-Risk Dynamicity for Failure-Aware Real-Time UAV Single-Object Tracking."

**Не thesis:** "A new UAV tracker with best AUC." Це вже щільно закрито MSTFT, MATA, UTPTrack, LGTrack, BDTrack, Mamba-based trackers.

**Ключова емпірична точка:** `uav0000164` — AUC=0.174 при 99% CONFIRMED. APCE бачить гострий peak, але не бачить identity switch. Це і є signature failure mode для статті.

**Що ми повинні показати:**

1. Neural dynamicity head краще за APCE/LSTM/rule thresholds визначає складні динамічні моменти.
2. Reliability heads ловлять false-confirmed і imminent failure краще за APCE/PSR/entropy.
3. Policy на основі SALT-RD дає deployment win: lower wrong-reinit, safer template updates, better risk-coverage, and/or lower GFLOPs at bounded AUC loss.

---

## Конкуренти 2024-2026 і наша ніша

| Paper | Що закриває | Чому це конкурент | Наша різниця |
|---|---|---|---|
| MSTFT 2026 | Mamba backbone, small UAV targets, dynamic template fusion | Сильний AUC benchmark: UAV123 AUC=79.4%, 45 FPS | Ми не замінюємо backbone; беремо response/temporal/motion verification як labels/features для controller |
| MATA 2026 | Modular async UAV tracking, ego-motion, NT2F, embedded protocol | Найближчий systems reference | MATA валідовує measurement для EKF; SALT-RD вчить dynamicity/reliability policy для compute/recovery/template safety |
| UTPTrack CVPR 2026 | Unified token pruning SR/DT/ST, 65%+ token pruning | CE/CTEM як novelty більше не працює | SALT-RD може керувати коли застосовувати pruning, а не винаходити pruning |
| ABTrack 2024/2025 | Adaptive ViT block bypass | Динамічний compute уже є | ABTrack bypass-ить blocks за task difficulty; SALT-RD прогнозує tracking risk/dynamicity і має GT IoU labels |
| BDTrack 2025 | UAV motion blur + dynamic early exit | Прямий конкурент до "dynamic scene" | BDTrack оптимізує backbone/early exit; SALT-RD робить tracker-agnostic controller і failure calibration |
| UncL-STARK 2026 | Heatmap uncertainty for depth adaptation | Близький uncertainty-guided compute baseline | Їх uncertainty локалізаційна; наша ключова задача — false-confirmed identity drift + recovery safety |
| LGTrack 2026 | Dynamic layer selection + occlusion robustness | Реальний lightweight UAV competitor | LGTrack є tracker; SALT-RD є wrapper/controller і може оцінювати LGTrack/SGLATrack/MSTFT |
| PTDT 2026 | Point-tracking-guided dynamic tokens/template update | Найближчий reference для point consistency | Ми використовуємо point tracking як offline teacher/feature для reliability/dynamicity, не як основний tracker |
| OOTU 2025 | BBox uncertainty regression | Базова робота для calibration/ECE | OOTU оцінює де bbox; SALT-RD оцінює чи це правильний об'єкт і чи варто діяти |
| LoRAT 2024 | PEFT/domain adaptation для trackers | Кращий fallback якщо reliability head не дає AUC | Phase 6: domain-adapt backbone, не core novelty Phase 1 |

---

## Крок -1 — структурна реорганізація

Не робити `git mv` всього repo в `saltv3/`. Це зламає imports, packaging, paths. Робимо **policy freeze + новий пакет `saltr/` поряд**.

```text
uav-tracker-detector/
  src/uav_tracker/             # frozen baseline/SALT v3, мінімальні зміни тільки для telemetry
  configs/prod/
  weights/
  tests/unit/
  FROZEN.md                    # "Do not modify src/uav_tracker except telemetry/config gates"

  saltr/
    src/salt_r/
      collect_features.py      # telemetry + GT IoU + dynamicity labels
      model.py                 # SALT-RD GRU/TCN heads
      train.py                 # supervised multi-head training
      eval.py                  # reliability/dynamicity/policy metrics
      policy.py                # maps probabilities to tracker actions
      integrate.py             # wrapper around frozen tracker
    data/
    configs/

  papers/
  HANDOFF_NEXT.md
  THOUGHTS.md
```

Config gates замість видалення ablation paths:

```yaml
enable_ce: false
enable_dynamic: false          # old LSTM dynamic branch disabled
enable_velocity_drift: false   # replaced by learned P(false_confirmed)
enable_salt_rd: false          # off until GO gate
```

---

## Критичні ризики і як їх побороти

### 1. Label leakage / self-teaching

Смертельний ризик для статті: навчити модель повторювати `_decide_state()`, APCE thresholds або `scene_class`.

**Побороти:** labels тільки з GT IoU, future IoU, GT target motion, camera/flow residuals або offline teacher outputs. Жоден target label не походить з TSA state machine.

### 2. "Dynamicity" може перетворитись на стару scene classification

Нам не потрібен клас `STATIC/DYNAMIC/OCCLUDED` як декоративна назва сцени. Нам потрібна **tracking-relevant dynamicity**.

**Побороти:** визначати dynamic labels через те, що впливає на tracking:

- target normalized velocity/acceleration;
- bbox scale/aspect change;
- camera ego-motion magnitude;
- residual target motion after ego-motion compensation;
- future degradation of IoU under cheap/normal tracker mode.

### 3. False-confirmed дуже рідкісний клас

Наївний BCE дасть 97-99% accuracy при random AUROC.

**Побороти:** AUPRC primary, focal/weighted BCE, hard-negative suite: `uav0000164`, `bike2`, `Gull2`, `Sheep1`, `StreetBasketball1`. Diagnostic suite не входить у train.

### 4. Compute policy може зменшити GFLOPs, але зламати AUC

Dynamicity head легко перетворити на aggressive skip policy.

**Побороти:** training/eval policy має oracle regret:

- full tracker output;
- cheap/pruned/bypass output;
- label `needs_full_compute = cheap_iou_drop > 0.03 OR cheap_failure`;
- report AUC-vs-GFLOPs Pareto, not only average FPS.

### 5. Score-map geometry зараз неповна

APCE/PSR/entropy замало для false-confirmed і dynamicity.

**Побороти:** розширити `TrackState`:

```python
score_map_stats = {
    "top1": ...,
    "top2": ...,
    "peak_margin": ...,
    "peak_width": ...,
    "n_secondary": ...,
    "peak_distance": ...,
    "heatmap_mass_topk": ...,
}
```

### 6. Split має бути по sequence і по domain

Frame-level split дає leakage. Alphabetical split дає випадкову науку.

**Побороти:** stratified group split by sequence across UAV123 + VisDrone-SOT + DTB70, плюс LODO:

- train: UAV123 + VisDrone + DTB70 train sequences;
- val: held-out sequences from each dataset;
- diagnostic: only hard negatives;
- LODO: train on two datasets, test on third.

### 7. Reproducibility

Hardcoded SGLATrack path і env drift зламають наступну сесію.

**Побороти:** env var для external tracker path, config hash у NPZ, exact tracker checkpoint id, `uav-tracker doctor`.

---

## Крок 0 — cleanup + telemetry

Мета: зробити frozen baseline, який стабільно генерує дані для SALT-RD.

1. Додати `FROZEN.md`.
2. Додати config gates і реально прочитати їх у runtime.
3. Не видаляти CE/DYNAMIC/VelocityDrift фізично: вони потрібні для ablations.
4. Розширити `TrackState`:
   - `score_map_stats`;
   - `motion_stats`;
   - `flow_stats`;
   - `appearance_stats`;
   - `compute_mode`.
5. Додати smoke:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/ -q
PYTHONPATH=src .venv/bin/python -m uav_tracker doctor
```

---

## Фаза 1 — SALT-RD v0 dataset

Canonical path: `saltr/src/salt_r/collect_features.py`.

### Features v0

Target: 24-32 scalar features per frame. Не треба одразу CNN score-map encoder.

```text
Score map:
  apce_raw, apce_norm, psr, entropy,
  peak_margin, peak_width, n_secondary, peak_distance, heatmap_mass_topk

Temporal response:
  apce_ratio_5, apce_ratio_20, entropy_delta_5,
  peak_margin_delta_5, confirmed_streak, low_conf_streak

Target dynamics:
  bbox_cx_velocity, bbox_cy_velocity, bbox_speed_norm,
  bbox_accel_norm, scale_ratio, aspect_ratio_delta, dist_to_search_border

Camera/flow:
  global_flow_mag, target_flow_mag, ego_motion_residual,
  flow_iou, flow_residual, flow_consistency

Appearance:
  cosine_static_template, cosine_recent_template,
  embedding_drift_static, embedding_drift_recent

Detector/recovery context, optional:
  n_candidates, hint_distance_norm, top1_top2_detector_gap
```

### Labels v0

All labels are GT/teacher-derived, never `_decide_state()`-derived.

```python
correct[t] = iou[t] >= 0.5

false_confirmed[t] = (
    iou[t] < 0.2
    and apce_raw[t] > 100
)

failure_in_5[t] = (
    iou[t] >= 0.5
    and mean(iou[t+1:t+6]) < 0.3
)

recoverable[t] = (
    iou[t] < 0.2
    and max(iou[t+5:t+15]) >= 0.5
)

target_dynamic[t] = percentile_rank(
    speed_norm + 0.5 * accel_norm + 0.5 * abs(scale_delta),
    by_sequence=True,
) >= 0.75

camera_dynamic[t] = percentile_rank(
    global_flow_mag + ego_motion_residual,
    by_sequence=True,
) >= 0.75

hard_dynamic_scene[t] = (
    (target_dynamic[t] or camera_dynamic[t])
    and (peak_margin_low[t] or flow_consistency_low[t] or iou[t+1:t+6].min() < 0.3)
)
```

`needs_full_compute[t]` має два режими:

1. **Proper oracle mode:** run full tracker and cheap/pruned mode, label full if cheap mode causes IoU drop or failure.
2. **Bootstrap mode:** approximate with `hard_dynamic_scene OR failure_in_5` until oracle data exists.

Proper oracle mode потрібен перед будь-якими paper claims про GFLOPs/FPS.

### NPZ schema

```text
features/{seq}:       float32 (n_frames, n_features)
feature_names:        list[str]
feature_units:        list[str]
labels/{seq}:         int8 (n_frames, n_labels)
label_names:          ["correct", "false_confirmed", "failure_in_5", "recoverable",
                       "target_dynamic", "camera_dynamic", "hard_dynamic_scene",
                       "needs_full_compute"]
iou_trace/{seq}:      float32 (n_frames,)
bbox_pred/{seq}:      float32 (n_frames, 4)
bbox_gt/{seq}:        float32 (n_frames, 4)
sequence_name/{seq}:  str
dataset/{seq}:        str
split/{seq}:          str
tracker_version:      str
tracker_config_hash:  str
created_at:           str
```

---

## Фаза 1 — model/training

Canonical path: `saltr/src/salt_r/model.py`, `train.py`.

### Model v0

Start simple. The contribution is labels + policy + evaluation, not a huge network.

```python
class SALTRD(nn.Module):
    # input: (B, T, 24-32)
    # GRU(input_dim, hidden=64, layers=2, dropout=0.2)
    # shared temporal state -> separate heads
    # heads:
    #   P(false_confirmed)
    #   P(failure_in_5)
    #   P(recoverable)
    #   P(target_dynamic)
    #   P(camera_dynamic)
    #   P(hard_dynamic_scene)
    #   P(needs_full_compute)
```

Loss:

- weighted BCE or focal BCE per head;
- higher weight for `false_confirmed`;
- sequence-balanced sampler;
- report base rate per label.

Alternatives only after v0:

- TCN if GRU overfits;
- tiny SSM/Mamba-like temporal block only if it materially improves latency/accuracy;
- score-map CNN encoder only after scalar telemetry GO/NO-GO.

---

## Фаза 1 — evaluation

Canonical path: `saltr/src/salt_r/eval.py`.

### Reliability metrics

- AUROC/AUPRC per head;
- base rate per head;
- ECE/Brier/NLL;
- false-confirmed recall@5% FPR;
- failure warning lead time;
- NT2F from MATA;
- bootstrap 95% CI by sequence.

### Dynamicity metrics

- AUROC/AUPRC for `target_dynamic`, `camera_dynamic`, `hard_dynamic_scene`;
- confusion matrix by dataset and target-size bin;
- dynamicity calibration curve;
- correlation with AUC drops and APCE drops;
- per-sequence timeline plot: IoU, APCE, P(dynamic), P(false_confirmed), compute decision.

### Policy/deployment metrics

- wrong re-init rate;
- template corruption rate;
- risk-coverage curve;
- abstention gain;
- AUC-vs-GFLOPs Pareto;
- FPS on Apple MPS for policy overhead;
- compute regret: AUC loss at 10/15/20% GFLOPs reduction.

---

## GO / NO-GO gate

| Metric | GO | STOP |
|---|---:|---:|
| AUPRC false_confirmed | > 0.30 | < 0.15 |
| AUROC false_confirmed | > 0.65 | < 0.55 |
| AUROC failure_in_5 | > 0.75 | < 0.65 |
| AUROC hard_dynamic_scene | > 0.75 | < 0.60 |
| AUROC needs_full_compute | > 0.70 | < 0.60 |
| ECE false_confirmed | < 0.12 | > 0.20 |
| Policy AUC loss at 15% GFLOPs saving | < 0.005 | > 0.020 |
| Wrong re-init reduction | > 25% | < 5% |

If SALT-RD cannot beat APCE/PSR/entropy baselines on the same labels, stop. Do not tune thresholds until it looks good.

---

## Фаза 2 — policy integration

Canonical path: `saltr/src/salt_r/policy.py`, `integrate.py`.

```python
probs = salt_rd.predict(window)

if probs.false_confirmed > 0.70:
    action.template_update = "block"
    action.recovery = "abstain_or_verify"
    action.compute = "full"

elif probs.hard_dynamic_scene > 0.65:
    action.compute = "full"
    action.search = "normal_or_expand_conservatively"
    action.template_update = "verify"

elif probs.needs_full_compute < 0.25 and probs.failure_in_5 < 0.20:
    action.compute = "cheap"
    action.template_update = "allow_if_appearance_stable"

if probs.recoverable > 0.60 and probs.false_confirmed < 0.40:
    action.recovery = "run_detector_or_matcher"
```

Policy must be evaluated offline first. Runtime integration only after replay simulation shows bounded regret.

---

## Повний фазовий план

```text
Фаза 0:  Freeze + config gates + telemetry extensions (0.5-1 день)
Фаза 1a: Collect SALT-RD scalar dataset with GT IoU labels (1-2 дні)
Фаза 1b: Train/eval SALT-RD multi-head GRU (1-2 дні)
Фаза 1c: Offline policy replay: risk, recovery, compute, template safety (1-2 дні)
Фаза 2:  Runtime integration behind enable_salt_rd flag (1-2 дні)
Фаза 3:  SALT-Match candidate accept/reject dataset for recovery (1 тиждень)
Фаза 4:  CoTracker3/PTDT-style point consistency offline teacher (1-2 тижні)
Фаза 5:  DINO/SAM/CoTracker distillation into lightweight features (2-4 тижні)
Фаза 6:  LoRAT domain adaptation fallback if AUC remains capped (3-6 тижнів)
```

Priority rule:

- If Phase 1 fails reliability/dynamicity GO gates, do not implement runtime policy.
- If reliability works but compute policy fails, publish reliability/recovery angle and leave compute as negative/secondary.
- If both fail, move to LoRAT/domain-adapted backbone and write the negative result honestly.

---

## ✅ Реалізовано — Phases 1a→2 + fixes + tests (commits 64f307b…7d9b693)

### Модулі saltr/

| Файл | Статус | Ключові деталі |
|---|---|---|
| `collect_features.py` | ✅ | 28 features, GT labels, `_TruncatedSequence`, `root=None` autodetect |
| `model.py` | ✅ | SALTRD GRU hidden=64 layers=2, ~43k params, 7 ModuleDict heads, sigmoid в head |
| `train.py` | ✅ | focal BCE, empirical pos_weight, early stop on AUPRC(fc), subclasses model.SALTRD |
| `eval.py` | ✅ | AUROC/AUPRC/ECE/Brier/NLL/recall@5%FPR, NT2F, bootstrap CI (seq-level), GO/NO-GO |
| `policy.py` | ✅ | TrackerAction, RiskThresholds, apply_policy (4-priority chain), replay_policy |
| `integrate.py` | ✅ | SALTRDRunner wraps frozen SALTRunner, FeatureBuffer, run_with_risk |
| `run_phase1.sh` | ✅ | collect→train→eval→predictions→policy replay (one command) |

### Важливі pipeline fixes (commit bc34a5b)

| Bug | Fix |
|---|---|
| Double-sigmoid in eval.py | `out[h].detach().cpu().numpy()` + `HEAD_NAMES` + `np.clip`; ECE 0.537→0.320 |
| IoU key mismatch в policy.py | raw `np.load` + strip `iou_trace/` prefix + warning on missing |
| `--dry-run` завантажував модель | SALTRunner.from_config() тільки коли `not dry_run` |
| Predictions export відсутній | `--predictions-output` зберігає `{seq_key: [{head: prob},...]}` JSON |
| Label-head mapping off-by-one | HEAD_NAMES lookup (не index) для named prediction mapping |
| train.py дублював SALTRD | підклас model.SALTRD, не окрема архітектура |

### Unit tests (200 passed — commit 7d9b693)

| Файл | Тестів | Що покрито |
|---|---|---|
| `test_saltr_collect_features.py` | 3 | flat labels, key collision, _TruncatedSequence |
| `test_saltr_model.py` | 6 | forward contract, param count, train/eval compat, no double-sigmoid |
| `test_saltr_eval.py` | 6 | ECE, NT2F, bootstrap CI, GO/NO-GO, predictions schema |
| `test_saltr_policy.py` | 7 | apply_policy cases, replay_policy real IoU |
| `test_saltr_integrate.py` | 4 | FeatureBuffer, extract_features shape/NaN |

---

## 📊 Поточні результати (val split, 49 sequences)

NPZ: `saltr/data/salt_rd_v0.npz` — 228 seqs, 161k frames (UAV123+VisDrone+DTB70)  
Checkpoint: `saltr/checkpoints/saltrd_best.pt` — epoch 4 (early stop)

| Head | base% | AUROC | AUPRC | ECE | Brier | R@5FPR |
|---|---|---|---|---|---|---|
| **false_confirmed** | 5.5 | **0.884** | **0.331** | 0.320 | 0.162 | 0.445 |
| **failure_in_5** | 0.1 | **0.863** | 0.010 | 0.244 | 0.069 | 0.333 |
| recoverable | 0.6 | 0.892 | 0.044 | 0.260 | 0.091 | 0.481 |
| target_dynamic | 5.6 | 0.769 | 0.162 | 0.375 | 0.192 | 0.264 |
| camera_dynamic | 24.7 | 0.573 | 0.302 | 0.237 | 0.241 | 0.063 |
| hard_dynamic_scene | 11.9 | 0.638 | 0.264 | 0.332 | 0.209 | 0.204 |
| needs_full_compute | 12.0 | 0.641 | 0.266 | 0.334 | 0.211 | 0.211 |

**NT2F(IoU≥0.5):** mean=0.555, std=0.375, never_failed=17/49  
**Bootstrap AUPRC(fc) 95%CI:** [0.216, 0.538] (n_seq=19)

**GO/NO-GO: BORDERLINE**

| Metric | Value | GO threshold | Status |
|---|---|---|---|
| AUPRC(fc) | 0.331 | > 0.30 | ✅ |
| AUROC(fc) | 0.884 | > 0.65 | ✅ |
| AUROC(fail5) | 0.863 | > 0.75 | ✅ |
| AUROC(hard_dyn) | 0.638 | > 0.75 | ⚠️ |
| AUROC(full_cmp) | 0.641 | > 0.70 | ⚠️ |
| ECE(fc) | 0.320 | < 0.12 | ❌ needs calibration |

**Policy replay (val):**
```
wrong_reinit_rate:       0.216  ← 21.6% recovery → wrong object
template_blocked_rate:   0.004  ← conservative at default thresholds
template_corruption_rate: 0.097 ← target for false_confirmed improvement
compute_cheap_rate:      0.000  ← needs_full_compute head too weak
abstention_gain:         0.051  ← blocking update gives +5% IoU
```

**Diagnostic sequences (4 hard seqs: bike2, Gull2, Sheep1, StreetBasketball1):**
- AUROC(fc)=0.604, AUPRC(fc)=0.279 (base=20.3%), NT2F=0.105 — model weaker on hardest cases

---

## Стартовий промпт для наступної coding сесії

> Pipeline complete. NPZ + checkpoint exist. BORDERLINE verdict. Next = calibration → LODO → operating point tuning.

```text
Прочитай HANDOFF_NEXT.md секцію "Поточні результати" та FROZEN.md.

СТАН:
- saltr/data/salt_rd_v0.npz — 228 seqs, 161k frames ✓
- saltr/checkpoints/saltrd_best.pt — trained, early stop epoch 4 ✓
- eval_val.json — val metrics з fixed pipeline ✓
- GO/NO-GO: BORDERLINE (ECE=0.320 > GO 0.12, hard_dyn AUROC=0.638 < GO 0.75)

ЗАДАЧА 1 — Temperature scaling для ECE
ECE(fc)=0.320 → target <0.12. Реалізуй у saltr/src/salt_r/eval.py:

  from torch import nn
  class TemperatureScaler(nn.Module):
      # Learn T on val split, apply on val/test
      # Scale logits (before sigmoid) by 1/T
      # Fit: NLL minimization over val set
  
  Після calibration перегенеруй eval_val.json і перевір ECE.
  НЕ роби calibration на train split.

ЗАДАЧА 2 — LODO eval (обов'язкова для paper)
  PYTHONPATH=src:saltr/src python saltr/src/salt_r/train.py \
    --npz saltr/data/salt_rd_v0.npz \
    --output saltr/checkpoints/lodo_dtb70/

  Але спочатку потрібен новий NPZ з LODO split:
  - train: uav123 + visdrone_sot sequences
  - val: dtb70 sequences  
  Або: зробити окремий збір тільки для DTB70 і оцінити checkpoint trained on
  UAV123+VisDrone.

ЗАДАЧА 3 — Аналіз weak heads
  hard_dynamic_scene AUROC=0.638 < 0.75. Перевір:
  - base rate на train split (чи не < 5% → дуже рідко)
  - correlation між hard_dynamic_scene labels і actual IoU drops
  - чи compute_labels правильно рахує hard_dynamic_scene (HANDOFF_NEXT.md §Labels v0)

ЗАДАЧА 4 — Policy operating point
  compute_cheap_rate=0.000 означає поточний поріг (needs_full_compute < 0.25) не спрацьовує.
  needs_full_compute base rate = 12%, AUROC=0.641.
  Спробуй поріг 0.40 і перевір compute regret (AUC delta при 10% cheap frames).
  
  Після calibration: запусти policy replay з оновленими ймовірностями:
  PYTHONPATH=src:saltr/src python -m salt_r.policy \
    --probs-json saltr/checkpoints/preds_val.json \
    --npz saltr/data/salt_rd_v0.npz \
    --output saltr/checkpoints/policy_val.json

ЗАДАЧА 5 — Feature diagnostics (якщо ECE не покращується після calibration)
  Перед додаванням нових features (v1 additions з papers/README.md):
  - feature importance: permutation importance або gradient-based
  - correlation matrix features vs labels
  - які features мають найбільший signal для false_confirmed

RED LINES (не робити):
- НЕ тюнити GO thresholds щоб отримати GO verdict
- НЕ тренувати на діагностичних sequences (bike2, Gull2, Sheep1, StreetBasketball1)
- НЕ робити calibration на train split
- НЕ видаляти CE/DYNAMIC/VelocityDrift з src/uav_tracker/ (потрібні для ablations)
```

---

## Papers у `papers/`, які читати першими

| Local file | Why |
|---|---|
| `papers/MSTFT_2026_Mamba_Based_Spatio_Temporal_Fusion_UAV_Tracking.pdf` | SOTA backbone + response/temporal/motion verification |
| `papers/MATA_2026_Architecture_and_Evaluation_Protocol_UAV_Tracking.pdf` | NT2F, ego-motion, embedded protocol |
| `papers/17_UTPTrack_Unified_Token_Pruning.pdf` / `papers/utptrack.pdf` | token pruning competitor |
| `papers/18_ABTrack_Adaptively_Bypassing_ViT_Blocks.pdf` | adaptive block bypass competitor |
| `papers/09_BDTrack_Motion_Blur_Robust_Dynamic_Early_Exit_UAV.pdf` | UAV dynamic early exit + motion blur competitor |
| `papers/10_UncL_STARK_Uncertainty_Guided_Depth_Adaptation.pdf` / `papers/uncl-stark.pdf` | uncertainty-guided compute baseline |
| `papers/11_LGTrack_Layer_Guided_UAV_Tracking.pdf` | 2026 UAV dynamic layer selection competitor |
| `papers/15_SMTrack_State_Aware_Mamba_Visual_Tracking.pdf` | temporal state/dynamic scenario competitor |
| `papers/ptdt.pdf` | point tracking-guided dynamic tokens/template update |
| `papers/ootu.pdf` | bbox uncertainty calibration baseline |
| `papers/CoTracker3_2025_Simpler_and_Better_Point_Tracking_by_Pseudo_Labeling_Real_Videos.pdf` | offline point teacher |
| `papers/LoRAT_2024_Tracking_Meets_LoRA_Faster_Training_Larger_Model_Stronger_Performance.pdf` | Phase 6 domain adaptation fallback |
