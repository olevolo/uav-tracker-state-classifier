# PROJECT X — Живий документ
**Останнє оновлення:** 2026-05-22  
Консолідований документ: замінює `SUPER_PLAN.md`, `ARCHITECT.md`, `RESULTS.md`, `FROZEN.md`, `bugs.md`.

---

## Мета і gate для публікації

Потрібен **повний причинно-наслідковий ланцюг**:
```
SALT-RD передбачає дію → дія спрацьовує → поведінка трекера змінюється
→ bbox trajectory змінюється → AUC на hard-сценах покращується → full-set AUC покращується
```
Гарного AUROC самого по собі **недостатньо**.

### Мінімальні gates для паперу

| Gate | Ціль |
|---|---|
| Hard subset AUC delta | **≥ +0.10** (interim 14 seqs: +0.08 прийнятно) |
| Full UAV123 AUC delta | **≥ +0.010** |
| Changed bbox frames (hard subset) | **> 0.5%** |
| Wrong reinit rate | не гірше baseline |
| Template corruption | не гірше baseline |
| TSA references в production | нуль |
| Checkpoint feature schema | `saltrd_v3_no_tsa_no_flow` |

**Publication scope:** мінімум 1 трекер + 1 детектор для workshop; 2 трекери + 2 детектори для повного паперу.

**GFLOPs:** ≥ 5% скорочення щоб CE/compute action залишилась в production; < 3% = CE off.

---

## Oracle ceilings (верхні межі)

| Action | Hard AUC delta | Full AUC delta | Harmful seqs |
|---|---|---|---|
| reinit | **+0.0834** | +0.0246 | 0 |
| search_expand | +0.0041 | +0.0005 | 10 |
| template_update | +0.0011 | +0.0274 | 10 |
| center_freeze | +0.000 | +0.0022 | 2 |

→ **Тільки reinit має oracle ceiling ≥ +0.08 без шкоди.** Решта — або слабкі, або шкідливі.

### SGLATrack baselines (reference)

| Dataset/subset | SGLATrack AUC | SALT-RD AUC (current) | Delta |
|---|---|---|---|
| Hard UAV (7 seqs) | 0.206 | ~0.142 | **−0.064** |
| Hard UAV (Phase 8, 7 seqs) | 0.176 | 0.222 | +0.046 |
| car7 | 0.595 | 0.314 | −0.281 |
| truck1 | 0.721 | 0.128 | −0.593 |
| bike2 | 0.176 | 0.056 | −0.120 |

> Поточний SALT-RD гірший за baseline через over-firing (42:1 class imbalance). Phase 0 фіксує цю проблему через `reinit_threshold: 0.95`.

---

## Поточний стан треків

| Трек | Статус | Gate |
|---|---|---|
| Phase 0 — threshold + cleanup | ✓ complete | car7/truck1 ≥ SGLATrack baseline |
| TRACK A-2 — crop_sim + feature schema v2 | ✓ code done | crop_sim AUC > 0.65 на car7/truck1 |
| TRACK B — BUG-30 RT-DETRv2 config | ✓ complete | config loads without error |
| TRACK C — ORTrack-DeiT wrapper | ✓ code done | baseline benchmark runs |

---

## ❗ Блокери — зробити ЗАРАЗ

### 🔴 1. `train_policy.py` — старий 10-dim candidate schema

**Файл:** `saltr/src/salt_r/train_policy.py:74`

```python
# ЗАРАЗ (неправильно):
CANDIDATE_FEATURE_DIM: int = 10
# читає: bbox_x, bbox_y, detector_score, cosine_sim, geometry_area_ratio...

# ТРЕБА:
CANDIDATE_FEATURE_DIM: int = 8
# читати через: CandidateEvent(**ev).to_feature_vector()
```

`feature_schema.py` і `candidate_events.to_feature_vector()` вже 8-dim.  
`train_candidate_scorer.py` імпортує `CANDIDATE_FEATURE_DIM` від `train_policy` — теж зламаний.

---

### 🟡 2. Per-dataset конфіги — `reinit_threshold: 0.65`

Потрібно виправити в:
- `configs/prod/saltrd_uav123.yaml:27`
- `configs/prod/saltrd_dtb70.yaml:27`
- `configs/prod/saltrd_visdrone_sot.yaml:27`

```bash
sed -i '' 's/reinit_threshold: 0.65/reinit_threshold: 0.95/' \
  configs/prod/saltrd_uav123.yaml configs/prod/saltrd_dtb70.yaml configs/prod/saltrd_visdrone_sot.yaml
```

---

## Повна послідовність виконання

```
Паралельно ─────────────────────────────────────────────────────────────

  TRACK A-2 (per dataset: uav123 / dtb70 / visdrone_sot):
    1. [✓] Оновити feature schema (8-dim, crop_sim + aspect_ratio_delta + size_delta_ratio)
    2. [✓] fix train_policy.py CANDIDATE_FEATURE_DIM=8
    3. V5+A-2 collection per dataset
       → feature diagnostic: AUC per feature на car7/truck1 events
       → gate: crop_sim AUC > 0.65
       → якщо < 0.55: розглянути ResNet-18 замість MobileNetV3

  TRACK B (per dataset):
    1. [✓] BUG-30: виправлено salt_detector_rtdetr.yaml
    2. RT-DETRv2 recall audit (--mode sgla, uav2/4/5/7/8, --max-frames 500)
       → gate: uav2–uav8 mean AUC > 0.206 AND car7/truck1 ≥ baseline
       → fallback 1: SAHI (tile_size=256, overlap=0.2) якщо recall низький
       → fallback 2: YOLO26m-P2 fine-tune якщо SAHI недостатньо

  TRACK C (per dataset):
    1. [✓] ORTrack DeiT-tiny ваги + wrapper + config
    2. Baseline benchmark per dataset (--mode sgla, --max-frames 500)
       → gate: ORTrack AUC ≥ SGLATrack на hard UAV subset

Після A-2 + B gates ────────────────────────────────────────────────────

  4. Scorer v2.1 training per dataset (8-dim features, нові checkpoints)
     → оновити checkpoint path в ВСІХ конфігах (salt.yaml, rtdetr, per-dataset)
  5. Regression check per dataset (car7/truck1/bike2 ≥ SGLATrack baseline)
  6. Full benchmark: uav123 / dtb70 / visdrone_sot
  7. DTB70 regression діагноз (−0.067 AUC: ізолювати CE vs policy)
  8. Action audit: causal chain demonstration
     → Table F: per-sequence [Baseline AUC, SALT-RD AUC, Delta,
        Action fired frames, Changed bbox frames, Best action, Failure mode]
     → Обов'язкова таблиця — без неї run не інтерпретований
  9. FPS/GFLOPs вимірювання (Table G)
     → GFLOPs: ≥ 5% скорочення → CE policy в production

Після SceneStateClassifier multi-head retrain ───────────────────────────

  10. Calibration plots (AUPRC > 0.10 per head)
  11. Threshold calibration (θ_fc, θ_risk, θ_rec, θ_dyn) на val set
  12. SceneStateClassifier code + integration
```

---

## Failure modes

### RC-1 — Wrong candidate (car7, truck1)
YOLO знаходить кандидатів, але не ту ціль (distractor). REINIT локує трекер на distractor.  
**Fix:** crop_sim identity gate (TRACK A-2). Очікуваний crop_sim AUC > 0.65.

### RC-2 — No candidate (uav2/4/5/6/8)
UAV < 5px — YOLO не бачить взагалі. Features не мають значення без кандидатів.  
**Fix:** RT-DETRv2 fallback (TRACK B) + SAHI якщо потрібно.

---

## Candidate pipeline архітектура

```
SALT-RD policy → REINIT triggered
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
  score_map top-k   YOLO26m-VisDrone   RT-DETRv2-S
  (завжди, free)    (primary, fast)    (fallback: тільки якщо 0 кандидатів)
        │               │                │
        └───────────────┴────────────────┘
                        │
              merged candidate list
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
        identity gate         geometry gate
        crop_sim ≥ 0.30       розмір / позиція
              │                    │
              └─────────┬──────────┘
                        ▼
                  scorer v2.1 rank
                        ▼
                  best candidate → REINIT

Runtime logic (salt_runner.py):
  candidates = score_map_top_k(tracker)       # always, free
  candidates += yolo.detect(frame, hint_bbox) # always, fast
  if len(candidates) == 0:
      candidates += rtdetrv2.detect(frame)    # fallback only
  candidates = [c for c in candidates if c.crop_sim >= 0.30]
  best = scorer.rank(candidates) if candidates else None
```

**Per-dataset профіль:**

| Dataset | YOLO | RT-DETRv2 |
|---|---|---|
| UAV123 (hard UAV) | часто = 0 кандидатів | активний на uav2/4/5/6/8 |
| DTB70 (ground vehicles) | достатній | майже не запускається |
| VisDrone-SOT | основний | safety net |

### TRACK B fallbacks (якщо RT-DETRv2 recall недостатній)

**Fallback 1 — SAHI sliced inference:**
```python
# RTDETRv2Detector.detect() — optional sahi_mode
tile_size=256, overlap=0.2
# UAV < 5px = ~0.5% full frame, але ~15% tile → recall суттєво зростає
```

**Fallback 2 — YOLO26m-P2 fine-tune (якщо SAHI не допомагає):**
```bash
yolo train model=yolo26m-p2.yaml \
  data=VisDrone.yaml \
  pretrained=/Users/voleksiuk/projects/visdrone-yolo26m/best.pt \
  imgsz=640 epochs=50 batch=16 \
  cls=0.5 box=7.5 dfl=1.5
```
Метрика: **recall at oracle-REINIT frames** (не mAP).

---

## Feature schema v2 (8-dim) — OFFLINE USE ONLY

### Чому видалені старі features

| Feature | AUC | Вердикт |
|---|---|---|
| bbox_x | 0.434 | ШКОДИТЬ — sub-random |
| bbox_y | 0.460 | ШКОДИТЬ |
| detector_score | 0.500 | ШУМ — YOLO identity-blind |
| cosine_sim | 0.497 | ШКОДИТЬ — завжди 0.0 (BUG-27) |
| geometry_area_ratio | 0.494 | ШКОДИТЬ |
| bbox_w | 0.520 | Шум, граничний |
| frame_area_ratio | 0.552 | Слабкий |
| bbox_h | 0.580 | Маргінальний |
| score_map_score | 0.612 | ★ Єдиний корисний |

### Фінальна v2 схема

| IDX | Feature | Джерело |
|---|---|---|
| 0 | `score_map_score` | tracker score-map peak |
| 1 | `bbox_h` | candidate geometry |
| 2 | `frame_area_ratio` | candidate geometry |
| 3 | `bbox_w` | candidate geometry |
| 4 | `dist_from_last` | spatial prior (BUG-29 DONE) |
| 5 | `crop_sim` | MobileNetV3 identity (TRACK A-2) |
| 6 | `aspect_ratio_delta` | shape consistency |
| 7 | `size_delta_ratio` | size consistency |

**`FEATURE_DIM = 8`** — `feature_schema.py`, `candidate_events.to_feature_vector()`, `train_policy.py` (потрібно оновити).

---

## Scope claim — що входить в результати

| Dataset | Включено в claim | Виключено (вимірюємо, але не claim) |
|---|---|---|
| UAV123 | всі sequences КРІМ → | uav2/4/5/6/8 (flying drone < 5px, BUG-8), bike2 (blind hold-out) |
| DTB70 | всі sequences КРІМ → | Gull2, Sheep1, StreetBasketball1, Surfing04 (extreme domain, blind hold-out) |
| VisDrone-SOT | повний набір | — |

**Gate sequences для regression guard:**
- UAV123: `car7, truck1, building1`
- DTB70: повний без 4 excluded
- VisDrone-SOT: повний набір

---

## RC-5 — Identity gap в PolicyNet (відкладено)

**Проблема:** 28-dim EvidenceExtractor містить тільки confidence features — PolicyNet вирішує КОЛИ діяти, але не бачить ЧИ правильна ціль трекується.

**Правильний fix:** `template_drift` feature (+1 dim → 29-dim) — rolling cosine sim поточного crop до initial template через MobileNetV3 **в runtime**.

**Чому не зараз:**
- runtime inference overhead (~2–3ms/frame)
- TRACK A-2 crop_sim (offline) вирішує RC-1 дешевше
- якщо scorer v2.1 закриває regression gates → PolicyNet identity може не знадобитись

**Умова для розгляду:** TRACK A+B пройшли gates, full benchmark показав що PolicyNet все одно приймає неправильні REINIT рішення.

---

## Frozen decisions (незмінні архітектурні правила)

| Правило | Причина |
|---|---|
| No online Farneback flow в production | indices 22–27 zeroed by `zero_production_features()` |
| CE layer 3 only, kr=0.50 | `_CE_LOC={3}`, +0.006 AUC vs no-pruning |
| LSTM disabled permanently | OnlineLSTMMotionPredictor → zero AUC benefit |
| No runtime thresholds на APCE/p_fc | всі рішення тільки від learned action heads |
| TSA permanently deleted | archived, zero references дозволені в production |
| No center-freeze | Oracle: +0.000 hard AUC; Phase 7: −0.036 regression |
| Dynamic template update disabled | Oracle AUPRC занадто низький; car7 regression risk |

**Verified baseline:** 449 unit tests passing. SGLATrack checkpoint: `$UAV_WEIGHTS_ROOT/sglatrack/sglatrack_ep0297.pth.tar`.

---

## SceneStateClassifier — детальний план

### Чому зараз

Мета: проактивно визначати стан сцени і дозволяти cheap compute тільки коли **впевнені** що наступні 10 кадрів теж будуть стабільними.

### Таксономія станів (пріоритетний порядок)

| Стан | Умова | Дія системи |
|---|---|---|
| `FALSE_CONFIRMED` | впевнений, але не та ціль | block template update, abstain recovery |
| `AT_RISK` | зараз OK, але впаде за 5–10 кадрів | full compute, verify template |
| `RECOVERING` | загублений але recoverable | run recovery pipeline |
| `DYNAMIC` | OK, але сцена складна | full compute, expand search |
| `STABLE` | OK зараз і буде OK наступні ~10 кадрів | **cheap compute** |

`STABLE` = "confident by elimination" — коли жодна risk-голова не спрацювала.  
`STABLE_AND_CONFIDENT` — **не потрібен** (позитивний сигнал `failure_in_10` має AUPRC=0.013 → gate відкритий 99.7% часу, нічого не фільтрує).

### Метрики поточної multi-head моделі

| Голова | AUROC | AUPRC | Base rate | Рішення |
|---|---|---|---|---|
| `false_confirmed` | 0.885 | 0.361 | 5.5% | ✅ → FALSE_CONFIRMED gate |
| `imminent_failure_dynamic` | 0.889 | 0.263 | 4.2% | ✅ → AT_RISK gate (замінює failure_in_10) |
| `recoverable` | 0.894 | 0.042 | 0.6% | ⚠️ слабкий AUPRC — моніторити |
| `target_dynamic` | 0.730 | 0.123 | 5.6% | ✅ |
| `hard_dynamic_scene` | 0.604 | 0.214 | 11.9% | ✅ → DYNAMIC gate |
| `failure_in_10` | 0.827 | **0.013** | 0.26% | ❌ base rate занадто низький |
| `camera_dynamic` | **0.457** | 0.227 | 24.7% | ❌ гірше random — не додавати в модель |

### Кроки реалізації

**Крок 1 — Ретрейн multi-head (блокує все інше):**

Поточна проблема: `saltrd_v21_*` тренувались 1 епоху, тільки `recovery_action` голова. Потрібне повноцінне тренування всіх 5 голів з pos_weight:
`false_confirmed`, `imminent_failure_dynamic`, `recoverable`, `hard_dynamic_scene`, `target_dynamic`.

Перевірити `train_policy.py` — чи є multi-head loss чи тільки одна голова. Якщо одна — це першочинна проблема.

Не додавати: `camera_dynamic` (AUROC=0.457, гірше random), `failure_in_10` (AUPRC=0.013).

**Крок 2 — Calibration після ретрейну:**

Перевірити `AUPRC > 0.10` для кожної голови.  
Якщо `recoverable` залишиться < 0.10 AUPRC — виключити з combiner (лишити тільки як evidence signal).

**Крок 3 — SceneStateClassifier код (після ретрейну):**

```python
def classify_scene(probs: dict[str, float]) -> SceneState:
    if probs["false_confirmed"] > θ_fc:            # 0.885 AUROC — надійний
        return SceneState.FALSE_CONFIRMED
    if probs["imminent_failure_dynamic"] > θ_risk: # замінює failure_in_10
        return SceneState.AT_RISK
    if probs["recoverable"] > θ_rec:
        return SceneState.RECOVERING
    if probs["hard_dynamic_scene"] > θ_dyn:
        return SceneState.DYNAMIC
    return SceneState.STABLE                       # cheap compute
```

Пороги θ — **калібрувати на val set**, не hardcode.

**Важливо:** `SceneState` (новий, SALT-RD) ≠ `SceneClass` (старий 6-class enum в scheduler, відключений).  
`RECOVERING` в SceneState ≠ ніякий існуючий стан — це новий стан що тригерить recovery pipeline.

### Обмеження від FROZEN decisions
- Template update policy відключена (oracle AUPRC низький, car7 regression ризик) → `RECOVERING` стан **не повинен** тригерити template update без окремого gate
- Всі SceneState рішення мають іти через learned heads — без hardcoded APCE/threshold logic

---

## Відкриті ризики

| # | Ризик | Пріоритет | Файл |
|---|---|---|---|
| RISK-1 | MobileNetV3 на module scope в `build_candidate_dataset.py` — будь-який import завантажує модель | Низький | `build_candidate_dataset.py:47` |
| RISK-2 | `saltrd.checkpoint` → archive path — після scorer retrain оновити path у ВСІХ конфігах | Середній | `salt.yaml:30`, `salt_detector_rtdetr.yaml` |
| RISK-3 | `candidate_scorer_checkpoint` в per-dataset конфігах — `salt_runner.py` його ніколи не читає | Низький | `saltrd_uav123/dtb70/visdrone_sot.yaml:29` |
| RISK-4 | `tsa:` блок залишився в двох ablation configs | Низький | `salt_detector_yolo26m.yaml:26`, `salt_detector_leafyolo.yaml:35` |
| RISK-5 | BLOCKER (з ARCHITECT review): `update_with_action()` в sglatrack.py завжди викликає `self.update(frame)` → кожна action variant = no-op | Критичний при інтеграції | `sglatrack.py` |
| RISK-6 | BLOCKER (з ARCHITECT review): `from_config()` instantiates `SALTRDController()` без model path → завжди `_safe_noop` | Критичний при інтеграції | `salt_runner.py` |

---

## Лог змін

### Phase 0 — 2026-05-22 ✓ complete
- `saltrd.reinit_threshold: 0.95` в `configs/prod/salt.yaml`
- `policy_reinit_v1/` + `policy_with_candidate_scorer/` → `saltr/archive/checkpoints/`
- BUG-R1: `reinit_confidence_threshold` прибраний з `tracker:` блоку (TypeError)
- BUG-R2: `salt_detector_rtdetr.yaml` reinit_threshold → 0.95
- BUG-R3: `ortrack_baseline.yaml` — `appearance_memory/motion_predictor: enabled: false`

### TRACK B — BUG-30 — 2026-05-22 ✓ complete
- `salt_detector_rtdetr.yaml`: видалені `tsa:`, `ce_loc`, `ce_keep_ratio_by_state`
- `saltrd.enabled: false` (pure detector ablation)

### TRACK A-2 — 2026-05-22 ✓ code done
- `feature_schema.py`: v2 candidate schema (8-dim)
- `candidate_events.py`: нові поля, `to_feature_vector()`
- `build_candidate_dataset.py`: MobileNetV3 `_crop_embed`, `_compute_crop_sim`, rolling template

### TRACK C — ORTrack-DeiT — 2026-05-22 ✓ code done
- `src/uav_tracker/trackers/transformer/ortrack.py` (pretrained=False patch)
- `configs/prod/ortrack_baseline.yaml`
- `__init__.py` реєстрація
- Weights symlink → `конференція/ORTrack-D-DeiT.pth.tar`

---

## Architect Review — 16:12 2026-05-22

### Changes reviewed
- `saltr/src/salt_r/train_policy.py` — CANDIDATE_FEATURE_DIM + CandidateEventDataset
- `saltr/src/salt_r/controller.py` — _build_candidate_features() rewritten to 8-dim
- `configs/prod/saltrd_uav123/dtb70/visdrone_sot.yaml` — reinit_threshold fix
- `saltr/src/salt_r/build_candidate_dataset.py` — aspect/size collection logic
- `docs/archive/` — SUPER_PLAN/ARCHITECT/RESULTS/FROZEN/bugs moved

### Findings

**✅ FIXED — `CANDIDATE_FEATURE_DIM = 8`** (було 10)
File: `train_policy.py:71`
`CandidateEventDataset` тепер читає правильні поля: score_map_score, bbox_h/w raw pixels, frame_area_ratio, dist_from_last, crop_sim, aspect_ratio_delta, size_delta_ratio. Порядок відповідає `to_feature_vector()`. ✅

**✅ FIXED — `reinit_threshold: 0.95`** у всіх трьох per-dataset конфігах. ✅

**✅ FIXED — `controller.py` оновлено до 8-dim** без MobileNetV3 в runtime. crop_sim = 0.0. ✅

---

**⚠️ WARNING — `aspect_ratio_delta` feature 6: різна формула в training vs runtime**
File: `controller.py:138` vs `build_candidate_dataset.py:259`

Training: `abs(cand_w/cand_h − tmpl_w/tmpl_h)` — дельта від aspect ratio **template** (GT bbox)
Runtime:  `abs(bw/bh − 1.0)` — дельта від **квадрату** (1:1)

Приклад: car (tmpl AR = 2.0), correct candidate (AR = 2.0):
- Training → 0.0 (ідеальний збіг з template) ✅
- Runtime  → 1.0 (виглядає як поганий кандидат) ❌

Distractor person (AR = 0.5):
- Training → 1.5 (great discriminator) ✅
- Runtime  → 0.5 (слабкий сигнал) ❌

**Action:** замінити в `controller.py:138`:
```python
# Замість:  abs(aspect - 1.0)
# Треба:    abs(bw/bh - tracker_w/tracker_h)
# де tracker_w/tracker_h — aspect ratio поточного tracker bbox

tracker_bbox = candidates[0].bbox  # або з evidence.frame.bbox
tracker_ar = tracker_w / max(tracker_h, 1e-6)
aspect_ratio_delta = abs(bw / max(bh, 1e-6) - tracker_ar)
```
Це відтворює формулу тренування (коли трекер on-target, tracker AR ≈ template AR).

---

**ℹ️ IDEA — crop_sim = 0.0 завжди в runtime → scorer partial blind**
File: `controller.py:141` (hardcoded 0.0)

На тренуванні ~X% прикладів мають `crop_sim > 0` (MobileNetV3 offline).
На інференсі всі кандидати отримують `crop_sim = 0.0`.
Scorer навчився "вище crop_sim → кращий кандидат", але в runtime не може цим скористатись.

**Action:** при тренуванні scorer додати **crop_sim dropout** — з імовірністю ~0.3 замінювати crop_sim на 0.0:
```python
# В CandidateEventDataset.__getitem__:
if self.training and random.random() < 0.3:
    cand_feat[5] = 0.0  # dropout crop_sim → scorer вчиться без нього
```
Це зробить scorer robustний до відсутності crop_sim в runtime, без втрати сигналу при collection.

---

**ℹ️ IDEA — `bbox_h` і `bbox_w` в raw pixels (не нормовані)**
File: `train_policy.py:361`, `controller.py:126`

UAV123 (typical frame: 1280×720) vs VisDrone (2000×1500+) — той самий цільовий об'єкт дасть різні bbox_h/bbox_w значення в пікселях. Scorer тренується per-dataset, тому в межах одного dataset це OK. Але якщо в майбутньому знадобиться cross-dataset scorer — потрібна нормалізація.

**Action (низький пріоритет):** розглянути нормалізацію `bbox_h / frame_h` і `bbox_w / frame_w` при переході до cross-dataset scorer. Зараз не блокує.


---

## Architect Review — 16:23 2026-05-22

### Changes reviewed
- `saltr/src/salt_r/controller.py` — aspect_ratio_delta fix + tracker_bbox param
- `saltr/src/salt_r/train_policy.py` — crop_sim_dropout_p added to CandidateEventDataset

### Findings

**✅ FIXED — aspect_ratio_delta тепер відносно tracker AR**
File: `controller.py:138`
`abs(cand_ar - tkr_ar)` замість `abs(cand_ar - 1.0)`. Формула узгоджена з тренуванням. ✅

**✅ FIXED — crop_sim dropout (p=0.3) реалізований**
File: `train_policy.py:391`
Dropout застосовується тільки до training dataloader (немає val split для candidate scorer), тому валідація не постраждає. ✅

---

**⚠️ WARNING — `salt_ostrack.yaml` пропущений: reinit_threshold=0.65**
File: `configs/prod/salt_ostrack.yaml:29`
Всі per-dataset configs виправлені, але `salt_ostrack.yaml` залишився з `0.65`.
Якщо запустити benchmark з цим конфігом — over-firing повернеться.
Action: `reinit_threshold: 0.65` → `0.95` в `configs/prod/salt_ostrack.yaml`.

---

**ℹ️ IDEA — BBox attribute access через `or` не захищає від w=0.0**
File: `controller.py:130–134`
```python
bw = float(getattr(bbox, 'w', None) or (bbox[2] if ...))
```
Якщо `bbox.w = 0.0` (degenerate box на краю кадру) → `getattr` повертає `0.0` → falsy → fallback до `bbox[2]` → теж `0.0`. Результат: `cand_area=0`, `frame_area_ratio=0`, `cand_ar=0`, `aspect_ratio_delta=tkr_ar` (максимальний penalty для degenerate candidate).
Поведінка корректна (degenerate кандидат отримає поганий score), але неявна.
Action (low priority): замінити `or`-chain на явний `getattr(bbox, 'w', bbox[2] if ... else 1.0)` або додати коментар що нульова ширина — OK, scorer відфільтрує.

