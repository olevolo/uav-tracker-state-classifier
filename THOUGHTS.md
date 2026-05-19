# THOUGHTS — Зовнішні огляди та стратегічні коментарі

**Дата:** 2026-05-19  
**Зміст:** Staff-level коментарі, архітектурні рекомендації та аналіз ключових статей.  
**Пов'язано:** `HANDOFF_NEXT.md`, `ANALYSIS.md`, `papers/`

---

## Поточне рішення — SALT-RD

Станом на 2026-05-19 центральна ідея проекту змінена з "SALT-R як reliability head" на **SALT-RD: proactive tracking-risk dynamicity + false-confirmed + recovery**.

Це важлива зміна. Якщо залишити тільки `P(false_confirmed)`, робота виглядатиме як вузький failure detector. Якщо поставити в центр **нейромережевий визначник динамічності сцени як проактивний tracking-risk модуль**, робота стає ширшою і deployment-релевантною: модель не просто каже "трекер може помилятися", а наперед оцінює tracking-risk, false-confirmed drift, recovery readiness, скільки compute витрачати, чи можна оновлювати template, і чи варто взагалі довіряти bbox.

**Правильна назва задачі:** proactive tracking-risk dynamicity, not generic scene classification.

Тобто label `dynamic` не має походити з ручного класу сцени або старого LSTM branch. Він має походити з:

- GT target velocity/acceleration/scale change;
- camera ego-motion або global flow magnitude;
- residual target motion after ego-motion compensation;
- score-map ambiguity under motion;
- future IoU degradation;
- oracle comparison між full compute і cheap/pruned compute режимами.

**Два речення статті:**
> We introduce SALT-RD, a lightweight proactive tracking-risk dynamicity controller for real-time UAV single-object tracking that predicts false-confirmed drift, near-future failure, recovery readiness, and the need for full computation from tracker-internal telemetry. Unlike adaptive-depth or token-pruning trackers that optimize average computation, SALT-RD is trained with GT/teacher-derived tracking-risk labels and controls template updates, re-acquisition, and compute allocation under calibrated deployment metrics.

---

## Детальний план робіт — що реально робити

### Фаза 0 — заморозити стару систему і відкрити telemetry

Мета: не дати старому entropy/v2/SALT v3 змішатися з новою наукою.

1. `src/uav_tracker/` вважати frozen baseline.
2. Дозволені зміни у frozen code тільки для:
   - config gates;
   - `TrackState` telemetry;
   - score-map/motion/flow/appearance stats.
3. CE, DYNAMIC і VelocityDrift не видаляти, а config-gate-ити для ablations.
4. Створити `saltr/src/salt_r/` як єдине місце для нової моделі.

### Фаза 1 — dataset без leakage

Найважливіший deliverable — не модель, а правильний NPZ.

Labels:

- `correct`: IoU >= 0.5;
- `false_confirmed`: IoU < 0.2 and tracker confidence high;
- `failure_in_5`: tracker currently correct, but near-future IoU collapses;
- `recoverable`: lost now, can become correct again soon;
- `target_dynamic`: high target velocity/acceleration/scale change;
- `camera_dynamic`: high ego-motion/global flow;
- `hard_dynamic_scene`: dynamicity plus ambiguity/future tracking risk;
- `needs_full_compute`: oracle label from full-vs-cheap tracker replay.

Головне правило: labels не можуть бути похідними від `_decide_state()`, `TargetState`, `scene_class`, APCE thresholds як teacher, або старої LSTM dynamic гілки.

### Фаза 2 — SALT-RD v0 model

Почати з малого temporal model:

- input: 24-32 scalar features;
- model: GRU або TCN, hidden 64, 2 layers;
- heads: false-confirmed, failure-in-5, recoverable, target-dynamic, camera-dynamic, hard-dynamic-scene, needs-full-compute;
- losses: weighted/focal BCE, sequence-balanced sampler;
- metrics: AUROC/AUPRC/ECE/Brier + bootstrap CI.

Не починати з CNN score-map encoder або Mamba head. Це можна додати тільки якщо scalar telemetry дає signal, але не вистачає capacity.

### Фаза 3 — offline policy replay

Перед runtime інтеграцією policy має бути перевірена offline:

- чи зменшує wrong re-init;
- чи блокує template corruption;
- чи покращує risk-coverage;
- чи дає AUC-vs-GFLOPs Pareto;
- чи не шкодить DTB70 hard cases.

Runtime інтеграція без replay — це ризик повторити SALT v3: локально красиво, на повному dataset майже нуль або regression.

### Фаза 4 — teacher features

Якщо Phase 1-3 дають GO:

- CoTracker3/PTDT-style point consistency для dynamic/recoverability labels;
- DINO/SAM appearance/foreground teachers offline;
- candidate-level SALT-Match dataset для recovery accept/reject;
- LoRAT/domain adaptation тільки як fallback або паралельна backbone improvement робота.

---

## Нові конкурентні статті для SALT-RD positioning

Ці роботи важливо згадати, бо вони закривають частину простору "dynamic compute" і "dynamic scenes". Вони не вбивають SALT-RD, але змушують дуже чітко формулювати novelty.

| Paper | Що важливо | Висновок для нас |
|---|---|---|
| **UTPTrack, CVPR 2026** | Joint token pruning для search region, static template, dynamic template; 65%+ pruning with near-baseline accuracy | Не продавати CE/token pruning як внесок. SALT-RD може бути policy, що вирішує коли pruning безпечний |
| **ABTrack, Pattern Recognition 2025 / arXiv 2024** | Bypass Decision Module для adaptive ViT block bypass | "Dynamic compute" вже є. Наша dynamicity має бути tracking-risk supervised, не просто block-skip controller |
| **BDTrack, ESWA 2025** | UAV motion blur + dynamic early exiting | Прямий конкурент до фрази "dynamic UAV scenes"; треба показати, що SALT-RD ловить false-confirmed/recovery risk, а не тільки blur/easy-vs-hard |
| **UncL-STARK, arXiv 2026** | Heatmap uncertainty керує encoder/decoder depth, 12% GFLOPs reduction при ~0.2% accuracy loss | Сильний baseline для uncertainty-guided compute. Наша різниця: identity-level false-confirmed і recovery safety |
| **LGTrack, arXiv 2026** | Dynamic layer selection + occlusion robustness for UAV tracking | Конкурент як lightweight UAV tracker; ми маємо порівнювати як wrapper/controller або брати як майбутній backbone |
| **SMTrack, arXiv 2026** | State-aware Mamba for temporal modeling in dynamic scenarios | Конкурент до temporal modeling. SALT-RD не має claim "ми перші моделюємо temporal dynamics"; claim — calibrated control policy |
| **TrackingMiM, arXiv 2025** | Mamba-in-Mamba UAV tracking, high speed/precision claims | Конкурент до backbone/FPS. Не змагатися напряму без відтворення; використовувати як related work |
| **AVTrack / Aba-ViTrack** | Dynamic layer activation / background-aware token termination | Related work для adaptive compute in tracking | Позиціонувати SALT-RD як risk-aware controller over tracker actions, not only internal layer skipping |

**Staff verdict:** якщо стаття звучить як "ми зробили dynamic neural scene classifier", вона слабка. Якщо вона звучить як "ми вчимо tracking-risk dynamicity labels і показуємо calibrated control of compute/recovery/template safety", вона має шанс.

---

## Ключові статті (papers/)

### MSTFT 2026 — Mamba-Based Spatio-Temporal Fusion for Small Object Tracking in UAV Videos
*Electronics 2026, 15, 256. Sun K., Zhang H., Chen H.*

**Результати:** UAV123 AUC=79.4%, UAV123@10fps=76.5%, UAV20L=75.8%, **45 FPS**

**Три внески:**
1. **Bidirectional Spatio-Temporal Mamba (BS-Mamba)** — горизонтальне + вертикальне двонаправлене сканування для малих цілей
2. **Dynamic Template Fusion з Adaptive Attention** — **threefold safety verification: response peak + temporal consistency + motion stability** → це прямо наш feature set для SALT-RD
3. **Small-Target-Aware Context Prediction Head** — Gaussian-weighted prior для локалізації малих цілей

**Що взяти для SALT-RD:**
- Triple verification mechanism = підтверджує наш feature set (response_peak, temporal_consistency, motion_stability)
- Dynamic template fusion strategy = reference для SALT-Match (що оновлювати, коли)
- UAV123 AUC=79.4% — це нова SOTA для порівняння (SGLATrack=73.7%)

**Ризик для нас:** MSTFT закриває Mamba як backbone. Якщо ми будемо говорити "замінити backbone" — це вже є. Наша ніша = reliability head, не backbone.

---

### MATA 2026 — Architecture and Evaluation Protocol for Transformer-Based Visual Object Tracking in UAV Applications
*arXiv:2603.03904v2, Borne A. et al. (French-German Research Institute)*

**Архітектура:** три блоки що працюють асинхронно:
- **Block A** (10Hz): ViT tracker (MixFormerV2 або OSTrack) → bbox + score
- **Block B** (30Hz): Ego-motion compensation (sparse optical flow → homography)
- **Block C** (30Hz): EKF estimation — об'єднує A і B, "Check bbox validity"

**Ключові внески:**

**1. NT2F метрика (Normalized Time to Failure)**
> "Measures the duration for which a tracker can successfully follow an object before a tracking failure occurs."
- Це формалізована версія нашої failure prediction задачі
- Відрізняється від AUC: не "середня точність", а "час до першого провалу"
- Важлива для UAV deployment evaluation де re-init неможливий

**2. Explainable VOT — PMF confidence score**
> MixFormerV2 дає PMF для bbox координат → AUC під піком = confidence
- Це аналог нашого APCE але на рівні bbox (не score map)
- "Check measure validity" = MATA вирішує споріднену задачу до SALT-RD: P(tracker_reliable)
- **Ключова різниця:** MATA валідує перед EKF; SALT-RD вчить dynamicity/reliability policy для recovery, template update і compute allocation

**3. EOP (Embedded-Oriented Protocol)**
> Hardware-independent evaluation з асинхронними модулями → Jetson AGX Orin
- 6.4× менша розбіжність між PC і embedded для success rate vs LTP protocol
- NT2F: 4.7× менша розбіжність

**Що взяти для SALT-RD:**
- **NT2F metric** — реалізувати у `saltr/src/salt_r/eval.py` поруч з AUROC/ECE
- **Ego-motion residual** (Block B) = наш `ego_motion_residual` feature (Farneback вже є)
- **"Check measure validity"** = архітектурний прецедент для нашого P(correct) head
- **PMF confidence** = краще ніж APCE для деяких архітектур — reference для майбутнього

**Формула NT2F:**
```
NT2F = (1/N) * Σ_i (t_failure_i - t_init_i) / sequence_length_i
де t_failure = перший кадр де IoU < threshold (зазвичай 0.2 або 0.5)
```

---

## Коментар 1 — Staff ML/CV Architect: Codex (перший аналіз)

> SALT v3 у поточній формі не варто далі "докручувати" як rule-based надбудову. Вона вже дала майже всю локальну користь, яку могла: recovery рятує окремі кейси типу uav2, але на повних датасетах не змінює стелю базового трекера, а на DTB70 активно шкодить.

**Висновок:** Головний напрямок — не APCE rules / CE tweaks / ще один guard, а **зміна джерела якості: domain-adapted tracker + навчений failure/recovery модуль з IoU labels.**

**Що не працює (підтверджено ablation):**
- APCE thresholds — rule-based, не generalized
- CE pruning — не стратегічна оптимізація (<1% GFLOPs, regression risk)
- Supervised TSA head (92.5%) — навчений повторювати правила, не GT
- Online MLP adaptation — circular (teacher = _decide_state)
- LSTM motion predictor — 97% false DYNAMIC, warmup не конвергує
- Template EMA updates — drift → regression

**Пріоритизація (P0→P5):**
- P0: IoU-supervised failure predictor (GT IoU labels, не APCE rules)
- P1: Domain fine-tune / tracker replacement (єдиний шлях підняти AUC)
- P2: ROI/hint-aware recovery (дешевий win)
- P3: Conservative recovery policy by domain (DTB70 fix)
- P4: Multi-template re-acquisition
- P5: CE pruning (тільки якщо потрібен >15-20% real speedup)

**Назва правильного підходу:**
> Перетворити SALT з rule-based state machine на **supervised failure-aware tracker wrapper**, паралельно адаптуючи tracker під UAV/aerial domain.

---

## Коментар 2 — Novelty Positioning: "не UAV tracker"

> Станом на 2026, простір UAV trackers дуже щільний: SGLATrack CVPR 2025, ORTrack, TATrack, Mamba/MSTFT, UTPTrack CVPR 2026.

**Що вже закрито:**
- SGLATrack CVPR 2025: layer-adaptive ViT для UAV tracking (efficiency)
- UTPTrack CVPR 2026: joint token pruning 65% (CE/CTEM мертві як внесок)
- MATA 2026: systems angle — transformer + EKF + optical flow + embedded
- ORTrack 2025: occlusion-robust via masking + distillation
- TATrack 2025: spatial-temporal prompt + bypass redundant layers
- OOTU 2025: uncertainty regression для bbox confidence — але це не false-confirmed

**Наша ніша:**
> Real-time aerial SOT with **calibrated failure probability, false-confirmed detection, and risk-aware class-agnostic recovery** on edge hardware.

**Thesis:** "Calibrated Failure-Aware Re-Acquisition for Real-Time UAV Single Object Tracking"

**Чому це відрізняється від OOTU:**
- OOTU: P(bbox_accurate | frame) — де саме об'єкт?
- Наш: P(tracker_on_wrong_object | APCE_high) — чи той об'єкт взагалі?

**Ключова емпірична точка:**
- `uav0000164`: AUC=0.174 при 99% CONFIRMED
- Жоден SOTA не вимірює і не вирішує false-confirmed failure mode

**Three contributions:**
1. IoU-supervised false-confirmed detector (AUROC baseline ≈ random — APCE threshold безглуздий тут)
2. Risk-aware recovery with abstention (learned accept/reject)
3. Calibrated failure probability (ECE < 0.10 vs OOTU point-uncertainty)

**Що НЕ буде достатньою новизною:**
- APCE thresholds, spatial hint для YOLO сам по собі
- Fine-tuning SGLATrack без нового training/eval protocol
- CE pruning (UTPTrack/TATrack вже сильні)
- Supervised TSA head якщо навчений повторювати правила
- SAM/Grounding-DINO recovery якщо просто заміна детектора

---

## Коментар 3 — Архітектурна стратегія: "Не тренуємо новий трекер"

**Базова ідея:**
> SGLATrack лишається основним трекером. Ми збираємо з нього внутрішні сигнали: APCE, score map, peak sharpness, bbox motion, scale, cosine до template, optical-flow consistency. Offline, де є GT/teacher labels, рахуємо справжній IoU, target/camera dynamicity і full-vs-cheap oracle regret. Тренуємо мережу передбачати **P(false_confirmed), P(failure_in_5), P(recoverable), P(target_dynamic), P(camera_dynamic), P(hard_dynamic_scene), P(needs_full_compute)**.

**Стекова архітектура:**

| Роль | Модель | Режим |
|------|--------|-------|
| Main tracker | SGLATrack-DeiT (frozen) | real-time |
| Dynamicity/reliability head | SALT-RD (наш GRU/TCN) | real-time, <1мс |
| Appearance teacher | DINOv3 | offline training only |
| Video/mask teacher | SAM2 / MobileSAM | offline pseudo-labels |
| Point consistency teacher | CoTracker3 | offline flow labels |
| Recovery candidates | YOLO26m / RT-DETR | real-time, LOST only |

**Teacher-моделі (DINO/SAM2/CoTracker3) не в real-time pipeline** — тільки для генерації labels під час тренування → distillation у легку голову.

**MVP порядок:**
1. SALT-RD v0: scalar telemetry тільки, frozen SGLATrack
2. SALT-RD v1: + score-map CNN features
3. SALT-Match v1: candidate accept/reject для recovery
4. SALT-Distill: + DINOv3/SAM2/CoTracker3 як offline teachers

---

## Коментар 4 — Сигнали та метрики (з посиланнями на paper)

**Додаткові сигнали (рекомендовані):**

**Score map geometry:**
- top1/top2 peak gap (ambiguity) — MSTFT 2026
- peak width (FWHM), secondary peaks count, peak distance
- temporal response consistency (deviation від середнього 5-10 кадрів)

**Motion:**
- ego-motion residual після компенсації руху камери — MATA 2026
- acceleration (не тільки velocity)
- dist_to_search_border (search risk)

**Point track consistency (offline teacher: CoTracker3):**
- points_inside_ratio, point_residual_med, fwd_bwd_error — PTDT 2026

**Appearance (offline teacher: DINOv3):**
- foreground-to-template similarity, background confusion
- nearest-neighbor entropy, target/background margin

**BBox uncertainty (OOTU-style):**
- sigma_x, sigma_y (variance over top-K peak locations)

**Recovery candidate:**
- n_candidates, top1/top2 gap, hint_distance, size_prior, detector_agreement

**Метрики beyond AUC:**

| Метрика | Навіщо |
|---------|--------|
| Failure AUROC / AUPRC | Primary для imbalanced labels |
| **False-confirmed recall@5%FPR** | Signature metric нашої роботи |
| ECE / Brier / NLL | Calibration vs OOTU |
| Wrong re-init rate | DTB70 regression source |
| Recovery precision/recall | Корисність recovery |
| **Abstention gain** | Δ AUC коли відмовляємось від bad update |
| NT2F / time-to-failure | MATA metric для порівняння |
| Template corruption rate | % updates при IoU < 0.5 |
| Risk-coverage curve | AUC vs % прийнятих рішень |
| AUC by target size | small / normal / large |

---

## Коментар 5 — Staff Review: Критичні ризики перед ML

> Repo зараз у перехідному стані між трьома системами: старий entropy/v2 scene-router, поточний SALT v3, і запланований SALT-RD. Перед серйозним ML треба стабілізувати межі системи.

**6 критичних ризиків:**

**1. Label leakage / self-teaching (СМЕРТЕЛЬНИЙ):**
> Старий train_tsa_classifier.py: scene labels → TargetState → APCE rules → circular. Для статті це смертельно: модель "гарно валідована" на власних правилах.
Рішення: окремий `saltr/src/salt_r/collect_features.py`, тільки GT/teacher-derived labels.

**2. False-confirmed рідкісний клас (1-3%):**
> З наївним BCELoss модель буде казати "0" завжди: 97% accuracy при AUROC=0.50.
Рішення: AUPRC primary, weighted BCE pos_weight≈40, focal loss, hard-negative oversampling.

**3. Score map features відсутні:**
> generate_ml_labels.py пише лише APCE/PSR/entropy. Треба top1/top2 gap, peak width.
Рішення: розширити TrackState.score_map_stats.

**4. Split некоректний:**
> Поточний: алфавітний, тільки UAV123.
Правильно: stratified group split по послідовностях, across all 3 datasets.

**5. Worktree dirty — три системи:**
> Будь-який агент може наступити на стару архітектуру.
Рішення: один migration commit, saltv3/ окрема папка.

**6. Reproducibility fragile:**
> SGLATrack path hardcoded, torch pin 2.1.0 vs env 2.11.0.

**Verdict:**
> План хороший, але repo треба "розчистити від старої науки" перед тренуванням. Інакше є ризик отримати ще одну красиву, але самореферентну модель, яка не вирішує головну проблему: впевнений трекер на неправильному об'єкті.

---

## Ключові інсайти для статті

1. **False-confirmed = унікальна наукова точка.** uav0000164 = 99% CONFIRMED, AUC=0.174. APCE вимірює якість піку, не identity. Ніхто в літературі це явно не вимірює і не вирішує.

2. **AUPRC baseline ≈ random для false_confirmed.** APCE threshold не може детектувати false-confirmed (APCE HIGH = умова входу). LSTM детектує через patterns у часі → це нова ML задача.

3. **Abstention gain > recovery gain.** На DTB70 система що "відмовляється" від ризикованих recovery дає кращий AUC, ніж та що завжди re-initіалізується.

4. **ECE ≠ Uncertainty.** OOTU дає point uncertainty, ми даємо calibrated probability. Різна задача, різна метрика, різна наукова позиція.

---

## Аналіз нових статей (papers/)

### CoTracker3 (Meta AI, arXiv:2410.11831, Oct 2024)

**Semi-supervised point tracking — 1000× менше даних ніж BootsTAPIR, SOTA на TAP-Vid.**

Output: `x_t, y_t, V_t ∈ [0,1] (visibility), C_t ∈ [0,1] (confidence)` per tracked point.  
Два режими: **online** (sliding window, real-time) та **offline** (full video, точніший).

**Для SALT-RD (Phase 4 — offline teacher):**
```python
# 9 query points on frame 0 inside bbox
tracks, vis, conf = cotracker3_offline(video, query_points)
points_inside_ratio[t] = sum(is_inside(tracks[t], bbox_pred[t])) / 9
point_visibility_mean[t] = mean(vis[t])
fwd_bwd_error[t] = ||forward_track - backward_track|| / diag
```
Confidence C_t = чи точка в межах GT → **це і є наш recoverable label для точок!**

---

### LoRAT (arXiv:2403.05231, Jul 2024)

**PEFT для ViT trackers. LoRA fine-tuning ViT-g до 25.8GB, LaSOT +3.9%, 3.2× faster.**

Два унікальні виклики vs NLP:
1. Unshared position embeddings (template ≠ search) → LoRA ламає структуру
2. CNN bbox head з inductive biases → LoRA не сходиться

**Рішення:** decoupled embeddings (shared absolute + independent type) + MLP head.

**Для SALT-RD Phase 6:**
- LoRAT = правильний спосіб domain-adapt SGLATrack на UAV/aerial
- Inference latency = 0 (LoRA weights merged)
- Можна тренувати на MPS без data-center GPU
- **Потрібна модифікація SGLATrack:** decoupled positional embeddings (з LoRAT repo)

---

### PTDT 2026 (Neurocomputing 678, 2026)

**Point Tracking-Guided Dynamic Tokens. UAV123 AUC=68.8%, UAV20L=72.1%.**

**3-condition template update gate (прямий reference для SALT-Match):**
```python
should_update = (score > σ) AND (pos_t in bbox) AND (point_tracker_success)
```

Mask-guided token pruning через NanoSAM (lightweight, init-only, one-time).  
70% token retention на шарах 3, 6, 9.

**Для SALT-RD:**
- 3-condition gate = архітектурний blueprint SALT-Match v1
- points_inside_ratio + fwd_bwd_error = PTDT-inspired features (Phase 4)
- Token pruning з foreground mask = reference для майбутнього CE refinement

---

### OOTU 2025 (Neurocomputing 648, 2025)

**End-to-end tracker + Uncertainty Head. TrackingNet AUC 83.5% vs 83.1%.**

KL loss для bbox uncertainty:
```
L_loc = (x_g - x_e)² / 2σ² + ½ log(σ²)  # self-calibrating
```
σ малий коли впевнений і правильний; σ великий коли невпевнений.

**Ключова різниця від нас:**
- OOTU: `σ²` = uncertainty про ТО ДЕ bbox (localization)
- SALT-RD: `P(false_confirmed)` = uncertainty про ТЕ ЧИЙ це об'єкт (identity), плюс `P(hard_dynamic_scene)` і `P(needs_full_compute)`

**OOTU σ може бути МАЛИМ при false_confirmed** (трекер локалізує дистрактора точно!) — це наш головний scientific argument проти OOTU як вирішення нашої проблеми.

**Для статті:** показати що σ_OOTU і P(false_confirmed)_SALT-RD некорельовані на uav0000164 → різні failure modes, різні metrics.

---

## Кориговані плани (Codex Staff — друге ревью)

### Що змінити у HANDOFF_NEXT.md:

**1. Не переносити в saltv3/ git mv — заморозити через policy**
```
# saltv3/README.md: "frozen — do not modify"
# saltr/ — нова папка поряд
```

**2. Config-gate замість видалення CE/DYNAMIC/VelocityDrift**
```yaml
# salt.yaml:
enable_ce: false          # для ablation baseline
enable_dynamic: false
enable_velocity_drift: false
```

**3. Label bug: units clarification**
```python
# Правильно: apce в TrackState вже нормований 0-256
# false_confirmed = iou < 0.2 AND apce > 100  (raw APCE, не /256)
# або якщо нормований: apce_norm > 100/256 = 0.39
# ТРЕБА явно писати які одиниці!
```

**4. NPZ schema для reproducibility**
```python
npz.save({
    "feature_names": ["apce_norm", "psr_norm", ...],
    "feature_units": ["[0,1]", "[0,1]", ...],
    "sequence_name": seq_names,
    "dataset": datasets,
    "frame_idx": frame_indices,
    "split": splits,  # train/val/diagnostic
    "tracker_version": "sglatrack_ep0297",
    "config_hash": hash(salt_config)
})
```

**5. Hard negatives = ОКРЕМИЙ diagnostic suite, не mixed з train/val**
```
train split    (80 seqs UAV123 + 25 VisDrone + 50 DTB70)
val split      (43 seqs UAV123 + 10 VisDrone + 20 DTB70)
diagnostic     (uav0000164, bike2, Gull2, Sheep1, StreetBasketball1)
LODO eval:     train UAV123+VisDrone → test DTB70 (generalization)
```

**6. Recovery labels insufficient → candidate-level dataset**
```python
# Для SALT-Match треба окремий NPZ:
# run_detector_on_all_sequences() → candidate bboxes
# label: IoU(candidate, gt) > 0.5 = same_object
# features: cosine_sim, size_ratio, hint_distance, n_candidates
```

**7. Bootstrap CI обов'язково**
```python
# Для статті: mean ± 95% CI по sequence resampling
from sklearn.utils import resample
for _ in range(1000):
    sample = resample(seq_metrics, replace=True)
    ci_samples.append(mean(sample))
CI = (np.percentile(ci_samples, 2.5), np.percentile(ci_samples, 97.5))
```

**8. Negative result policy**
```
Якщо false_confirmed AUROC < 0.60 після fair attempt:
→ НЕ докручувати features
→ Перейти до LoRAT domain adaptation (Phase 6 moved to P0)
→ Задокументувати: "scalar telemetry insufficient for identity-level failure detection"
→ Цей negative result + LoRAT = окрема стаття
```

---

## Papers Analysis — SALT-RD Competitive Positioning (2026-05-19)

### Positioning Summary

| SALT-RD claim | Closest competitor | Novel? | Key differentiator |
|---|---|---|---|
| `false_confirmed` detection (high APCE, wrong identity) | None — unaddressed in all papers | **Yes, fully novel** | uav0000164: 99% CONFIRMED, AUC=0.174. σ_OOTU and UncL-STARK confidence are HIGH here — they cannot detect it |
| GT-supervised multi-label tracking-risk policy | All use task-loss or heuristic supervision | **Yes, novel training regime** | Only work using GT IoU-derived labels for false_confirmed/failure_in_5/recoverable |
| `failure_in_5` proactive failure prediction | MATA NT2F (retrospective measure) | **Yes** — MATA measures duration, SALT-RD predicts 5 frames ahead | Frame as: "MATA quantifies failure duration; SALT-RD predicts failure proactively" |
| Calibrated failure probability (ECE/AUPRC/Brier) | OOTU (localization uncertainty KL loss) | **Yes** — OOTU calibrates localization σ², SALT-RD calibrates identity failure P | Must show: σ_OOTU negatively correlates with P(false_confirmed) on uav0000164 |
| `needs_full_compute` oracle-supervised compute | UTPTrack CVPR26, ABTrack PR25, BDTrack ESWA25, UncL-STARK 26 | **Partially** — oracle from full-vs-cheap replay is new | Reframe: "others decide by visual difficulty; SALT-RD by oracle regret — catches visually-easy false-confirmed cases" |
| Wrapper over frozen tracker | All adaptive-compute papers modify backbone | **Yes** | Zero changes to SGLATrack; wrappable around any ViT tracker |
| `hard_dynamic_scene` from GT-derived labels | BDTrack, SMTrack — scene difficulty from backbone | **Partially** | SALT-RD dynamicity is tracking-outcome-derived, not appearance-category derived |
| NT2F evaluation metric | MATA 2026 (introduced NT2F) | **Not novel as metric** | Claim: "first to evaluate proactive failure predictor using NT2F" |

### Features to add for v1 (from papers)

From MSTFT: `response_peak_ratio` (peak/map_mean), `temporal_response_consistency` (APCE deviation from 5-frame mean), `motion_stability` (bbox centroid delta / search_size), `scale_consistency` (area / 5-frame mean area)

From UncL-STARK: `heatmap_top3_mass` (top-3 sum of spatially-softmaxed score map) — BETTER calibrated confidence proxy than APCE; specifically, σ_OOTU and this are both HIGH during false_confirmed → empirical differentiator

From MATA: `ego_motion_confidence` (fraction of Lucas-Kanade flow points tracked successfully)

From PTDT/CoTracker3 (Phase 4): `points_inside_ratio`, `fwd_bwd_error`, `point_visibility_mean`

### Critical empirical argument (must be in paper)

uav0000164: σ_OOTU small (distractor well-localized), UncL-STARK confidence HIGH, APCE HIGH — but IoU < 0.2, P(false_confirmed) HIGH. Show scatter plot proving near-zero correlation between localization-uncertainty and identity-failure-probability. This is the paper's central empirical differentiator.

### eval.py must-implement metrics

- NT2F at IoU ∈ {0.5, 0.2} (MATA formula: mean over sequences of (t_failure - t_init) / seq_len)
- False-confirmed recall @ 5% FPR (SALT-RD signature metric — no competitor reports this)
- AUC vs % full-compute frames Pareto (comparable to UncL-STARK GFLOPs table)
- ECE / Brier / NLL on P(false_confirmed) and P(failure_in_5)
- AUROC / AUPRC per head with base rates
