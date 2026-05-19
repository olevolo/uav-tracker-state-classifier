# SALT v3 — Повний технічний аналіз

**Дата:** 2026-05-19  
**Мета:** реал-тайм відстеження об'єктів у відеопотоках БПЛА на периферійному пристрої (Apple MPS)

---

## 1. Результати по всіх датасетах

### 1.1 Порівняльна таблиця (SGLATrack baseline vs SALT v3)

| Датасет | Послідовностей | SGLATrack AUC | SALT v3 AUC | Δ AUC | SGLATrack FPS | SALT FPS | GFLOPs/кадр |
|---------|:-:|:---:|:---:|:---:|:---:|:---:|:---:|
| **UAV123** (повний) | 123 | 0.718 | **0.720** | **+0.002** | 80 | 62 | 1.27 |
| **VisDrone-SOT** (test-dev) | 35 | 0.672 | 0.672 | 0.000 | 72 | 38 | 1.27 |
| **DTB70** | 70 | ~0.737* | 0.670 | **−0.067** | ~94* | 46 | 1.27 |

*SGLATrack на DTB70 не вимірювався напряму — наведено еталон зі статті на UAV123.

### 1.2 GFLOPs деталізація

| Конфігурація | GFLOPs/кадр | vs baseline |
|---|:---:|:---:|
| SGLATrack (no CE) | 1.2663 | — |
| SALT CE kr=0.85 | 1.3116 | **+3.6%** ❌ |
| SALT CE kr=0.75 | 1.2947 | +2.3% ❌ |
| SALT CE kr=0.50 (активний) | 1.2554 | −0.9% |
| CTEM kr=0.50 | 1.1847 | −6.4% ✓ |
| Paper (pre-SGLA trunk) | 0.9000 | — |

> ⚠️ Стаття заявляє 0.90 GFLOPs — це лише перші 5 блоків (до SGLA маршрутизатора). Повна модель = **1.27 GFLOPs**.

### 1.3 UAV123 діагностичний підмножина (6 послідовностей)

| Послідовність | Характер | SGLATrack | SALT v3 | Δ |
|---|---|:---:|:---:|:---:|
| car13 | Легка, перекриття | 0.750 | 0.749 | −0.001 |
| uav2 | Важка, справжня втрата | 0.136 | **0.507** | **+0.371** |
| bike2 | Зміна вигляду | 0.176 | 0.176 | 0.000 |
| car7 | Швидкий рух | 0.595 | **0.612** | **+0.017** |
| building1 | Легка, статична | 0.872 | 0.871 | −0.001 |
| truck1 | Відволікаючий об'єкт | 0.721 | 0.778 | +0.057 |
| **MEAN** | | **0.541** | **0.616** | **+0.075** |

---

## 2. Все що пробували — що дало / не дало результат

### ✅ Дало позитивний результат

| Підхід | Зміна AUC | Примітка |
|--------|:---:|---|
| SGLATrack замість KCF | +0.444 (+0.293→0.737) | Головний стрибок |
| YOLO26m VisDrone для відновлення | uav2: +0.356 | Найбільший SALT внесок |
| APCE як первинний сигнал стану | Усунуто false-LOST на car13 | Flow-IoU давав фіктивні LOST |
| OCCLUDED→LOST ескалація (25 кадрів) | Дозволила recovery на uav2 | Без неї recovery ніколи не спрацьовував |
| Staged ескалація (mean APCE guard) | Усунуто false-escalation на car7 | Просте counting → regression |
| APCE trend gating (Guard A) | bike2: +0.018 | Блокує recovery коли трекер самовідновлюється |
| Guard B (displacement+cosine) | Прибрало bad recovery IoU=0.000 | Recovery quality: 0.476 → 0.951 |
| BUG-02 fix (shared embed helper) | car7: 0.570 → не регресії | Несумісний простір вкладень порушував cosine guards |
| CE виправлення Q1/Q2/Q4 | MEAN: 0.610 → 0.616 | Виявлено 3 архітектурні баги |
| APCECalibrator (адаптивні пороги) | Стабільність на VisDrone | Абсолютні пороги не узагальнювалися |
| VelocityDriftMonitor | Виявляє false-CONFIRMED | uav0000164: 99% CONFIRMED при AUC=0.174 |
| APCE gate для CE (threshold=25) | Захист re-acquisition | truck1 failure при APCE=12-15 |
| BUG-12 fix (32×32 embed) | VisDrone: 27→40 FPS | Матриця 12288×64 = вузьке місце |
| Supervised TSA голова (92.5%) | Краще confidence scoring | Не впливає на маршрутизацію безпосередньо |

### ❌ Не дало результату / спричинило регресію

| Підхід | Результат | Причина |
|--------|:---:|---|
| LSTM предиктор руху | 97% false DYNAMIC, 0 приросту AUC | LSTM не конвергує за 15 кадрів |
| TTT HeadAdaptor (self-supervised) | Zero gradient | Pseudo-GT = передбачення самого трекера |
| CosineAppearanceMemory drift | drift=0.000 на всіх послідовностях | UAV123 недостатньо варіативний |
| Template EMA оновлення (90%+10%) | car7: 0.570→0.321 | 5 оновлень за 500 кадрів накопичують дрейф |
| Search factor expansion (OCCLUDED 5.5×) | car13: 0.750→0.690 | Маленькі цілі втрачають роздільну здатність |
| CTEM (косинусна схожість) | MEAN: 0.442 vs CE 0.616 | Середнє шаблону — занадто грубе для малих цілей БПЛА |
| CE kr=0.75-0.65 | MEAN: 0.554-0.571 | TSA зворотній зв'язок: прунінг → нижчий APCE → більше LOST |
| Оригінальний CE (баги Q1/Q2/Q4) | MEAN: 0.551 | Шар 3,6,9 — але лише 3 досяжний; Q1+Q2+Q4 некоректні |
| Розширений OCCLUDED temporal voting | Не покращило | Справжні та фальшиві recovery мають однаковий консекутивний streak |
| Online MLP адаптація | Нічого | Адаптується до тих самих правил, що вже закодовані |

---

## 3. Навчання / Самонавчання / Адаптація

### 3.1 Що зараз активно

**a) APCECalibrator (адаптивні пороги)**
- Алгоритм: ковзне вікно 100 APCE-значень per-sequence
- Пороги: `LOST = min(20, max(10, p5 × 1.5))`, `OCCLUDED = max(80, p75 × 0.5)`
- Адаптується: тільки вниз для LOST (щоб спіймати важкі послідовності), тільки вгору для OCCLUDED (щоб не скорочувати вікно)
- Проблема: перші 30 кадрів = фіксовані значення (cold-start)

**b) Supervised TSA голова (inference-only)**
- Навчено: 50 epochs на UAV123, `--mode sglatrack` (реальні ознаки)
- Точність: 92.5% на val
- Використання: confidence scoring для TelemetryEntry, НЕ впливає на рішення про стан (рішення = правила APCE)
- Реальний вплив: мінімальний (confidence ≠ routing)

**c) Guard-3 EMA (_ref_embedding)**
- Кожні 50 кадрів: `ref_embedding = 0.80 × old + 0.20 × new`
- Проблема: _ref_embedding дрейфує разом із трекером → cosine guard стає ненадійним для template update
- Наслідок: template оновлення вимкнено (BUG-17)

### 3.2 Що вимкнено і чому

**LSTM (OnlineLSTMMotionPredictor)**
- Вимкнено: `motion_predictor.enabled: false`
- Причина: warm-up 15 кадрів → LSTM residual зростає монотонно → 97% фальшивих DYNAMIC
- Залишено в коді: може бути корисним при іншій стратегії навчання

**Online MLP адаптація (3 SGD кроки/20 кадрів)**
- Вимкнено: `adapt_enabled: false`
- Причина: вчиться відтворювати `_decide_state()` — тобто адаптується до самого себе
- Буфер забруднений: APCE 20-80 = неоднозначна зона, SGD на ній шумить

**Synthetic warmup MLP**
- Видалено: була 300 кроків Adam на синтетичних зразках
- Замінено: навченим на реальних даних supervised головою (92.5% acc)
- Вплив: маргінальний, бо голова не впливає на routing

### 3.3 Як правильно навчати

Поточна підготовлена голова навчена на **правилах** (teacher = `_decide_state()`). Це обмежений self-supervision: модель вчиться наближати правило, а не покращувати його.

Для справжнього покращення потрібно:
1. Регенерувати NPZ з SGLATrack (BUG-01: всі `flow_features` = нулі)
2. Навчати з **справжніми IoU labels** (коли трекер правий/неправий відносно GT)
3. Проблема: GT IoU доступне лише офлайн, не під час inference

---

## 4. Класифікація сцен — чи правильно?

### 4.1 Поточний механізм

```
APCE < 20  → LOST     (40/416 кадрів на uav2 = 10%)
APCE 20-80 → OCCLUDED (85/133 кадрів на uav2 = 64%)
APCE ≥ 80  → CONFIRMED
```

### 4.2 Відомі проблеми класифікації

**Проблема 1: False CONFIRMED (прихований дрейф)**
- Послідовність uav0000164 (VisDrone): AUC=0.174, але TSA = 99% CONFIRMED
- Трекер впевнено стежить за НЕПРАВИЛЬНИМ об'єктом (дистрактор/фон)
- APCE вимірює якість піку, не семантичну правильність
- VelocityDriftMonitor частково вирішує: freeze score + PSR decay
- Але: якщо дистрактор теж рухається подібно до цілі → не детектується

**Проблема 2: OCCLUDED ≠ окклюзія**
- bike2: 85% кадрів OCCLUDED при APCE 40-80
- Насправді: трекер просто має слабший сигнал через зміну вигляду
- OCCLUDED стан не розрізняє: справжнє перекриття vs зміна вигляду vs motion blur

**Проблема 3: Пороги не масштабуються між датасетами**
- UAV123 (низька висота): APCE 200-255 = хороше відстеження
- VisDrone (велика висота): APCE 80-150 = хороше відстеження (маленька ціль)
- DTB70 (різні умови): невідомий розподіл APCE

**Проблема 4: DYNAMIC ніколи не спрацьовує**
- Залежить від LSTM (вимкнено) → normalized_lstm_residual = 0 завжди
- Стан існує в коді, але мертвий на практиці

---

## 5. Взаємодія детектор-трекер

### 5.1 Поточний конвеєр

```
SGLATrack (первинний)
    ↓ APCE > 20 → CONFIRMED/OCCLUDED → CE pruning
    ↓ APCE < 20 → LOST → consecutive_lost++
    
    [після 25 OCCLUDED + mean_APCE < 60] → escalate LOST
    [після 5 consecutive LOST] → YOLO26m
    
YOLO26m VisDrone (відновлення)
    ↓ detect(frame, hint_bbox)
    ↓ Guard 1: size consistency (±70%)
    ↓ Guard 3: cosine similarity > 0.25
    ↓ Guard 5: displacement + crowded scene (cosine > 0.50 при ≥2 кандидати)
    ↓ Temporal voting: 2 кадри
    → tracker.init(winner_bbox)
```

### 5.2 Проблеми взаємодії

**Проблема 1: hint_bbox ігнорується YOLO26m і LEAF-YOLO**
- Детектор приймає `hint_bbox` як параметр, але НЕ використовує для crop/sort
- Тільки RT-DETR сортує по proximity до hint
- Наслідок: YOLO26m шукає по всьому кадру, знаходить дистрактори

**Проблема 2: Recovery не спрацьовує на коротких послідовностях**
- Потрібно: 25 OCCLUDED + 5 LOST = 30 кадрів мінімум
- VisDrone-SOT max streak OCCLUDED = 11 кадрів (всього 35 послідовностей)
- Результат: детектор майже ніколи не викликається на VisDrone

**Проблема 3: Class mismatch при recovery**
- YOLO26m навчено на VisDrone-DET (pedestrian, car, van, truck, bicycle...)
- Немає класу "UAV/drone"
- bike2: знаходить правильний клас (bicycle), але неправильного велосипедиста

**Проблема 4: Recovery IoU = 0.000 на DTB70 hard sequences**
- Gull2: чайка на морі — YOLO26m не знає птахів добре
- StreetBasketball1: гравці схожі між собою → cosine guard не допомагає
- Sheep1: стадо = багато однакових кандидатів

---

## 6. Warm-up і фонова обробка

### 6.1 Ланцюжок ініціалізації

```
SALTRunner.from_config(salt.yaml)
    → SGLATracker._load()               # DeiT-tiny + SGLATrack weights
    → TargetStateAssessor.__init__()    # APCECalibrator + VelocityDriftMonitor
      → _load_head(tsa_head.pth)        # Supervised MLP
    → VisDroneYOLO26m._load()           # lazy, on first detect()
    → SALTRunner.prepare()              # eager tracker load
    → detector.warmup()                 # 1 dummy inference
```

**Час ініціалізації:** ~3-5 секунди (завантаження SGLATrack + YOLO26m warmup)

### 6.2 Per-frame overhead (виміряний)

| Компонент | Час (мс) | Коли |
|---|:---:|---|
| SGLATrack update (full) | 9-22 мс | кожен кадр |
| TSA Farneback (fast path) | 0.0-0.4 мс | при APCE < 120 або перші 5 кадрів |
| TSA fast-path skip | ~0 мс | APCE > 120 + 3+ CONFIRMED кадрів |
| YOLO26m detection | ~23 мс | лише при LOST (рідко) |
| CosineMemory store (32×32) | ~1 мс | кожні 10 кадрів |

### 6.3 Фонова обробка

**Немає.** Весь конвеєр синхронний на головному потоці. Детектор не працює у фоні. YOLO26m викликається блокуюче коли потрібне відновлення. Це обмеження поточної архітектури.

### 6.4 TSA temporal gating (оптимізація)

```python
# Пропускаємо Farneback якщо:
if prev_apce > 120 AND current_apce > 120 AND consecutive_confirmed >= 3 AND frame_idx > 5:
    return CONFIRMED  # без optical flow
```
- Економить ~5-8 мс/кадр на стабільних послідовностях
- ~40% кадрів пропускають повний TSA розрахунок на легких послідовностях

---

## 7. Поведінка при кожному стані та проблеми

### CONFIRMED (APCE ≥ 80) — ~90% кадрів

- CE token pruning kr=0.50 при layer 3
- TSA fast-path (пропуск Farneback)
- APCE gate: якщо prev_apce < 25 → full tokens (захист re-acquisition)
- **Проблема:** false CONFIRMED — трекер впевнений, але неправильний об'єкт
- **Проблема:** CE може видалити lateral-motion tokens → car7 регресія на kr=0.75-0.65

### OCCLUDED (APCE 20-80) — ~5-15% кадрів

- Full tokens, full depth
- Ескалація: якщо consecutive_occluded ≥ 25 AND mean(APCE_last25) < threshold × 0.75 → LOST
- APCE trend gating: якщо поточний APCE зріс ≥15% vs попереднього LOST → пропустити (трекер відновлюється)
- **Проблема:** OCCLUDED не розрізняє справжнє перекриття і motion blur
- **Проблема:** Ескалація занадто повільна для VisDrone (max streak = 11 кадрів < 25)

### LOST (APCE < 20) — ~1-10% кадрів

- Full tokens, full depth
- Після 5 consecutive LOST → детектор
- Тільки після _RECOVERY_WARMUP_FRAMES = 10 кадрів від початку
- **Проблема:** bike2 і car7 — обидва генерують streak 5+ LOST → recovery спрацьовує некоректно
- **Проблема:** DTB70 hard sequences (птахи, природні сцени) — YOLO26m не навчений

### DISTRACTOR_RISK (VelocityDriftMonitor) — ~1-3% кадрів

- Full tokens (як OCCLUDED)
- Спрацьовує при: freeze score + PSR decay + flow disagrees
- **Проблема:** car13 + truck1 + car7 показують ~3-8 DIS кадрів — іноді надмірно агресивний
- **Проблема:** False DIS на статичних цілях (building1) захищений flow guard, але не ідеально

### DYNAMIC — НІКОЛИ не спрацьовує

- Залежить від LSTM (вимкнено)
- Стан мертвий без motion predictor

---

## 8. Чому результати слабкі

### 8.1 Фундаментальне обмеження

**SALT не підвищує точність трекера — він покращує ефективність і відновлення.**

SGLATrack AUC=0.718/0.737 — це стеля поточного трекера. SALT отримує +0.002 на повному UAV123 в основному через recovery на `uav*` послідовностях. На решті 120+ послідовностях SALT = SGLATrack.

**DTB70: −0.067** — тут SALT активно шкодить:
- Gull2 (птах), Sheep1 (вівця), Surfing04 (серфінг) — YOLO26m VisDrone не знає цих категорій
- Recovery спрацьовує, але ставить трекер на неправильний об'єкт
- Без recovery: SALT = SGLATrack; з recovery: SALT < SGLATrack

### 8.2 Структурні обмеження

1. **CE pruning один шар** (`_CE_LOC = {3}`, не 3 шари): max теоретична економія ~7% FLOPs
2. **TSA classification помилкова на ~10%**: false CONFIRMED = найгірший випадок
3. **Детектор без spatial hint**: YOLO26m ігнорує `hint_bbox`
4. **Немає class-aware recovery**: детектор шукає будь-що, не цільовий клас
5. **Recovery threshold не адаптується**: 5 consecutive LOST — однаково для легких і важких сцен
6. **FPS penalty**: 62fps vs 80fps SGLATrack — 23% overhead за мізерний приріст точності

### 8.3 VisDrone і DTB70 проблеми

**VisDrone-SOT (AUC=0.672 = SGLATrack):**
- Цілі <20px (велика висота) — APCE розподіл різний від UAV123
- Recovery ніколи не спрацьовує (послідовності занадто короткі)
- SALT = SGLATrack = нуль цінності від конвеєра

**DTB70 (AUC=0.670, −0.067 vs ref):**
- Hard sequences (птахи, морські тварини) — YOLO26m не навчений
- Recovery шкодить більше ніж допомагає

---

## 9. ChatGPT Deep Research Prompt

```
I'm building a real-time UAV single-object tracking system for edge devices (Apple Silicon MPS/CUDA). 
Current implementation: SGLATrack (DeiT-tiny, 1.27 GFLOPs) with APCE-based state assessment and 
YOLO26m recovery detector. Results: UAV123 AUC=0.720 (+0.002 vs baseline), DTB70 AUC=0.670 
(-0.067 vs baseline), VisDrone-SOT AUC=0.672 (=baseline). The system is not improving meaningfully 
over the bare tracker on most datasets.

Please do deep research on the following specific questions:

## 1. APCE-based tracking quality estimation

What is the state of the art in estimating per-frame tracking failure probability using internal 
tracker signals (score maps, heatmaps, response maps)? Are there methods that go beyond APCE/PSR 
thresholds to provide calibrated probability-of-failure estimates? Specifically:
- Papers using uncertainty quantification in ViT-based trackers (2023-2026)
- Methods for distinguishing "tracker on wrong object" (false-confirmed) from "tracker on correct 
  but occluded object" without ground truth
- Any papers that combine optical flow consistency + tracker confidence for failure detection

## 2. Token pruning for ViT trackers with small targets

Current CE token pruning in SGLATrack is implemented as single-stage (only layer 3 fires, not 3 
stages as claimed), giving <7% FLOPs reduction. What approaches exist for:
- Scale-aware token pruning that adjusts keep_ratio based on target size (bbox/frame ratio)
- Multi-stage pruning that correctly integrates with SGLA routing
- Dynamic token merging (ToMe) rather than elimination for ViT trackers
- Papers 2024-2026 specifically addressing small target tracking with token pruning

## 3. Recovery pipeline design

Current approach: OCCLUDED escalation → YOLO26m → cosine+spatial guards → temporal voting.
Problems: (1) detector class mismatch, (2) no spatial hint usage, (3) works only when 25+ 
consecutive OCCLUDED frames occur. What are the best practices for:
- Detector-tracker fusion for re-initialization after tracking failure
- Class-agnostic re-detection for open-vocabulary scenarios (birds, unusual objects)
- Short-term failure recovery (5-15 frames lost) without detector re-init
- Template re-acquisition using appearance matching without explicit detection

## 4. Domain generalization across aerial tracking datasets

SGLATrack is trained on LaSOT/GOT-10k (ground-level). SALT sees: UAV123 (low-altitude drone), 
VisDrone-SOT (high-altitude surveillance), DTB70 (drone perspective, various objects). 
What is the state of the art in:
- Fine-tuning transformer trackers for aerial domains without overfitting
- Multi-dataset training strategies (TrackingNet + UAV123 + VisDrone simultaneously)
- Test-time adaptation for ViT trackers that actually works (unlike TTT approaches with self-
  referential gradients)
- UAV-specific tracking papers 2024-2026 that outperform SGLATrack (AUC 0.737 on UAV123)

## 5. Alternative architectures for real-time UAV tracking

What trackers in 2024-2026 achieve >0.75 AUC on UAV123 at >60 FPS on consumer hardware? 
Specifically looking for:
- Trackers that achieve better than AUC=0.737 on UAV123 with <2 GFLOPs
- Lightweight backbone alternatives to DeiT-tiny for aerial tracking
- State-space model trackers (Mamba) for sequential UAV tracking
- Any approach that explicitly models UAV-specific challenges: small targets, fast camera motion, 
  similar-appearance distractors

## 6. Self-supervised tracking improvement

Current self-supervised signals: APCE (quality), optical flow consistency (pseudo-GT), 
cosine appearance memory (drift). What new self-supervised signals have been proposed for:
- Online tracker improvement during inference without GT labels
- Cycle-consistency between forward and backward tracking as training signal
- Using detection model outputs as pseudo-labels for tracker adaptation (not self-referential)
- DINO/SAM features for appearance-based self-supervision during tracking

Please provide specific paper titles, venues, years, key metrics on UAV123/VisDrone/DTB70, 
and code availability. Focus on 2024-2026 papers that could realistically improve on 
AUC=0.737 UAV123 baseline at real-time speed on edge hardware.
```

---

## 10. Пріоритетні напрямки покращення

На основі аналізу, від найбільш до найменш перспективних:

| Пріоритет | Напрямок | Очікуваний приріст | Складність |
|-----------|----------|:-:|:-:|
| 1 | Spatial hint для YOLO26m (crop search region) | +0.05-0.10 на DTB70 | Низька |
| 2 | Адаптивний порядок recovery threshold (per-sequence) | +0.02-0.05 | Середня |
| 3 | Multi-template appearance matching (без детектора) | +0.03-0.07 на bike2-type | Висока |
| 4 | Scale-aware CE pruning (більше keep для малих цілей) | +0.01-0.03 FPS | Середня |
| 5 | Fine-tune SGLATrack на UAV/VisDrone domain | +0.05-0.15 (фундаментально) | Дуже висока |
| 6 | Замінити YOLO26m на class-agnostic detector (SAM2/Grounding-DINO) | +0.03-0.08 на DTB70 | Висока |
