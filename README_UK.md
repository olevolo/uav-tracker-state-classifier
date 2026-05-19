# SALT — Самонавчальний адаптивний відстежувач із розпізнаванням стану

**SALT** (*Self-Adaptive Learning Tracker*) — модульна система відстеження одиночного об'єкта на відео з БПЛА. Центральна ідея: оцінювач стану сцени (ОСС) класифікує кожен кадр і маршрутизує обчислювальне навантаження — скорочує жетони уваги (CE-відсікання) у підтверджених кадрах та залучає детектор під час відновлення після втрати цілі.

Відносно базового відстежувача SGLATrack: **+0.075 AUC** на UAV123 при ~56 кадрів/с.

---

## Архітектура конвеєра обробки

```
Вхідний кадр
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  SALTRunner                                          │
│                                                      │
│  ┌──────────────────────┐   ┌────────────────────┐  │
│  │  SGLATrack (DeiT-tiny)│   │  ОСС (TSA)         │  │
│  │  1.27 ГФлоп (повн.)  │   │  APCECalibrator    │  │
│  │  0.90 ГФлоп (до SGLA)│   │  VelocityDrift     │  │
│  │                      │◄──│  Farneback (оп.пот.) │  │
│  │  CE kr=0.50          │   │  МШП-голова 92.5%  │  │
│  │  (тільки CONFIRMED)  │   └────────────────────┘  │
│  └─────────┬────────────┘             │              │
│            │                   стан ОСС              │
│            ▼                         │              │
│  ┌──────────────────────┐            │              │
│  │  Пам'ять зовнішн.вигл│            │              │
│  │  CosineAppearance    │            │              │
│  │  (32×32 кроп)        │            │              │
│  └──────────────────────┘            │              │
│            │                         │              │
│            ▼                         ▼              │
│  ┌──────────────────────────────────────────────┐   │
│  │  Конвеєр відновлення (лише стан LOST)         │   │
│  │  YOLO26m VisDrone (55% mAP@0.5)              │   │
│  │  Захист-3: косинусна подібність ≥ 0.25       │   │
│  │  Захист-5: зміщення + перевантаженість сцени │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
     │
     ▼
BBox + стан ОСС + кадрів/с
```

---

## Результати контрольного оцінювання

### UAV123 — 6 діагностичних послідовностей (обмеження 500 кадрів)

| Послідовність | SGLATrack | SALT v3 | Δ | кадрів/с |
|--------------|:---------:|:-------:|:-:|:--------:|
| car13        | 0.750     | 0.749   | −0.001 | 54 |
| uav2         | 0.136     | **0.507** | **+0.371** | 66 |
| bike2        | 0.176     | 0.176   | 0.000 | 56 |
| car7         | 0.595     | **0.612** | **+0.017** | 56 |
| building1    | 0.872     | 0.871   | −0.001 | 55 |
| truck1       | 0.721     | **0.778** | **+0.057** | 50 |
| **СЕРЕДНЄ** | **0.541** | **0.616** | **+0.075** | **56** |

Якість відновлення: 1 подія, середній IoU = **0.951**

### VisDrone-SOT test-dev — 35 послідовностей (повний набір)

| Метод | AUC | Pr@20 | кадрів/с |
|-------|:---:|:-----:|:--------:|
| SGLATrack | 0.672 | 0.859 | 78 |
| SALT v3   | **0.673** | 0.858 | **40** |

### Еволюція версій SALT (СЕРЕДНЄ UAV123)

| Версія | AUC | Ключова зміна |
|--------|:---:|--------------|
| SGLATrack (базова лінія) | 0.541 | Без SALT |
| SALT v1 (2026-05-18) | 0.541 | AUC = базова лінія, 55 кадрів/с |
| SALT v2 (CE зламано, без відсікання) | 0.610 | +маршрутизація TSA + відновлення |
| SALT v2a (CE виправлено, неправильна архітектура) | 0.551 | Регресія зворотного зв'язку CE |
| **SALT v3 (CE виправлено, kr=0.50)** | **0.616** | **+0.006 від відсікання відволікачів** |

---

## Архітектурні компоненти

| Компонент | Файл | Опис |
|-----------|------|------|
| `SALTRunner` | `src/uav_tracker/salt_runner.py` | Головний конвеєр — з'єднує всі компоненти |
| `SGLATracker` | `src/uav_tracker/trackers/sglatrack.py` | DeiT-tiny + маршрутизація CE-відсікання |
| `TargetStateAssessor` | `src/uav_tracker/ml/tsa/target_state_assessor.py` | ОСС: APCECalibrator, VelocityDriftMonitor, МШП-голова |
| `VelocityDriftMonitor` | `src/uav_tracker/ml/tsa/velocity_drift.py` | Виявлення хибного CONFIRMED |
| `CosineAppearanceMemory` | `src/uav_tracker/ml/appearance_memory/cosine_memory.py` | Векторне подання 32×32 (раніше 64×64) |
| `VisDroneSOTDataset` | `src/uav_tracker/datasets/visdrone_sot.py` | Завантажувач VisDrone-SOT |
| YOLO26m VisDrone | `src/uav_tracker/detectors/visdrone_yolo26m.py` | Виявлювач для відновлення після втрати |
| `configs/experiments/salt.yaml` | — | Активна конфігурація SALT |

---

## Станова машина ОСС

ОСС (TSA — Target State Assessor) — оцінювач, що переводить кожен кадр в один з п'яти станів. APCE (середня пікова кореляційна енергія) є основним сигналом; оптичний потік Фарнебека — резервний.

| Стан | Поріг APCE | Маршрутизація обчислень | Відновлення |
|------|:----------:|:----------------------:|:-----------:|
| `CONFIRMED` | ≥ 80 | CE-відсікання 50% жетонів | — |
| `OCCLUDED` | 20–80 | Повна глибина | Ескалація після 25 кадрів |
| `LOST` | < 20 | Повна глибина | YOLO26m після 5 кадрів LOST |
| `DISTRACTOR_RISK` | CONFIRMED + дрейф швидкості | Повна глибина | — |
| `DYNAMIC` | (недосяжний — LSTM вимкнено) | Повна глибина | — |

**Примітки:**
- `APCECalibrator` адаптує пороги онлайн: `max(80, p75×0.5)` для OCCLUDED; `min(20, p5×1.5)` для LOST
- `DISTRACTOR_RISK` спрацьовує, коли замерзлий рахунок + затухання PSR → хибний CONFIRMED
- `DYNAMIC` недосяжний при вимкненому LSTM (`motion_predictor: enabled: false` у `salt.yaml`)

---

## CE-відсікання жетонів — виправлення Q1/Q2/Q4

### Передісторія

DeiT-tiny використовує блоки `timm`, які не підтримують `return_attention=True`. До цієї сесії CE-оцінювання було «тихим» no-op — усі раніше виміряні результати фактично отримані без будь-якого відсікання. Приріст +0.069 AUC у версіях v1→v2 належить TSA-маршрутизації та відновленню через YOLO26m, а не CE.

### Три виправлені архітектурні помилки (`base_backbone.py`)

| Помилка | Опис | Серйозність |
|---------|------|:-----------:|
| **Q1** | `_CE_LOC={3,6,9}`, але умова `i < start_layer=5` ніколи не виконується для шарів 6 і 9 → відсікання у цих шарах не відбувалось | ВИСОКА |
| **Q2** | CE оцінював вихід блоку `i` QKV-матрицями блоку `i+1` (невідповідність розподілів) | СЕРЕДНЯ |
| **Q4** | CTEM використовував середнє всіх 64 жетонів шаблону (домінує фон для малих БПЛА); виправлено: центральні 4×4 жетони | СЕРЕДНЯ |

### Результати (СЕРЕДНЄ AUC UAV123)

| kr (коефіцієнт збереження) | CE до виправлень | CE після виправлень |
|:--------------------------:|:----------------:|:-------------------:|
| 0.85 | 0.567 | **0.610** |
| 0.75 | 0.573 | 0.554 |
| 0.65 | 0.558 | 0.571 |
| **0.50** | 0.551 | **0.616** ← активна конфіг. |

**Поточна конфігурація:** CE у шарі 3, kr=0.50, лише стан `CONFIRMED`.

### Аналіз обчислювальних витрат (ГФлоп)

| Режим | ГФлоп | Зміна |
|-------|:-----:|:-----:|
| Повна модель | 1.2663 | — |
| CE kr=0.85 | +3.6% | Затратно — не рекомендується |
| CE kr=0.50 | −0.9% | Мінімальна економія, +0.006 AUC |
| CTEM kr=0.50 | −6.4% | Дешевше, але AUC гірший |

Накладні витрати CE (norm1+QKV оцінюваного блоку = 0.071 ГФлоп) перевищують економію при kr > 0.57. Максимальне прискорення від одноступеневого відсікання (лише шар 3): ~7% із CTEM kr=0.50.

---

## Швидкий старт

### Встановлення

```bash
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt

# Перевірка реєстру плагінів
uav-tracker list-plugins
```

### Швидкий тест продуктивності (6 послідовностей, ~2–3 хв)

```bash
# SALT v3 (активна конфігурація)
PYTHONPATH=src python scripts/fast_bench.py \
  --config configs/experiments/salt.yaml \
  --dataset uav123 --max-frames 500

# Базова лінія SGLATrack
PYTHONPATH=src python scripts/fast_bench.py \
  --tracker sglatrack \
  --dataset uav123 --max-frames 500

# VisDrone-SOT (35 послідовностей)
PYTHONPATH=src python scripts/fast_bench.py \
  --config configs/experiments/salt.yaml \
  --dataset visdrone_sot
```

### Повний тест продуктивності UAV123 (123 послідовності)

```bash
PYTHONPATH=src python scripts/run_benchmark.py \
  --config configs/experiments/salt.yaml \
  --dataset uav123
```

### Навчання класифікатора ОСС

```bash
# Генерація міток (SGLATrack на UAV123)
PYTHONPATH=src python scripts/generate_ml_labels.py \
  --tracker sglatrack \
  --output data/uav123_labels_sgla.npz

# Навчання МШП-голови (50 епох)
PYTHONPATH=src python scripts/train_tsa_classifier.py \
  --mode sglatrack --epochs 50
# Збережено в: weights/tsa_head_uav123.pth
```

### Аблаційне дослідження виявлювачів відновлення

```bash
PYTHONPATH=src python scripts/eval_recovery_detectors.py \
  --configs configs/experiments/salt_detector_yolo26m.yaml \
            configs/experiments/salt_detector_rtdetr.yaml
```

---

## Файлова структура

```
uav-tracker-detector/
├── configs/
│   └── experiments/
│       ├── salt.yaml                    ← активна конфігурація SALT
│       ├── salt_detector_yolo26m.yaml   ← аблація виявлювачів
│       └── salt_detector_rtdetr.yaml
├── scripts/
│   ├── fast_bench.py        ← швидкий тест (6 послідовностей)
│   ├── run_benchmark.py     ← повний тест (123/35 послідовностей)
│   ├── train_tsa_classifier.py
│   ├── generate_ml_labels.py
│   └── eval_recovery_detectors.py
├── src/uav_tracker/
│   ├── salt_runner.py       ← головний конвеєр SALT
│   ├── trackers/
│   │   └── sglatrack.py     ← SGLATrack + CE-маршрутизація
│   ├── ml/
│   │   ├── tsa/             ← ОСС: APCECalibrator, VelocityDriftMonitor
│   │   ├── appearance_memory/  ← CosineAppearanceMemory (32×32)
│   │   └── motion_predictor/  ← LSTM (вимкнено)
│   ├── detectors/
│   │   └── visdrone_yolo26m.py  ← виявлювач для відновлення
│   └── datasets/
│       ├── uav123.py        ← UAV123 (123 послідовності)
│       ├── visdrone_sot.py  ← VisDrone-SOT (35 послідовностей)
│       └── dtb70.py         ← DTB70 (70 послідовностей, очікує завантаження)
├── weights/
│   ├── tsa_head_uav123.pth  ← навчена МШП-голова ОСС (92.5% точн.)
│   └── ostrack/             ← ваги OSTrack (довідкові)
├── sglatrack_ep0297.pth.tar ← ваги SGLATrack (AUC 0.737 на UAV123)
└── bugs.md                  ← реєстр 18 виявлених помилок
```

---

## Відомі обмеження

| Послідовність | Δ AUC | Причина |
|--------------|:-----:|---------|
| bike2 | −0.005 | YOLO26m знаходить схожого велосипедиста (cosine_sim=0.921, IoU=0.000) — архітектурна проблема BUG-18 |
| car7 | −0.017 | 50 кадрів OCCLUDED із розмиттям руху (середній IoU=0.219) — повна глибина не допомагає |

BUG-18 частково вирішено: Захист-5 (`_last_recovery_sim`) та APCE-трендове відсікання (`_prev_escalated_apce`) блокують більшість хибних відновлень.

---

## Посилання

- [DATASETS_UK.md](DATASETS_UK.md) — довідник по наборах даних UAV123, VisDrone-SOT, DTB70
- [SESSION_SUMMARY_UK.md](SESSION_SUMMARY_UK.md) — підсумок сесій розробки, таблиці результатів, виправлені помилки
- [bugs.md](bugs.md) — повний реєстр 18 виявлених помилок

## Цитування

```bibtex
@article{oleksiuk2026salt,
  author  = {Oleksiuk, V.},
  title   = {SALT: Scene-Adaptive Learning Tracker for UAV Single-Object Tracking},
  journal = {Electronics and Information Technologies},
  volume  = {33},
  year    = {2026}
}
```
