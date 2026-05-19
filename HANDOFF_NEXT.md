# HANDOFF — SALT-R: Calibrated Failure-Aware Re-Acquisition

**Дата:** 2026-05-19 — фінальна версія з урахуванням Staff ML/CV Architect review

**Читай також:** `THOUGHTS.md` — всі зовнішні огляди, рекомендації, аналіз MSTFT/MATA papers  
**Papers:** `papers/MSTFT_2026_*.pdf`, `papers/MATA_2026_*.pdf`

---

## Наукова позиція

**Thesis:** "Calibrated Failure-Aware Re-Acquisition for Real-Time UAV SOT"  
Не "кращий tracker". Система яка знає, коли вона впевнено помиляється.

**Головна проблема:** `uav0000164` — AUC=0.174 при 99% CONFIRMED. APCE не бачить identity switch. Жоден SOTA (SGLATrack, UTPTrack, ORTrack) не вимірює і не вирішує це.

**SOTA стеля (з THOUGHTS.md):**  
- MSTFT 2026: UAV123 AUC=79.4% при 45 FPS — новий SOTA для backbone порівняння  
- MATA 2026: NT2F metric + ego-motion — systems evaluation reference  
- UTPTrack CVPR 2026: 65% joint token pruning — CE/CTEM мертві як внески

---

## Крок -1 — saltv3/ реорганізація (30 хв, ПЕРШ ЗА ВСЕ)

**Мета:** чітко відокремити старий код від нового. Три системи в одному repo — головний ризик для агентів.

```bash
# Структура після реорганізації:
uav-tracker-detector/
  saltv3/                    ← ВСЕ поточне (freeze, не змінювати)
    src/
    configs/
    scripts/
    tests/
    weights/
    notebooks/
    ...
    
  saltr/                     ← НОВА система (з нуля)
    src/
      salt_r/
        model.py             ← GRU/TCN reliability head
        collect_features.py  ← feature collector  
        train.py
        eval.py
        integrate.py         ← wrapper around saltv3
    configs/
    data/
    
  papers/                    ← залишається (вже є)
  THOUGHTS.md                ← залишається
  HANDOFF_NEXT.md            ← залишається
```

**Практично:**
```bash
mkdir -p saltr/src/salt_r saltr/configs saltr/data
# Нові скрипти пишемо тільки в saltr/
# saltv3/ — frozen, тільки читаємо

# saltr/src/salt_r/integrate.py:
import sys
sys.path.insert(0, '../saltv3/src')
from uav_tracker.salt_runner import SALTRunner  # читаємо, не змінюємо
```

**Важливо:** saltv3/ не змінюється після цього кроку. SALT-R є wrapper навколо нього.

## ⚠️ Критичні ризики (Codex Staff Review)

### Ризик 1: Label leakage / self-teaching — СМЕРТЕЛЬНИЙ ДЛЯ СТАТТІ
```
Старий train_tsa_classifier.py: scene_labels → TargetState → APCE rules → circular
Якщо знову навчити модель повторювати _decide_state() → "гарна валідація" на власних правилах
→ модель не вирішує реальну помилку, просто апроксимує пороги
```
**Побороти:** окремий `scripts/collect_salt_r_features.py` з ТІЛЬКИ GT IoU labels.  
`train_tsa_classifier.py` залишити як legacy або видалити з active path.

### Ризик 2: False-confirmed дуже рідкісний клас (1-3%)
```
З наївним BCELoss: модель буде казати "0" завжди і досягне 97% accuracy
при AUROC = 0.50 (random)
```
**Побороти:** weighted BCE (pos_weight ≈ 40), Focal Loss, AUPRC як primary metric.  
Виділити hard-negative suite: `uav0000164, bike2, Gull2, Sheep1, StreetBasketball1`

### Ризик 3: Features замалі — score map geometry відсутня
```
Зараз generate_ml_labels.py пише тільки APCE/PSR/entropy у slots 11-13
Треба: top1/top2 peak gap, peak width, secondary peaks, peak distance
Для цього: зберігати score_map або похідні в TrackState.aux
```
**Побороти:** розширити `TrackState` з `score_map_stats` dict, заповнювати в SGLATracker.

### Ризик 4: Split по послідовностях — зараз НЕПРАВИЛЬНИЙ
```
Поточний UAV123 split: алфавітний (bike1...uav7), не stratified, тільки UAV123
```
**Правильно:** stratified group split по sequence, across UAV123 + VisDrone-SOT + DTB70,  
рівний розподіл easy/hard/failure sequences в train і val.

### Ризик 5: Worktree dirty — три системи в одному repo
```
Repo зараз містить: legacy v2 (entropy/scene-router), SALT v3, заготовки SALT-R
Будь-який наступний агент може "наступити" на стару архітектуру
```
**Побороти:** один migration commit з чітким розділенням перед ML тренуванням.

### Ризик 6: Reproducibility
```
SGLATrack path: hardcoded /Users/voleksiuk/projects/SGLATrack
torch pin: 2.1.0, але env має 2.11.0
```
**Побороти:** env var для external repos, smoke test.

---

## Крок 0 — Cleanup (1-2 год, ДО будь-якого ML)

### 0a. Вимкнути CE і мертвий код
```python
# sglatrack.py: _STATE_COMPUTE_MAP CONFIRMED → [1.0, 1.0, 1.0]
# base_backbone.py: ctem_prune_tokens() і _ce_prune_tokens_from_qk() → archive/
# target_state_assessor.py: видалити DYNAMIC branch (_decide_state)
# salt_runner.py: видалити _ref_embedding EMA update block (~line 417)
# salt_runner.py: вимкнути VelocityDriftMonitor override (тимчасово)
```

### 0b. Розширити TrackState з score_map_stats
```python
# types.py — додати до TrackState:
@dataclass
class TrackState:
    ...
    score_map_stats: dict = field(default_factory=dict)
    # Заповнити в sglatrack.py update():
    # {"peak_margin": top1-top2, "peak_width": FWHM/16,
    #  "n_secondary": count, "peak_distance": dist(top1,top2)}
```

### 0c. Верифікація
```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/unit/ -q  # all pass
PYTHONPATH=src .venv/bin/python scripts/fast_bench.py --mode salt --dataset uav123 2>&1 | grep MEAN
# AUC ≈ 0.610 (без CE і VelocityDrift зміниться)
```

---

## Три нових скрипти (SALT-R v0)

### scripts/collect_salt_r_features.py
```python
"""
Збирає features + GT IoU labels з SGLATrack на UAV123/VisDrone/DTB70.
НЕ використовує _decide_state() або scene_class як labels.
НЕ залежить від train_tsa_classifier.py.
"""

Features (18 signals):
  Score map: apce, psr, entropy, peak_margin, peak_width, n_secondary, peak_distance
  Temporal:  apce_t/apce_mean10, entropy_deviation
  Dynamics:  velocity, scale_ratio, dist_to_border
  Flow:      flow_iou, flow_residual
  Appearance: cosine_static, embedding_drift
  + 2 reserved slots

Labels від GT IoU (не від правил):
  correct[t]         = iou[t] >= 0.5
  false_confirmed[t] = iou[t] < 0.2 AND apce[t] > 100/256
  failure_in_5[t]    = mean(iou[t+1:t+6]) < 0.3 AND iou[t] > 0.5
  recoverable[t]     = iou[t] < 0.2 AND max(iou[t+5:t+15]) > 0.5

Split: stratified по послідовностях (не по кадрах!)
  train_seqs / val_seqs = train_test_split(seqs, stratify=difficulty_bucket)

Output: data/salt_r_features.npz
```

### scripts/train_salt_r.py
```python
"""
Тренує SALT-R v0: GRU(18→48→3) multi-head на GT IoU labels.
"""

class SALTR(nn.Module):
    # GRU(input=18, hidden=48, layers=2, dropout=0.3)
    # Head: FC(48→16) → ReLU → Dropout(0.2) → FC(16→3) → Sigmoid
    # 3 outputs: P(false_confirmed), P(failure_in_5), P(recoverable)

Loss: weighted BCE, pos_weight_false_confirmed ≈ 40
Optimizer: AdamW(lr=1e-3, weight_decay=1e-4)
Primary metric: AUPRC (краще для imbalanced) + AUROC
```

### scripts/eval_salt_r.py
```python
"""
Evaluation: AUROC, AUPRC, ECE/Brier, false-confirmed recall@5%FPR,
wrong-reinit simulation, abstention gain.
"""

Метрики:
  AUROC і AUPRC для кожного head
  ECE (calibration curve) — чи P=0.8 реально = 80%?
  false_confirmed recall@5%FPR — signature metric для статті
  Wrong-reinit simulation: скільки поточних recovery були б заблоковані
```

---

## GO / NO-GO gate

| Метрика | GO | СТОП |
|---------|:---:|:---:|
| AUPRC false_confirmed | > 0.30 | < 0.15 |
| AUROC false_confirmed | > 0.65 | < 0.55 |
| AUROC failure_in_5 | > 0.75 | < 0.65 |
| ECE | < 0.12 | > 0.20 |

*(AUPRC 0.30 при base rate ~2% = 15× краще за random — це реальна новизна)*

---

## Якщо GO: інтеграція в SALT

```python
# В TargetStateAssessor.assess() після _decide_state():
if self.salt_r is not None:
    p_false_conf, p_fail5, p_rec = self.salt_r.predict(self._feature_window)
    
    if p_false_conf > 0.7 and state == TargetState.CONFIRMED:
        state = TargetState.DISTRACTOR_RISK  # preventive

# В SALTRunner recovery:
if p_rec < 0.3:
    self._lost_cooldown = 30  # abstain — recovery не варто
```

---

## Повний фазовий план (після GO)

```
Фаза 0:  Cleanup + TrackState.score_map_stats (1-2 год)
Фаза 1:  SALT-R v0 — scalar telemetry (1-2 дні) ← GO/NO-GO
Фаза 2a: hint_bbox ROI + conservative recovery by domain (1 день)  
Фаза 2b: SALT-R v1 — + score map CNN features (2-3 дні)
Фаза 3:  SALT-Match — candidate accept/reject (1 тиждень)
Фаза 4:  Point consistency + ego-motion signals (1-2 тижні)
Фаза 5:  SALT-Distill — DINOv2/SAM2/CoTracker3 teachers (2-4 тижні)
Фаза 6:  Domain adaptation SGLATrack fine-tune (3-6 тижнів)
```

**НЕ ДИВИТИСЬ на Фази 2-6 до підтвердження Фази 1.**

---

## Стартовий промпт для нової сесії

```
Прочитай HANDOFF_NEXT.md повністю, включно з розділом "Критичні ризики".

ЗАДАЧА: реалізуй Крок 0 + Фазу 1 (SALT-R v0).

Крок 0 спочатку:
- Вимкни CE pruning: _STATE_COMPUTE_MAP CONFIRMED → [1.0, 1.0, 1.0]
- Видали DYNAMIC branch з _decide_state()
- Видали _ref_embedding EMA update
- Розшир TrackState з score_map_stats (peak_margin, peak_width тощо)
- Перевір: pytest pass + fast_bench MEAN ≈ 0.610

Фаза 1: три нових скрипти
- collect_salt_r_features.py — GT IoU labels ТІЛЬКИ, без _decide_state rules
- train_salt_r.py — GRU multi-head, weighted BCE, pos_weight_false_confirmed=40
- eval_salt_r.py — AUPRC primary, ECE, false-confirmed recall@5%FPR

КРИТИЧНО: не використовувати scene_class або _decide_state() як labels.
КРИТИЧНО: split по послідовностях (не по кадрах).
КРИТИЧНО: hard negatives: uav0000164, bike2, Gull2, Sheep1, StreetBasketball1.

GO якщо AUPRC false_confirmed > 0.30 AND AUROC failure_5 > 0.75.
СТОП якщо ні — не йти далі.
```

---

## Ключові papers для SALT-R (в papers/)

### MSTFT 2026 (papers/MSTFT_2026_*.pdf)
**UAV123 AUC=79.4%, 45 FPS** — новий SOTA для порівняння (SGLATrack=73.7%)

**Що взяти для features:**
- Triple safety verification: response_peak + temporal_consistency + motion_stability
  → підтверджує наш feature set (peak_margin, temporal APCE deviation, velocity)
- Dynamic Template Fusion: condition для template update = наш SALT-Match reference

**Наша позиція vs MSTFT:**
- MSTFT: новий backbone (Mamba) — "кращий трекер"
- SALT-R: reliability head поверх будь-якого трекера — "самосвідомий трекер"
- Ці роботи ДОПОВНЮЮТЬ одна одну, не конкурують

---

### MATA 2026 (papers/MATA_2026_*.pdf)
**arXiv:2603.03904v2 — NVIDIA Jetson AGX Orin validated**

**NT2F metric (Normalized Time to Failure):**
```
NT2F_i = (t_failure_i - t_init_i) / sequence_length_i
NT2F = mean across sequences
t_failure = перший кадр де IoU < threshold (0.2 або 0.5)
```
Реалізувати у eval_salt_r.py. Порівнювати SALT-R vs SALT v3 vs SGLATrack на NT2F.

**Ego-motion residual (Block B у MATA):**
```python
# Sparse optical flow на background points → homography
# ego_motion_residual = ||target_flow - global_flow|| / diag
# Вже є через Farneback, потрібно тільки розрахувати окремо background flow
```

**"Check measure validity" в MATA:**
= архітектурний прецедент для нашого P(correct) head
MATA валідує перед EKF, ми — перед recovery і template update

**PMF confidence (MixFormerV2 in MATA):**
- AUC під піком PMF bbox coordinates = calibrated confidence
- Аналог нашого APCE + peak_margin
- Для майбутнього: MATA framework + SALT-R reliability head = combined system

---

## Документи які треба прочитати в новій сесії

| Файл | Що знайти |
|------|-----------|
| `THOUGHTS.md` | Всі коментарі експертів + paper summaries |
| `papers/MSTFT_2026_*.pdf` | Triple verification mechanism (секція 3.2) |
| `papers/MATA_2026_*.pdf` | NT2F формула (секція 3) + EOP protocol |
| `ANALYSIS.md` | Технічний аналіз SALT v3 + Deep Research prompt |
| `bugs.md` | 18 bugs — що вже виправлено |
