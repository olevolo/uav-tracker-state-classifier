# CLAUDE.md

# Safety Scope

This task is limited to offline computer-vision benchmarking on public single-object tracking datasets such as LaSOT, GOT-10k, UAV123, DTB70, and VisDrone-SOT.

The system is not connected to any real UAV, weapon system, battlefield system, live camera, navigation stack, actuator, targeting system, or deployment environment.

The requested work is diagnosis-only:
- telemetry validation;
- confidence calibration;
- label quality checks;
- feature quality analysis;
- offline classifier evaluation.

Do not implement real-world guidance, targeting, weaponization, deployment, or autonomous control.

## Target Paper (what we are aiming for)

The deliverable that defines this project is the journal article draft at:

```
docs/paper/csc_uav_tracking_article.tex   ← LaTeX source (XeLaTeX/LuaLaTeX)
docs/paper/csc_uav_tracking_article.pdf   ← rendered draft
docs/paper/csc_uav_tracking_article.docx  ← Word version
```

Title: *"State-Aware Evaluation and Control of UAV Visual Trackers Using
ClassStateClassifier with False-Confirmed Failure Detection"*
Authors: Volodymyr Oleksiuk, Serhiy Velhosh — Department of Radiophysics
and Computer Technologies, Ivan Franko National University of Lviv.

What the paper claims (and what implementation must therefore deliver):

- Six-state taxonomy: `confirmed`, `uncertain`, `occluded`, `lost`,
  `distractor`, `false_confirmed`.
- `false_confirmed` is the central novelty — a high-confidence wrong
  prediction that conventional Success AUC / Precision do not surface.
- CSC is **tracker-agnostic** — operates purely on telemetry
  (confidence, response-map stats, bbox dynamics, motion entropy,
  appearance consistency, detector agreement, template age, token-keep
  ratio, temporal stability).
- Six trackers are evaluated: SGLATrack, ORTrack, Aba-ViTrack,
  ParallelTracker, SiamHFFT, DSATrack.
- New metrics: False Confirmed Rate (FCR), False Confirmed Duration
  (FCD), Unsafe Update Rate (UUR), Recovery@K, State-Conditioned AUC,
  State Transition Matrix.
- Illustrative target effect: **27–42% reduction in FCR** with 3–7 FPS
  overhead.
- Final evaluation reserved for UAV123 + UAV123@10fps.

All numerical values currently in the draft are marked illustrative —
implementation must produce real measured results to replace them
before submission.

## Project Overview

This repository implements **ClassStateClassifier (CSC)** for UAV visual object tracking.

CSC is a lightweight state-aware diagnostic and optional control layer for visual trackers. It does not replace a tracker. Instead, it observes tracker telemetry and predicts the current semantic tracking state:

1. `confirmed`
2. `uncertain`
3. `occluded`
4. `lost`
5. `distractor`
6. `false_confirmed`

The central research novelty is the `false_confirmed` state: a critical failure mode where the tracker remains internally confident but localizes the wrong object or background region.

The project is part of a PhD research direction on **real-time self-learning/adaptive models for auto-guidance and UAV tracking**.

## Scientific Goal

The goal is to build a framework that evaluates and optionally improves UAV trackers through state-aware failure diagnosis.

Conventional metrics such as Success AUC, Precision@20, FPS, and FLOPs summarize tracker quality but do not explain hidden failure behavior. CSC adds a diagnostic layer that identifies whether the tracker is stable, uncertain, occluded, lost, attracted by a distractor, or confidently wrong.

The key contribution is not only improving average tracking accuracy, but exposing and reducing dangerous failure states such as `false_confirmed`.

## Important Research Constraints

Do not fabricate real experimental results.

If numbers are needed for draft tables, mark them clearly as:
`illustrative`, `synthetic`, `model`, or `placeholder`.

Never present generated numbers as actual measured results.

UAV123 and UAV123@10fps must be used only for final evaluation. Do not train CSC on UAV123. Do not tune thresholds directly on UAV123, unless the experiment is explicitly marked as exploratory and not final.

Preferred protocol:

```text
Train CSC:
LaSOT selected categories, optionally GOT-10k, TrackingNet, COCO-derived pairs

Validation / UAV adaptation:
DTB70, UAVDT, VisDrone-SOT

Final test:
UAV123, UAV123@10fps
```

## Research Context

The previous article introduced an entropy-guided hybrid tracker for real-time UAV tracking. It combined KCF + Kalman prediction with a deeper Siamese tracker and used motion entropy / residual flow to decide when to switch tracking modes.

This project generalizes that adaptive idea:
instead of switching only between trackers, CSC diagnoses semantic tracking states and can guide update control, pruning, verification, or re-detection.

## Target Repository Structure

Use this structure unless the existing project already has a better one:

```text
csc_uav_tracking/
  configs/
    datasets.yaml
    trackers.yaml
    csc_model.yaml
    metrics.yaml
    experiments.yaml

  scripts/
    download_lasot_categories.py
    prepare_lasot_subset.py
    prepare_uav123.py
    run_tracker.py
    extract_telemetry.py
    generate_state_labels.py
    train_csc.py
    evaluate_csc.py
    evaluate_trackers.py
    run_ablation.py
    export_tables.py
    plot_results.py

  src/
    csc_uav_tracking/
      datasets/
      trackers/
      telemetry/
      states/
      models/
      metrics/
      control/
      evaluation/
      utils/

  data/
    raw/
    processed/
    results/

  notebooks/
  tests/
```

## Dataset Requirements

Supported datasets:

- LaSOT selected categories
- UAV123
- UAV123@10fps
- DTB70
- UAVDT
- VisDrone-SOT
- optionally GOT-10k
- optionally TrackingNet
- optionally COCO

For the first proof-of-concept, use selected LaSOT categories:

```text
car
person
truck
bus
bicycle
motorcycle
boat
dog
horse
bird
drone
airplane
```

The Hugging Face LaSOT mirror stores categories as zip files:

```text
car.zip
person.zip
truck.zip
...
drone.zip
airplane.zip
```

Do not assume category folders exist before extraction.

## Dataset Statistics (measured 2026-05-22)

All datasets reside at `~/uav-tracker-data/`.

| Dataset | Role | Sequences | Frames | Size |
|---|---|---|---|---|
| **LaSOT** (11 categories × 20) | CSC training | 220 | 581,150 | 43 GB |
| **GOT-10k val** | CSC training | 180 | 21,007 | ~2.1 GB (extracted) |
| **DTB70** | Validation / UAV adaptation | 70 | 15,777 | 1.4 GB |
| **VisDrone-SOT** | Validation / UAV adaptation | 35 | 32,922 | 11 GB |
| **UAV123** | Final test only | 123 | 113,476 | 13 GB |

**LaSOT categories present:** bicycle, bird, boat, bus, car, dog, drone, horse, motorcycle, person, truck (11 of 12; `airplane` not in this subset).

**GOT-10k layout:**
```
~/uav-tracker-data/GOT_10k/
├── val/   ← 180 sequences, full GT — use for CSC training
└── test/  ← 181 sequences, init bbox only — not for label generation
```

**Paths for loaders:**
```bash
export LASOT_DATA_ROOT=~/uav-tracker-data/LaSOT
export GOT10K_DATA_ROOT=~/uav-tracker-data/GOT_10k
export UAV_DATA_ROOT=~/uav-tracker-data   # covers DTB70, VisDrone-SOT, UAV123
```

**Protocol:**
```text
Train CSC:    LaSOT (11 cats × 20 seqs)  +  GOT-10k val (180 seqs)
Validate:     DTB70  +  VisDrone-SOT
Final test:   UAV123 (never used for training or threshold tuning)
```

## Dataset Loader Requirements

Each dataset loader should return a common `Sequence` object:

```python
@dataclass
class Sequence:
    name: str
    dataset: str
    frames: list[Path]
    gt_bboxes: np.ndarray  # shape [T, 4], xywh
    full_occlusion: Optional[np.ndarray] = None
    out_of_view: Optional[np.ndarray] = None
    attributes: dict = field(default_factory=dict)
```

Bounding boxes should use `xywh` format unless explicitly converted.

## Tracker Adapter Requirements

Implement tracker adapters behind one common API:

```python
class BaseTrackerAdapter:
    name: str

    def initialize(self, frame: np.ndarray, init_bbox: np.ndarray) -> None:
        ...

    def track(self, frame: np.ndarray) -> TrackerOutput:
        ...
```

`TrackerOutput` must support optional telemetry:

```python
@dataclass
class TrackerOutput:
    bbox: np.ndarray
    confidence: Optional[float] = None
    response_map: Optional[np.ndarray] = None
    search_region: Optional[np.ndarray] = None
    template_embedding: Optional[np.ndarray] = None
    search_embedding: Optional[np.ndarray] = None
    attention_map: Optional[np.ndarray] = None
    token_keep_ratio: Optional[float] = None
    active_layers: Optional[int] = None
    raw: dict = field(default_factory=dict)
```

The system must work even if a tracker exposes only `bbox` and no internal telemetry. Missing values should be represented as `None` and handled safely.

## Telemetry Extraction

CSC should be trained on tracker telemetry, not directly on frames.

Extract the following feature groups when available:

### Localization dynamics

- center displacement
- velocity
- acceleration
- scale change
- aspect ratio change
- center error if ground truth is available
- IoU if ground truth is available

### Confidence features

- raw confidence
- confidence EMA
- confidence delta
- normalized score

### Response-map features

- response peak
- response entropy
- APCE
- PSR

### Motion features

- motion entropy
- Kalman residual
- optical-flow residual
- frame-to-frame displacement

### Appearance features

- template-search cosine similarity
- embedding drift
- template age

### Detector / candidate agreement

- detector score
- IoU with detector candidate
- candidate ambiguity

### Transformer/token features

- token keep ratio
- active layers
- attention entropy

Telemetry should be saved as `.parquet` or `.csv` per sequence.

## Automatic Training Label Generation

CSC is a neural classifier. The rule-based logic below is not the final model and must not be used as the runtime predictor.

These rules are used only to generate supervised training labels from datasets where ground-truth bounding boxes are available.

During training, we can compute IoU between the predicted tracker bbox and the ground-truth bbox. During real-time inference, ground truth is unavailable, so CSC must predict the state from telemetry features only.

Training label generation uses:

- predicted bbox
- ground-truth bbox
- IoU
- tracker confidence
- occlusion flag
- out-of-view flag
- optional detector/candidate agreement

Runtime CSC input uses only telemetry:

- confidence
- response-map features
- bbox dynamics
- motion entropy
- appearance similarity
- detector agreement if available
- token/layer features if available
- temporal history

Runtime CSC must not use ground-truth IoU.

## State Labels

CSC predicts six states:

```python
CONFIRMED = 0
UNCERTAIN = 1
OCCLUDED = 2
LOST = 3
DISTRACTOR = 4
FALSE_CONFIRMED = 5
```

Initial label rules:

```python
if iou < tau_fail and confidence >= tau_conf:
    state = FALSE_CONFIRMED
elif iou >= 0.5 and confidence >= 0.55:
    state = CONFIRMED
elif occlusion_flag == 1 or out_of_view_flag == 1:
    state = OCCLUDED
elif iou < 0.1 and confidence < 0.4:
    state = LOST
elif 0.2 <= iou < 0.5:
    state = UNCERTAIN
elif iou < 0.3 and candidate_overlap_high:
    state = DISTRACTOR
else:
    state = UNCERTAIN
```

Important: `false_confirmed` should have priority over `lost` when confidence is high.

Default thresholds for first experiments:

```text
tau_fail = 0.2
tau_conf = 0.65
tau_lost = 0.1
tau_low_conf = 0.4
```

These are initial values. They must be validated and calibrated.

## CSC Model

Implement at least two model variants:

1. `CSC_MLP`
2. `CSC_TCN`

Optionally implement:

3. `CSC_GRU`

Recommended first model:

```text
Model: TCN
Temporal window: 16 frames
Hidden dim: 64
Output classes: 6
Loss: weighted focal cross entropy
```

Class weights:

```text
confirmed: 1.0
uncertain: 1.2
occluded: 1.5
lost: 1.8
distractor: 2.0
false_confirmed: 2.5
```

Reason: `false_confirmed` is rare but safety-critical.

## Real-Time Calibration

Implement lightweight online calibration, not full online retraining.

Allowed:

- EMA normalization of tracker-specific confidence
- temperature scaling
- adaptive threshold calibration
- safe pseudo-label calibration only on stable confirmed frames

Forbidden:

- updating CSC on uncertain/lost/occluded/distractor/false_confirmed states
- tuning final results on UAV123
- silently changing thresholds during final evaluation

## Metrics

Keep standard tracking metrics:

- Success AUC
- Precision@20
- Center Location Error
- FPS
- Latency
- FLOPs or estimated compute

Add CSC metrics:

### False Confirmed Rate

```text
FCR = N_false_confirmed / N_total
```

### False Confirmed Duration

```text
FCD = mean length of continuous false_confirmed segments
```

### Time to False Confirmation

```text
TTFC = t_first_false_confirmed - t_last_confirmed
```

### Unsafe Update Rate

```text
UUR = updates_during_false_confirmed / all_template_updates
```

### Recovery@K

```text
Recovery@K = recovered_failure_episodes_within_K_frames / total_failure_episodes
```

Use `K = 30` by default.

### State-conditioned AUC

Calculate AUC separately for frames belonging to each state.

### State Transition Matrix

Calculate transitions such as:

```text
confirmed -> uncertain
uncertain -> occluded
occluded -> false_confirmed
false_confirmed -> lost
lost -> confirmed
```

## Evaluation Modes

### Evaluation-only mode

CSC diagnoses tracker states but does not modify tracker output.

Use this for all trackers.

Expected output table:

```text
Tracker | AUC | Precision@20 | FPS | FCR | FCD | Lost Rate | Recovery@30
```

### Control mode

CSC modifies tracker behavior if hooks are available.

Supported actions:

- freeze template update
- reduce token pruning
- increase token keep ratio
- activate deeper layers
- widen search region
- trigger detector verification
- trigger reinitialization

Control policy:

```text
confirmed:
  normal tracking, aggressive pruning allowed

uncertain:
  reduce pruning, cautious update

occluded:
  freeze template update, use motion prediction

lost:
  wider search or re-detection

distractor:
  reject candidate, verify with detector/appearance

false_confirmed:
  block update, reduce pruning, force verification
```

## Experiments

### MVP Experiment

Datasets:

```text
Train: LaSOT selected categories
Validation: DTB70 or VisDrone-SOT
Final test: UAV123
```

Trackers:

```text
KCF
one modern pretrained tracker
```

States:

```text
confirmed
uncertain
lost
false_confirmed
```

Metrics:

```text
AUC
Precision@20
FCR
FCD
Recovery@30
Macro-F1
F1 false_confirmed
```

### Full Experiment

Datasets:

```text
Train: LaSOT selected categories + GOT-10k
Validation: DTB70 + UAVDT + VisDrone-SOT
Test: UAV123 + UAV123@10fps
```

Trackers:

```text
SGLATrack
ORTrack
Aba-ViTrack
ParallelTracker
SiamHFFT
DSATrack
```

If some tracker is difficult to integrate, use KCF/OSTrack/MobileTrack as fallback.

### Ablation Study

Run:

```text
confidence only
confidence + bbox dynamics
+ motion entropy
+ appearance consistency
+ detector agreement
full CSC + calibration
```

Report:

```text
Macro-F1
F1 false_confirmed
FCR after control
AUC
FPS
```

## Plotting and Reports

Generate:

- confusion matrix
- FCR by tracker
- FCR by scene group
- state transition matrix
- failure timeline plots
- examples of false_confirmed episodes
- AUC/FPS/FCR comparison table
- ablation table

All result tables must be saved to:

```text
data/results/tables/
```

All figures must be saved to:

```text
data/results/figures/
```

## Coding Style

Use Python 3.10+.

Prefer:

- clear dataclasses
- typed function signatures
- config-driven scripts
- deterministic outputs
- safe file handling
- no hardcoded absolute paths
- tqdm for long loops
- structured logging

Avoid:

- hidden global state
- silent exception swallowing
- notebook-only implementation
- mixing dataset preparation, training, and evaluation in one giant script
- modifying original raw datasets

## Testing Requirements

Add tests for:

- IoU calculation
- bbox conversion
- state label rules
- FCR calculation
- FCD calculation
- Recovery@K calculation
- missing telemetry handling
- dataset sequence loading

## Important Implementation Notes

If Hugging Face LaSOT stores categories as `.zip` files, download `category.zip`, extract it, then validate that image files exist.

Expected extracted structure:

```text
LaSOT_selected_categories/
  car/
    car-1/
      img/
      groundtruth.txt
      full_occlusion.txt
      out_of_view.txt
```

If no image files are found after extraction, fail loudly and print a clear diagnostic message.

## Deliverables

The implementation should produce:

1. dataset download/preparation scripts
2. dataset loaders
3. tracker adapter interface
4. telemetry extraction pipeline
5. automatic state-label generation
6. CSC model training
7. CSC evaluation
8. state-aware metrics
9. reports/tables/figures
10. optional control policy

## First Task for Claude

Start by implementing the dataset and telemetry foundation.

Concrete first tasks:

1. Fix or create `scripts/download_lasot_categories.py` to download category zip files from Hugging Face.
2. Implement `Sequence` dataclass.
3. Implement `LaSOTDataset` loader for extracted category folders.
4. Implement bbox utilities: `xywh_to_xyxy`, `xyxy_to_xywh`, `iou_xywh`, `center_error`.
5. Implement initial state label generation for:
   - confirmed
   - uncertain
   - lost
   - false_confirmed
6. Add unit tests for bbox utilities and label rules.
7. Add a CLI command that loads one LaSOT sequence and prints:
   - number of frames
   - number of ground-truth boxes
   - first bbox
   - available occlusion/out-of-view flags

Do not start model training before dataset loading and label generation are tested.
