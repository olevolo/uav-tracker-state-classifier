# Current Project Phase Baseline

**Date:** 2026-05-20  
**Purpose:** current status snapshot for Codex/staff review of SALT-RD implementation.  
**Last reviewed:** Phase 2A e-process implemented; 6 P1-P3 bugs fixed; 62 tests green on 2026-05-20.

## Role For Review

Codex is acting as Staff Computer Vision + AI/ML Architect/Engineer with scientific review responsibilities.

Review focus:
- scientific validity of SALT-RD;
- no label leakage or self-teaching;
- correct sequence-level evaluation protocol;
- calibrated dynamicity/reliability prediction metrics;
- engineering reproducibility and clean architecture boundaries.

## Current Phase

The project is in **Phase 2A — e-process sequential alerts implemented; Phase 2B (DAM memory) is next**.

The intended research direction is:

> Proactive Tracking-Risk Dynamicity for Failure-Aware Real-Time UAV Single Object Tracking.

SALT-RD is not meant to be a new tracker backbone. It should be a small neural dynamicity/reliability/recovery controller on top of frozen SGLATrack/SALT v3 that proactively predicts tracking-risk, false-confirmed drift, real tracking failure, recovery readiness, and compute need from GT/teacher-derived labels.

## Current Repo State

- Active branch: `main`.
- v0/v1/v2 trained and evaluated; all P1/P2 bugs fixed; e-process implemented.
- `saltr/src/salt_r/` contains: `collect_features.py`, `model.py`, `train.py`, `eval.py`, `policy.py`, `eprocess.py`.
- `saltr/results/` contains: `eval_val_v2.json`, `preds_val_v2.json`, `policy_val_v2.json`, `eprocess_val_v2.json`, `eprocess_sweep_v2.csv`.
- `saltr/data/`: `salt_rd_v0.npz` (228 seq, ~161k frames), `salt_rd_v1_labels.npz`, `salt_rd_v2_labels.npz`.

## Current Planning Documents

- `HANDOFF_NEXT.md` is the current final implementation plan.

## Current SALT-RD Status

### v2 Model Results (val split, calibrated)

| Head | AUROC | AUPRC | Comment |
|---|---:|---:|---|
| false_confirmed | 0.884 | 0.336 | GRU +37.9pp vs best baseline |
| imminent_failure_dynamic | 0.902 | 0.323 | Strong short-horizon signal |
| imminent_failure_dynamic_10 | **0.897** | **0.329** | 10-frame horizon holds |
| imminent_failure_dynamic_20 | **0.889** | **0.339** | 20-frame horizon holds |
| failure_in_5 | 0.853 | 0.011 | Ranking good, AUPRC low (sparse) |

**Policy replay (v2 vs v0):** template_corruption −44%, wrong_reinit −33%.

**ECE(false_confirmed)=0.316, AUROC(needs_full_compute)=0.648** — both blocking GO gates.

### e-Process Phase 2A Results (formal mode, alpha=0.10, epsilon=0.5)

| Metric | e-process | Raw P(ifd)>0.5 | Target |
|---|---:|---:|---:|
| Median lead time | **11.5 frames** | — | ≥ 3 |
| Event recall | 0.062 | 0.750 | ≥ 0.60 |
| FA per 1000 frames | **0.21** | 171 | ≤ 100 |
| Seq-level FAR | 0.167 | — | ≤ 0.10 |

**Diagnosis:** e-process precision is 44% (4 TP / 9 total alerts) vs baseline 1% (48/4682). Lead time is excellent (11.5f). Bottleneck is **recall** — the formal martingale is too conservative for the current risk score quality. Adding DAM memory features (Phase 2B) should strengthen the signal so e-process accumulates evidence faster.

### Bug Fixes Completed (this session)

| Bug | File | Fix |
|---|---|---|
| P1a — ifd10/20 labels counted already-failed frames | collect_features.py | Added `iou_trace[t] >= 0.5` and `len==horizon` guards |
| P1b — lead_time hardcoded to ifd5 and 6-frame window | eval.py | Parameterized `label_name`, `horizon` |
| P2a — policy docs didn't mention v2 head gap | policy.py | Added NOTE comment |
| P2b — fake AUROC 0.5 for "correct" label in tables | eval.py | Added `model_predicted: false` + note field |
| P2c — recompute_labels_v2() silently accepted v0 input | collect_features.py | Added v1 schema validation |
| P3 — train log said AUPRC(fc) but it was composite | train.py | Renamed to "validation selection score" |

### Test Suite

62 passed (33 original + 29 eprocess unit tests).

## Staff Review Red Lines (unchanged from HANDOFF_NEXT.md)

- Do not train on `_decide_state()`, TSA states, old scene labels.
- Do not claim compute/FPS without oracle labels.
- Do not calibrate on train split.
- Do not use diagnostic sequences in train.
- Do not tune GO gates to get a better verdict.
- Do not make `hard_dynamic_scene_v2` central again.
- Do not use `pt_inside_gt_ratio` as runtime feature (GT-relative, teacher-only).

## Role For Review

Codex is acting as Staff Computer Vision + AI/ML Architect/Engineer with scientific review responsibilities.

Review focus:
- scientific validity of SALT-RD;
- no label leakage or self-teaching;
- correct sequence-level evaluation protocol;
- calibrated dynamicity/reliability prediction metrics;
- engineering reproducibility and clean architecture boundaries.

## Current Phase

The project is in **Phase 1c calibration / policy operating-point validation**.

The intended research direction is:

> Proactive Tracking-Risk Dynamicity for Failure-Aware Real-Time UAV Single Object Tracking.

SALT-RD is not meant to be a new tracker backbone. It should be a small neural dynamicity/reliability/recovery controller on top of frozen SGLATrack/SALT v3 that proactively predicts tracking-risk, false-confirmed drift, real tracking failure, recovery readiness, and compute need from GT/teacher-derived labels.

The current reviewed plan is:

1. Create `saltr/` as the new implementation area.
2. Add `FROZEN.md` and freeze legacy/current `src/uav_tracker/` after a small, explicit freeze-prep change.
3. Add config gates, not deletions, for CE / dynamic / velocity-drift behavior.
4. Add stable score-map statistics to tracker telemetry.
5. Implement SALT-RD v0 in `saltr/src/salt_r/` only.
6. Treat Phase 2 runtime wrapper as scaffold until offline policy replay shows bounded regret.
7. Next: cleanup, calibration, LODO, and policy operating-point sweep.

## Current Repo State

- Active branch: `main`.
- Phase 0 commit: `ecfcb0f` — all Phase 0 items delivered.
- SALT-RD implementation commits through `7d9b693` are present on `main`.
- `saltr/src/salt_r/` exists with implemented `collect_features.py`, `model.py`, `train.py`, `eval.py`, `policy.py`, and `integrate.py`.
- `FROZEN.md` committed, policy freeze active.
- Config gates wired: `enable_ce` in sglatrack.py, `enable_velocity_drift` in target_state_assessor.py.
- `TrackState` extended with `score_map_stats` and 4 other telemetry fields.
- `enable_salt_rd` is read by `SALTRDRunner.from_config()` only; frozen `SALTRunner` production path still ignores it.
- `saltr/data/salt_rd_v0.npz` exists: 228 sequences, ~161k frames.
- `saltr/checkpoints/saltrd_best.pt` exists: checkpoint metadata epoch 5.
- `saltr/checkpoints/eval_val.json` contains fixed eval metrics after the double-sigmoid fix.

## Current Planning Documents

- `HANDOFF_NEXT.md` is the current final implementation plan.
- `THOUGHTS.md` contains staff-level commentary, paper summaries, and corrected strategy.
- `ANALYSIS.md` contains SALT v3 technical results and ablations.
- `papers/README.md` indexes the key papers.

Final-plan verdict:

- Direction is scientifically stronger after the 2026-05-19 update: neural scene dynamicity plus reliability/failure prediction, not another UAV tracker.
- Main novelty remains false-confirmed detection: high APCE/confidence but low IoU / wrong identity.
- The plan now correctly prioritizes GT/teacher-derived dynamicity labels, GT IoU labels, sequence-level splits, AUPRC, calibration, NT2F, bootstrap CI, AUC-vs-GFLOPs policy replay, and negative-result policy.
- Implementation must stay disciplined; the biggest remaining risk is accidentally rebuilding old TSA/rule training under a new name.

## Current Technical Baseline

SALT v3 exists as the current experimental runtime:

- `src/uav_tracker/salt_runner.py`
- `src/uav_tracker/trackers/sglatrack.py`
- `src/uav_tracker/ml/tsa/`
- `src/uav_tracker/detectors/visdrone_yolo26m.py`
- `src/uav_tracker/detectors/rtdetr.py`
- `src/uav_tracker/datasets/visdrone_sot.py`
- `src/uav_tracker/datasets/dtb70.py`

Last verified in this session:

- SALT-RD targeted tests passed: `26 passed`.
- `compileall saltr/src/salt_r` passed.
- `eval.py --predictions-output` works.
- `policy.py` replay works on val predictions.
- Earlier full-suite report in latest implementation notes: `200 passed`.

## Current SALT-RD Status

SALT-RD is in **Phase 1c-Calib / policy operating-point validation** (2026-05-20).

### Completed

- Phase 0 freeze/config/telemetry scaffold (commit `ecfcb0f`)
- `collect_features.py` collection loop implemented (commit `64f307b`)
- flat-sequence dynamicity label bug fixed (ABS_MIN_MOTION floor + strict `>`)
- sequence-key collision fixed (compound keys `dataset/seq_name`)
- `_TruncatedSequence` added — `max_frames` now caps runner work before execution
- GT-derived dynamic labels — `_compute_bbox_motion_arrays` uses GT bboxes, not predicted boxes
- dataset root autodetection — `_get_dataset_loaders` passes `root=None`
- Papers competitive analysis written to `papers/README.md`
- `model.py` / `train.py` / `eval.py` implemented and checkpoint-compatible.
- `policy.py` implemented with `TrackerAction`, thresholds, and replay.
- `integrate.py` implemented as runtime wrapper/scaffold, not deployment-validated.
- Fixed eval path: no double-sigmoid, named `HEAD_NAMES` prediction mapping, predictions JSON export.
- Fixed policy replay IoU key loading (`iou_trace/` prefix stripping).
- Fixed `--dry-run` so it no longer loads `SALTRunner`.
- Added SALT-RD unit tests for collect/model/eval/policy/integrate.

### Current Results

- NPZ: `saltr/data/salt_rd_v0.npz` — 228 sequences, ~161k frames.
- Global `false_confirmed` base rate: ~8.44%.
- Val split: 49 sequences.
- `false_confirmed`: AUROC `0.884`, AUPRC `0.331`, ECE `0.320`, recall@5%FPR `0.445`.
- `failure_in_5`: AUROC `0.863`, but AUPRC only `0.010` due very low base rate.
- `hard_dynamic_scene`: AUROC `0.638`, below GO.
- `needs_full_compute`: AUROC `0.641`, below GO.
- GO/NO-GO: **BORDERLINE**.
- Policy replay val: `wrong_reinit_rate=0.216`, `template_corruption_rate=0.097`, `compute_cheap_rate=0.000`.
- Diagnostic split remains weak: `false_confirmed` AUROC around `0.604`, AUPRC around `0.279` with base around `20.3%`.

### Current Blockers / Cleanup Before Paper Metrics

- Fix `policy.py` summary accounting: report `n_evaluated` and `n_skipped`, not only `len(all_probs)`.
- Strengthen `test_eval_does_not_double_sigmoid` to call real `eval._run_inference()`.
- Fix `run_phase1.sh --datasets` usage/parser mismatch.
- Move or ignore generated JSON artifacts under `saltr/checkpoints/`; use `saltr/results/` for versioned publishable snapshots.
- Add calibration. Since model currently returns probabilities, either add `return_logits=True` or calibrate through `logit(p)` before temperature scaling.
- Run LODO evaluation before any generalization claim.
- Do not claim compute/FPS win until `needs_full_compute` has oracle labels and a bounded-regret policy sweep.

Existing legacy training code still reflects the old rule/scene-label path:

- `scripts/train_tsa_classifier.py`
- `scripts/generate_ml_labels.py`
- `src/uav_tracker/datasets/uav123_ml.py`

These should not become the SALT-RD training path unless rewritten to use GT/teacher-derived labels only.

## Final Plan Review Notes

The final `HANDOFF_NEXT.md` is acceptable as the next execution plan with these review constraints:

- **Freeze semantics:** `src/uav_tracker/` may receive one final freeze-prep change for config gates and `score_map_stats`. After `FROZEN.md`, Claude should not continue feature work there.
- **Canonical SALT-RD path:** new Phase 1 code belongs in `saltr/src/salt_r/`, not in legacy `scripts/`. The canonical collector is `saltr/src/salt_r/collect_features.py`.
- **Config gates must be real:** adding `enable_ce`, `enable_dynamic`, and `enable_velocity_drift` to YAML is insufficient unless runtime code reads and honors them.
- **Do not delete ablation paths:** CE, DYNAMIC, and VelocityDrift should be disabled/config-gated, not removed, because the paper needs ablations.
- **Diagnostic split counts must be corrected:** `bike2`, `Gull2`, `Sheep1`, `StreetBasketball1`, and `uav0000164` must be removed from train/val before reporting split counts. The plan's nominal counts cannot all remain true if diagnostics are excluded.
- **Dataset names must be validated:** `uav0000164` may not be the exact sequence key exposed by the local VisDrone loader. Claude must verify actual sequence names before hardcoding diagnostics.
- **APCE units are now conceptually fixed:** the plan uses `apce_norm > 100/256.0`. Implementation must store both `raw_apce` and `apce_norm`, or clearly document the one used.
- **Dynamicity is not scene classification:** target labels must come from target/camera motion, future IoU, flow residual, point teacher, or full-vs-cheap oracle replay. Do not train on old scene labels.
- **Compute claims require oracle replay:** any GFLOPs/FPS claim from `needs_full_compute` must compare full vs cheap/pruned/bypass outputs and report AUC-vs-GFLOPs regret.
- **Bootstrap CI must be sequence-level:** frame-level bootstrap is invalid because adjacent frames are correlated.
- **Negative result policy is good, but pre-register attempts:** the "3+ feature sets" limit should be explicit before experiments to avoid p-hacking.
- **LODO eval is required for claims of generalization:** at minimum, report train UAV123+VisDrone -> test DTB70.

## Staff Review Red Lines

Block the change if Claude:

- trains SALT-RD using `scene_class`, `_decide_state()`, TSA states, APCE rules, or any rule-generated labels as the target;
- treats neural dynamicity as old `STATIC/DYNAMIC/OCCLUDED` scene classification;
- uses frame-level random split instead of sequence-level split;
- reports accuracy without AUROC/AUPRC and base-rate context for rare labels;
- treats false-confirmed detection as solved by APCE thresholding;
- removes ablation paths instead of config-gating them;
- mutates frozen SALT v3 code without clear reason after the policy freeze;
- claims scientific improvement without leave-one-dataset-out or at least named hard-negative diagnostics;
- reports diagnostic-suite performance after training on those same diagnostic sequences;
- implements bootstrap confidence intervals over frames instead of sequences;
- adds config fields that are never read by runtime code;
- silently changes SGLATrack/SALT v3 behavior after `FROZEN.md`.

## Known Unit/Schema Issue To Watch

The older plan mixed APCE units:

```python
false_confirmed[t] = iou[t] < 0.2 AND apce[t] > 100/256
```

The final plan clarifies this as normalized APCE:

```python
false_confirmed = iou < 0.2 and apce_norm > 100 / 256.0
```

Implementation requirement:

```python
raw_apce = TrackState.apce
apce_norm = raw_apce / 256.0
```

The feature schema must explicitly name raw vs normalized APCE, and `feature_names[0]` must match what `false_confirmed` reads.

## Required Phase 1 Output

Before moving to SALT-Match, DINO/SAM/CoTracker teachers, LoRAT, or tracker fine-tuning, Phase 1 should produce:

- versioned feature NPZ with feature names, units, sequence names, dataset names, frame indices, split labels, IoU trace, and target labels;
- SALT-RD v0 temporal model;
- labels for false-confirmed, failure-in-5, recoverable, target-dynamic, camera-dynamic, hard-dynamic-scene, and needs-full-compute;
- metrics: AUROC, AUPRC, ECE, Brier, false-confirmed recall at 5% FPR, dynamicity AUROC/AUPRC, NT2F, policy replay metrics, and base rates;
- named hard-negative diagnostic results for `uav0000164`, `bike2`, `Gull2`, `Sheep1`, `StreetBasketball1`;
- leave-one-dataset-out results, especially train on UAV123+VisDrone and test on DTB70;
- sequence-level bootstrap 95% confidence intervals;
- GO/NO-GO decision based on the thresholds in `HANDOFF_NEXT.md`.

## Current Paper Context

The final plan and `THOUGHTS.md` use these papers as positioning anchors:

- MSTFT 2026: backbone/SOTA comparison, dynamic template fusion, triple safety verification.
- MATA 2026: NT2F, ego-motion residual, embedded evaluation protocol.
- OOTU 2025: bbox localization uncertainty; compare calibration but distinguish from identity failure.
- PTDT 2026: point-tracking-guided dynamic token/template update gates.
- CoTracker3: offline point tracking teacher for point consistency features.
- LoRAT 2024: Phase 6 fallback for parameter-efficient domain adaptation if SALT-RD v0 fails.
- UTPTrack 2026: token pruning competitor; SALT-RD should control pruning, not claim pruning novelty.
- ABTrack 2024/2025 and UncL-STARK 2026: adaptive compute/uncertainty-guided depth baselines.
- BDTrack 2025 and LGTrack 2026: UAV dynamic/efficient tracking competitors.
