# Current Project Phase Baseline

**Date:** 2026-05-19  
**Purpose:** baseline snapshot before Claude starts implementation work. Use this file to compare what changed.  
**Last reviewed:** final `HANDOFF_NEXT.md` and `THOUGHTS.md` review on 2026-05-19.

## Role For Review

Codex is acting as Staff Computer Vision + AI/ML Architect/Engineer with scientific review responsibilities.

Review focus:
- scientific validity of SALT-RD;
- no label leakage or self-teaching;
- correct sequence-level evaluation protocol;
- calibrated dynamicity/reliability prediction metrics;
- engineering reproducibility and clean architecture boundaries.

## Current Phase

The project is in **Phase 1a implementation (data collection) SALT-RD migration phase**.

The intended research direction is:

> Proactive Tracking-Risk Dynamicity for Failure-Aware Real-Time UAV Single Object Tracking.

SALT-RD is not meant to be a new tracker backbone. It should be a small neural dynamicity/reliability/recovery controller on top of frozen SGLATrack/SALT v3 that proactively predicts tracking-risk, false-confirmed drift, real tracking failure, recovery readiness, and compute need from GT/teacher-derived labels.

The final reviewed plan is:

1. Create `saltr/` as the new implementation area.
2. Add `FROZEN.md` and freeze legacy/current `src/uav_tracker/` after a small, explicit freeze-prep change.
3. Add config gates, not deletions, for CE / dynamic / velocity-drift behavior.
4. Add stable score-map statistics to tracker telemetry.
5. Implement SALT-RD v0 in `saltr/src/salt_r/` only.
6. Stop after Phase 1 unless GO metrics are reached.

## Current Repo State

- Active branch: `main`.
- Phase 0 commit: `ecfcb0f` — all Phase 0 items delivered.
- `saltr/src/salt_r/` exists with collect_features.py, model/train/eval/policy/integrate stubs.
- `FROZEN.md` committed, policy freeze active.
- Config gates wired: `enable_ce` in sglatrack.py, `enable_velocity_drift` in target_state_assessor.py.
- `TrackState` extended with `score_map_stats` and 4 other telemetry fields.
- collect_features.py collection loop: implemented in this session (Phase 1a).

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

- `uav-tracker doctor` passed.
- Unit tests passed: `174 passed, 5 deselected`.

## Current SALT-RD Status

SALT-RD is in **Phase 1a data-collection implementation** (2026-05-19).

### Completed

- Phase 0 freeze/config/telemetry scaffold (commit `ecfcb0f`)
- `collect_features.py` collection loop implemented (commit `64f307b`)
- flat-sequence dynamicity label bug fixed (ABS_MIN_MOTION floor + strict `>`)
- sequence-key collision fixed (compound keys `dataset/seq_name`)
- `_TruncatedSequence` added — `max_frames` now caps runner work before execution
- GT-derived dynamic labels — `_compute_bbox_motion_arrays` uses GT bboxes, not predicted boxes
- dataset root autodetection — `_get_dataset_loaders` passes `root=None`
- Papers competitive analysis written to `papers/README.md`
- `model.py` / `train.py` / `eval.py` / `policy.py` / `integrate.py` — implemented (Phase 1b)

### Still blocking before real NPZ collection

- Verify `--dry-run` succeeds on all 3 datasets with autodetect roots
- Confirm label base rates after first NPZ: `false_confirmed` expected 1-3%

### Not produced yet

- `saltr/data/salt_rd_features.npz` — collection loop not yet executed
- trained SALT-RD checkpoint
- SALT-RD eval report with AUROC/AUPRC/ECE/Brier/NT2F/dynamicity/policy metrics

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
