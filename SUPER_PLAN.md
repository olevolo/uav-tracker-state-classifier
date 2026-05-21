# SUPER PLAN — SALT-RD As The Only Learned Tracking Controller

Date: 2026-05-21

## Executive Decision

We are not migrating from TSA to SALT-RD gradually. We are deleting TSA as a product concept.

Final production system:

```text
SGLATrack raw telemetry
        -> SALT-RD EvidenceExtractor
        -> SALT-RD PolicyNet
        -> SALT-RD Controller
        -> SGLATrack / Detector / TemplateManager actions
```

There must be no production `TargetStateAssessor`, no `TargetState`, no
`CONFIRMED / OCCLUDED / LOST / DYNAMIC / DISTRACTOR_RISK`, no APCE threshold state
machine, no `5 LOST frames`, and no TSA fallback path.

APCE, PSR, entropy, score-map peaks, bbox motion, detector candidates, and template
history remain valid numeric evidence. They are not decisions.

## Success Criterion

This project succeeds only when the following causal chain is demonstrated:

```text
SALT-RD predicts action
    -> action fires
    -> tracker behavior changes
    -> bbox trajectory changes
    -> hard-scene AUC improves
    -> full-set AUC also improves
```

Good AUROC, low template corruption, or a clean advisory log is not enough.

Required final gates:

| Gate | Required result |
|---|---:|
| Hard subset AUC delta | >= +0.10 (oracle ceiling ~+0.08 reinit-only on 14 seqs; all 18 hard seqs + compute routing makes +0.10–+0.12 realistic) |
| Full UAV123 AUC delta | >= +0.010 (oracle full-set reinit gain is +0.025; positive improvement required, not just no-regression) |
| Action coverage on hard subset | changed bbox frames > 0.5% |
| Wrong reinit rate | not worse than current baseline |
| Template corruption | not worse than current baseline |
| Production TSA references | zero in runtime code |
| SALT-RD -> TargetState adapter | absent |
| Rule threshold action triggers | absent from production |
| Checkpoint feature schema | `saltrd_v3_no_tsa_no_flow` with flow indices dropped |

## Current Work Review

### What Aligns With This Plan

1. **No-flow ablation is valuable.**
   The recent `v2_no_flow` result is a real architecture finding: offline Farneback
   flow created train/runtime mismatch. Production SALT-RD must use a no-flow feature
   schema. Flow can remain offline teacher/debug only.

2. **Oracle action audit is finally the right type of experiment.**
   `saltr/results/oracle_action_audit.json` evaluates actions by AUC utility, not just
   classifier metrics. This is exactly the right direction.

   Current oracle result:

   | Action | Hard oracle gain | Full oracle gain | Decision |
   |---|---:|---:|---|
   | reinit | +0.0834 | +0.02463 | build learned policy |
   | search_expand | +0.00408 | +0.00049 | kill |
   | template_update | +0.00109 | +0.00274 | kill |
   | center_freeze | +0.00003 | +0.00215 | kill |

   Interpretation: the next AUC-improving action should be learned reinitialization,
   not CE, not search expansion, not center freeze, and not template update.

3. **SGLATrack telemetry work is useful.**
   APCE, PSR, entropy, score-map stats, top-k candidates, and internal embeddings are
   still useful as evidence. The bug was treating them as rules or assuming SGLATrack
   token embeddings are good identity features.

4. **Entropy should stay.**
   Removing entropy hurt diagnostic false-confirmed detection. Entropy is not the bug.
   It stays as an evidence feature.

5. **CE is not broken anymore, but it is secondary.**
   CE works technically after the Q1/Q2/Q4 fixes, but its measured compute gain is small
   and it does not solve hard-scene AUC. It remains only as a future learned compute
   actuator, not a central contribution.

### What Conflicts With This Plan

1. **Runtime still depends on TSA.**
   `src/uav_tracker/salt_runner.py` still constructs `TargetStateAssessor`, imports
   `TargetState`, routes compute by previous TSA state, and triggers recovery via
   OCCLUDED/LOST counters.

2. **`SGLATracker.update_with_state()` is a legacy control API.**
   `src/uav_tracker/trackers/sglatrack.py` still maps state integers to compute and
   search behavior. This must become `update_with_action(action)`.

3. **`saltr/src/salt_r/advisor.py` is still threshold/rule-based.**
   `stage3_policy(tsa_state_int)` still consumes TSA state and uses handcrafted gates.
   It must be replaced by model action outputs.

4. **`saltr/src/salt_r/interventions.py` is rule-based.**
   `decide_intervention()` contains thresholds for false-confirmed, ifd10, ifd20,
   memory margin, and Kalman residual. This is not allowed in the final architecture.
   It can be used only for retrospective analysis, not production control.

5. **`HANDOFF_NEXT.md` still describes TSA compatibility/fallback.**
   It says not to remove TSA directly and describes Shadow -> Advisory/Veto ->
   Primary Controller. This is now obsolete.

6. **Data artifact state is broken.**
   `saltr/data` is currently missing. The new oracle audit excluded four key hard
   sequences (`uav123/bike2`, `dtb70/Gull2`, `dtb70/Sheep1`, `dtb70/StreetBasketball1`).
   Any final decision must rerun oracle/action experiments on canonical data with all
   required sequences available.

### Latest Commit Review — 2026-05-21

Reviewed current `main` at `6737c91` plus untracked `SUPER_PLAN.md` and
`saltr/results/oracle_action_audit.json`.

#### Accepted Findings

1. **No-flow result is valid and important.**

   `saltr/results/ablation_entropy_flow_0521.json` shows:

   | Run | Val fc AUROC | Val fc AUPRC | Diagnostic fc AUROC | Diagnostic fc AUPRC |
   |---|---:|---:|---:|---:|
   | v2_retrained baseline | 0.8854 | 0.3381 | 0.5978 | 0.2813 |
   | v2_no_entropy | 0.8840 | 0.3334 | 0.5829 | 0.2710 |
   | v2_no_flow | 0.8831 | 0.3504 | 0.6967 | 0.3806 |
   | v2_no_entropy_no_flow | 0.8814 | 0.3470 | 0.6905 | 0.3757 |

   Conclusion: entropy stays; flow is removed from production feature schema.

2. **Oracle action audit is the right direction.**

   Current untracked `saltr/results/oracle_action_audit.json` gives:

   | Action | Hard oracle gain | Full oracle gain | Decision |
   |---|---:|---:|---|
   | reinit | +0.0834 | +0.02463 | build learned policy |
   | search_expand | +0.00408 | +0.00049 | kill |
   | template_update | +0.00109 | +0.00274 | kill |
   | center_freeze | +0.00003 | +0.00215 | kill |

   This confirms the next AUC path is learned reinitialization, not template update,
   not center freeze, not search expansion, and not CE.

### Action Oracle Verdict

This is the current action-priority decision. Do not reopen these branches without a
new oracle audit that contradicts the table.

| Action branch | Verdict | Current oracle result | Why |
|---|---|---:|---|
| center-freeze | KILL | ~0.000 hard AUC gain | By the time false-confirmed is visible, the real target has already moved; freezing the last center preserves the wrong trajectory. |
| reinit | BUILD | +0.083 hard AUC, +0.0246 full AUC | Strongest action signal and no harmful sequences in current audit. This is the only immediate AUC path. |
| template update | KILL for now | +0.001 hard AUC, many harmful cases | Does not explain hard-scene AUC gap; keep dynamic template updates disabled until a counterfactual head proves value. |
| search expand | KILL for now | +0.004 hard AUC, harmful sequences | Expansion dilutes small UAV target resolution and does not recover drift reliably. |

Implementation consequence:

1. Do not implement center-freeze policy.
2. Do not implement search-expand policy.
3. Do not implement template-update policy yet.
4. Build only the learned reinit/candidate policy first.
5. If learned reinit fails, fix candidate generation/identity scoring before returning to
   other actions.

3. **Cleanup of broken symlinks is directionally good.**

   Commit `6737c91` removed broken `saltr/data` and `saltr/checkpoints/v2_corrected`
   symlinks. This avoids circular symlink bugs, but it did not restore canonical data.

#### Blocking Issues Found

1. **Production checkpoint is not the no-flow checkpoint.**

   `saltr/checkpoints/production/saltrd_best.pt` currently has:

   ```text
   drop_feature_indices = None
   label_schema = v2
   feature_names include flow columns 22..27
   ```

   Therefore current production still represents the older full-flow risk model, not
   the winning `v2_no_flow` ablation. Do not call current production checkpoint final.

   Required fix:

   - produce or restore `saltr/checkpoints/production_no_flow/saltrd_best.pt`;
   - checkpoint metadata must include `drop_feature_indices=[22,23,24,25,26,27]`;
   - update production only after eval/advisor/runtime auto-apply that metadata.

2. **No-flow is a CLI ablation, not a runtime contract yet.**

   `train.py` writes `drop_feature_indices` into checkpoint metadata, but `eval.py`
   only zeros features when `--drop-features` is manually passed. Runtime advisor also
   does not yet enforce checkpoint drop indices as the single source of truth.

   Required fix:

   - `eval.py` must read checkpoint `drop_feature_indices` and apply them by default;
   - runtime `EvidenceExtractor` / controller must apply checkpoint feature schema;
   - CLI overrides must fail if they conflict with checkpoint metadata in production mode.

3. **`saltr/data` is missing.**

   `ls saltr/data` currently fails. This blocks canonical training, oracle relabeling,
   and full hard-subset audit.

   Required fix:

   - restore canonical NPZ artifacts;
   - do not use symlinks that point to themselves;
   - add `salt_r.check_artifacts` to fail fast when expected NPZs are missing.

4. **Oracle audit is incomplete.**

   `oracle_action_audit.json` reports:

   ```text
   hard_subset_available = 14
   hard_subset_missing = 4
   missing = uav123/bike2, dtb70/Gull2, dtb70/Sheep1, dtb70/StreetBasketball1
   ```

   These are exactly the hard identity/reinit cases we care about. Current oracle
   gain is promising but not final.

   Required fix:

   - rerun oracle audit after restoring `saltr/data`;
   - Table F must include those missing sequences;
   - if reinit gain disappears when they are included, revisit candidate generation.

5. **Oracle audit recommendation says "rule-based reinit".**

   Current `phase4_recommendation.next` says:

   ```text
   Implement conservative rule-based reinit (Phase 5)
   ```

   This conflicts with this plan. The correct next step is:

   ```text
   Generate oracle/counterfactual reinit labels and train learned reinit policy.
   ```

6. **`saltr/README_PROD.md` is stale.**

   It still describes `SALTRDAdvisor`, `should_block_template_update()`,
   `tracker.update_with_state()`, and production checkpoint diag AUROC `0.598`.
   That is the old advisory/veto architecture, not the final plan.

   Required fix:

   - update README after the new contracts exist;
   - until then mark it as archived/stale;
   - do not use README_PROD as implementation source of truth.

7. **Stage3 integration is obsolete under the new decision.**

   Commit `5bc78d7` added `SALTRDState`, `get_state()`, `stage3_policy()`, and runner
   telemetry. This was useful while we considered TSA compatibility. It now conflicts
   with the final architecture because it:

   - consumes `tsa_state_int`;
   - maps risks to actions using handcrafted thresholds;
   - still leaves TSA in the runner;
   - does not create a learned action policy.

   Required fix:

   - do not build on `stage3_policy()`;
   - replace it with `SALTRDController` + learned action heads;
   - remove Stage3 state interpretation from runtime.

8. **`interventions.py` is analysis-only now.**

   `decide_intervention()` is threshold logic. It can be kept temporarily only for
   offline comparison, but it cannot be used in production rollout.

9. **Full UAV123 advisory benchmark confirmed no trajectory effect.**

   Existing full UAV123 result showed `Baseline 0.673 -> Advisory 0.673`, delta `0.000`.
   That means risk-only/advisory work does not meet the new success criterion.

   Required fix:

   - next rollout must report changed bbox frames;
   - if changed bbox frames are zero, the experiment is a NO-GO regardless of AUROC.

#### Updated Immediate Priority

The actual next sequence is now:

1. Restore `saltr/data` and canonical NPZs.
2. Materialize `production_no_flow` checkpoint with metadata.
3. Make eval/runtime auto-apply checkpoint feature schema.
4. Rerun oracle action audit with all hard sequences.
5. Generate learned reinit labels.
6. Train learned reinit/candidate policy.
7. Run rollout and require bbox changes + hard AUC improvement.

### Latest Work Review Addendum — commits `27136f7` / `ee12be1`

Reviewed latest Phase 5 + Phase 8 work:

- `saltr/src/salt_r/advisor.py`
- `src/uav_tracker/salt_runner.py`
- `src/uav_tracker/ml/tsa/saltrd_adapter.py`
- `scripts/fast_bench.py`
- `tests/unit/test_saltr_phase5_phase8.py`
- untracked `saltr/src/salt_r/phase7_eval.py`
- untracked `tests/unit/test_saltr_phase7.py`
- `saltr/results/phase8_saltrd_primary_benchmark.json`

#### What Is Valuable

1. **The Phase 8 benchmark finally shows trajectory-level movement.**

   `saltr/results/phase8_saltrd_primary_benchmark.json` reports:

   | Mode | Hard mean AUC | Hard mean Pr@20 | Delta vs baseline |
   |---|---:|---:|---:|
   | baseline | 0.176 | 0.285 | — |
   | advisory_stage2 | 0.176 | 0.285 | 0.000 |
   | primary_stage3 | 0.222 | 0.373 | +0.046 AUC / +0.088 Pr@20 |

   Per-sequence signal:

   | Sequence | Baseline AUC | Primary Stage3 AUC | Delta |
   |---|---:|---:|---:|
   | uav2 | 0.492 | 0.487 | -0.005 |
   | uav4 | 0.069 | 0.090 | +0.021 |
   | uav6 | 0.070 | 0.374 | +0.304 |
   | bike2/uav3/uav5/uav7 | mostly neutral | mostly neutral | ~0 |

   This is the first result that satisfies part of the causal chain:

   ```text
   policy routing changed -> trajectory changed -> hard AUC improved
   ```

   However, it does **not** satisfy the final architecture because the routing is still
   expressed through old `TargetState` integers and handcrafted policy thresholds.

2. **Feature schema auto-apply direction is good.**

   `feature_schema.py`, `eval.py`, `shadow_mode.py`, and `advisor.py` now move toward
   checkpoint-driven no-flow feature masking. This direction is correct.

   But the current production checkpoint still fails the metadata check:

   ```text
   saltr/checkpoints/production/saltrd_best.pt
   feature_schema = <missing>
   drop_feature_indices = <missing>
   ```

   Therefore the code path exists, but the shipped production artifact is still not the
   no-flow production checkpoint. This remains a blocker.

3. **Spatially constrained detector recovery is the likely useful executor primitive.**

   The crop/offset implementation in `salt_runner.py` is a plausible executor for a
   future learned reinit action. Keep the mechanical ability to run detector on a crop,
   offset detections back to full-frame coordinates, and log candidate/reinit outcomes.

   But the decision to use that crop must come from a learned action head, not from
   `p_fc` / APCE-ratio thresholds.

#### What Is Architecturally Rejected

These additions must **not** become production architecture:

1. **`saltrd_adapter.py` is the wrong direction.**

   The adapter maps `SALTRDState -> TargetState` for compatibility:

   ```text
   SALT-RD -> CONFIRMED/OCCLUDED/DYNAMIC/LOST
   ```

   This is exactly what the final plan forbids. SALT-RD must not be squeezed back into
   TSA classes. Delete this adapter during the TSA removal phase. Do not add tests that
   require it.

2. **`SALTRDState`, `get_state()`, and `stage3_policy()` are transitional only.**

   `advisor.py` still implements:

   - handcrafted thresholds (`p_fc >= 0.60`, `p_fc >= 0.30`, rising edge, top2 ratio);
   - `get_state(tsa_state_int)`;
   - `stage3_policy(tsa_state_int)`;
   - `allow_ce_pruning` / `force_full_compute` through old state routing.

   This explains the Phase 8 AUC gain, but it is not the final controller. The next
   implementation must extract the useful behavior into learned action labels:

   ```text
   evidence window -> learned action head -> FULL_COMPUTE / CE_LIGHT / RECOVER_CROP / NOOP
   ```

   It must not keep `TargetState` or `stage3_policy()`.

3. **Center-freeze remains KILL.**

   `advisor.update_center_freeze()` and `tracker.override_search_center()` are still
   wired into `salt_runner.py`. The oracle audit showed center-freeze gives ~0 hard AUC
   gain. Do not continue this branch. If `override_search_center()` stays, it may only be
   a generic executor used by a future learned action, never a p_fc/APCE threshold policy.

4. **Template update remains KILL for now.**

   `salt_runner.py` currently attempts dynamic template update through
   `advisor.should_block_template_update()` and fixed APCE/PSR/cosine thresholds.
   This branch caused the car7 regression historically and the oracle gain is near zero.
   Disable/remove runtime dynamic template update until a learned counterfactual template
   action head passes its gates.

5. **Phase 7 early recovery trigger is rule-based.**

   `advisor.should_trigger_early_recovery()` uses:

   ```text
   p_fc >= 0.55 and apce_ratio5 < 0.75
   ```

   `advisor.update_recovery_hint()` uses:

   ```text
   p_fc >= 0.45 and apce_ratio5 < 0.85
   ```

   These can be archived as experimental baselines, but final SALT-RD must replace them
   with a learned `RECOVER_CROP` / `RECOVER_FULL` / `NOOP` action head.

6. **The new tests lock in the wrong architecture.**

   `tests/unit/test_saltr_phase5_phase8.py` validates:

   - proactive hint thresholds;
   - `SALTRDState -> TargetState` adapter.

   Untracked `tests/unit/test_saltr_phase7.py` validates:

   - `p_fc >= 0.55` plus APCE-ratio threshold;
   - crop bbox helper behavior copied into a mock.

   These tests should not be accepted as final production tests. Replace them with tests
   that assert:

   - no runtime import of TSA modules;
   - no `TargetState` adapter;
   - learned action outputs control recovery/compute;
   - crop/offset mechanics are deterministic executor code only.

7. **`phase7_eval.py` is not a clean final evaluator.**

   The untracked evaluator treats advisory and constrained recovery as the same run:

   ```text
   auc_phase7 = auc_advisory
   ```

   That makes action attribution muddy. Final rollout must report per-frame:

   - model action probabilities/logits;
   - selected action;
   - executor result;
   - whether bbox changed;
   - AUC delta by sequence.

8. **The untracked Phase 7 summary is a hard FAIL.**

   `saltr/results/phase7_summary.json` reports:

   | Scope | Baseline | Phase7/advisory | Delta |
   |---|---:|---:|---:|
   | hard mean AUC | 0.1729 | 0.1369 | -0.036 |
   | standard min delta | — | — | -0.488 |

   Regressions include:

   - `bike2`: -0.078
   - `uav3`: -0.058
   - `uav7`: -0.097
   - `uav8`: -0.055
   - `car7`: -0.143
   - `car13`: -0.488
   - `truck1`: -0.340

   This is stronger evidence that threshold/advisory recovery must not be productionized.
   Keep crop execution mechanics, but delete the p_fc/APCE trigger.

#### Critical Code Bugs / Risks To Fix

1. **Production checkpoint metadata is still missing.**

   Fix before any more benchmark claims:

   ```text
   feature_schema = saltrd_v2_online_no_flow
   drop_feature_indices = [22,23,24,25,26,27]
   ```

2. **`salt_runner.py` calls `should_block_template_update()` while building telemetry.**

   Current aux construction calls the advisor again:

   ```python
   _aux["template_update_blocked"] = (
       _template_attempted and not _template_updated
       and not _advisor.should_block_template_update() is False
   )
   ```

   This has side effects (`n_blocked`, `n_allowed`, `_last_gate_reason`) and can corrupt
   telemetry. In the final architecture this whole gate is removed. If it remains during
   transition, compute the block decision once and store it in a local variable.

3. **`salt_runner.py` still performs TSA temporal gating and LOST/OCCLUDED escalation.**

   Even with `use_saltrd_primary`, the runner still imports `TargetState`, computes TSA,
   maintains `consecutive_lost`, `consecutive_occluded`, and fires detector recovery after
   fixed counters. This is incompatible with the final plan.

4. **The reported +0.046 hard AUC is not yet a deployment result.**

   Reasons:

   - only seven UAV123 hard sequences;
   - no full UAV123 regression result in this artifact;
   - no VisDrone/DTB70 check;
   - production checkpoint is not no-flow;
   - action path is threshold/TSA-compatible, not learned-controller-native.

   Treat it as a useful proof that compute/recovery routing can change AUC, not as a
   finished SALT-RD result.

#### Updated Interpretation

The latest work changes the plan in one important way:

> Reinit/recovery is still the main AUC path, but learned compute routing may also be
> useful because Phase 8 improved hard AUC through routing on uav6.

So the final controller should train/evaluate two learned action groups before optional
template update:

1. **Recovery/reinit action head**: `NOOP / PREPARE_RECOVERY / RECOVER_CROP / RECOVER_FULL / REJECT_REINIT`.
2. **Compute action head**: `FULL_COMPUTE / CE_LIGHT / CE_OFF`.

Both must be learned from counterfactual/oracle labels. Neither may be implemented as a
threshold mapping to old `TargetState`.

## Target Architecture

### 1. EvidenceExtractor

New module:

```text
saltr/src/salt_r/evidence.py
saltr/src/salt_r/feature_schema.py
```

Responsibilities:

- convert `TrackState` + frame + tracker internals into a typed feature vector;
- maintain rolling history;
- enforce the production no-flow schema;
- expose top-k score-map candidates;
- expose detector candidates when recovery is evaluated;
- never output a state label or decision.

Evidence groups:

| Group | Examples | Runtime? |
|---|---|---|
| response quality | APCE, PSR, entropy, peak margin, peak width | yes |
| score-map geometry | top-k peaks, secondary peak ratio, peak distance | yes |
| bbox dynamics | velocity, acceleration, scale ratio, border distance | yes |
| template history | last update age, update count, update context | yes |
| recovery history | last reinit age, detector candidate stats | yes |
| optical flow | Farneback/teacher consistency | no production |
| teacher identity | DINO/CoTracker/TAPIR features | offline teacher only unless proven fast |

### 2. TrackerAction

New module:

```text
saltr/src/salt_r/actions.py
```

Canonical action contract:

```python
class ComputeAction(Enum):
    FULL = "full"
    PRUNE_LIGHT = "prune_light"
    PRUNE_MEDIUM = "prune_medium"

class SearchAction(Enum):
    KEEP = "keep"
    EXPAND = "expand"
    FREEZE = "freeze"
    CENTER_ON_REINIT_HINT = "center_on_reinit_hint"

class TemplateAction(Enum):
    KEEP_CURRENT = "keep_current"
    UPDATE = "update"
    BLOCK_UPDATE = "block_update"

class RecoveryAction(Enum):
    NONE = "none"
    SCORE_CANDIDATES = "score_candidates"
    REINIT = "reinit"
    REJECT_REINIT = "reject_reinit"

@dataclass
class TrackerAction:
    compute: ComputeAction
    search: SearchAction
    template: TemplateAction
    recovery: RecoveryAction
    bbox_hint: BBox | None = None
    detector_hint: BBox | None = None
```

This is not a rule system. It is the execution format for model-selected actions.

### 3. PolicyNet

New/updated module:

```text
saltr/src/salt_r/policy_model.py
```

The model must output actions, not just risks:

```python
SALT_RD outputs:
    risk_probs:
        false_confirmed
        ifd10
        ifd20
        recoverable
    action_logits:
        compute
        search
        template
        recovery
        reinit_candidate
    optional:
        bbox_delta
        reinit_candidate_score
```

Runtime action selection:

```python
output = policy_net(features_window)
action = decode_argmax_action(output.action_logits)
```

No production thresholds for APCE, p_fc, p_ifd10, memory margin, or e-process.

### 4. Controller

New module:

```text
saltr/src/salt_r/controller.py
```

Responsibilities:

- run EvidenceExtractor;
- run PolicyNet;
- decode action logits;
- apply deterministic safety constraints only:
  - NaN/Inf guard;
  - bbox clipping to image;
  - detector unavailable -> cannot execute reinit;
  - model output shape/version mismatch -> fail closed;
- return `SALT_RD_Decision`.

Important: deterministic safety constraints are not tracking policy. They do not encode
APCE thresholds or handcrafted risk rules.

### 5. Tracker API

Replace:

```python
update_with_state(frame, target_state_int, ...)
```

with:

```python
update_with_action(frame, action: TrackerAction) -> TrackState
```

`SGLATracker` should use action fields directly:

- `action.compute` controls CE/full compute;
- `action.search` controls search factor or hint center;
- `action.template` controls whether template update is attempted;
- `action.recovery` controls detector/reinit path.

## Tracker / Detector Modularity Plan

### Why This Matters

We do **not** need many trackers/detectors to prove the first engineering result. The
minimal success path is:

```text
SGLATrack + one detector + learned SALT-RD actions
    -> hard AUC improves
    -> full set does not regress
```

But for a paper-quality claim, we should show at least limited modularity:

```text
SALT-RD is not a one-off patch for SGLATrack+YOLO26m.
```

Recommended paper evidence:

| Claim level | Required backends | Why |
|---|---|---|
| workshop/internal | 1 tracker + 1 detector | enough to prove controller can move trajectory |
| solid paper | 2 trackers + 2 detectors | shows tracker/detector decoupling |
| strong general controller claim | 3 trackers + 2-3 detectors + per-backend calibration | shows real backend-agnostic behavior |

Do not delay the first working SALT-RD system waiting for many backends. Build the clean
adapter boundary now, then run backend matrix after the controller works.

### Current Coupling Audit

Existing useful pieces:

- `src/uav_tracker/trackers/base.py` already defines a tracker protocol with
  `init()`, `update()`, and `flops_per_update()`.
- `src/uav_tracker/detectors/base.py` already defines a detector protocol.
- `uav_tracker.registry` already has `TRACKERS` and `DETECTORS`.
- Implemented tracker backends include `sglatrack`, `kcf_henriques`, and
  `ostrack_256` (OSTrack currently depends on weights and may be stubby).
- Implemented detector backends include `yolo26m_visdrone`, `rtdetr`, `leaf_yolo`,
  and generic YOLOv8.

Current coupling problems:

1. `SALTRunner.from_config()` is hardwired to TSA construction and old optional modules.
2. `SGLATrack` has non-standard `update_with_state()` and SALT-RD advisor hooks.
3. Detector signatures are inconsistent:
   - base protocol uses `detect(frame, hint=...)`;
   - several implementations use `hint_bbox=...`;
   - `YOLOv8Detector` uses the hint for crop search;
   - `VisDroneYOLO26m`, `LEAF-YOLO`, and some others mostly ignore the hint.
4. Recovery candidate selection is inside `SALTRunner._best_detection()`, not a clean
   detector/candidate scorer module.
5. Tracker telemetry is not standardized. SGLATrack exposes APCE/PSR/entropy/score-map
   stats; KCF/OSTrack may not expose the same evidence.

### Required Backend Contracts

#### Tracker Backend Contract

Add a SALT-RD-capable tracker adapter interface:

```python
class SALTRDTrackerBackend(Protocol):
    name: str
    capabilities: TrackerCapabilities

    def init(self, frame: np.ndarray, bbox: BBox) -> None: ...
    def update_with_action(self, frame: np.ndarray, action: TrackerAction) -> TrackState: ...
    def extract_evidence(self) -> TrackerEvidence: ...
    def apply_reinit(self, frame: np.ndarray, bbox: BBox) -> None: ...
    def flops_per_update(self, action: TrackerAction | None = None) -> float: ...
```

`TrackerCapabilities`:

```python
supports_ce: bool
supports_template_update: bool
supports_score_map: bool
supports_topk_candidates: bool
supports_internal_embeddings: bool
supports_search_hint: bool
```

`TrackerEvidence` must be typed and may include missing masks:

```python
apce: float | None
psr: float | None
entropy: float | None
score_map_stats: dict
topk_tracker_candidates: list[BBox]
last_bbox: BBox
confidence: float
missing_feature_mask: np.ndarray
```

Important: missing evidence is allowed. It must be represented by feature masks, not by
pretending every tracker has SGLATrack features.

#### Detector Backend Contract

Standardize all detectors to:

```python
detect(frame: np.ndarray, hint: BBox | None = None, mode: DetectorMode = FULL) -> CandidateSet
```

Where:

```python
class DetectorMode(Enum):
    FULL = "full"
    CROP_AROUND_HINT = "crop_around_hint"
    FULL_THEN_RANK_BY_HINT = "full_then_rank_by_hint"
```

`CandidateSet`:

```python
candidates: list[DetectionCandidate]
source: str
crop_bbox: BBox | None
runtime_ms: float
```

`DetectionCandidate`:

```python
bbox: BBox
score: float
class_id: int | None
class_name: str | None
source: str
features: dict
```

The detector must only generate candidates. It must not decide whether recovery happens.
SALT-RD decides action; detector executes candidate generation.

### Backend Matrix For Experiments

Start with a small matrix:

| Role | Backend | Status | Purpose |
|---|---|---|---|
| tracker A | SGLATrack | primary | current strongest baseline and feature-rich backend |
| tracker B | OSTrack-256 | secondary if weights are valid | tests whether SALT-RD can work with another transformer tracker |
| tracker C | KCF-HOG | sanity/low-end | tests whether system gracefully handles weak telemetry |
| detector A | YOLO26m VisDrone | primary | current recovery detector, UAV/VisDrone-oriented |
| detector B | RT-DETRv2 | secondary | better candidate ranking/proximity behavior, different detector family |
| detector C | YOLOv8 crop mode | engineering baseline | validates hint/crop executor consistency |

Do **not** treat KCF as a competitive paper tracker. It is an adapter/capability stress
test only.

### Experiment Plan

#### Stage A — Backend Smoke Matrix

Goal: verify adapters without retraining.

Run:

```text
trackers = [SGLATrack, OSTrack-256 if weights valid, KCF-HOG]
detectors = [YOLO26m, RT-DETRv2, YOLOv8-crop]
datasets = [UAV123 hard subset, UAV123 standard subset]
```

Measure:

- baseline tracker AUC;
- detector candidate count per recovery frame;
- candidate oracle recall;
- runtime/FPS;
- feature availability mask coverage;
- failure cases where backend lacks required evidence.

Exit gate:

- every backend combination either runs or fails with a clear capability error;
- no backend-specific `if tracker_name == ...` in controller logic;
- all detector candidates go through the same `CandidateSet` path.

#### Stage B — Frozen Controller Transfer Test

Goal: test whether the current SGLATrack-trained SALT-RD policy transfers.

Run the same learned policy on:

```text
SGLATrack + YOLO26m
SGLATrack + RT-DETRv2
OSTrack + YOLO26m
OSTrack + RT-DETRv2
```

Expected:

- SGLATrack transfer should be meaningful.
- OSTrack transfer may degrade because telemetry distribution differs.

This is a diagnostic, not a final result.

#### Stage C — Per-Tracker Calibration / Lightweight Adaptation

If transfer fails, do not immediately fine-tune the tracker. First train tiny backend
adaptation layers:

1. shared SALT-RD policy trunk;
2. tracker-specific input normalization;
3. tracker-id embedding;
4. optional small LoRA/adapter on SALT-RD, not on the tracker.

Artifacts:

```text
saltr/checkpoints/controller_sgla/
saltr/checkpoints/controller_ostrack/
saltr/results/backend_matrix_transfer.json
saltr/results/backend_matrix_adapted.json
```

This preserves the claim: the controller adapts to tracker telemetry without changing
the base tracker.

#### Stage D — Detector Candidate Fine-Tuning

Detector fine-tuning is more likely to help than tracker fine-tuning for recovery,
because oracle action audit says reinit is the highest-upside action.

Fine-tune only if candidate oracle recall is low:

| Symptom | Fix |
|---|---|
| target absent from detector candidates | fine-tune detector or lower conf/create crop mode |
| target present but not top-ranked | train SALT-RD candidate scorer, not detector |
| wrong class/domain mismatch | class-agnostic candidate mode or detector fine-tune |
| crop misses target | learned crop/recovery action, not detector fine-tune |

Detector fine-tune data:

- UAV123 train split boxes converted to detection format;
- VisDrone-SOT boxes where available;
- DTB70 only if we keep DTB70 in the target domain;
- hard negative crops around distractors.

Recommended fine-tune modes:

1. **candidate recall mode**: prioritize recall at low confidence; SALT-RD candidate
   scorer handles precision.
2. **class-agnostic objectness mode**: one target-like object class for recovery.
3. **hard-negative mode**: include wrong cyclists/sheep/players/birds as negatives if
   class labels permit.

Do not fine-tune detector to make final decisions. It remains a candidate generator.

#### Stage E — Tracker Fine-Tuning Decision

Only fine-tune trackers if all are true:

1. learned recovery actions fire correctly;
2. detector candidate recall is adequate;
3. AUC is still capped because the base tracker loses target before candidates matter;
4. hard subset failures are tracker-localization failures, not candidate failures.

Tracker fine-tune options:

| Option | Use when | Risk |
|---|---|---|
| LoRA/adapter on transformer tracker | need domain adaptation with low parameter count | may blur SALT-RD contribution |
| full tracker fine-tune on UAV123/VisDrone | base tracker ceiling is too low | overfit, harder paper comparison |
| train only template/update head | failures are update corruption | car7 risk; only after oracle gain |

For the paper, tracker fine-tuning should be a separate row:

```text
SGLATrack
SGLATrack + fine-tune
SGLATrack + SALT-RD
SGLATrack + fine-tune + SALT-RD
```

This keeps the controller contribution interpretable.

### Do We Need This For The Paper?

Minimum: yes, but limited.

Required for paper credibility:

1. **Detector ablation**: at least YOLO26m vs RT-DETRv2 or YOLOv8-crop.
2. **Tracker sanity transfer**: at least SGLATrack vs one alternative tracker.
3. **Backend capability table**: show which evidence/actions each backend supports.
4. **Main table remains SGLATrack**, because that is the current project baseline.

Not required for first paper version:

- full fine-tuning of every tracker;
- exhaustive detector zoo;
- claiming universal tracker-agnostic control.

Best claim:

> SALT-RD is a learned controller interface for tracker trust and recovery. It is
> demonstrated primarily on SGLATrack, with detector and tracker backend ablations showing
> which parts are backend-independent and which require calibration.

### Backend Result Tables To Add

#### Table H — Backend Capability Matrix

| Backend | Type | Score map | APCE/PSR | Top-k candidates | CE action | Template action | Search hint | Runtime status |
|---|---|---|---|---|---|---|---|---|
| SGLATrack | tracker | yes | yes | TO_FILL | yes | limited | yes | primary |
| OSTrack-256 | tracker | TO_FILL | TO_FILL | TO_FILL | no/TO_FILL | TO_FILL | yes | secondary |
| KCF-HOG | tracker | response map | peak only | TO_FILL | no | yes/native | yes | sanity |
| YOLO26m | detector | n/a | n/a | n/a | n/a | n/a | currently ignores hint unless wrapped | primary |
| RT-DETRv2 | detector | n/a | n/a | n/a | n/a | n/a | rank/filter by hint | secondary |
| YOLOv8-crop | detector | n/a | n/a | n/a | n/a | n/a | crop mode | baseline |

#### Table I — Tracker/Detector Matrix

| Tracker | Detector | Hard AUC | Full UAV123 AUC | Candidate recall | Wrong reinit | FPS | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| SGLATrack | YOLO26m | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | primary |
| SGLATrack | RT-DETRv2 | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | detector ablation |
| SGLATrack | YOLOv8-crop | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | crop baseline |
| OSTrack-256 | YOLO26m | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | transfer |
| OSTrack-256 | RT-DETRv2 | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | transfer |
| KCF-HOG | YOLO26m | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | sanity |

#### Table J — Fine-Tuning Decision Table

| Model | Bottleneck metric | Before | After fine-tune/adapt | Did SALT-RD still add value? | Decision |
|---|---:|---:|---:|---:|---|
| detector YOLO26m | candidate recall | TO_FILL | TO_FILL | TO_FILL | TO_FILL |
| detector RT-DETRv2 | candidate recall | TO_FILL | TO_FILL | TO_FILL | TO_FILL |
| tracker SGLATrack adapter | hard AUC | TO_FILL | TO_FILL | TO_FILL | TO_FILL |
| tracker OSTrack adapter | hard AUC | TO_FILL | TO_FILL | TO_FILL | TO_FILL |

## Implementation Phases

### Phase 0 — Artifact Repair And Baseline Freeze

Goal: make every subsequent result reproducible.

Tasks:

1. Restore canonical data artifacts:
   - `saltr/data/salt_rd_v2_labels.npz`
   - train/val/diagnostic splits
   - OOF predictions if still needed
   - sidecars only if part of a reproducible experiment
2. Fix the missing `saltr/data` state.
3. Rerun current oracle action audit with all hard sequences available.
4. Save baseline table:
   - SGLATrack only;
   - current TSA/SALT archived baseline;
   - current SALT-RD advisory;
   - current no-flow checkpoint;
   - oracle action audit.

Exit gate:

- audit covers all expected hard sequences or explicitly documents an approved exclusion;
- `oracle_action_audit.json` is reproducible from committed inputs;
- no circular symlink or missing-data note remains.

### Phase 1 — Define New Contracts

Goal: introduce the clean architecture without changing behavior yet.

Tasks:

1. Add `saltr/src/salt_r/actions.py`.
2. Add `saltr/src/salt_r/evidence.py`.
3. Add `saltr/src/salt_r/feature_schema.py`.
4. Add unit tests for:
   - no-flow feature schema;
   - feature ordering parity with training;
   - action serialization;
   - action decode shape checks.

Exit gate:

- new modules pass tests;
- no production behavior change yet.

### Phase 2 — Replace SGLATrack State API

Goal: remove the state-int control surface.

Tasks:

1. Add `SGLATracker.update_with_action(frame, action)`.
2. Keep `update()` as plain SGLATrack update.
3. Remove or deprecate `update_with_state()`.
4. Replace `_STATE_COMPUTE_MAP` and `_STATE_SEARCH_MAP` with action handling.
5. Keep CE disabled by default until the learned compute head is validated.

Exit gate:

- pure SGLATrack AUC is unchanged;
- action `FULL` is equivalent to baseline full compute;
- no `TargetState` import inside `sglatrack.py`.

### Phase 3 — Delete TSA Runtime Path

Goal: production runner no longer imports or constructs TSA.

Tasks:

1. Rewrite `src/uav_tracker/salt_runner.py` so the loop is:

   ```python
   track_state = tracker.update_with_action(frame, previous_action)
   evidence = evidence_extractor.step(track_state, frame)
   decision = controller.step(evidence)
   previous_action = decision.action
   ```

2. Remove:
   - `tsa` dataclass field;
   - `_prev_tsa_state_int`;
   - OCCLUDED/LOST counters;
   - APCE threshold escalation;
   - TSA temporal gating;
   - TSA telemetry.
3. Keep detector and template machinery as executors, not decision makers.

Exit gate:

```bash
rg "TargetState|TargetStateAssessor|ml/tsa|CONFIRMED|OCCLUDED|DISTRACTOR_RISK|tsa_" src/uav_tracker saltr/src configs/prod
```

must return no production-control references.

### Phase 4 — Remove TSA Files And Config

Goal: no dead API remains.

Tasks:

1. Delete `src/uav_tracker/ml/tsa/`.
2. Delete `scripts/train_tsa_classifier.py`.
3. Remove TSA weights if tracked.
4. Remove `tsa:` config sections.
5. Update README and docs.
6. Move old analysis docs to archive if needed.

Exit gate:

- code import graph has no TSA module;
- tests no longer instantiate TSA;
- configs have no TSA keys.

### Phase 5 — Action Oracle Dataset

Goal: train actions from counterfactual utility, not handcrafted thresholds.

Primary next action from current evidence: `reinit`.

Tasks:

1. Build `saltr/src/salt_r/oracle_actions.py`.
2. For each frame/candidate, compute future utility:

   ```text
   utility = AUC_next_50
           + min_iou_next_20_bonus
           - wrong_reinit_penalty
           - template_corruption_penalty
           - compute_penalty
   ```

3. Generate labels:
   - `recovery_action`: NONE / SCORE_CANDIDATES / REINIT / REJECT_REINIT;
   - `reinit_candidate_score`;
   - later: compute/search/template labels.
4. Start with the reinit action because oracle gain is largest.

Exit gate:

- action labels are not derived from APCE thresholds or TSA states;
- per-sequence utility is available;
- harmful-sequence list is reported.

### Phase 6 — Learned Reinit Policy

Goal: turn the oracle reinit opportunity into real runtime AUC gain.

Tasks:

1. Add recovery/reinit heads to PolicyNet.
2. Train on oracle reinit labels.
3. Evidence must include detector candidate features:
   - detector confidence;
   - bbox size;
   - proximity to tracker prediction;
   - score-map top-k candidate relation;
   - candidate temporal consistency;
   - optional offline identity teacher score.
4. Runtime:
   - model chooses whether to run detector/reinit;
   - model scores candidate;
   - controller applies selected candidate.

Exit gate:

- reinit action fires on hard scenes;
- bbox trajectory changes;
- hard AUC delta >= +0.10 (oracle ceiling is +0.08 on 14 seqs; all 18 hard seqs required before final gate);
- full UAV123 AUC delta >= +0.010 (positive improvement required; oracle predicts +0.025 achievable).

### Phase 7 — Template Update Policy

Goal: fix car7-style template corruption only if oracle says it matters.

Current oracle result says template update gives tiny hard gain and many harmful cases.
Therefore this phase is secondary.

Tasks:

1. Build counterfactual labels:
   - update at frame t;
   - do not update at frame t;
   - compare AUC/IoU over next 20/50 frames.
2. Train a learned `template_action` head.
3. Include car7, bike2, truck1, and crowded DTB70 frames as hard negatives.

Exit gate:

- car7 does not regress;
- template action produces measurable AUC or safety benefit;
- otherwise keep dynamic template updates disabled.

### Phase 8 — Compute / CE Policy

Goal: use CE only after tracking-control works.

Decision:

- CE stays in the code as an actuator.
- CE is disabled by default during TSA deletion and reinit-policy work.
- No `ce_keep_ratio_by_state`.
- Later, model output chooses:
  - `FULL`;
  - `PRUNE_LIGHT`;
  - `PRUNE_MEDIUM`.

Training:

- use full-vs-pruned counterfactual replay;
- label CE safe only if future IoU/AUC does not degrade;
- compute penalty is secondary to tracking quality.

Exit gate:

- no hard-scene AUC loss;
- measurable FPS/GFLOP gain;
- if no measurable gain, keep CE off.

### Phase 9 — Evaluation And Reporting

Required outputs:

1. Full-set table:

   | Method | UAV123 AUC | Hard AUC | FPS | GFLOPs | Notes |
   |---|---:|---:|---:|---:|---|

2. Action causality table:

   | Sequence | Action | Fired frames | Changed bbox frames | AUC delta |
   |---|---|---:|---:|---:|

3. Per-action ablation:

   | Policy | Hard AUC | Full AUC | Wrong reinit | Template corruption |
   |---|---:|---:|---:|---:|

4. Failure report:
   - sequences where action harmed;
   - sequences where model did not fire despite oracle opportunity;
   - detector candidate unavailable cases.

## Bug Fix Checklist

| Bug / Risk | Fix |
|---|---|
| TSA false-confirmed blind spot | delete TSA state machine; learned false-confirmed/reinit policy |
| Dead `DYNAMIC` state | delete old classes entirely |
| APCE threshold brittleness | APCE becomes feature only |
| OCCLUDED/LOST delayed recovery | learned recovery/reinit action |
| Flow train/runtime mismatch | no-flow production schema |
| Entropy concern | keep entropy as feature |
| CE routing by state | learned compute action only |
| Template update car7 regression | dynamic updates disabled until learned counterfactual head passes |
| Rule-based `interventions.py` | remove from production; keep only analysis if needed |
| `advisor.stage3_policy(tsa_state_int)` | replace with model action output |
| Missing `saltr/data` | restore canonical artifacts before final gates |
| Oracle audit missing hard sequences | rerun with all hard sequences before final decision |
| `saltrd_adapter.py` maps SALT-RD back to TSA | delete; SALT-RD must own action API directly |
| Phase 7 p_fc/APCE recovery trigger | convert to learned recovery action labels/head |
| Center-freeze runtime path | remove/disable; oracle says KILL |
| Production checkpoint missing no-flow metadata | rebuild/materialize before any production benchmark |
| Telemetry calls policy methods with side effects | compute decisions once; telemetry must be passive |

## What To Kill Immediately

- TSA as runtime module.
- `TargetState` enum in production control.
- `update_with_state(target_state_int)`.
- `ce_keep_ratio_by_state`.
- `decide_intervention()` as production controller.
- `saltrd_adapter.py` and any `SALTRDState -> TargetState` compatibility layer.
- `SALTRDState`, `get_state()`, and `stage3_policy()` as production APIs.
- `should_trigger_early_recovery()` as a threshold controller.
- `update_recovery_hint()` as a threshold controller.
- `update_center_freeze()` as runtime behavior.
- dynamic template update through `should_block_template_update()`.
- untracked `phase7_eval.py` as final evaluator.
- tests that assert p_fc/APCE thresholds or TargetState adapter behavior.
- Any plan section saying TSA remains fallback.
- Any claim based only on AUROC without action/AUC rollout.

## What To Keep

- SGLATrack as base tracker.
- YOLO26m/detector as executor for learned reinit action.
- APCE/PSR/entropy/score-map stats as evidence.
- No-flow SALT-RD checkpoint direction.
- Oracle action audit methodology.
- Crop detector executor mechanics: crop frame, run detector, offset detections back
  to full-frame coordinates. The crop decision must be learned.
- Phase 8 result as evidence that action/routing can move AUC, not as final architecture.
- CE implementation as future compute actuator, disabled by default.

## Immediate Next 7 Tasks

1. Fix artifact baseline first: restore `saltr/data`, rebuild/materialize
   `production_no_flow`, and verify checkpoint metadata.
2. Archive or remove current Phase 5/7/8 threshold-controller runtime paths:
   `saltrd_adapter.py`, center-freeze, p_fc/APCE early recovery, and Stage3
   `TargetState` routing.
3. Keep only passive telemetry and executor primitives: tracker update, detector crop
   executor, candidate offset, bbox clipping, provenance logging.
4. Add `actions.py`, `evidence.py`, and extend `feature_schema.py` into a strict
   checkpoint/runtime contract.
5. Implement `SGLATracker.update_with_action()` and `SALTRunner` controller loop with
   learned action outputs only.
6. Rerun oracle action audit with all hard sequences and generate learned
   recovery/reinit + compute-routing labels.
7. Train/evaluate learned recovery/compute policy and require
   `action -> bbox/compute -> hard AUC+` with full-set no-regression.

## Claude Implementation Rules

This section is written for another implementation agent. Follow it literally.

### Hard Constraints

1. Do not keep TSA as fallback.
2. Do not keep TSA as debug runtime state.
3. Do not keep old `TargetState` enum in production code.
4. Do not introduce a new rule-based state machine under a different name.
5. Do not use APCE thresholds, p_fc thresholds, e-process thresholds, memory-margin
   thresholds, or fixed lost-frame counters to choose tracking actions.
6. Do not claim success from AUROC/AUPRC alone.
7. Do not claim success unless bbox trajectory changes and hard-scene AUC improves.
8. Do not use online Farneback flow as a production feature.
9. Do not use detector as controller. Detector only returns candidates.
10. Do not let template update run unless a learned model action allows it.

### Allowed Deterministic Safety Code

The only deterministic checks allowed in production are non-policy safety guards:

- tensor shape and schema version validation;
- NaN/Inf checks;
- bbox clipping to image bounds;
- detector unavailable -> cannot execute detector action;
- missing checkpoint -> fail fast;
- invalid action enum -> fail fast;
- no candidate available -> action is not executable;
- image/frame index consistency checks.

These checks must not encode tracking-risk rules.

### Forbidden Patterns

Reject any patch containing these patterns in production control paths:

```python
if apce < ...
if psr < ...
if p_fc > ...
if p_ifd10 > ...
if consecutive_lost >= ...
if state == TargetState...
if state_name == "CONFIRMED"
if old_tsa_state ...
```

Exception: tests may search for these strings to assert they are absent, and archived
analysis scripts may contain old code if clearly outside runtime.

### Production Reference Search

Before considering implementation done, run:

```bash
rg -n "TargetState|TargetStateAssessor|ml/tsa|CONFIRMED|OCCLUDED|LOST|DYNAMIC|DISTRACTOR_RISK|tsa_" \
  src/uav_tracker saltr/src configs/prod tests
```

Allowed hits:

- tests that assert old symbols are absent;
- archived docs/scripts only if outside `src/`, `saltr/src`, `configs/prod`, and active tests.

No hits are allowed in production control code.

## Detailed File-by-File Implementation Plan

### A. New `saltr/src/salt_r/actions.py`

Purpose: one canonical action contract used by train, eval, runner, controller, and tests.

Must define:

```python
class ComputeAction(str, Enum):
    FULL = "full"
    PRUNE_LIGHT = "prune_light"
    PRUNE_MEDIUM = "prune_medium"

class SearchAction(str, Enum):
    KEEP = "keep"
    EXPAND = "expand"
    FREEZE = "freeze"
    CENTER_ON_REINIT_HINT = "center_on_reinit_hint"

class TemplateAction(str, Enum):
    KEEP_CURRENT = "keep_current"
    UPDATE = "update"
    BLOCK_UPDATE = "block_update"

class RecoveryAction(str, Enum):
    NONE = "none"
    SCORE_CANDIDATES = "score_candidates"
    REINIT = "reinit"
    REJECT_REINIT = "reject_reinit"

@dataclass(frozen=True)
class TrackerAction:
    compute: ComputeAction = ComputeAction.FULL
    search: SearchAction = SearchAction.KEEP
    template: TemplateAction = TemplateAction.KEEP_CURRENT
    recovery: RecoveryAction = RecoveryAction.NONE
    bbox_hint: BBox | None = None
    detector_hint: BBox | None = None
```

Implementation details:

- `TrackerAction` should be immutable (`frozen=True`) to avoid accidental mutation between frames.
- Add `to_json()` and `from_json()` helpers for telemetry and replay artifacts.
- Do not import TSA or old tracker state symbols.
- Do not include thresholds in this file.

Tests:

- serialization round trip;
- invalid enum value raises `ValueError`;
- default action is full compute, keep search, no template update, no recovery.

### B. New `saltr/src/salt_r/feature_schema.py`

Purpose: single source of truth for feature order and production drop/zero rules.

Required constants:

```python
FEATURE_SCHEMA_VERSION = "saltrd_v3_no_tsa_no_flow"
FLOW_FEATURE_INDICES = (22, 23, 24, 25, 26, 27)
PRODUCTION_ZERO_FEATURE_INDICES = FLOW_FEATURE_INDICES
```

Current v2-compatible base feature order:

| Index | Name | Production note |
|---:|---|---|
| 0 | apce | numeric evidence |
| 1 | apce_norm | numeric evidence |
| 2 | psr | numeric evidence |
| 3 | response_entropy | keep; entropy helped |
| 4 | peak_margin | numeric evidence |
| 5 | peak_width | numeric evidence |
| 6 | n_secondary | numeric evidence |
| 7 | peak_distance | numeric evidence |
| 8 | heatmap_mass_topk | numeric evidence |
| 9 | apce_ratio_5 | rolling numeric evidence |
| 10 | apce_ratio_20 | rolling numeric evidence |
| 11 | entropy_delta_5 | rolling numeric evidence |
| 12 | peak_margin_delta_5 | rolling numeric evidence |
| 13 | high_apce_streak_legacy | v2 parity only; do not use for new hand rules |
| 14 | low_apce_streak_legacy | v2 parity only; do not use for new hand rules |
| 15 | bbox_vx | numeric evidence |
| 16 | bbox_vy | numeric evidence |
| 17 | bbox_speed | numeric evidence |
| 18 | bbox_accel | numeric evidence |
| 19 | bbox_scale_ratio | numeric evidence |
| 20 | bbox_aspect_delta | numeric evidence |
| 21 | dist_to_border | numeric evidence |
| 22 | global_flow_mag | zero in production |
| 23 | target_flow_mag | zero in production |
| 24 | ego_motion_residual | zero in production |
| 25 | flow_iou | zero in production |
| 26 | flow_residual | zero in production |
| 27 | flow_consistency | zero in production |

Important:

- For old v2 risk checkpoints, keep the feature order exactly for compatibility.
- For new v3 action policy, prefer continuous rolling features over threshold-derived
  streaks. If indices 13/14 remain for checkpoint compatibility, document them as
  legacy scalar evidence and never use them as action rules.

Required functions:

```python
def feature_names(schema: str) -> list[str]: ...
def zero_production_features(x: np.ndarray) -> np.ndarray: ...
def validate_feature_matrix(x: np.ndarray, expected_dim: int) -> None: ...
def schema_metadata() -> dict[str, Any]: ...
```

Tests:

- flow indices are zeroed in runtime;
- shape mismatch fails;
- schema metadata is stored in checkpoints;
- eval/runtime both apply the same schema.

### C. New `saltr/src/salt_r/evidence.py`

Purpose: convert tracker outputs into model evidence.

Core classes:

```python
@dataclass
class EvidenceFrame:
    frame_idx: int
    bbox: BBox
    base_features: np.ndarray
    score_map_stats: dict[str, Any]
    candidates: list[CandidateEvidence]
    template_context: TemplateContext
    recovery_context: RecoveryContext

class EvidenceExtractor:
    def reset(self) -> None: ...
    def step(self, frame: np.ndarray, track_state: TrackState, tracker: Any) -> EvidenceFrame: ...
```

Candidate evidence must include:

- candidate bbox;
- score-map rank;
- score-map score;
- score ratio to top candidate;
- distance to current tracker prediction;
- distance to previous bbox;
- size ratio to current tracker bbox;
- detector score when available;
- optional offline teacher score when generating labels.

Implementation guardrails:

- `EvidenceExtractor` does not make decisions.
- `EvidenceExtractor` does not know `TrackerAction`.
- `EvidenceExtractor` does not call detector.
- `EvidenceExtractor` does not import TSA.
- `EvidenceExtractor` may compute numeric rolling features but cannot choose state/action.

Tests:

- frame 0 produces valid features;
- no-flow production features are zero;
- top-k candidates are stable and serializable;
- reset clears rolling buffers;
- extractor parity with old `collect_features.py` for v2-compatible fields.

### D. New `saltr/src/salt_r/policy_model.py`

Purpose: learned model with explicit action heads.

Model structure:

```python
class SALTRDPolicyNet(nn.Module):
    trunk: GRU/TCN
    risk_heads:
        false_confirmed
        imminent_failure_dynamic_10
        imminent_failure_dynamic_20
        recoverable
    action_heads:
        compute_action
        search_action
        template_action
        recovery_action
    candidate_head:
        reinit_candidate_score
```

Loss:

```text
total_loss =
    risk_loss
  + lambda_recovery * recovery_action_ce
  + lambda_candidate * candidate_ranking_loss
  + lambda_template * template_action_ce
  + lambda_compute * compute_action_ce
```

Start with recovery/reinit only:

- implement action heads for all actions, but train only recovery/reinit labels first;
- template/compute/search heads may be masked until labels exist.

Checkpoint metadata must include:

```json
{
  "model_family": "saltrd_policy",
  "feature_schema": "saltrd_v3_no_tsa_no_flow",
  "n_base_features": 28,
  "zero_feature_indices": [22, 23, 24, 25, 26, 27],
  "action_schema": "...",
  "trained_heads": [...],
  "train_split": "...",
  "label_source": "oracle_counterfactual",
  "git_commit": "...",
  "created_at": "..."
}
```

Tests:

- forward pass returns all heads;
- checkpoint load reconstructs exact head shapes;
- old risk checkpoint is not silently loaded as policy checkpoint;
- model refuses feature schema mismatch.

### E. New `saltr/src/salt_r/controller.py`

Purpose: only runtime owner of SALT-RD decisions.

Core API:

```python
@dataclass
class SALTRDDecision:
    action: TrackerAction
    risk_probs: dict[str, float]
    action_probs: dict[str, dict[str, float]]
    selected_candidate: CandidateEvidence | None
    model_confidence: float
    safety_fallback_applied: bool
    reason: str

class SALTRDController:
    def reset(self) -> None: ...
    def step(self, frame: np.ndarray, track_state: TrackState, tracker: Any) -> SALTRDDecision: ...
```

Allowed safety fallback:

- if model unavailable: fail fast in production; optional explicit `--baseline` mode may run tracker only;
- if detector unavailable and model asks recovery: convert to non-executable action and log;
- if candidate score tensor invalid: no reinit;
- if bbox invalid: clip or reject.

Forbidden:

- no `if p_fc > 0.6`;
- no APCE thresholds;
- no old state mapping;
- no lost-frame counters.

Tests:

- controller does not import TSA;
- action changes are reproducible from fixed model output;
- unavailable detector prevents only recovery execution, not all tracking;
- model schema mismatch fails fast.

### F. Update `src/uav_tracker/trackers/sglatrack.py`

Current bad API:

```python
update_with_state(frame, target_state_int, ...)
```

New API:

```python
update_with_action(frame, action: TrackerAction) -> TrackState
```

Implementation details:

- default action is full compute, normal search;
- CE keep ratio comes from `action.compute`;
- search factor/hint comes from `action.search`;
- template update is not performed inside basic update unless action explicitly allows it;
- reset embedding caches on init/reset;
- preserve existing telemetry: APCE, PSR, entropy, score_map_stats, top-k candidates.

CE handling:

| Action | CE behavior |
|---|---|
| `FULL` | `ce_keep_rate = 1.0` |
| `PRUNE_LIGHT` | conservative keep, e.g. 0.75 after validation |
| `PRUNE_MEDIUM` | e.g. 0.50 only after compute policy passes |

Do not include `ce_keep_ratio_by_state`.

Tests:

- `FULL` matches old full-compute output within tolerance;
- CE disabled by default;
- no TargetState import;
- invalid action fails;
- score-map stats still populated.

### G. Rewrite `src/uav_tracker/salt_runner.py`

Target runtime loop:

```python
previous_action = TrackerAction()
for frame in frames:
    track_state = tracker.update_with_action(frame, previous_action)
    decision = saltrd_controller.step(frame, track_state, tracker)
    applied = apply_action(decision.action, frame, track_state)
    previous_action = applied.next_tracker_action
    yield telemetry(decision, applied)
```

Runner responsibilities:

- sequence lifecycle;
- tracker init/reset;
- pass action to tracker;
- call detector only when controller action requires candidate scoring/reinit;
- apply selected reinit candidate;
- write telemetry;
- never infer state itself.

Telemetry fields:

```json
{
  "saltrd_action_compute": "...",
  "saltrd_action_search": "...",
  "saltrd_action_template": "...",
  "saltrd_action_recovery": "...",
  "saltrd_action_confidence": 0.0,
  "saltrd_changed_bbox": true,
  "saltrd_reinit_candidate_score": 0.0,
  "apce_raw": 0.0,
  "psr_raw": 0.0,
  "entropy_raw": 0.0,
  "score_map_stats": {}
}
```

Remove telemetry fields:

- `target_state`;
- `tsa_confidence`;
- `tsa_template`;
- `tsa_reinit`;
- `saltrd_state` if it is a manually interpreted state rather than model output;
- any old TSA class names.

Tests:

- runner never imports TSA;
- first frame telemetry has SALT-RD fields only;
- detector is called only when action requests it;
- action fired count matches telemetry;
- changed bbox count is measurable.

### H. Delete/Archive Old TSA Files

Delete from active code:

- `src/uav_tracker/ml/tsa/target_state.py`
- `src/uav_tracker/ml/tsa/target_state_assessor.py`
- `src/uav_tracker/ml/tsa/velocity_drift.py`
- `src/uav_tracker/ml/tsa/__init__.py`
- `scripts/train_tsa_classifier.py`
- TSA weights if tracked.

Docs:

- update `README.md`;
- update `README_UK.md`;
- update `HANDOFF_NEXT.md`;
- archive old TSA analysis only under `docs/archive/` or `saltr/archive/`.

Tests:

- remove tests that instantiate TSA;
- add tests that assert old symbols are absent.

## Oracle Reinit Policy Details

The current oracle audit says reinit has the only meaningful hard-scene AUC upside.
Therefore implement reinit first.

### Candidate Generation

Candidate sources:

1. detector candidates from YOLO26m/selected detector;
2. SGLATrack score-map top-k decoded candidates;
3. optional teacher candidate identity scores during offline label generation.

For each frame, store:

```json
{
  "sequence_key": "...",
  "frame_idx": 123,
  "current_bbox": [x, y, w, h],
  "gt_bbox": [x, y, w, h],
  "candidate_source": "detector|score_map|teacher",
  "candidate_bbox": [x, y, w, h],
  "candidate_features": [...],
  "candidate_iou_now": 0.0,
  "utility_next_20": 0.0,
  "utility_next_50": 0.0,
  "label_reinit": 0/1,
  "label_reject": 0/1
}
```

### Utility Function

Start with:

```text
future_iou_gain_20 = mean_iou(candidate_rollout, t:t+20) - mean_iou(baseline, t:t+20)
future_iou_gain_50 = mean_iou(candidate_rollout, t:t+50) - mean_iou(baseline, t:t+50)

wrong_reinit_penalty = 1.0 if candidate_iou_now < 0.3 and baseline_iou_now >= 0.5 else 0.0
fragmentation_penalty = 0.05 if candidate causes large bbox jump but no IoU gain else 0.0

utility = future_iou_gain_50
        + 0.5 * future_iou_gain_20
        - wrong_reinit_penalty
        - fragmentation_penalty
```

Label:

```text
REINIT if utility > +0.03 and candidate_iou_now >= 0.3
REJECT_REINIT if candidate_iou_now < 0.3 or utility < -0.01
NONE otherwise
```

These thresholds are for offline label construction only. They are not runtime policy.

### Required Reinit Dataset Diagnostics

Write:

```text
saltr/results/reinit_oracle_dataset_summary.json
```

Must contain:

- number of sequences;
- number of frames;
- number of candidate frames;
- positive reinit base rate;
- reject reinit base rate;
- per-dataset base rates;
- per-hard-sequence positives;
- candidate source distribution;
- maximum oracle gain by sequence;
- missing sequence list.

Stop condition:

- if positive reinit base rate < 0.5%, switch to ranking loss and event-balanced sampler;
- if hard sequences have zero candidates, detector/candidate generation must be fixed before training.

## Training Plan

### Training Run 0 — Freeze Current Risk Baseline

Purpose: preserve the best current risk model after no-flow fix.

Inputs:

- current v2 labels;
- production no-flow feature schema;
- no TSA state features;
- no online flow;
- no action heads required.

Expected known result from current work:

| Checkpoint | Val fc AUROC | Val fc AUPRC | Diagnostic fc AUROC | Note |
|---|---:|---:|---:|---|
| v2_retrained baseline | ~0.885 | ~0.338 | ~0.598 | strict baseline |
| v2_no_flow | ~0.883 | ~0.350 | ~0.697 | current best risk baseline |

Required artifact:

```text
saltr/checkpoints/production_no_flow/saltrd_best.pt
saltr/results/eval_val_production_no_flow.json
saltr/results/eval_diagnostic_production_no_flow.json
```

### Training Run 1 — Reinit Candidate Scorer

Purpose: decide which candidate is worth reinitializing to.

Model:

- candidate pair/ranking model;
- input = frame evidence window + candidate features;
- output = candidate utility score.

Loss options:

1. BCE on `label_reinit`;
2. pairwise ranking loss within frame;
3. hybrid BCE + ranking.

Use ranking if base rate is sparse.

Metrics:

- candidate AUROC;
- candidate AUPRC;
- top-1 candidate IoU;
- oracle recall recovered by model at top-1;
- wrong reinit rate.

Go gate:

| Metric | Gate |
|---|---:|
| candidate AUPRC | >= 0.30 or 5x base rate |
| top-1 oracle recall on hard frames | >= 0.50 |
| wrong reinit rate on val | <= baseline |
| diagnostic hard AUC rollout | +0.03 or better |

### Training Run 2 — Recovery Action Head

Purpose: choose `NONE / SCORE_CANDIDATES / REINIT / REJECT_REINIT`.

Model:

- shared trunk over evidence window;
- recovery action head;
- candidate scorer attached when candidates exist.

Loss:

```text
recovery_action_ce + candidate_ranking_loss
```

Sampling:

- sequence-balanced sampler;
- hard-event oversampling;
- keep an untouched val split;
- keep diagnostic split only for final stress test, not early stopping.

Metrics:

- action macro-F1;
- reinit recall on oracle-positive frames;
- reject precision on wrong-candidate frames;
- hard AUC rollout;
- full AUC rollout.

Go gate:

| Metric | Gate |
|---|---:|
| reinit action recall on oracle positives | >= 0.40 |
| wrong reinit rate | not worse than current baseline |
| hard subset AUC delta | >= +0.10 (interim with 14 seqs: >= +0.08 acceptable) |
| full UAV123 AUC delta | >= +0.010 |

### Training Run 3 — Template Action Head

Only run after reinit policy succeeds or if oracle re-run shows template action gain.

Purpose:

- learned `KEEP_CURRENT / UPDATE / BLOCK_UPDATE`.

Do not start from APCE/p_fc gates. Labels must come from update-vs-no-update
counterfactual replay.

Go gate:

| Metric | Gate |
|---|---:|
| car7 AUC delta | >= 0.000 |
| template corruption rate | lower than baseline |
| hard subset AUC delta | positive or neutral |
| full UAV123 AUC delta | >= -0.005 |

If template oracle gain remains near zero, keep dynamic template updates disabled.

### Training Run 4 — Compute / CE Action Head

Run after the clean action API exists. It does not need to wait for template update.
The latest Phase 8 result suggests compute routing can affect hard-scene AUC, but the
current implementation routes through `TargetState`. This training run is how we keep
the useful idea while deleting the bad architecture.

Purpose:

- learned `FULL / PRUNE_LIGHT / PRUNE_MEDIUM`.
- optionally learned `CE_OFF` when the controller predicts high recovery risk.

Labels:

- full-vs-pruned counterfactual replay;
- CE safe only when future IoU/AUC does not degrade.
- include the Phase 8 hard cases (`uav4`, `uav6`) in the counterfactual report, because
  current threshold routing found gain there.

Go gate:

| Metric | Gate |
|---|---:|
| hard AUC delta | >= 0.000 |
| full AUC delta | >= -0.005 |
| GFLOPs reduction | meaningful, target >= 5% |
| FPS gain | measurable on MPS |
| no `TargetState` dependency | required |
| no p_fc/APCE threshold routing | required |

If GFLOPs gain remains < 3%, keep CE off in production.

## Calibration Plan

Calibration is required for reporting, confidence telemetry, and deployment monitoring.
It must not become a handwritten runtime controller.

### What To Calibrate

| Output | Calibration method | Used for action? |
|---|---|---|
| `false_confirmed` risk | temperature / vector temperature | no direct threshold action |
| `ifd10/ifd20` risk | temperature | monitoring only unless model has learned action |
| recovery action logits | multiclass temperature scaling | action confidence reporting |
| candidate utility score | Platt or isotonic on val | candidate score reporting |
| template action logits | multiclass temperature scaling | confidence reporting |
| compute action logits | multiclass temperature scaling | confidence reporting |

### Calibration Artifacts

Write:

```text
saltr/results/calibration_val_policy.json
saltr/results/preds_val_policy_calibrated.json
saltr/results/calibration_diagnostic_policy.json
```

Required fields:

```json
{
  "checkpoint": "...",
  "feature_schema": "...",
  "split": "val",
  "heads": {
    "false_confirmed": {"ece_before": 0.0, "ece_after": 0.0, "temperature": 1.0},
    "recovery_action": {"ece_before": 0.0, "ece_after": 0.0, "temperature": 1.0},
    "candidate_score": {"ece_before": 0.0, "ece_after": 0.0, "method": "..."}
  }
}
```

Go gates:

| Calibration metric | Gate |
|---|---:|
| recovery action ECE | <= 0.10 |
| candidate score ECE | <= 0.15 |
| false_confirmed ECE | improve vs no-flow baseline |
| calibration does not change ranking | AUROC/AUPRC unchanged for binary heads |

If ECE remains high:

1. check split leakage and base rates;
2. use vector temperature;
3. use isotonic for reporting only;
4. do not add runtime thresholds to compensate.

## Evaluation Commands To Add

Add scripts or Make targets so Claude does not run ad-hoc commands.

Required commands:

```bash
# 1. Restore/check data
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.check_artifacts

# 2. Generate oracle action dataset
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.oracle_actions \
  --npz saltr/data/salt_rd_v2_labels.npz \
  --output saltr/results/reinit_oracle_dataset.npz \
  --summary saltr/results/reinit_oracle_dataset_summary.json

# 3. Train policy
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.train_policy \
  --oracle saltr/results/reinit_oracle_dataset.npz \
  --output saltr/checkpoints/policy_reinit_v1/

# 4. Calibrate policy
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.calibrate_policy \
  --checkpoint saltr/checkpoints/policy_reinit_v1/saltrd_policy_best.pt \
  --split val \
  --output saltr/results/calibration_val_policy_reinit_v1.json

# 5. Rollout evaluation
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.rollout_policy \
  --checkpoint saltr/checkpoints/policy_reinit_v1/saltrd_policy_best.pt \
  --split uav123_full \
  --output saltr/results/rollout_uav123_policy_reinit_v1.json

# 6. Hard subset rollout
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.rollout_policy \
  --checkpoint saltr/checkpoints/policy_reinit_v1/saltrd_policy_best.pt \
  --split hard \
  --output saltr/results/rollout_hard_policy_reinit_v1.json
```

Every command must write:

- input artifact paths;
- md5 of inputs;
- checkpoint md5;
- git commit;
- feature schema;
- action schema;
- random seed;
- dataset split;
- wall-clock time.

## Required Test Matrix

### Unit Tests

| Test file | Required coverage |
|---|---|
| `tests/unit/test_saltr_actions.py` | action enum, serialization, invalid values |
| `tests/unit/test_saltr_feature_schema.py` | no-flow zeroing, schema metadata, shape validation |
| `tests/unit/test_saltr_evidence.py` | feature extraction, reset, candidate serialization |
| `tests/unit/test_saltr_policy_model.py` | forward shapes, checkpoint metadata, schema mismatch |
| `tests/unit/test_saltr_controller.py` | model output -> action decode, safety fallback only |
| `tests/unit/test_sglatrack_action_api.py` | `update_with_action`, no `TargetState`, CE disabled default |
| `tests/unit/test_no_tsa_runtime.py` | no active imports/symbols from TSA |

### Integration Tests

| Test | Required behavior |
|---|---|
| runner one-sequence smoke | emits SALT-RD telemetry only |
| detector action smoke | detector called only when action asks |
| reinit action smoke | selected candidate changes bbox |
| baseline action smoke | default full action matches SGLATrack-only trajectory |
| artifact reproducibility | result JSON contains md5/provenance |

### Regression Tests

Required named regressions:

- car7: no template corruption regression;
- uav2: reinit/recovery opportunity is detected;
- bike2: wrong cyclist reinit is rejected or not worsened;
- truck1: re-acquisition is not broken;
- building1/car13: standard/easy scenes do not regress.

## Historical Results To Preserve

These are current known results. Do not overwrite them without a new artifact name.

### Risk Model Results

| Run | Key result | Interpretation |
|---|---:|---|
| v0 | fc AUROC ~0.884, fc AUPRC ~0.331 | first useful trust head |
| v0 calibrated | fc ECE ~0.264, fail5 ECE ~0.015, recoverable ECE ~0.098 | temperature helped some heads |
| v1 | fc AUROC ~0.890, fc AUPRC ~0.356, ifd AUROC ~0.900 | label split helped |
| v2 | ifd5 AUROC ~0.902, ifd10 ~0.897, ifd20 ~0.889 | long-horizon risk works |
| v2 failure_in_10/20 | AUROC ~0.827/~0.785, AUPRC ~0.017/~0.022 | sparse labels; use hazard/ranking if needed |
| v2 no-flow | diag fc AUROC ~0.697 vs strict baseline ~0.598 | production schema should be no-flow |
| no-entropy | diag fc AUROC ~0.583 | entropy helps; keep it |
| no-entropy no-flow | diag fc AUROC ~0.691 | worse than no-flow alone |

### Memory / Teacher Results

| Run | Val fc | Diagnostic fc | Decision |
|---|---:|---:|---|
| proxy memory full | val AUROC ~0.857, diag AUROC ~0.796 | validates direction but hurts val |
| proxy pos-only | val AUROC ~0.852, diag AUROC ~0.774 | RAM driver, but proxy too coarse |
| proxy neg-only | diag AUROC ~0.496 | harmful alone |
| SGLA score_weighted | val AUROC ~0.858, diag AUROC ~0.584 | kill |
| SGLA peak_local | val AUROC ~0.858, diag AUROC ~0.584 | kill |
| LK point v2.3 | diag AUROC ~0.546 | kill |

### Policy / Oracle Results

| Action | Hard oracle gain | Full oracle gain | Decision |
|---|---:|---:|---|
| reinit | +0.0834 | +0.02463 | build learned policy |
| search_expand | +0.00408 | +0.00049 | kill for now |
| template_update | +0.00109 | +0.00274 | secondary; many harmful cases |
| center_freeze | +0.00003 | +0.00215 | kill for now |

### Runtime / Rollout Results So Far

| Run | Scope | Hard AUC delta | Full AUC delta | Verdict |
|---|---|---:|---:|---|
| Stage2 advisory/veto | 7 UAV123 hard seqs | 0.000 | not reported | risk-only path has no trajectory effect |
| Phase7 p_fc/APCE constrained recovery | 8 UAV123 hard + 4 standard seqs | -0.036 | not reported | hard FAIL, standard regressions |
| Stage3 primary via TSA routing | 7 UAV123 hard seqs | +0.046 | not reported | useful signal, rejected architecture |

The Stage3 result is evidence that controller actions can improve hard scenes, but it
does not count as final SALT-RD because it still uses `TargetState` routing, handcrafted
thresholds, and a production checkpoint without no-flow metadata.

Important caveat: current oracle audit missed 4 hard sequences because `saltr/data` was
unavailable. Rerun before finalizing.

## Required New Training / Calibration Result Tables

Claude must fill these tables as artifacts and update this section after each run.

### Table A — Oracle Dataset Summary

| Artifact | Sequences | Frames | Candidate frames | Reinit positives | Positive rate | Missing hard seqs | Status |
|---|---:|---:|---:|---:|---:|---|---|
| `reinit_oracle_dataset.npz` | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL |

Required status:

- PASS only if all required hard sequences are included or exclusion is explicitly approved.

### Table B — Reinit Candidate Model

| Run | Val AUROC | Val AUPRC | Candidate top-1 recall | Wrong reinit rate | Diagnostic hard recall | Status |
|---|---:|---:|---:|---:|---:|---|
| candidate_scorer_v1 | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL |

Target:

- AUPRC >= 0.30 or >= 5x base rate;
- top-1 recall >= 0.50 on oracle-positive hard frames;
- wrong reinit not worse than baseline.

### Table C — Recovery Action Head

| Run | Action macro-F1 | Reinit recall | Reject precision | Val action ECE | Hard AUC delta | Full AUC delta | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| policy_reinit_v1 | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL |

Target:

- hard AUC delta >= +0.10 (interim checkpoint with 14 hard seqs: >= +0.08 acceptable; final gate requires all 18 seqs);
- full AUC delta >= +0.010 (positive improvement; oracle ceiling +0.025);
- action ECE <= 0.10;
- bbox changed frames > 0.5% on hard subset.

### Table D — Calibration Results

| Run | Head | ECE before | ECE after | Method | Temperature/Params | AUROC change | AUPRC change | Status |
|---|---|---:|---:|---|---|---:|---:|---|
| policy_reinit_v1 | false_confirmed | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL |
| policy_reinit_v1 | recovery_action | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL |
| policy_reinit_v1 | candidate_score | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL |

Expected:

- ranking metrics should not change for temperature scaling;
- calibrated probabilities improve ECE;
- no runtime thresholds are added because of calibration.

### Table E — Rollout Results

| Method | Full UAV123 AUC | Hard subset AUC | Changed bbox frames | Reinit actions | Compute actions | Wrong reinit rate | FPS | GFLOPs | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SGLATrack baseline | TO_FILL | TO_FILL | 0 | 0 | 0 | 0 | TO_FILL | TO_FILL | baseline |
| Old SALT/TSA archived | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | archive only |
| SALT-RD risk-only/no-flow | TO_FILL | TO_FILL | 0 | 0 | 0 | 0 | TO_FILL | TO_FILL | diagnostic only |
| Stage3 primary via TSA routing | not reported | 0.222 on 7 hard seqs | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | rejected architecture, useful signal |
| SALT-RD learned reinit v1 | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | GO/NO-GO |
| SALT-RD learned reinit+compute v1 | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | GO/NO-GO |

GO condition:

- learned reinit v1 hard AUC delta >= +0.10 (interim with 14 seqs: >= +0.08 acceptable; final gate needs all 18 seqs);
- learned reinit v1 full AUC delta >= +0.010 (positive improvement required; oracle shows +0.025 is achievable);
- changed bbox frames > 0.5%;
- wrong reinit rate not worse.
- any compute-action claim must report actual CE/full-compute frame counts and measured
  FPS/GFLOPs, not only action probabilities.

### Table F — Per-Sequence Causality

| Sequence | Baseline AUC | SALT-RD AUC | Delta | Action fired frames | Changed bbox frames | Best action | Failure mode |
|---|---:|---:|---:|---:|---:|---|---|
| uav123/uav2 | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL |
| uav123/bike2 | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | wrong reinit risk |
| uav123/car7 | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | template corruption risk |
| uav123/truck1 | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | reacquisition |
| dtb70/Gull2 | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | class mismatch |
| dtb70/Sheep1 | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | distractor herd |
| dtb70/StreetBasketball1 | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | similar persons |

This table is mandatory. If it is not filled, the run is not interpretable.

### Table G — Final Result Table To Publish Internally

| System | Controller type | TSA present? | Learned actions? | UAV123 AUC | Hard AUC | VisDrone AUC | DTB70 AUC | FPS MPS | GFLOPs/frame | Main claim |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| SGLATrack | none | no | no | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | base tracker |
| SALT v3 old | rule-based TSA | yes | no | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | archived baseline |
| SALT-RD risk-only | learned risk, no actions | no | no | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | diagnosis only |
| Stage3 primary experiment | learned risk + rule routing | yes | partial/rule-based | TO_FILL | 0.222 on 7 hard seqs | TO_FILL | TO_FILL | TO_FILL | TO_FILL | rejected architecture, proves action path matters |
| SALT-RD learned reinit | learned controller | no | recovery/reinit | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | hard-scene AUC gain |
| SALT-RD learned reinit+compute | learned controller | no | recovery + compute | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | hard-scene gain plus optional efficiency |
| SALT-RD + CE policy | learned controller | no | recovery + compute | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | TO_FILL | optional compute gain |

Final acceptable outcome:

- The first production version may be `SALT-RD learned reinit` without CE.
- CE policy is optional and must not delay proving hard-scene AUC gain.
- If learned reinit does not improve hard AUC, stop and revisit candidate generation
  before training more risk heads.
