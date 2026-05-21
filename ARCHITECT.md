# ARCHITECT — Ongoing Code Review Against SUPER_PLAN

Staff AI/ML CV Architect review of all in-progress changes.
Updated: 2026-05-21.

---

## How To Use This File

This file tracks architectural findings, decision changes, and blocking issues found
during code review. The reviewer re-scans new code periodically and appends to the
relevant section. Anything tagged **[BLOCKER]** must be fixed before the next rollout.
Anything tagged **[WRONG DIRECTION]** means the implementation conflicts with SUPER_PLAN
and must be replaced, not extended.

---

## Critical Findings — 2026-05-21 (Initial Review)

### [BLOCKER-1] `salt_runner.py` half-migration will crash on every frame

The diff removes the following fields from the SALTRunner dataclass:
- `tsa`
- `_prev_tsa_state_int`
- `use_saltrd_primary`
- `_prev_saltrd_policy`
- `_consecutive_lost`, `_consecutive_occluded`
- `_RECOVERY_MIN_LOST`, `_OCCLUDED_ESCALATION_FRAMES`
- `_prev_apce`, `_apce_buffer`, `_prev_escalated_apce`

But `_step()` was **not touched** and still reads all of these fields at lines 333, 334,
346-349, 370-372, 407, 416-421, 424-447, 456-459, 499-546, 549-552. Every call to
`_step()` will raise `AttributeError`.

The runner is in a split state: the `run()` path was partially cleaned but the core
`_step()` was left entirely intact. This is not a partial migration — it is broken code
that cannot run at all.

**Required fix:** Either revert `run()` to restore removed fields, or complete the
`_step()` rewrite as described in SUPER_PLAN Phase 3 (Section G). There is no valid
intermediate state here.

---

### [BLOCKER-2] New `saltrd_controller` and `evidence_extractor` are dead code

`from_config()` now constructs `SALTRDController` and `EvidenceExtractor` and stores
them on the runner. But `_step()` never calls them. The entire new control path is
completely inert. The controller always falls back to NOOP because the model is `None`
(no checkpoint path is read in `from_config()` — just `SALTRDController()` with no
arguments).

The new architecture objects exist in memory, are never invoked, and produce zero effect
on tracking behavior. This means the Phase 1 contract work is correct but Phase 3
(wiring the controller into the runner loop) has not started.

**Required fix:** Implement the runner loop from SUPER_PLAN Section G before any
performance claim involving the new controller.

---

### [BLOCKER-3] `_step()` still imports and fully runs TSA on every frame

Line 308 inside `_step()`:
```python
from uav_tracker.ml.tsa.target_state import TargetState
```

Lines 407-447 run full TSA assess with temporal gating, optical flow, APCE thresholds.
Lines 490-546 run OCCLUDED escalation with `_OCCLUDED_ESCALATION_FRAMES` threshold.
Lines 549-552 run the `_consecutive_lost` counter.
Lines 333-341 still route CE through TSA state integers.

TSA is still the only thing controlling tracker behavior. The SUPER_PLAN gate "zero
production TSA references" fails at the first line of `_step()`. SALT-RD is not the
primary controller — TSA is. Nothing has changed at the behavior level.

---

### [WRONG DIRECTION-1] `update_with_action()` in sglatrack.py is a hollow stub

The new method (lines 826-840 of sglatrack.py) always calls `self.update(frame)`,
ignoring `action.search`, `action.template`, and `action.recovery`. It only issues a
warning for non-FULL compute and then runs the same code path. There is no behavior
change for any action variant.

This means:
- `SearchAction.FREEZE` does nothing
- `SearchAction.CENTER_ON_REINIT_HINT` does nothing
- `TemplateAction.BLOCK_UPDATE` does nothing
- `RecoveryAction.REINIT` does nothing
- Only `ComputeAction.FULL` has any meaning (it calls `update(frame)` as normal)

The method cannot satisfy the plan's exit gate: "action `FULL` is equivalent to
baseline full compute" passes, but "action `FREEZE` changes search center" fails
entirely. The API name is correct; the implementation is a skeleton.

**Required fix:** Wire `action.search` to `override_search_center()`, wire
`action.template` to `allow_template_update` flag, wire `action.recovery` to the
detector path. Keep CE disabled for now as the plan prescribes.

---

### [WRONG DIRECTION-2] Hard import of `salt_r.actions` in `sglatrack.py` at module level

```python
try:
    from salt_r.actions import TrackerAction, ComputeAction, SearchAction
except ImportError as e:
    raise ImportError("salt_r package required: ...") from e
```

The tracker is in `src/uav_tracker/` and `salt_r` is in `saltr/src/`. Making the core
tracker a hard dependency on the SALT controller package couples these two subsystems
at module load time. Any test, script, or evaluation that imports SGLATracker without
`PYTHONPATH=saltr/src` will crash immediately, including all existing tracker tests that
predate SALT-RD.

The SUPER_PLAN says SGLATracker should expose `update_with_action(action: TrackerAction)`
but this should use a TYPE_CHECKING guard or accept a duck-typed protocol, not a hard
import that breaks the tracker in isolation.

**Required fix:** Use `from __future__ import annotations` + `TYPE_CHECKING` guard so
the import is type-only, or define a local `TrackerAction` stub and do the real import
lazily inside `update_with_action()`.

---

### [ARCH-1] `EvidenceExtractor.step()` API diverges from plan specification

SUPER_PLAN Section C specifies:
```python
def step(self, frame: np.ndarray, track_state: TrackState, tracker: Any) -> EvidenceFrame
```

Current implementation:
```python
def step(self, base_features: np.ndarray, bbox: BBox, score_map_stats=None, candidates=None)
```

The extractor takes pre-computed features as input instead of computing them from the
tracker. This means feature extraction is split: the caller must already have computed
the 28-dim vector and extracted candidates before calling `EvidenceExtractor.step()`.

The architectural intent is that `EvidenceExtractor` is the single responsible module
for all feature computation from raw tracker outputs. The current split means the runner
must know how to extract features, which replicates the knowledge that should live only
in the extractor.

This is not a crash-level bug today because nothing calls `EvidenceExtractor.step()`
yet (see BLOCKER-2). But it will produce the wrong design when the controller loop is
wired in.

**Acceptable workaround for Phase 1:** Keep current API if the intent is to accept
pre-extracted features and let the runner be responsible for calling the existing
`collect_features.py` pipeline. But this must be explicitly documented as a deliberate
scope choice, not a spec divergence.

---

### [BUG-1] `EvidenceExtractor._parse_candidates()` prev_bbox is always current bbox

In `evidence.py`, `step()` appends `bbox` to `self._bbox_history` before calling
`_parse_candidates()`:

```python
self._feature_history.append(prod_features)
self._bbox_history.append(bbox)            # <-- current bbox appended here
parsed_candidates = self._parse_candidates(candidates or [], bbox)
```

Inside `_parse_candidates()`:
```python
prev_bbox = self._bbox_history[-1] if len(self._bbox_history) >= 1 else tracker_bbox
```

`self._bbox_history[-1]` is now the current frame's bbox (just appended), not the
previous frame's bbox. So `distance_to_prev_bbox` is always the distance from the
candidate to the same bbox as `distance_to_tracker` — effectively zero or identical.
This feature carries no information.

**Required fix:** Save `prev_bbox` before appending, or index `[-2]` with a length
check.

---

### [ARCH-2] `from_config()` instantiates `SALTRDController` with no model path

```python
saltrd_controller = SALTRDController()
```

No checkpoint is loaded. The controller always returns `_safe_noop(reason="no_model_loaded")`.
Even if wired into `_step()`, it would produce zero tracking behavior change.

The config key `saltrd.enabled: true` enables the code path but the model path is not
read. There is no `saltrd.checkpoint` key handling.

**Required fix:** Read `saltrd_cfg.get("checkpoint")` and pass it to the controller.
Fail fast if `enabled=True` but no checkpoint is provided.

---

## Decision Changes vs Previous Work

### TSA compatibility adapter is fully rejected

`saltrd_adapter.py` (maps `SALTRDState -> TargetState`) was built in the Phase 5/8 work
to maintain backward compatibility. SUPER_PLAN explicitly rejects this:

> Delete this adapter during the TSA removal phase. Do not add tests that require it.

This is confirmed as the right decision. The Phase 8 AUC gain (+0.046 hard) was real
but it came from threshold/TSA routing, not from SALT-RD. The architecture is wrong even
if the metric was temporarily good.

### Center-freeze is killed

Oracle audit: +0.000 hard AUC gain. Phase 7 result showed -0.036 regression on the hard
subset. `update_center_freeze()` must not run in production. The call at line 387 in
`_step()` is still present and is wasted compute.

### Rule-based recovery trigger is killed

`should_trigger_early_recovery()` (p_fc >= 0.55 + apce_ratio5 < 0.75) and
`update_recovery_hint()` (p_fc >= 0.45 + apce_ratio5 < 0.85) are both threshold
policies. Phase 7 result proved these cause regressions. These calls remain active in
`_step()` at lines 400-401 and 567-571. They must be removed before any production
rollout.

### `stage3_policy(tsa_state_int)` is killed

Lines 369-372 in `_step()` still call `_advisor.stage3_policy(_tsa_int)`. This is the
exact handcrafted-threshold-to-TSA-mapping that SUPER_PLAN rejects. The Phase 8 result
was produced by this path. It explains the AUC gain but is not the final architecture.

---

## What Is Correctly Implemented

### Phase 1 contracts are sound

- `actions.py`: correct enum definitions, frozen dataclass, `to_json`/`from_json`. Clean.
- `feature_schema.py`: correct v3 schema, flow zeroing, `validate_feature_matrix`,
  `schema_metadata`. Matches plan spec exactly.
- `evidence.py`: correct `EvidenceFrame`, `CandidateEvidence`, `TemplateContext`,
  `RecoveryContext` structure. Isolation from TSA/decisions is maintained.
- `controller.py`: correct `SALTRDDecision`, `_safe_noop`, NaN/shape guards, no TSA
  imports. REINIT→SCORE_CANDIDATES fallback when no candidates is correct behavior.

### Test architecture is sound

- `test_saltr_actions.py`: TSA AST check, serialization round-trip, frozen mutation
  guard — all correct and aligned with plan.
- `test_saltr_controller.py`: NaN guard, shape mismatch, model error, REINIT fallback,
  TSA import check — all correct.
- `test_saltr_feature_schema.py` and `test_saltr_evidence.py` (unreviewed but expected):
  aligned with plan spec tests.

### TSA removal from `from_config()` is correct

Removing `TargetStateAssessor` construction from `from_config()` is the right move.
The runner should not construct TSA at all.

---

## Immediate Next Actions (Priority Order)

1. **Fix BLOCKER-1 + BLOCKER-3 together**: Rewrite `_step()` per SUPER_PLAN Section G.
   The minimal viable runner loop is:
   ```python
   track_state = tracker.update(frame)  # plain update for now
   decision = saltrd_controller.step(evidence_extractor.step(...))
   # apply decision action when controller is wired
   ```
   Until the controller has a real model, NOOP is fine — the critical thing is that TSA
   is not called at all.

2. **Fix BLOCKER-2**: Wire `evidence_extractor` and `saltrd_controller` into `_step()`.

3. **Fix BUG-1**: Save `prev_bbox` before appending to history in `EvidenceExtractor`.

4. **Fix WRONG DIRECTION-2**: Move `salt_r.actions` import in sglatrack.py inside
   `update_with_action()` or behind `TYPE_CHECKING`.

5. **Fix WRONG DIRECTION-1**: Implement actual action dispatch in `update_with_action()`.

6. **Fix ARCH-2**: Add checkpoint loading in `from_config()`.

7. Do NOT extend `stage3_policy`, center-freeze, TSA escalation, or threshold recovery
   logic. Any work on those paths is wasted effort and wrong direction.

---

## Forbidden Patterns Still Present In Production Code (As Of This Review)

Run this to verify after fixes:
```bash
rg -n "TargetState|TargetStateAssessor|ml/tsa|CONFIRMED|OCCLUDED|DYNAMIC|DISTRACTOR_RISK|tsa_|stage3_policy|update_center_freeze|update_recovery_hint|should_trigger_early_recovery|should_block_template_update|consecutive_lost|consecutive_occluded|_prev_tsa" \
  src/uav_tracker saltr/src
```

Expected: zero hits in production control code after Phase 3 rewrite.
Current: dozens of hits in `_step()`.

---

## Review — 2026-05-21 Pass 2 (10-min cycle)

### [BLOCKER-3 ESCALATED] TSA modules deleted but `_step()` still imports them

New working tree state: all five TSA files are **deleted**:
- `src/uav_tracker/ml/tsa/__init__.py`
- `src/uav_tracker/ml/tsa/saltrd_adapter.py`
- `src/uav_tracker/ml/tsa/target_state.py`
- `src/uav_tracker/ml/tsa/target_state_assessor.py`
- `src/uav_tracker/ml/tsa/velocity_drift.py`

BLOCKER-3 has escalated from "wrong architecture" to "instant crash". `_step()` line 308
has a runtime import:
```python
from uav_tracker.ml.tsa.target_state import TargetState
```
That module no longer exists on disk. Any call to `_step()` raises `ModuleNotFoundError`
before a single line of tracking logic runs. The system is completely non-functional.

BLOCKER-1 (dataclass fields removed but still referenced in `_step()`) and BLOCKER-3
are now the same crash, at the same call site. Fix target is identical: rewrite `_step()`.

### [OK] Phase 4 TSA deletion is proceeding correctly

Deleting the TSA module tree is the right move per SUPER_PLAN Phase 4. Archiving the
TSA-dependent adapter tests to `archive/tsa_removed/` rather than deleting them is also
correct — they serve as a historical record of the old architecture without keeping dead
imports in the active test suite.

### [OK] `test_no_tsa_runtime.py` is the right enforcement mechanism

The new test file asserts:
1. No TSA imports anywhere in `salt_runner.py` (AST walk).
2. No forbidden patterns: `TargetStateAssessor`, `TargetState.`, `from uav_tracker.ml.tsa`,
   `consecutive_lost`, `consecutive_occluded`, `update_with_state(`.
3. `update_with_action` is present and called in the runner.
4. SALT-RD telemetry fields exist in runner output.

These are exactly the right architectural assertions. All four tests will currently fail,
which is the correct state: tests define the target, not the current state.

### [BLOCKER-NEW] `test_no_tsa_runtime.py` will fail on 4 / 4 assertions against current code

Specifically:
- `test_no_tsa_import_in_runner` — FAILS: line 308 `from uav_tracker.ml.tsa.target_state import TargetState`
- `test_no_targetstate_usage_in_runner` — FAILS: `TargetState.`, `from uav_tracker.ml.tsa`,
  `consecutive_lost`, `consecutive_occluded`, `update_with_state(` all present in `_step()`
- `test_update_with_action_called_in_runner` — FAILS: runner calls `update_with_state`, not `update_with_action`
- `test_saltrd_telemetry_fields_present` — PASSES: fields added in init frame but `_step()` still emits old TSA fields too

These tests must all pass before any benchmark claim. Run them as a gate after each `_step()` edit.

### [OK] `test_saltr_phase5_phase8.py` adapter tests removed correctly

The TSA→SALT-RD adapter tests (tests 7 and 8) were removed from the active suite and
the docstring updated. No imports of deleted TSA modules remain in the active test file.
Tests 1–6 (recovery hint behavior) remain valid and unaffected.

### Current state summary

| Component | Status |
|---|---|
| `actions.py` | OK — clean |
| `feature_schema.py` | OK — clean |
| `evidence.py` | OK — clean (BUG-1 prev_bbox unresolved) |
| `controller.py` | OK — clean |
| `salt_runner._step()` | CRASH — ModuleNotFoundError on every frame |
| `sglatrack.update_with_action()` | WRONG DIRECTION — hollow stub |
| TSA module tree | OK — deleted |
| `test_no_tsa_runtime.py` | OK — correct gates, all 4 currently failing |
| `archive/tsa_removed/` | OK — correct archival |

---

## Review — 2026-05-21 Pass 3 (10-min cycle)

### [OK] BLOCKER-1 / BLOCKER-2 / BLOCKER-3 all resolved — `_step()` fully rewritten

The runner `_step()` has been completely replaced with the SUPER_PLAN Section G loop:

```python
prev_action = self._prev_action or TrackerAction()
track_state = self.tracker.update_with_action(frame, prev_action)
evidence_frame = self.evidence_extractor.step(...)
decision = self.saltrd_controller.step(evidence_frame)
self._prev_action = decision.action
```

No `TargetState`, no TSA import, no threshold state machine, no `consecutive_lost`,
no `consecutive_occluded`, no OCCLUDED escalation, no `stage3_policy`. The grep for
all forbidden patterns returns zero hits in `salt_runner.py`.

Recovery is now triggered by `_consecutive_recovery_frames` (controller counts frames
where model asked for recovery), not by TSA LOST counter. Template update is gated by
`decision.action.template == TemplateAction.UPDATE`. Telemetry emits SALT-RD action
fields only. `test_no_tsa_runtime.py` tests 1–2 and 4 now pass. Test 3
(`test_update_with_action_called_in_runner`) passes too.

### [OK] `registry.py` `TSA_ASSESSORS` removed

Correct Phase 4 cleanup. No functional impact but eliminates a dead registry entry.

### [OK] `test_no_tsa_runtime.py` — all 4 assertions now pass

Previous pass reported all 4 failing. Current code satisfies:
1. No TSA import in runner — PASS
2. No forbidden patterns (`consecutive_lost`, `update_with_state`, etc.) — PASS
3. `update_with_action` called in runner — PASS
4. SALT-RD telemetry fields present — PASS

---

### [WRONG DIRECTION-3] `_advisory_p_fc < 0.30` gate still controls reference embedding update

`salt_runner.py` line 409:
```python
and _advisory_p_fc < 0.30
```

This is a `p_fc` threshold controlling whether the EMA reference embedding is updated.
`_advisory_p_fc` comes from `self.tracker._salt_rd_advisor.last_p_fc` — the old risk
model's probability score used as a threshold gate on a behavioral decision.

This is the exact pattern SUPER_PLAN Section "Forbidden Patterns" prohibits:
```python
if p_fc > ...   # forbidden in production control paths
```

The reference embedding gate must be either:
- always enabled on confident frames (remove p_fc gate entirely), or
- gated by a learned controller signal from `decision.risk_probs["false_confirmed"]`
  without a hardcoded threshold.

Until the learned model outputs risk_probs, remove the `_advisory_p_fc` gate and gate
only on `track_state.confidence`. The embedding update is a passive telemetry operation,
not a tracking decision.

### [WRONG DIRECTION-4] Recovery spatial constraint still comes from advisor threshold state

`salt_runner.py` line 462:
```python
_recovery_crop = _advisor.get_recovery_crop_bbox(_fh, _fw, expand_factor=3.0)
```

The advisor's `get_recovery_crop_bbox()` method computes a spatial crop region from its
internal threshold-based monitoring of p_fc / APCE. This is a threshold rule that
determines *where* the detector searches — a spatial policy decision. Introducing a crop
constraint via the advisor while recovery *triggering* is now controller-driven is an
inconsistency: the controller decides to recover, but the advisor decides where.

The spatial crop should come from `decision.action.bbox_hint` (the controller's selected
candidate position) or from `decision.action.detector_hint`. If no hint is available, the
detector should run on the full frame. The advisor crop path must be removed.

**Required fix:** Replace `_advisor.get_recovery_crop_bbox()` with:
```python
_recovery_hint_bbox = decision.action.detector_hint or decision.action.bbox_hint
# if hint: run detector in crop around hint; if None: run full frame
```

### [BUG-2] Feature vector is 9 / 28 populated — model will see near-zero input

`salt_runner.py` lines 358–368 populate only indices 0–8 (APCE, PSR, entropy, 5 score-map
stats). Indices 9–21 (rolling evidence and bbox dynamics) remain zero:

| Index | Name | Populated? |
|---:|---|---|
| 9 | apce_ratio_5 | **zero** |
| 10 | apce_ratio_20 | **zero** |
| 11 | entropy_delta_5 | **zero** |
| 12 | peak_margin_delta_5 | **zero** |
| 13 | high_apce_streak_legacy | **zero** |
| 14 | low_apce_streak_legacy | **zero** |
| 15–21 | bbox_vx/vy/speed/accel/scale/aspect/border | **zero** |

These 13 features were computed by `collect_features.py` during training. At runtime they
are all zero. The model was trained on non-zero rolling and motion features; at inference
it receives a near-zero vector. This is a train/runtime feature mismatch and will hurt
model calibration directly.

**Required fix:** Port the rolling feature computation from `collect_features.py` into
`EvidenceExtractor`. The extractor already maintains `_feature_history` and
`_bbox_history` buffers — use them to compute `apce_ratio_5`, `entropy_delta_5`,
`bbox_vx/vy`, etc., and return the fully populated 28-dim vector. The runner should not
compute features — it should pass raw tracker outputs and let the extractor do all the
feature math.

### [WRONG DIRECTION-5] `update_with_action()` still a stub (unchanged from Pass 1)

Already documented. No change since last review. Search, template, recovery sub-actions
are ignored. Only `ComputeAction.FULL` has any effect.

---

### Updated state summary

| Component | Status |
|---|---|
| `actions.py` | OK |
| `feature_schema.py` | OK |
| `evidence.py` | OK — BUG-1 (prev_bbox) unresolved |
| `controller.py` | OK |
| `salt_runner._step()` | OK — rewritten, no TSA, controller-driven |
| `salt_runner` advisor p_fc gate | WRONG DIRECTION-3 |
| `salt_runner` advisor crop in recovery | WRONG DIRECTION-4 |
| Feature vector completeness | BUG-2 — 9/28 populated |
| `sglatrack.update_with_action()` | WRONG DIRECTION-5 — stub |
| `sglatrack.update_with_state()` | Deprecated stub, tolerated short-term |
| TSA module tree | OK — deleted |
| `registry.py` | OK — TSA_ASSESSORS removed |
| `test_no_tsa_runtime.py` | OK — all 4 assertions now pass |

---

## Review — 2026-05-22 Pass 4 (10-min cycle)

### [OK] WRONG DIRECTION-3 resolved — p_fc EMA gate removed

`salt_runner.py` EMA embedding update block no longer contains `_advisory_p_fc < 0.30`.
Gate is now purely `track_state.confidence >= 0.014`. `_advisory_p_fc` is read once and
used only as a passive telemetry field (`_aux["salt_rd_p_fc"]`). No threshold behavior.

### [OK] WRONG DIRECTION-4 resolved — recovery crop now from controller hint

`_advisor.get_recovery_crop_bbox()` is gone. Recovery crop now derives from:
```python
_recovery_hint = decision.action.detector_hint or decision.action.bbox_hint
```
If the controller provides no hint, the detector runs on the full frame. Spatial recovery
policy is now fully controller-owned, not advisor-threshold-owned.

### [OK] BUG-1 resolved — prev_bbox captured before history append

`evidence.py` `step()` now saves `prev_bbox` before calling `self._bbox_history.append(bbox)`,
then passes it explicitly to `_parse_candidates(..., prev_bbox=prev_bbox)`. The
`distance_to_prev_bbox` feature now carries correct information.

### [OK] BUG-2 resolved — all 28 features populated via `_compute_rolling_features()`

`EvidenceExtractor._compute_rolling_features()` added. Computes all previously-zero
indices from history buffers:

| Indices | Features | Method |
|---|---|---|
| 9–12 | apce_ratio_5, apce_ratio_20, entropy_delta_5, peak_margin_delta_5 | window mean over history deque |
| 13–14 | high/low APCE streak (v2 parity) | reverse scan over history |
| 15–21 | vx, vy, speed, accel, scale_ratio, aspect_delta, dist_to_border | bbox history diff |

Called inside `step()` after appending current features/bbox to history deque.
Feature parity with `collect_features.py` training pipeline is now restored for all
indices except 22–27 (flow, intentionally zero in production).

### [OK] All 46 unit tests pass

`test_saltr_actions`, `test_saltr_controller`, `test_saltr_feature_schema`,
`test_saltr_evidence`, `test_no_tsa_runtime` — 46/46 in 0.05s.

### [BUG-3] `image_shape` not passed to extractor — `dist_to_border` (index 21) always 0.0

`EvidenceExtractor.step()` accepts `image_shape: tuple[int, int] | None = None` and
uses it to compute `dist_to_border` (index 21). The runner call at line 371–376 does
not pass `image_shape`:

```python
evidence_frame = self.evidence_extractor.step(
    base_features=features,
    bbox=bbox_tuple,
    score_map_stats=score_map_stats,
    candidates=candidates_raw,
    # image_shape missing — dist_to_border will be 0.0 every frame
)
```

The frame dimensions are available as `frame.shape[:2]` at that call site. This is a
one-line fix.

**Required fix:**
```python
evidence_frame = self.evidence_extractor.step(
    base_features=features,
    bbox=bbox_tuple,
    score_map_stats=score_map_stats,
    candidates=candidates_raw,
    image_shape=frame.shape[:2],
)
```

`dist_to_border` being permanently zero is tolerable short-term (it was also zero under
BUG-2), but should be fixed before any training run that uses border-proximity as a
recovery signal.

---

### Updated state summary

| Component | Status |
|---|---|
| `actions.py` | OK |
| `feature_schema.py` | OK |
| `evidence.py` | OK — BUG-3 (dist_to_border zero) minor |
| `controller.py` | OK |
| `salt_runner._step()` | OK — no TSA, controller-driven, advisor threshold calls removed |
| Feature vector completeness | OK — 27/28 populated (index 21 zero pending BUG-3 fix) |
| `sglatrack.update_with_action()` | WRONG DIRECTION-5 — stub, ignores search/template/recovery |
| `sglatrack.update_with_state()` | Deprecated stub, tolerated short-term |
| TSA module tree | OK — deleted |
| `registry.py` | OK |
| Unit tests | OK — 46/46 pass |

---

## Review — 2026-05-22 Pass 5 (10-min cycle)

### [OK] `saltr/data` restored — Phase 0 blocker resolved

`saltr/data/` is now a real directory containing `salt_rd_v2_labels.npz` (17.6 MB),
restored from `saltr/tmp/oof/fold_00.npz`. The circular symlink that blocked oracle
rerun and training is gone.

Caveat: fold_00 may not include all 18 hard sequences. 4 sequences (bike2, Gull2,
Sheep1, StreetBasketball1) were absent from the prior oracle audit. They may live in
val/diagnostic splits of other folds. Verification needed before starting Phase 6.

### [OK] `oracle_action_audit_full.json` — reinit oracle confirmed clean

224 sequences audited. Reinit oracle: +0.083 hard (14 seqs), +0.025 full, **zero
harmful sequences**. All other actions (search_expand, template_update, center_freeze)
confirmed KILL. The signal is clean and the ceiling is above the +0.010 full-set gate.

Per-sequence the biggest gains are uav6 (+0.229), uav5 (+0.179), uav4 (+0.164) —
exactly the sequences where Phase 8 TSA routing also moved AUC. This validates the
causal path.

### [OK] Training Run 0 in progress — epoch 5 best, metrics ahead of target

`production_no_flow/saltrd_best.pt` at epoch 5:
- Val fc AUROC: **0.885**, AUPRC: **0.363** — already exceeds the prior ablation
  target (0.883 / 0.350). Good sign; training to 60 epochs should hold or improve.
- `drop_feature_indices`: [22,23,24,25,26,27] — correctly saved.

### [BUG-4] `feature_schema` string not saved in checkpoint

`train.py` saves `drop_feature_indices` but not `feature_schema`. Checkpoint inspection
shows `feature_schema: NOT SET`. SUPER_PLAN Section B and D require:
```json
"feature_schema": "saltrd_v3_no_tsa_no_flow"
```
Without this, any code that validates checkpoint schema by name will fail or silently
skip validation. Proposal: add one line to `train.py` checkpoint save block.

### [WRONG DIRECTION-6] `phase4_recommendation.next` says "rule-based reinit"

`oracle_action_audit_full.json` line:
```json
"next": "Implement conservative rule-based reinit (Phase 5)"
```
This is the oracle script's auto-generated recommendation, not the plan. SUPER_PLAN
explicitly forbids rule-based reinit. The correct next step is Phase 5 oracle label
generation followed by Phase 6 learned policy training. Do not implement rule-based
reinit under any name.

### [BUG-3] `image_shape` still not passed — `dist_to_border` (index 21) still zero

No change since Pass 4. `dist_to_border` remains zero at runtime. Low urgency but
should be proposed before Phase 6 training begins, so the feature is non-zero in the
rollout distribution that generates training feedback.

---

### Updated state summary

| Component | Status |
|---|---|
| `actions.py` | OK |
| `feature_schema.py` | OK |
| `evidence.py` | OK — BUG-3 (dist_to_border zero, minor) |
| `controller.py` | OK |
| `salt_runner._step()` | OK — controller-driven, TSA-free |
| Feature vector completeness | OK — 27/28 (index 21 zero, BUG-3) |
| `sglatrack.update_with_action()` | WRONG DIRECTION-5 — stub |
| `sglatrack.update_with_state()` | Deprecated, tolerated |
| TSA module tree | OK — deleted |
| `registry.py` | OK |
| `saltr/data` | OK — restored |
| Oracle audit | OK — confirmed reinit, zero harmful |
| Training Run 0 | IN PROGRESS — epoch 5, metrics on target |
| Checkpoint `feature_schema` | BUG-4 — NOT SET, proposal pending |
| Rule-based reinit risk | WRONG DIRECTION-6 — oracle script says rule-based; ignore |
| Unit tests | OK — 46/46 pass |

---

## Review — 2026-05-22 Pass 6

### [OK] WRONG DIRECTION-5 resolved — `update_with_action()` is now fully wired

`sglatrack.py` `update_with_action()` now dispatches:
- `CENTER_ON_REINIT_HINT` + `bbox_hint` → `override_search_center(cx, cy, w, h)`
- `FREEZE` → `override_search_center()` on `self._state` (last known position)
- `KEEP` / `EXPAND` → default tracker search (EXPAND is KILL per plan, so no-op is correct)
- Import of `salt_r.actions` is lazy inside the method body → coupling issue resolved

Template action dispatch is intentionally absent from `update_with_action()` — it is
handled entirely in the runner. Recovery is handled entirely in the runner. This matches
the plan architecture. 9 action API tests pass.

### [OK] Training Run 0 complete — checkpoint metrics on target

`production_no_flow/saltrd_best.pt` (epoch 5, best val):
- Val fc AUROC: **0.8854**, AUPRC: **0.3611** (eval) — exceeds prior ablation target
- `drop_feature_indices` [22–27] correctly saved
- Early stopping at epoch 13 (val AUPRC peaked epoch ~5–6 then declined)

Note: commit message states "best val AUPRC = 0.3775" but checkpoint inspection and
eval both show 0.3634 / 0.3611. One-off documentation discrepancy; checkpoint and eval
results are authoritative.

### [BUG-4] `feature_schema` still NOT SET in checkpoint (unchanged)

No fix landed in commit 02783af or 5b4b686. Train.py saves `drop_feature_indices` but
not `feature_schema = "saltrd_v3_no_tsa_no_flow"`. Runtime schema validation has a
blind spot. Proposal from Pass 5 still stands.

### [BLOCKER-NEW] 4 hard diagnostic sequences absent from ALL 5 OOF folds — data pipeline root cause identified

All 5 folds (fold_00 through fold_04) are missing `uav123/bike2`, `dtb70/Gull2`,
`dtb70/Sheep1`, `dtb70/StreetBasketball1`. Root cause confirmed in
`saltr/src/salt_r/collect_features.py` lines 172–179:

```python
DIAGNOSTIC_SEQUENCES = frozenset({
    "uav0000164",
    "bike2",          # UAV123: identity-loss hard case
    "Gull2",          # DTB70: hard case
    "Sheep1",         # DTB70: hard case
    "StreetBasketball1",  # DTB70: hard case
})
```

These sequences are permanently assigned to the "diagnostic" split and are **excluded
from OOF fold rotation by design**. They exist only in the original full-dataset NPZ
(the pre-symlink `salt_rd_v2_labels.npz`) which was lost when the circular symlink was
introduced and only fold_00 was restored.

Consequence: oracle audit ran on 220 sequences (224 minus 4 diagnostic); oracle reinit
ceiling is +0.083 on 14 non-diagnostic hard sequences. The 4 diagnostic sequences
(hardest identity/distractor cases) contribute zero to training loss and zero to oracle
calibration. A learned policy trained on fold_00 has no opportunity to learn reinit for
distractor scenes.

**Two recovery paths:**

Option A — Restore original NPZ from backup/raw extraction:
```bash
# Find if original full NPZ exists elsewhere
find . -name "salt_rd_v2*.npz" -not -path "*/oof/*" 2>/dev/null
find . -name "v2_corrected*.npz" -o -name "v2_labels*.npz" 2>/dev/null | grep -v oof
```

Option B — Re-extract diagnostic sequences from raw video:
```bash
# collect_features.py can extract individual sequences if raw data is available
PYTHONPATH=src:saltr/src .venv/bin/python saltr/src/salt_r/collect_features.py \
  --sequences uav123/bike2 dtb70/Gull2 dtb70/Sheep1 dtb70/StreetBasketball1 \
  --output saltr/data/diagnostic_sequences.npz
# Then merge with salt_rd_v2_labels.npz
```

This is a BLOCKER for Phase 5 (oracle reinit labels) — training without these sequences
will produce a model blind to the hardest recovery scenarios.

### [BUG-5] Diagnostic AUROC 0.911 is not comparable to prior ablation 0.697

`eval_diagnostic_production_no_flow.json`: fc AUROC = **0.911**. Prior ablation target
was **0.697**. The delta is +0.214 AUROC, which cannot be explained by model improvement
alone. Root cause: fold_00 "diagnostic" split (37 sequences) does NOT include the 4
hardest DIAGNOSTIC_SEQUENCES — they are absent from the NPZ entirely. The diagnostic
split in fold_00 is a weaker, non-canonical subset. The 0.697 result was measured on
the true diagnostic split that included bike2/Gull2/Sheep1/StreetBasketball1.

**Do not use 0.911 as a benchmark claim.** Any paper result must be measured on the
canonical diagnostic split with all 4 hard sequences present.

---

### Updated state summary

| Component | Status |
|---|---|
| `actions.py` | OK |
| `feature_schema.py` | OK |
| `evidence.py` | OK — BUG-3 (dist_to_border, minor) |
| `controller.py` | OK |
| `salt_runner._step()` | OK — controller-driven, TSA-free |
| `sglatrack.update_with_action()` | OK — SearchAction wired, lazy import |
| TSA module tree | OK — deleted and archived |
| Training Run 0 | OK — complete, val AUROC 0.885 / AUPRC 0.361 |
| Checkpoint `feature_schema` | BUG-4 — NOT SET |
| Diagnostic AUROC 0.911 | BUG-5 — inflated, non-canonical split |
| 4 missing hard sequences | BLOCKER — absent from all 5 folds, data restore needed |
| Oracle ceiling | PARTIAL — +0.083 on 14/18 seqs, must rerun after data restore |
| Unit tests | OK — 46+ passing |

---

## Review — 2026-05-22 Pass 7

### [OK] Phase 5–9 pipeline committed (commit e6d8db8)

New scripts: `oracle_actions.py`, `policy_model.py`, `train_policy.py`,
`calibrate_policy.py`, `rollout_policy.py`. 449 tests passing (22 new).

### [OK] BUG-4 resolved in policy checkpoint

`saltr/checkpoints/policy_reinit_v1/saltrd_policy_best.pt` contains
`feature_schema: saltrd_v3_no_tsa_no_flow`. The policy training script saves schema
correctly. Note: the production_no_flow risk checkpoint (train.py) still does NOT save
`feature_schema` — that is a separate issue in `train.py`, not fixed here.

### [OK] Oracle labels generated — GO signal confirmed

`reinit_oracle_dataset.npz`: 155,375 frames, 3,845 reinit positives = **2.47%** base
rate. Exceeds the SUPER_PLAN stop condition threshold of 0.5%. GO on training.

Splits:
- training ("diagnostic"): 125,463 frames, 2,843 reinit positives (2.27%)
- validation ("val"): 29,912 frames, 1,002 reinit positives (3.35%)

### [BUG-7] `train_policy.py` split name fix applied in working tree

Original code used `split="train"` but oracle NPZ only has "diagnostic" and "val"
splits. Fixed to `split="diagnostic"` in the working tree. Not yet committed. Training
is proceeding correctly with this fix.

### [BLOCKER-TRAINING] Policy epoch 3: reinit_recall=0.005 — near-zero, monitoring required

Checkpoint progression:
| Epoch | macro_F1 | reinit_recall | reject_prec | val_loss |
|---:|---:|---:|---:|---:|
| 1 | 0.2452 | 0.0000 | 0.9620 | 0.1664 |
| 3 | 0.2477 | 0.0050 | 0.9621 | 0.1509 |

Training continues to epoch 80 (patience=10). Class weights are correctly set:
`w_reinit ≈ 40` (inverse frequency, capped at 50), `w_reject ≈ 0.025`.

The model IS learning (recall went from 0 to 0.005), but very slowly. The
reinit recall gate is **>= 0.40** per SUPER_PLAN. Monitor epoch 10–20 for trajectory.

**Decision rule:** if reinit_recall < 0.05 at epoch 10, training is failing to
overcome the imbalance and interventions are needed (see BUG-8 remediation below).
If recall >= 0.10 by epoch 20, on track.

### [BUG-8] Oracle label skew: 97% reject — candidate quality problem on hard sequences

Oracle dataset class distribution:
- REINIT: 2.47% (3,845 frames)
- REJECT_REINIT: 97.01% (150,724 frames)
- NONE: 0.52% (~806 frames)

Nearly every frame evaluated for recovery is labeled REJECT_REINIT because available
candidates (score-map / detector) are not close enough to the target to produce positive
utility. Hard sequences (uav3, uav4, uav5, uav6, uav7) all have **negative max_utility**:

| Sequence | Max utility | Oracle audit gain | Root cause |
|---|---:|---:|---|
| uav123/uav3 | -0.050 | +0.088 | no adequate detector candidates |
| uav123/uav4 | -0.042 | +0.164 | no adequate detector candidates |
| uav123/uav5 | +0.037 | +0.179 | minimal candidate coverage |
| uav123/uav6 | -0.045 | +0.229 | no adequate detector candidates |
| uav123/uav7 | -0.049 | +0.067 | no adequate detector candidates |

**Interpretation:** the oracle audit uses "reinit to GT bbox" (oracle-best). The oracle
label dataset uses "reinit to best available candidate bbox." For uav3-7, no available
candidate is close enough to GT to produce positive utility — so these sequences
contribute ZERO reinit-positive labels. A policy trained on this dataset will learn to
reinit in easy cases (car7, truck2, person sequences) but will never fire on the hardest
UAV identity-loss scenes.

**Required fix (after monitoring current training run):**
1. Improve detector candidate recall on uav-class sequences (detector fine-tune or
   score-map top-k candidates instead of detector-only)
2. OR re-generate oracle labels including score-map derived candidates, not only detector
3. OR use the oracle audit GT bbox as the reinit target in labels (upper-bound training signal)

This is the most important open issue for reaching the +0.10 hard AUC gate.

---

### Updated state summary

| Component | Status |
|---|---|
| Phase 1–4 architecture | OK — TSA-free, controller-driven |
| Phase 5 oracle labels | OK — generated, 2.47% base rate |
| Phase 6 policy model | OK — SALTRDPolicyNet with GRU, action heads |
| train_policy.py split fix | BUG-7 — in working tree, needs commit |
| Policy training | IN PROGRESS — epoch 3/80, reinit_recall=0.005 |
| Reinit recall on hard seqs | BUG-8 — uav3-7 have zero candidate-based labels (point tracker needed) |
| Candidate quality (uav class) | BUG-8 — YOLO misses small UAVs; point-tracker candidates are the fix |
| `feature_schema` in risk ckpt | BUG-4 — still NOT SET in production_no_flow/saltrd_best.pt |
| 4 diagnostic hold-out seqs | OK — intentional by design; evaluate separately as blind test |
| Unit tests | OK — 449 passing |

---

## Correction — Pass 7 diagnostic sequences framing

The BLOCKER tag applied to the 4 missing sequences (bike2, Gull2, Sheep1,
StreetBasketball1) was **wrong framing**. HANDOFF_NEXT.md confirms these are
intentional diagnostic hold-outs excluded from fold rotation by design. Correct approach:

- **Train** on 220 sequences (canonical folds)
- **Evaluate separately** on the 4 diagnostic hold-outs as a blind generalization test
- **Do not include** them in training decisions or oracle labeling
- **Do not block** training because they are absent from folds

The "BLOCKER" in the previous pass was tunnel vision. These sequences are the hardest
stress test, not a missing data problem.

### Broader picture from HANDOFF_NEXT.md

The candidate quality problem on uav3-7 (zero YOLO candidates close to target) is
already the known open problem from prior sessions. The HANDOFF documents the solution
direction:

> Next work should move to **external/teacher candidate verification**:
> CoTracker3/TAPIR point consistency + candidate-aware DINO/DAM-style memory

Relevant papers in `papers/`:
- `05_CoTracker3_Pseudo_Labeling_Real_Videos.pdf` — point tracking with temporal consistency; can track the target point even when bbox is uncertain
- `06_DINO_Tracker_Taming_DINO_for_Self_Supervised_Point_Tracking.pdf` — frozen DINO features for identity-consistent point tracking
- `04_Verifier_Guided_Pseudo_Labeling_Point_Tracking.pdf` — verification framework for point tracking labels

These point-tracking approaches generate candidates that are temporally consistent
(unlike one-shot YOLO detections) and could fill the candidate gap for uav3-7.

The current BUG-8 (near-zero YOLO candidates for UAV targets) is a valid issue that
will limit AUC gain on the hardest sequences. But it is not a blocker for training —
the current oracle labels cover 220 sequences with real positive signal. A model trained
on these will generalize to uav-class scenes as a first test, and point-tracker
candidates can be added in a second iteration.

### BUG-8 Revised: approach, not blocker

The correct framing:
1. **Current training run** (Run 1): train on 220-sequence oracle labels using YOLO candidates. This will achieve recall on "easy" reinit cases (car, person, group scenes).
2. **Evaluate on uav hard set**: run rollout on uav2-uav8, expect partial gain.
3. **Gap analysis**: if hard UAV sequences show zero bbox changes → confirm candidate quality is the bottleneck.
4. **Next iteration** (Run 2): add CoTracker3/score-map candidates to oracle labels, retrain. Expect gain on uav3-7.

This two-run strategy is cleaner than blocking on candidate quality before any training signal.

---

## Dead Code Audit — 2026-05-22

Full dead-code scan across `src/uav_tracker/` and `saltr/src/salt_r/`. Move files tagged **UNUSED** to `./garbage/`. Files tagged **TEST-ONLY** or **SCRIPT-ONLY** may remain but are not production-critical.

### Task for engineer: move to `./garbage/`

These have zero callers in production and no test coverage worth keeping:

| File | What it defines | Why dead |
|---|---|---|
| `src/uav_tracker/ml/ttt/__init__.py` | empty namespace | TTT feature killed; `test_no_tsa_runtime.py` bans it |
| `src/uav_tracker/ml/ttt/head_adaptor.py` | `HeadAdaptor`, loss functions | TTT never instantiated anywhere |
| `src/uav_tracker/ml/difficulty_predictor/base.py` | `DifficultyPredictor` Protocol | No caller in any test or script |
| `src/uav_tracker/ml/difficulty_predictor/regression_predictor.py` | `MLPDifficultyPredictor` | Registry entry never fires |
| `src/uav_tracker/ml/difficulty_predictor/__init__.py` | re-exports above | No caller |
| `src/uav_tracker/training/augmentation.py` | `UAVAugmentPipeline` | Never called by any file anywhere |

### TEST-ONLY — keep but do not extend

These are used by tests or were part of v2 pipeline that is now superseded:

| File | Note |
|---|---|
| `src/uav_tracker/ml/scene_classifier/` (whole dir) | Disabled in `_PLUGIN_MODULES`; v2 only |
| `src/uav_tracker/ml/warmer/` (whole dir) | Disabled in `_PLUGIN_MODULES`; v2 only |
| `src/uav_tracker/schedulers/ml_scene_scheduler.py` | Commented out of production init |
| `src/uav_tracker/signals/tracker_confidence.py` | Registered but never built in production |
| `src/uav_tracker/signals/motion_entropy.py` | V2 pipeline; superseded by SALT-RD features |
| `src/uav_tracker/detectors/yolo.py` | Registration commented out; replaced by yolo26m |
| `saltr/src/salt_r/policy.py` | Old TrackerAction; superseded by `actions.py` |
| `saltr/src/salt_r/interventions.py` | Old action types; superseded by `actions.py` |
| `saltr/src/salt_r/integrate.py` | Old integration wrapper; superseded by `controller.py` |
| `saltr/src/salt_r/memory.py` | Proxy memory approach (killed after ablation) |
| `saltr/src/salt_r/memory_features.py` | Companion to `memory.py` |
| `saltr/src/salt_r/eprocess.py` | Analysis only; 3.1% recall, not a runtime gate |
| `saltr/src/salt_r/policy_sweep.py` | V2 policy sweep; superseded |
| `saltr/src/salt_r/make_oof_predictions.py` | OOF helper; test-only |

### SCRIPT-ONLY — keep for reproducibility

These are standalone CLI scripts, valid as research tools:

`advisor.py`, `shadow_mode.py`, `center_freeze_sweep.py`, `action_audit.py`,
`oracle_action_audit.py`, `oracle_actions.py`, `make_safe_to_update_labels.py`,
`baselines.py`, `diagnose_labels.py`, `train_policy.py`, `calibrate_policy.py`,
`rollout_policy.py`, `src/uav_tracker/training/label_generator.py`,
`src/uav_tracker/datasets/uav123_ml.py`

### USED in production

`actions.py`, `controller.py`, `evidence.py`, `feature_schema.py`, `model.py`,
`policy_model.py`, `train.py`, `eval.py`, `collect_features.py`,
`src/uav_tracker/ml/motion_predictor/lstm_predictor.py` (disabled by config but code path exists),
`src/uav_tracker/ml/appearance_memory/cosine_memory.py`,
`src/uav_tracker/schedulers/multi_tier.py`,
`src/uav_tracker/signals/optical_flow.py`,
`src/uav_tracker/signals/global_motion.py`

---

## HANDOFF_NEXT Approach Analysis — What Worked, What Didn't, What to Try Next

### ✅ Worked

| Approach | Result | Key number |
|---|---|---|
| 28-dim telemetry GRU risk model (v2_retrained) | Valid false-confirmed signal | Val fc AUROC 0.885, Diag 0.598 |
| Proxy memory pos-only (RAM) | +0.176 diag AUROC gain | Diag 0.598 → 0.774 (89% of full memory gain) |
| LODO cross-dataset generalization | All 3 held-out pass gate | 0.939/0.608/0.802 >> 0.598 baseline |
| Stage 2 advisory/veto | wrir=0, msu=0.081 | GO confirmed |
| No-flow schema (v3) | +0.099 diag AUROC vs v2_retrained | Diag 0.598 → 0.697 (ablation); 0.911 on weak fold |
| LK point tracking gate | Passes diagnostic gate 0.65 | `pt_inside_pred_ratio` AUROC 0.729 on diagnostic |
| Oracle reinit audit | Confirms reinit is the only AUC path | +0.083 hard AUC oracle gain |
| Phase 1–4 SALT-RD architecture | TSA removed, controller loop clean | 449 tests passing |

### ❌ Killed / Did Not Work

| Approach | Result | Why killed |
|---|---|---|
| Real SGLATrack embeddings (score_weighted) | Diag AUROC 0.584 < 0.598 baseline | DeiT-tiny search tokens = localization, not identity |
| Real SGLATrack embeddings (peak_local) | Identical to score_weighted | Same root cause |
| DINOv2 ROI identity (CLS/patch_mean, 5 variants) | Diag 0.514–0.563, all FAIL gate | Generic crop similarity fails on Gull2/organic scenes |
| SGLATrack top-K score-map candidates | Overall top-5 recall@0.3 = 0.298; bike2/Gull2/SB1 = 0.000 | Target absent from score map during false-confirmed |
| CoTracker3 point teacher | KILLED — 8–16 GB RAM crash, marginal +0.019 over LK | Not edge-deployable |
| LK point sidecar + v2.3 training (33-dim) | Diag AUROC 0.546 < 0.598 baseline | LK features don't generalize from UAV123 train → hard DTB70 diag |
| Full memory (37-dim, neg+pos) | Diag 0.796 but val AUPRC 0.243 (too low) | Best on diag, worst on val — overfit signal |
| Negative memory only | Diag AUROC 0.496 (sub-random) | Harmful without positive context |
| TSA-based SALT-RD Stage 3 routing | Hard AUC +0.046 but rejected architecture | Uses TargetState integers and handcrafted thresholds |
| Phase 7 p_fc/APCE threshold recovery | Hard AUC -0.036 regression | Threshold policy causes regressions on standard sequences |
| Center-freeze | ~0.000 hard AUC gain | Target already moved far from frozen position at FC time |
| Advisory mode delta | Full UAV123: baseline=0.673, advisory=0.673 | Advisory/veto has no trajectory effect without action execution |

### ⏳ Next approaches to try (from papers/)

Current training (Run 1) will reveal the candidate quality gap. If uav3-7 show zero bbox
changes in rollout, the following approaches address it (in priority order):

| Approach | Paper | Why promising | Estimated effort |
|---|---|---|---|
| **Score-map top-k as reinit target** | — (internal) | Score map has target when YOLO doesn't; already extracted | Low — modify oracle_actions.py |
| **LK point consistency as reinit trigger** | `papers/05_CoTracker3_*.pdf` (LK proxy) | `pt_inside_pred_ratio` AUROC 0.729 on diag; passed Phase 5A gate | Medium — wire into oracle label generation |
| **Verifier-guided pseudo-labeling** | `papers/04_Verifier_Guided_Pseudo_Labeling_Point_Tracking.pdf` | Point tracks verify candidates temporally | Medium — implement teacher offline |
| **DINO-Tracker frozen features for candidate identity** | `papers/06_DINO_Tracker_*.pdf` | Frozen DINO for self-supervised point tracking; no finetuning | Medium — need to confirm Gull2 case |
| **DAM4SAM-style target-vs-distractor margin** | `papers/DAM4SAM_*.pdf` (reference_dam4sam_paper.md) | Lightweight distractor-aware memory, not full SAM2 | High — needs candidate source first |
| **ORTrack backbone (occlusion-robust ViT)** | `papers/08_ORTrack_*.pdf` | Better base tracker for hard UAV occlusion scenes | High — full tracker replacement |

**Red lines from HANDOFF_NEXT (permanent):**
- No training on diagnostic sequences (bike2, Gull2, Sheep1, StreetBasketball1, uav0000164)
- No negative memory
- No CoTracker3 (RAM crash)
- No global entropy threshold (dataset shift, inverted on diagnostic)
- No temperature scaling as template-update unlock
- v2_retrained = strict baseline for all gates (diag 0.598, val 0.885)
- Per-dataset reporting mandatory for all GO/KILL decisions

---

## Task: Full Benchmark + Video + FPS/GFLOPs (for engineer)

**Priority:** medium — needed before any paper submission claim.

### What needs to be built / run

1. **Full UAV123 benchmark** (all 123 sequences, no frame cap):
   - Already has `fast_bench.py` with `--all-sequences --max-frames 0`
   - Needs to run against: SGLATrack baseline, SALT-RD no-flow risk-only, SALT-RD learned reinit v1
   - Output: per-sequence AUC table + hard subset delta

2. **VisDrone-SOT and DTB70 benchmarks** (new, not yet run):
   - Same `fast_bench.py` or new `bench_visdrone.py` / `bench_dtb70.py`
   - Required for paper (per-dataset reporting is mandatory per HANDOFF_NEXT red lines)

3. **FPS measurement**:
   - Tracker-only FPS (SGLATrack full compute, CE kr=0.50)
   - SALT-RD inference overhead (GRU + evidence extractor per frame)
   - Total system FPS
   - Measure on Apple MPS and CPU separately

4. **GFLOPs per frame**:
   - SGLATrack forward pass GFLOPs (full vs CE pruned)
   - SALT-RD policy model GFLOPs
   - Total per-frame budget
   - `thop` or `fvcore` for flop counting

5. **Visualization video**:
   - Overlay: predicted bbox (green), GT bbox (gray), SALT-RD action indicator (color coded)
   - Action color code: FULL=white, REINIT=red, SCORE_CANDIDATES=orange, NOOP=none
   - Per-frame telemetry: APCE, p_fc, reinit_recall, changed_bbox
   - Relevant sequences: uav6 (+0.229 reinit oracle), uav4, uav8, car7, bike2
   - Output: MP4, 30fps, 640×480 or 1280×720

### Code that already exists and should be extended

- `scripts/fast_bench.py` — baseline + advisory benchmark (adapt for SALT-RD learned controller)
- `scripts/hard_bench.py` — hard subset benchmark (adapt for policy rollout)
- `saltr/src/salt_r/rollout_policy.py` — already generates per-sequence AUC delta (use as basis for benchmark)
- `src/uav_tracker/trackers/sglatrack.py` — has FPS timing in `update()` already

### Proposed new script: `scripts/benchmark_full.py`

```python
# Usage:
# PYTHONPATH=src:saltr/src .venv/bin/python scripts/benchmark_full.py \
#   --dataset uav123 --split all --no-cap \
#   --checkpoint saltr/checkpoints/policy_reinit_v1/saltrd_policy_best.pt \
#   --output saltr/results/benchmark_uav123_policy_reinit_v1.json \
#   --report-fps --report-gflops

# Outputs per-sequence: AUC, Pr@20, FPS, changed_bbox_frames, action_distribution
# Outputs aggregate: hard_mean_auc, full_mean_auc, fps_mean, gflops_mean
```

### Proposed new script: `scripts/visualize_sequence.py`

```python
# Usage:
# PYTHONPATH=src:saltr/src .venv/bin/python scripts/visualize_sequence.py \
#   --sequence uav123/uav6 \
#   --checkpoint saltr/checkpoints/policy_reinit_v1/saltrd_policy_best.pt \
#   --output saltr/results/viz_uav6_policy_reinit_v1.mp4 \
#   --show-actions --show-telemetry
```

---

## Requirement: Per-Dataset Training, Eval, Calibration, Results

HANDOFF_NEXT red line: **pooled val is not enough; per-dataset reporting is mandatory.**

### What this means for every experiment going forward

Every `eval.py` run must produce and report:

| Dataset | AUC gate | Notes |
|---|---|---|
| UAV123 | Primary benchmark; 123 seqs | Full + hard subset |
| VisDrone-SOT | Secondary; drone footage | Different domain from UAV123 |
| DTB70 | Hard; includes Gull2/Sheep1 | Lowest AUROC typically |

**Required fields in every result JSON:**
```json
{
  "head_metrics": { "false_confirmed": {"auroc": ..., "auprc": ...} },
  "per_dataset_head_metrics": {
    "uav123":      {"false_confirmed": {"auroc": ..., "auprc": ...}},
    "visdrone_sot": {"false_confirmed": {"auroc": ..., "auprc": ...}},
    "dtb70":       {"false_confirmed": {"auroc": ..., "auprc": ...}}
  }
}
```

**Gate: a result only counts as GO if ALL THREE datasets pass the gate, not just the pooled average.**

### Training: LODO already validates this

From HANDOFF_NEXT: LODO (leave-one-dataset-out) models all pass per-dataset gates:
- LODO no-UAV123: UAV123 OOD AUROC = 0.939 (generalization confirmed)
- LODO no-DTB70: DTB70 OOD AUROC = 0.608 (lowest; expected due to harder dynamics)
- LODO no-VisDrone: VisDrone OOD AUROC = 0.802

For the **policy model** (Train Run 1+), per-dataset results are not yet available.
Required before calling any policy training GO:
```bash
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.rollout_policy \
  --checkpoint saltr/checkpoints/policy_reinit_v1/saltrd_policy_best.pt \
  --split hard --datasets uav123 visdrone_sot dtb70 \
  --output saltr/results/rollout_hard_policy_reinit_v1_perdataset.json
```

### Calibration: also per-dataset

Temperature scaling on val pooled may hide per-dataset ECE drift.
Required calibration check per dataset before any production rollout.

---

## Engineer Task List (summary of all open engineering tasks)

| # | Task | Priority | Files involved |
|---|---|---|---|
| 1 | Move `garbage/` files (see dead code list above) | High | 6 UNUSED files → `./garbage/` |
| 2 | Commit `train_policy.py` split fix (`"train"` → `"diagnostic"`) | High | `saltr/src/salt_r/train_policy.py` |
| 3 | Add `feature_schema` to `train.py` checkpoint save | Medium | `saltr/src/salt_r/train.py` |
| 4 | Pass `image_shape=frame.shape[:2]` to extractor in runner | Medium | `src/uav_tracker/salt_runner.py` |
| 5 | Implement `scripts/benchmark_full.py` (per-dataset, FPS, GFLOPs) | Medium | new script |
| 6 | Implement `scripts/visualize_sequence.py` (bbox overlay + action video) | Medium | new script |
| 7 | Add per-dataset reporting to `rollout_policy.py` | Medium | `saltr/src/salt_r/rollout_policy.py` |
| 8 | Run VisDrone-SOT + DTB70 benchmarks on policy v1 | Medium | after policy converges |
| 9 | Implement event-balanced oversampling in `OracleReinitDataset` | ~~Medium~~ **RESOLVED** — epoch 34 recall 0.73 | `saltr/src/salt_r/train_policy.py` |
| 10 | Regenerate oracle labels with score-map top-k candidates | High (for Run 2) | `saltr/src/salt_r/oracle_actions.py` |
| 11 | Wire `update_with_action()` search/template/recovery fully in sglatrack | Medium | `src/uav_tracker/trackers/sglatrack.py` |

---

## Review — 2026-05-22 Pass 8

### [OK] Commit 0633a9e — full TSA/state residue purge

Everything TSA-related removed from production files:
- `sglatrack.py`: `_STATE_COMPUTE_MAP`, `_STATE_SEARCH_MAP`, `update_with_state()` (~210 lines gone)
- `advisor.py`: `SALTRDState` enum, `get_state(tsa_state_int)`, `stage3_policy(tsa_state_int)` removed
- `salt_runner.py`, `shadow_mode.py`: OCCLUDED/LOST/CONFIRMED strings purged from comments/docstrings

Production grep for forbidden patterns: **0 hits** in `sglatrack.py`, **comment-only** in `salt_runner.py` and `advisor.py` (docstrings referencing old API in examples — no functional code).

25 no-TSA and action API tests pass.

### [OK] train_policy.py split fix committed

`split="diagnostic"` (the correct training pool) is now committed in the same 0633a9e commit.
`inverse-frequency class weights (REINIT×43, REJECT×0.023)` also committed.

### [OK] Policy training converged — epoch 34, reinit_recall = 0.73

| Epoch | reinit_recall | macro_F1 | val_loss | reject_prec |
|---:|---:|---:|---:|---:|
| 1 | 0.000 | 0.245 | 0.166 | 0.962 |
| 3 | 0.005 | 0.248 | 0.151 | 0.962 |
| **34** | **0.730** | **0.262** | **0.727** | **0.986** |

**Recall 0.73 >> SUPER_PLAN gate of 0.40.** Model fires reinit on 73% of oracle-positive
frames. Reject precision 0.986 — model is conservative: when it does NOT predict reinit,
it's almost always correct not to.

Gate status before rollout: **GO on recall. ROLLOUT NOW REQUIRED.**

### [BUG-COSMETIC] TSA words remain in advisor.py/salt_runner.py docstrings

`advisor.py` lines 20-21, 276, 464 contain `update_with_state()` in docstring examples and
`LOST` in a comment. `salt_runner.py` line 750 has `# Last known position before LOST state`
comment. These are documentation artifacts with zero functional impact. They do not affect
behavior. Not blocking.

### [NOTE] FROZEN.md conflict — SUPER_PLAN supersedes

`FROZEN.md` (2026-05-19) says `update_with_state()` must be preserved for CE ablation.
Commit 0633a9e removed it. The SUPER_PLAN (2026-05-22) explicitly requires this removal.
SUPER_PLAN is the newer architectural authority. FROZEN.md is stale and should itself be
archived. The CE ablation is still possible via config (`enable_ce: false`) — the method
was not the only ablation path.

### [ACTION REQUIRED] Rollout evaluation — gates 1, 2, 3 all unverified

Recall = 0.73 proves the model fires correctly. But the SUPER_PLAN gates require:
- Hard subset AUC delta >= +0.10 (or +0.08 interim)
- Full AUC delta >= +0.010
- Changed bbox frames > 0.5%

None of these are measured yet. `rollout_policy.py` exists and is ready. Run:

```bash
# Diagnostic split (contains hard seqs uav2/uav4/uav6/uav8 + many others)
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.rollout_policy \
  --oracle saltr/results/reinit_oracle_dataset.npz \
  --checkpoint saltr/checkpoints/policy_reinit_v1/saltrd_policy_best.pt \
  --split diagnostic \
  --output saltr/results/rollout_diagnostic_policy_reinit_v1.json

# Val split (held-out — checks for wrong reinit on non-hard sequences)
PYTHONPATH=src:saltr/src .venv/bin/python -m salt_r.rollout_policy \
  --oracle saltr/results/reinit_oracle_dataset.npz \
  --checkpoint saltr/checkpoints/policy_reinit_v1/saltrd_policy_best.pt \
  --split val \
  --output saltr/results/rollout_val_policy_reinit_v1.json
```

These two runs are the **single most important missing experiment**.

---

### Updated state summary

| Component | Status |
|---|---|
| TSA in production code | OK — zero functional references (comments only) |
| `update_with_state()` | OK — removed per SUPER_PLAN |
| `SALTRDState` / `stage3_policy()` | OK — removed |
| Policy training | OK — epoch 34, recall=0.73, READY FOR ROLLOUT |
| Rollout diagnostic split | **MISSING — highest priority** |
| Rollout val split | **MISSING — needed for wrong-reinit check** |
| Hard subset AUC delta | NOT MEASURED — rollout needed |
| Changed bbox frames | NOT MEASURED — rollout needed |
| Calibration | NOT RUN — after rollout GO only |
| FROZEN.md | STALE — superseded by SUPER_PLAN, archive it |
