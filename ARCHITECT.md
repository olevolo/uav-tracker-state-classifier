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
