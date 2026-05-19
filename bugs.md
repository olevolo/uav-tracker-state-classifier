# Bug & Dead Code Inventory

Findings from the 2026-05-19 CV/ML architecture review. Items are ranked by
impact on correctness or benchmark validity. Fixes noted where already applied.

## Status Summary

| Bug | Description (brief) | Status |
|-----|---------------------|--------|
| BUG-01 | generate_ml_labels zeros for flow_features/iou_trace | ⚠️ Partially Fixed |
| BUG-02 | Guard-3 EMA creates fresh CosineAppearanceMemory every 50 frames | ✅ Fixed |
| BUG-03 | motion_predictor.update() called on LOST/drifted frames | ✅ Fixed |
| BUG-04 | fast_bench.py re-instantiates SGLATracker per sequence | ✅ Fixed |
| BUG-05 | DYNAMIC state is unreachable — missing comment | ✅ Fixed |
| BUG-06 | _template_window / _template_window_size are unused dead fields | ✅ Fixed |
| BUG-07 | head_adaptor / _DeferredHeadAdaptor dead code in salt_runner.py | ✅ Fixed |
| BUG-08 | Multiple inline import logging inside function bodies | ✅ Fixed |
| BUG-09 | Stale _last_flow_feat comment in target_state_assessor.py | ✅ Fixed |
| BUG-10 | configs/experiments/v2_full_ml.yaml stale architecture config | ⏳ Open |
| BUG-11 | run_benchmark.py imports typer without it being in requirements | ✅ Not a bug |
| BUG-12 | CosineAppearanceMemory _PROJ_IN is 12288 instead of 3072 | ✅ Fixed |
| BUG-13 | salt.yaml tsa.name references non-existent TSA | ✅ Fixed |
| BUG-14 | _prev_tsa_state_int initialized to CONFIRMED regardless of frame-0 quality | ✅ Fixed |
| BUG-15 | update_online() no-op called every frame | ✅ Fixed |
| BUG-16 | _RECOVERY_WARMUP_FRAMES comment says "flow IoU unreliable" (stale) | ✅ Fixed |
| BUG-17 | Template EMA update causes regression (car7: 0.570→0.321) | ✅ Fixed |
| BUG-18 | Recovery cannot distinguish good re-init from bad at decision time | ✅ Fixed |

---

## CRITICAL — affects benchmark results or correctness

### BUG-01: `generate_ml_labels.py` writes all-zero `flow_features` and `iou_trace`
**File:** `scripts/generate_ml_labels.py` lines 285–292  
**Impact:** `uav123_labels.npz` has `flow_features/{seq}` = zeros for all 123
sequences. The script uses KCF which has no APCE/PSR/entropy signals. Any
training that reads these fields (TSA classifier, pretrain script) silently
falls back to proxy values instead of real tracker responses.  
`iou_trace/{seq}` is also all zeros — the per-frame IoU between tracker and GT
was never computed.  
**Fix:** Run `generate_ml_labels.py` with SGLATrack to populate real
APCE/PSR/entropy into `flow_features`, and compute real IoU into `iou_trace`.

**Status:** ⚠️ Partially Fixed (2026-05-19) — `--tracker sglatrack` mode added.
Running with `--tracker sglatrack` instantiates SGLATracker and populates
`flow_features[i,11]=APCE/256`, `[12]=PSR/3000`, `[13]=entropy/5` with real values,
and computes actual `iou_trace[i]=IoU(pred_bbox, gt[i])` per frame.
The existing NPZ (generated with KCF) still has all-zero fields — regenerate:
```bash
PYTHONPATH=src .venv/bin/python scripts/generate_ml_labels.py \
    --tracker sglatrack --output data/uav123_labels_sgla.npz
```

---

### BUG-02: Guard-3 EMA creates a fresh `CosineAppearanceMemory` every 50 frames
**File:** `src/uav_tracker/salt_runner.py` lines ~386–399  
**Impact:** Each call instantiates a new `CosineAppearanceMemory(max_templates=1)`,
which generates a fresh random projection matrix (`self._proj`). The EMA blends
embeddings from different random projections — the result is meaningless. The
0.80×old + 0.20×new calculation compares apples and oranges.  
**Fix:** Create a single module-level or class-level `_embed_helper` instance
and reuse it across all Guard-3 calls.

**Status:** ✅ Fixed in session 2026-05-19

---

### BUG-03: `motion_predictor.update()` is called on LOST/drifted frames
**File:** `src/uav_tracker/salt_runner.py` lines ~594–600  
**Impact:** The motion predictor (LSTM, currently disabled) receives
`track_state.bbox` unconditionally — including frames where the tracker is
LOST and the bbox is a frozen/drifted estimate. If the LSTM is ever re-enabled
it will be trained on bad positions during extended loss events.  
**Fix:** Gate the update: only call `motion_predictor.update()` when
`state_int in (CONFIRMED, DYNAMIC)`.

**Status:** ✅ Fixed in session 2026-05-19

---

### BUG-04: `fast_bench.py:run_sglatrack()` re-instantiates `SGLATracker` per sequence
**File:** `scripts/fast_bench.py` lines 175–183  
**Impact:** Creates `SGLATracker(device="auto")` fresh for every sequence,
triggering `_load()` (full model weight load) N times per benchmark run.
SALT correctly caches its runner via `_cache={}`. This makes SGLATrack's FPS
look worse than it is, biasing the comparison.  
**Fix:** Move SGLATracker instantiation outside the per-sequence loop, same
pattern as `run_salt()`.

**Status:** ✅ Fixed in session 2026-05-19

---

## HIGH — dead code that adds noise to benchmarks or confuses the codebase

### BUG-05: DYNAMIC state is unreachable in current configuration
**File:** `src/uav_tracker/ml/tsa/target_state_assessor.py` `_decide_state()` lines ~276–287  
**Impact:** The DYNAMIC branch fires only when `normalized_lstm_residual >
motion_threshold`. The LSTM is disabled (`motion_predictor: enabled: false` in
`salt.yaml`), so `lstm_pred_bbox` is always `None` → `lstm_residual = 0.0`
always → normalized residual = 0.0 → DYNAMIC never fires. The state is
effectively dead in the current pipeline but the code path is tested and
maintained as if it matters.  
**Fix:** Either keep as a documented future hook, or add an explicit comment
that DYNAMIC requires a functioning motion predictor to ever fire.

**Status:** ✅ Fixed in session 2026-05-19

---

### BUG-06: `_template_window` and `_template_window_size` are unused dead fields
**File:** `src/uav_tracker/trackers/sglatrack.py` lines 218–219, 538  
**Impact:** Both fields are initialized in `__init__` and cleared in `init()`
but never read. Template logic now uses `try_update_template()` with
`_template_update_count` and `_template_last_update`. The old window fields
are leftover from a previous design.  
**Fix:** Remove `_template_window` and `_template_window_size`.

**Status:** ✅ Fixed in session 2026-05-19

---

### BUG-07: `head_adaptor` / `_DeferredHeadAdaptor` is dead code
**File:** `src/uav_tracker/salt_runner.py` lines ~570–610, ~889–906  
**Impact:** TTT HeadAdaptor is `enabled: false` in salt.yaml (zero gradient
effect confirmed by ablation). `self.head_adaptor` is always `None` at runtime
but `_step()` has a full `if self.head_adaptor is not None` branch that
resolves a deferred pattern every frame. The `_DeferredHeadAdaptor` class at
the bottom of the file is never instantiated.  
**Fix:** Remove the entire `head_adaptor` field, the deferred resolver class,
and the `_step()` branch. Remove `head_adaptor` from the `SALTRunner` dataclass.

**Status:** ✅ Fixed in session 2026-05-19

---

### BUG-08: Multiple inline `import logging` inside function bodies in `salt_runner.py`
**File:** `src/uav_tracker/salt_runner.py` lines 225, 376, 387, 417, 443, 486, 510, 570, 722, 757  
**Impact:** `import logging as _logging` / `_tlog` / `_log2` / `_log3` etc.
are scattered inside method bodies. Python caches these after first import but
the repeated `import` statements are misleading — they look like new imports
but are just repeated lookups. Makes the code harder to read.  
**Fix:** Single `import logging` at module top; use `logger = logging.getLogger(__name__)`.

**Status:** ✅ Fixed in session 2026-05-19

---

### BUG-09: `_last_flow_feat` was a class-level attribute (latent bug, now fixed)
**File:** `src/uav_tracker/ml/tsa/target_state_assessor.py`  
**Impact:** Was shared between all instances of `TargetStateAssessor` (one
shared slot for all sequences). When the field was assigned in `update_online()`
it only shadowed the class attribute via instance assignment, so between
`assess()` and `update_online()` calls the value was correct, but two concurrent
instances would corrupt each other. Fixed in `__init__`, but the comment at the
old location should be removed.

**Status:** ✅ Fixed in session 2026-05-19

---

## MEDIUM — misleading configs or stale infrastructure

### BUG-10: `configs/experiments/v2_full_ml.yaml` is a stale architecture config
**File:** `configs/experiments/v2_full_ml.yaml`  
**Impact:** References the old V2 pipeline (KCF tier-0, OSTrack tier-2,
`ml_scene_scheduler`) that was superseded by the SALT/TSA architecture. The
`ml_scene_scheduler` routing it configures was replaced by TSA state-based
compute routing. This config cannot be run correctly with the current codebase
without the deleted `stark.py` tracker and removed entropy components.  
**Fix:** Move to `configs/archive/` or delete. Do not keep in the active
experiments directory.

**Status:** ⏳ Open — `configs/archive/v2_full_ml.yaml` was created but `configs/experiments/v2_full_ml.yaml` still exists and was not removed.

---

### BUG-11: `run_benchmark.py` imports `typer` without it being in requirements
**File:** `scripts/run_benchmark.py` line ~75 (`import typer`)  
**Impact:** `typer` is not in `requirements.txt`. The script will raise
`ModuleNotFoundError` on a clean install. The `fast_bench.py` correctly uses
only `argparse`.  
**Fix:** Either add `typer` to `requirements.txt` / `pyproject.toml`, or
migrate `run_benchmark.py` to use `argparse`.

**Status:** ✅ Not a bug — typer already in requirements.txt and pyproject.toml

---

### BUG-12: `CosineAppearanceMemory` is the actual SALT FPS bottleneck
**File:** `src/uav_tracker/ml/appearance_memory/cosine_memory.py` lines 207–208  
**Impact:** `flat @ self._proj` is a `(12288,) @ (12288×64)` matmul executed
every 10 frames when storing templates. On VisDrone-SOT easy sequences (97%
CONFIRMED, no recovery) this is the dominant overhead causing 41fps vs 74fps —
not Farneback, not the MLP, not the detector. The random projection matrix is
`(12288, 64)` = 3MB float32; the multiply processes ~786K FLOPs per call.  
**Fix:** Reduce the input image size before projection (currently uses
`frame.flatten()` on full-resolution crop), or switch to a smaller embedding
(e.g. 16×16 grayscale = 256d → much cheaper projection).

**Status:** ✅ Fixed in session 2026-05-19 — `_PROJ_IN` reduced to 3072 (32×32×3) and crop resized to 32×32 before projection.

---

### BUG-13: `configs/experiments/salt.yaml` `tsa.name: mobilenetv3_tiny` references a non-existent TSA
**File:** `configs/experiments/salt.yaml` line 18  
**Impact:** `tsa.name: mobilenetv3_tiny` is set but `SALTRunner.from_config()`
ignores it — it always constructs `TargetStateAssessor()` directly without
using the registry. The `name` field is a documentation lie that implies a
swappable TSA but no such registry exists.  
**Fix:** Remove `tsa.name` from the YAML, or implement `TSA_ASSESSORS.build()`
if swappability is genuinely needed.

**Status:** ✅ Fixed in session 2026-05-19 — `tsa:` block in salt.yaml has no `name:` field.

---

## LOW — minor / cosmetic

### BUG-14: `_prev_tsa_state_int` initialized to CONFIRMED regardless of frame-0 quality
**File:** `src/uav_tracker/salt_runner.py` line 91, `_reset()` line 789  
**Impact:** Frame 1 uses CONFIRMED compute budget (CE keep_ratio 0.5) even if
the initial GT bbox is degenerate. Minor — most sequences have valid frame-0 GT.

**Status:** ✅ Fixed (2026-05-19) — GT bbox validity check added in `run()`. If
`gt[0].w <= 0 or gt[0].h <= 0`, `_prev_tsa_state_int` is set to `OCCLUDED` so
frame 1 uses full compute (no CE pruning). Normal sequences unaffected.

---

### BUG-15: `update_online()` is now a no-op but callers still invoke it every frame
**File:** `src/uav_tracker/salt_runner.py` line ~598–610  
**Impact:** After MLP removal, `tsa.update_online(assessment_record)` is called
every frame but does nothing. The call itself is cheap but the
`TargetStateAssessment` record is constructed just to pass to it.  
**Fix:** Remove the `update_online()` call from `_step()` and the
`TargetStateAssessment` construction block that feeds it.

**Status:** ✅ Fixed in session 2026-05-19 — no `# ---- TSA online update ----` block and no `update_online()` call in `_step()`.

---

### BUG-17: Template EMA update causes regression even with strict guards (car7: 0.570→0.321)
**File:** `src/uav_tracker/salt_runner.py` (try_update_template block, now disabled)  
**Impact:** 5 template blends (90%+10%) at frames ~100/200/300/400/499 cause
AUC to drop from 0.570 to 0.321 on car7. Root cause: `_ref_embedding` used
for the cosine_sim guard is itself EMA-updated every 50 frames, so by frame 100
it has already drifted from frame-0. The cosine_sim ≈ 0.99 at these frames
means "similar to the drifted ref" not "similar to the original template" —
the guard is not detecting drift correctly.  
**Status:** Disabled again (same as original SALT). Re-enable only after:
(a) `_ref_embedding` EMA update is removed (frozen at frame-0), and
(b) blend fraction ≤5% with ≥200 frame gap.

**Status:** ✅ Fixed in session 2026-05-19 — `try_update_template` call is commented out/disabled in `_step()`.

---

### BUG-16: `_RECOVERY_WARMUP_FRAMES` comment says "flow IoU unreliable" (stale)
**File:** `src/uav_tracker/salt_runner.py` lines 102–104  
**Impact:** The stated reason is "prev_gray history is too short before frame 10",
but `_prev_gray` is set from frame 0 — after frame 1, one full frame of history
exists and Farneback is valid. The warmup may be overly conservative, delaying
recovery on short sequences (some VisDrone-SOT clips are only 90 frames).  
**Note:** The actual reason the warmup exists is the optical-flow IoU false-LOST
issue during startup (car13 showed 5 false-LOST at frames 1–5). With APCE as
primary signal (not flow-IoU), the 10-frame warmup may now be unnecessary.

**Status:** ✅ Fixed (2026-05-19) — comment updated to reflect car13 false-LOST history
and APCE-primary rationale. `_prev_gray` / Farneback timing is no longer cited.

---

### BUG-18: Recovery cannot distinguish good re-init from bad one at decision time
**Sequences:** bike2 (−0.023 vs SGLATrack), car7 (−0.025)
**Impact:** Both sequences trigger recovery with exactly 5 consecutive LOST frames.
car7 recovery at frame 356 gives IoU=0.951 (correct). bike2 recovery at frame 107
gives IoU=0.000 (wrong cyclist found by YOLO26m, cosine_sim=0.921 due to similar
bicycle appearance). Any threshold change breaks one to fix the other — they
produce identical LOST streak lengths in the recovery-enabled run.
**Root cause:** YOLO26m VisDrone is trained on vehicles, not cyclists. For
appearance-change sequences (bike2), the detector finds the right class, right
size, but a different individual. The cosine_sim threshold (0.25) is too loose
to reject same-class distractors.
**Fix path:** Post-recovery validation: after re-init, if the next frame's
tracker APCE drops back below the OCCLUDED threshold within 3 frames, undo the
re-init and set cooldown=50. This catches cases where the detector placed the
tracker on a wrong-but-similar object (AUC immediately returns to 0 → APCE
collapses), vs. a correct recovery (APCE stays high for 3+ frames).

**Status:** ✅ Fixed in session 2026-05-19 — Guard5 (`_last_recovery_sim`) and APCE trend gating (`_prev_escalated_apce`) implemented to distinguish good/bad recovery.
