# CSC-v4 BUILD CONTRACT (read before implementing any module)

You are one of ~10 agents building **CSC-v4** in parallel. V4 turns CSC from
"state-classifier + hand-policy" into **diagnosis + action-utility**:
normalized features → redesigned FC/LA labels with subtypes → prototype/distractor
memory + candidate verifier → hazard + per-action-gain heads (replacing forecast) →
SPRT sequential-evidence control → budgeted re-detector with an abort signal.

## GROUND RULES (all agents)
1. **Additive only.** Write NEW files under `csc_lib/csc/v4/` and `tools/v4_*`.
   Do NOT edit V3 files (`csc_lib/csc/{model,config,inference,features,calibration}.py`,
   `csc_lib/csc/labeling/*`, `tools/run_with_csc.py`, `tools/train_csc.py`) or anything in
   `outputs/csc_training/csc_prod`. V3 stays frozen. (Reading them to mirror style is encouraged.)
2. **Import shared types from `csc_lib.csc.v4.v4types`** (enums, Candidate, Prototype,
   V4Prediction, ActionDecision, Action/ACTION_NAMES, HEAD_DIMS). Never redefine them.
3. **sys.path note:** the live tracker is `src/uav_tracker/trackers/sglatrack.py` (it
   shadows salrtd/). Tools that load data/trackers must prepend, in order:
   `PROJECT_ROOT/salrtd/src`, `PROJECT_ROOT/src`, `PROJECT_ROOT` to sys.path (mirror the
   header of `tools/la_smoke.py`).
4. **Python 3.10+, typed signatures, dataclasses, no hardcoded abs paths** (use
   `Path(__file__).resolve().parents[...]`). Each module ends with a `if __name__ == "__main__":`
   smoke that constructs the class on synthetic/dummy inputs and prints shapes/asserts — so it
   runs standalone without datasets. Keep it CPU-only and fast.
5. **Stub cross-module deps against the interface** in this contract (don't block on another
   agent's file). Mark integration points with `# INTEGRATION:` comments.
6. Compile-clean: `python -m py_compile <yourfile>` must pass. Prefer numpy/torch already in the venv.
7. Telemetry field names are the V3 ones (see `outputs/.../telemetry/<seq>.jsonl`): confidence,
   apce, psr, response_entropy, sm_local_top2_ratio, sm_local_peak_margin, sm_peak_distance,
   sm_heatmap_mass_topk, sm_n_secondary, sm_peak_width, sm_top1, sm_top2, last_cosine_sim,
   initial_template_sim, appearance_drift, + bbox.

## SHARED TYPES (already written, in csc_lib/csc/v4/types.py)
DerivedStateV4(CC0/CU1/LA2/FC3), FCSubtype(NONE/DISTRACTOR/BACKGROUND),
LASubtype(NONE/FALSE/SMOOTH/ABRUPT/OCCLUDED/CANDIDATE), Action(HOLD/MOTION_BRIDGE/RELOCATE/
WIDEN/GLOBAL_SEARCH/TEMPLATE_UPDATE/FREEZE), ACTION_NAMES, N_ACTIONS, Candidate, Prototype,
V4Prediction, ActionDecision, HEAD_DIMS.

## MODULE ASSIGNMENTS (one per agent)

### A1 — `csc_lib/csc/v4/features_v4.py`  (normalized response-structure features)
Fix the V3 negative-transfer (raw response feats). Implement:
- `class V4FeatureCalibrator`: `.fit(values: np.ndarray)` storing robust-z (median/IQR) AND
  percentile (empirical CDF) params; `.transform(x: float)->float in [-clip,clip] or [0,1]`;
  `.save(path)`, `@classmethod load(path)` (JSON). One per feature.
- `fit_v4_calibrators(rows: list[dict]) -> dict[str, V4FeatureCalibrator]` for the response
  features (response_entropy, sm_local_top2_ratio, sm_local_peak_margin, sm_peak_distance,
  sm_heatmap_mass_topk, sm_n_secondary, sm_peak_width) — SEPARATE from APCE/PSR/confidence.
- `build_v4_features(row: dict, calibrators: dict, prev=None) -> np.ndarray` returning a
  fixed-length normalized vector (document slot order in a FEATURE_NAMES_V4 list). Robust to
  missing keys (None -> 0 after norm).
Smoke: fit on random rows, transform, assert finite + shape.

### A2 — `csc_lib/csc/v4/memory.py`  (prototype + distractor memory)
- `class PrototypeMemory(max_recent=5, max_distractor=8, ema=0.7)`:
  `update_anchor(emb)` (frame-0, set once), `update_recent(emb, frame_idx)` (on CC),
  `add_distractor(emb, frame_idx)` (on FC/LA secondary peaks), `sims(emb)->dict`
  {sim_to_init, sim_to_recent, sim_to_distractor(max)}. Cosine sim; handle empty stores (nan).
  Embeddings are 1-D np arrays (e.g. tracker `_last_search_peak_local` / `_last_template_embedding`).
Smoke: feed random embeddings, assert sims in [-1,1] / nan when empty.

### A3 — `csc_lib/csc/v4/verifier.py`  (candidate extraction + verification)
- `extract_candidates(score_map: np.ndarray, search_bbox, image_size, k=5, nms_radius=1,
  min_score_ratio=0.05) -> list[Candidate]` (mirror the logic of
  `src/uav_tracker/trackers/sglatrack.py::_select_candidate_peak_indices` / `_extract_candidate_diagnostics`;
  read them). Map grid peak -> pixel cx,cy,w,h.
- `class CandidateVerifier(memory: PrototypeMemory)`: `.score(c: Candidate, motion_prior=None)->float`
  and `.verify(c, margin=...)->bool` using sim_to_init/recent (high), sim_to_distractor (low),
  score_rank, peak_margin, motion/scale plausibility. This is the guard that prevents
  catastrophic relocate jumps (person9/car6_2 -0.5). Return calibrated [0,1].
Smoke: synthetic 16x16 score_map -> candidates; verify against a dummy memory.

### A4 — `csc_lib/csc/v4/labeling_v4.py`  (redesigned FC/LA labels + subtypes)
Replace "FC = LOST && high confidence" (confidence is degenerate). Implement:
- `label_frame_v4(iou, occ, oov, tel: dict, cand_sims: dict|None) -> dict` returning
  `{derived (DerivedStateV4), fc_subtype (FCSubtype), la_subtype (LASubtype)}`. Rules:
  FC = iou<tau_fail AND targetness/peaky-map high (low entropy, high top2/peak_margin) AND
  (identity_to_target low OR identity_to_distractor high) AND not pure occlusion;
  FC_D vs FC_B by sim_to_distractor; LA subtypes by motion smoothness (needs prev/next center —
  pass a small window), occlusion flags, candidate availability; LA_FALSE when iou actually ok.
- `build_v4_labels(telemetry_rows, gt_bboxes, occ, oov) -> list[dict]` (sequence-level; compute
  IoU, motion features, call label_frame_v4). Document thresholds as args with defaults.
Smoke: synthetic sequence -> labels; assert every label has the 3 keys, FC over LA priority holds.

### A5 — `tools/v4_action_sweep.py`  (offline counterfactual action-gain labels)
The most important new supervision. For a sequence's passive run, for each frame in a loss/risk
window, simulate each Action and record ΔIoU vs passive (use GT). Output per-frame
`action_gain[action_name] = ΔIoU` + `best_action`, `recoverable_by_bridge/relocate`,
`needs_global_search`, `do_not_act` (all actions ≤0), `template_update_safe`.
- You may APPROXIMATE the sweep cheaply offline WITHOUT re-running the tracker: model each action's
  effect on the search center / next IoU using the stored predictions + GT + telemetry (document
  the approximation clearly; mark it `# APPROX`). A full tracker-in-the-loop sweep is future work.
- CLI: `--passive_dir <run> --dataset uav123 --out <parquet/jsonl>`; reuse la_smoke loaders
  (build_index, seq_iou, gt_array via `import la_smoke`).
Smoke: run on a tiny synthetic example (no dataset needed) via a `--selftest` flag.

### A6 — `csc_lib/csc/v4/model_v4.py`  (V4 multi-head model)
- `class CSCv4(nn.Module)`: a causal TCN encoder (mirror `csc_lib/csc/model.py` CSCTCN: kernel 3,
  dilations [1,2,4,8], hidden 64, not bidirectional) + heads sized by `HEAD_DIMS`
  (derived 4 / fc_subtype 3 / la_subtype 6 / hazard 3 sigmoid / action_utility 7 regression /
  do_not_act 1 / template_update_safe 1). `forward(x:(B,T,F))` returns a dict of logits;
  `predict(x, last_step_only=True) -> V4Prediction`. feature_dim configurable (from features_v4).
Smoke: build with feature_dim=20, run (2,32,20) tensor, assert all head shapes + a V4Prediction.

### A7 — `csc_lib/csc/v4/sprt_gate.py`  (sequential-evidence control gate)
Replace one-frame `top2ratio≥τ AND entropy≥τ`. Implement:
- `class SPRTGate(alpha=0.05, beta=0.1, max_evidence=...)`: `.update(llr: float)->str` in
  {'accumulate','fire','clear'} via Wald SPRT thresholds; `.reset()`. `false_alert_budget`.
- `expected_gain_gate(action_utility: dict, costs: dict, min_gain=0.0) -> (action, gain)` choosing
  the action with max (predicted ΔIoU − cost), or HOLD if none clears min_gain.
- `llr_from_evidence(per_frame_features: dict) -> float` (a simple calibrated log-likelihood-ratio
  proxy from gate features; document).
Smoke: feed a stream of high/low evidence, assert it fires only after sustained evidence.

### A8 — `csc_lib/csc/v4/redetect.py`  (budgeted multi-crop SGLATrack re-detector)
Class-agnostic emergency re-detect REUSING the SGLATrack backbone (no new model). Implement:
- `make_crop_pyramid(last_good_center, velocity_prior, image_size, frame) -> list[crop_spec]`
  (local expanded -> 2x -> 3x -> sparse 3x3 full-frame grid).
- `class MultiCropRedetector(tracker, budget=RedetectBudget(max_fps=2,min_interval=5,max_attempts=6))`:
  `.maybe_redetect(frame, last_good, velocity, frame_idx) -> list[Candidate] | None` — runs the
  SAME template-search forward on the batched crops (use tracker._model / tracker._z_tensor READ-ONLY;
  do NOT mutate tracker state), decodes top-k peaks via verifier.extract_candidates. Enforce the
  budget (sparse trigger). Return candidates for the verifier to judge (NEVER auto-jump).
- `@dataclass RedetectBudget`.
Smoke: with a dummy tracker stub exposing _model/_z_tensor, assert budget gating (no fire before
min_interval) + returns list. Mark tracker-coupled parts `# INTEGRATION:`.

### A9 — `csc_lib/csc/v4/avtrack_sidecar.py`  (AVTrack sidecar re-detector)
Independent-failure-profile re-detector. AVTrack adapter = `src/uav_tracker/trackers/avtrack.py`,
weights `~/uav-tracker-weights/avtrack/AVTrack-DeiT.pth.tar` (~0.7 GFLOPs, class-agnostic).
- `class AVTrackSidecar(device='cpu')`: lazy-load AVTrack once; `.propose(frame, crops, template_hint)
  -> list[Candidate]` running AVTrack's template-query on the crops + extracting score-map peaks
  (AVTrack has score/size/offset maps). It only PROPOSES a center; SGLATrack stays primary; the
  CandidateVerifier (A3) accepts/rejects. Keep load lazy + guarded (weights may be absent).
Smoke: construct without loading (lazy); assert API + a dummy propose path. Mark heavy parts `# INTEGRATION:`.

### A10 — `csc_lib/csc/v4/control_v4.py`  (LA-triage + action selection + abort + Meta-Updater)
The orchestrator (writes against A2/A3/A6/A7/A8/A9 interfaces; stub where needed).
- `class V4Controller(model, memory, verifier, sprt, redetector=None, sidecar=None, cfg=V4ControlConfig())`:
  `.decide(pred: V4Prediction, tel: dict, candidates: list[Candidate], frame_ctx) -> ActionDecision`.
  Policy: false-LA -> HOLD (no freeze, no override); smooth-LA -> MOTION_BRIDGE (+displacement cap +
  ABORT if telemetry doesn't improve in 2-3 frames); candidate-LA -> verify top-k, RELOCATE only if a
  verified candidate beats last-good by margin; occlusion-LA -> HOLD+FREEZE; persistent unknown LA ->
  GLOBAL_SEARCH (budgeted redetect+verify, 2-frame vote). FC suspected-not-verified -> FREEZE only;
  FC verified -> FREEZE + reject bbox + search last-good/global. Gate every action through the SPRT +
  expected_gain (do_not_act). 
- `class MetaUpdater`: `.template_update_safe(pred, tel) -> bool` (state-gated: target_present high,
  uncertainty low, identity_match high, distractor_conflict low; allow a short post-recovery window).
- `@dataclass V4ControlConfig` (all thresholds, defaults sane).
- `abort_check(before_tel, after_tel) -> bool` (targetness up / entropy down / top2_ratio down /
  sim up / p_LA down; else rollback).
Smoke: drive the controller with synthetic V4Predictions across a fake LA episode; assert it HOLDs on
false-LA, BRIDGEs on smooth-LA, and ABORTs when telemetry doesn't improve.

### A11 — `tools/v4_fast_eval.py` + `tools/v4_diagnose.py`  (fast-track tools)
- `tools/v4_diagnose.py`: quick offline separability/calibration diagnostics for the new
  labels/features (per-feature AUROC for FC-subtype and LA-subtype separation; calibrator coverage).
  Reuse la_smoke loaders + agg_full style. CLI `--passive_dir --dataset`.
- `tools/v4_fast_eval.py`: a FAST ΔAUC/FCR/FCD harness specialized for v4 runs (per-seq + tercile +
  guards), mirroring `tools/agg_full.py` + `la_smoke.fc_stats`, but reading v4 run dirs. CLI
  `--run_dir --baseline --dataset`.
Smoke: `--selftest` path that runs on synthetic arrays (no dataset).

## DELIVERABLE per agent
Your file(s), compile-clean, with a working `__main__` smoke, typed, importing v4.types. Return a
short report: what you built, the public API, integration points (`# INTEGRATION:`), and anything you
stubbed/approximated. Do NOT run training or tracker eval — implementation + smoke only.
