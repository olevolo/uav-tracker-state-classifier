# UAV Entropy-Guided Tracker — Implementation Plan (v2, Modular + Iterative)

**Reference:** Oleksiuk, V., Velhosh, S. (2026). *Entropy-Guided Tracker Switching Method for Unmanned Aerial Vehicle Real-Time Tracking.* Electronics and Information Technologies, 33.

This v2 plan supersedes v1. It (a) makes modularity a first-class design constraint — adding a new tracker, detector, or switching signal is a **config change, not a code change**; (b) treats the paper as a **starting reference, not a contract** — where the paper is weak or sub-optimal, we propose and empirically validate improvements; (c) is **strictly iterative** — every phase produces a working, demo-able end-to-end system.

Work is executed by three cooperating Claude agents — Architect/Coordinator, Engineer, and DevOps. See §5 and the `agents/*.md` files.

---

## 1. Design Principles (read these first)

These principles override any tactical decision below when in conflict.

### 1.1 Modularity is non-negotiable
Every swappable component is behind a `Protocol` + a registry + a config. Four core extension points:

| Extension point | What you plug in | Examples we'll ship |
|---|---|---|
| `Tracker` | a SOT backbone | KCF, SiamFC, MobileTrack, MixFormer, OSTrack |
| `Detector` | a full-frame detector / re-detector | YOLOv8-n, YOLOv10-n, DFINE, Grounding DINO (zero-shot) |
| `SwitchSignal` | a per-frame scalar the scheduler reads | MotionEntropy, CircularResultant, APCE, TrackerConfidence, FlowDivergence, LearnedGate |
| `Scheduler` | a policy mapping signals → tracker tier | HysteresisBinary, CUSUM, MultiTierThreshold, LearnedPolicy |

**Acceptance test for modularity:** adding a new tracker should require (1) one new file under `src/uav_tracker/trackers/`, (2) registering it with `@TRACKERS.register("name")`, (3) a YAML snippet under `configs/trackers/`. No edits to the hybrid runner, the scheduler, or the evaluator. Enforced by `tests/contract/test_plugin_contract.py`.

### 1.2 Paper fidelity ≠ number matching
We implement the paper's **logic** faithfully (entropy calculation, hysteresis state machine, global-motion compensation). We do **not** contract to reproduce its exact numbers, because:
- The paper runs on a Colab ephemeral VM with variable hardware.
- OpenCV's KCF differs from Henriques 2015 reference implementation.
- "Variant of MobileTrack" is under-specified.
- 123 UAV123 sequences with OPE-only eval is noisy (±0.02 AUC is the protocol's inherent variance).

We'll treat Table 2 numbers as **reference points** (see §10 for the fidelity contract). If our implementation systematically diverges, we investigate. If it beats the paper, we document it.

### 1.3 Deviate deliberately where the paper is weak
See §2 for a critical assessment. Where we deviate, we (a) keep the paper's original as one registered plugin, (b) add our improved version as another, (c) compare empirically in ablations. Nothing is removed; everything is additive.

### 1.4 Iterate end-to-end
Every phase from Phase 1 onward produces a CLI command that takes a sequence and emits metrics. No phase leaves the repo in a "half-wired" state. If a phase doesn't end with a green demo, it isn't done.

### 1.5 Keep research hooks open
- Logs are structured (JSONL telemetry per frame) so any analysis can be bolted on post-hoc.
- All components emit signals, not just scalars — 2D vectors, histograms, confidence distributions — so a future learned scheduler can consume rich features.
- Seeds are controllable per-subcomponent for repeatable ablations.

---

## 2. Critical Assessment of the Paper & Our Deviations

This section is the Architect's responsibility to keep current. Each item below is tagged with a proposed remedy and whether it lands as a default or an optional plugin.

### 2.1 Orientation-only entropy loses magnitude information
**Issue.** Paper computes Shannon entropy over orientation bins weighted by magnitude. But identical orientation distributions with wildly different magnitudes yield the same `H̃`. A target jittering in all directions at 0.5 px is not the same as one darting in all directions at 20 px.
**Remedy.** Ship two signals: `MotionEntropy` (paper's method) and `JointMotionEntropy` (2D histogram over `(orientation, log(magnitude))`). Compare in §13 ablation.
**Default:** `MotionEntropy` (paper) for fidelity; document the limitation.

### 2.2 Shannon entropy on small samples is high-variance
**Issue.** With 16 bins and ~20 vectors (typical UAV ROI), `H̃` fluctuates frame-to-frame even under coherent motion. Paper smooths with EMA, but the bias remains.
**Remedy.** Also ship `CircularResultant` signal: `R = |(1/N) Σ e^{iθ_i}|`, use `1 − R` as a disorder metric. `R` is the standard directional-statistics tool (Mardia & Jupp) and is **statistically stable** at small N.
**Default:** offer both; let the scheduler config choose. Our prior: `1 − R` will be more robust.

### 2.3 Cyclic boundary artifact in the histogram
**Issue.** Two vectors at 359° and 1° fall in different bins but are essentially the same direction. With 22.5° bins this is rare but real.
**Remedy.** Either smooth the histogram with a wrapped Gaussian kernel, or use the circular statistics above which have no cyclic artifact.
**Default:** add a `smooth_sigma` knob (0 = paper default, ~5° = wrapped smoothing).

### 2.4 Hard thresholds don't generalize
**Issue.** `E_hi=0.65`, `E_lo=0.50` are empirical for UAV123/OTB100. On LaSOT, GOT-10k, or a different UAV dataset, recalibration is required.
**Remedy.** Add `AdaptiveThresholdScheduler` that learns thresholds from per-sequence percentiles (e.g., trigger deep when `H̄_t` exceeds its own rolling 90th percentile).
**Default:** paper's fixed thresholds; adaptive as a plugin.

### 2.5 5-frame confirmation adds latency when speed matters most
**Issue.** The confirmation window is designed to avoid flapping, but during fast maneuvers — exactly when you need the deep tracker — it delays the switch by ~50 ms at 100 FPS.
**Remedy.** Ship a `TrajectoryAwareScheduler` that shortens confirmation when `dH̄/dt` is large (entropy rising fast). Keeps the flap-guard property via a soft cost term instead of a hard count.
**Default:** paper's binary hysteresis; trajectory-aware as a plugin for comparison.

### 2.6 Warm-standby wastes compute
**Issue.** Paper runs deep tracker every 10 frames during LIGHT mode "as a safety net." At T4 throughput that's significant GPU draw. On battery-constrained UAV hardware this matters.
**Remedy.** Off by default; only activate during `d H̄/dt > 0` (entropy climbing). Gate via a separate `WarmStandbyPolicy` plugin.
**Default:** off. Can be flipped per experiment.

### 2.7 No recovery from complete target loss
**Issue.** Paper explicitly flags this in §Discussion as future work: "If the target is not found by either tracker for a certain period, trigger a full-frame detection."
**Remedy.** First-class `Detector` plugin point. Scheduler can escalate to DETECT tier when trackers report low confidence for N frames.
**Default:** no detector in the paper-fidelity config; ship YOLOv8-n detector as an opt-in third tier from Phase 6 onward.

### 2.8 No confidence output
**Issue.** Hybrid returns a bbox; downstream auto-guidance needs calibrated uncertainty.
**Remedy.** Augment output schema with `bbox_confidence ∈ [0,1]` and `track_state ∈ {LOCKED, UNCERTAIN, LOST}`. Back-fill from each tracker's native score.
**Default:** always emitted; consumers can ignore.

### 2.9 OPE is too generous
**Issue.** One-Pass Evaluation doesn't re-init on failure, so early loss doesn't propagate — giving low AUC but hiding behavior. Modern practice adds restart-based eval and long-term metrics.
**Remedy.** Report OPE (for paper comparability) **plus** restart-based eval on a subset, plus long-term metrics (precision, recall over track length) for the detection-equipped config.
**Default:** OPE is primary; extended metrics computed on top.

### 2.10 Global-motion estimation brittle on low-texture frames
**Issue.** RANSAC homography fails on sky/water frames with few background keypoints.
**Remedy.** Three-level fallback: (1) RANSAC homography, (2) LMedS affine, (3) frame-reliability flag → reuse previous global estimate + mark frame "entropy-unreliable" so scheduler holds its state.
**Default:** full three-level fallback from day one (strict improvement, no downside).

### 2.11 MobileTrack is the paper's deep tier — reference unresolved
**Issue.** The paper cites "a variant of MobileTrack" (Xue 2022 per earlier assumption; the precise reference is **not currently resolvable** from web research alone — see `scripts/manifests/weights.sha256` and `src/uav_tracker/trackers/siamese/mobiletrack.py`). SiamFC is a widely-available comparison baseline (Table 2 row), not the paper's method.
**Remedy.** `mobiletrack` is registered in TRACKERS as the paper's primary deep tier; its architecture currently inherits SiamFC's AlexNet-style backbone as a placeholder. Swap the backbone once the paper's MobileTrack reference is pinned (user action: source Oleksiuk & Velhosh 2026 PDF, cite the actual MobileTrack reference). `SiamFCTracker` stays registered as `siamfc` for Table 2 baseline comparison.
**Default:** MobileTrack (scaffold) for paper-fidelity experiments; SiamFC for baseline comparison.

### 2.12 Single-object only
**Out of v1 scope.** But the `Tracker` protocol is defined to allow future multi-object extension (returns a list). No code paths assume cardinality 1.

---

## 3. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          HybridRunner                                    │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │                    Scheduler (plugin)                            │   │
│   │   reads { SwitchSignal }*  →  decides tier ∈ {T0, T1, T2, ...}   │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                 │                  │                  │                  │
│                 ▼                  ▼                  ▼                  │
│      ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐    │
│      │ T0: Fast        │  │ T1: Heavy        │  │ T2: Detector      │    │
│      │ (KCF+Kalman,    │  │ (SiamFC,         │  │ (YOLO,            │    │
│      │  STARK-lite,    │  │  MobileTrack,    │  │  DFINE,           │    │
│      │  ByteTrack-CF)  │  │  MixFormer, …)   │  │  Grounding DINO)  │    │
│      └─────────────────┘  └──────────────────┘  └───────────────────┘    │
│                 ▲                  ▲                  ▲                  │
│                 └──────────────────┴──────────────────┘                  │
│                                    │                                     │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │                SignalBus (plugin registry)                       │   │
│   │   MotionEntropy · CircularResultant · APCE · TrackerConfidence   │   │
│   │   · FlowDivergence · LearnedGate · {your_signal}                 │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                    ▲                                     │
│                                    │                                     │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │                    FrameContext                                  │   │
│   │   raw frame · prev frame · current bbox · optical flow cache     │   │
│   │   · global motion estimate · tracker response maps · telemetry   │   │
│   └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘

Input:  sequence of frames + initial bbox
Output: per-frame (bbox, confidence, tier, signals, timings) → JSONL + CSV
```

**Key invariants:**
- `Tracker`, `Detector`, `SwitchSignal`, `Scheduler` are all `Protocol`s with zero coupling to each other's internals.
- `FrameContext` is a passive dataclass; components read what they need, write to their own namespace.
- The `HybridRunner` is the only component that knows about tiers — all plugins see only their local contract.
- N-tier is first-class: binary LIGHT/DEEP is a special case of `MultiTierScheduler` with 2 tiers.

---

## 4. Plugin Architecture Specification

### 4.1 Registry pattern

```python
# src/uav_tracker/registry.py
from typing import Callable, TypeVar, Generic

T = TypeVar("T")

class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def deco(cls: Callable[..., T]) -> Callable[..., T]:
            if name in self._items:
                raise ValueError(f"{self._kind}:{name} already registered")
            self._items[name] = cls
            return cls
        return deco

    def build(self, name: str, **kwargs) -> T:
        if name not in self._items:
            raise KeyError(f"unknown {self._kind}: {name}. Known: {list(self._items)}")
        return self._items[name](**kwargs)

    def names(self) -> list[str]: return list(self._items)

TRACKERS: Registry["Tracker"] = Registry("tracker")
DETECTORS: Registry["Detector"] = Registry("detector")
SIGNALS: Registry["SwitchSignal"] = Registry("signal")
SCHEDULERS: Registry["Scheduler"] = Registry("scheduler")
```

### 4.2 Core Protocols (owned by Architect, in `*/base.py`)

```python
# src/uav_tracker/trackers/base.py
@dataclass(frozen=True)
class BBox:
    x: float; y: float; w: float; h: float   # left, top, width, height

@dataclass
class TrackState:
    bbox: BBox
    confidence: float                          # [0, 1], calibrated best-effort
    status: Literal["locked", "uncertain", "lost"]
    aux: dict[str, Any] = field(default_factory=dict)  # response map, etc.

class Tracker(Protocol):
    name: str
    tier_hint: int                             # 0=lightest, higher=heavier; advisory
    def init(self, frame: np.ndarray, bbox: BBox) -> None: ...
    def update(self, frame: np.ndarray) -> TrackState: ...
    def flops_per_update(self) -> float: ...   # static estimate
```

```python
# src/uav_tracker/detectors/base.py
@dataclass
class Detection:
    bbox: BBox
    score: float
    class_id: int | None = None

class Detector(Protocol):
    name: str
    def detect(self, frame: np.ndarray, hint: BBox | None = None) -> list[Detection]: ...
    def flops_per_call(self) -> float: ...
```

```python
# src/uav_tracker/signals/base.py
@dataclass
class SignalReport:
    value: float                                # scalar summary
    vector: np.ndarray | None = None            # optional richer payload
    reliable: bool = True                       # False = scheduler should ignore
    aux: dict[str, Any] = field(default_factory=dict)

class SwitchSignal(Protocol):
    name: str
    range: tuple[float, float]                  # expected value range, for normalization
    def step(self, ctx: FrameContext, state: TrackState) -> SignalReport: ...
    def reset(self) -> None: ...
```

```python
# src/uav_tracker/schedulers/base.py
@dataclass
class SchedulerDecision:
    tier: int                                   # which tier index to use
    reason: str                                 # human-readable
    switched: bool                              # did tier change this frame

class Scheduler(Protocol):
    name: str
    tiers: int                                  # number of tiers it manages
    def decide(
        self,
        signals: dict[str, SignalReport],
        current_tier: int,
        frame_idx: int,
    ) -> SchedulerDecision: ...
    def reset(self) -> None: ...
```

### 4.3 Config schema (Hydra)

Declarative composition — *all* experiment variation lives here:

```yaml
# configs/experiments/entropy_hybrid.yaml
runner:
  _target_: uav_tracker.runner.HybridRunner
  trackers:
    - {tier: 0, name: kcf_kalman, args: {sigma: 0.125}}
    - {tier: 1, name: siamfc, args: {device: cuda, dtype: float16}}
  signals:
    - {name: motion_entropy, args: {n_bins: 16, alpha: 0.8, magnitude_threshold: 1.0}}
    - {name: tracker_confidence}             # always-on, cheap
  scheduler:
    name: hysteresis_binary
    args:
      signal: motion_entropy
      thresh_up: 0.65
      thresh_down: 0.50
      confirm_frames: 5
      cooldown_frames: 5
      warm_standby_cadence: 0                # paper uses 10; we default off (see §2.6)
  detector: null                             # opt-in in later phases
```

Swapping the signal is a one-line change (`motion_entropy` → `circular_resultant`). Swapping the scheduler is a one-line change. Plugging a new tracker is a registry + YAML addition.

### 4.4 Contract tests
`tests/contract/test_plugin_contract.py` enumerates every registered plugin and asserts:
- Protocol conformance (mypy + runtime `isinstance(..., Protocol)` via `runtime_checkable`).
- `reset()` leaves state equivalent to construction.
- Repeated `step`/`update` on identical inputs give identical outputs (determinism).
- No plugin imports another plugin directly — only `base.py` and `types.py`.

---

## 5. Agent Team (unchanged from v1 — see `agents/*.md`)

Three agents: Architect/Coordinator, Engineer, DevOps. Governance, handoff rules, and acceptance gates unchanged. Phase assignments updated below.

---

## 6. Local Environment Setup

Unchanged from v1. See `agents/devops.md` §5 for full macOS/Linux/Colab setup. One-command bootstrap: `make setup`.

---

## 7. DevOps & Infrastructure

Unchanged from v1. See `agents/devops.md` §6. Key CI workflows: `ci.yml` (PR), `smoke-eval.yml` (PR), `nightly-eval.yml` (T4), `docker-images.yml`, `release.yml`.

**New addition in v2:** `plugin-registry-health.yml` runs once per week, enumerates every registered plugin, and asserts the plugin-contract suite still passes.

---

## 7a. Git Workflow (cross-agent)

> **Activation:** this section applies once the repo has a remote + enforced branch protection. While running locally without a remote, see §11 preamble for the simpler local-master workflow (direct commits to `master` with conventional commits, close-out commit per phase). The rules below are the target state, not the current state.

All three agents collaborate through **trunk-based development with short-lived feature branches merged to `master` via pull requests**. No direct pushes to `master`. CI must be green. Every PR requires approval from at least one **cross-agent reviewer**.

### 7a.1 Branch naming convention
- Architect: `architect/<scope>-<slug>` — e.g., `architect/adr-0004-registry`, `architect/plan-v3`.
- Engineer: `engineer/<scope>-<slug>` — e.g., `engineer/eng-0003-kcf`, `engineer/phase-4-motion-entropy`.
- DevOps: `devops/<scope>-<slug>` — e.g., `devops/infra-0001-pyproject`, `devops/ci-smoke-eval`.

Scope prefixes: `adr`, `plan`, `doc`, `phase`, `eng`, `infra`, `ci`, `release`, `hotfix`, `research`.

### 7a.2 Commit discipline
- **Conventional Commits:** `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `ci:`, `perf:`, `revert:`.
- Atomic commits — one logical change per commit.
- Subject ≤ 72 chars; body wraps at 72.
- Signed commits (GPG or DCO sign-off).
- No `--no-verify`, no `--no-gpg-sign`, no bypass of pre-commit hooks.

### 7a.3 Pull-request lifecycle
1. Sync master: `git checkout master && git pull --ff-only`.
2. Create branch per §7a.1.
3. Commit iteratively. Rebase-interactive to tidy history before opening PR.
4. Push: `git push -u origin <branch>`.
5. Open PR using `.github/PULL_REQUEST_TEMPLATE.md`. Fill `owner:` + `reviewer:` front-matter.
6. CI runs automatically: `ci`, `smoke-eval`, `plugin-contract`, `security-review` (DevOps paths only).
7. Request review from a **cross-agent reviewer** per §7a.5.
8. Address feedback via additional commits. **Don't force-push a PR branch once it has review comments** (reviewers lose their place).
9. **Squash-merge** into `master` — keeps master history linear and readable.
10. Branch auto-deleted post-merge.

### 7a.4 Branch protection rules (enforced on `master`)
- ≥1 approving review from a user other than the PR author.
- All required status checks green: `ci`, `smoke-eval`, `plugin-contract`, `security-review` (DevOps paths).
- PR must be up-to-date with `master` (merge queue enabled when available).
- No direct pushes (admins included).
- Signed commits required.
- Force-pushes to `master` disabled.
- Admin-bypass disabled.

### 7a.5 Cross-agent reviewer matrix
| PR author | Required reviewer (primary) | Optional reviewer |
|---|---|---|
| Engineer | Architect | DevOps (if touching env/infra surface) |
| DevOps | Architect | Engineer (if affecting DX or runtime shape) |
| Architect (docs-only: PLAN, ADRs, docs/**) | Engineer or DevOps | — |
| Architect (touching `*/base.py` or Protocols) | Engineer | Architect confirms + tags DevOps if registry changes |

**An agent never merges their own PR**, regardless of content. Docs-only Architect PRs still go through a reviewer.

### 7a.6 Conflict resolution
- Own branch conflicts: `git fetch origin && git rebase origin/master`; resolve; `git push --force-with-lease`.
- `--force` and `--force-with-lease` are **forbidden on `master`**.
- If a conflict spans another agent's owned paths, tag that agent — they resolve in their namespace.
- Unresolved cross-agent conflicts → Architect decides within 24 h.

### 7a.7 Release engineering
- DevOps tags `v<semver>` on `master` after a clean nightly-eval.
- `release.yml` auto-publishes: PyPI wheel, GHCR images (`cpu`, `gpu`, `jetson`), GitHub Release from `CHANGELOG.md`.
- Pre-1.0: `v0.X.Y` during Phases 0–8. `v1.0.0` when §15 Definition of Done is met.
- Phase close-outs (the `chore(phase-N): close-out` commits defined in §11 preamble) are **not** release tags — they are CHANGELOG-only checkpoints. Release tags cut at natural `v0.X` points, typically after Phase 2, 4, 7, and 8.

### 7a.8 Hotfixes
- Branch: `devops/hotfix-<slug>` from the latest release tag.
- PR to `master` labeled `hotfix`. Architect may approve with single-reviewer (no waiting for second).
- Required: post-mortem in `docs/incidents/YYYY-MM-DD-<slug>.md` within 48 h.
- Cherry-pick to active release branch if LTS variants are maintained.

### 7a.9 Keeping `master` green
- Red master = stop-the-line. The merging agent reverts within 30 minutes.
- Revert PR: `git revert <sha>`, commit prefixed `revert:`, single-reviewer fast-track allowed.
- Recurrent-flake CI steps: DevOps owns root-cause fix, not retry loops.

### 7a.10 Agents never do
- Push directly to `master` (or to any protected release branch).
- Force-push `master`.
- Merge own PRs.
- Bypass CI, hooks, or required reviews.
- Disable branch protection to expedite merges.

---

## 8. Technology Stack

Unchanged from v1. Python 3.10, PyTorch 2.1.0, OpenCV-contrib 4.9, Hydra, Typer, pytest, ruff, mypy, uv.

**New v2 deps for detection tier and richer signals:**
- `ultralytics` (YOLOv8/v10) — detection.
- `torchmetrics` — calibrated confidence metrics.
- `ruptures` — change-point detection for CUSUM scheduler.
- `pycircstat` — circular statistics (optional; we can hand-roll if dep is flaky).

---

## 9. Repository Layout (v2)

```
uav-entropy-tracker/
├── pyproject.toml / uv.lock / requirements*.txt / Makefile / etc.
├── PLAN.md / AGENTS.md / agents/*.md / docs/…
├── configs/
│   ├── trackers/{kcf_kalman,siamfc,mobiletrack,transt,mixformer}.yaml
│   ├── detectors/{yolov8n,yolov10n,dfine,grounding_dino}.yaml
│   ├── signals/{motion_entropy,circular_resultant,apce,tracker_confidence,flow_divergence}.yaml
│   ├── schedulers/{hysteresis_binary,cusum,multi_tier,adaptive_threshold,trajectory_aware}.yaml
│   ├── datasets/{uav123,otb100,lasot_airplane}.yaml
│   └── experiments/
│       ├── paper_kcf.yaml
│       ├── paper_mobiletrack.yaml
│       ├── paper_entropy_hybrid.yaml        # paper fidelity
│       ├── improved_hybrid.yaml             # our best config
│       ├── ablation_signals.yaml
│       ├── ablation_schedulers.yaml
│       ├── ablation_detection_tier.yaml
│       └── edge_jetson.yaml
├── src/uav_tracker/
│   ├── __init__.py
│   ├── types.py                  # BBox, TrackState, FrameContext, Detection, SignalReport
│   ├── registry.py               # Registry pattern
│   ├── runner.py                 # HybridRunner (the orchestrator)
│   ├── trackers/
│   │   ├── base.py / kcf_kalman.py / siamese/{base,siamfc,mobiletrack}.py / transt.py
│   ├── detectors/
│   │   ├── base.py / yolo.py / dfine.py / null_detector.py
│   ├── signals/
│   │   ├── base.py / motion_entropy.py / circular_resultant.py
│   │   ├── apce.py / tracker_confidence.py / flow_divergence.py
│   │   ├── global_motion.py     # shared utility used by motion signals
│   │   └── optical_flow.py      # shared utility
│   ├── schedulers/
│   │   ├── base.py / hysteresis_binary.py / cusum.py
│   │   ├── multi_tier.py / adaptive_threshold.py / trajectory_aware.py
│   ├── kalman/constant_velocity.py
│   ├── datasets/{base,uav123,otb100,lasot,registry}.py
│   ├── metrics/{success,precision,flops,timing,restart_ope}.py
│   ├── evaluation/{ope,report,sweep}.py
│   ├── viz/{entropy_plot,success_plot,overlay,video,signal_dashboard}.py
│   └── cli.py
├── scripts/                     # download_datasets, download_weights, run_benchmark,
│                                # run_ablation, demo, export_onnx, etc.
├── notebooks/                   # data sanity, signal sandbox, ablation analysis, paper figures
├── tests/
│   ├── unit/                    # per-module math + behavior
│   ├── contract/                # plugin protocol conformance
│   ├── integration/             # end-to-end on tiny fixtures
│   ├── property/                # hypothesis-based (entropy bounds, scheduler invariants)
│   └── fixtures/
├── infra/{docker,github,terraform}
├── .github/workflows/
├── data/ weights/ results/ .devcontainer/
```

---

## 10. Fidelity Contract (softened in v2)

We make two distinct fidelity claims:

### 10.1 Methodological fidelity (hard contract — Architect gate)
Each of the following is verified by a dedicated test; failure blocks merge.

- Motion entropy pipeline matches the paper's stages: Shi-Tomasi → LK → global flow subtraction → 16-bin magnitude-weighted histogram → Shannon entropy → `H̃ = H / log₂(N)` → EMA with `α = 0.8`.
- Hysteresis state machine has the paper's transition rules (E_hi / E_lo / 5-frame confirm / 5-frame cooldown).
- KCF integrates with Kalman for ROI smoothing and occlusion coasting.
- Global motion is estimated separately on background keypoints and subtracted.
- OPE evaluation protocol matches OTB (init on frame 0, no re-init on failure).

### 10.2 Numerical fidelity (soft target — informational)
We'll run `paper_entropy_hybrid.yaml` on UAV123 and OTB100 and **report** the delta to Table 2:

| Check | Expected behavior |
|---|---|
| If within ±0.03 AUC | declare "numerically consistent with paper"; proceed |
| If outside ±0.03 AUC but gap is systematic across baselines | declare "hardware/impl drift"; document the scaling factor |
| If only Entropy-Hybrid is off | investigate the entropy engine or scheduler specifically |

We do **not** gate merges on Table 2 numbers. We gate on §10.1 methodological fidelity + our own numerical floor (hybrid beats KCF by ≥0.1 AUC on UAV123; see §12).

---

## 11. Iterative Phases

Each phase ends with a **working, demo-able CLI command** and a **small increment of scientific insight**. Phases are ordered so that each one is standalone-valuable — you could stop after any of them and still have a useful artifact.

### Phase close-out convention
After the last ticket of a phase merges, the Architect opens a `chore(phase-N): close-out — <summary>` PR that:
- Updates `CHANGELOG.md` under `## [Unreleased]` with a dated `### Phase N — YYYY-MM-DD — <name>` block listing **Added / Changed / Removed / Fixed** entries sourced from the phase's merged PRs.
- Pastes the exit-demo command + output from §11 into the CHANGELOG block as proof.
- Lists the contributing PRs by number + branch name.
- Cites the ADRs that landed in the phase.

The close-out PR is squash-merged like any other (cross-agent reviewer per §7a.5; no self-merge). No git tag is required for phase close-outs — the dated CHANGELOG block is the discoverable record. Release tags (§7a.7) remain a separate `v<semver>` cadence.

**Local-master workflow (current mode):** until a git remote + branch protection are live (see §7a.4), agents commit directly to `master` with conventional commits. No feature branches, no PR flow — keep ceremony out of a single-repo bootstrap. Each phase still ends with a `chore(phase-N): close-out` commit that lands the CHANGELOG block plus any cross-cutting doc updates; that commit is the canonical per-phase checkpoint. When the repo moves to a remote with enforced branch protection, the full §7a PR flow applies and phase close-outs become PRs.

### Phase 0 — Foundation (1 day)
**Agents:** `[A]` PLAN + AGENTS + ADRs. `[D]` pyproject + Makefile + Docker skeletons + CI skeleton. `[E]` `cli.py doctor` stub.
**Exit demo:**
```
$ make setup && uav-tracker doctor
✓ Python 3.10.14  ✓ OpenCV 4.9 (contrib)  ✓ Torch 2.1.0  ✓ ffmpeg
✓ data root writable  ✓ weights root writable
```
**Deliverable:** nothing tracks yet, but the scaffold is solid.

---

### Phase 1 — OPE Skeleton + Fast Tracker (2 days) — `demoable`
**Goal:** prove the evaluation harness works end-to-end with one tracker.
**Agents:** `[A]` Tracker & Dataset Protocols (ADR-0002, ADR-0003). `[D]` UAV123 + OTB100 download + checksum. `[E]` `KCFKalmanTracker`, dataset loaders, OPE runner, Success/Precision metrics.
**Exit demo:**
```
$ uav-tracker evaluate --tracker kcf_kalman --dataset uav123 --limit 5
Sequence group1_1: AUC=0.41 Pr@20=0.48 FPS=168
... 5 sequences ...
Summary: AUC=0.43 Pr@20=0.52 FPS=162 GFLOPs/frame=0.02
```
**Insight gained:** we know our KCF baseline is close to paper's 0.432.

---

### Phase 2 — Plugin System + Second Tracker (2-3 days) — `demoable`
**Goal:** prove the registry pattern by adding a deep tracker via plugin.
**Agents:** `[A]` Registry design (ADR-0004), plugin-contract test spec. `[D]` Siamese weights download. `[E]` `SiamFCBackend`, `Tracker` registry wiring, config-driven construction.
**Exit demo:**
```
$ uav-tracker evaluate --config configs/experiments/paper_mobiletrack.yaml --limit 5
Summary: AUC=0.68 Pr@20=0.76 FPS=78 GFLOPs/frame=1.2

$ uav-tracker list-plugins
trackers: [kcf_kalman, siamfc]
detectors: []
signals: []
schedulers: []
```
**Insight gained:** plugin contract works; SiamFC serves as the paper's heavy tier.

---

### Phase 3 — First Switching (Trivially Simple) (2 days) — `demoable`
**Goal:** prove the Scheduler abstraction with the simplest possible switch.
**Agents:** `[A]` SwitchSignal + Scheduler Protocols (ADR-0005). `[E]` `TrackerConfidenceSignal`, `HysteresisBinaryScheduler`, `HybridRunner`.
**Exit demo:**
```
$ uav-tracker evaluate --config configs/experiments/hybrid_confidence.yaml --limit 5
Summary: AUC=0.56 Pr@20=0.64 FPS=105 time_in_tier1=31%
```
Scheduler uses KCF's own correlation-peak confidence as the switching signal. Not the paper's method — but proves the composition works.
**Insight gained:** hybrid runner correctly switches, composes, and measures.

---

### Phase 4 — Paper's Motion Entropy Signal (3 days) — `demoable, paper-fidelity`
**Goal:** implement the paper's core contribution.
**Agents:** `[A]` ADR-0006 on global-motion fallback strategy. `[E]` `MotionEntropy` signal: Shi-Tomasi + pyramidal LK + RANSAC homography with LMedS fallback + magnitude-weighted 16-bin histogram + Shannon `H̃` + EMA. Property tests (delta/uniform closed forms, monotone under dispersion). Synthetic fixtures (translating rectangle → `H̃<0.15`; noise rectangle → `H̃>0.85`).
**Exit demo:**
```
$ uav-tracker evaluate --config configs/experiments/paper_entropy_hybrid.yaml --limit 5
Summary: AUC=0.59 Pr@20=0.69 FPS=100 GFLOPs/frame=0.6 time_in_tier1=22%

$ uav-tracker plot-signals --sequence group1_1 --out entropy_timeline.png
(renders H̃(t) with mode bands and switch events)
```
**Insight gained:** paper's logic faithfully implemented. Deviations from Table 2 quantified.

---

### Phase 5 — Alternate Signals & Schedulers (3 days) — `demoable, research-add`
**Goal:** ship the deviations from §2 and compare them to the paper's defaults.
**Agents:** `[A]` ADR-0007 on signal/scheduler comparison protocol. `[E]` implement `CircularResultant`, `APCE`, `FlowDivergence` signals; `CUSUMScheduler`, `AdaptiveThresholdScheduler`, `TrajectoryAwareScheduler`. Ablation script.
**Exit demo:**
```
$ uav-tracker ablate --sweep configs/experiments/ablation_signals.yaml --limit 20
Signal                 AUC    Pr@20   FPS   Switches/seq
motion_entropy         0.59   0.69    100   14
circular_resultant     0.60   0.70    102   11       ← candidate new default
apce                   0.57   0.66    110   19
flow_divergence        0.55   0.62    98    22
```
**Insight gained:** empirical verdict on whether circular statistics actually beat Shannon entropy on this task. Publishable result in its own right.

---

### Phase 6 — Detection Tier (3 days) — `demoable, research-add`
**Goal:** ship the paper's "future work" recommendation.
**Agents:** `[A]` ADR-0008 on 3-tier scheduling semantics. `[D]` YOLOv8-n weights + model card. `[E]` `Detector` protocol, `YOLOv8Detector`, `MultiTierScheduler` (generalizes binary hysteresis to N tiers). Re-detection trigger: `confidence<τ_lost` for ≥M frames → DETECT tier → reinitialize trackers on best-IoU detection vs last-known bbox.
**Exit demo:**
```
$ uav-tracker evaluate --config configs/experiments/hybrid_with_detection.yaml --limit 20
Summary: AUC=0.62 Pr@20=0.73 FPS=88 GFLOPs/frame=0.9
  time_in_tier0=68% time_in_tier1=29% time_in_tier2=3%
  recoveries=7 (from complete loss)
```
**Insight gained:** detection tier closes the "target disappears + reappears far" failure mode that paper flags.

---

### Phase 7 — Full Benchmark & Ablation Reproduction (3 days) — `paper-reproduction`
**Goal:** produce every table and figure for the paper + our extensions.
**Agents:** `[D]` self-hosted T4 runner, nightly-eval workflow. `[E]` `run_benchmark.py`, `run_ablation.py`, seed-pinned, result-cached. Reports:
  - Table 2 reproduction (paper baselines + entropy hybrid).
  - Table 3 reproduction (ablation).
  - Extended Table 4 (our additions: signals × schedulers × detection ON/OFF).
  - Per-attribute breakdown on UAV123 (FM, OCC, IV, LR).
  - Restart-based eval on a 20-sequence subset.
**Exit demo:** `make reproduce` regenerates all results + markdown tables + figures from a clean checkout.

---

### Phase 8 — Visualization, Demo, & Paper Figures (1.5 days) — `paper-ready`
**Agents:** `[E]` entropy-vs-time plots with mode bands & switches; OPE success/precision curves; demo MP4 with on-screen bbox + mode badge + live signal gauges. `[A]` finalize figure set for article. `[D]` Git LFS for figures.
**Exit demo:**
```
$ uav-tracker demo --sequence group1_wakeboard2 --out demo.mp4
$ uav-tracker figures --out-dir results/figures/
(generates all figures needed for the article)
```

---

### Phase 9 — Edge Deployment (stretch, 2.5 days)
**Agents:** `[D]` Jetson Orin Nano provisioning, `Dockerfile.jetson`, TensorRT build. `[E]` ONNX export for each heavy tracker, TRT backend under `Tracker` protocol. `[A]` edge-deployment report (FPS/W tradeoffs).
**Exit demo:** hybrid (with entropy signal, ours config) ≥50 FPS at <10 W on Orin Nano.

---

### Phase 10 — Research Extensions (open-ended)
Optional, post-article. Each ships as a new plugin under the existing architecture — no refactor required. Ideas:

- `LearnedGateSignal` — small MLP trained on hand-crafted features (entropy, confidence, APCE, flow magnitude) to predict "will KCF fail in next K frames".
- `BayesianOptScheduler` — online tuning of thresholds per sequence.
- Multi-object extension via `MultiObjectHybridRunner`.
- Online adaptation of Siamese template (SiamR-CNN-style).
- Cross-modal signals: IMU-derived ego-motion as a reliability prior.

---

## 12. Testing Strategy (v2)

### 12.1 Unit tests `[E]`
- Entropy math: delta/uniform closed forms; `H̃ ∈ [0,1]`; monotonic under dispersion.
- Circular resultant: `R=1` for coherent, `R=0` for uniform.
- Histogram: weight conservation; empty → 0.
- Scheduler state machines: golden traces; no-flap invariant; min-gap invariant.
- Kalman: matches analytic CV solution.

### 12.2 Property tests `[E]`
- Every `SwitchSignal`: `range[0] ≤ value ≤ range[1]` for arbitrary synthetic contexts.
- Every `Scheduler`: between two switches, at least `cooldown_frames` elapse.
- `HybridRunner`: deterministic given fixed seed.

### 12.3 Contract tests `[A]`-owned, `[E]`-maintained
Enumerate registry, assert Protocol conformance, assert `reset()` is idempotent, assert no cross-plugin imports.

### 12.4 Integration tests `[E]`
Tiny 20-frame fixtures with known ground truth:
- Smooth translation → tracker stays in tier 0.
- Injected erratic motion → tracker switches to tier 1.
- Target-exit-then-reenter → tracker escalates to tier 2 (when detector config active).

### 12.5 Smoke eval in CI `[D]`
3 UAV123 sequences (one easy, one occlusion, one fast motion). CPU-only. Must complete in <5 min; AUC bands asserted.

### 12.6 Performance regression `[E]`+`[D]`
`pytest-benchmark` guards: KCF ≥150 FPS, entropy signal ≥500 FPS, hybrid (tier-0 steady state) ≥90 FPS.

### 12.7 Numerical reproducibility `[D]`
Fixed seed → identical metrics across runs on same image. Golden metrics regenerated only via `--update-goldens`.

---

## 13. Experiments to Run

| # | Experiment | Config | What it answers |
|---|---|---|---|
| 1 | Paper Table 2 | `experiments/paper_table2.yaml` | do we hit reference numbers on the paper's configs? |
| 2 | Paper Table 3 ablation | `experiments/paper_table3.yaml` | does entropy switch beat APCE in paper's setup? |
| 3 | Signals sweep | `experiments/ablation_signals.yaml` | which switching signal is best (Shannon vs circular vs APCE vs divergence)? |
| 4 | Schedulers sweep | `experiments/ablation_schedulers.yaml` | hysteresis vs CUSUM vs adaptive vs trajectory-aware? |
| 5 | Detection tier impact | `experiments/ablation_detection.yaml` | what does a third tier buy us (AUC gain vs FLOPs cost)? |
| 6 | Threshold sensitivity | `experiments/sweep_thresholds.yaml` | how fragile are E_hi/E_lo defaults across datasets? |
| 7 | Warm-standby cost/benefit | `experiments/ablation_warm_standby.yaml` | is the 10-frame standby worth its compute? |
| 8 | Per-attribute breakdown | reuses (1) outputs | which attributes benefit most from switching? |
| 9 | Restart-based eval | `experiments/restart_eval.yaml` | does AUC advantage hold under harder protocol? |
| 10 | Edge benchmark | `experiments/edge_jetson.yaml` | FPS/watt on Orin Nano per-config |

Experiment results always include provenance (git SHA, dataset SHA, weights SHA, hardware, image tag). Every experiment is re-runnable via cached per-sequence results.

---

## 14. Risks & Mitigations (v2)

| Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|
| UAV123 upstream URL rot | M | H | S3/B2 mirror + checksum manifest | `[D]` |
| MobileTrack weights unreachable | M | H | SiamFC default; adapter layer; document in ADR | `[A]`+`[D]` |
| Plugin architecture over-engineered | L | M | Contract test enforces minimal surface; Architect pushes back on gratuitous abstraction | `[A]` |
| Paper numbers don't reproduce | M | M | Soft contract (see §10); investigate but don't block | `[A]`+`[E]` |
| Too many plugins → choice paralysis | M | L | Default configs curated; ablation scripts tell you what to pick | `[A]` |
| CI GPU runner cost creep | M | M | Auto-suspend + monthly cap | `[D]` |
| Low-texture frames break global-motion | M | M | Three-level fallback from day one | `[E]` |
| Detector weights size explodes repo | L | M | Never commit weights; download script only | `[D]` |
| Scope creep into training/learning | H | M | Phase 10 is the only home for learning; everything else rule-based | `[A]` |

---

## 15. Definition of Done

Per **phase**: exit demo runs cleanly, smoke-eval green, phase tests green, Architect acceptance comment posted.

Per **repo**: 
- [ ] Phases 0–8 merged.
- [ ] `make reproduce` regenerates all tables from a clean checkout.
- [ ] `make setup` under 15 min on clean macOS + Ubuntu.
- [ ] CI green on main; nightly green 5 consecutive nights.
- [ ] `uav-tracker list-plugins` shows ≥3 trackers, ≥1 detector, ≥4 signals, ≥4 schedulers.
- [ ] Plugin contract test green — adding a new tracker takes <50 LOC + YAML.
- [ ] Paper figures + demo MP4 committed via LFS.
- [ ] All §2 deviations have an ADR + empirical result.

---

## 16. Open Questions (Architect drives resolution via ADRs)

From paper, unchanged:
1. Siamese variant: stock SiamFC default; MobileTrack-if-available as optional plugin.
2. APCE calibration split: OTB50.
3. Cooldown vs confirm as independent knobs (both default 5).
4. Residual-entropy threshold `r_max`: sweep on camera-shake sequences.
5. Edge target: Jetson Orin Nano.
6. Experiment tracker: W&B offline + CSV always.
7. GPU runner: GCP T4 with auto-suspend.

New in v2:
8. Default switching signal: paper's `MotionEntropy` for fidelity, or our `CircularResultant` for robustness? → decide post Phase 5 ablation.
9. Detection tier on by default? → no; opt-in via experiment config.
10. Multi-object: defer to Phase 10 or never? → defer; keep Protocol multi-object-capable but single-object runner.

---

## 17. First Wave of Tickets

Created immediately by Architect after this v2 plan is accepted:

**`[A]` tickets**
- `ADR-0001`: agent model & handoff (ratifies §5 / AGENTS.md).
- `ADR-0002`: Dataset/Sequence Protocol.
- `ADR-0003`: Tracker Protocol + TrackState schema.
- `ADR-0004`: Registry pattern + plugin-contract tests.
- `ADR-0005`: SwitchSignal + Scheduler Protocols.
- `ADR-0006`: Global-motion fallback strategy.
- `ADR-0007`: Signal/scheduler comparison protocol.
- `ADR-0008`: 3-tier scheduling semantics.

**`[D]` tickets**
- `INFRA-0001…0006`: unchanged from v1 (pyproject, CI, Dockerfiles, downloads, pre-commit, gitleaks).
- `INFRA-0007`: plugin-registry health workflow.
- `INFRA-0008`: YOLO weights download (phase 6 gate).

**`[E]` tickets**
- `ENG-0001`: `cli.py` with `doctor` + `list-plugins` + `evaluate`.
- `ENG-0002`: `datasets/{uav123,otb100}.py` + tests.
- `ENG-0003`: `trackers/kcf_kalman.py` + tests.
- `ENG-0004`: `registry.py` + registry unit tests.
- `ENG-0005`: `runner.py` HybridRunner skeleton.
- `ENG-0006`: `signals/motion_entropy.py` + property tests.
- `ENG-0007`: `schedulers/hysteresis_binary.py` + property tests.

---

## 18. Communication & Reporting

- **Daily:** each agent posts standup on `status/current-phase` issue.
- **Per PR:** owner + reviewer front-matter; smoke-eval green.
- **End of phase:** Architect posts acceptance review + records `docs/status/phase-N.md` with the demo command's output and the insight gained.
- **Post Phase 5 and Phase 6:** publish a short note in `docs/research/` summarizing empirical findings on signal comparison and detection tier value — these are citable artifacts for the PhD thesis.
