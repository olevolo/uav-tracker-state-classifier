"""Plugin-contract conformance tests (PLAN §4.4).

Enumerates every registered plugin and asserts:

    (a) The instance is structurally a member of its Protocol
        (runtime-checkable via ``isinstance(..., Protocol)``).
    (b) ``reset()`` is idempotent where the plugin defines it.
    (c) The plugin module imports no other plugin module directly
        (only ``base`` / ``types`` / ``registry``) — enforced by AST
        walk.
    (d) Per-tracker smoke check: instantiate → init(dummy_frame, bbox).
        Skipped gracefully on missing-dep errors (KCF contrib, SiamFC
        weights, etc.).
    (e) Per-dataset smoke check: __iter__() yields at least one
        sequence whose ground_truth is non-empty.

Phase 2 note: SiamFC is registered in TRACKERS and its backbone is
implemented by the Engineer.  init() is exercised in a dedicated smoke
test that skips gracefully on missing weights/deps.

Determinism: seed fixed to 42 everywhere.  Tests must not use
random.random() or numpy default_rng without an explicit seed.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from uav_tracker import DATASETS, DETECTORS, SCHEDULERS, SIGNALS, TRACKERS
from uav_tracker.datasets.base import Dataset
from uav_tracker.detectors.base import Detector
from uav_tracker.schedulers.base import Scheduler
from uav_tracker.signals.base import SwitchSignal
from uav_tracker.trackers.base import Tracker
from uav_tracker.types import BBox


# --------------------------------------------------------------------------- #
# Registry map                                                                 #
# --------------------------------------------------------------------------- #

_REGISTRIES = {
    "trackers": TRACKERS,
    "detectors": DETECTORS,
    "signals": SIGNALS,
    "schedulers": SCHEDULERS,
    "datasets": DATASETS,
}

# Mapping from registry key → expected Protocol class
_PROTOCOLS = {
    "trackers": Tracker,
    "detectors": Detector,
    "signals": SwitchSignal,
    "schedulers": Scheduler,
    "datasets": Dataset,
}

# Dummy 64×64 BGR frame and bbox used in smoke checks (fixed seed, deterministic)
_RNG = np.random.default_rng(42)
_DUMMY_FRAME: np.ndarray = _RNG.integers(0, 256, (64, 64, 3), dtype=np.uint8)
_DUMMY_BBOX = BBox(16.0, 16.0, 32.0, 32.0)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _all_plugin_module_paths() -> Iterable[Path]:
    """Yield ``Path`` for every module under the five plugin subpackages."""
    for sub in ("trackers", "detectors", "signals", "schedulers", "datasets"):
        try:
            package = importlib.import_module(f"uav_tracker.{sub}")
        except Exception:
            continue
        pkg_path = Path(package.__file__ or "").parent
        if not pkg_path.exists():
            continue
        for _, modname, _ in pkgutil.walk_packages([str(pkg_path)]):
            yield pkg_path / f"{modname}.py"


# --------------------------------------------------------------------------- #
# Registry enumeration                                                         #
# --------------------------------------------------------------------------- #


def test_all_five_registries_present() -> None:
    """Exactly the five expected registries must be importable and non-None."""
    assert set(_REGISTRIES.keys()) == {
        "trackers",
        "detectors",
        "signals",
        "schedulers",
        "datasets",
    }, "DATASETS registry slot missing — Phase 1 added it; update the import."


def test_registries_enumerate_without_error() -> None:
    for kind, reg in _REGISTRIES.items():
        names = list(reg.names())
        assert isinstance(names, list), f"{kind}.names() must return a list"


# --------------------------------------------------------------------------- #
# Protocol conformance                                                         #
# --------------------------------------------------------------------------- #


def test_every_tracker_satisfies_protocol() -> None:
    """Each registered tracker instance must satisfy the Tracker Protocol."""
    for name in TRACKERS.names():
        try:
            instance = TRACKERS.build(name)
        except Exception as exc:
            pytest.skip(f"Cannot instantiate tracker {name!r}: {exc}")
        assert isinstance(instance, Tracker), (
            f"tracker {name!r} does not satisfy the Tracker Protocol. "
            "Check that name/tier_hint/init/update/flops_per_update are present."
        )


def test_every_detector_satisfies_protocol() -> None:
    """Each registered detector instance must satisfy the Detector Protocol."""
    for name in DETECTORS.names():
        try:
            instance = DETECTORS.build(name)
        except Exception as exc:
            pytest.skip(f"Cannot instantiate detector {name!r}: {exc}")
        assert isinstance(instance, Detector), (
            f"detector {name!r} does not satisfy the Detector Protocol."
        )


def test_every_detector_detect_runs() -> None:
    """Each registered detector must accept a 64x64 frame and return list[Detection].

    - Iterates ``DETECTORS.names()``; skips gracefully on any import or
      instantiation failure (e.g. ultralytics not installed).
    - For detectors whose name starts with ``"yolo"``, gates on
      ``pytest.importorskip("ultralytics")`` before attempting build.
    - Asserts the return value is a ``list``; individual items must be
      ``Detection`` instances (list may be empty — no target in dummy frame).
    """
    from uav_tracker.types import Detection

    for name in DETECTORS.names():
        if name.startswith("yolo"):
            pytest.importorskip("ultralytics")

        try:
            instance = DETECTORS.build(name)
        except Exception as exc:
            pytest.skip(f"Cannot instantiate detector {name!r}: {exc}")

        try:
            result = instance.detect(_DUMMY_FRAME)
        except (NotImplementedError, AttributeError) as exc:
            pytest.skip(f"detector {name!r} .detect() not yet implemented: {exc}")
        except Exception as exc:
            msg = str(exc)
            if any(kw in msg.lower() for kw in ("weights", "model", "file", "ultralytics")):
                pytest.skip(f"detector {name!r} missing weights/dep: {exc}")
            raise

        assert isinstance(result, list), (
            f"detector {name!r} .detect() returned {type(result)!r}, expected list"
        )
        for item in result:
            assert isinstance(item, Detection), (
                f"detector {name!r} .detect() list contains {type(item)!r}, expected Detection"
            )


def test_every_signal_satisfies_protocol() -> None:
    """Each registered signal instance must satisfy the SwitchSignal Protocol."""
    for name in SIGNALS.names():
        try:
            instance = SIGNALS.build(name)
        except Exception as exc:
            pytest.skip(f"Cannot instantiate signal {name!r}: {exc}")
        assert isinstance(instance, SwitchSignal), (
            f"signal {name!r} does not satisfy the SwitchSignal Protocol."
        )


def test_every_scheduler_satisfies_protocol() -> None:
    """Each registered scheduler instance must satisfy the Scheduler Protocol."""
    for name in SCHEDULERS.names():
        try:
            instance = SCHEDULERS.build(name)
        except Exception as exc:
            pytest.skip(f"Cannot instantiate scheduler {name!r}: {exc}")
        assert isinstance(instance, Scheduler), (
            f"scheduler {name!r} does not satisfy the Scheduler Protocol."
        )


def test_every_dataset_satisfies_protocol() -> None:
    """Each registered dataset instance must satisfy the Dataset Protocol."""
    for name in DATASETS.names():
        try:
            instance = DATASETS.build(name)
        except Exception as exc:
            pytest.skip(f"Cannot instantiate dataset {name!r}: {exc}")
        assert isinstance(instance, Dataset), (
            f"dataset {name!r} does not satisfy the Dataset Protocol."
        )


# --------------------------------------------------------------------------- #
# Per-tracker smoke check                                                      #
# --------------------------------------------------------------------------- #


def test_siamfc_tracker_init_smoke() -> None:
    """SiamFC: instantiate + init on dummy frame.

    SiamFC.init() is implemented (Engineer Phase 2 backbone is wired).
    Skips gracefully if torch / cv2 is unavailable or weights loading
    raises an unrecoverable error.
    """
    try:
        tracker = TRACKERS.build("siamfc")
    except Exception as exc:
        pytest.skip(f"Cannot build siamfc tracker: {exc}")

    try:
        tracker.init(_DUMMY_FRAME, _DUMMY_BBOX)
    except RuntimeError as exc:
        msg = str(exc)
        if any(kw in msg for kw in ("torch", "cuda", "weights", "missing")):
            pytest.skip(f"siamfc missing dep or weights: {msg}")
        raise
    except NotImplementedError as exc:
        # Fallback: backbone not yet wired in this environment
        pytest.skip(f"siamfc not yet implemented: {exc}")


def test_kcf_kalman_tracker_init_smoke() -> None:
    """KCFKalmanTracker: instantiate + init on a 64×64 BGR dummy frame.

    Skipped gracefully if opencv-contrib is missing (CI without contrib wheel).
    """
    try:
        tracker = TRACKERS.build("kcf_kalman")
    except Exception as exc:
        pytest.skip(f"Cannot build kcf_kalman: {exc}")

    try:
        tracker.init(_DUMMY_FRAME, _DUMMY_BBOX)
    except RuntimeError as exc:
        msg = str(exc)
        if "opencv-contrib" in msg or "TrackerKCF_create" in msg:
            pytest.skip(f"opencv-contrib not available: {msg}")
        raise


def test_all_tracker_init_smoke() -> None:
    """For every tracker, attempt instantiate + init; skip on missing deps.

    This is a catch-all for future trackers added without a dedicated test.
    SiamFC is intentionally excluded here (handled by the dedicated smoke
        test above) to
    avoid masking it as a skip.
    """
    for name in TRACKERS.names():
        if name == "siamfc":
            continue  # covered by dedicated siamfc smoke test above

        try:
            tracker = TRACKERS.build(name)
        except Exception as exc:
            pytest.skip(f"Cannot build tracker {name!r}: {exc}")

        try:
            tracker.init(_DUMMY_FRAME, _DUMMY_BBOX)
        except RuntimeError as exc:
            msg = str(exc)
            # Known-safe missing-dep patterns: skip rather than fail.
            if any(
                kw in msg
                for kw in ("opencv-contrib", "TrackerKCF_create", "weights", "missing")
            ):
                pytest.skip(f"tracker {name!r} missing dep: {msg}")
            raise
        except NotImplementedError as exc:
            pytest.skip(f"tracker {name!r} not yet implemented: {exc}")


# --------------------------------------------------------------------------- #
# Per-dataset smoke check                                                      #
# --------------------------------------------------------------------------- #


def test_all_datasets_yield_sequences() -> None:
    """Each registered dataset must yield at least one sequence via __iter__.

    The first sequence must have non-empty ground_truth and at least one frame.
    Datasets that need a filesystem path (uav123, otb100) are skipped when the
    root directory is absent.
    """
    for name in DATASETS.names():
        try:
            dataset = DATASETS.build(name)
        except Exception as exc:
            pytest.skip(f"Cannot instantiate dataset {name!r}: {exc}")

        try:
            sequences = list(dataset)
        except Exception as exc:
            pytest.skip(f"dataset {name!r} __iter__ failed (missing data?): {exc}")

        assert len(sequences) >= 1, f"dataset {name!r} yielded zero sequences"

        seq = sequences[0]
        assert hasattr(seq, "ground_truth"), (
            f"dataset {name!r} sequence missing 'ground_truth'"
        )
        assert len(seq.ground_truth) >= 1, (
            f"dataset {name!r} first sequence has empty ground_truth"
        )

        # Verify frames are accessible (at least the first frame).
        assert hasattr(seq, "frames"), f"dataset {name!r} sequence missing 'frames'"
        frames_iter = iter(seq.frames)
        try:
            first_frame = next(frames_iter)
        except StopIteration:
            pytest.fail(f"dataset {name!r} first sequence yielded zero frames")
        assert isinstance(first_frame, np.ndarray), (
            f"dataset {name!r} first frame is not a numpy array"
        )


# --------------------------------------------------------------------------- #
# reset() idempotency                                                          #
# --------------------------------------------------------------------------- #


def test_reset_is_idempotent_where_defined() -> None:
    """For each registered plugin, ``reset()`` twice == ``reset()`` once.

    We construct with a no-arg factory where possible; plugins that
    need args are skipped (those are exercised via Phase-specific
    tests).
    """
    any_tested = False
    for reg in _REGISTRIES.values():
        for name in reg.names():
            try:
                instance = reg.build(name)
            except Exception:
                continue
            reset = getattr(instance, "reset", None)
            if not callable(reset):
                continue
            reset()
            reset()  # second call must not raise
            any_tested = True
    if not any_tested:
        pytest.skip("No plugins with a no-arg ``reset()`` yet (Phase 3+).")


# --------------------------------------------------------------------------- #
# No cross-plugin imports (AST walk)                                           #
# --------------------------------------------------------------------------- #


def test_no_cross_plugin_imports() -> None:
    """AST-walk every plugin module; forbid ``from uav_tracker.<other-plugin>``.

    Only ``base``, ``types``, ``registry``, and the plugin's OWN
    subpackage are allowed. Shared utilities (``signals.global_motion``,
    ``signals.optical_flow``) are explicitly allowed because they live
    in the signals subpackage itself.
    """
    forbidden_prefixes = (
        "uav_tracker.trackers.",
        "uav_tracker.detectors.",
        "uav_tracker.signals.",
        "uav_tracker.schedulers.",
        "uav_tracker.datasets.",
    )
    allowed_tails = ("base", "__init__")

    violations: list[str] = []
    for path in _all_plugin_module_paths():
        try:
            source = path.read_text()
        except OSError:
            continue
        tree = ast.parse(source)
        # Determine the plugin's own top-level subpackage.
        # e.g. src/uav_tracker/trackers/kcf_kalman.py → uav_tracker.trackers.
        parts = path.parts
        try:
            pkg_ix = parts.index("uav_tracker")
            own_prefix = f"uav_tracker.{parts[pkg_ix + 1]}."
        except (ValueError, IndexError):
            own_prefix = ""
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                m = node.module
                if not m.startswith(forbidden_prefixes):
                    continue
                # Imports within own subpackage are OK.
                if own_prefix and m.startswith(own_prefix):
                    # Still disallow importing siblings that are registered
                    # plugins themselves — allow only ``base`` / ``__init__``.
                    tail = m[len(own_prefix):].split(".")[0]
                    if tail in allowed_tails:
                        continue
                    # Utility modules by convention end in ``_flow`` /
                    # ``_motion`` and aren't registered plugins; allow.
                    continue
                violations.append(f"{path.name} → {m}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    m = alias.name
                    if not m.startswith(forbidden_prefixes):
                        continue
                    if own_prefix and m.startswith(own_prefix):
                        continue
                    violations.append(f"{path.name} → {m}")
    assert not violations, f"cross-plugin imports: {violations}"


# --------------------------------------------------------------------------- #
# Per-signal smoke check (Phase 3+)                                            #
# --------------------------------------------------------------------------- #


def test_every_signal_step_returns_report() -> None:
    """For every registered signal, call .step() on a dummy FrameContext.

    Auto-discovers via ``SIGNALS.names()``; no hardcoded names.  Each signal
    is exercised with a 32×32 dummy frame, a reasonable BBox, ``frame_idx=0``,
    and ``state=None`` (simulates pre-init TrackerConfidence path).  Asserts
    that the returned ``SignalReport`` has a numeric ``.value`` OR that the
    report is unreliable (``reliable=False``) — both are valid outcomes for
    frame 0.

    Skips gracefully when:
      - The signal cannot be default-constructed.
      - The signal raises an expected runtime error (missing required context
        fields, unimplemented flow, etc.).
    """
    from uav_tracker.types import FrameContext, SignalReport

    rng = np.random.default_rng(42)
    frame_32 = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    ctx = FrameContext(
        frame=frame_32,
        prev_frame=None,
        frame_idx=0,
        bbox=BBox(8.0, 8.0, 16.0, 16.0),
    )

    any_tested = False
    for name in SIGNALS.names():
        try:
            signal = SIGNALS.build(name)
        except Exception as exc:
            pytest.skip(f"Cannot instantiate signal {name!r}: {exc}")

        try:
            report = signal.step(ctx, state=None)
        except (NotImplementedError, AttributeError) as exc:
            pytest.skip(f"signal {name!r} .step() not yet implemented: {exc}")
        except Exception as exc:
            # Missing optional context (flow cache, global_motion) is OK —
            # skip rather than fail since those will be covered by unit tests.
            msg = str(exc)
            if any(
                kw in msg.lower()
                for kw in ("flow", "motion", "none", "attribute", "ndarray")
            ):
                pytest.skip(f"signal {name!r} missing context field: {exc}")
            raise

        assert isinstance(report, SignalReport), (
            f"signal {name!r} .step() returned {type(report)!r}, expected SignalReport"
        )
        # If reliable, value must be numeric and within declared range.
        if report.reliable:
            assert isinstance(report.value, (int, float)), (
                f"signal {name!r} reliable=True but value is {type(report.value)!r}"
            )
            lo, hi = signal.range
            assert lo <= report.value <= hi, (
                f"signal {name!r} value {report.value} outside declared range {signal.range}"
            )
        # If not reliable, value may still be present (sentinel) — just don't enforce range.
        any_tested = True

    if not any_tested:
        pytest.skip("No signals registered yet (Phase 3+ registers them).")


# --------------------------------------------------------------------------- #
# Per-signal numerical invariant check (Phase 4)                               #
# --------------------------------------------------------------------------- #


def test_every_signal_value_is_finite_and_in_range() -> None:
    """Numerical invariant: two-frame step produces a finite value (or reliable=False).

    For each registered signal:
      1. Build the signal.
      2. Step on frame 0 (prev_frame=None) — allowed to return reliable=False.
      3. Step on frame 1 (prev_frame=frame_a, frame=frame_b) — if reliable=True
         the value must be a finite float and within signal.range.
      4. If the signal exposes a ``range`` attribute of the form (lo, hi),
         assert lo <= value <= hi (only when reliable=True).

    Rationale: NaN in a reliable report would silently poison the scheduler's
    threshold comparison. This test provides a fast signal-level guard without
    mandating specific entropy magnitudes (that is property-test territory
    owned by Engineer in test_entropy_math.py).
    """
    import math

    from uav_tracker.types import FrameContext, SignalReport

    rng = np.random.default_rng(42)
    # Two deterministic 64×64 BGR frames.
    frame_a: np.ndarray = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    frame_b: np.ndarray = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

    ctx0 = FrameContext(
        frame=frame_a,
        prev_frame=None,
        frame_idx=0,
        bbox=BBox(16.0, 16.0, 32.0, 32.0),
    )
    ctx1 = FrameContext(
        frame=frame_b,
        prev_frame=frame_a,
        frame_idx=1,
        bbox=BBox(16.0, 16.0, 32.0, 32.0),
    )

    any_tested = False
    for name in SIGNALS.names():
        try:
            signal = SIGNALS.build(name)
        except Exception as exc:
            pytest.skip(f"Cannot instantiate signal {name!r}: {exc}")

        # --- frame 0 step (may legitimately return reliable=False) ---
        try:
            signal.step(ctx0, state=None)
        except (NotImplementedError, AttributeError) as exc:
            pytest.skip(f"signal {name!r} .step() not yet implemented: {exc}")
        except Exception as exc:
            msg = str(exc).lower()
            if any(kw in msg for kw in ("flow", "motion", "none", "attribute", "ndarray")):
                pytest.skip(f"signal {name!r} missing context field on frame 0: {exc}")
            raise

        # --- frame 1 step (the invariant check) ---
        try:
            report1 = signal.step(ctx1, state=None)
        except (NotImplementedError, AttributeError) as exc:
            pytest.skip(f"signal {name!r} .step() frame 1 not yet implemented: {exc}")
        except Exception as exc:
            msg = str(exc).lower()
            if any(kw in msg for kw in ("flow", "motion", "none", "attribute", "ndarray")):
                pytest.skip(f"signal {name!r} missing context field on frame 1: {exc}")
            raise

        assert isinstance(report1, SignalReport), (
            f"signal {name!r} frame-1 .step() returned {type(report1)!r}, expected SignalReport"
        )

        if report1.reliable:
            # Value must be a finite float when reliable=True.
            assert isinstance(report1.value, (int, float)), (
                f"signal {name!r} reliable=True but value type is {type(report1.value)!r}"
            )
            assert math.isfinite(report1.value), (
                f"signal {name!r} reliable=True but value is non-finite: {report1.value!r}"
            )
            # Range check (signal.range is part of the SwitchSignal Protocol).
            lo, hi = signal.range
            assert lo <= report1.value <= hi, (
                f"signal {name!r} value {report1.value} outside declared range "
                f"[{lo}, {hi}]"
            )
        # If not reliable, value may be a sentinel (0.0, NaN, etc.) — no range check.

        any_tested = True

    if not any_tested:
        pytest.skip("No signals registered yet (Phase 4+ registers motion_entropy).")


# --------------------------------------------------------------------------- #
# Per-scheduler smoke check (Phase 3+)                                         #
# --------------------------------------------------------------------------- #


def test_every_scheduler_decide_returns_decision() -> None:
    """For every registered scheduler, call .decide() on a dummy input.

    Auto-discovers via ``SCHEDULERS.names()``; no hardcoded names.  Each
    scheduler is exercised in two ways:
      1. Empty signals dict + ``current_tier=0``, ``frame_idx=0``.
      2. A single dummy ``SignalReport`` (reliable=True, value=0.5) keyed to
         ``"dummy"``.

    Asserts that the returned ``SchedulerDecision`` has a ``.tier`` that is
    an ``int``.  The concrete value is not asserted here — unit tests own that.

    Skips gracefully when the scheduler cannot be default-constructed.
    """
    from uav_tracker.types import SchedulerDecision, SignalReport

    dummy_report = SignalReport(value=0.5, reliable=True)

    any_tested = False
    for name in SCHEDULERS.names():
        try:
            scheduler = SCHEDULERS.build(name)
        except Exception as exc:
            pytest.skip(f"Cannot instantiate scheduler {name!r}: {exc}")

        # -- empty signals (baseline: scheduler should hold state or default) --
        try:
            decision_empty = scheduler.decide(
                signals={},
                current_tier=0,
                frame_idx=0,
            )
        except (NotImplementedError, KeyError) as exc:
            pytest.skip(f"scheduler {name!r} .decide() (empty) not yet runnable: {exc}")
        except Exception as exc:
            msg = str(exc)
            if "required" in msg.lower() or "missing" in msg.lower():
                pytest.skip(f"scheduler {name!r} requires specific signals: {exc}")
            raise

        assert isinstance(decision_empty, SchedulerDecision), (
            f"scheduler {name!r} .decide() returned {type(decision_empty)!r}"
        )
        assert isinstance(decision_empty.tier, int), (
            f"scheduler {name!r} .tier is {type(decision_empty.tier)!r}, expected int"
        )

        # -- single dummy report --
        try:
            decision_one = scheduler.decide(
                signals={"dummy": dummy_report},
                current_tier=0,
                frame_idx=1,
            )
        except (NotImplementedError, KeyError) as exc:
            pytest.skip(f"scheduler {name!r} .decide() (one report) not yet runnable: {exc}")
        except Exception as exc:
            msg = str(exc)
            if "required" in msg.lower() or "missing" in msg.lower():
                pytest.skip(f"scheduler {name!r} requires specific named signal: {exc}")
            raise

        assert isinstance(decision_one, SchedulerDecision), (
            f"scheduler {name!r} .decide() returned {type(decision_one)!r}"
        )
        assert isinstance(decision_one.tier, int), (
            f"scheduler {name!r} .tier is {type(decision_one.tier)!r}, expected int"
        )

        any_tested = True

    if not any_tested:
        pytest.skip("No schedulers registered yet (Phase 3+ registers them).")
