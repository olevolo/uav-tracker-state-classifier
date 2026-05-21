"""test_saltr_policy_scripts.py — Import-level and structural tests for policy scripts.

Tests:
1. train_policy module is importable
2. calibrate_policy module is importable
3. rollout_policy module is importable
4. No TSA imports in any of the three modules
5. All three modules have a main() function
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure salt_r is on the path before any imports
# ---------------------------------------------------------------------------

_SALT_R_SRC = str(Path(__file__).parents[2] / "saltr" / "src")
if _SALT_R_SRC not in sys.path:
    sys.path.insert(0, _SALT_R_SRC)

# Module names to test
_POLICY_MODULE_NAMES = [
    "salt_r.train_policy",
    "salt_r.calibrate_policy",
    "salt_r.rollout_policy",
]


# ---------------------------------------------------------------------------
# Test 1: train_policy is importable
# ---------------------------------------------------------------------------

def test_train_policy_importable():
    """train_policy module must be importable without side-effects."""
    mod = importlib.import_module("salt_r.train_policy")
    assert mod is not None, "salt_r.train_policy failed to import"


# ---------------------------------------------------------------------------
# Test 2: calibrate_policy is importable
# ---------------------------------------------------------------------------

def test_calibrate_policy_importable():
    """calibrate_policy module must be importable without side-effects."""
    mod = importlib.import_module("salt_r.calibrate_policy")
    assert mod is not None, "salt_r.calibrate_policy failed to import"


# ---------------------------------------------------------------------------
# Test 3: rollout_policy is importable
# ---------------------------------------------------------------------------

def test_rollout_policy_importable():
    """rollout_policy module must be importable without side-effects."""
    mod = importlib.import_module("salt_r.rollout_policy")
    assert mod is not None, "salt_r.rollout_policy failed to import"


# ---------------------------------------------------------------------------
# Test 4: No TSA imports in any of the three modules
# ---------------------------------------------------------------------------

_TSA_PATTERNS = ["tsa", "TSA", "tracker_state_annotation", "TrackerStateAnnotation"]


def _source_for_module(module_name: str) -> str:
    """Return the source code of a module."""
    mod = importlib.import_module(module_name)
    src_file = inspect.getfile(mod)
    return Path(src_file).read_text(encoding="utf-8")


@pytest.mark.parametrize("module_name", _POLICY_MODULE_NAMES)
def test_no_tsa_imports(module_name: str):
    """None of the three policy scripts must import from TSA modules."""
    source = _source_for_module(module_name)
    for pattern in _TSA_PATTERNS:
        # Check import lines only: "import tsa", "from tsa import ...", etc.
        import_lines = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith(("import ", "from "))
        ]
        for line in import_lines:
            assert pattern not in line, (
                f"{module_name}: found TSA import pattern '{pattern}' in line: {line!r}"
            )


# ---------------------------------------------------------------------------
# Test 5: All three modules have a main() function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_name", _POLICY_MODULE_NAMES)
def test_module_has_main(module_name: str):
    """Each policy script must expose a top-level main() callable."""
    mod = importlib.import_module(module_name)
    assert hasattr(mod, "main"), f"{module_name} has no 'main' attribute"
    assert callable(mod.main), f"{module_name}.main is not callable"
