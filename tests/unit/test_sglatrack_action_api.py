"""Tests for SGLATracker.update_with_action — no TargetState, no TSA."""
import ast
import inspect
import textwrap
import pytest


def test_no_targetstate_import():
    from uav_tracker.trackers import sglatrack as m
    # Check update_with_action method source specifically
    cls = m.SGLATracker
    method_src = textwrap.dedent(inspect.getsource(cls.update_with_action))
    tree = ast.parse(method_src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                assert "tsa" not in (name or "").lower(), f"TSA import in update_with_action: {name}"
                assert "target_state" not in (name or "").lower()


def test_update_with_action_exists():
    from uav_tracker.trackers.sglatrack import SGLATracker
    assert hasattr(SGLATracker, "update_with_action"), "update_with_action missing"
    assert callable(SGLATracker.update_with_action)


def test_update_with_action_accepts_tracker_action():
    import inspect
    from uav_tracker.trackers.sglatrack import SGLATracker
    sig = inspect.signature(SGLATracker.update_with_action)
    params = list(sig.parameters.keys())
    assert "action" in params, f"Expected 'action' param, got: {params}"


def test_full_action_is_default_compute():
    """ComputeAction.FULL should not warn."""
    import warnings
    from salt_r.actions import TrackerAction, ComputeAction
    # Just verify TrackerAction.compute defaults to FULL
    a = TrackerAction()
    assert a.compute == ComputeAction.FULL


def test_update_with_action_importable_without_saltr():
    """SGLATracker must be importable even if salt_r is not on PYTHONPATH."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path = [p for p in sys.path if 'saltr' not in p]; "
         "from uav_tracker.trackers.sglatrack import SGLATracker; print('ok')"],
        capture_output=True, text=True,
        cwd="/Users/voleksiuk/projects/uav-tracker-detector",
        env={**__import__("os").environ, "PYTHONPATH": "src"},
    )
    assert result.returncode == 0, f"Import failed without saltr/src: {result.stderr}"
    assert "ok" in result.stdout
