"""Assert that salt_runner.py does not import or use TSA in production control."""
import ast
import pathlib


def _load_runner_ast():
    path = pathlib.Path("src/uav_tracker/salt_runner.py")
    return ast.parse(path.read_text()), path.read_text()


def test_no_tsa_import_in_runner():
    tree, src = _load_runner_ast()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names]
            for n in [mod] + names:
                if n is None:
                    continue
                assert "tsa" not in n.lower(), f"TSA import found: {n}"
                assert "target_state" not in n.lower(), f"TargetState import: {n}"
                assert "saltrd_adapter" not in n.lower(), f"adapter import: {n}"


def test_no_targetstate_usage_in_runner():
    _, src = _load_runner_ast()
    # These are the forbidden runtime-control patterns from the plan
    forbidden = [
        "TargetStateAssessor",
        "TargetState.",           # enum access
        "from uav_tracker.ml.tsa",
        "consecutive_lost",
        "consecutive_occluded",
        "update_with_state(",
    ]
    for pattern in forbidden:
        assert pattern not in src, f"Forbidden pattern still in salt_runner.py: {pattern!r}"


def test_update_with_action_called_in_runner():
    _, src = _load_runner_ast()
    assert "update_with_action" in src, "Runner must call update_with_action"


def test_saltrd_telemetry_fields_present():
    _, src = _load_runner_ast()
    for field in ["saltrd_action_compute", "saltrd_action_recovery", "saltrd_changed_bbox"]:
        assert field in src, f"Expected telemetry field missing: {field}"
