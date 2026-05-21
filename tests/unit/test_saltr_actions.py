import json
import pytest
from salt_r.actions import (
    ComputeAction, SearchAction, TemplateAction, RecoveryAction,
    TrackerAction, BBox,
)


def test_default_action():
    a = TrackerAction()
    assert a.compute == ComputeAction.FULL
    assert a.search == SearchAction.KEEP
    assert a.template == TemplateAction.KEEP_CURRENT
    assert a.recovery == RecoveryAction.NONE
    assert a.bbox_hint is None
    assert a.detector_hint is None


def test_frozen():
    a = TrackerAction()
    with pytest.raises((AttributeError, TypeError)):
        a.compute = ComputeAction.PRUNE_LIGHT  # type: ignore


def test_serialization_round_trip():
    a = TrackerAction(
        compute=ComputeAction.PRUNE_LIGHT,
        search=SearchAction.CENTER_ON_REINIT_HINT,
        template=TemplateAction.BLOCK_UPDATE,
        recovery=RecoveryAction.REINIT,
        bbox_hint=(10.0, 20.0, 30.0, 40.0),
        detector_hint=(5.0, 6.0, 7.0, 8.0),
    )
    j = a.to_json()
    b = TrackerAction.from_json(j)
    assert a == b


def test_json_is_serializable():
    a = TrackerAction(recovery=RecoveryAction.REINIT)
    s = json.dumps(a.to_json())
    b = TrackerAction.from_json(json.loads(s))
    assert a == b


def test_invalid_enum_raises():
    with pytest.raises(ValueError):
        ComputeAction("invalid_value")


def test_no_tsa_imports():
    import importlib
    import ast
    import inspect
    import salt_r.actions as m
    src = inspect.getsource(m)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                assert "tsa" not in (name or "").lower(), f"TSA import found: {name}"
                assert "target_state" not in (name or "").lower(), f"TargetState import found: {name}"
