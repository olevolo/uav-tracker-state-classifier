"""Unit tests for the plugin ``Registry``.

The ``Registry[T]`` generic lives in Architect's ``src/uav_tracker/registry.py``.
We test its externally-visible contract per PLAN §4.1:

    * ``register("name")(cls)`` stores the class.
    * Duplicate registrations raise ``ValueError``.
    * Unknown names raise ``KeyError``.
    * ``names()`` returns a list-like of registered strings.
    * ``build(name, **kwargs)`` forwards kwargs to the stored class.
"""

from __future__ import annotations

import pytest

from uav_tracker.registry import Registry


class _Dummy:
    """Trivial target for registration — captures kwargs for asserts."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def test_register_then_build() -> None:
    reg: Registry[_Dummy] = Registry("dummy")
    reg.register("first")(_Dummy)
    instance = reg.build("first", a=1, b="two")
    assert isinstance(instance, _Dummy)
    assert instance.kwargs == {"a": 1, "b": "two"}


def test_register_twice_raises() -> None:
    reg: Registry[_Dummy] = Registry("dummy")
    reg.register("shared")(_Dummy)
    with pytest.raises(ValueError):
        reg.register("shared")(_Dummy)


def test_build_unknown_raises_key_error() -> None:
    reg: Registry[_Dummy] = Registry("dummy")
    with pytest.raises(KeyError):
        reg.build("missing")


def test_names_reports_registered() -> None:
    reg: Registry[_Dummy] = Registry("dummy")
    reg.register("a")(_Dummy)
    reg.register("b")(_Dummy)
    names = list(reg.names())
    assert set(names) == {"a", "b"}
