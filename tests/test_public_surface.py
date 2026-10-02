"""HexSet's public surface: every module with no leading underscore in its
path, as far as its `__all__` lists. Everything else -- an underscore module,
or a name a public module does not list -- is internal and may change in any
release."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil

import pytest

import hexset

# The extras a public module may need: without one it is skipped, not failed.
OPTIONAL = {"catanatron", "gymnasium", "onnx", "onnxruntime", "pettingzoo"}


def _public_modules() -> list[str]:
    names = ["hexset"]
    for info in pkgutil.walk_packages(hexset.__path__, "hexset.", onerror=lambda name: None):
        if not any(part.startswith("_") for part in info.name.split(".")[1:]):
            names.append(info.name)
    return sorted(names)


def _load(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as error:
        if error.name and error.name.split(".")[0] in OPTIONAL:
            pytest.skip(f"{name} needs the {error.name.split('.')[0]} extra")
        raise


def _defined(module) -> list[str]:
    """The functions and classes `module` itself defines without an underscore."""
    return [
        name for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isfunction(value) or inspect.isclass(value))
        and getattr(value, "__module__", None) == module.__name__
    ]


@pytest.mark.parametrize("name", _public_modules())
def test_a_public_module_lists_its_surface(name):
    module = _load(name)
    if not hasattr(module, "__all__"):
        assert not _defined(module), f"{name} defines {_defined(module)} but lists no __all__"
        return
    for exported in module.__all__:
        assert not exported.startswith("_"), f"{name} exports the internal name {exported}"
        assert hasattr(module, exported), f"{name}.__all__ lists {exported}, which it does not have"
    assert len(set(module.__all__)) == len(module.__all__), f"{name}.__all__ repeats a name"


def _own_doc(value) -> str | None:
    """`value`'s own docstring: a decorated function's, read through the
    wrapper; a class's from the class itself, not one inherited from a base
    class or the signature `dataclass` writes for a class that has none."""
    value = inspect.unwrap(value)
    if not inspect.isclass(value):
        return value.__doc__
    doc = value.__dict__.get("__doc__")
    if doc and dataclasses.is_dataclass(value) and doc.startswith(f"{value.__name__}("):
        return None
    return doc


@pytest.mark.parametrize("name", _public_modules())
def test_every_public_function_and_class_is_documented(name):
    module = _load(name)
    undocumented = [
        exported for exported in getattr(module, "__all__", [])
        if (inspect.isroutine(inspect.unwrap(getattr(module, exported)))
            or inspect.isclass(getattr(module, exported)))
        and not (_own_doc(getattr(module, exported)) or "").strip()
    ]
    assert not undocumented, f"{name} has no docstring for {undocumented}"
