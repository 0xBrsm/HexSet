from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path

# `hexset-ui`'s version is bumped together with the engine's
# (`engine/pyproject.toml`) -- see `engine/tests/test_versions.py`, which
# fails the moment the two drift.
try:
    __version__ = metadata.version("hexset-ui")
except metadata.PackageNotFoundError:
    try:
        with open(Path(__file__).resolve().parent.parent.parent / "pyproject.toml", "rb") as f:
            __version__ = tomllib.load(f)["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        __version__ = "0+unknown"
