# SPDX-License-Identifier: GPL-3.0-only
"""A Catan rules engine: board, rules, trading, information sets, records,
search, evaluation, training environments and the browser/HTTP/MCP server."""

from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path

# `hexset` and its subpackages are one distribution (`../pyproject.toml`).
# Read `pyproject.toml` from the source tree first: an editable install's
# dist-info metadata is only regenerated on reinstall and goes stale
# silently. Fall back to installed metadata only for a wheel, which ships no
# `pyproject.toml`.
try:
    with open(Path(__file__).resolve().parent.parent / "pyproject.toml", "rb") as f:
        __version__ = tomllib.load(f)["project"]["version"]
except (OSError, KeyError, tomllib.TOMLDecodeError):
    try:
        __version__ = metadata.version("hexset")
    except metadata.PackageNotFoundError:
        __version__ = "0+unknown"

