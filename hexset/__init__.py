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
# silently. Only this distribution's own file counts -- an installed wheel
# ships none, and whatever `pyproject.toml` sits beside the install directory
# is somebody else's -- so anything else falls back to installed metadata.
def _source_version(root: Path) -> str | None:
    """The version `root/pyproject.toml` declares, if it is hexset's own."""
    try:
        with open(root / "pyproject.toml", "rb") as f:
            project = tomllib.load(f).get("project", {})
    except (OSError, tomllib.TOMLDecodeError):
        return None
    if project.get("name") != "hexset":
        return None
    version = project.get("version")
    return version if isinstance(version, str) else None


__version__ = _source_version(Path(__file__).resolve().parent.parent)
if __version__ is None:
    try:
        __version__ = metadata.version("hexset")
    except metadata.PackageNotFoundError:
        __version__ = "0+unknown"

