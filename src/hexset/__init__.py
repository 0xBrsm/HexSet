# SPDX-License-Identifier: GPL-3.0-only
"""Graph-native Settlers of Catan engine, bots and ledger."""

from __future__ import annotations

import subprocess
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any

# `hexset`, `heximax`, `hexset.bench`, `hexset.server` and `hexset.clients`
# are all one distribution now (`../../pyproject.toml`) -- see
# `tests/test_versions.py`. Read `pyproject.toml` from the source tree first:
# it's the one file a checkout always has current, whereas an editable
# install's dist-info metadata is only regenerated on reinstall and silently
# goes stale otherwise (this is what made the dunder read "0.26.0" for four
# releases while the tree had moved on). Fall back to installed metadata
# only when there's no source tree to read (a real wheel install, which
# ships no `pyproject.toml`).
try:
    with open(Path(__file__).resolve().parent.parent.parent / "pyproject.toml", "rb") as f:
        __version__ = tomllib.load(f)["project"]["version"]
except (OSError, KeyError, tomllib.TOMLDecodeError):
    try:
        __version__ = metadata.version("hexset")
    except metadata.PackageNotFoundError:
        __version__ = "0+unknown"


def build_info() -> dict[str, Any]:
    """`{"version": ..., "git_commit": ...}` for a consumer's provenance record.

    `git_commit` is the commit this package's own files are checked out at,
    read from whatever git repo contains this file -- None if that fails (a
    wheel install with no `.git` directory, or `git` unavailable), because a
    provenance field that is sometimes wrong is worse than one that is
    sometimes absent. Consumers (e.g. HexN's `hexn.run.manifest`) stamp
    this into their own run records rather than reproducing the git call
    themselves, so there is exactly one place that knows how to ask.
    """
    commit: str | None = None
    try:
        out = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode == 0:
            commit = out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"version": __version__, "git_commit": commit}
