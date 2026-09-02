# SPDX-License-Identifier: GPL-3.0-only
"""Graph-native Settlers of Catan engine, bots and ledger."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

__version__ = "0.1.0"


def build_info() -> dict[str, Any]:
    """`{"version": ..., "git_commit": ...}` for a consumer's provenance record.

    `git_commit` is the commit this package's own files are checked out at,
    read from whatever git repo contains this file -- None if that fails (a
    wheel install with no `.git` directory, or `git` unavailable), because a
    provenance field that is sometimes wrong is worse than one that is
    sometimes absent. Consumers (e.g. HexNet's `hexnet.run.manifest`) stamp
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
