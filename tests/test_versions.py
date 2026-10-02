# SPDX-License-Identifier: GPL-3.0-only
"""One distribution, one `pyproject.toml`, one version. Read with `tomllib`
rather than by import, so stale dist-info cannot mask a drift.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import hexset

ROOT = Path(__file__).resolve().parent.parent


def _version(pyproject_path: Path) -> str:
    with open(pyproject_path, "rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_hexset_dunder_version_matches_pyproject():
    assert hexset.__version__ == _version(ROOT / "pyproject.toml")
