# SPDX-License-Identifier: GPL-3.0-only
"""HexSet adopted the engine's 0.13.x line as its single version
(`CHANGELOG.md`, "one version line"): the root `pyproject.toml` and
`engine/pyproject.toml` are bumped together and must always agree, and the
installed `hexset` package must report that same version.

Reads both files directly with `tomllib` rather than importing either
package, so a version drift is caught even if `importlib.metadata` is
serving stale dist-info from an old install.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import hexset

ROOT = Path(__file__).resolve().parent.parent.parent


def _version(pyproject_path: Path) -> str:
    with open(pyproject_path, "rb") as f:
        return tomllib.load(f)["project"]["version"]


def test_root_and_engine_pyproject_versions_agree():
    root_version = _version(ROOT / "pyproject.toml")
    engine_version = _version(ROOT / "engine" / "pyproject.toml")
    assert root_version == engine_version


def test_hexset_dunder_version_matches_pyproject():
    assert hexset.__version__ == _version(ROOT / "engine" / "pyproject.toml")
