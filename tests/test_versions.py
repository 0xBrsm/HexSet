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


def test_a_pyproject_beside_the_install_that_is_not_hexsets_is_not_read(tmp_path):
    """An installed package can sit beside another project's
    `pyproject.toml` (say, `site-packages` inside a checkout): only a file
    naming this distribution is the source tree's."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "other"\nversion = "9.9.9"\n')
    assert hexset._source_version(tmp_path) is None
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "hexset"\nversion = "9.9.9"\n')
    assert hexset._source_version(tmp_path) == "9.9.9"
    (tmp_path / "pyproject.toml").write_text("not = [toml")
    assert hexset._source_version(tmp_path) is None
    assert hexset._source_version(tmp_path / "missing") is None
