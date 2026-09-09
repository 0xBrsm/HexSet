# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.build_info`: the provenance stamp a consumer records about us.

Only the shape is pinned here -- `git_commit` legitimately varies (None off a
wheel install, a real SHA in a checkout), so the test asserts what a consumer
like `hexnet.run.manifest` actually relies on: the key exists and, when
present, is a string.
"""

from __future__ import annotations

import hexset


def test_build_info_has_version_matching_the_package():
    info = hexset.build_info()
    assert info["version"] == hexset.__version__


def test_build_info_git_commit_is_a_string_or_absent():
    commit = hexset.build_info()["git_commit"]
    assert commit is None or isinstance(commit, str)


def test_installed_build_info_does_not_claim_an_enclosing_repository(tmp_path, monkeypatch):
    from hexset import _source
    installed = tmp_path / ".venv" / "lib" / "python3.11" / "site-packages" / "hexset" / "__init__.py"
    monkeypatch.setattr(hexset, "__file__", str(installed))
    def unexpected_git(*args, **kwargs):
        raise AssertionError("installed package must not query ancestor Git")
    monkeypatch.setattr(_source.subprocess, "run", unexpected_git)
    assert hexset.build_info()["git_commit"] is None
