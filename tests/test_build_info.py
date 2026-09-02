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
