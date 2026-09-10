# SPDX-License-Identifier: GPL-3.0-only
"""Graph-native Settlers of Catan engine, bots and ledger."""

from __future__ import annotations

import tomllib
from importlib import metadata
from pathlib import Path

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


# `build_info()` was here: `{"version", "git_commit"}`, served by
# `GET /api/version` and stamped into a consumer's run records. It is gone.
# The commit only ever mattered as research provenance -- identifying the code
# that produced a benchmark number -- and that belongs to the run document, not
# to a package-level accessor and not to a server endpoint. It lives in
# `hexset.experiment.provenance()`, alongside the rest of the fingerprint a
# result actually needs: the dirty flag (a commit alone cannot reproduce a run
# off a modified tree), dependency versions and their VCS revisions, the
# determinism-relevant environment, and checkpoint hashes.
#
# A consumer that stamped `build_info()` into its own records (HexN's
# `hexn.run.manifest`) reads `hexset.experiment.provenance()` instead, and gets
# strictly more than it had. `GET /api/version` now answers with the API's own
# contract version (`hexset.server.api.API_VERSION`), which is what that route
# always should have meant.
