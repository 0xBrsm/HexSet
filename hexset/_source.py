# SPDX-License-Identifier: GPL-3.0-only
"""Which source checkout this package runs from, for experiment metadata
(`hexset.experiment.provenance`) and run journals (`hexset.arena`). Internal."""
from pathlib import Path
import subprocess
import tomllib


def _source_tree(package_parent: Path) -> bool:
    """Whether `package_parent` is a hexset source tree: the directory holding
    the `hexset` package also holds hexset's own `pyproject.toml`. An installed
    wheel's directory (`site-packages`) holds none."""
    try:
        with open(package_parent / "pyproject.toml", "rb") as f:
            return tomllib.load(f).get("project", {}).get("name") == "hexset"
    except (OSError, tomllib.TOMLDecodeError):
        return False


def git_value(source_file: str, *args: str) -> str | None:
    """`git <args>` in the checkout `source_file` runs from, or None when it
    does not run from one (an installed package) or git cannot answer."""
    source = Path(source_file).resolve()
    tree = source.parent.parent
    # Recognise the source tree by its own pyproject, wherever it sits in the
    # checkout -- at the root, under `src/`, or deeper. Counting parents or
    # listing layouts reads a new layout as "installed" and silently drops the
    # SHA from every record. A wheel inside another project's .venv has no
    # hexset pyproject beside it, so it never claims that project's SHA.
    if not _source_tree(tree):
        return None
    root = next((p for p in (tree, *tree.parents) if (p / ".git").exists()), None)
    if root is None:
        return None
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True,
            text=True, check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def source(source_file: str) -> dict:
    """The checkout `source_file` runs from: its commit, and whether tracked
    files differ from it (`dirty`). Both None for an installed package, whose
    version identifies it."""
    commit = git_value(source_file, "rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "dirty": None}
    changes = git_value(source_file, "status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": None if changes is None else bool(changes)}
