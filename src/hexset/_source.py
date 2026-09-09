# SPDX-License-Identifier: GPL-3.0-only
"""Source-checkout identification shared by build and experiment metadata."""
from pathlib import Path
import subprocess


def git_value(source_file: str, *args: str) -> str | None:
    source = Path(source_file).resolve()
    root = source.parents[2]
    # A wheel inside another project's .venv must not claim that project's SHA.
    if not (root / ".git").exists() or (root / "src" / "hexset" / source.name).resolve() != source:
        return None
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True,
            text=True, check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
