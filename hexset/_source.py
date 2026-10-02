# SPDX-License-Identifier: GPL-3.0-only
"""Which source checkout this package runs from, for experiment metadata
(`hexset.experiment.provenance`). Internal."""
from pathlib import Path
import subprocess


def git_value(source_file: str, *args: str) -> str | None:
    source = Path(source_file).resolve()
    # `hexset` sits at the root of the published tree and under `src/` in the
    # development one, so find the checkout rather than counting a fixed number
    # of parents: a wrong guess reads as "installed" and silently drops the SHA
    # from every record written out of a source checkout.
    root = next((p for p in source.parents if (p / ".git").exists()), None)
    # A wheel inside another project's .venv must not claim that project's SHA.
    if root is None:
        return None
    here = {(root / prefix / "hexset" / source.name) for prefix in (".", "src")}
    if not any(c.resolve() == source for c in here):
        return None
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True,
            text=True, check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
