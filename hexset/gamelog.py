# SPDX-License-Identifier: GPL-3.0-only
"""A long run's journal: one JSON line per finished unit of work, written the
moment the unit finishes, so a run that dies keeps everything it did and a
rerun picks up where it stopped.

Each line is appended, flushed and fsynced, then the file is closed -- a
killed process, a dead container or a disk that drops out from under the
page cache loses at most the line being written. A reader stops at a torn
last line and keeps everything before it; a torn line anywhere else means
the file was edited or corrupted, and is refused.

`hexset.arena.compete(journal=...)` writes one of these per game. Anything
else that runs a batch -- replays, fits, sweeps -- can use `append` and
`read_lines` the same way.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = [
    "JOURNAL_VERSION",
    "append",
    "read_lines",
    "cut_torn_tail",
]


#: Bumped whenever a journal's line format changes incompatibly.
JOURNAL_VERSION = 1


def append(path: str | os.PathLike, entry: dict[str, Any]) -> None:
    """Append `entry` as one JSON line, durably: written, flushed and fsynced
    before this returns."""
    line = json.dumps(entry, separators=(",", ":"), allow_nan=False) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_lines(path: str | os.PathLike) -> list[dict[str, Any]]:
    """Every complete line of the journal at `path`, in file order. A torn
    last line -- the one a dying writer was part-way through -- is dropped;
    a line that does not parse anywhere before it raises `ValueError`."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.split("\n")
    # A file that ends in a newline splits into a trailing empty string.
    torn = lines.pop() if lines else ""
    entries = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number} is not a journal line: {error}") from error
    if torn.strip():
        try:
            entries.append(json.loads(torn))
        except json.JSONDecodeError:
            pass  # the line a dying writer did not finish
    return entries


def cut_torn_tail(path: str | os.PathLike) -> None:
    """Drop a torn last line from `path` in place, so the next `append`
    starts a line of its own rather than finishing a dead writer's."""
    with open(path, "rb+") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if not size:
            return
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return
        handle.seek(0)
        data = handle.read()
        handle.truncate(data.rfind(b"\n") + 1)
        handle.flush()
        os.fsync(handle.fileno())
