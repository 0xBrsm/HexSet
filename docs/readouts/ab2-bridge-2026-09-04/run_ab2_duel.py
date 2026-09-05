# SPDX-License-Identifier: GPL-3.0-only
"""Driver for `hexset.catanatron.duel` with heximax-family DC entrants.

`hexset.catanatron.duel.main()` never imports `hexset.bots` (or
`hexset.bots.heximax`), so `DC:heximax-notrade` (and any other heximax
preset) fails inside the worker pool with `KeyError: 'heximax-notrade'` --
`search2`/`search2-notrade` are module-level entries in `hexset.arena.PRESETS`
and work with the bare CLI, but heximax's presets are registered by
`hexset.bots.heximax` at *import* time (see `hexset/arena.py`'s own comment
on this), so nothing calls that import when the CLI only ever imports
`hexset.catanatron.duel` and `hexset.arena`.

This script is *not* a change to `duel.py` -- it is a one-line-earlier
caller: import `hexset.bots` in this (parent) process before `run_duel`
creates its `multiprocessing.Pool`. `Pool`'s default start method on Linux is
fork, which copies this process's already-populated
`hexset.arena.PRESETS` into every worker, so the import only has to happen
once, here.

Also runs with `PYTHONHASHSEED` already pinned to "0" in the *shell*
environment before Python starts (see the `PYTHONHASHSEED=0` prefix in the
invocation) so `_ensure_pythonhashseed_zero` sees the pin is real (not merely
the string "0" in `os.environ`, but the interpreter's actual hash seed) and
skips its re-exec -- the re-exec path is for a plain `python -m
hexset.catanatron.duel` invocation with no pin set, and would otherwise blow
away the `hexset.bots` import done here by replacing the process before
`run_duel` runs.
"""

from __future__ import annotations

import argparse
import os
import sys

assert os.environ.get("PYTHONHASHSEED") == "0", (
    "run this script with PYTHONHASHSEED=0 already set in the environment "
    "(e.g. `PYTHONHASHSEED=0 python run_ab2_duel.py ...`) -- it must be pinned "
    "before the interpreter starts, exactly as hexset.catanatron.duel's own "
    "docstring requires."
)

import hexset.bots  # noqa: F401  -- registers heximax's arena presets

from hexset.catanatron.duel import run_duel, provenance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", required=True)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run_duel(args.players, args.num, args.workers, args.seed)
    print(result.report())
    print("hexset.bots pre-imported by run_ab2_duel.py (see module docstring)")


if __name__ == "__main__":
    main()
