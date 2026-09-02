# SPDX-License-Identifier: GPL-3.0-only
"""Every heuristic bot lives here, so they can share code.

`search2` (`hexset.bots.search2`: `SearchBot`, `greedy`, `RandomBot`, the
`STANCES` a per-seat vector is read through) and `heximax`
(`hexset.bots.heximax`, split by concern into `belief`/`evaluate`/`search`/
`trade`/`presets`) both build on the same handcrafted evaluation,
`hexset.bots.evaluate` -- the old `hexset/evaluate.py`, moved here alongside
its consumers rather than left a level up from all of them. A one-line shim
at `hexset.evaluate` re-exports it for existing callers (`hexset.tuning`,
`hexset.fitting`, `hexset.dataset`, HexNet); a deprecated `heximax` top-level
package re-exports `hexset.bots.heximax` the same way, for `import heximax`.

This module re-exports `search2`'s public names (so `from hexset.bots import
SearchBot` keeps working exactly as it did when `bots.py` was a single file)
and most of `heximax`'s (`Heximax`, `Belief`, `HonestEvaluator`, `Weights`,
`TRADING_WEIGHTS`, `NO_TRADE_WEIGHTS`, `MODES`, `BY_MODE`). Importing
`.heximax` here is what makes `import hexset.bots` register the "heximax"/
"heximax-omni"/"heximax-notrade" presets and the "heximax-trading"/
"heximax-notrade" evaluator names -- previously only an explicit `import
heximax` did that; now any consumer of this package's bots gets it too, since
the two live in the same package. See `heximax`'s own module docstring for
the import-cycle this creates with `hexset.arena`/`hexset.mcts`, and how it
resolves.

**Rule for this file: no re-exported name may equal a submodule's name.**
`hexset.bots.heximax`, imported as a side effect above, is the submodule
(the package at `hexset/bots/heximax/`) for as long as nothing here rebinds
that attribute. An earlier version of this file did -- re-exporting the
`heximax(...)` factory function under its own name shadowed the submodule,
so `hexset.bots.heximax` (and `import hexset.bots.heximax as x`) meant
whichever one was bound *last*, depending on import order: a real
public-API bug, not merely confusing. The factory is deliberately left
un-re-exported here for that reason; reach it at its home,
`from hexset.bots.heximax import heximax`, the same way `search2`'s and
`evaluate`'s own names are reached at `hexset.bots.search2`/
`hexset.bots.evaluate` rather than through this package (neither module
name collides with a re-export here either, and this file should stay that
way as it grows).
"""

from __future__ import annotations

from .search2 import (
    Bot,
    RandomBot,
    SearchBot,
    STANCES,
    greedy,
    options_for,
    own,
    paranoid,
    relative,
)
from .heximax import (
    BY_MODE,
    MODES,
    Belief,
    Heximax,
    HonestEvaluator,
    NO_TRADE_WEIGHTS,
    TRADING_WEIGHTS,
    View,
    Weights,
)

__all__ = [
    "BY_MODE",
    "Belief",
    "Bot",
    "Heximax",
    "HonestEvaluator",
    "MODES",
    "NO_TRADE_WEIGHTS",
    "RandomBot",
    "SearchBot",
    "STANCES",
    "TRADING_WEIGHTS",
    "View",
    "Weights",
    "greedy",
    "options_for",
    "own",
    "paranoid",
    "relative",
]
