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
and `heximax`'s (`Heximax`, `heximax`, `Belief`, `HonestEvaluator`, `Weights`,
`TRADING_WEIGHTS`, `NO_TRADE_WEIGHTS`, `MODES`, `BY_MODE`). Importing
`heximax` here is what makes `import hexset.bots` register the "heximax"/
"heximax-omni"/"heximax-notrade" presets and the "heximax-trading"/
"heximax-notrade" evaluator names -- previously only an explicit `import
heximax` did that; now any consumer of this package's bots gets it too, since
the two live in the same package. See `heximax`'s own module docstring for
the import-cycle this creates with `hexset.arena`/`hexset.mcts`, and how it
resolves.

One name collides on purpose: this file's own `from .heximax import (...,
heximax, ...)` line rebinds the attribute `hexset.bots.heximax` from the
submodule (the package at `hexset/bots/heximax/`, which Python bound there
automatically while resolving that same import) to the re-exported *factory
function* -- the explicit `from ... import heximax` runs last and wins.
Neither `import hexset.bots.heximax as x` nor `from hexset.bots import
heximax` reaches the submodule after that, because both resolve through
this same (now-rebound) attribute; only `sys.modules["hexset.bots.heximax"]`
still holds it, e.g. via `importlib.import_module("hexset.bots.heximax")`.
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
    Weights,
    heximax,
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
    "Weights",
    "greedy",
    "heximax",
    "options_for",
    "own",
    "paranoid",
    "relative",
]
