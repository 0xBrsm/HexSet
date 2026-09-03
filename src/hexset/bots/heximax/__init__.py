# SPDX-License-Identifier: GPL-3.0-only
"""heximax: the handcrafted baseline that does not read the opponents' hands.

Lives at `hexset.bots.heximax`, a subpackage of `hexset.bots` -- the package
that holds every heuristic bot (`hexset.bots.search2` is the other one) so
they can share code, chiefly the handcrafted evaluation at
`hexset.bots.evaluate`.

`hexset.bots.search2.SearchBot` over `hexset.bots.evaluate.Evaluator` --
`search2` -- is the project's one clean held-out referent, and it cheats: its
evaluation reads every seat's true hand, its tree expands opponents from
their true hands and development cards, and a steal or a dev-card buy is
valued on one frozen draw. heximax is the next generation of that bot, built
to the design in `agents/reference/heximax.md`. It is **information-set
honest by default**: every quantity about an opponent is read through the
public ledger (`game.ledger`, `known[s]` + `unknown`) and the public counts,
never through `state.hands[opponent]` or `state.dev_cards[opponent]`. Its own
hand is exact. An `omniscient` mode keeps the old reading, so the price of
honesty can be measured rather than assumed.

A package, so a downstream copy takes the directory: `heximax/evaluate.py`,
`heximax/search.py`, `heximax/presets.py`, one concern a file. The
information set itself, `View`, lives in the engine at `hexset.view` -- a
seat's view of the game is engine functionality, reached through
`Game.state(seat, hidden=True)`, not something a bot builds for itself.

* `evaluate` -- `hexset.bots.evaluate.Evaluator`'s term set read through the
  view (`HonestEvaluator`), with progress zeroed where the piece supply is
  exhausted, and two weight profiles (`TRADING_WEIGHTS`, a trading table,
  and `NO_TRADE_WEIGHTS`, a no-trade table). Reaches the shared evaluator by
  `from ..evaluate import ...` (the sibling `hexset.bots.evaluate`), not
  through the `hexset.evaluate` compat shim -- see that shim's own docstring
  for why the distinction matters here specifically. Duplicates rather than
  imports `hexset.bots.evaluate`'s term functions (see the duplication note
  in `evaluate.py`'s own module docstring); not merged, on purpose, for now.
* `search`   -- max^n over `HonestEvaluator` with a node budget and
  iterative deepening (`Heximax`), opponents expanded from determinized
  samples of the belief (PIMC over `k` worlds), and every hidden draw
  averaged over its distribution. See its module docstring for the leaf
  budget's cost accounting.
* `presets`  -- registers "heximax"/"heximax-omni"/"heximax-notrade" with
  `hexset.arena` and "heximax-trading"/"heximax-notrade" with
  `hexset.tuning`, as an import-time side effect of importing this package
  (and therefore of `import hexset.bots`, which imports this package).

Trading is not an action and needs no adapter: `Heximax.valuation` publishes
`tanh(marginal / MARGINAL_SCALE)` and `Heximax.accepts` gates on a strictly
positive change in its own evaluation, and the engine's one trade event a
turn does the rest (`hexset.trading`). `max_trades=0` publishes nothing and
refuses everything.

Instantiate through the `heximax(board, ...)` factory, not `Heximax(...)`
directly, unless you are building a custom evaluator or weight vector by
hand: `heximax(board, mode="honest")` is the shipped bot, `mode="omniscient"`
reads every true hand at the same weights (the honest/omni dial this module
exists to measure the cost of), and `mode="notrade"` plays the no-trade
weight table with trading switched off. `weights=` overrides either mode's
profile with a candidate vector while leaving the mode's other defaults (the
trade switch, `omniscient`) unchanged -- the hook `hexset.tuning` fits
through. Importing this package, or `hexset.bots` (which imports it), or
anything that imports either, registers the three presets above with
`hexset.arena` and the two evaluator names above with `hexset.tuning`; a
process that never imports `hexset.bots` (or the deprecated `heximax` shim)
cannot spawn or fit heximax by name.

Engine dependency: this package is a consumer of `hexset` (actions, board,
cards, economy, game, ledger, mcts, placement, robber, state, trading,
victory, view), of the sibling `hexset.bots.search2`/`hexset.bots.evaluate` (the
shared STANCES and the shared evaluation), and of `hexset.arena`/
`hexset.tuning` for registration. It does not modify any of them. Nesting
under `hexset.bots` means `hexset.bots`, `hexset.arena` and `hexset.mcts` now
have a real import cycle through this package (`hexset.bots` -> `heximax` ->
`hexset.arena`/`hexset.mcts` -> `hexset.bots`, for the names those two
modules borrow back); it resolves because `hexset.arena`'s and
`hexset.mcts`'s own uses of `hexset.bots` names are deferred to inside the
functions/methods that need them rather than sitting at module-import time --
see the comment at each of those two import sites.

`bot.choose()`'s own choices, and the leaves it spends reaching them, are
checked on every position by
`test_choices_are_byte_identical_to_the_recorded_census`. History -- the
optimization and structural passes, with their per-change breakdowns -- is
in `agents/reference/heximax.md`.
"""

from __future__ import annotations

from hexset.view import View
from .evaluate import NO_TRADE_WEIGHTS, TRADING_WEIGHTS, HonestEvaluator, Weights
from .search import BY_MODE, DEFAULT_MAX_NODES, MODES, Heximax, heximax

# Import-time side effect only: registers "heximax"/"heximax-omni"/
# "heximax-notrade" with `hexset.arena` and "heximax-trading"/
# "heximax-notrade" with `hexset.tuning`. See `presets`'s own docstring.
from . import presets  # noqa: F401

__all__ = [
    "BY_MODE",
    "DEFAULT_MAX_NODES",
    "Heximax",
    "HonestEvaluator",
    "MODES",
    "NO_TRADE_WEIGHTS",
    "TRADING_WEIGHTS",
    "View",
    "Weights",
    "heximax",
]
