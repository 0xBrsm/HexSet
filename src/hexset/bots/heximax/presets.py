# SPDX-License-Identifier: GPL-3.0-only
"""Preset registration: importing `heximax` makes it spawnable and fittable.

`hexset.arena` and `hexset.tuning` know heximax only by name, the same way
they know the network-backed kinds `hexnet.netbot` provides -- neither
imports this package, so importing `heximax` (directly, or via anything that
does: `hexset.bench.duel`, `hexset.server`) is what makes the "heximax" entrant
kind spawnable and the "heximax-trading"/"heximax-notrade" evaluator names
fittable. A process that never imports `heximax` gets a plain
"unknown"/`KeyError` on the name rather than this module's numpy and
`hexset.mcts` imports forced on it.
"""

from __future__ import annotations

import random

from hexset.arena import Entrant, register_entrant_kind, register_preset
from hexset.board.board import Board
from hexset.tuning import register_heximax_evaluator

from .evaluate import NO_TRADE_WEIGHTS, TRADING_WEIGHTS
from .search import Heximax, heximax


def _spawn(entrant: Entrant, board: Board, rng: random.Random) -> Heximax:
    return heximax(
        board,
        rng,
        mode=entrant.mode,
        depth=entrant.depth,
        width=entrant.width,
        max_offers=entrant.max_offers,
        stance=entrant.stance,
        k=entrant.k,
        weights=entrant.weights,
        accept_margin=entrant.accept_margin,
        propose_margin=entrant.propose_margin,
    )


register_entrant_kind("heximax", _spawn)

# The honest handcrafted baseline (design note `heximax.md` §5). The
# placement prior is composed into the bot rather than wrapped around it, so
# `placement` stays False here and `spawn` returns the bot itself.
# `heximax-omni` is the same bot reading every true hand, kept to measure
# what honesty costs; `heximax-notrade` plays the no-trade table at an offer
# budget of zero.
register_preset("heximax", Entrant("heximax", kind="heximax", depth=2, width=6, max_offers=3))
register_preset(
    "heximax-omni",
    Entrant("heximax-omni", kind="heximax", depth=2, width=6, max_offers=3, mode="omniscient"),
)
register_preset(
    "heximax-notrade",
    Entrant("heximax-notrade", kind="heximax", depth=2, width=6, max_offers=0, mode="notrade"),
)

# `hexset.tuning.entrant_for(evaluator="heximax-trading"/"heximax-notrade")`
# climbs from these starting points -- `heximax-trading` shares
# `evaluate.Weights` with "default" (`HonestEvaluator` wraps the same
# `Evaluator`) but starts from heximax's own `TRADING_WEIGHTS` rather than
# the bare default, and `heximax-notrade` starts from `NO_TRADE_WEIGHTS`.
register_heximax_evaluator("heximax-trading", "honest", lambda: TRADING_WEIGHTS)
register_heximax_evaluator("heximax-notrade", "notrade", lambda: NO_TRADE_WEIGHTS)
