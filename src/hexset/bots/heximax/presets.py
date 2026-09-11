# SPDX-License-Identifier: GPL-3.0-only
"""Preset registration: importing `heximax` makes it spawnable.

`hexset.arena` knows heximax only by name, the same way it knows the
network-backed kinds `hexn.netbot` provides -- it does not import this
package, so importing `heximax` (directly, or via anything that does:
`hexset.bench.duel`, `hexset.server`) is what makes the "heximax" entrant
kind spawnable. A process that never imports `heximax` gets a plain
"unknown"/`KeyError` on the name rather than this module's numpy and
`hexset.mcts` imports forced on it.
"""

from __future__ import annotations

import random

from hexset.arena import Entrant, register_entrant_kind, register_preset
from hexset.board.board import Board

from .search import DEFAULT_MAX_NODES, Heximax, heximax


def _spawn(entrant: Entrant, board: Board, rng: random.Random) -> Heximax:
    # `entrant.stance` is `None` unless a caller asked for something else --
    # `hexset.arena.Entrant`'s own field default -- so `stance` is left out of
    # this call entirely in that case, and `heximax()`'s own default (`"win"`)
    # applies. That is the bot stating its own default once, rather than this
    # module restating it (see the presets below, which used to pass
    # `stance="win"` explicitly for exactly that reason).
    kwargs: dict = dict(
        mode=entrant.mode,
        depth=entrant.depth,
        width=entrant.width,
        max_nodes=entrant.max_nodes if entrant.max_nodes is not None else DEFAULT_MAX_NODES,
        max_trades=entrant.max_trades,
        k=entrant.k,
        weights=entrant.weights,
        temperature=entrant.temperature,
        native_action_compat=entrant.native_action_compat,
    )
    if entrant.stance is not None:
        kwargs["stance"] = entrant.stance
    return heximax(board, rng, **kwargs)


register_entrant_kind("heximax", _spawn)

# The honest handcrafted baseline (design note `heximax.md` §5). The
# placement prior is composed into the bot rather than wrapped around it, so
# `placement` stays False here and `spawn` returns the bot itself.
# `heximax-notrade` plays the no-trade table with the trade switch off.
#
# Neither passes `stance` -- `Entrant`'s own field default is
# `None`, which `_spawn` above leaves out of the `heximax()` call, so
# `heximax()`'s own default (`"win"`) applies. `agents/reference/heximax.md`,
# "Registration 2026-09-04: the objective -- a win-probability stance against
# the relative-VP stance" and its post-data note, ratifying `win` as
# heximax's default; stated once, on the bot, rather than on every preset
# that spawns it.
register_preset("heximax", Entrant("heximax", kind="heximax", depth=2, width=6))
register_preset(
    "heximax-notrade",
    Entrant(
        "heximax-notrade", kind="heximax", depth=2, width=6, max_trades=0, mode="notrade",
    ),
)
