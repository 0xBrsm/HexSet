# SPDX-License-Identifier: GPL-3.0-only
"""Register the single adaptive Heximax entrant."""
from __future__ import annotations

import random

from hexset.arena import Entrant, register_entrant_kind, register_preset
from hexset.board.board import Board
from .search import Heximax, heximax


def _spawn(entrant: Entrant, board: Board, rng: random.Random) -> Heximax:
    kwargs = dict(
        depth=entrant.depth, width=entrant.width, max_trades=entrant.max_trades,
        k=entrant.k, weights=entrant.weights, temperature=entrant.temperature,
        pin_weights=entrant.pin_weights, expansion_value=entrant.expansion_value,
    )
    if entrant.stance is not None:
        kwargs["stance"] = entrant.stance
    return heximax(board, rng, **kwargs)


register_entrant_kind("heximax", _spawn)
register_preset("heximax", Entrant("heximax", kind="heximax", depth=2, width=6))
