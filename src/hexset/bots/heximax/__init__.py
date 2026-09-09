# SPDX-License-Identifier: GPL-3.0-only
"""Heximax: a handcrafted expectimax/max-n baseline using per-seat information.

The evaluator reads the player's own hand, public counts and public resource
ledger. Search expands opponents from sampled beliefs and averages dice,
steals and development-card draws over their distributions. The default uses
one determinization, depth two and a bounded leaf budget.

Use ``heximax(board, ...)`` to construct a bot. ``mode="honest"`` enables
trading with TRADING_WEIGHTS; ``mode="notrade"`` uses NO_TRADE_WEIGHTS and
declines trades. ``weights=`` overrides the selected profile.

Importing this package registers the heximax and heximax-notrade arena presets.
Shared evaluation terms live in hexset.bots.evaluate; search objectives live
in hexset.bots.stances, and the engine owns the information-set View.
"""

from __future__ import annotations

from hexset.view import View
from .evaluate import NO_TRADE_WEIGHTS, TRADING_WEIGHTS, HonestEvaluator, Weights
from .search import BY_MODE, DEFAULT_MAX_NODES, HEXIMAX_TRADE_FLOOR, MODES, Heximax, heximax

# Import-time side effect only: registers "heximax"/"heximax-notrade" with `hexset.arena`. See `presets`'s own docstring.
from . import presets  # noqa: F401

__all__ = [
    "BY_MODE",
    "DEFAULT_MAX_NODES",
    "HEXIMAX_TRADE_FLOOR",
    "Heximax",
    "HonestEvaluator",
    "MODES",
    "NO_TRADE_WEIGHTS",
    "TRADING_WEIGHTS",
    "View",
    "Weights",
    "heximax",
]
