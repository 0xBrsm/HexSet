# SPDX-License-Identifier: GPL-3.0-only
"""Heximax: a handcrafted expectimax/max-n baseline using per-seat information.

The evaluator reads the player's own hand, public counts and public resource
ledger. Search expands opponents from sampled beliefs and averages dice,
steals and development-card draws over their distributions. The default uses
one determinization, depth two and a bounded leaf budget.

Use ``heximax(board, ...)`` for the adaptive policy. ``pin_weights=0``
or ``pin_weights=1`` holds an endpoint for testing; ``max_trades=0``
independently disables trading. Explicit ``weights`` supports fixed custom
experiments. Importing this package registers the single ``heximax`` preset.
Shared evaluation terms live in hexset.bots.evaluate; search objectives live
in hexset.bots.stances, and the engine owns the information-set View.
"""

from __future__ import annotations

from hexset.view import View
from .evaluate import BALANCED_WEIGHTS, NO_TRADE_WEIGHTS, TRADING_WEIGHTS, HonestEvaluator, Weights
from .search import DEFAULT_MAX_NODES, HEXIMAX_TRADE_FLOOR, Heximax, heximax

# Import-time registration of the single Heximax preset.
from . import presets  # noqa: F401

__all__ = [
    "BALANCED_WEIGHTS",
    "DEFAULT_MAX_NODES",
    "HEXIMAX_TRADE_FLOOR",
    "Heximax",
    "HonestEvaluator",
    "NO_TRADE_WEIGHTS",
    "TRADING_WEIGHTS",
    "View",
    "Weights",
    "heximax",
]
