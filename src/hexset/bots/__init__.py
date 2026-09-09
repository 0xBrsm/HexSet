# SPDX-License-Identifier: GPL-3.0-only
"""Bots, trade gates, and heuristic search objectives.

Importing this package registers the Heximax arena presets. The factory is
available from ``hexset.bots.heximax`` without shadowing that submodule.
"""
from .base import Bot, RandomBot, TradeGate
from .stances import STANCES, own, paranoid, relative
from .heximax import (
    BY_MODE, MODES, Heximax, HonestEvaluator, NO_TRADE_WEIGHTS,
    TRADING_WEIGHTS, View, Weights,
)

__all__ = [
    "Bot", "RandomBot", "TradeGate", "STANCES", "own", "paranoid", "relative",
    "BY_MODE", "MODES", "Heximax", "HonestEvaluator", "NO_TRADE_WEIGHTS",
    "TRADING_WEIGHTS", "View", "Weights",
]
