# SPDX-License-Identifier: GPL-3.0-only
"""What a bot is to the engine, and the pieces any bot can compose.

`Bot` is the whole interface: `choose(game) -> Action`. `TradeGate` is the
optional trading half, `TradesBy` seats one bot's moves over another's
trading, and `RandomBot` is the one bot shipped here. `determinized` samples
information-set worlds for a search; `stances` turns a per-seat vector into
one seat's objective.

A bot this package does not ship joins a table one of two ways: registered in
process (`hexset.arena.register_entrant_kind`, `register_preset`,
`register_spec`, loaded by `hexset.arena.load_runtime`), or as a client of the
HTTP API (`hexset.clients.botclient` is an example).
"""
from .base import Bot, RandomBot, TradeGate, TradesBy, seat_at
from .stances import STANCES, own, paranoid, relative

__all__ = [
    "Bot", "RandomBot", "TradeGate", "TradesBy", "seat_at",
    "STANCES", "own", "paranoid", "relative",
]
