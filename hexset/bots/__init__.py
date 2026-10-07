# SPDX-License-Identifier: GPL-3.0-only
"""What a bot is to the engine, and the pieces any bot can compose.

`Bot` is the whole interface: `choose(game) -> Action`. `TradeGate` is the
optional trading half, `TradesBy` seats one bot's moves over another's
trading, `Handoff` seats one bot's moves up to a round and another's after
it, `PlaysAgainst` is the optional hook for playing as one side against
named seats, `Coalition` seats any bot as one of a side against the rest of
its table, and `RandomBot` is the one bot shipped here. `determinized`
samples information-set worlds for a search; `stances` turns a per-seat
vector into one seat's objective.

A bot this package does not ship joins a table one of two ways: registered in
process (`hexset.arena.register_entrant_kind`, `register_preset`,
`register_spec`, loaded by `hexset.arena.load_runtime`), or as a client of the
HTTP API (`hexset.clients.botclient` is an example).
"""
from .base import (
    Bot, Handoff, PlaysAgainst, RandomBot, TradeGate, TradesBy, handed_off, play_against, seat_at,
)
from .coalition import Coalition, targets_at
from .stances import STANCES, own, paranoid, relative

__all__ = [
    "Bot", "RandomBot", "TradeGate", "TradesBy", "Handoff", "handed_off", "seat_at",
    "PlaysAgainst", "play_against", "Coalition", "targets_at",
    "STANCES", "own", "paranoid", "relative",
]
