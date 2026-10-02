# SPDX-License-Identifier: GPL-3.0-only
"""Bot interfaces and the uniform random policy.

A bot only needs ``choose(game)``. Trading is optional: a gate implements
``gains_many(view, received, counterparties)`` and must declare a
nonnegative ``trade_floor`` when it returns positive gains. ``hexset.trading``
also accepts boolean ``accepts``/``accepts_many`` gates. ``TradesBy`` seats
one bot's moves over another's trading.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ..actions import Action, options_for
from ..game import Game
from ..trading import Bundle
from ..view import View

__all__ = [
    "Bot",
    "TradeGate",
    "TradesBy",
    "seat_at",
    "RandomBot",
]


class Bot(Protocol):
    """Choose a legal action for the seat currently entitled to act."""

    def choose(self, game: Game) -> Action: ...


class TradeGate(Protocol):
    """Value exchanges from one seat's information set."""

    trade_floor: float

    def gains_many(
        self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[float]: ...


class TradesBy:
    """``mover``'s moves over ``trader``'s trading.

    ``choose`` is the mover's. Every other name -- each trade hook the engine
    and the arena look up (``candidates``, ``offer``, ``respond``,
    ``respond_any``, ``pick``, ``consent_gain``, ``gains_many``,
    ``estimate_many``, ``observe_trade``, ``trade_params``, the offer budget)
    -- is the trader's, read, written and deleted through, so
    ``hexset.trading.retune`` on the seat retunes the trader and reinstalls
    its protocol.
    Neither side knows about the other: any bot can move and any gate can
    trade. The trader is only ever asked about exchanges, so a search bot used
    as one never searches a move.
    """

    __slots__ = ("mover", "trader")

    def __init__(self, mover: Bot, trader: object) -> None:
        object.__setattr__(self, "mover", mover)
        object.__setattr__(self, "trader", trader)

    def choose(self, game: Game) -> Action:
        return self.mover.choose(game)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "trader"), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self.trader, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self.trader, name)

    def __reduce__(self):
        return (TradesBy, (self.mover, self.trader))


def seat_at(bot: object, game: Game) -> None:
    """`seat_at(game)` on every part of `bot` that has one: a network bot's
    gate, a search's, and both halves of a `TradesBy` seat, so a gate asked
    about a trade before its first move prices it on `game`. Other bots have
    nothing to seat."""
    if isinstance(bot, TradesBy):
        seat_at(bot.mover, game)
        seat_at(bot.trader, game)
        return
    method = getattr(bot, "seat_at", None)
    if method is not None:
        method(game)


@dataclass
class RandomBot:
    """Sample uniformly from legal actions; decline player trades."""

    rng: random.Random = field(default_factory=random.Random)
    trade_floor: float = 0.0

    def choose(self, game: Game) -> Action:
        return self.rng.choice(options_for(game))
