# SPDX-License-Identifier: GPL-3.0-only
"""Bot interfaces and the uniform random policy.

A bot only needs ``choose(game)``. Player trading is optional: a gate can
implement ``gains_many(view, received, counterparties)`` and must declare a
nonnegative ``trade_floor`` when it returns positive gains. Boolean
``accepts``/``accepts_many`` gates are also supported by ``hexset.trading``.
Server rounds additionally accept ``offer``, ``respond``, ``pick``, and
``estimate_many`` methods; trading.py supplies their defaults structurally.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ..actions import Action, options_for
from ..game import Game
from ..trading import Bundle
from ..view import View


class Bot(Protocol):
    """Choose a legal action for the seat currently entitled to act."""

    def choose(self, game: Game) -> Action: ...


class TradeGate(Protocol):
    """Value exchanges from one seat's information set."""

    trade_floor: float

    def gains_many(
        self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[float]: ...


@dataclass
class RandomBot:
    """Sample uniformly from legal actions; decline player trades."""

    rng: random.Random = field(default_factory=random.Random)
    trade_floor: float = 0.0

    def choose(self, game: Game) -> Action:
        return self.rng.choice(options_for(game))
