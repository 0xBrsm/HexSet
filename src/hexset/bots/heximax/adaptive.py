# SPDX-License-Identifier: GPL-3.0-only
"""One policy: interpolate the current no-trade and best trading profiles."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from hexset.bots.evaluate import TERM_NAMES, Weights
from hexset.game import Game
from .evaluate import BALANCED_WEIGHTS, NO_TRADE_WEIGHTS
from .search import Heximax


def trading_profile(activity: float) -> tuple[Weights, float]:
    """Zero is exactly no-trade; one is exactly the balanced trading policy."""
    if not 0.0 <= activity <= 1.0:
        raise ValueError("trading activity must be between zero and one")
    if activity == 0.0:
        return NO_TRADE_WEIGHTS, .25
    if activity == 1.0:
        return BALANCED_WEIGHTS, .125
    weights = Weights(**{
        name: (1.0 - activity) * getattr(NO_TRADE_WEIGHTS, name)
              + activity * getattr(BALANCED_WEIGHTS, name)
        for name in TERM_NAMES
    })
    return weights, (1.0 - activity) * .25 + activity * .125


@dataclass
class TradeActivity:
    """Fraction of the last eight eligible turns with a completed exchange.

    Start at the no-trade endpoint. Startup zeros leave the finite window
    after eight observations, so they create no permanent floor or ceiling.
    This is public realized activity, not opponents' hidden willingness.
    """
    history: deque[bool] = field(default_factory=lambda: deque([False] * 8, maxlen=8))
    last_turn: int = -1

    @property
    def mean(self) -> float:
        return sum(self.history) / len(self.history)

    def observe(self, *, turn: int, actor: int, hand_sizes: tuple[int, ...],
                trade_participants: tuple[tuple[int, int], ...]) -> None:
        if turn <= self.last_turn:
            return
        self.last_turn = turn
        if hand_sizes[actor] < 2 or not any(
            n >= 2 for seat, n in enumerate(hand_sizes) if seat != actor
        ):
            return
        self.history.append(any(actor in pair for pair in trade_participants))


@dataclass
class AdaptiveHeximax(Heximax):
    activity: TradeActivity = field(default_factory=TradeActivity)

    def observe_trade(self, **public_event) -> None:
        self.activity.observe(**public_event)

    def choose(self, game: Game):
        alpha = 0.0 if self.max_trades == 0 or game.max_trades == 0 else self.activity.mean
        weights, expansion = trading_profile(alpha)
        evaluator = self.evaluator
        evaluator.weights = evaluator.inner.weights = weights
        evaluator.vector = evaluator.inner.vector = tuple(getattr(weights, k) for k in TERM_NAMES)
        evaluator.expansion_value = expansion
        # Hold this profile throughout the search; choose clears its caches.
        return super().choose(game)
