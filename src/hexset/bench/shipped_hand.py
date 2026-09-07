# SPDX-License-Identifier: GPL-3.0-only
"""The hand valuation heximax and search2 shipped before 2026-09-06, frozen.

An instrument, not a bot: the three old hand terms (`progress`, `held`,
`surplus_card`) were deleted from `hexset.bots.evaluate` when `hand_terms`
replaced them, and a strength gate against "the current shipped weights"
needs both term sets alive in one process to play them head to head on the
same board. Everything else -- the search, the trade gates, the seven board
terms and their weights -- is the shipped code, reached by subclassing.

Registers two entrant kinds with `hexset.arena`: `heximax-shipped-hand` and
`search2-shipped-hand`, each the shipped preset of that bot with this term
set and the weights it was fitted at. There is no frozen publication scale
to go with them: the marginal-scale layer this file once had to freeze was
deleted with the public valuation vector, so both term sets now publish
gains in the same unit -- win probability -- and are directly comparable.
"""

from __future__ import annotations

import random
from typing import Sequence

from hexset.arena import Entrant, register_entrant_kind, register_preset
from hexset.board.board import Board
from hexset.bots.evaluate import Evaluator, Survey, Weights
from hexset.bots.heximax.evaluate import HonestEvaluator
from hexset.bots.heximax.search import Heximax
from hexset.bots.search2 import SearchBot
from hexset.economy import COSTS, Purchase
from hexset.robber import DISCARD_THRESHOLD
from hexset.state import MAX_CITIES, MAX_SETTLEMENTS, GameState

# Roads were left out of the old build-progress term: at one wood and one
# brick nearly any hand is half way to a road, so including it scored almost
# every hand alike.
OLD_PROGRESS_PURCHASES = (Purchase.SETTLEMENT, Purchase.CITY, Purchase.DEV_CARD)

# `Weights` as it shipped, in the slots the new fields occupy: `buy_progress`
# holds the old `progress`, `spare_card` the old flat `card`, `robber_risk`
# the old `surplus_card`. The term functions below are the matching ones, so
# the product is the shipped score to the last bit.
SHIPPED_WEIGHTS = Weights(
    buy_progress=0.01843,
    spare_card=0.005406,
    robber_risk=-0.3891,
)


def old_hand_terms(
    hand: Sequence[float], walk: Survey
) -> tuple[float, float, float]:
    """(progress, cards held, cards over seven), the shipped trio."""
    best = 0.0
    for purchase in OLD_PROGRESS_PURCHASES:
        if purchase is Purchase.SETTLEMENT and walk.settlements >= MAX_SETTLEMENTS:
            continue
        if purchase is Purchase.CITY and walk.cities >= MAX_CITIES:
            continue
        cost = COSTS[purchase]
        toward = sum(
            [min(hand[r], n) for r, n in enumerate(cost) if n]
        ) / sum(cost)
        if toward > best:
            best = toward
    held = sum(hand)
    return best, held, max(0.0, held - DISCARD_THRESHOLD)


class ShippedHandEvaluator(Evaluator):
    """`Evaluator` with the pre-redesign hand terms."""

    def hand_terms(
        self, state: GameState, player: int, walk: Survey
    ) -> tuple[float, float, float]:
        return old_hand_terms(state.hands[player], walk)


class ShippedHandHonestEvaluator(HonestEvaluator):
    """`HonestEvaluator` with the pre-redesign hand terms."""

    def _hand_terms_of(
        self, state: GameState, seat: int, hand: Sequence[float], belief, walk=None,
    ) -> tuple[float, float, float]:
        return old_hand_terms(hand, self._walk(state, seat) if walk is None else walk)


def _spawn_heximax(entrant: Entrant, board: Board, rng: random.Random) -> Heximax:
    kwargs: dict = dict(
        depth=entrant.depth,
        width=entrant.width,
        rng=rng,
        max_trades=entrant.max_trades,
        placement=True,
        mode="honest",
    )
    if entrant.stance is not None:
        kwargs["stance"] = entrant.stance
    return Heximax(
        ShippedHandHonestEvaluator(board, entrant.weights or SHIPPED_WEIGHTS),
        **kwargs,
    )


def _spawn_search2(entrant: Entrant, board: Board, rng: random.Random) -> SearchBot:
    kwargs: dict = dict(
        depth=entrant.depth,
        width=entrant.width,
        rng=rng,
        max_trades=entrant.max_trades,
    )
    if entrant.stance is not None:
        kwargs["stance"] = entrant.stance
    return SearchBot(
        ShippedHandEvaluator(board, entrant.weights or SHIPPED_WEIGHTS),
        **kwargs,
    )


register_entrant_kind("heximax-shipped-hand", _spawn_heximax)
register_entrant_kind("search2-shipped-hand", _spawn_search2)

# Presets under the same names, so any harness that takes entrant names --
# `hexset.bench.trade_census`, `hexset.arena` -- can seat the shipped hand
# valuation beside the new one at one table.
register_preset(
    "heximax-shipped-hand",
    Entrant(
        "heximax-shipped-hand", kind="heximax-shipped-hand", depth=2, width=6,
        weights=SHIPPED_WEIGHTS,
    ),
)
register_preset(
    "search2-shipped-hand",
    Entrant(
        "search2-shipped-hand", kind="search2-shipped-hand", depth=2, width=6,
        weights=SHIPPED_WEIGHTS,
    ),
)
