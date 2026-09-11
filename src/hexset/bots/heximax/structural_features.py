# SPDX-License-Identifier: GPL-3.0-only
"""Opt-in structural feature family for ordinary Heximax.

This module deliberately sits beside the shipped evaluator.  Importing it is
required to register the three candidate entrant kinds; importing
``hexset.bots`` alone therefore leaves the production presets unchanged.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hexset.actions import Action
from hexset.arena import Entrant, register_entrant_kind, register_preset
from hexset.economy import COSTS, Purchase
from hexset.game import Game
from hexset.state import GameState
from hexset.victory import NO_OWNER
from hexset.roads import longest_road
from .evaluate import (
    NO_TRADE_WEIGHTS, PURCHASE_VALUE, HonestEvaluator, Survey, affordable,
)
from .search import DEFAULT_MAX_NODES, Heximax

# These are preregistered constants.  The first is applied to the raw residual
# purchase progress (which already includes the purchase's value), and the
# second is a penalty in the evaluator's VP units.
BACKUP_PROGRESS_COEFFICIENT = 0.45
AWARD_FRAGILITY_COEFFICIENT = -0.25
STRUCTURAL_TEMPERATURE = 2.476644394795811

_PURCHASES = tuple(PURCHASE_VALUE)


def _progress(hand: Sequence[float], purchase: Purchase) -> float:
    """The existing value-weighted progress for one purchase."""
    cost = COSTS[purchase]
    total = sum(cost)
    if not total:
        return 0.0
    return PURCHASE_VALUE[purchase] * sum(
        min(float(hand[r]), n) for r, n in enumerate(cost) if n
    ) / total


def backup_progress(
    hand: Sequence[float], walk: Survey, *, deck_left: int,
) -> float:
    """Best second purchase after reserving cards for the existing best one.

    The first argmax is exactly the existing ``hand_terms`` purchase loop:
    only purchases open on the board participate and ties retain its insertion
    order.  Its committed cards are the per-resource minima for that purchase;
    the residual therefore cannot be spent twice.  The second argmax excludes
    that purchase and uses the same open-purchase and value-weighted formula.
    """
    open_purchases = [
        p for p in _PURCHASES if affordable(p, walk, deck_left)
    ]
    if not open_purchases:
        return 0.0
    best = 0.0
    selected: Purchase | None = None
    for purchase in open_purchases:
        value = _progress(hand, purchase)
        if value > best:
            best, selected = value, purchase
    if selected is None:
        return 0.0

    committed = COSTS[selected]
    residual = [
        max(0.0, float(hand[r]) - min(float(hand[r]), committed[r]))
        for r in range(len(committed))
    ]
    second = 0.0
    for purchase in open_purchases:
        if purchase is selected:
            continue
        second = max(second, _progress(residual, purchase))
    return second


def _state_key(state: GameState) -> tuple:
    """All public inputs read by award fragility and longest-road traversal."""
    return (
        tuple(state.vertex_owner), tuple(state.vertex_building),
        tuple(state.edge_owner), state.longest_road_holder,
        state.largest_army_holder, tuple(state.knights_played),
    )


class StructuralEvaluator(HonestEvaluator):
    """HonestEvaluator plus one or both preregistered structural features."""

    def __init__(self, board, weights=None, *, backup=False, fragility=False):
        super().__init__(board, weights or NO_TRADE_WEIGHTS)
        self.backup_enabled = bool(backup)
        self.fragility_enabled = bool(fragility)
        self._structural_cache: dict[tuple, tuple[float, float]] = {}

    def clear_structural_cache(self) -> None:
        """Clear state-derived structural values at the start of a choose."""
        self._structural_cache.clear()

    def _award_fragility(self, state: GameState, seat: int) -> float:
        key = (_state_key(state), seat)
        cached = self._structural_cache.get(key)
        if cached is not None:
            return cached[1]

        result = 0.0
        if state.longest_road_holder == seat:
            holder = longest_road(state, seat)
            opponents = [longest_road(state, p) for p in range(state.num_players) if p != seat]
            max_opponent = max(opponents, default=0)
            gap = holder - max_opponent
            result += 2.0 / (1.0 + max(0, gap))
        if state.largest_army_holder == seat:
            holder = state.knights_played[seat]
            max_opponent = max(
                (state.knights_played[p] for p in range(state.num_players) if p != seat),
                default=0,
            )
            gap = holder - max_opponent
            result += 2.0 / (1.0 + max(0, gap))
        # Preserve a backup value if this call followed _features_for; cache
        # entries are state/seat exact and never depend on a hidden hand.
        prior = self._structural_cache.get(key)
        backup = prior[0] if prior is not None else 0.0
        self._structural_cache[key] = (backup, result)
        return result

    def _features(
        self, state: GameState, seat: int, hand: Sequence[float],
    ) -> tuple[float, float]:
        backup = 0.0
        if self.backup_enabled:
            backup = backup_progress(hand, self._walk(state, seat), deck_left=len(state.deck))
        fragility = self._award_fragility(state, seat) if self.fragility_enabled else 0.0
        return backup, fragility

    def score(self, state, seat, hand, *, knower=None, belief=None) -> float:
        total = super().score(state, seat, hand, knower=knower, belief=belief)
        backup, fragility = self._features(state, seat, hand)
        return total + BACKUP_PROGRESS_COEFFICIENT * backup + AWARD_FRAGILITY_COEFFICIENT * fragility

    def score_many(self, state: GameState, knower: int, hands: np.ndarray) -> np.ndarray:
        total = super().score_many(state, knower, hands)
        if self.backup_enabled:
            for row in range(hands.shape[0]):
                for seat in range(state.num_players):
                    total[row, seat] += BACKUP_PROGRESS_COEFFICIENT * backup_progress(
                        hands[row, seat], self._walk(state, seat), deck_left=len(state.deck)
                    )
        if self.fragility_enabled:
            for seat in range(state.num_players):
                total[:, seat] += AWARD_FRAGILITY_COEFFICIENT * self._award_fragility(state, seat)
        return total


class StructuralHeximax(Heximax):
    """Heximax variant whose opt-in evaluator cache is per decision."""

    def choose(self, game: Game) -> Action:
        self.evaluator.clear_structural_cache()
        return super().choose(game)


@dataclass(frozen=True)
class StructuralSpec:
    label: str
    backup: bool
    fragility: bool


SPECS = {
    "backup": StructuralSpec("backup", True, False),
    "fragility": StructuralSpec("fragility", False, True),
    "combined": StructuralSpec("combined", True, True),
}


def _spawn(entrant: Entrant, board, rng: random.Random) -> Heximax:
    spec = SPECS[entrant.name.rsplit("-", 1)[-1]]
    evaluator = StructuralEvaluator(
        board, entrant.weights or NO_TRADE_WEIGHTS,
        backup=spec.backup, fragility=spec.fragility,
    )
    return StructuralHeximax(
        evaluator, depth=entrant.depth, width=entrant.width,
        max_nodes=entrant.max_nodes or DEFAULT_MAX_NODES, k=entrant.k, rng=rng,
        stance=entrant.stance or "win", max_trades=0,
        mode="notrade",
        temperature=(entrant.temperature if entrant.temperature is not None else STRUCTURAL_TEMPERATURE),
        placement=True,
    )


for _label in SPECS:
    _kind = "heximax-structural-" + _label
    register_entrant_kind(_kind, _spawn)
    register_preset(_kind, Entrant(
        _kind, kind=_kind, depth=2, width=6, max_nodes=600, k=1,
        mode="notrade", max_trades=0, weights=NO_TRADE_WEIGHTS,
        temperature=STRUCTURAL_TEMPERATURE, placement=False,
    ))


__all__ = [
    "AWARD_FRAGILITY_COEFFICIENT", "BACKUP_PROGRESS_COEFFICIENT",
    "STRUCTURAL_TEMPERATURE",
    "SPECS", "StructuralEvaluator", "StructuralHeximax", "StructuralSpec", "backup_progress",
]
