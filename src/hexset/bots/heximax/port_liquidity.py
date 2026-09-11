# SPDX-License-Identifier: GPL-3.0-only
"""Opt-in production-weighted port liquidity for ordinary Heximax.

The shipped evaluator is deliberately unchanged.  An arm sets the existing
flat ``port`` weight to zero and adds a public, resource-specific liquidity
term instead::

    sum(rate_r * (1 / ratio_r - 1 / BASE_TRADE_RATIO))

``rate_r`` is the same per-resource production walk used by ``Survey.rate``:
cities multiply production and a robber-blocked hex contributes nothing.
Gold contributes to total production but has no fixed resource rate and is
therefore omitted from this trade-specific term.
"""
from __future__ import annotations

import random
from dataclasses import asdict, replace
from typing import Sequence

import numpy as np

from hexset.arena import Entrant, register_entrant_kind, register_preset
from hexset.board.ports import BASE_TRADE_RATIO
from hexset.board.terrain import NUM_RESOURCES
from hexset.state import GameState
from ..evaluate import ROLLS
from .evaluate import HonestEvaluator, NO_TRADE_WEIGHTS, Survey, Weights
from .search import DEFAULT_MAX_NODES, Heximax


LIQUIDITY_COEFFICIENTS = (1.3925, 2.785, 5.57)
FLAT_PORT_4X = 4.0 * NO_TRADE_WEIGHTS.port


def _board_key(board) -> tuple:
    """The immutable board inputs captured by ``Evaluator.__init__``."""
    return (
        tuple(board.terrain), tuple(board.tokens),
        tuple((p.edge, p.vertices, p.resource, p.ratio) for p in board.ports),
        tuple(board.topology.edges),
    )


def _state_key(state: GameState) -> tuple:
    """All public state fields read by the liquidity calculation."""
    return (
        tuple(state.vertex_owner), tuple(state.vertex_building),
        tuple(state.edge_owner), state.robber,
    )


def production_rates(
    evaluator: HonestEvaluator, state: GameState, seat: int,
) -> tuple[float, ...]:
    """Per-resource expected cards per roll for ``seat``.

    This mirrors ``Evaluator.survey``'s exact yield loop, including building
    multiplicity and robber exclusion.  The evaluator's precomputed ``yields``
    is used so terrain/token interpretation cannot drift from the native walk.
    """
    rates = [0.0] * NUM_RESOURCES
    robber = state.robber
    for vertex, owner in enumerate(state.vertex_owner):
        if owner != seat:
            continue
        built = state.vertex_building[vertex]
        for hex_index, resource, pips in evaluator.inner.yields[vertex]:
            if hex_index == robber or resource is None:
                continue
            rates[int(resource)] += built * pips
    return tuple(rate / ROLLS for rate in rates)


def liquidity_from_rates(
    rates: Sequence[float], ratios: Sequence[int | float],
) -> float:
    """Production-weighted value of improved maritime trade ratios."""
    if len(rates) != NUM_RESOURCES or len(ratios) != NUM_RESOURCES:
        raise ValueError("rates and ratios must have one entry per resource")
    return sum(
        float(rate) * (1.0 / float(ratio) - 1.0 / BASE_TRADE_RATIO)
        for rate, ratio in zip(rates, ratios)
    )


class PortLiquidityEvaluator(HonestEvaluator):
    """Native evaluator plus an optional production-weighted liquidity term."""

    def __init__(
        self, board, weights: Weights | None = None, *,
        liquidity_coefficient: float = 0.0,
    ) -> None:
        super().__init__(board, weights)
        self.liquidity_coefficient = float(liquidity_coefficient)
        self._liquidity_cache: dict[tuple, float] = {}
        self._liquidity_board_key = _board_key(board)

    def clear_liquidity_cache(self) -> None:
        self._liquidity_cache.clear()

    def _liquidity(self, state: GameState, seat: int, walk: Survey | None = None) -> float:
        # The complete key is intentionally public and includes the board
        # captured by this evaluator, occupancy, edges, robber and seat.
        key = (self._liquidity_board_key, _state_key(state), seat)
        cached = self._liquidity_cache.get(key)
        if cached is not None:
            return cached
        actual_walk = self._walk(state, seat) if walk is None else walk
        value = liquidity_from_rates(
            production_rates(self, state, seat), actual_walk.ratios,
        )
        self._liquidity_cache[key] = value
        return value

    def score(self, state, seat, hand, *, knower=None, belief=None) -> float:
        total = super().score(state, seat, hand, knower=knower, belief=belief)
        if self.liquidity_coefficient:
            total += self.liquidity_coefficient * self._liquidity(state, seat)
        return total

    def score_many(self, state, knower, hands: np.ndarray) -> np.ndarray:
        total = super().score_many(state, knower, hands)
        if self.liquidity_coefficient:
            total = total + self.liquidity_coefficient * np.asarray(
                [self._liquidity(state, seat) for seat in range(state.num_players)]
            )[None, :]
        return total


class PortLiquidityHeximax(Heximax):
    """Heximax that bounds the opt-in board cache to one decision."""

    def choose(self, game):
        self.evaluator.clear_liquidity_cache()
        return super().choose(game)


def candidate_weights(arm: str) -> tuple[Weights, float]:
    """Return the exact vector and liquidity coefficient for a candidate arm."""
    if arm == "drop-flat-port":
        return replace(NO_TRADE_WEIGHTS, port=0.0), 0.0
    if arm == "flat-port-4x":
        return replace(NO_TRADE_WEIGHTS, port=FLAT_PORT_4X), 0.0
    if arm.startswith("liquidity-"):
        coefficient = float(arm.removeprefix("liquidity-"))
        if coefficient not in LIQUIDITY_COEFFICIENTS:
            raise ValueError(f"unknown liquidity coefficient: {coefficient}")
        return replace(NO_TRADE_WEIGHTS, port=0.0), coefficient
    raise ValueError(f"unknown port-liquidity arm: {arm}")


ARM_LABELS = (
    "control", "drop-flat-port", "flat-port-4x",
    "liquidity-1.3925", "liquidity-2.785", "liquidity-5.57",
)
CANDIDATE_LABELS = ARM_LABELS[1:]


def _spawn(entrant: Entrant, board, rng: random.Random) -> Heximax:
    arm = entrant.name.removeprefix("heximax-port-liquidity-")
    weights, coefficient = candidate_weights(arm)
    # Entrant.weights is an identity field in manifests; the arm's immutable
    # map is the executable definition, while this assertion catches a stale
    # worker/preset crossing before it can play a game.
    if entrant.weights != weights:
        raise ValueError(f"weights do not match port-liquidity arm {arm}")
    evaluator = PortLiquidityEvaluator(
        board, weights, liquidity_coefficient=coefficient,
    )
    return PortLiquidityHeximax(
        evaluator, depth=entrant.depth,
        width=entrant.width, max_nodes=(entrant.max_nodes or DEFAULT_MAX_NODES),
        k=entrant.k, rng=rng, stance=entrant.stance or "win",
        max_trades=0, mode="notrade", temperature=entrant.temperature,
    )


def entrant_for(arm: str) -> Entrant:
    if arm not in CANDIDATE_LABELS:
        raise ValueError(f"candidate arm required: {arm}")
    weights, _ = candidate_weights(arm)
    name = "heximax-port-liquidity-" + arm
    return Entrant(
        name, kind=name, weights=weights, depth=2, width=6, max_nodes=600,
        k=1, mode="notrade", max_trades=0,
    )


for _arm in CANDIDATE_LABELS:
    _entrant = entrant_for(_arm)
    register_entrant_kind(_entrant.kind, _spawn)
    register_preset(_entrant.name, _entrant)


def arm_manifest(arm: str) -> dict:
    """Serializable phenotype for one arm, including the unchanged search."""
    if arm == "control":
        weights = NO_TRADE_WEIGHTS
        kind = "heximax"
        name = "heximax-notrade"
        coefficient = 0.0
    else:
        weights, coefficient = candidate_weights(arm)
        kind = name = "heximax-port-liquidity-" + arm
    return {
        "arm": arm, "name": name, "kind": kind, "mode": "notrade",
        "depth": 2, "width": 6, "max_nodes": 600, "k": 1,
        "max_trades": 0, "stance": "win", "temperature": None,
        "weights": asdict(weights), "liquidity_coefficient": coefficient,
    }


__all__ = [
    "ARM_LABELS", "CANDIDATE_LABELS", "FLAT_PORT_4X",
    "LIQUIDITY_COEFFICIENTS", "PortLiquidityEvaluator", "PortLiquidityHeximax", "arm_manifest",
    "candidate_weights", "entrant_for", "liquidity_from_rates", "production_rates",
]
