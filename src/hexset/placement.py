# SPDX-License-Identifier: GPL-3.0-only
"""A heuristic opening-placement prior, fitted to 40,803 recorded games.

`RandomBot` picks a setup vertex uniformly, so every handcrafted entrant in the
arena opens at random.  Self-play is different and worth being precise about:
`Collector._ask` makes no distinction by phase, so a setup settlement is offered
to the training policy exactly like any other action and the network chooses its
own opening.  Near-random at initialisation, learned after that — whether it
learns anything worth having is the open question this prior exists to answer.

The three terms are the only ones the fit kept.  Pip count dominates; touching
more distinct resources and touching a scarce one each add a little.  Number
diversity, port access, complementarity between the two settlements, pip balance
and denial of a rival's best corner were all offered to the fit and were all
null at fixed pips.

Weights come from a conditional logit over the four seats of a game, which
cancels the board and any game-level effect because exactly one seat wins.  They
are expressed in pips, since scaling every weight together cannot change which
vertex ranks first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from .actions import Action, ActionType, legal_actions
from .board.board import Board, pips, scarce_resources
from .board.terrain import TERRAIN_RESOURCE, Resource
from .game import Game, Phase, to_move
from .state import GameState

# Fitted exchange rates: one extra distinct resource is worth 1.19 pips, and
# reaching a scarce resource a further 0.91.  Held-out log-loss 1.3746 against
# 1.3863 for chance, picking the winner 29.9% of the time against 25%.  That
# gap is small because the opening only weakly determines the winner, not
# because the weights are poorly fitted — the train/test gap is 0.002.
RESOURCE_WEIGHT = 1.19
SCARCE_WEIGHT = 0.91


def score(board: Board, vertices: Iterable[int], scarce: frozenset[Resource]) -> float:
    """Rate a whole opening: the vertices one seat holds, candidate included.

    Hexes are counted once per vertex that touches them, not once overall: two
    settlements sharing a hex both collect from it, so it really does pay twice.
    """
    topology = board.topology
    total = 0
    kinds: set[Resource] = set()
    for vertex in vertices:
        for hex_index in topology.vertex_hexes[vertex]:
            total += pips(board.tokens[hex_index])
            resource = TERRAIN_RESOURCE[board.terrain[hex_index]]
            if resource is not None:
                kinds.add(resource)
    return total + RESOURCE_WEIGHT * len(kinds) + SCARCE_WEIGHT * len(kinds & scarce)


def rank(state: GameState, player: int, candidates: Iterable[int]) -> list[tuple[float, int]]:
    """Score each candidate vertex as the opening it would complete."""
    held = [v for v, owner in enumerate(state.vertex_owner) if owner == player]
    scarce = scarce_resources(state.board)
    return sorted(
        ((score(state.board, [*held, vertex], scarce), vertex) for vertex in candidates),
        key=lambda pair: (-pair[0], pair[1]),
    )


def best(state: GameState, player: int, candidates: Iterable[int]) -> int:
    """The highest-scoring candidate, ties broken by vertex index for repeatability.

    Greedy per pick rather than over the pair, which is what the corpus shows
    humans doing: setup play there is pip-greedy with no denial component.
    """
    options = rank(state, player, candidates)
    if not options:
        raise ValueError(f"no candidate vertices for player {player}")
    return options[0][1]


class Bot(Protocol):
    def choose(self, game: Game) -> Action: ...


@dataclass
class PlacementBot:
    """Any bot, with its setup settlements chosen by the prior instead.

    A wrapper rather than a bot in its own right so that the same entrant can be
    run with and without it: the only difference between the two arms of the
    duel is then the eight setup picks, which is the comparison worth having.
    Setup roads are left alone — this measures placement, not road policy.
    """

    inner: Bot

    def choose(self, game: Game) -> Action:
        if game.phase is not Phase.SETUP_SETTLEMENT:
            return self.inner.choose(game)
        options = [a for a in legal_actions(game) if a.type is ActionType.SETUP_SETTLEMENT]
        if not options:
            return self.inner.choose(game)
        chosen = best(game.state, to_move(game), [action.a for action in options])
        return Action(ActionType.SETUP_SETTLEMENT, chosen)


__all__ = ["PlacementBot", "best", "rank", "scarce_resources", "score"]
