"""Reading a policy's values where its decisions are finished.

A policy that decides a setup turn (a settlement, then its road) or a Road
Building card (the card, then each free road) as one choice may have been
taught its win head only at positions where a decision starts. A search that
walks one move at a time still reaches the positions inside such a decision
and asks for a value there, off the head's ground. `Finishing` answers those
reads where the decision is finished instead: on an exact copy of the game,
the deciding seat plays out the rest of the decision with the policy's own
most probable moves, and the value is read there. Its prior is the policy's,
read at the position itself, so the search still chooses among the moves it
was asked about. A checkpoint asks for it with `finish_decisions=1` in its
metadata (`hexset.clients._modelmeta.finish_decisions_config`).
"""
from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from hexset.actions import apply, legal_actions
from hexset.game import Game, Phase, imagine, pending_free_roads, to_move

# A setup turn is one road after its settlement; a Road Building card two
# roads after the card. The bound only guards a policy whose moves never
# leave the decision.
MAX_FINISHING_MOVES = 8


def inside(game: Game) -> bool:
    """Whether `game` stands inside a setup turn or a Road Building card."""
    return game.phase is Phase.SETUP_ROAD or bool(pending_free_roads(game))


def finished(policy, game: Game) -> Game:
    """An exact copy of `game` with the decision under way played to its end
    by `policy`'s most probable moves. `game` itself is untouched; no chance
    is drawn, since placing a road draws none."""
    copy = imagine(game, random.Random(0), randomize_deck=False)
    for _ in range(MAX_FINISHING_MOVES):
        if not inside(copy):
            break
        seat = to_move(copy)
        options = tuple(legal_actions(copy))
        if not options:
            break
        ((prior, _),) = policy.score_rows([(copy, seat, options)])
        apply(copy, options[int(np.argmax(prior))])
    return copy


class Finishing:
    """`policy`, reading every value inside a setup turn or a Road Building
    card where that decision is finished (the module docstring). Every other
    name is the policy's."""

    def __init__(self, policy) -> None:
        self._policy = policy

    def __getattr__(self, name: str):
        return getattr(self._policy, name)

    def _finish(self, rows: Sequence[tuple]) -> list[tuple]:
        done: dict[int, Game] = {}
        out = []
        for row in rows:
            game = row[0]
            if inside(game):
                if id(game) not in done:
                    done[id(game)] = finished(self._policy, game)
                row = (done[id(game)], *row[1:])
            out.append(row)
        return out

    def score_rows(self, rows: Sequence[tuple[Game, int, tuple]]):
        """`(prior, value)` per row: the prior at the row's own position, the
        value where its decision is finished."""
        scored = self._policy.score_rows(rows)
        if not any(inside(row[0]) for row in rows):
            return scored
        values = self._policy.value_rows([(game, seat) for game, seat, *_ in self._finish(rows)])
        return [(prior, value) for (prior, _), value in zip(scored, values)]

    def value_rows(self, rows: Sequence[tuple[Game, int]]) -> list[tuple[float, ...]]:
        """The value per row, where its decision is finished."""
        return self._policy.value_rows(self._finish(rows))
