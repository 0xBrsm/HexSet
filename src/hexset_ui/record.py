"""A played game as a board plus a sequence of actions, and the replay that
checks the two still agree.

`hexset_ui.journal` is what actually writes games to disk here; this module is
the shape a journal's contents are checked against. A record stores the board in
full and the actions taken, never encoded features, so a change to the encoder
cannot invalidate it.

The board is written out rather than stored as the seed that generated it: a
seed only reproduces a board for as long as the board generator is untouched.
Hex coordinates go in too, so a Seafarers layout records exactly like the base
board.

Dice and steals are the one thing not stored. They come back from the seeded
random stream, which means a change to how the engine draws randomness shows up
as a replay mismatch rather than as a game that quietly replays wrong.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

from .actions import Action, ActionType, apply, is_legal, legal_actions
from .board.board import Board, make_board
from .board.coords import Hex
from .board.ports import Port
from .board.terrain import Resource, Terrain
from .board.topology import build as build_topology
from .game import Game, start


class ReplayError(RuntimeError):
    """Raised when a record no longer describes the game it claims to."""


@dataclass(frozen=True)
class Record:
    layout: tuple[tuple[int, int, int], ...]
    terrain: tuple[int, ...]
    tokens: tuple[int, ...]
    ports: tuple[tuple[int, int, int | None], ...]
    num_players: int
    seed: int
    actions: tuple[tuple[int, int, int], ...]
    winner: int | None
    turns: int
    # Trade offers, sparse by step: the two bundles and who the proposer wanted
    # to take it. An offer will not fit in an action triple, and proposals are a
    # small fraction of the actions in a game, so widening every action to carry
    # mostly-empty fields would cost far more than storing the few that exist.
    offers: tuple[
        tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...
    ] = ()

    @property
    def decided(self) -> bool:
        return self.winner is not None


def board_fields(board: Board) -> dict[str, tuple]:
    """The parts of a board a record needs. Port vertices are left out — they
    follow from the edge once the topology is rebuilt."""
    return {
        "layout": tuple(tuple(h) for h in board.topology.hexes),
        "terrain": tuple(int(t) for t in board.terrain),
        "tokens": tuple(board.tokens),
        "ports": tuple(
            (p.edge, p.ratio, None if p.resource is None else int(p.resource))
            for p in board.ports
        ),
    }


def board_of(record: Record) -> Board:
    topology = build_topology(Hex(*h) for h in record.layout)
    ports = tuple(
        Port(
            edge=edge,
            vertices=(topology.edges[edge][0], topology.edges[edge][1]),
            resource=None if resource is None else Resource(resource),
            ratio=ratio,
        )
        for edge, ratio, resource in record.ports
    )
    return make_board(
        topology,
        tuple(Terrain(t) for t in record.terrain),
        tuple(record.tokens),
        ports,
    )


def actions_of(record: Record) -> Iterator[Action]:
    """The recorded actions, with trade offers put back on the ones that had them.

    Everything that walks a record goes through here, so the reconstruction
    lives in one place and cannot drift between replaying, featurising and
    behaviour analysis.
    """
    offers = {
        step: (tuple(g), tuple(w), tuple(k)) for step, g, w, k in record.offers
    }
    for step, (kind, a, b) in enumerate(record.actions):
        give, want, ask = offers.get(step, ((), (), ()))
        yield Action(ActionType(kind), a, b, give, want, ask)


def replay(record: Record) -> Game:
    """Re-play a record, checking it still describes the game it claims to."""
    game = start(board_of(record), record.num_players, random.Random(record.seed))
    for step, action in enumerate(actions_of(record)):
        if not is_legal(game, action, legal_actions(game)):
            raise ReplayError(
                f"step {step}: {action} is not legal in {game.phase.name}"
            )
        apply(game, action)

    if (game.won_by, game.turns) != (record.winner, record.turns):
        raise ReplayError(
            f"replay ended {game.won_by} after {game.turns} turns, "
            f"record says {record.winner} after {record.turns}"
        )
    return game
