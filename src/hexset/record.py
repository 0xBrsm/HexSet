# SPDX-License-Identifier: GPL-3.0-only
"""Record played games as data that outlives the code that produced them.

A record stores the board in full and the actions taken, not encoded features.
Feature layouts change every time the encoder is touched, and a dataset frozen
in one of them is worth nothing afterwards; a replayable action sequence can be
re-encoded however the model wants, as many times as it wants.

The board is written out rather than stored as the seed that generated it, for
the same reason: a seed only reproduces a board for as long as the board
generator is untouched. Hex coordinates go in too, so a Seafarers layout records
exactly like the base board.

Dice and steals are the one thing not stored. They come back from the seeded
random stream, which means a change to how the engine draws randomness shows up
as a replay mismatch rather than as quietly wrong training data.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, Sequence

from .actions import Action, ActionType, apply, legal_actions
from .arena import MAX_ACTIONS
from .board.board import Board, make_board
from .board.coords import Hex
from .board.ports import Port
from .board.terrain import Resource, Terrain
from .board.topology import build as build_topology
from .bots import Bot
from .game import Game, is_over, start, to_move


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


def record_game(
    bots: Sequence[Bot],
    board: Board,
    seed: int,
    *,
    action_cap: int = MAX_ACTIONS,
) -> Record:
    """Play one game and record it. Each bot is seated at its own index."""
    game = start(board, len(bots), random.Random(seed))
    actions: list[tuple[int, int, int]] = []
    offers: list[
        tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    ] = []
    while not is_over(game) and len(actions) < action_cap:
        action = bots[to_move(game)].choose(game)
        apply(game, action)
        if action.give or action.want:
            offers.append(
                (
                    len(actions),
                    tuple(action.give),
                    tuple(action.want),
                    tuple(action.ask),
                )
            )
        actions.append((int(action.type), action.a, action.b))

    return Record(
        num_players=len(bots),
        seed=seed,
        actions=tuple(actions),
        offers=tuple(offers),
        winner=game.won_by,
        turns=game.turns,
        **board_fields(board),
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
        if action not in legal_actions(game):
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


def to_json(record: Record) -> str:
    return json.dumps(asdict(record), separators=(",", ":"))


def from_json(line: str) -> Record:
    raw = json.loads(line)
    return Record(
        layout=tuple(tuple(h) for h in raw["layout"]),
        terrain=tuple(raw["terrain"]),
        tokens=tuple(raw["tokens"]),
        ports=tuple(tuple(p) for p in raw["ports"]),
        num_players=raw["num_players"],
        seed=raw["seed"],
        actions=tuple(tuple(a) for a in raw["actions"]),
        offers=tuple(
            (step, tuple(give), tuple(want), tuple(ask))
            for step, give, want, ask in raw.get("offers", ())
        ),
        winner=raw["winner"],
        turns=raw["turns"],
    )


def write(path: str, records: Iterable[Record]) -> int:
    """Append records as JSON lines. Returns how many were written."""
    written = 0
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(to_json(record) + "\n")
            written += 1
    return written


def read(path: str) -> Iterator[Record]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield from_json(line)
