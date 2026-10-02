# SPDX-License-Identifier: GPL-3.0-only
"""The information-set record: the boundary between the rules and the model.

A position stated **in the rules' own terms, already filtered to what the
perspective seat may legally know** -- own hand and own development cards
exact, every other seat by count plus `hexset.ledger`'s public-knowledge
reconstruction. There is nowhere in it for a hidden card to be. Everything
downstream (one-hot encoding, rotation, scaling) is mechanical and lives in
`hexn.export_onnx.RecordEncoder`, which also rotates the perspective seat to
slot 0; rows here stay in board-seat order. **Everything here is numpy, and
stays that way**: `import hexset.onnx_record` must succeed with torch absent.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .actions import Action, ActionSpace, legal_actions
from .board.board import Board
from .board.terrain import NUM_RESOURCES
from .cards import NUM_DEV_CARDS
from .devcards import dev_count
from .economy import hand_size
from .encoding import StaticGraph
from .game import Game, to_move
from .state import NO_OWNER
from .victory import award_points

__all__ = [
    "CONTRACT_VERSION",
    "RECORD_FIELDS",
    "record_shapes",
    "record_from_game",
    "record_batch",
]


# The record/graph contract, read off an exported graph's metadata; earlier
# contracts are refused. Bump only when `RECORD_FIELDS`, the export's output
# tuple, or the action space changes.
CONTRACT_VERSION = "6"

# Board, then position, then information set, then legality.
# `hexn.export_onnx` reuses this order for the graph's input names.
RECORD_FIELDS: tuple[str, ...] = (
    "terrain",
    "token",
    "port_code",
    "robber",
    "vertex_owner",
    "vertex_building",
    "edge_owner",
    "bank",
    "knights_played",
    "award_points",
    "longest_road_holder",
    "largest_army_holder",
    "phase",
    "free_roads",
    "deck_size",
    "turns",
    "perspective",
    "own_hand",
    "hand_totals",
    "own_dev",
    "dev_totals",
    "ledger_known",
    "ledger_unknown",
    "action_mask",
)


def record_shapes(graph: StaticGraph, players: int, space: ActionSpace) -> dict[str, tuple]:
    """Every input field's per-row shape, i.e. `RECORD_FIELDS` minus the
    batch axis. `hexn.export_onnx` extends it with its own output shapes."""
    return {
        "terrain": (graph.num_hexes,),
        "token": (graph.num_hexes,),
        "port_code": (graph.num_vertices,),
        "robber": (),
        "vertex_owner": (graph.num_vertices,),
        "vertex_building": (graph.num_vertices,),
        "edge_owner": (graph.num_edges,),
        "bank": (NUM_RESOURCES,),
        "knights_played": (players,),
        "award_points": (players,),
        "longest_road_holder": (),
        "largest_army_holder": (),
        "phase": (),
        "free_roads": (),
        "deck_size": (),
        "turns": (),
        "perspective": (),
        "own_hand": (NUM_RESOURCES,),
        "hand_totals": (players,),
        "own_dev": (NUM_DEV_CARDS,),
        "dev_totals": (players,),
        "ledger_known": (players, NUM_RESOURCES),
        "ledger_unknown": (players,),
        "action_mask": (space.size,),
    }


def _port_code(board: Board) -> np.ndarray:
    """One code per vertex: `-1` no port, `0` generic, `1 + r` a port for
    resource `r`."""
    codes = np.full(board.topology.num_vertices, NO_OWNER, dtype=np.int64)
    for port in board.ports:
        value = 0 if port.resource is None else 1 + int(port.resource)
        for v in port.vertices:
            codes[v] = value
    return codes


def record_from_game(
    game: Game,
    perspective: int | None,
    space: ActionSpace,
    options: Sequence[Action] | None = None,
) -> dict[str, np.ndarray]:
    """The information-set record for `perspective`, one unbatched row per
    field. `perspective` defaults to `to_move(game)`, since the mask belongs
    to whoever is to move; `options` defaults to `legal_actions(game)`."""
    state = game._state
    players = state.num_players
    if perspective is None:
        perspective = to_move(game)
    if not 0 <= perspective < players:
        raise ValueError(f"no such player: {perspective}")

    if options is None:
        options = legal_actions(game)
    mask = np.zeros(space.size, dtype=bool)
    for action in options:
        mask[space.index(action)] = True

    own_dev = np.array(
        [held + fresh for held, fresh in zip(state.dev_cards[perspective], state.new_dev_cards[perspective])],
        dtype=np.int64,
    )
    hand_totals = np.array(
        [hand_size(state, s) for s in range(players)], dtype=np.int64
    )
    dev_totals = np.array(
        [dev_count(state, s) for s in range(players)], dtype=np.int64
    )
    award = np.array([award_points(state, s) for s in range(players)], dtype=np.int64)

    # Board-seat order like every other field; `RecordEncoder` rotates it and
    # drops the perspective seat's own entry (exact via `own_hand` above).
    ledger_known = np.asarray(
        [game.ledger.seats[s].known for s in range(players)], dtype=np.int64
    )
    ledger_unknown = np.asarray(
        [game.ledger.seats[s].unknown for s in range(players)], dtype=np.int64
    )

    return {
        "terrain": np.asarray([int(t) for t in state.board.terrain], dtype=np.int64),
        "token": np.asarray(state.board.tokens, dtype=np.int64),
        "port_code": _port_code(state.board),
        "robber": np.int64(state.robber),
        "vertex_owner": np.asarray(state.vertex_owner, dtype=np.int64),
        "vertex_building": np.asarray(state.vertex_building, dtype=np.int64),
        "edge_owner": np.asarray(state.edge_owner, dtype=np.int64),
        "bank": np.asarray(state.bank, dtype=np.int64),
        "knights_played": np.asarray(state.knights_played, dtype=np.int64),
        "award_points": award,
        "longest_road_holder": np.int64(state.longest_road_holder),
        "largest_army_holder": np.int64(state.largest_army_holder),
        "phase": np.int64(int(game.phase)),
        "free_roads": np.int64(game.free_roads),
        "deck_size": np.int64(len(state.deck)),
        "turns": np.int64(game.turns),
        "perspective": np.int64(perspective),
        "own_hand": np.asarray(state.hands[perspective], dtype=np.int64),
        "hand_totals": hand_totals,
        "own_dev": own_dev,
        "dev_totals": dev_totals,
        "ledger_known": ledger_known,
        "ledger_unknown": ledger_unknown,
        "action_mask": mask,
    }


def record_batch(
    games_and_perspectives: Sequence[tuple[Game, int | None]], space: ActionSpace
) -> dict[str, np.ndarray]:
    """`record_from_game` for several positions, stacked on a leading batch
    axis. Every game must share `space`'s topology and player count."""
    if not games_and_perspectives:
        raise ValueError("need at least one position")
    rows = [record_from_game(game, seat, space) for game, seat in games_and_perspectives]
    return {name: np.stack([row[name] for row in rows]) for name in RECORD_FIELDS}
