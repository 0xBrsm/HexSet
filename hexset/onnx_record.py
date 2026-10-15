# SPDX-License-Identifier: GPL-3.0-only
"""The information-set record: the boundary between the rules and the model.

A position stated **in the rules' own terms, already filtered to what the
perspective seat may legally know** -- own hand and own development cards
exact, every other seat by count plus `hexset.ledger`'s reconstruction as
the perspective knows it (`PublicLedger.for_seat`). There is nowhere in it for a hidden card to be. Everything
downstream (one-hot encoding, rotation, scaling) is mechanical and lives in
`hexn.export_onnx.RecordEncoder`, which also rotates the perspective seat to
slot 0; rows here stay in board-seat order. **Everything here is numpy, and
stays that way**: `import hexset.onnx_record` must succeed with torch absent.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .actions import Action, ActionSpace, ActionType, legal_actions
from .board.board import Board
from .board.terrain import NUM_RESOURCES
from .cards import NUM_DEV_CARDS
from .devcards import dev_count
from .economy import hand_size
from .encoding import StaticGraph
from .game import Game, to_move
from .roads import roads_to_take_all
from .state import NO_OWNER
from .victory import award_points

__all__ = [
    "CONTRACT_VERSION",
    "RECORD_FIELDS",
    "record_shapes",
    "NO_MOVE",
    "record_from_game",
    "record_rows",
    "record_batch",
]


# The record/graph contract, read off an exported graph's metadata. Bump only
# when `RECORD_FIELDS`, the export's output tuple, or the action space changes.
# Contract 7 adds `roads_to_take` and `to_move` to contract 6's record; a
# contract-6 graph declares a subset of these fields and is still served
# (`hexset.server.constants.RECORD_CONTRACTS`).
CONTRACT_VERSION = "7"

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
    "roads_to_take",
    "to_move",
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
        "roads_to_take": (players,),
        "to_move": (),
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


#: The fields a board fixes for the whole game, worked out once per board.
BOARD_FIELDS: tuple[str, ...] = ("terrain", "token", "port_code")
_board_rows: dict[int, tuple[Board, dict[str, np.ndarray]]] = {}


def _board_fields(board: Board) -> dict[str, np.ndarray]:
    """`BOARD_FIELDS` for `board`, read-only and shared by every record of
    it. Keyed by identity and held beside the board, so a recycled id is
    never mistaken for it."""
    entry = _board_rows.get(id(board))
    if entry is None or entry[0] is not board:
        rows = {
            "terrain": np.asarray([int(t) for t in board.terrain], dtype=np.int64),
            "token": np.asarray(board.tokens, dtype=np.int64),
            "port_code": _port_code(board),
        }
        for row in rows.values():
            row.flags.writeable = False
        if len(_board_rows) >= 64:
            _board_rows.clear()
        entry = _board_rows[id(board)] = (board, rows)
    return entry[1]


#: The mask of a record whose perspective seat has no move of its own -- a
#: value-only row for a seat that is not acting. The graph normalises its
#: prior over the legal entries, so a mask needs at least one, and this one is
#: the same for every position: it says nothing about any seat's cards.
NO_MOVE = Action(ActionType.END_TURN)


def _row(
    game: Game,
    perspective: int | None,
    space: ActionSpace,
    options: Sequence[Action] | None,
    indices: Sequence[int] | None,
) -> dict:
    """One record's fields past `BOARD_FIELDS`, as plain ints and lists, and
    under `"mask"` the action indices the mask sets: what `record_from_game`
    and `record_rows` both turn into arrays."""
    state = game._state
    players = state.num_players
    if perspective is None:
        perspective = to_move(game)
    if not 0 <= perspective < players:
        raise ValueError(f"no such player: {perspective}")
    if indices is None:
        if options is None:
            options = legal_actions(game, perspective) or (NO_MOVE,)
        indices = [space.index(action) for action in options]

    # Board-seat order like every other field; `RecordEncoder` rotates it and
    # drops the perspective seat's own entry (exact via `own_hand` below).
    rows = game.ledger.for_seat(perspective).seats
    seats = range(players)
    return {
        "robber": state.robber,
        "vertex_owner": state.vertex_owner,
        "vertex_building": state.vertex_building,
        "edge_owner": state.edge_owner,
        "bank": state.bank,
        "knights_played": state.knights_played,
        "award_points": [award_points(state, s) for s in seats],
        "longest_road_holder": state.longest_road_holder,
        "largest_army_holder": state.largest_army_holder,
        "phase": int(game.phase),
        "free_roads": game.free_roads,
        "deck_size": len(state.deck),
        "turns": game.turns,
        # Board-seat order, each at most `roads.ROADS_TO_TAKE_CAP`; the graph scales it.
        "roads_to_take": roads_to_take_all(state),
        "to_move": to_move(game),
        "perspective": perspective,
        "own_hand": state.hands[perspective],
        "hand_totals": [hand_size(state, s) for s in seats],
        "own_dev": [held + fresh for held, fresh in
                    zip(state.dev_cards[perspective], state.new_dev_cards[perspective])],
        "dev_totals": [dev_count(state, s) for s in seats],
        "ledger_known": [rows[s].known for s in seats],
        "ledger_unknown": [rows[s].unknown for s in seats],
        "mask": indices,
    }


#: The fields `_row` states, in `RECORD_FIELDS` order, every one int64, and
#: those of them that are one number a row.
_ROW_FIELDS: tuple[str, ...] = tuple(
    name for name in RECORD_FIELDS if name not in BOARD_FIELDS and name != "action_mask"
)
_SCALAR_FIELDS = frozenset({
    "robber", "longest_road_holder", "largest_army_holder", "phase", "free_roads",
    "deck_size", "turns", "to_move", "perspective",
})


def record_from_game(
    game: Game,
    perspective: int | None,
    space: ActionSpace,
    options: Sequence[Action] | None = None,
) -> dict[str, np.ndarray]:
    """The information-set record for `perspective`, one unbatched row per
    field. `perspective` defaults to `to_move(game)`.

    `options` defaults to the perspective seat's own legal actions
    (`legal_actions(game, perspective)`), and to `NO_MOVE` alone where it has
    none, so the mask never encodes another seat's moves: they would read
    that seat's hand."""
    row = _row(game, perspective, space, options, None)
    out = {name: field.copy() for name, field in _board_fields(game._state.board).items()}
    for name in _ROW_FIELDS:
        out[name] = (np.int64(row[name]) if name in _SCALAR_FIELDS
                     else np.asarray(row[name], dtype=np.int64))
    mask = np.zeros(space.size, dtype=bool)
    mask[row["mask"]] = True
    out["action_mask"] = mask
    return {name: out[name] for name in RECORD_FIELDS}


def record_rows(
    rows: Sequence[tuple[Game, int | None, Sequence[Action] | None]],
    space: ActionSpace,
    indices: Sequence[Sequence[int]] | None = None,
) -> dict[str, np.ndarray]:
    """`record_from_game` for several `(game, perspective, options)` rows,
    stacked on a leading batch axis, built straight into the batch.
    `indices`, one list per row, are its options' action indices where the
    caller has them already (`space.index`); they set the mask in their
    place. Every game must share `space`'s topology and player count."""
    if not rows:
        raise ValueError("need at least one position")
    built = [
        _row(game, seat, space, options, None if indices is None else indices[i])
        for i, (game, seat, options) in enumerate(rows)
    ]
    n = len(rows)
    boards = [game._state.board for game, _, _ in rows]
    out: dict[str, np.ndarray] = {}
    if all(board is boards[0] for board in boards):
        for name, field in _board_fields(boards[0]).items():
            out[name] = np.tile(field, (n, 1))
    else:
        for name in BOARD_FIELDS:
            out[name] = np.stack([_board_fields(board)[name] for board in boards])
    for name in _ROW_FIELDS:
        out[name] = np.asarray([row[name] for row in built], dtype=np.int64)
    mask = np.zeros((n, space.size), dtype=bool)
    for i, row in enumerate(built):
        mask[i, row["mask"]] = True
    out["action_mask"] = mask
    return {name: out[name] for name in RECORD_FIELDS}


def record_batch(
    games_and_perspectives: Sequence[tuple[Game, int | None]], space: ActionSpace
) -> dict[str, np.ndarray]:
    """`record_from_game` for several positions, stacked on a leading batch
    axis. Every game must share `space`'s topology and player count."""
    if not games_and_perspectives:
        raise ValueError("need at least one position")
    return record_rows([(game, seat, None) for game, seat in games_and_perspectives], space)
