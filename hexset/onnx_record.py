# SPDX-License-Identifier: GPL-3.0-only
"""The information-set record: the boundary between the rules and the model.

A position stated **in the rules' own terms, already filtered to what the
perspective seat may legally know** -- own hand and own development cards
exact, every other seat by count plus `hexset.ledger`'s reconstruction as
the perspective knows it (`PublicLedger.for_seat`), and from contract 8 the
expected counts and odds the perspective's own view gives (`hexset.view.View`).
There is nowhere in it for a hidden card to be. Everything
downstream (one-hot encoding, rotation, scaling) is mechanical and lives in
`hexn.export_onnx.RecordEncoder`, which also rotates the perspective seat to
slot 0; rows here stay in board-seat order. **Everything here is numpy, and
stays that way**: `import hexset.onnx_record` must succeed with torch absent.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .actions import Action, ActionSpace, ActionType, legal_actions
from .board.board import MAX_ROLL, MIN_ROLL, Board
from .board.terrain import NUM_RESOURCES
from .cards import NUM_DEV_CARDS
from .devcards import dev_count
from .economy import hand_size
from .cards import DevCard
from .encoding import AGE_SLOTS, StaticGraph, dev_age_rows, expected_hands, knights_needed_all
from .game import Game, roll_odds, to_move
from .roads import reach_all, roads_to_take_all
from .state import NO_OWNER
from .victory import award_points
from .view import View

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
# Contract 7 adds `roads_to_take` and `to_move` to contract 6's record.
# Contract 8 adds `reach`, `own_matured`, `dev_card_played`, `expected_hand`,
# `dev_ages`, `knights_needed` and `unseen_dev`, and keeps every
# earlier field: a contract-6 or -7 graph declares a subset of these fields and
# is still served (`hexset.server.constants.RECORD_CONTRACTS`). Contract 8.1
# adds the table's rules (`winning_points`, `discard_limit`, `friendly_robber`,
# `balanced_dice`) and the next roll's odds (`roll_odds`), so a contract-8
# graph is still served beside it. The record is int64 throughout but for
# `expected_hand`, `dev_ages` and `roll_odds` (float32) and `action_mask` (bool).
CONTRACT_VERSION = "8.1"

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
    "reach",
    "own_matured",
    "dev_card_played",
    "expected_hand",
    "dev_ages",
    "knights_needed",
    "unseen_dev",
    "winning_points",
    "discard_limit",
    "friendly_robber",
    "balanced_dice",
    "roll_odds",
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
        "reach": (players, graph.num_vertices),
        "own_matured": (NUM_DEV_CARDS,),
        "dev_card_played": (),
        "expected_hand": (players, NUM_RESOURCES),
        "dev_ages": (players, AGE_SLOTS),
        "knights_needed": (players,),
        "unseen_dev": (NUM_DEV_CARDS,),
        "winning_points": (),
        "discard_limit": (),
        "friendly_robber": (),
        "balanced_dice": (),
        "roll_odds": (MAX_ROLL - MIN_ROLL + 1,),
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
    view = View.from_game(game, perspective)
    unseen = view.unseen_dev_cards()
    odds = dict(roll_odds(game))
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
        # Contract 8, board-seat order like the rest. `reach`: per seat, per
        # vertex, the new roads its network needs to touch it, -1 unreachable
        # (`roads.reach_all`).
        "reach": reach_all(state),
        "own_matured": state.dev_cards[perspective],
        "dev_card_played": game.dev_card_played,
        # Each seat's expected count of each resource on the perspective's
        # view (`View.expected_counts`); its own row exact.
        "expected_hand": expected_hands(view),
        # Each seat's longest-held cards' ages, oldest first, -1 past its cards.
        "dev_ages": dev_age_rows(state),
        # Knights each seat must still play for Largest Army, -1 out of reach.
        "knights_needed": knights_needed_all(state, perspective, unseen[DevCard.KNIGHT]),
        "unseen_dev": unseen,
        # Contract 8.1: the table's rules, and each sum 2..12's chance on the
        # next roll of the seat on turn (`game.roll_odds`), 0 for a sum the
        # balanced dice deck has run out of.
        "winning_points": state.rules.winning_points,
        "discard_limit": state.rules.discard_limit,
        "friendly_robber": int(state.rules.friendly_robber),
        "balanced_dice": int(state.rules.balanced_dice),
        "roll_odds": [odds.get(total, 0.0) for total in range(MIN_ROLL, MAX_ROLL + 1)],
        "mask": indices,
    }


#: The fields `_row` states, in `RECORD_FIELDS` order; those of them that are
#: one number a row; and each one's dtype, int64 but for the float32 few.
_ROW_FIELDS: tuple[str, ...] = tuple(
    name for name in RECORD_FIELDS if name not in BOARD_FIELDS and name != "action_mask"
)
_SCALAR_FIELDS = frozenset({
    "robber", "longest_road_holder", "largest_army_holder", "phase", "free_roads",
    "deck_size", "turns", "to_move", "perspective", "dev_card_played",
    "winning_points", "discard_limit", "friendly_robber", "balanced_dice",
})
_DTYPES: dict[str, type] = {
    name: np.float32 if name in ("expected_hand", "dev_ages", "roll_odds") else np.int64
    for name in _ROW_FIELDS
}


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
        out[name] = (_DTYPES[name](row[name]) if name in _SCALAR_FIELDS
                     else np.asarray(row[name], dtype=_DTYPES[name]))
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
        out[name] = np.asarray([row[name] for row in built], dtype=_DTYPES[name])
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
