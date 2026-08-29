"""The position, stated in the rules' own terms and filtered to what one seat
may legally know — the interface between the engine and any way of choosing
an action, whether that is `search2.py` or a `.onnx` file.

See `docs/onnx-contract-v2.md` for the field-by-field contract this mirrors.
This module says what is true and visible; it never says how a network reads
it, so it imports nothing model-shaped and knows no feature layout.

Filtering here is load-bearing, not incidental: own hand and dev cards are
exact, everyone else contributes a total alone. A caller that wants more than
this record describes is asking the engine to leak a hidden card.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .actions import Action, ActionSpace, ActionType
from .board.terrain import NUM_RESOURCES
from .devcards import holdings
from .game import Game
from .victory import award_points

NUM_PAIRS = NUM_RESOURCES * NUM_RESOURCES


def action_mask(space: ActionSpace, options: Sequence[Action]) -> np.ndarray:
    """Mark already-enumerated actions without enumerating them again."""
    mask = np.zeros(space.size, dtype=bool)
    for action in options:
        mask[space.index(action)] = True
    return mask


def pair_index(give: Sequence[int], want: Sequence[int]) -> int:
    """The flat pair slot for a one-for-one offer's two one-hot bundles."""
    return give.index(1) * NUM_RESOURCES + want.index(1)


def pair_mask(options: Sequence[Action]) -> np.ndarray:
    """Which one-for-one offers were legal, as a flat `(NUM_PAIRS,)` bool."""
    mask = np.zeros(NUM_PAIRS, dtype=bool)
    for option in options:
        if option.type is ActionType.PROPOSE_TRADE:
            mask[pair_index(option.give, option.want)] = True
    return mask


def _port_code(board, num_vertices: int) -> np.ndarray:
    code = np.full(num_vertices, -1, dtype=np.int64)
    for port in board.ports:
        value = 0 if port.resource is None else 1 + int(port.resource)
        for v in port.vertices:
            code[v] = value
    return code


def build_record(
    game: Game, seat: int, options: Sequence[Action], space: ActionSpace
) -> dict[str, np.ndarray]:
    """`Game + seat + options -> dict[str, np.ndarray]`, per
    `docs/onnx-contract-v2.md`. `options` is normally `legal_actions(game)`
    (after any offer-budget trim), the same list `action_mask`/`pair_mask`
    are built from.
    """
    state = game.state
    board = state.board
    players = state.num_players

    own_dev = holdings(state, seat)
    return {
        # Board.
        "terrain": np.array(board.terrain, dtype=np.int64),
        "token": np.array(board.tokens, dtype=np.int64),
        "port_code": _port_code(board, board.topology.num_vertices),
        # Position.
        "robber": np.array(state.robber, dtype=np.int64),
        "vertex_owner": np.array(state.vertex_owner, dtype=np.int64),
        "vertex_building": np.array(state.vertex_building, dtype=np.int64),
        "edge_owner": np.array(state.edge_owner, dtype=np.int64),
        "bank": np.array(state.bank, dtype=np.int64),
        "knights_played": np.array(state.knights_played, dtype=np.int64),
        "award_points": np.array(
            [award_points(state, p) for p in range(players)], dtype=np.int64
        ),
        "longest_road_holder": np.array(state.longest_road_holder, dtype=np.int64),
        "largest_army_holder": np.array(state.largest_army_holder, dtype=np.int64),
        "phase": np.array(int(game.phase), dtype=np.int64),
        "free_roads": np.array(game.free_roads, dtype=np.int64),
        "deck_size": np.array(len(state.deck), dtype=np.int64),
        "turns": np.array(game.turns, dtype=np.int64),
        "perspective": np.array(seat, dtype=np.int64),
        # Information set, in board-seat order -- exact for `seat`, totals only
        # for everyone else.
        "own_hand": np.array(state.hands[seat], dtype=np.int64),
        "hand_totals": np.array(
            [sum(state.hands[p]) for p in range(players)], dtype=np.int64
        ),
        "own_dev": np.array(own_dev, dtype=np.int64),
        "dev_totals": np.array(
            [
                sum(state.dev_cards[p]) + sum(state.new_dev_cards[p])
                for p in range(players)
            ],
            dtype=np.int64,
        ),
        # Legality.
        "action_mask": action_mask(space, options),
        "pair_mask": pair_mask(options),
    }
