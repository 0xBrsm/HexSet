"""The robber on a board with no desert, and never at sea.

Seafarers: with no desert, "the robber stands beside the game board. As soon
as the first '7' is rolled or the first knight is played, the robber is
placed on any of the terrain hexes." Sea is not terrain.
"""

from __future__ import annotations

import random

import pytest

from hexset.actions import ActionType, legal_actions
from hexset.board.board import make_board
from hexset.board.maps import MINI_LAYOUT
from hexset.board.terrain import Terrain
from hexset.board.topology import build as build_topology
from hexset.encoding import NUM_TERRAIN, encode, encode_batch
from hexset.game import Phase, move_robber_to, start
from hexset.robber import occupants
from hexset.state import OFF_BOARD


def island_board():
    """The mini layout with sea on hex 0 and forest everywhere else: no
    desert."""
    topology = build_topology(MINI_LAYOUT)
    n = topology.num_hexes
    return make_board(
        topology, (Terrain.SEA,) + (Terrain.FOREST,) * (n - 1), (0,) + (6,) * (n - 1)
    )


def a_robber_move():
    """The robber's first move, as a seven leaves it to the roller."""
    game = start(island_board(), 2, random.Random(0))
    game.phase = Phase.ROBBER
    game.resume_phase = Phase.ROLL
    return game


def test_a_board_with_no_desert_starts_the_robber_beside_it():
    game = start(island_board(), 2, random.Random(0))
    assert game.state(0, hidden=False).robber == OFF_BOARD


def test_its_first_move_takes_any_terrain_hex_and_never_the_sea():
    game = a_robber_move()
    hexes = game.state(0, hidden=False).board.num_hexes
    targets = {a.a for a in legal_actions(game) if a.type is ActionType.MOVE_ROBBER}
    assert targets == set(range(1, hexes))
    with pytest.raises(ValueError, match="sea"):
        move_robber_to(game, 0)
    move_robber_to(game, 1)
    assert game.state(0, hidden=False).robber == 1


def test_a_robber_beside_the_board_marks_no_hex():
    game = start(island_board(), 2, random.Random(0))
    robber_column = NUM_TERRAIN + 2
    assert not encode(game, 0).hexes[:, robber_column].any()
    (batched,) = encode_batch([game], [0])
    assert not batched.hexes[:, robber_column].any()


def test_nobody_sits_under_a_robber_beside_the_board():
    game = start(island_board(), 2, random.Random(0))
    state = game.state(0, hidden=False)
    assert occupants(state, state.robber) == ()
