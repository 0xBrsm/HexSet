from __future__ import annotations

import pytest
from helpers import ROLL, a_vertex_touching, mini_board

from catan.board.terrain import Resource
from catan.state import (
    NO_OWNER,
    Building,
    can_place_road,
    can_place_settlement,
    gold_claims,
    new_game,
    place_road,
    place_settlement,
    production,
    upgrade_to_city,
)

def test_new_game_starts_empty_with_robber_on_desert():
    state = new_game(mini_board(), 3)
    assert state.robber == 0
    assert set(state.vertex_owner) == {NO_OWNER}
    assert set(state.edge_owner) == {NO_OWNER}
    assert state.hands == [[0] * 5 for _ in range(3)]


@pytest.mark.parametrize("players", [0, 1, 7])
def test_unsupported_player_counts_rejected(players):
    with pytest.raises(ValueError):
        new_game(mini_board(), players)


def test_settlement_blocks_its_neighbours():
    state = new_game(mini_board(), 2)
    v = a_vertex_touching(state.board, 1)
    place_settlement(state, 0, v, connected=False)

    assert state.vertex_building[v] == Building.SETTLEMENT
    assert not can_place_settlement(state, 1, v, connected=False)
    for n in state.board.topology.vertex_neighbors[v]:
        assert not can_place_settlement(state, 1, n, connected=False)


def test_settlement_needs_a_road_outside_setup():
    state = new_game(mini_board(), 2)
    v = a_vertex_touching(state.board, 1)

    assert not can_place_settlement(state, 0, v, connected=True)
    assert can_place_settlement(state, 0, v, connected=False)


def test_road_requires_connection():
    state = new_game(mini_board(), 2)
    topology = state.board.topology
    v = a_vertex_touching(state.board, 1)

    near = topology.vertex_edges[v][0]
    assert not can_place_road(state, 0, near)

    place_settlement(state, 0, v, connected=False)
    assert can_place_road(state, 0, near)
    place_road(state, 0, near)
    assert not can_place_road(state, 1, near)


def test_road_extends_from_an_existing_road():
    state = new_game(mini_board(), 2)
    topology = state.board.topology
    v0 = a_vertex_touching(state.board, 1)
    place_settlement(state, 0, v0, connected=False)

    e0 = topology.vertex_edges[v0][0]
    place_road(state, 0, e0)
    v1 = next(v for v in topology.edges[e0] if v != v0)
    e1 = next(e for e in topology.vertex_edges[v1] if e != e0)
    assert can_place_road(state, 0, e1)


def test_opponent_building_blocks_road_continuation():
    state = new_game(mini_board(), 2)
    topology = state.board.topology

    v0 = a_vertex_touching(state.board, 1)
    place_settlement(state, 0, v0, connected=False)
    e0 = topology.vertex_edges[v0][0]
    place_road(state, 0, e0)
    v1 = next(v for v in topology.edges[e0] if v != v0)
    e1 = next(e for e in topology.vertex_edges[v1] if e != e0)
    place_road(state, 0, e1)
    v2 = next(v for v in topology.edges[e1] if v != v1)

    place_settlement(state, 1, v2, connected=False)
    beyond = [e for e in topology.vertex_edges[v2] if e != e1]
    assert beyond
    for e in beyond:
        assert not can_place_road(state, 0, e)


def test_settlement_then_city_doubles_yield():
    state = new_game(mini_board(), 2)
    v = a_vertex_touching(state.board, 1)
    place_settlement(state, 0, v, connected=False)

    assert production(state, ROLL)[0][Resource.WOOD] == 1
    upgrade_to_city(state, 0, v)
    assert production(state, ROLL)[0][Resource.WOOD] == 2


def test_yield_scales_with_adjacent_producing_hexes():
    state = new_game(mini_board(), 2)
    v = a_vertex_touching(state.board, 3)
    place_settlement(state, 0, v, connected=False)
    assert production(state, ROLL)[0][Resource.WOOD] == 3


def test_robber_blocks_its_hex():
    state = new_game(mini_board(), 2)
    v = a_vertex_touching(state.board, 1)
    place_settlement(state, 0, v, connected=False)
    blocked = state.board.topology.vertex_hexes[v][0]

    state.robber = blocked
    assert production(state, ROLL)[0][Resource.WOOD] == 0


def test_wrong_roll_and_desert_produce_nothing():
    state = new_game(mini_board(), 2)
    v = a_vertex_touching(state.board, 1)
    place_settlement(state, 0, v, connected=False)

    assert production(state, ROLL + 1) == [[0] * 5, [0] * 5]

    desert_vertex = next(
        v for v, hexes in enumerate(state.board.topology.vertex_hexes) if hexes == (0,)
    )
    state2 = new_game(state.board, 2)
    place_settlement(state2, 0, desert_vertex, connected=False)
    assert production(state2, ROLL) == [[0] * 5, [0] * 5]


def test_gold_yields_a_choice_not_a_resource():
    state = new_game(mini_board(gold=True), 2)
    v = a_vertex_touching(state.board, 2)
    place_settlement(state, 0, v, connected=False)

    assert production(state, ROLL) == [[0] * 5, [0] * 5]
    assert gold_claims(state, ROLL)[0] == 2
