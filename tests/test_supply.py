"""The piece supply: 15 roads, 5 settlements, 4 cities a player, enforced in
placement legality so every path that builds reads one rule."""

from __future__ import annotations

import random

from helpers import independent_vertices, mini_board

from catan.actions import ActionType, legal_actions
from catan.board.board import random_base_board
from catan.game import Phase, start
from catan.state import (
    MAX_CITIES,
    MAX_ROADS,
    MAX_SETTLEMENTS,
    NO_OWNER,
    Building,
    can_place_road,
    can_place_settlement,
    can_upgrade_to_city,
    city_count,
    new_game,
    road_count,
    settlement_count,
)


def test_the_supply_is_the_standard_one():
    assert (MAX_ROADS, MAX_SETTLEMENTS, MAX_CITIES) == (15, 5, 4)


def _settle(state, player: int, vertices, building=Building.SETTLEMENT) -> None:
    for v in vertices:
        state.vertex_owner[v] = player
        state.vertex_building[v] = building


def test_a_sixth_settlement_is_refused_and_a_fifth_is_not():
    state = new_game(random_base_board(random.Random(0)), 4)
    spots = independent_vertices(state.board, MAX_SETTLEMENTS + 1)
    _settle(state, 0, spots[: MAX_SETTLEMENTS - 1])
    assert settlement_count(state, 0) == MAX_SETTLEMENTS - 1
    assert can_place_settlement(state, 0, spots[-2], connected=False)
    _settle(state, 0, [spots[-2]])
    assert settlement_count(state, 0) == MAX_SETTLEMENTS
    assert not can_place_settlement(state, 0, spots[-1], connected=False)
    # Another player still may: the supply is per player.
    assert can_place_settlement(state, 1, spots[-1], connected=False)


def test_upgrading_frees_a_settlement_from_the_count():
    state = new_game(random_base_board(random.Random(0)), 4)
    spots = independent_vertices(state.board, MAX_SETTLEMENTS + 1)
    _settle(state, 0, spots[:MAX_SETTLEMENTS])
    assert not can_place_settlement(state, 0, spots[-1], connected=False)
    state.vertex_building[spots[0]] = Building.CITY
    assert settlement_count(state, 0) == MAX_SETTLEMENTS - 1
    assert can_place_settlement(state, 0, spots[-1], connected=False)


def test_a_fifth_city_is_refused_and_a_fourth_is_not():
    state = new_game(random_base_board(random.Random(0)), 4)
    spots = independent_vertices(state.board, MAX_CITIES + 1)
    _settle(state, 0, spots[: MAX_CITIES - 1], Building.CITY)
    _settle(state, 0, spots[MAX_CITIES - 1 :])  # two settlements left to upgrade
    assert city_count(state, 0) == MAX_CITIES - 1
    assert can_upgrade_to_city(state, 0, spots[-2])
    state.vertex_building[spots[-2]] = Building.CITY
    assert city_count(state, 0) == MAX_CITIES
    assert not can_upgrade_to_city(state, 0, spots[-1])
    assert state.vertex_building[spots[-1]] == Building.SETTLEMENT  # still theirs, still a settlement


def test_a_sixteenth_road_is_refused_and_a_fifteenth_is_not():
    state = new_game(random_base_board(random.Random(0)), 4)
    topology = state.board.topology
    v = independent_vertices(state.board, 1)[0]
    _settle(state, 0, [v])
    beside = list(topology.vertex_edges[v])
    # Fill the supply with roads elsewhere, leaving the edges at `v` free.
    elsewhere = [e for e in range(topology.num_edges) if e not in beside]
    for e in elsewhere[: MAX_ROADS - 1]:
        state.edge_owner[e] = 0
    assert road_count(state, 0) == MAX_ROADS - 1
    assert can_place_road(state, 0, beside[0])
    state.edge_owner[beside[0]] = 0
    assert road_count(state, 0) == MAX_ROADS
    assert not can_place_road(state, 0, beside[1])
    assert state.edge_owner[beside[1]] == NO_OWNER
    # Another player's supply is their own.
    state.vertex_owner[beside_v := topology.edges[beside[1]][1]] = 1
    state.vertex_building[beside_v] = Building.SETTLEMENT
    assert can_place_road(state, 1, beside[1])


def test_legal_actions_stop_offering_a_piece_that_is_not_in_the_box():
    rng = random.Random(3)
    game = start(random_base_board(rng), 4, rng)
    state = game.state
    # Skip setup: put the game in the main phase with a rich hand.
    game.phase = Phase.MAIN
    game.current_player = 0
    state.hands[0] = [10, 10, 10, 10, 10]
    topology = state.board.topology
    spots = independent_vertices(state.board, MAX_SETTLEMENTS)
    _settle(state, 0, spots)
    # A two-road path from the first settlement to a vertex two steps away, so
    # a settlement *spot* exists (the distance rule blocks every neighbour of
    # a building) and the only thing standing in the way is the supply.
    def two_step_path():
        for v0 in spots:
            for e1 in topology.vertex_edges[v0]:
                v1 = next(v for v in topology.edges[e1] if v != v0)
                for e2 in topology.vertex_edges[v1]:
                    v2 = next(v for v in topology.edges[e2] if v != v1)
                    if v2 != v0 and state.vertex_building[v2] == Building.NONE and all(
                        state.vertex_building[n] == Building.NONE
                        for n in topology.vertex_neighbors[v2]
                    ):
                        return v0, e1, e2
        raise AssertionError("no distance-2 vertex free of buildings; pick another board seed")

    v0, e1, e2 = two_step_path()
    state.edge_owner[e1] = state.edge_owner[e2] = 0
    kinds = {a.type for a in legal_actions(game)}
    assert ActionType.BUILD_SETTLEMENT not in kinds  # the box is empty
    assert ActionType.BUILD_CITY in kinds  # five settlements, no cities yet
    assert ActionType.BUILD_ROAD in kinds  # two roads of fifteen
    state.vertex_building[v0] = Building.CITY
    kinds = {a.type for a in legal_actions(game)}
    assert ActionType.BUILD_SETTLEMENT in kinds  # a settlement went back in the box


def test_the_mini_board_is_untouched_below_the_caps():
    state = new_game(mini_board(), 2)
    assert settlement_count(state, 0) == city_count(state, 0) == road_count(state, 0) == 0
