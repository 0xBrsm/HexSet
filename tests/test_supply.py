# SPDX-License-Identifier: GPL-3.0-only
"""The piece supply: 15 roads, 5 settlements, 4 cities a player, enforced in
placement legality so every path that builds reads one rule."""

from __future__ import annotations

import random

from helpers import independent_vertices

from hexset.actions import ActionType, legal_actions
from hexset.board.board import random_base_board
from hexset.game import Phase, start
from hexset.state import (
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
    assert can_place_settlement(state, 1, spots[-1], connected=False)


def test_a_fifth_city_is_refused_and_a_fourth_is_not():
    state = new_game(random_base_board(random.Random(0)), 4)
    spots = independent_vertices(state.board, MAX_CITIES + 1)
    _settle(state, 0, spots[: MAX_CITIES - 1], Building.CITY)
    _settle(state, 0, spots[MAX_CITIES - 1 :])
    assert city_count(state, 0) == MAX_CITIES - 1
    assert can_upgrade_to_city(state, 0, spots[-2])
    state.vertex_building[spots[-2]] = Building.CITY
    assert city_count(state, 0) == MAX_CITIES
    assert not can_upgrade_to_city(state, 0, spots[-1])
    assert state.vertex_building[spots[-1]] == Building.SETTLEMENT


def test_a_sixteenth_road_is_refused_and_a_fifteenth_is_not():
    state = new_game(random_base_board(random.Random(0)), 4)
    topology = state.board.topology
    v = independent_vertices(state.board, 1)[0]
    _settle(state, 0, [v])
    beside = list(topology.vertex_edges[v])
    elsewhere = [e for e in range(topology.num_edges) if e not in beside]
    for e in elsewhere[: MAX_ROADS - 1]:
        state.edge_owner[e] = 0
    assert road_count(state, 0) == MAX_ROADS - 1
    assert can_place_road(state, 0, beside[0])
    state.edge_owner[beside[0]] = 0
    assert road_count(state, 0) == MAX_ROADS
    assert not can_place_road(state, 0, beside[1])
    assert state.edge_owner[beside[1]] == NO_OWNER
    state.vertex_owner[beside_v := topology.edges[beside[1]][1]] = 1
    state.vertex_building[beside_v] = Building.SETTLEMENT
    assert can_place_road(state, 1, beside[1])


def test_legal_actions_stop_offering_a_piece_that_is_not_in_the_box():
    rng = random.Random(3)
    game = start(random_base_board(rng), 4, rng)
    state = game._state
    # Skip setup; hand is deliberately rich so only the supply can bind.
    game.phase = Phase.MAIN
    game.current_player = 0
    state.hands[0] = [10, 10, 10, 10, 10]
    topology = state.board.topology
    spots = independent_vertices(state.board, MAX_SETTLEMENTS)
    _settle(state, 0, spots)
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
    assert ActionType.BUILD_SETTLEMENT not in kinds
    assert ActionType.BUILD_CITY in kinds
    assert ActionType.BUILD_ROAD in kinds
    state.vertex_building[v0] = Building.CITY
    kinds = {a.type for a in legal_actions(game)}
    assert ActionType.BUILD_SETTLEMENT in kinds
