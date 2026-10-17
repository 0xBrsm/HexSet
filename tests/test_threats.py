# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

from helpers import give, grow_road, independent_vertices

from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.cards import DevCard
from hexset.roads import longest_road
from hexset.state import NO_OWNER, Building, new_game
from hexset.threats import spots, threats
from hexset.victory import update_longest_road

VICTIM, RIVAL = 0, 1


def a_table(seed: int = 3):
    """Four seats, each with one settlement, apart."""
    board = random_base_board(random.Random(seed))
    state = new_game(board, 4)
    for seat, v in enumerate(independent_vertices(board, 4)):
        state.vertex_owner[v] = seat
        state.vertex_building[v] = Building.SETTLEMENT
    return state


def a_route_held(state, length: int):
    """VICTIM with a route of `length` and Longest Road."""
    grow_road(state, VICTIM, length)
    update_longest_road(state)
    assert state.longest_road_holder == VICTIM
    return state


def test_a_road_that_takes_longest_road_is_a_steal_and_says_what_it_lacks():
    state = a_table()
    grow_road(state, VICTIM, 5)
    grow_road(state, RIVAL, 5)      # one more road lengthens it to 6
    state.road_lengths[:] = [longest_road(state, p) for p in range(4)]
    state.longest_road_holder = VICTIM
    steals = [t for t in threats(state, VICTIM) if t.kind == "steal"]
    assert steals and all(t.rival == RIVAL and t.build == "road" and t.cost.award == "longest_road" for t in steals)
    assert all(t.lacks == (1, 1, 0, 0, 0) for t in steals)          # an empty hand: one wood, one brick short
    give(state, RIVAL, Resource.WOOD)
    give(state, RIVAL, Resource.BRICK)
    assert all(t.short == 0 for t in threats(state, VICTIM) if t.kind == "steal")


def test_hands_are_whatever_the_caller_holds():
    state = a_table()
    grow_road(state, VICTIM, 5)
    grow_road(state, RIVAL, 5)
    state.road_lengths[:] = [longest_road(state, p) for p in range(4)]
    state.longest_road_holder = VICTIM
    give(state, RIVAL, Resource.WOOD)
    give(state, RIVAL, Resource.BRICK)
    known = [[0] * 5 for _ in range(4)]       # the victim can certify nothing the rival holds
    assert all(t.short == 2 for t in threats(state, VICTIM, known) if t.kind == "steal")


def test_settling_where_the_victim_could_is_a_block():
    state = a_table()
    grow_road(state, VICTIM, 2)
    topology = state.board.topology
    open_ = sorted(spots(state, VICTIM))
    assert open_
    # A rival road ending at one of those spots, on an edge the victim does not need to reach it, so the
    # rival can settle it.
    for u, edge in ((u, e) for u in open_ for e in topology.vertex_edges[u] if state.edge_owner[e] == NO_OWNER):
        state.edge_owner[edge] = RIVAL
        if u in spots(state, VICTIM):
            break
        state.edge_owner[edge] = NO_OWNER
    else:
        raise AssertionError("no spot a rival road can reach beside the victim's")
    state.road_lengths[:] = [longest_road(state, p) for p in range(4)]
    blocks = [t for t in threats(state, VICTIM) if t.rival == RIVAL and t.build == "settlement" and t.at == u]
    assert blocks and blocks[0].kind == "block" and u in blocks[0].cost.spots


def test_a_settlement_that_cuts_the_held_route_below_five_steals_it():
    state = a_route_held(a_table(), 6)
    topology = state.board.topology
    own = {e for e, o in enumerate(state.edge_owner) if o == VICTIM}
    # A vertex inside the route, free and settleable once a rival road reaches it.
    inner = [v for v in range(topology.num_vertices)
             if sum(e in own for e in topology.vertex_edges[v]) == 2 and state.vertex_owner[v] == NO_OWNER
             and all(state.vertex_building[n] == Building.NONE for n in topology.vertex_neighbors[v])
             and any(state.edge_owner[e] == NO_OWNER for e in topology.vertex_edges[v])]
    cut = next(v for v in inner
               if longest_road(state, VICTIM) - _route_after_cut(state, v) >= 2)
    edge = next(e for e in topology.vertex_edges[cut] if state.edge_owner[e] == NO_OWNER)
    state.edge_owner[edge] = RIVAL
    state.road_lengths[:] = [longest_road(state, p) for p in range(4)]
    hit = [t for t in threats(state, VICTIM) if t.build == "settlement" and t.at == cut]
    assert hit and hit[0].kind == "steal" and hit[0].cost.award == "longest_road" and hit[0].cost.route >= 2


def _route_after_cut(state, v) -> int:
    owner, building = state.vertex_owner[v], state.vertex_building[v]
    state.vertex_owner[v], state.vertex_building[v] = RIVAL, Building.SETTLEMENT
    try:
        return longest_road(state, VICTIM)
    finally:
        state.vertex_owner[v], state.vertex_building[v] = owner, building


def test_a_knight_takes_largest_army_only_from_a_rival_holding_a_card():
    state = a_table()
    state.knights_played[VICTIM] = 3
    state.knights_played[RIVAL] = 3
    state.largest_army_holder = VICTIM
    assert not [t for t in threats(state, VICTIM) if t.build == "knight"]
    state.dev_cards[RIVAL][DevCard.KNIGHT] += 1
    knights = [t for t in threats(state, VICTIM) if t.build == "knight"]
    assert len(knights) == 1 and knights[0].kind == "steal" and knights[0].cost.award == "largest_army"


def test_a_knight_that_raises_the_bar_is_a_block():
    state = a_table()
    state.knights_played[VICTIM] = 2
    state.knights_played[RIVAL] = 2
    state.dev_cards[RIVAL][DevCard.KNIGHT] += 1
    knights = [t for t in threats(state, VICTIM) if t.build == "knight"]
    assert len(knights) == 1 and knights[0].kind == "block" and knights[0].cost.knights == 1


def test_every_threat_costs_something_and_the_position_is_untouched():
    state = a_route_held(a_table(), 5)
    grow_road(state, RIVAL, 4)
    state.road_lengths[:] = [longest_road(state, p) for p in range(4)]
    state.longest_road_holder = VICTIM
    edges, vertices = list(state.edge_owner), list(state.vertex_owner)
    found = threats(state, VICTIM)
    assert found and all(t.cost for t in found)
    assert state.edge_owner == edges and state.vertex_owner == vertices and state.longest_road_holder == VICTIM
