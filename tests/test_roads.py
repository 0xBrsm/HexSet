# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

from helpers import mini_board

from hexset.board.board import random_base_board
from hexset.roads import ROADS_TO_TAKE_CAP, longest_road, roads_to_take, roads_to_take_all
from hexset.state import MAX_ROADS, NO_OWNER, Building, new_game, road_count


def a_game(players: int = 2):
    return new_game(mini_board(), players)


def lay(state, player, edges):
    """Set roads directly, bypassing the connectivity rule under test elsewhere."""
    for e in edges:
        state.edge_owner[e] = player


def occupy(state, player, vertex, building=Building.SETTLEMENT):
    state.vertex_owner[vertex] = player
    state.vertex_building[vertex] = building


def a_path(state, length: int, start_edge: int | None = None) -> list[int]:
    """A chain of connected edges, walked greedily from one endpoint."""
    topology = state.board.topology
    edge = 0 if start_edge is None else start_edge
    path = [edge]
    cursor = topology.edges[edge][1]
    while len(path) < length:
        step = next(
            e
            for e in topology.vertex_edges[cursor]
            if e not in path
        )
        path.append(step)
        a, b = topology.edges[step]
        cursor = b if a == cursor else a
    return path


def test_a_chain_counts_every_segment():
    state = a_game()
    path = a_path(state, 4)
    lay(state, 0, path)
    assert longest_road(state, 0) == 4


def test_a_loop_counts_in_full():
    state = a_game()
    ring = state.board.topology.hex_edges[3]
    lay(state, 0, ring)
    assert longest_road(state, 0) == 6


def test_branches_do_not_stack():
    state = a_game()
    topology = state.board.topology
    junction = next(
        v for v in range(topology.num_vertices) if len(topology.vertex_edges[v]) == 3
    )
    spokes = topology.vertex_edges[junction]
    lay(state, 0, spokes)

    # A route through one junction can use two spokes, never all three.
    assert longest_road(state, 0) == 2


def test_disconnected_networks_do_not_add_up():
    state = a_game()
    topology = state.board.topology
    first = a_path(state, 3)
    far = next(
        e
        for e in range(topology.num_edges)
        if e not in first
        and not set(topology.edges[e]) & {v for f in first for v in topology.edges[f]}
    )
    lay(state, 0, first)
    lay(state, 0, [far])

    assert longest_road(state, 0) == 3


def test_opponent_building_breaks_a_route():
    state = a_game()
    topology = state.board.topology
    path = a_path(state, 4)
    lay(state, 0, path)

    shared = set(topology.edges[path[1]]) & set(topology.edges[path[2]])
    occupy(state, 1, shared.pop())

    assert longest_road(state, 0) == 2


def test_your_own_building_does_not_break_a_route():
    state = a_game()
    topology = state.board.topology
    path = a_path(state, 4)
    lay(state, 0, path)

    shared = set(topology.edges[path[1]]) & set(topology.edges[path[2]])
    occupy(state, 0, shared.pop())

    assert longest_road(state, 0) == 4


def test_an_extra_edge_counts_as_built_without_touching_the_state():
    state = a_game()
    path = a_path(state, 4)
    lay(state, 0, path[:3])
    before = list(state.edge_owner)
    assert longest_road(state, 0, extra_edge=path[3]) == 4
    assert longest_road(state, 0, extra_edge=path[0]) == 3      # already the player's
    assert state.edge_owner == before
    lay(state, 0, path[3:])
    assert longest_road(state, 0) == 4


# --- roads_to_take: the fewest new roads that take Longest Road, 0 for the holder


def needing(state, lengths, holder=NO_OWNER):
    """Set the engine's cached lengths and holder directly, so a test reads
    one seat's search against a length to beat it does not have to build."""
    state.road_lengths[:] = lengths
    state.longest_road_holder = holder
    return state


def test_the_holder_needs_no_roads():
    state = needing(a_game(), [6, 5], holder=0)
    assert roads_to_take(state, 0) == 0


def test_an_unheld_card_is_taken_at_five():
    state = a_game()
    path = a_path(state, 5)
    lay(state, 0, path[:4])
    needing(state, [4, 0])
    assert roads_to_take(state, 0) == 1
    assert roads_to_take(state, 1) == ROADS_TO_TAKE_CAP     # no network to build from


def test_one_road_joining_two_routes_counts_both():
    state = a_game()
    path = a_path(state, 7)
    lay(state, 0, path[:3] + path[4:])
    needing(state, [3, 6], holder=1)
    # 3 + the gap + 3 = 7 beats 6: one road, not the four a bare extension needs.
    assert roads_to_take(state, 0) == 1


def test_an_opponent_building_at_the_gap_stops_the_join():
    state = a_game()
    topology = state.board.topology
    path = a_path(state, 7)
    lay(state, 0, path[:3] + path[4:])
    gap_end = (set(topology.edges[path[3]]) & set(topology.edges[path[4]])).pop()
    occupy(state, 1, gap_end)
    needing(state, [3, 6], holder=1)
    assert roads_to_take(state, 0) > 1


def test_four_or_more_reads_as_the_cap():
    state = a_game()
    lay(state, 0, a_path(state, 1))
    needing(state, [1, 8], holder=1)
    assert roads_to_take(state, 0) == ROADS_TO_TAKE_CAP


def test_no_pieces_left_is_out_of_reach():
    state = new_game(random_base_board(random.Random(0)), 2)
    topology = state.board.topology
    path = a_path(state, 5)
    lay(state, 0, path[:4])
    # The other eleven pieces as single roads touching neither the route nor
    # each other, so the route stays 4 and only the piece count changes.
    taken = {v for e in path for v in topology.edges[e]}
    spare = []
    for e in range(topology.num_edges):
        ends = set(topology.edges[e])
        if len(spare) < MAX_ROADS - 4 and not ends & taken:
            spare.append(e)
            taken |= ends
    lay(state, 0, spare)
    needing(state, [4, 0])
    assert road_count(state, 0) == MAX_ROADS
    assert longest_road(state, 0) == 4
    assert roads_to_take(state, 0) == ROADS_TO_TAKE_CAP


def test_a_new_road_is_seen_through_the_cache():
    state = a_game()
    path = a_path(state, 6)
    lay(state, 0, path[:3])
    needing(state, [3, 5], holder=1)
    assert roads_to_take(state, 0) == 3
    lay(state, 0, path[3:4])
    needing(state, [4, 5], holder=1)
    assert roads_to_take(state, 0) == 2
    assert roads_to_take_all(state) == (2, 0)
