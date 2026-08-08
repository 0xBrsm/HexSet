from __future__ import annotations

from helpers import mini_board

from catan.board.coords import Hex
from catan.roads import longest_road, road_lengths
from catan.state import Building, new_game


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


def test_no_roads_is_zero():
    state = a_game()
    assert longest_road(state, 0) == 0
    assert road_lengths(state) == [0, 0]


def test_a_chain_counts_every_segment():
    state = a_game()
    path = a_path(state, 4)
    lay(state, 0, path)
    assert longest_road(state, 0) == 4


def test_only_the_owner_gets_credit():
    state = a_game()
    lay(state, 0, a_path(state, 3))
    assert longest_road(state, 1) == 0


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

    # Three spokes from one junction: a route can use two of them, never all three.
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


def test_a_blocked_junction_still_anchors_routes_on_both_sides():
    state = a_game()
    topology = state.board.topology
    path = a_path(state, 5)
    lay(state, 0, path)

    shared = set(topology.edges[path[0]]) & set(topology.edges[path[1]])
    occupy(state, 1, shared.pop())

    # One segment either side of the block: the longer is four.
    assert longest_road(state, 0) == 4


def test_search_terminates_on_a_dense_network():
    state = new_game(mini_board(), 2)
    topology = state.board.topology
    centre = topology.hex_index[Hex(0, 0, 0)]
    edges = set(topology.hex_edges[centre])
    for h in topology.hex_neighbors[centre]:
        edges.update(topology.hex_edges[h])
    lay(state, 0, edges)

    assert longest_road(state, 0) >= 6
