# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from helpers import mini_board

from hexset.roads import longest_road
from hexset.state import Building, new_game


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
