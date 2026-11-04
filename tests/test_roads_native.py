# SPDX-License-Identifier: GPL-3.0-only
"""Compare native integer queries with the independent Python DFS and BFS."""
from __future__ import annotations

import itertools
import pickle
import random
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from hexset import roads
from hexset.board.board import random_base_board
from hexset.state import MAX_ROADS, new_game

native = pytest.importorskip("hexset._road_native")


@dataclass(frozen=True)
class Topology:
    edges: tuple
    num_edges: int
    num_vertices: int
    vertex_edges: tuple


def probe(edges, owners, buildings):
    incident = [[] for _ in buildings]
    for e, pair in enumerate(edges):
        for v in pair:
            incident[v].append(e)
    topology = Topology(edges, len(edges), len(buildings), tuple(tuple(row) for row in incident))
    return SimpleNamespace(board=SimpleNamespace(topology=topology), edge_owner=owners,
                           vertex_owner=buildings)


def compare(state, seat, budget):
    edges = native.topology(state.board.topology.edges, len(state.vertex_owner))
    eo, vo = state.edge_owner, state.vertex_owner
    before = (list(eo), list(vo))
    expected_paths, actual_paths = ["discard me"], ["discard me"]
    expected = roads._longest_with_new_roads_python(state, seat, budget, expected_paths)
    actual = roads.longest_with_new_roads(state, seat, budget, actual_paths)
    assert actual == expected
    assert actual_paths == expected_paths
    assert native.with_new(edges, eo, vo, seat, budget, False) == expected
    assert roads.longest_road(state, seat) == roads._longest_road_python(state, seat)
    for extra in (None, 0, len(eo) - 1):
        assert roads.longest_road(state, seat, extra) == roads._longest_road_python(state, seat, extra)
    for beat in (5, 7, 12):
        want = next((k for k in range(max(1, beat - eo.count(seat)), budget + 1)
                     if expected[k] >= beat), roads.ROADS_TO_TAKE_CAP)
        assert native.to_take(edges, eo, vo, seat, beat, budget, roads.ROADS_TO_TAKE_CAP) == want
    assert (list(eo), list(vo)) == before


def test_every_occupancy_of_a_cycle_with_a_branch():
    # A loop with a branch at a revisited junction: trails are not simple paths.
    edges = ((0, 1), (1, 2), (2, 0), (2, 3))
    for owners in itertools.product((-1, 0, 1), repeat=len(edges)):
        for buildings in itertools.product((-1, 0, 1), repeat=4):
            state = probe(edges, owners, buildings)
            compare(state, 0, 2)


@pytest.mark.parametrize("players", (2, 4))
def test_constructed_board_positions_and_every_seat(players, monkeypatch):
    rng = random.Random(20260909)
    board = random_base_board(rng)
    for position in range(32):
        state = new_game(board, players)
        # Connected, disconnected, blocked and empty networks, within piece counts.
        available = list(range(board.topology.num_edges))
        rng.shuffle(available)
        for seat in range(players):
            for _ in range(rng.randrange(1, 12)):
                state.edge_owner[available.pop()] = seat
        for v in range(board.topology.num_vertices):
            if rng.random() < 0.15:
                state.vertex_owner[v] = rng.randrange(players)
        for seat in range(players):
            compare(state, seat, position % 4)
        args = (board.topology, tuple(state.edge_owner), tuple(state.vertex_owner), players)
        got = native.reach(roads._native_topology(board.topology), args[1], args[2], players, MAX_ROADS)
        with monkeypatch.context() as context:
            context.setattr(roads, "_native", None)
            expected = roads._reach_all.__wrapped__(*args)
        assert got == expected


def test_paths_use_more_than_64_candidate_edges():
    # Arbitrary topologies have no fixed board-size or machine-word mask limit.
    edges = tuple((0, v) for v in range(1, 81))
    state = probe(edges, [-1] * len(edges), [0] + [-1] * 80)
    compare(state, 0, 3)
    expected = roads._longest_with_new_roads_python(state, 0, 3)
    assert expected == [0, 1, 2, 2]


def test_reach_obeys_piece_budget_and_opponent_buildings():
    edges = tuple((v, v + 1) for v in range(20))
    state = probe(edges, [0] + [-1] * 19, [0] + [-1] * 20)
    packed = native.topology(edges, len(state.vertex_owner))
    got = native.reach(packed, state.edge_owner, state.vertex_owner, 2, MAX_ROADS)
    assert got[0] == (0, 0, *range(1, 15), *([-1] * 5))
    assert got[1] == (-1,) * 21  # a seat with no building reaches nothing
    state.vertex_owner[3] = 1
    got = native.reach(packed, state.edge_owner, state.vertex_owner, 2, MAX_ROADS)
    assert got[0] == (0, 0, 1, 2, *([-1] * 17))


def test_road_and_building_changes_do_not_use_stale_native_data():
    edges = ((0, 1), (1, 2), (2, 3))
    state = probe(edges, [0, 0, -1], [-1] * 4)
    assert roads.longest_road(state, 0) == 2
    state.edge_owner[2] = 0
    assert roads.longest_road(state, 0) == 3
    state.vertex_owner[1] = 1
    assert roads.longest_road(state, 0) == 2
    state.vertex_owner[1] = 0
    assert roads.longest_road(state, 0) == 3


def test_packed_adjacency_is_shared_without_entering_pickled_topologies():
    rng = random.Random(20260909)
    first, second = (random_base_board(rng).topology for _ in range(2))
    packed = roads._native_topology(first)
    assert roads._native_topology(first) is packed
    assert roads._native_topology(second) is packed
    restored = pickle.loads(pickle.dumps(first))
    assert restored == first
    assert roads._native_topology(restored) is packed


@pytest.mark.parametrize("args", [
    (((0, 1),), [], [-1, -1]),
    (((0, 2),), [-1], [-1, -1]),
    (((-1, 1),), [-1], [-1, -1]),
    (((0,),), [-1], [-1]),
])
def test_invalid_graphs_raise_without_native_memory_access(args):
    with pytest.raises(ValueError):
        native.longest(native.topology(args[0], len(args[2])), args[1], args[2], 0, None)


def test_noninteger_graph_and_invalid_hypothetical_edge():
    packed = native.topology(((0, 1),), 2)
    with pytest.raises(TypeError):
        native.longest(packed, [object()], [-1, -1], 0, None)
    with pytest.raises(IndexError):
        native.longest(packed, [-1], [-1, -1], 0, 2)
    with pytest.raises(ValueError):
        native.with_new(packed, [-1], [-1, -1], 0, -1, False)
    with pytest.raises(ValueError):
        native.longest(object(), [-1], [-1, -1], 0, None)
