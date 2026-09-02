# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

from hexset.actions import Action, ActionType, apply, legal_actions
from hexset.board.board import make_board, random_base_board
from hexset.board.maps import MINI_LAYOUT
from hexset.board.terrain import Resource, Terrain
from hexset.board.topology import build as build_topology
from hexset.game import Phase, start
from hexset.placement import PlacementBot, best, rank, scarce_resources, score
from hexset.state import place_settlement


def test_scarcity_is_read_off_the_board_not_hardcoded():
    board = random_base_board(random.Random(0))
    assert scarce_resources(board) == {Resource.BRICK, Resource.ORE}


def test_a_board_with_one_resource_has_nothing_scarce():
    topology = build_topology(MINI_LAYOUT)
    n = topology.num_hexes
    board = make_board(topology, (Terrain.FOREST,) * n, (4,) * n)
    assert scarce_resources(board) == frozenset()


def test_a_hex_shared_by_two_settlements_is_counted_twice():
    """Both settlements collect from a shared hex, so it really does pay twice."""
    topology = build_topology(MINI_LAYOUT)
    n = topology.num_hexes
    board = make_board(topology, (Terrain.FOREST,) * n, (6,) * n)
    shared = next(
        (a, b, h)
        for a, hexes_a in enumerate(topology.vertex_hexes)
        for b, hexes_b in enumerate(topology.vertex_hexes)
        if a < b and b not in topology.vertex_neighbors[a]
        for h in set(hexes_a) & set(hexes_b)
    )
    a, b, _ = shared
    scarce = scarce_resources(board)
    pair = score(board, [a, b], scarce)
    apart = score(board, [a], scarce) + score(board, [b], scarce)
    # Pips add up because the shared hex is counted once per vertex; the
    # diversity term does not, because it is taken over the union.
    assert pair - apart == pytest.approx(-(1.19 + 0.91 * len(scarce)))


def test_the_prior_prefers_more_pips_when_nothing_else_differs():
    topology = build_topology(MINI_LAYOUT)
    n = topology.num_hexes
    tokens = tuple(6 if h == 0 else 2 for h in range(n))
    board = make_board(topology, (Terrain.FOREST,) * n, tokens)
    state = start(board, 2, rng=random.Random(0))._state
    ordered = rank(state, 0, range(topology.num_vertices))
    top = ordered[0][1]
    assert 0 in topology.vertex_hexes[top]


def test_ties_break_on_vertex_index_so_the_prior_is_repeatable():
    board = random_base_board(random.Random(3))
    state = start(board, 4, rng=random.Random(0))._state
    candidates = list(range(board.topology.num_vertices))
    first = best(state, 0, candidates)
    assert first == best(state, 0, list(reversed(candidates)))


def test_the_prior_scores_the_opening_it_would_complete_not_the_vertex_alone():
    """A candidate is rated with the seat's existing settlements, not on its own."""
    board = random_base_board(random.Random(5))
    state = start(board, 4, rng=random.Random(0))._state
    candidates = [v for v in range(board.topology.num_vertices)]
    alone = best(state, 0, candidates)

    held = next(
        v
        for v in candidates
        if v != alone and alone not in board.topology.vertex_neighbors[v]
    )
    place_settlement(state, 0, held, connected=False)
    scarce = scarce_resources(board)
    for value, vertex in rank(state, 0, [v for v in candidates if v != held]):
        assert value == score(board, [held, vertex], scarce)
        break


def test_a_random_opener_recovers_none_of_the_prior_gap():
    """Calibrates the metric: choosing at random must score near zero, not near one."""
    import hexset.bench.placement_policy as pp

    picks = []
    for index in range(12):
        board = random_base_board(random.Random(f"0:{index}:board"))
        picks.extend(pp.walk_setup("random", board, seed=index))

    summary = pp.summarise(picks)
    assert summary["picks"] == 12 * 8
    assert abs(summary["recovered"]) < 0.2
    assert 0.35 < summary["percentile"] < 0.65
    assert summary["pips"]["prior"] > summary["pips"]["field"]


def test_the_prior_itself_recovers_all_of_the_prior_gap():
    """Pins the other end of the scale, so a score near one means what it says."""
    import hexset.bench.placement_policy as pp

    picks = []
    for index in range(12):
        board = random_base_board(random.Random(f"0:{index}:board"))
        picks.extend(pp.walk_setup("random-placement", board, seed=index))

    summary = pp.summarise(picks)
    assert summary["agreement"] == 1.0
    assert summary["percentile"] == 0.0
    assert summary["recovered"] == 1.0


def test_the_wrapper_only_intercepts_setup_settlements():
    board = random_base_board(random.Random(1))
    game = start(board, 4, rng=random.Random(2))
    sentinel = Action(ActionType.END_TURN)

    class Inner:
        def __init__(self):
            self.calls = 0

        def choose(self, _game):
            self.calls += 1
            return sentinel

    inner = Inner()
    bot = PlacementBot(inner)

    assert game.phase is Phase.SETUP_SETTLEMENT
    chosen = bot.choose(game)
    assert chosen.type is ActionType.SETUP_SETTLEMENT
    assert inner.calls == 0

    game.phase = Phase.SETUP_ROAD
    assert bot.choose(game) is sentinel
    assert inner.calls == 1


def test_the_wrapper_picks_a_legal_vertex_throughout_setup():
    board = random_base_board(random.Random(4))
    game = start(board, 4, rng=random.Random(6))
    bot = PlacementBot(None)

    placed: list[int] = []
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        options = legal_actions(game)
        if game.phase is Phase.SETUP_ROAD:
            apply(game, options[0])
            continue
        action = bot.choose(game)
        assert action in options
        placed.append(action.a)
        apply(game, action)

    assert len(placed) == 8
    assert len(set(placed)) == 8
