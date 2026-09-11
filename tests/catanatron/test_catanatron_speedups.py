# SPDX-License-Identifier: GPL-3.0-only
"""Exact scores, copy isolation and lifecycle for native acceleration."""
import random
from collections import defaultdict

import pytest

pytest.importorskip("catanatron.game")

from catanatron.game import Game
from catanatron.models.enums import CITY, SETTLEMENT
from catanatron.models.player import Color, RandomPlayer
from catanatron.models.board import Board
from catanatron.players import value
from hexset.catanatron.speedups import (
    BoardFeatureCache, catanatron_speedups, clone_board_mutable_structures,
    optimized_base_fn, verify_runtime,
)


def position(seed=17):
    return Game([RandomPlayer(c) for c in Color], seed=seed)


def test_runtime_matches_the_pinned_source():
    verify_runtime()


@pytest.mark.parametrize("mode", ["basic", "cached"])
def test_scores_match_native_on_every_perspective_through_real_games(mode):
    native = value.base_fn()
    # Nondefault weights ensure we check every feature, including the native
    # zero-weight reachability term and its small floating-point contributions.
    params = {k: (i + 1) / 7 for i, k in enumerate(value.DEFAULT_WEIGHTS)}
    varied_native = value.base_fn(params)
    states = []
    for seed in (7, 19):
        game = position(seed)
        rng = random.Random(seed)
        for tick in range(300):
            if tick % 7 == 0:
                states.append(game.copy())
            if game.winning_color() is not None:
                break
            game.execute(rng.choice(game.playable_actions))
    with catanatron_speedups(mode) as cache:
        fast = value.base_fn()
        varied_fast = value.base_fn(params)
        for game in states:
            for color in game.state.colors:
                assert fast(game, color) == native(game, color)
                assert varied_fast(game, color) == varied_native(game, color)
        if cache is not None:
            assert cache.hits > 0
            assert cache.misses > 0


def test_cache_reads_dynamic_hand_points_and_longest_road_live():
    game = position()
    cache = BoardFeatureCache()
    fast = optimized_base_fn(cache=cache)
    native = value.base_fn()
    color = game.state.colors[0]
    first = fast(game, color)
    for suffix, delta in (("WHEAT_IN_HAND", 8), ("VICTORY_POINTS", 1),
                          ("LONGEST_ROAD_LENGTH", 2), ("KNIGHT_IN_HAND", 1),
                          ("PLAYED_KNIGHT", 1)):
        key = 'P0_' + suffix
        game.state.player_state[key] += delta
        assert fast(game, color) == native(game, color)
    assert fast(game, color) != first
    assert cache.misses == 1
    assert cache.hits > 1


def test_cache_invalidates_board_dependencies_and_bounds_memory():
    game = position()
    cache = BoardFeatureCache(max_entries=2)
    color = game.state.colors[0]
    cache.terms(game, color)
    board = game.state.board
    # Independent mutable dependencies, even when the resulting synthetic
    # state has redundant representations that are not mutually consistent.
    mutations = (
        lambda: setattr(board, 'robber_coordinate', next(c for c in board.map.land_tiles if c != board.robber_coordinate)),
        lambda: game.state.buildings_by_color[color][SETTLEMENT].append(0),
        lambda: game.state.buildings_by_color[color][CITY].append(1),
        lambda: board.buildings.update({0: (color, SETTLEMENT)}),
        lambda: board.roads.update({(0, 1): color}),
        lambda: board.connected_components[color].append({0, 1}),
        lambda: board.board_buildable_ids.discard(0),
    )
    for mutate in mutations:
        before = cache.misses
        mutate()
        cached = cache.terms(game, color)
        assert cache.misses == before + 1
        assert cached == BoardFeatureCache().terms(game, color)
        assert len(cache.entries) <= 2
    other = position(18)
    cache.terms(other, other.state.colors[0])
    assert cache.map is other.state.board.map
    assert len(cache.entries) == 1


def test_board_copy_matches_native_and_isolates_all_mutable_fields():
    board = position().state.board
    color = Color.RED
    board.connected_components[color] = [{0, 1}, {3, 4}]
    board.buildable_edges_cache[color] = [(0, 1), (3, 4)]
    board.player_port_resources_cache[color] = {'WOOD', 'ORE'}
    expected = board.copy()
    actual = clone_board_mutable_structures(board)
    assert vars(actual) == vars(expected)
    assert isinstance(actual.connected_components, defaultdict)
    for name in ('map', 'buildable_subgraph'):
        assert getattr(actual, name) is getattr(board, name)
    for name in ('buildings', 'roads', 'connected_components', 'board_buildable_ids',
                 'road_lengths', 'buildable_edges_cache', 'player_port_resources_cache'):
        assert getattr(actual, name) is not getattr(board, name)
    actual.connected_components[color][0].add(9)
    actual.buildable_edges_cache[color].append((8, 9))
    actual.player_port_resources_cache[color].add('SHEEP')
    assert vars(board) == vars(expected)


def test_context_restores_after_failure_and_rejects_nesting():
    before = Board.copy, value.base_fn
    with pytest.raises(LookupError):
        with catanatron_speedups('cached'):
            assert (Board.copy, value.base_fn) != before
            with pytest.raises(RuntimeError):
                with catanatron_speedups('basic'):
                    pass
            raise LookupError()
    assert (Board.copy, value.base_fn) == before
    with catanatron_speedups('off') as cache:
        assert cache is None
        assert (Board.copy, value.base_fn) == before
