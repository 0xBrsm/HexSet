# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

from hexset.board.board import make_board, random_base_board
from hexset.board.maps import MINI_LAYOUT
from hexset.board.terrain import Resource, Terrain
from hexset.board.topology import build as build_topology
from hexset.cards import DevCard
from hexset.evaluate_tiered import (
    ROLLS,
    TERM_NAMES,
    VARIETY_PIPS,
    Evaluator,
    Weights,
)
from hexset.game import start
from hexset.state import new_game, place_road, place_settlement, upgrade_to_city
from hexset.victory import WINNING_POINTS, victory_points
from helpers import ROLL, a_vertex_touching, give, independent_vertices, mini_board

MINI_PIPS = 3  # every mini-board producer bears the same token


def mixed_board():
    """One hex of each terrain, so a junction can touch three distinct resources."""
    topology = build_topology(MINI_LAYOUT)
    terrain = (
        Terrain.DESERT,
        Terrain.FOREST,
        Terrain.HILLS,
        Terrain.PASTURE,
        Terrain.FIELDS,
        Terrain.MOUNTAINS,
        Terrain.FOREST,
    )
    tokens = (0,) + (ROLL,) * (len(terrain) - 1)
    return make_board(topology, terrain, tokens)


def a_state(board, players: int = 4):
    return new_game(board, players, random.Random(0))


def term(evaluator, state, player, name, *, knower=None):
    values = evaluator.terms(state, player, knower=knower)
    return values[TERM_NAMES.index(name)]


def test_an_empty_position_scores_every_seat_alike():
    board = mini_board()
    scores = Evaluator(board).evaluate(a_state(board))
    assert len(set(scores)) == 1


def test_production_counts_pips_and_a_bonus_for_each_resource():
    board = mini_board()
    state = a_state(board)
    place_settlement(state, 0, a_vertex_touching(board, 3), connected=False)

    rate = Evaluator(board).snapshot(state).production[0]
    assert rate == (3 * MINI_PIPS + VARIETY_PIPS) / ROLLS


def test_a_city_doubles_the_pips_a_settlement_earns():
    board = mini_board()
    state = a_state(board)
    vertex = a_vertex_touching(board, 3)
    place_settlement(state, 0, vertex, connected=False)
    evaluator = Evaluator(board)
    before = evaluator.snapshot(state).bare_production[0]

    upgrade_to_city(state, 0, vertex)
    assert evaluator.snapshot(state).bare_production[0] == 2 * before


def test_the_robber_removes_the_hex_it_sits_on():
    board = mini_board()
    state = a_state(board)
    vertex = a_vertex_touching(board, 3)
    place_settlement(state, 0, vertex, connected=False)
    evaluator = Evaluator(board)
    before = evaluator.snapshot(state).bare_production[0]

    state.robber = board.topology.vertex_hexes[vertex][0]
    after = evaluator.snapshot(state).bare_production[0]
    assert after == pytest.approx(before - MINI_PIPS / ROLLS)


def test_gold_pays_pips_but_earns_no_variety_bonus():
    board = mini_board(gold=True)
    state = a_state(board)
    place_settlement(state, 0, a_vertex_touching(board, 3), connected=False)

    snapshot = Evaluator(board).snapshot(state)
    assert snapshot.bare_production[0] > 0
    assert snapshot.production[0] == snapshot.bare_production[0]


def test_variety_makes_three_resources_beat_one():
    diverse, plain = mixed_board(), mini_board()
    diverse_state, plain_state = a_state(diverse), a_state(plain)

    spot = next(
        v
        for v, hexes in enumerate(diverse.topology.vertex_hexes)
        if len(hexes) == 3 and 0 not in hexes
    )
    place_settlement(diverse_state, 0, spot, connected=False)
    place_settlement(plain_state, 0, a_vertex_touching(plain, 3), connected=False)

    diverse_rate = Evaluator(diverse).snapshot(diverse_state).production[0]
    plain_rate = Evaluator(plain).snapshot(plain_state).production[0]
    assert diverse_rate == pytest.approx(plain_rate + 2 * VARIETY_PIPS / ROLLS)


def test_enemy_production_tracks_the_strongest_opponent_only():
    board = mini_board()
    state = a_state(board)
    spots = independent_vertices(board, 3)
    place_settlement(state, 1, spots[0], connected=False)
    place_settlement(state, 2, spots[1], connected=False)
    upgrade_to_city(state, 2, spots[1])

    evaluator = Evaluator(board)
    snapshot = evaluator.snapshot(state)
    assert term(evaluator, state, 0, "enemy_production") == max(
        snapshot.bare_production[1], snapshot.bare_production[2]
    )
    # Seat 2 is the strongest, so it is measured against seat 1, not itself.
    assert term(evaluator, state, 2, "enemy_production") == snapshot.bare_production[1]


def test_settling_never_reduces_reachable_production():
    """The trap the previous evaluation fell into, and the reason for this test."""
    board = random_base_board(random.Random(4))
    evaluator = Evaluator(board)
    state = new_game(board, 4, random.Random(0))
    vertex = next(
        v for v, hexes in enumerate(board.topology.vertex_hexes) if len(hexes) == 3
    )
    place_settlement(state, 0, vertex, connected=False)
    before, _ = evaluator.reachable(state, 0, evaluator.snapshot(state).sources[0])

    edge = board.topology.vertex_edges[vertex][0]
    place_road(state, 0, edge)
    after, _ = evaluator.reachable(state, 0, evaluator.snapshot(state).sources[0])
    assert after >= before


def test_buildable_nodes_falls_as_the_board_fills():
    board = random_base_board(random.Random(5))
    evaluator = Evaluator(board)
    state = new_game(board, 4, random.Random(0))
    vertex = next(
        v for v, hexes in enumerate(board.topology.vertex_hexes) if len(hexes) == 3
    )
    place_settlement(state, 0, vertex, connected=False)
    _, open_spots = evaluator.reachable(state, 0, evaluator.snapshot(state).sources[0])

    # Spread the crowding over the three other players: the supply caps any one
    # of them at five settlements, and the ring two steps out can hold more.
    filled = 0
    for neighbour in board.topology.vertex_neighbors[vertex]:
        for beyond in board.topology.vertex_neighbors[neighbour]:
            if state.vertex_building[beyond] == 0 and beyond != vertex:
                if not any(
                    state.vertex_building[n]
                    for n in board.topology.vertex_neighbors[beyond]
                ):
                    place_settlement(state, 1 + filled % 3, beyond, connected=False)
                    filled += 1
    _, crowded = evaluator.reachable(state, 0, evaluator.snapshot(state).sources[0])
    assert crowded < open_spots


def test_hand_synergy_rewards_a_hand_useful_for_two_builds():
    board = mini_board()
    evaluator = Evaluator(board)

    empty = a_state(board)
    assert evaluator.hand_synergy(empty, 0) == 0.0

    both = a_state(board)
    for resource, count in (
        (Resource.WHEAT, 3),
        (Resource.ORE, 3),
        (Resource.SHEEP, 1),
        (Resource.WOOD, 1),
        (Resource.BRICK, 1),
    ):
        give(both, 0, resource, count)
    assert evaluator.hand_synergy(both, 0) == 1.0


def test_touched_tiles_counts_hexes_not_buildings():
    board = mini_board()
    state = a_state(board)
    vertex = a_vertex_touching(board, 3)
    place_settlement(state, 0, vertex, connected=False)

    evaluator = Evaluator(board)
    assert evaluator.touched_tiles(state, 0) == 3
    upgrade_to_city(state, 0, vertex)
    assert evaluator.touched_tiles(state, 0) == 3


def test_the_discard_penalty_is_flat_above_the_threshold():
    board = mini_board()
    evaluator = Evaluator(board)

    safe = a_state(board)
    give(safe, 0, Resource.WOOD, 7)
    assert term(evaluator, safe, 0, "discard_penalty") == 0.0

    over = a_state(board)
    give(over, 0, Resource.WOOD, 8)
    assert term(evaluator, over, 0, "discard_penalty") == 1.0

    far_over = a_state(board)
    give(far_over, 0, Resource.WOOD, 12)
    assert term(evaluator, far_over, 0, "discard_penalty") == 1.0


def test_a_victory_point_outranks_everything_below_it():
    """The tiers are a priority order, so this has to hold by construction."""
    board = mini_board()
    evaluator = Evaluator(board)
    weights = evaluator.weights

    # The largest plausible contribution from every tier under the first.
    generous = (
        weights.production * 3
        + abs(weights.enemy_production) * 3
        + weights.reachable_production_1 * 10
        + weights.buildable_nodes * 20
        + weights.hand_synergy * 1
        + weights.longest_road * 15
        + weights.hand_devs * 25
        + weights.army_size * 14
        + weights.num_tiles * 19
        + weights.hand_resources * 40
    )
    assert weights.victory_point > generous


def test_victory_points_stay_exactly_representable_alongside_tie_breaks():
    """Tiers this far apart risk the smallest terms vanishing into rounding."""
    weights = Weights()
    biggest = weights.victory_point * 12
    assert biggest + weights.num_tiles != biggest


def test_hidden_cards_count_only_for_the_seat_that_holds_them():
    board = mini_board()
    state = a_state(board)
    state.dev_cards[0][DevCard.VICTORY_POINT] += 1
    state.dev_cards[0][DevCard.KNIGHT] += 1
    evaluator = Evaluator(board)

    assert term(evaluator, state, 0, "victory_point") == 0
    assert term(evaluator, state, 0, "victory_point", knower=0) == 1
    assert term(evaluator, state, 0, "hand_devs") == 0
    assert term(evaluator, state, 0, "hand_devs", knower=0) == 2
    # Another seat's cards stay invisible even to a knower.
    assert term(evaluator, state, 1, "hand_devs", knower=0) == 0


def test_a_winning_position_outscores_every_other_seat():
    board = mini_board()
    state = a_state(board)
    # Four cities and two settlements: ten points inside the piece supply.
    # Upgrade as we go: the supply never lets a player hold six settlements.
    spots = independent_vertices(board, 6)
    for vertex in spots[:4]:
        place_settlement(state, 0, vertex, connected=False)
        upgrade_to_city(state, 0, vertex)
    for vertex in spots[4:]:
        place_settlement(state, 0, vertex, connected=False)
    assert victory_points(state, 0) >= WINNING_POINTS

    scores = Evaluator(board).evaluate(state)
    assert scores[0] == max(scores)
    assert scores[0] > sum(abs(s) for s in scores[1:])


def test_terms_line_up_with_the_weights_they_multiply():
    board = mini_board()
    state = a_state(board)
    evaluator = Evaluator(board)
    assert len(evaluator.terms(state, 0)) == len(TERM_NAMES)
    assert evaluator.vector == tuple(
        getattr(evaluator.weights, name) for name in TERM_NAMES
    )


def test_a_shared_snapshot_scores_the_same_as_a_private_one():
    board = random_base_board(random.Random(2))
    state = new_game(board, 4, random.Random(0))
    for vertex in independent_vertices(board, 4):
        place_settlement(state, vertex % 4, vertex, connected=False)

    evaluator = Evaluator(board)
    shared = evaluator.evaluate(state, 0)
    private = [evaluator.score(state, p, knower=0) for p in range(4)]
    assert shared == private


def test_for_game_reads_the_board_off_the_game():
    rng = random.Random(0)
    game = start(random_base_board(rng), 4, rng)
    assert len(Evaluator.for_game(game).evaluate(game.state)) == 4
