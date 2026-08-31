from __future__ import annotations

import random

import pytest

from catan.board.board import make_board, random_base_board
from catan.board.maps import MINI_LAYOUT
from catan.board.terrain import TERRAIN_RESOURCE, Resource, Terrain
from catan.board.topology import build as build_topology
from catan.cards import DevCard
from catan.evaluate import FITTED_SCARCE, ROLLS, WIN_SCORE, Evaluator, Weights
from catan.game import start
from catan.state import new_game, place_settlement, upgrade_to_city
from catan.victory import WINNING_POINTS, victory_points
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


def distinct_junction(board, count: int = 3) -> int:
    """A junction touching `count` hexes, all of them different resources."""
    for v, hexes in enumerate(board.topology.vertex_hexes):
        resources = {TERRAIN_RESOURCE[board.terrain[h]] for h in hexes}
        if len(hexes) == count and len(resources) == count and None not in resources:
            return v
    raise AssertionError(f"no junction touching {count} distinct resources")


def test_an_empty_position_scores_every_seat_alike():
    board = mini_board()
    scores = Evaluator(board).evaluate(a_state(board))
    assert len(set(scores)) == 1


def test_production_is_expected_cards_per_turn():
    board = mini_board()
    state = a_state(board)
    vertex = a_vertex_touching(board, 3)
    place_settlement(state, 0, vertex, connected=False)

    rate, kinds = Evaluator(board).production(state, 0)
    assert rate == 3 * MINI_PIPS / ROLLS
    assert kinds == 1


def test_a_city_produces_twice_a_settlement():
    board = mini_board()
    state = a_state(board)
    vertex = a_vertex_touching(board, 3)
    place_settlement(state, 0, vertex, connected=False)
    evaluator = Evaluator(board)
    settlement_rate, _ = evaluator.production(state, 0)

    upgrade_to_city(state, 0, vertex)
    city_rate, _ = evaluator.production(state, 0)
    assert city_rate == 2 * settlement_rate


def test_the_robber_removes_the_hex_it_sits_on():
    board = mini_board()
    state = a_state(board)
    vertex = a_vertex_touching(board, 3)
    place_settlement(state, 0, vertex, connected=False)
    evaluator = Evaluator(board)
    before, _ = evaluator.production(state, 0)

    state.robber = board.topology.vertex_hexes[vertex][0]
    after, _ = evaluator.production(state, 0)
    assert after == pytest.approx(before - MINI_PIPS / ROLLS)


def test_gold_pays_a_rate_but_no_diversity():
    board = mini_board(gold=True)
    state = a_state(board)
    place_settlement(state, 0, a_vertex_touching(board, 3), connected=False)

    rate, kinds = Evaluator(board).production(state, 0)
    assert rate > 0
    assert kinds == 0


def test_diverse_production_beats_concentrated_production():
    """Same pip total, three resources against one."""
    diverse, plain = mixed_board(), mini_board()
    diverse_state, plain_state = a_state(diverse), a_state(plain)
    place_settlement(diverse_state, 0, distinct_junction(diverse), connected=False)
    place_settlement(plain_state, 0, a_vertex_touching(plain, 3), connected=False)

    diverse_rate, diverse_kinds = Evaluator(diverse).production(diverse_state, 0)
    plain_rate, plain_kinds = Evaluator(plain).production(plain_state, 0)
    assert diverse_rate == plain_rate
    assert (diverse_kinds, plain_kinds) == (3, 1)
    assert Evaluator(diverse).score(diverse_state, 0) > Evaluator(plain).score(
        plain_state, 0
    )


def test_progress_tells_a_city_hand_from_a_junk_hand():
    board = mini_board()
    evaluator = Evaluator(board)

    empty = a_state(board)
    assert evaluator.progress(empty, 0) == 0.0

    city = a_state(board)
    give(city, 0, Resource.WHEAT, 2)
    give(city, 0, Resource.ORE, 3)
    assert evaluator.progress(city, 0) == 1.0

    junk = a_state(board)
    give(junk, 0, Resource.WOOD, 5)
    assert evaluator.progress(junk, 0) == 0.25


def test_a_winning_position_dominates_every_other_term():
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

    evaluator = Evaluator(board)
    assert evaluator.score(state, 0) > WIN_SCORE
    assert evaluator.score(state, 1) < WIN_SCORE


def test_hidden_victory_cards_count_only_for_the_seat_that_holds_them():
    board = mini_board()
    state = a_state(board)
    state.dev_cards[0][DevCard.VICTORY_POINT] += 1
    evaluator = Evaluator(board)

    public = evaluator.score(state, 0)
    known = evaluator.score(state, 0, knower=0)
    assert known - public == evaluator.weights.victory_point
    assert evaluator.score(state, 1, knower=0) == evaluator.score(state, 1)


def test_weights_are_what_the_score_is_built_from():
    board = mini_board()
    state = a_state(board)
    place_settlement(state, 0, a_vertex_touching(board, 3), connected=False)

    silent = Weights(
        victory_point=0.0,
        production=0.0,
        diversity=0.0,
        scarce=0.0,
        progress=0.0,
        road=0.0,
        knight=0.0,
        card=0.0,
        surplus_card=0.0,
        port=0.0,
    )
    assert Evaluator(board, silent).score(state, 0) == 0.0


def test_for_game_reads_the_board_off_the_game():
    rng = random.Random(0)
    game = start(random_base_board(rng), 4, rng)
    scores = Evaluator.for_game(game).evaluate(game.state)
    assert len(scores) == 4


def test_scarcity_counts_only_the_short_resources_the_seat_reaches():
    """A resource with fewer hexes than the commonest counts; a plentiful one does not."""
    assert Evaluator(random_base_board(random.Random(0))).scarce == {
        Resource.BRICK,
        Resource.ORE,
    }

    topology = build_topology(MINI_LAYOUT)
    n = topology.num_hexes
    # The desert is first so the robber starts there rather than on the one hex
    # this test needs to be producing.
    terrain = (Terrain.DESERT, Terrain.HILLS) + (Terrain.FOREST,) * (n - 2)
    board = make_board(topology, terrain, (0,) + (6,) * (n - 1))
    evaluator = Evaluator(board)
    assert evaluator.scarce == {Resource.BRICK}

    touching = next(v for v, hs in enumerate(topology.vertex_hexes) if 1 in hs)
    away = next(v for v, hs in enumerate(topology.vertex_hexes) if 1 not in hs)
    for vertex, expected in ((touching, 1), (away, 0)):
        state = new_game(board, 2)
        place_settlement(state, 0, vertex, connected=False)
        assert evaluator.survey(state, 0).scarce == expected


def test_the_scarcity_default_is_the_fitted_value_and_cannot_drift_from_it():
    """The one weight taken from the opening fit rather than fitted against the engine."""
    assert Weights().scarce == pytest.approx(FITTED_SCARCE, abs=1e-4)


def test_the_survey_agrees_with_the_rules():
    """The fast walk must equal the canonical functions it replaced.

    `Evaluator.survey` folds production, building points and port rates into
    one pass for speed, which means the same arithmetic now lives in two
    places. This is what stops the fast one drifting from the real one.
    """
    from catan.actions import apply, legal_actions
    from catan.economy import trade_ratios
    from catan.board.ports import BASE_TRADE_RATIO
    from catan.victory import building_points

    rng = random.Random(11)
    game = start(random_base_board(rng), 4, rng)
    evaluator = Evaluator(game.state.board)

    checked = with_a_port = with_a_building = 0
    for _ in range(400):
        options = legal_actions(game)
        if not options:
            break
        apply(game, rng.choice(options))
        for player in range(4):
            walk = evaluator.survey(game.state, player)
            assert walk.buildings == building_points(game.state, player)
            assert walk.port_gain == sum(
                BASE_TRADE_RATIO - r for r in trade_ratios(game.state, player)
            )
            assert (walk.rate, walk.kinds) == evaluator.production(game.state, player)
            checked += 1
            with_a_port += walk.port_gain > 0
            with_a_building += walk.buildings > 0

    # Agreeing on zero everywhere would pass vacuously, and the port branch is
    # the one that would go unexercised: nobody owns a port until they settle
    # on one.
    assert checked > 1000
    assert with_a_building > 100
    assert with_a_port > 100
