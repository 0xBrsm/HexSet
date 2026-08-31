# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

from hexset.actions import (
    YEAR_OF_PLENTY_PAIRS,
    Action,
    ActionType,
    apply,
    build_space,
    legal_actions,
    legal_mask,
    space_for,
    _offer_actions,
)
from hexset.board.board import random_base_board
from hexset.economy import expected_total, total_in_play
from hexset.game import Phase, is_over, start
from hexset.play import play_random_game, step_randomly
from hexset.trading import Offer, responders
from hexset.victory import WINNING_POINTS, victory_points


def a_game(players: int = 4, seed: int = 0):
    rng = random.Random(seed)
    return start(random_base_board(rng), players, rng)


def test_space_size_is_the_sum_of_its_blocks():
    space = build_space(54, 72, 19, 4)
    assert space.size == sum(space.sizes)
    assert space.offsets[0] == 0


def test_every_index_round_trips():
    space = build_space(54, 72, 19, 4)
    for index in range(space.size):
        assert space.index(space.decode(index)) == index


def test_every_action_round_trips():
    space = build_space(54, 72, 19, 4)
    samples = [
        Action(ActionType.ROLL),
        Action(ActionType.END_TURN),
        Action(ActionType.BUILD_ROAD, 71),
        Action(ActionType.BUILD_SETTLEMENT, 53),
        Action(ActionType.BUILD_CITY, 0),
        Action(ActionType.MOVE_ROBBER, 18, 3),
        Action(ActionType.PLAY_KNIGHT, 0, 4),
        Action(ActionType.BANK_TRADE, 4, 0),
        Action(ActionType.PLAY_YEAR_OF_PLENTY, len(YEAR_OF_PLENTY_PAIRS) - 1),
        Action(ActionType.DISCARD, 2),
    ]
    for action in samples:
        assert space.decode(space.index(action)) == action


def test_the_space_grows_with_the_board():
    small = build_space(24, 30, 7, 4)
    large = build_space(54, 72, 19, 4)
    assert large.size > small.size


def test_more_players_widen_only_the_robber_blocks():
    three = build_space(54, 72, 19, 3)
    four = build_space(54, 72, 19, 4)
    grew = [
        kind
        for kind in ActionType
        if four.sizes[kind] != three.sizes[kind]
    ]
    assert grew == [ActionType.MOVE_ROBBER, ActionType.PLAY_KNIGHT]


def test_setup_offers_only_setup_actions():
    game = a_game()
    assert {a.type for a in legal_actions(game)} == {ActionType.SETUP_SETTLEMENT}

    apply(game, legal_actions(game)[0])
    assert {a.type for a in legal_actions(game)} == {ActionType.SETUP_ROAD}


def test_the_opening_road_must_touch_the_new_settlement():
    game = a_game()
    apply(game, legal_actions(game)[0])
    offered = {a.a for a in legal_actions(game)}
    topology = game.state.board.topology
    assert offered == set(topology.vertex_edges[game.last_settlement])


def test_rolling_is_the_only_option_without_cards():
    game = a_game()
    while game.phase is not Phase.ROLL:
        apply(game, legal_actions(game)[0])
    assert legal_actions(game) == [Action(ActionType.ROLL)]


def test_the_mask_agrees_with_the_action_list():
    game = a_game()
    rng = random.Random(1)
    space = space_for(game)
    for _ in range(200):
        if is_over(game):
            break
        mask = legal_mask(game, space)
        expected = {space.index(a) for a in legal_actions(game)}
        assert {i for i, ok in enumerate(mask) if ok} == expected
        step_randomly(game, rng)


def test_a_finished_game_offers_nothing():
    game = play_random_game(num_players=3, rng=random.Random(5))
    assert is_over(game)
    assert legal_actions(game) == []


@pytest.mark.parametrize("seed", range(8))
def test_random_games_finish_with_a_legal_winner(seed):
    game = play_random_game(num_players=4, rng=random.Random(seed))

    assert game.won_by is not None
    assert victory_points(game.state, game.won_by) >= WINNING_POINTS


@pytest.mark.parametrize("seed", range(5))
def test_resources_survive_a_whole_game(seed):
    game = play_random_game(num_players=4, rng=random.Random(seed))
    assert total_in_play(game.state) == expected_total()
    assert all(n >= 0 for n in game.state.bank)
    assert all(n >= 0 for hand in game.state.hands for n in hand)


@pytest.mark.parametrize("players", [2, 3, 4])
def test_every_supported_player_count_plays(players):
    game = play_random_game(num_players=players, rng=random.Random(11))
    assert is_over(game)


def test_a_live_game_always_offers_something():
    rng = random.Random(7)
    game = a_game(players=4, seed=7)
    while not is_over(game):
        assert legal_actions(game), f"stuck in {game.phase.name}"
        step_randomly(game, rng)


def test_fast_offer_enumeration_matches_responder_rules():
    rng = random.Random(19)
    game = a_game(players=4, seed=19)
    game.phase = Phase.MAIN

    for _ in range(100):
        game.current_player = rng.randrange(game.state.num_players)
        game.offers_made = rng.randrange(10)
        game.state.hands = [
            [rng.randrange(4) for _ in range(5)]
            for _ in range(game.state.num_players)
        ]

        expected = []
        player = game.current_player
        if game.offers_made < 8:
            for given in range(5):
                if not game.state.hands[player][given]:
                    continue
                for wanted in range(5):
                    if wanted == given:
                        continue
                    give = tuple(int(r == given) for r in range(5))
                    want = tuple(int(r == wanted) for r in range(5))
                    if responders(game.state, Offer(player, give, want)):
                        expected.append(
                            Action(ActionType.PROPOSE_TRADE, give=give, want=want)
                        )

        assert _offer_actions(game) == expected


def test_free_roads_are_placed_before_the_turn_can_end():
    game = a_game(players=2)
    rng = random.Random(2)
    while game.phase is not Phase.MAIN:
        step_randomly(game, rng)

    game.free_roads = 2
    kinds = {a.type for a in legal_actions(game)}
    if ActionType.BUILD_ROAD in kinds:
        assert ActionType.END_TURN not in kinds


def test_a_stranded_free_road_does_not_deadlock():
    game = a_game(players=2)
    rng = random.Random(3)
    while game.phase is not Phase.MAIN:
        step_randomly(game, rng)

    # Nowhere legal to build, so the turn must still be endable.
    game.free_roads = 2
    game.state.edge_owner = [0] * len(game.state.edge_owner)
    kinds = {a.type for a in legal_actions(game)}
    assert ActionType.END_TURN in kinds
