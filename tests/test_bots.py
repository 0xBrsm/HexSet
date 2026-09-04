# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

from hexset.actions import ActionType, apply, legal_actions
from hexset.board.board import pips, random_base_board
from hexset.board.terrain import Resource
from hexset.bots import RandomBot, SearchBot, greedy
from hexset.cards import DevCard
from hexset.bots.evaluate import Evaluator
from hexset.game import (
    ROLL_ODDS,
    Phase,
    imagine,
    is_over,
    roll_dice,
    start,
    to_move,
)
from hexset.state import place_settlement, upgrade_to_city
from hexset.trading import NO_VALUATION, one_for_one
from hexset.victory import victory_points
from helpers import clear_hand, give, independent_vertices, mini_board


def a_game(seed: int = 0, players: int = 4):
    rng = random.Random(seed)
    board = random_base_board(rng)
    return start(board, players, rng)


def snapshot(game):
    state = game._state
    return (
        game.phase,
        game.current_player,
        game.turns,
        state.vertex_owner[:],
        state.vertex_building[:],
        state.edge_owner[:],
        state.robber,
        [hand[:] for hand in state.hands],
        state.bank[:],
        state.deck[:],
        game.rng.getstate(),
    )


def play_out(game, bot, cap: int = 20000) -> int:
    moves = 0
    while not is_over(game):
        apply(game, bot.choose(game))
        moves += 1
        if moves > cap:
            raise AssertionError("bot did not finish a game")
    return moves


def test_roll_odds_are_the_dice():
    assert sum(weight for _, weight in ROLL_ODDS) == pytest.approx(1.0)
    assert [roll for roll, _ in ROLL_ODDS] == list(range(2, 13))
    assert all(weight == pips(roll) / 36 for roll, weight in ROLL_ODDS)


def test_an_explicit_roll_is_resolved_as_rolled():
    game = a_game()
    while game.phase is not Phase.ROLL:
        apply(game, legal_actions(game)[0])

    assert roll_dice(game, 8) == 8
    assert game.last_roll == 8


def test_imagining_leaves_the_real_game_untouched():
    game = a_game()
    for _ in range(60):
        apply(game, RandomBot(random.Random(1)).choose(game))
    before = snapshot(game)

    copy = imagine(game, random.Random(2))
    play_out(copy, RandomBot(random.Random(3)))

    assert snapshot(game) == before
    assert is_over(copy)


def test_imagining_hides_the_deck_it_copies():
    game = a_game()
    copy = imagine(game, random.Random(4))
    assert sorted(copy._state.deck) == sorted(game._state.deck)
    assert copy._state.deck != game._state.deck


def test_hidden_deck_randomization_can_be_deferred():
    game = a_game()
    copy = imagine(game, random.Random(4), randomize_deck=False)
    assert copy._state.deck == game._state.deck
    assert copy._state.deck is not game._state.deck


def test_to_move_is_the_discarding_player_not_the_roller():
    game = a_game()
    while game.phase is not Phase.ROLL:
        apply(game, legal_actions(game)[0])
    game.current_player = 0
    game._state.hands[2] = [4, 4, 0, 0, 0]
    roll_dice(game, 7)

    assert game.phase is Phase.DISCARD
    assert to_move(game) == 2
    assert all(a.type is ActionType.DISCARD for a in legal_actions(game))


def test_a_random_bot_finishes_a_game():
    game = a_game(seed=3)
    play_out(game, RandomBot(random.Random(3)))
    assert is_over(game)


def a_trade_that_wins_the_game_for_the_counterparty():
    """Player 0 is nine points and one ore short of a winning city.

    The exchange is good for player 1 in isolation -- it moves their hand
    closer to a settlement -- and fatal in context, because it ends the game
    in somebody else's favour. This is the whole of what the private gate
    exists for (`hexset.trading`): the public vectors would advertise it.
    """
    board = mini_board()
    game = start(board, 4, random.Random(0))
    game.phase = Phase.MAIN
    game.current_player = 0

    spots = independent_vertices(board, 4)
    for vertex in spots[:3]:
        place_settlement(game._state, 0, vertex, connected=False)
        upgrade_to_city(game._state, 0, vertex)
    place_settlement(game._state, 0, spots[3], connected=False)
    game._state.dev_cards[0][DevCard.VICTORY_POINT] += 2
    assert victory_points(game._state, 0) == 9

    clear_hand(game._state, 0)
    give(game._state, 0, Resource.WHEAT, 2)
    give(game._state, 0, Resource.ORE, 2)
    give(game._state, 0, Resource.WOOD, 1)

    clear_hand(game._state, 1)
    give(game._state, 1, Resource.ORE, 1)
    give(game._state, 1, Resource.BRICK, 1)
    give(game._state, 1, Resource.WHEAT, 1)
    return game


def test_the_gate_takes_a_good_exchange_and_refuses_a_bad_one():
    game = a_trade_that_wins_the_game_for_the_counterparty()
    bot = SearchBot(Evaluator(game._state.board), depth=2, width=6, rng=random.Random(0))
    # Seat 0 is one ore short of a city: the ore is worth more than the wood.
    wanted = one_for_one(int(Resource.WOOD), int(Resource.ORE))
    assert bot.accepts(game.state(0), wanted, 1) is True
    assert bot.accepts(game.state(0), tuple(-n for n in wanted), 1) is False


def test_a_bot_that_never_trades_publishes_nothing_and_refuses_everything():
    game = a_trade_that_wins_the_game_for_the_counterparty()
    quiet = greedy(Evaluator(game._state.board), random.Random(0), max_trades=0)
    assert quiet.valuation(game.state(1)) == NO_VALUATION
    assert quiet.accepts(game.state(1), one_for_one(4, 0), 0) is False


def test_the_published_vector_names_one_want_and_one_give():
    game = a_trade_that_wins_the_game_for_the_counterparty()
    bot = greedy(Evaluator(game._state.board), random.Random(0))
    published = bot.valuation(game.state(0))
    assert sorted(published) == [-1.0, 0.0, 0.0, 0.0, 1.0]
    # The give side can only be a card actually held.
    assert game._state.hands[0][published.index(-1.0)] > 0


def test_a_seat_with_an_empty_hand_advertises_nothing():
    game = a_trade_that_wins_the_game_for_the_counterparty()
    clear_hand(game._state, 2)
    bot = greedy(Evaluator(game._state.board), random.Random(0))
    assert bot.valuation(game.state(2)) == NO_VALUATION
