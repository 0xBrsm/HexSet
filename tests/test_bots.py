# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

from hexset.actions import ActionType, apply, legal_actions
from hexset.board.board import pips, random_base_board
from hexset.board.terrain import Resource
from hexset.bots import RandomBot
from hexset.game import (
    ROLL_ODDS,
    Phase,
    imagine,
    is_over,
    roll_dice,
    start,
    to_move,
)


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
