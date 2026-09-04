# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

from hexset.board.board import random_base_board
from hexset.cards import DevCard
from hexset.bots.evaluate import FITTED_SCARCE, Evaluator, Weights
from hexset.game import start
from hexset.state import new_game, place_settlement
from helpers import a_vertex_touching, mini_board


def a_state(board, players: int = 4):
    return new_game(board, players, random.Random(0))


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
    scores = Evaluator.for_game(game).evaluate(game._state)
    assert len(scores) == 4


def test_the_scarcity_default_is_the_fitted_value_and_cannot_drift_from_it():
    """The one weight taken from the opening fit rather than fitted against the engine."""
    assert Weights().scarce == pytest.approx(FITTED_SCARCE, abs=1e-4)
