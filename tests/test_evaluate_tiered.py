# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

from hexset.board.board import random_base_board
from hexset.cards import DevCard
from hexset.evaluate_tiered import TERM_NAMES, Evaluator
from hexset.game import start
from hexset.state import new_game, place_settlement
from helpers import independent_vertices, mini_board


def a_state(board, players: int = 4):
    return new_game(board, players, random.Random(0))


def term(evaluator, state, player, name, *, knower=None):
    values = evaluator.terms(state, player, knower=knower)
    return values[TERM_NAMES.index(name)]


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
    assert len(Evaluator.for_game(game).evaluate(game._state)) == 4
