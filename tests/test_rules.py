# SPDX-License-Identifier: GPL-3.0-only
"""Game types: the rules variants beyond standard Catan.

`hexset.rules` carries the two numbers that vary between game types (VPs to
win, discard limit); the engine, the bots and the catanatron bridge read
them off the position instead of the module-level standard constants, so
one engine plays every game type.
"""

from __future__ import annotations

import random

import pytest

from helpers import independent_vertices, mini_board

from hexset.bots.evaluate import WIN_SCORE, Evaluator
from hexset.bots.heximax.evaluate import ViewEvaluator
from hexset.game import Phase, _check_win, roll_dice, start
from hexset.robber import discard_count
from hexset.rules import COLONIST_1V1, GAME_TYPES, STANDARD, Rules
from hexset.state import Building, copy_state, new_game
from hexset.victory import winner


def test_presets_carry_the_documented_numbers():
    assert (STANDARD.winning_points, STANDARD.discard_limit) == (10, 7)
    assert (COLONIST_1V1.winning_points, COLONIST_1V1.discard_limit) == (15, 9)
    assert GAME_TYPES["standard"] is STANDARD
    assert GAME_TYPES["colonist-1v1"] is COLONIST_1V1


def test_rules_reject_nonsense():
    with pytest.raises(ValueError):
        Rules(winning_points=0)
    with pytest.raises(ValueError):
        Rules(discard_limit=-1)


def _cities(state, player, count):
    for v in independent_vertices(state.board, count):
        state.vertex_owner[v] = player
        state.vertex_building[v] = Building.CITY


def test_winner_uses_the_game_types_threshold():
    std = new_game(mini_board(), 2, random.Random(0), rules=STANDARD)
    col = new_game(mini_board(), 2, random.Random(0), rules=COLONIST_1V1)
    _cities(std, 0, 5)  # 10 VP
    _cities(col, 0, 5)
    assert winner(std) == 0
    assert winner(col) is None  # 10 is not 15


def test_discard_uses_the_game_types_limit():
    std = new_game(mini_board(), 2, random.Random(0), rules=STANDARD)
    col = new_game(mini_board(), 2, random.Random(0), rules=COLONIST_1V1)
    std.hands[0] = [2, 2, 2, 2, 0]
    col.hands[0] = [2, 2, 2, 2, 0]  # 8 cards
    assert discard_count(std, 0) == 4  # over 7: half
    assert discard_count(col, 0) == 0  # 8 <= 9: nothing
    col.hands[0] = [2, 2, 2, 2, 2]  # 10 cards
    assert discard_count(col, 0) == 5


def test_check_win_uses_the_game_types_threshold():
    game = start(mini_board(), 2, random.Random(0), rules=COLONIST_1V1)
    game.phase = Phase.MAIN
    game.current_player = 0
    _cities(game._state, 0, 5)  # 10 VP: a standard game would end here
    _check_win(game)
    assert game.won_by is None
    assert game.phase is Phase.MAIN
    _cities(game._state, 0, 8)  # 16 VP
    _check_win(game)
    assert game.won_by == 0
    assert game.phase is Phase.GAME_OVER


def test_check_win_still_ends_a_standard_game_at_ten():
    game = start(mini_board(), 2, random.Random(0))
    game.phase = Phase.MAIN
    game.current_player = 0
    _cities(game._state, 0, 5)
    _check_win(game)
    assert game.won_by == 0


def test_seven_discard_quotas_use_the_game_types_limit():
    game = start(mini_board(), 2, random.Random(0), rules=COLONIST_1V1)
    game.phase = Phase.ROLL
    game.current_player = 0
    game._state.hands[0] = [2, 2, 2, 2, 0]  # 8 cards: safe under colonist rules
    game._state.hands[1] = [0, 0, 0, 0, 0]
    assert roll_dice(game, 7) == 7
    assert game.discard_quota == [0, 0]
    assert game.phase is Phase.ROBBER  # nobody discards, straight to the robber

    game.phase = Phase.ROLL
    game._state.hands[0] = [2, 2, 2, 2, 2]  # 10 cards: over 9
    assert roll_dice(game, 7) == 7
    assert game.discard_quota == [5, 0]
    assert game.phase is Phase.DISCARD


def test_copy_state_preserves_the_rules():
    state = new_game(mini_board(), 2, random.Random(0), rules=COLONIST_1V1)
    assert copy_state(state).rules is COLONIST_1V1


def test_start_defaults_to_standard_rules():
    game = start(mini_board(), 2, random.Random(0))
    assert game._state.rules is STANDARD


def test_evaluator_win_bonus_uses_the_game_types_threshold():
    """The plain evaluator: a 10-VP position scores the win bonus under
    standard rules and not under colonist rules; the gap is exactly the
    bonus, so nothing else in the score moved."""
    std = new_game(mini_board(), 2, random.Random(0), rules=STANDARD)
    col = new_game(mini_board(), 2, random.Random(0), rules=COLONIST_1V1)
    _cities(std, 0, 5)
    _cities(col, 0, 5)
    evaluator = Evaluator(std.board)
    assert evaluator.score(col, 0) == pytest.approx(evaluator.score(std, 0) - WIN_SCORE)


def test_heximax_win_bonus_uses_the_game_types_threshold():
    std = new_game(mini_board(), 2, random.Random(0), rules=STANDARD)
    col = new_game(mini_board(), 2, random.Random(0), rules=COLONIST_1V1)
    _cities(std, 0, 5)
    _cities(col, 0, 5)
    evaluator = ViewEvaluator(std.board)
    hand = [float(c) for c in std.hands[0]]
    assert evaluator.score(col, 0, hand) == pytest.approx(
        evaluator.score(std, 0, hand) - WIN_SCORE
    )


def test_heximax_evaluation_cache_distinguishes_rules():
    std = new_game(mini_board(), 2, random.Random(0), rules=STANDARD)
    _cities(std, 0, 5)
    col = copy_state(std)
    col.rules = COLONIST_1V1
    evaluator = ViewEvaluator(std.board)
    standard_scores = evaluator.evaluate(std)
    colonist_scores = evaluator.evaluate(col)
    assert colonist_scores[0] == pytest.approx(standard_scores[0] - WIN_SCORE)
    assert colonist_scores == pytest.approx(ViewEvaluator(col.board).evaluate(col))
