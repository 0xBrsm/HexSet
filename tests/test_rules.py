# SPDX-License-Identifier: GPL-3.0-only
"""Game types. `hexset.rules` carries what varies between them -- VPs to win,
the discard limit, the shield, the dice -- and every reader takes those off
the position rather than a module-level constant.
"""

from __future__ import annotations

import random

import pytest

from helpers import independent_vertices, mini_board

from hexset.game import Phase, _check_win, roll_dice, start
from hexset.rules import (
    DUEL_VARIANT_GAME,
    DUEL_VARIANT,
    GAME_TYPES,
    STANDARD,
    STANDARD_GAME,
    Rules,
)
from hexset.state import Building
from hexset.victory import relative_points


def test_presets_carry_the_documented_numbers():
    assert (STANDARD.winning_points, STANDARD.discard_limit) == (10, 7)
    assert (DUEL_VARIANT.winning_points, DUEL_VARIANT.discard_limit) == (15, 9)
    assert GAME_TYPES["standard"] is STANDARD_GAME
    assert GAME_TYPES["duel-variant"] is DUEL_VARIANT_GAME
    assert STANDARD_GAME.rules is STANDARD
    assert DUEL_VARIANT_GAME.rules is DUEL_VARIANT


def test_rules_reject_nonsense():
    with pytest.raises(ValueError):
        Rules(winning_points=0)
    with pytest.raises(ValueError):
        Rules(discard_limit=-1)


def _cities(state, player, count):
    for v in independent_vertices(state.board, count):
        state.vertex_owner[v] = player
        state.vertex_building[v] = Building.CITY


def test_check_win_uses_the_game_types_threshold():
    game = start(mini_board(), 2, random.Random(0), game_type=DUEL_VARIANT_GAME)
    game.phase = Phase.MAIN
    game.current_player = 0
    _cities(game._state, 0, 5)
    _check_win(game)
    assert game.won_by is None
    assert game.phase is Phase.MAIN
    _cities(game._state, 0, 8)
    _check_win(game)
    assert game.won_by == 0
    assert game.phase is Phase.GAME_OVER


def test_seven_discard_quotas_use_the_game_types_limit():
    game = start(mini_board(), 2, random.Random(0), game_type=DUEL_VARIANT_GAME)
    game.phase = Phase.ROLL
    game.current_player = 0
    game._state.hands[0] = [2, 2, 2, 2, 0]  # 8 cards: under the duel limit
    game._state.hands[1] = [0, 0, 0, 0, 0]
    assert roll_dice(game, 7) == 7
    assert game.discard_quota == [0, 0]
    assert game.phase is Phase.ROBBER

    game.phase = Phase.ROLL
    game._state.hands[0] = [2, 2, 2, 2, 2]
    assert roll_dice(game, 7) == 7
    assert game.discard_quota == [5, 0]
    assert game.phase is Phase.DISCARD


def test_relative_points_are_scaled_by_the_game_target():
    """The same finish in a longer game is not a bigger one: the divisor is the
    points the game was played to, not always ten."""
    points = (10, 6)
    assert relative_points(points) == pytest.approx((0.4, -0.4))
    assert relative_points(
        (15, 11), winning_points=DUEL_VARIANT.winning_points
    ) == pytest.approx((4 / 15, -4 / 15))
    # Zero-sum at any scale.
    assert sum(relative_points((15, 9, 8), winning_points=15)) == pytest.approx(0.0)


def test_seven_odds_follow_the_dice_the_rules_roll():
    from hexset.chance import BALANCED_SEVEN_ODDS, SEVEN_ODDS

    assert STANDARD.seven_odds == SEVEN_ODDS == 6 / 36
    assert DUEL_VARIANT.seven_odds == BALANCED_SEVEN_ODDS
    assert Rules(balanced_dice=True).seven_odds == BALANCED_SEVEN_ODDS
