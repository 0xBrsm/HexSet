# SPDX-License-Identifier: GPL-3.0-only
"""The game contract: a ruleset is only dealt to a table it is played at.

A ruleset and the seat counts it is played at are one contract: the
15-point game is a two-seat game, and a four-seat game under its settings is
not a format at all. `start` refuses to deal one.
"""
import random

import pytest

from hexset.game import Phase, start, to_move
from hexset.rules import (
    DUEL_VARIANT_GAME,
    DUEL_VARIANT,
    GAME_TYPES,
    STANDARD,
    STANDARD_GAME,
    GameType,
    game_type,
)

from helpers import mini_board


def a_board():
    return mini_board()


# --- what ships ------------------------------------------------------------


def test_the_two_shipped_types_are_the_two_that_are_played():
    assert set(GAME_TYPES) == {"standard", "duel-variant"}
    assert STANDARD_GAME.rules is STANDARD
    assert STANDARD_GAME.seats == (2, 3, 4)
    assert DUEL_VARIANT_GAME.rules is DUEL_VARIANT
    assert DUEL_VARIANT_GAME.seats == (2,)


def test_game_type_looks_up_by_name():
    assert game_type("duel-variant") is DUEL_VARIANT_GAME
    assert game_type("standard") is STANDARD_GAME
    with pytest.raises(ValueError, match="unknown game type"):
        game_type("ranked-1v1")


@pytest.mark.parametrize("live", [5])
def test_standard_does_not_play_beyond_four(live):
    assert not STANDARD_GAME.plays(live)
    with pytest.raises(ValueError, match="'standard' game type is played at"):
        STANDARD_GAME.check(live)


def test_a_refusal_names_what_does_ship():
    with pytest.raises(ValueError) as caught:
        DUEL_VARIANT_GAME.check(4)
    message = str(caught.value)
    assert "standard at 2, 3, 4" in message
    assert "duel-variant at 2" in message
    assert "Declare your own GameType" in message


# --- the contract at `start` ----------------------------------------------


def test_start_deals_the_types_rules():
    game = start(a_board(), 2, random.Random(0), game_type=DUEL_VARIANT_GAME)
    assert game.state(0, hidden=False).rules is DUEL_VARIANT


def test_a_four_seat_duel_is_refused():
    """The cell that prompted all of this. There is no four-player 15-point
    game; before the contract, this dealt one."""
    with pytest.raises(ValueError, match="'duel-variant' game type is played at 2 seats"):
        start(a_board(), 4, random.Random(0), game_type=DUEL_VARIANT_GAME)


# --- live seats, not dealt seats ------------------------------------------


def test_the_duel_accepts_four_dealt_with_two_retired():
    """What the server and the arena actually deal: four seats, two closed."""
    game = start(a_board(), 4, random.Random(0), game_type=DUEL_VARIANT_GAME, locked={2, 3})
    assert game.num_players == 4
    assert game.locked == frozenset({2, 3})
    assert game.state(0, hidden=False).rules is DUEL_VARIANT


def test_retiring_down_to_one_seat_is_refused():
    with pytest.raises(ValueError, match=r"not 1 \(4 dealt, 3 retired\)"):
        start(a_board(), 4, random.Random(0), locked={1, 2, 3})


def test_a_seat_that_is_not_at_the_table_cannot_retire():
    with pytest.raises(ValueError, match=r"cannot retire \[7\] at a 4-seat table"):
        start(a_board(), 4, random.Random(0), locked={7})


@pytest.mark.parametrize("retired", [{0, 1}])
def test_setup_opens_on_a_seat_that_is_playing(retired):
    """Retiring at `start` points the snake past the closed seats from the
    outset, whichever ones they are -- including the seat the snake would
    otherwise have opened on."""
    game = start(
        a_board(), 4, random.Random(0), game_type=DUEL_VARIANT_GAME, locked=retired
    )
    assert game.phase is Phase.SETUP_SETTLEMENT
    assert to_move(game) not in retired
    assert game.current_player == min(set(range(4)) - retired)


def test_first_may_be_a_retired_seat():
    game = start(
        a_board(), 4, random.Random(0), game_type=DUEL_VARIANT_GAME,
        locked={0, 1}, first=0,
    )
    assert game.first == 0
    assert to_move(game) == 2


# --- custom types are the consumer's ---------------------------------------


def test_a_custom_type_is_honoured():
    """Anything outside the two shipped formats is a consumer's to declare;
    the engine's job is to hold them to what they declared, not to refuse."""
    six_handed = GameType("house-six", STANDARD, (5, 6))
    game = start(a_board(), 6, random.Random(0), game_type=six_handed)
    assert game.num_players == 6
    with pytest.raises(ValueError, match="'house-six' game type is played at 5, 6 seats"):
        start(a_board(), 4, random.Random(0), game_type=six_handed)


@pytest.mark.parametrize(
    "seats, complaint",
    [
        ((), "declares no seat counts"),
        ((1, 2), "allows fewer than two seats"),
    ],
)
def test_a_game_type_checks_its_own_seat_list(seats, complaint):
    with pytest.raises(ValueError, match=complaint):
        GameType("nonsense", STANDARD, seats)


def test_a_game_type_needs_a_name():
    with pytest.raises(ValueError, match="needs a name"):
        GameType("", STANDARD, (2,))
