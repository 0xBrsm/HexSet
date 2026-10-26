# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.bots.Closes`: a closer's move wherever it names one, the mover's
everywhere else, and the `closer` metadata key that seats a checkpoint
under one."""
from __future__ import annotations

import random

import pytest

from hexset.actions import Action, ActionType, options_for
from hexset.arena import Entrant, closed, register_entrant_kind, register_preset
from hexset.board.board import random_base_board
from hexset.bots import Closes, TradesBy, play_against, seat_at
from hexset.clients._modelmeta import closer_config, closer_of
from hexset.clients.netbot import declared_closer
from hexset.game import Phase, start


def a_game():
    rng = random.Random(0)
    game = start(random_base_board(rng), 4, rng)
    game.phase = Phase.MAIN
    game.current_player = 0
    return game


class Mover:
    """Plays the first legal move, and records what it is asked and told."""

    trade_floor = 0.25

    def __init__(self):
        self.asked = 0
        self.seated = []
        self.against = None

    def choose(self, game):
        self.asked += 1
        return options_for(game)[0]

    def seat_at(self, game):
        self.seated.append(game)

    def play_against(self, seats):
        self.against = tuple(seats)


class Closer:
    """Names `move(game)` as its closing move."""

    def __init__(self, move):
        self.move = move
        self.seated = []

    def closing(self, game):
        return self.move(game)

    def seat_at(self, game):
        self.seated.append(game)


def test_the_closer_s_move_is_played_where_it_names_one():
    game = a_game()
    last = options_for(game)[-1]
    mover = Mover()
    assert Closes(mover, Closer(lambda g: last)).choose(game) == last
    assert mover.asked == 0


def test_the_mover_chooses_where_the_closer_names_none():
    game = a_game()
    mover = Mover()
    assert Closes(mover, Closer(lambda g: None)).choose(game) == options_for(game)[0]
    assert mover.asked == 1


def test_a_closer_s_illegal_move_is_never_played():
    game = a_game()
    illegal = Action(ActionType.BUILD_CITY, 0, 0)
    assert illegal not in options_for(game)
    mover = Mover()
    assert Closes(mover, Closer(lambda g: illegal)).choose(game) == options_for(game)[0]
    assert mover.asked == 1


def test_every_other_name_is_the_mover_s():
    mover = Mover()
    seat = Closes(mover, Closer(lambda g: None))
    assert seat.trade_floor == 0.25
    seat.trade_floor = 0.5
    assert mover.trade_floor == 0.5


def test_seating_reaches_the_mover_and_the_closer_and_through_a_trader():
    game = a_game()
    mover, closer = Mover(), Closer(lambda g: None)
    seat_at(TradesBy(Closes(mover, closer), Mover()), game)
    assert mover.seated == [game] and closer.seated == [game]
    play_against(Closes(mover, closer), (1, 2))
    assert mover.against == (1, 2)


def test_closed_without_a_closer_is_the_bot_itself():
    mover = Mover()
    assert closed(mover, None, a_game().state(0, hidden=False).board, random.Random(0)) is mover


def test_closed_builds_the_named_closer_without_drawing_from_the_seat_s_generator():
    register_entrant_kind("test-closer", lambda entrant, board, rng: Closer(lambda g: None))
    register_preset("test-closer", Entrant("test-closer", kind="test-closer"), listed=False)
    rng = random.Random(7)
    state = rng.getstate()
    seat = closed(Mover(), "test-closer", a_game().state(0, hidden=False).board, rng)
    assert isinstance(seat, Closes) and isinstance(seat.closer, Closer)
    assert rng.getstate() == state


@pytest.mark.parametrize("meta,expected", [({}, None), ({"closer": ""}, None), ({"closer": "x"}, "x")])
def test_the_metadata_key_names_the_closer(meta, expected):
    assert closer_config(meta) == expected


def test_a_checkpoint_naming_no_closer_declares_none():
    class Loaded:
        pass

    assert closer_of(Loaded()) is None
    assert declared_closer(Loaded(), "f.onnx") is None


def test_a_closer_nothing_registers_is_refused_naming_the_file_and_key():
    class Loaded:
        closer = "no-such-closer-kind"

    with pytest.raises(ValueError, match=r"f\.onnx: metadata key `closer`"):
        declared_closer(Loaded(), "f.onnx")
