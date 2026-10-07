# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.bots.Handoff`: one bot's moves through a round and another's after
it, and the `handoff:<round>:<first>|<second>` spelling that seats it."""
from __future__ import annotations

import pickle
import random

import pytest

from hexset.arena import (
    MAX_ACTIONS, _play_one, deal_board, deal_game, entrant_from_name, lineup_from_names,
    play_game, spawn,
)
from hexset.bots import Handoff, RandomBot, TradesBy, handed_off, play_against, seat_at
from hexset.rules import STANDARD_GAME
from hexset.trading import retune
from traders import ScarcityTrader


class Recording(RandomBot):
    """Random moves, each one noted with the turn count it was made at."""

    def __init__(self, seed):
        super().__init__(random.Random(seed))
        self.at = []

    def choose(self, game):
        self.at.append(game.turns)
        return super().choose(game)


class Told:
    """A bot that only records what it was seated at and told to play against."""

    def __init__(self):
        self.seated, self.told = [], []

    def choose(self, game):  # pragma: no cover - never seated to move
        raise AssertionError("never asked to move")

    def seat_at(self, game):
        self.seated.append(game)

    def play_against(self, seats):
        self.told.append(frozenset(seats))


def test_moves_are_first_s_through_the_round_and_second_s_after():
    first, second = Recording(1), Recording(2)
    bots = [Handoff(first, second, 3)] + [RandomBot(random.Random(s)) for s in (3, 4, 5)]
    game = play_game(deal_game(5, 0, 4), bots)
    # Three rounds of four seats: every turn count below 12 is first's
    # (setup included), every one from 12 on second's.
    assert first.at and second.at and game.turns > 12
    assert max(first.at) < 12 <= min(second.at)


def test_the_round_counts_the_seats_still_playing():
    game = deal_game(5, 0, 4, locked=(3,))
    game.turns = 5
    assert not handed_off(game, 2) and handed_off(game, 1)
    game.turns = 6
    assert handed_off(game, 2)
    assert handed_off(game, 0)


def test_every_other_name_is_first_s():
    board = random_base_board_for_test()
    trader = spawn(entrant_from_name("test-trader"), board, random.Random(1))
    seat = Handoff(trader, RandomBot(random.Random(2)), 5)
    assert seat.gains_many == trader.gains_many
    assert seat.trade_params is trader.trade_params
    retune(seat, max_offers=0)
    assert trader.trade.max_offers == 0 and seat.trade_offer_budget == 0

    again = pickle.loads(pickle.dumps(Handoff(RandomBot(random.Random(2)), RandomBot(random.Random(3)), 7)))
    assert isinstance(again.first, RandomBot) and isinstance(again.second, RandomBot) and again.at == 7

    with pytest.raises(ValueError, match="count of rounds"):
        Handoff(RandomBot(), RandomBot(), -1)


def test_both_bots_are_seated_and_told():
    a, b = Told(), Told()
    seat = Handoff(a, b, 4)
    game = deal_game(5, 0, 4)
    seat_at(seat, game)
    play_against(seat, {1, 2})
    assert a.seated == b.seated == [game]
    assert a.told == b.told == [frozenset({1, 2})]


def test_the_spec_names_the_round_and_both_bots():
    entrant = entrant_from_name("handoff:14:random|test-trader")
    assert (entrant.name, entrant.kind) == ("handoff:14:random|test-trader", "handoff")
    assert (entrant.option("at"), entrant.option("first"), entrant.option("second")) == (14, "random", "test-trader")
    assert pickle.loads(pickle.dumps(entrant)) == entrant

    seat = spawn(entrant, random_base_board_for_test(), random.Random(4))
    assert isinstance(seat, Handoff) and seat.at == 14
    assert isinstance(seat.first, RandomBot) and isinstance(seat.second, ScarcityTrader)

    # A `~` names the whole seat's trader; a second spec nests.
    traded = spawn(entrant_from_name("handoff:2:random|random~test-trader"), random_base_board_for_test(), random.Random(4))
    assert isinstance(traded, TradesBy) and isinstance(traded.mover, Handoff)
    assert isinstance(traded.trader, ScarcityTrader)
    nested = entrant_from_name("handoff:2:random|handoff:9:random|test-trader")
    assert nested.option("second") == "handoff:9:random|test-trader"

    lineup = lineup_from_names(["handoff:1:random|random"] * 2 + ["random"] * 2)
    assert [e.name for e in lineup] == ["handoff:1:random|random#0", "handoff:1:random|random#1", "random#0", "random#1"]


@pytest.mark.parametrize("name, match", [
    ("handoff:x:random|random", "reads handoff"),
    ("handoff:3:random", "reads handoff"),
    ("handoff:3:|random", "reads handoff"),
    ("handoff::random|random", "reads handoff"),
    ("handoff:3:random|nobody", "unknown bot"),
    ("handoff:3:coalition:random|random", "seated whole"),
])
def test_a_malformed_spec_is_refused(name, match):
    with pytest.raises(ValueError, match=match):
        entrant_from_name(name)


def test_a_handoff_plays_as_its_first_until_it_hands_over():
    """`second` draws nothing from the seat's generator, so a handoff past the
    end of the game is its `first`, game for game."""
    def outcome(name, index):
        lineup = tuple(lineup_from_names([name, "random", "random", "random"]))
        o = _play_one((lineup, index, 7, MAX_ACTIONS, True, True, "round", STANDARD_GAME, 1200))
        return o.winner, o.turns, o.points

    for index in range(3):
        alone = outcome("test-trader", index)
        assert outcome("handoff:999:test-trader|random", index) == alone
    assert any(outcome("handoff:0:test-trader|random", i) != outcome("test-trader", i) for i in range(3))


def random_base_board_for_test():
    return deal_board(0, 0)
