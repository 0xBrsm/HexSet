# SPDX-License-Identifier: GPL-3.0-only
"""`Entrant.no_trade` (`<entrant>!notrade`): a seat out of player trading,
whatever its gate declares. The table embargoes it with every other seat
(`arena.seat_bots`), so no exchange clears with it and no robber deal asks,
counts or spares it -- under either mechanism -- while a lineup without the
suffix plays exactly as before."""
from __future__ import annotations

import random

import pytest

from hexset.arena import MAX_ACTIONS, NO_TRADE, _play_one, entrant_from_name, lineup_from_names, seat_bots
from hexset.board.board import random_base_board
from hexset.game import start
from hexset.rules import STANDARD_GAME
from traders import TRADER  # registers `test-trader`
from journals import fresh

DEALER = f"{TRADER}^answer,ask"
# `test-trader` moves at random, so its games rarely finish; the first turns
# of each are enough to trade and deal in, through the arena's own game loop.
TURNS = 60


def _outcomes(names, games, trade_mode="round"):
    entrants = tuple(lineup_from_names(names))
    return [_play_one((entrants, index, 7, MAX_ACTIONS, True, trade_mode, STANDARD_GAME, TURNS, "random"), fresh())
            for index in range(games)]


def _touches(outcome, seat):
    """Every cleared exchange and robber deal in which `seat` took any part."""
    trades = [t for t in outcome.cleared if seat in (t.a, t.b)]
    deals = [d for d in outcome.deals
             if d.mover == seat or seat in d.promised or seat in d.accepted or seat in d.kept
             or getattr(d, "taker", None) == seat]
    return trades, deals


def test_the_suffix_marks_the_seat_and_keeps_the_rest_of_the_name():
    plain = entrant_from_name(DEALER)
    closed = entrant_from_name(DEALER + NO_TRADE)
    assert closed.no_trade and not plain.no_trade
    assert closed.name == DEALER + NO_TRADE
    assert closed.deals == plain.deals and closed.kind == plain.kind
    assert entrant_from_name("random!notrade").no_trade
    lineup = lineup_from_names([TRADER + NO_TRADE, TRADER, TRADER, TRADER])
    assert [e.no_trade for e in lineup] == [True, False, False, False]


def test_seat_bots_embargoes_a_no_trade_seat_with_every_other_and_nothing_else():
    rng = random.Random(0)
    game = start(random_base_board(rng), 4, rng)
    seat_bots(game, [None] * 4, no_trade=(2,))
    assert game.embargoes == frozenset({(2, 0), (2, 1), (2, 3)})
    rng = random.Random(0)
    untouched = start(random_base_board(rng), 4, rng)
    seat_bots(untouched, [None] * 4)
    assert untouched.embargoes == frozenset()


@pytest.mark.parametrize("trade_mode", ["round", "auto"])
def test_no_exchange_clears_with_a_no_trade_seat(trade_mode):
    games = 8
    control = _outcomes([TRADER] * 4, games, trade_mode)
    assert any(_touches(o, o.seating[0])[0] for o in control), "the control must trade, or the test proves nothing"
    closed = _outcomes([TRADER + NO_TRADE, TRADER, TRADER, TRADER], games, trade_mode)
    assert all(not _touches(o, o.seating[0])[0] for o in closed)
    assert any(o.cleared for o in closed), "the other seats still trade with each other"


def test_a_no_trade_seat_neither_asks_for_nor_takes_part_in_a_robber_deal():
    games = 12
    control = _outcomes([DEALER] * 4, games)
    assert any(_touches(o, o.seating[0])[1] for o in control), "the control must deal, or the test proves nothing"
    closed = _outcomes([DEALER + NO_TRADE, DEALER, DEALER, DEALER], games)
    for o in closed:
        _, deals = _touches(o, o.seating[0])
        assert all(d.mover != o.seating[0] for d in deals)
        assert not deals


def test_a_table_of_no_trade_seats_trades_and_deals_not_at_all():
    closed = _outcomes([DEALER + NO_TRADE] * 4, 8)
    assert all(not o.cleared for o in closed)
    assert all(not d.promised and not d.accepted for o in closed for d in o.deals)
