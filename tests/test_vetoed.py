# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.bots.Vetoed`: a second gate that can only remove the first one's
exchanges, and the `trader_veto` route that seats a checkpoint over its
trader with its own gate as that veto."""
from __future__ import annotations

import random

import pytest

from hexset.arena import traded
from hexset.board.board import random_base_board
from hexset.bots import TradesBy, Vetoed, seat_at
from hexset.clients._modelmeta import trader_veto_config, trader_veto_of
from hexset.game import Phase, start
from hexset.trading import (
    RESPONSE_ACCEPT, RESPONSE_COUNTER, RESPONSE_PASS, Offer, Response, TradeParams, bundle,
)
from traders import TRADER, ScarcityTrader

GOOD, BAD = bundle(brick=1, wood=-1), bundle(ore=1, sheep=-1)


def a_game():
    rng = random.Random(0)
    game = start(random_base_board(rng), 4, rng)
    game.phase = Phase.MAIN
    game.current_player = 0
    return game


class Proposer:
    """Offers, accepts, counters and picks whatever it is told to."""

    trade_floor = 0.5

    def __init__(self, answer=None):
        self.answer = answer
        self.seen = None
        self.seated = []

    def candidates(self, view, counterparties, *, turn=None, already_offered=()):
        return [(c, b) for c in counterparties for b in (GOOD, BAD)]

    def respond(self, view, offer):
        return self.answer

    def respond_any(self, view, offer):
        return self.answer

    def pick(self, view, responses):
        self.seen = list(responses)
        return next((i for i, r in enumerate(responses) if r.kind != RESPONSE_PASS), None)

    def gains_many(self, view, received, counterparties):
        return [1.0] * len(received)

    def seat_at(self, game):
        self.seated.append(game)


class Veto:
    """Gains from `GOOD` only, at a floor of zero."""

    trade_floor = 0.0

    def __init__(self):
        self.seated = []

    def gains_many(self, view, received, counterparties):
        return [0.01 if tuple(b) == GOOD else -0.01 for b in received]

    def seat_at(self, game):
        self.seated.append(game)


def test_offers_keep_only_what_the_veto_gains_from():
    gate = Vetoed(Proposer(), Veto())
    view = a_game().state(0)
    assert gate.candidates(view, [1, 2], turn=0) == [(1, GOOD), (2, GOOD)]


@pytest.mark.parametrize("kind", [RESPONSE_ACCEPT, RESPONSE_COUNTER])
def test_an_answer_the_veto_refuses_is_a_pass(kind):
    view = a_game().state(0)
    # Answers are signed towards the actor: the veto reads the negation.
    refused = Vetoed(Proposer(Response(0, kind, tuple(-n for n in BAD))), Veto())
    kept = Vetoed(Proposer(Response(0, kind, tuple(-n for n in GOOD))), Veto())
    offer = Offer(1, tuple(-n for n in BAD))
    assert refused.respond(view, offer) == Response(0, RESPONSE_PASS, None)
    assert kept.respond(view, offer).kind == kind
    assert refused.respond_any(view, offer) == Response(0, RESPONSE_PASS, None)


def test_a_pass_is_never_touched():
    view = a_game().state(0)
    gate = Vetoed(Proposer(Response(0, RESPONSE_PASS, None)), Veto())
    assert gate.respond(view, Offer(1, GOOD)) == Response(0, RESPONSE_PASS, None)


def test_the_proposer_picks_among_what_the_veto_leaves():
    view = a_game().state(0)
    proposer = Proposer()
    gate = Vetoed(proposer, Veto())
    answers = [Response(1, RESPONSE_ACCEPT, BAD), Response(2, RESPONSE_COUNTER, GOOD)]
    assert gate.pick(view, answers) == 1
    assert proposer.seen[0] == Response(1, RESPONSE_PASS, None)


def test_gains_are_the_proposers_where_the_veto_agrees():
    view = a_game().state(0)
    assert Vetoed(Proposer(), Veto()).gains_many(view, [GOOD, BAD], [1, 1]) == [1.0, -1.0]


def test_seating_reaches_both_gates_and_names_read_through_to_the_proposer():
    game = a_game()
    proposer, veto = Proposer(), Veto()
    seat = TradesBy(object(), Vetoed(proposer, veto))
    seat_at(seat, game)
    assert proposer.seated == [game] and veto.seated == [game]
    assert seat.trade_floor == 0.5


def test_traded_with_veto_seats_the_mover_as_its_traders_veto():
    game = a_game()
    mover = ScarcityTrader(random.Random(1))
    seat = traded(mover, TRADER, game._state.board, random.Random(2), veto=True)
    assert isinstance(seat, TradesBy) and isinstance(seat.trader, Vetoed)
    assert seat.trader.veto is mover and seat.mover is mover
    plain = traded(mover, TRADER, game._state.board, random.Random(2))
    assert not isinstance(plain.trader, Vetoed)


def test_a_seat_that_does_not_trade_cannot_veto():
    game = a_game()
    mover = ScarcityTrader(random.Random(1), TradeParams(max_offers=0))
    with pytest.raises(ValueError, match="cannot veto"):
        traded(mover, TRADER, game._state.board, random.Random(2), veto=True)


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("on", True),
                                            ("0", False), ("", False), (None, False), ("nope", False)])
def test_the_metadata_key_reads_as_a_switch(value, expected):
    meta = {} if value is None else {"trader_veto": value}
    assert trader_veto_config(meta) is expected


def test_a_checkpoint_without_the_key_does_not_veto():
    class Loaded:
        trader = TRADER

    assert trader_veto_of(Loaded()) is False
