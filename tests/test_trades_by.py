# SPDX-License-Identifier: GPL-3.0-only
"""`TradesBy`: one bot's moves over another's trading, and the `~` lineup
spelling that seats it."""
from __future__ import annotations

import pickle
import random

import pytest

from hexset.arena import MAX_ACTIONS, _play_one, entrant_from_name, lineup_from_names, spawn
from hexset.board.board import random_base_board
from hexset.bots import RandomBot, TradesBy
from hexset.rules import STANDARD_GAME
from hexset.trading import TradeParams, retune
from traders import ScarcityTrader


def test_moves_are_the_movers_and_every_other_name_is_the_traders():
    trader = spawn(entrant_from_name("test-trader"), random_base_board(random.Random(0)), random.Random(1))
    mover = RandomBot(random.Random(2))
    seat = TradesBy(mover, trader)
    assert seat.choose.__self__ is seat and seat.mover is mover
    assert seat.gains_many == trader.gains_many
    assert seat.trade_params is trader.trade_params
    assert seat.trade_offer_budget == trader.trade_offer_budget == 2

    # A run's override reaches whoever actually trades.
    retune(seat, max_offers=0)
    assert trader.trade.max_offers == 0 and seat.trade_offer_budget == 0

    again = pickle.loads(pickle.dumps(TradesBy(RandomBot(random.Random(2)), RandomBot(random.Random(3)))))
    assert isinstance(again.mover, RandomBot) and isinstance(again.trader, RandomBot)


def test_a_lineup_names_a_trader_after_a_tilde():
    lineup = lineup_from_names(["network:/tmp/x.pt~test-trader", "random~test-trader", "random", "random"])
    assert [e.name for e in lineup] == ["network~test-trader", "random~test-trader", "random#0", "random#1"]
    assert (lineup[0].kind, lineup[0].weights, lineup[0].trader) == ("network", "/tmp/x.pt", "test-trader")
    assert lineup[2].trader is None
    assert pickle.loads(pickle.dumps(lineup)) == lineup

    board = random_base_board(random.Random(0))
    seat = spawn(lineup[1], board, random.Random(4))
    assert isinstance(seat, TradesBy)
    assert isinstance(seat.mover, RandomBot) and isinstance(seat.trader, ScarcityTrader)


def test_a_trader_is_named_once_and_must_exist():
    with pytest.raises(ValueError, match="unknown bots"):
        lineup_from_names(["random~nobody"])
    with pytest.raises(ValueError, match="one trader"):
        entrant_from_name("random~test-trader~random")
    with pytest.raises(ValueError, match="no gate of its own"):
        from dataclasses import replace

        replace(entrant_from_name("random~test-trader"), trade=TradeParams(max_offers=0))


def test_a_random_mover_trades_through_a_registered_gate_at_a_real_table():
    """Random play never trades on its own; through another bot's gate it
    clears exchanges, while every move stays random."""
    entrants = tuple(lineup_from_names(["random~test-trader"] + ["test-trader"] * 3))
    alone = tuple(lineup_from_names(["random"] + ["test-trader"] * 3))

    def own_trades(lineup, index):
        outcome = _play_one((lineup, index, 7, MAX_ACTIONS, True, True, "round", STANDARD_GAME, 1200))
        return sum(1 for t in outcome.cleared if outcome.seating[0] in (t.a, t.b))

    assert sum(own_trades(alone, i) for i in range(4)) == 0
    assert sum(own_trades(entrants, i) for i in range(4)) > 0


def test_retuning_a_seat_whose_trader_carries_a_protocol_reinstalls_it_on_the_trader():
    """The hooks live on the trader, so a retune through the seat sets and
    takes them back there."""
    from dataclasses import replace

    from hexset.trading import TradeProtocol, install
    from traders import FRAGMENTED

    trader = ScarcityTrader(random.Random(1), replace(FRAGMENTED, fragment_trades=False))
    install(trader, TradeProtocol(trader, trader.trade, seed=0))
    seat = TradesBy(RandomBot(random.Random(2)), trader)
    assert seat.respond.__self__ is trader._protocol

    retune(seat, max_give_cards=1)
    assert trader.trade.max_give_cards == 1 and trader._protocol.params.max_give_cards == 1
    retune(seat, max_offers=0)
    assert getattr(trader, "respond", None) is None and getattr(seat, "pick", None) is None
