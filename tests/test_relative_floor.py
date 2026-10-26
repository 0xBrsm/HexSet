# SPDX-License-Identifier: GPL-3.0-only
"""`TradeParams.trade_floor_relative`: a gate's floor read relative to the
table's points, where the deciding seat's view is at hand."""
from __future__ import annotations

import random

import pytest

from hexset.board.board import random_base_board
from hexset.actions import apply
from hexset.bots import RandomBot, Vetoed
from hexset.game import Phase, start
from hexset.trading import TradeParams, bundle, clears_floor, trade_floor_of
from hexset.trading._engine import relative_points_factor
from hexset.victory import public_victory_points, victory_points


def a_game():
    """A game past its setup, every seat on two points, seat 0 to act."""
    rng = random.Random(0)
    game = start(random_base_board(rng), 4, rng)
    bot = RandomBot(random.Random(1))
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, bot.choose(game))
    game.phase = Phase.MAIN
    game.current_player = 0
    return game


def behind(game):
    """Seat 1 a city up on seat 0: two points ahead in public."""
    st = game._state
    v = next(v for v in range(len(st.vertex_owner)) if st.vertex_owner[v] == 1)
    st.vertex_building[v] = 2
    return game


class Gate:
    """Gains `gain` from everything, under `params`."""

    def __init__(self, gain, params):
        self.gain = gain
        self.trade_params = params
        self.trade_floor = params.trade_floor

    def gains_many(self, view, received, counterparties):
        return [self.gain] * len(received)


def test_the_metadata_key_round_trips():
    on = TradeParams.from_meta({"trade_floor": "0.036", "trade_floor_relative": "1"})
    assert on.trade_floor_relative and on.trade_floor == 0.036
    assert TradeParams.from_meta(on.as_meta()) == on
    off = TradeParams.from_meta({"trade_floor": "0.03"})
    assert not off.trade_floor_relative and "trade_floor_relative" not in off.as_meta()


def test_the_factor_is_own_points_over_the_table_s_points():
    game = a_game()
    view = game.state(0)
    st = view.state
    own = victory_points(st, 0)
    table = own + sum(public_victory_points(st, s) for s in range(1, 4))
    assert relative_points_factor(view) == pytest.approx(own / max(table, 1))


def test_a_relative_floor_scales_only_where_a_view_is_given():
    game = a_game()
    view = game.state(0)
    gate = Gate(0.0, TradeParams(trade_floor=0.04, trade_floor_relative=True))
    factor = relative_points_factor(view)
    assert trade_floor_of(gate) == 0.04
    assert trade_floor_of(gate, view) == pytest.approx(0.04 * factor)
    plain = Gate(0.0, TradeParams(trade_floor=0.04))
    assert trade_floor_of(plain, view) == 0.04


def test_behind_on_points_the_veto_clears_what_level_it_would_not():
    gate = Gate(0.0, TradeParams(trade_floor=0.04, trade_floor_relative=True))
    assert relative_points_factor(a_game().state(0)) == pytest.approx(1 / 4)
    view = behind(a_game()).state(0)
    assert relative_points_factor(view) == pytest.approx(2 / 9)
    gate = Gate(0.0, TradeParams(trade_floor=0.12, trade_floor_relative=True))
    gain = 0.028
    assert clears_floor(gain, gate, view) and not clears_floor(gain, gate)
    assert not clears_floor(gain, gate, a_game().state(0))


def test_vetoed_reads_the_relative_floor_from_its_view():
    game = behind(a_game())
    view = game.state(0)
    factor = relative_points_factor(view)

    class Proposer:
        trade_floor = 0.5

        def candidates(self, view, counterparties, turn=None, already_offered=()):
            return [(1, bundle(brick=1, wood=-1))]

    gain = 0.04 * factor * 1.1
    veto = Gate(gain, TradeParams(trade_floor=0.04, trade_floor_relative=True))
    assert Vetoed(Proposer(), veto).candidates(view, [1], turn=0) == [(1, bundle(brick=1, wood=-1))]
    veto = Gate(0.04 * factor * 0.9, TradeParams(trade_floor=0.04, trade_floor_relative=True))
    assert Vetoed(Proposer(), veto).candidates(view, [1], turn=0) == []
