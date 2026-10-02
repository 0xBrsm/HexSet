# SPDX-License-Identifier: GPL-3.0-only
"""When the turn's one trade event opens: on entering MAIN for a gate that
does not say, or at the first main-phase point its `trade_now` says yes."""

from __future__ import annotations

import random

from hexset.actions import Action, ActionType, apply
from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.bots import RandomBot
from hexset.game import Phase, run_trade_event, start
from hexset.record import moves, record_game, replay
from helpers import deal

WOOD, BRICK, SHEEP, WHEAT, ORE = (int(r) for r in Resource)


class Taker:
    """Wants one resource: gains from any exchange that brings it that
    resource and costs it nothing else it holds of that kind."""

    trade_floor = 0.0

    def __init__(self, wants=ORE):
        self.wants = wants

    def gains_many(self, view, received, counterparties):
        return [1.0 if r[self.wants] > 0 else -1.0 for r in received]


class Asker(Taker):
    """A `Taker` that also says when to trade, from a script of answers."""

    def __init__(self, answers):
        super().__init__(ORE)
        self.answers = list(answers)
        self.asked = 0

    def trade_now(self, game):
        self.asked += 1
        return self.answers.pop(0) if self.answers else False


def a_table(actor):
    rng = random.Random(0)
    game = start(random_base_board(rng), 4, rng)
    game.phase = Phase.MAIN
    game.current_player = 0
    deal(game, 0, Resource.WOOD, 16)
    deal(game, 1, Resource.ORE, 1)
    game.gates = (actor, Taker(WOOD), None, None)
    return game


def a_bank_trade(game):
    return Action(ActionType.BANK_TRADE, WOOD, BRICK)


def test_a_gate_without_trade_now_trades_on_entering_main():
    game = a_table(Taker())
    run_trade_event(game)
    assert game.trade_event_turn == game.turns
    assert [(t.a, t.b, t.received[ORE]) for t in game.trades] == [(0, 1, 1)]


def test_a_no_leaves_the_window_open_and_the_next_yes_opens_it():
    actor = Asker([False, False, True])
    game = a_table(actor)
    run_trade_event(game)                         # entering MAIN: no
    assert actor.asked == 1 and not game.trades
    assert game.trade_event_turn != game.turns
    apply(game, a_bank_trade(game))               # after a main-phase action: no
    assert actor.asked == 2 and not game.trades
    before = len(game.trades)
    apply(game, a_bank_trade(game))               # yes: the event runs inside this apply
    assert actor.asked == 3
    assert game.trade_event_turn == game.turns
    assert [(t.a, t.b, t.received[ORE]) for t in game.trades[before:]] == [(0, 1, 1)]
    apply(game, a_bank_trade(game))               # used: never asked again this turn
    assert actor.asked == 3


def test_a_turn_it_never_says_yes_in_does_not_trade():
    actor = Asker([])
    game = a_table(actor)
    run_trade_event(game)
    apply(game, a_bank_trade(game))
    apply(game, Action(ActionType.END_TURN))
    assert not game.trades
    assert game.phase is Phase.ROLL               # the next seat's turn; nobody asked there
    assert actor.asked == 2


class LateTrader(RandomBot):
    """A random player that trades ore-for-anything and says yes on its
    second main-phase decision point of a turn -- never the first. Checks
    the hook's contract as it goes: only in MAIN, never during free roads,
    at most once between two of its own choices."""

    trade_floor = 0.0

    def __init__(self, rng):
        super().__init__(rng)
        self.asked_since_choice = False
        self.turn = None
        self.points = 0
        self.yes = 0

    def gains_many(self, view, received, counterparties):
        return [1.0 if sum(r) >= 0 else -1.0 for r in received]

    def choose(self, game):
        self.asked_since_choice = False
        return super().choose(game)

    def trade_now(self, game):
        assert game.phase is Phase.MAIN and game.free_roads == 0
        assert not self.asked_since_choice, "asked twice at one decision point"
        self.asked_since_choice = True
        if self.turn != game.turns:
            self.turn, self.points = game.turns, 0
        self.points += 1
        if self.points == 2:
            self.yes += 1
            return True
        return False


def test_a_recorded_game_files_a_mid_turn_trade_at_its_step_and_replays():
    rng = random.Random(4)
    board = random_base_board(rng)
    bots = [LateTrader(random.Random(40 + s)) for s in range(4)]
    # A short capped game rather than a finished one: random hands only grow,
    # and every trade event prices the whole enumeration of them.
    record = record_game(bots, board, 4, turn_cap=12)
    assert sum(b.yes for b in bots) > 0
    kinds = {ActionType(record.actions[step][0]) for step, *_ in record.trades}
    assert record.trades, "nobody traded"
    # Never on entering MAIN from the roll: that is always a seat's first point.
    assert ActionType.ROLL not in kinds
    game = replay(record)   # raises unless it ends on the record's own turn
    assert game.turns == record.turns == 12
    assert sum(len(t) for _, _, t in moves(record)) == len(record.trades)
