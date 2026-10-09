# SPDX-License-Identifier: GPL-3.0-only
"""Robber deals as the seat asked (`hexset.deals`): what a promise is worth,
the plain reading of a mover, and who answers at all."""
from __future__ import annotations

import random

import pytest

from hexset import deals as D
from hexset.actions import Action, ActionType
from hexset.board.board import random_base_board
from hexset.chance import UNSEEN
from hexset.game import Phase, observe, start
from hexset.robber import occupants
from hexset.seat import Seat
from hexset.trading import RESPONSE_ACCEPT, RESPONSE_COUNTER, RESPONSE_PASS, Offer

from helpers import clear_hand, deal
from tests.test_game import run_setup

WOOD, BRICK, SHEEP, WHEAT, ORE = range(5)


class Gate:
    """A gate that prices every card at one."""

    trade_floor = 0.0

    def gains_many(self, view, received, counterparties):
        return [float(sum(r)) for r in received]


def a_robber_move(seed: int = 0):
    """Seat 0 to move the robber; seats 1-3 holding three wood each."""
    game = run_setup(start(random_base_board(random.Random(seed)), 4, random.Random(seed)))
    for s in range(4):
        clear_hand(game._state, s)
    deal(game, 0, ORE)
    for s in (1, 2, 3):
        deal(game, s, WOOD, 3)
    game.current_player = 0
    game.phase = Phase.ROBBER
    return game


def block_chance(model, game, seat, promised=None):
    state = game._state
    hexes, _ = D.hit_chances(model, state, 0, None, promised or {}, seat)
    return sum(p for h, p in hexes.items() if seat in occupants(state, h))


ANSWERS = D.DealParams(answer=True)
ASK = D.Ask(0, D.NO_BLOCK, WOOD)


def test_params_name_only_known_kinds_and_answer_nothing_by_default():
    assert D.DealParams().answer is False
    assert D.DealParams(kinds=(D.NO_STEAL, D.NO_BLOCK)).kinds == D.KINDS
    with pytest.raises(ValueError, match="deal kinds"):
        D.DealParams(kinds=("bribe",))


def test_the_seat_the_mover_will_hit_promises_and_one_it_will_not_does_not():
    game = a_robber_move()
    model, gate = D.PlainMover(), Gate()
    chance = {s: block_chance(model, game, s) for s in (1, 2, 3)}
    target = max(chance, key=chance.get)
    value = {s: D.promise_value(game, s, gate, ASK, (), model) for s in (1, 2, 3)}
    # The gate prices a one-for-one swap at nothing, so a promise is worth
    # only the robbery it saves.
    assert value[target] > 0 and value[target] == max(value.values())
    assert D.default_answer(game, target, gate, ASK, (), model)
    spared = min(chance, key=chance.get)
    if chance[spared] < 1e-6:
        assert not D.default_answer(game, spared, gate, ASK, (), model)


def test_a_no_steal_promise_is_worth_the_steal_it_saves():
    game = a_robber_move()
    model, gate = D.PlainMover(), Gate()
    ask = D.Ask(0, D.NO_STEAL, WOOD)
    for s in (1, 2, 3):
        hexes, robbed = D.hit_chances(model, game._state, 0, None, {}, s)
        stolen = sum(p * robbed[h] for h, p in hexes.items())
        value = D.promise_value(game, s, gate, ask, (), model)
        assert (value > 0) == (stolen > 1e-6)


def test_the_plain_mover_reads_promises_as_kept():
    game = a_robber_move()
    state = game._state
    model = D.PlainMover()
    free = model.hexes(state, 0, None, {})
    assert abs(sum(free.values()) - 1) < 1e-9
    assert block_chance(model, game, 1, {1: D.NO_BLOCK}) < 1e-6 < block_chance(model, game, 1)
    for h in free:
        robbed = model.victims(state, 0, h, {1: D.NO_STEAL})
        assert robbed.get(1, 0.0) < 1e-6 or set(robbed) == {1}


def test_only_a_seat_that_answers_this_kind_and_holds_the_card_is_asked():
    game = a_robber_move()
    asked = []

    class Own(Gate):
        def deal_answer(self, game, seat, ask, promised, model):
            asked.append((seat, promised, type(model)))
            return True

    gate = Own()
    assert not D.answer(game, 1, gate, None, ASK)
    assert not D.answer(game, 1, gate, D.DealParams(), ASK)
    assert not D.answer(game, 1, gate, D.DealParams(answer=True, kinds=(D.NO_STEAL,)), ASK)
    assert not D.answer(game, 1, gate, ANSWERS, D.Ask(0, D.NO_BLOCK, ORE))
    assert not D.answer(game, 0, gate, ANSWERS, D.Ask(0, D.NO_BLOCK, ORE))
    assert asked == []
    assert D.answer(game, 2, gate, ANSWERS, ASK, (1,))
    assert asked == [(2, (1,), D.PlainMover)]


def test_a_hosted_seat_answers_from_its_own_view():
    game = a_robber_move()
    model = D.PlainMover()
    chance = {s: block_chance(model, game, s) for s in (1, 2, 3)}
    target = max(chance, key=chance.get)
    seat = Seat(observe(game, target), target, object(), Gate(), deals=ANSWERS)
    assert seat.deal_answer(0, D.NO_BLOCK, WOOD)
    assert not Seat(observe(game, target), target, object(), Gate()).deal_answer(0, D.NO_BLOCK, WOOD)


def test_a_robber_model_registers_once():
    D.register_robber_model("plain", D.PlainMover)
    with pytest.raises(ValueError, match="already registered"):
        D.register_robber_model("plain", object)
    with pytest.raises(ValueError, match="unknown robber model"):
        D.robber_model("nobody-registered-this")


# --- a promise binds: the hosted seat pays what it promised -----------------


class Promises(Gate):
    """Promises every deal and turns down every trade on its merits."""

    def deal_answer(self, game, seat, ask, promised, model):
        return True

    def gains_many(self, view, received, counterparties):
        return [-1.0] * len(received)


def a_promise(kind=D.NO_BLOCK):
    """Seat 1, holding three wood, has promised the mover (seat 0) a wood."""
    game = a_robber_move()
    seat = Seat(observe(game, 1), 1, object(), Promises(), deals=ANSWERS)
    assert seat.deal_answer(0, kind, WOOD)
    return seat


def robber_hex(seat, on_payer):
    state = seat.game._state
    return next(h for h in D.candidate_hexes(state, None)
                if (1 in occupants(state, h)) == on_payer and (on_payer or not occupants(state, h)))


def collect(*, give=ORE):
    received = [0] * 5
    received[WOOD] = 1
    if give is not None:
        received[give] = -1
    return Offer(0, tuple(received))


def test_a_spared_seat_pays_the_card_it_promised_without_pricing_it():
    seat = a_promise()
    seat.play(0, Action(ActionType.MOVE_ROBBER, robber_hex(seat, False), 4))
    assert seat.answer(collect()).kind == RESPONSE_ACCEPT
    assert seat.answer(collect(give=None)).kind == RESPONSE_ACCEPT
    assert seat.answer(Offer(0, (2, 0, 0, 0, -1))).kind == RESPONSE_PASS     # more than the promise
    assert seat.answer(Offer(0, (1, -1, 0, 0, -1))).kind == RESPONSE_PASS    # two cards back
    assert seat.answer(Offer(2, collect().received)).kind == RESPONSE_PASS   # not the mover
    seat.traded(0, 1, collect().received)
    assert seat.answer(collect()).kind == RESPONSE_PASS                      # paid


def test_a_seat_the_robber_hits_anyway_owes_nothing():
    seat = a_promise()
    seat.play(0, Action(ActionType.MOVE_ROBBER, robber_hex(seat, True), 1), stolen=WOOD)
    assert seat.answer(collect()).kind == RESPONSE_PASS


def test_a_no_steal_promise_is_kept_by_robbing_someone_else():
    seat = a_promise(D.NO_STEAL)
    state = seat.game._state
    hex_index, other = next((h, o) for h in D.candidate_hexes(state, None) if 1 in occupants(state, h)
                            for o in occupants(state, h) if o not in (0, 1))
    seat.play(0, Action(ActionType.MOVE_ROBBER, hex_index, other), stolen=UNSEEN)
    assert seat.answer(collect()).kind == RESPONSE_ACCEPT
    other = a_promise(D.NO_STEAL)
    other.play(0, Action(ActionType.MOVE_ROBBER, robber_hex(other, True), 1), stolen=WOOD)
    assert other.answer(collect()).kind == RESPONSE_PASS


def test_a_debt_lapses_with_the_movers_turn():
    seat = a_promise()
    seat.play(0, Action(ActionType.MOVE_ROBBER, robber_hex(seat, False), 4))
    seat.play(0, Action(ActionType.END_TURN))
    seat.game.current_player, seat.game.phase = 0, Phase.MAIN
    assert seat.answer(collect()).kind == RESPONSE_PASS


def test_nothing_is_promised_once_the_robber_has_moved_this_turn():
    seat = a_promise()
    seat.play(0, Action(ActionType.MOVE_ROBBER, robber_hex(seat, False), 4))
    assert not seat.deal_answer(0, D.NO_BLOCK, WOOD)


def test_a_seat_with_no_gate_promises_nothing():
    game = a_robber_move()
    assert not Seat(observe(game, 1), 1, object(), None, deals=ANSWERS).deal_answer(0, D.NO_BLOCK, WOOD)


def test_every_card_the_mover_could_give_back_collects_the_debt():
    """A mover collecting with "the card for any of mine" lets the payer name
    the card: each one-for-one it could name is a payment, so the payer is
    free to take the one it values most."""
    seat = a_promise()
    seat.play(0, Action(ActionType.MOVE_ROBBER, robber_hex(seat, False), 4))
    for back in (BRICK, SHEEP, WHEAT, ORE):
        assert seat.answer(collect(give=back)).kind == RESPONSE_ACCEPT


def test_a_debt_collected_with_an_open_offer_is_paid_by_naming_the_best_card():
    """At a table whose open offers carry "any" cards, "your wood for any of
    mine" is answered with a counter: the wood for the card this seat's gate
    values most."""
    class Prefers(Promises):
        def gains_many(self, view, received, counterparties):
            return [float(r[WHEAT]) for r in received]

    game = a_robber_move()
    seat = Seat(observe(game, 1), 1, object(), Prefers(), deals=ANSWERS)
    assert seat.deal_answer(0, D.NO_BLOCK, WOOD)
    seat.play(0, Action(ActionType.MOVE_ROBBER, robber_hex(seat, False), 4))
    answer = seat.answer(Offer(0, (1, 0, 0, 0, 0), any=-1))
    assert answer.kind == RESPONSE_COUNTER and answer.bundle == (1, 0, 0, -1, 0)
