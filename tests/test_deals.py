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
    model = D.PlainMover()
    target = next(s for s in (2, 3) if D.targeted(game, s, ASK, (1,), model))
    other = 5 - target
    assert D.answer(game, target, gate, ANSWERS, ASK, (1,))
    assert asked == [(target, (1,), D.PlainMover)]
    if not D.targeted(game, other, ASK, (1,), model):
        assert not D.answer(game, other, gate, ANSWERS, ASK, (1,))
        assert len(asked) == 1                 # a seat the mover is not coming for is not asked


def test_the_target_is_the_seat_likeliest_to_be_hit():
    game = a_robber_move()
    model = D.PlainMover()
    chance = {s: block_chance(model, game, s) for s in (1, 2, 3)}
    top = max(chance.values())
    for s in (1, 2, 3):
        assert D.targeted(game, s, ASK, (), model) == (chance[s] > 0 and chance[s] >= top - 1e-12)
    assert not D.targeted(game, 0, ASK, (), model)          # never the mover itself
    ns = D.Ask(0, D.NO_STEAL, WOOD)
    stolen = {}
    for s in (1, 2, 3):
        hexes, robbed = D.hit_chances(model, game._state, 0, None, {}, s)
        stolen[s] = sum(p * robbed[h] for h, p in hexes.items())
    for s in (1, 2, 3):
        assert D.targeted(game, s, ns, (), model) == (stolen[s] > 0 and stolen[s] >= max(stolen.values()) - 1e-12)


def test_a_promise_already_made_moves_the_target():
    game = a_robber_move()
    model = D.PlainMover()
    target = next(s for s in (1, 2, 3) if D.targeted(game, s, ASK, (), model))
    # the mover reads the target's promise as kept, so its hexes are all but ruled out
    assert not D.targeted(game, target, ASK, (target,), model)


def test_a_hosted_seat_reads_whether_it_is_targeted_holding_the_card_or_not():
    game = a_robber_move()
    model = D.PlainMover()
    target = next(s for s in (1, 2, 3) if D.targeted(game, s, ASK, (), model))
    clear_hand(game._state, target)
    seat = Seat(observe(game, target), target, object(), Gate(), deals=ANSWERS)
    assert seat.deal_targeted(0, D.NO_BLOCK, WOOD)
    assert not seat.deal_answer(0, D.NO_BLOCK, WOOD)        # nothing to pay with
    assert not Seat(observe(game, target), target, object(), Gate(),
                    deals=D.DealParams()).deal_targeted(0, D.NO_BLOCK, WOOD)


def test_a_hosted_seat_answers_from_its_own_view():
    game = a_robber_move()
    model = D.PlainMover()
    chance = {s: block_chance(model, game, s) for s in (1, 2, 3)}
    target = max(chance, key=chance.get)
    seat = Seat(observe(game, target), target, object(), Gate(), deals=ANSWERS)
    assert seat.deal_answer(0, D.NO_BLOCK, WOOD)
    assert not Seat(observe(game, target), target, object(), Gate(),
                    deals=D.DealParams()).deal_answer(0, D.NO_BLOCK, WOOD)
    # Answering is a hosted seat's default.
    assert Seat(observe(game, target), target, object(), Gate()).deal_answer(0, D.NO_BLOCK, WOOD)


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


# --- the hosted seat as the robber's mover -------------------------------------


class Asker(Gate):
    """A gate that deals as the mover: asks for wood, spares the first promise,
    offers ore back, takes any exchange and proposes nothing of its own."""

    deal_params = D.DealParams(ask=True)

    def deal_ask(self, game, seat):
        return D.Ask(seat, D.NO_BLOCK, WOOD)

    def deal_accept(self, game, seat, ask, promised):
        return promised[:1]

    def deal_return(self, game, seat, payers, resource):
        return ORE

    def gains_many(self, view, received, counterparties):
        return [1.0] * len(received)

    def candidates(self, view, counterparties, **kw):
        return []


def a_mover(game=None):
    game = game or a_robber_move()
    return Seat(observe(game, 0), 0, object(), Asker())


def test_a_hosted_mover_asks_once_a_robber_move():
    seat = a_mover()
    ask = seat.deal_ask()
    assert ask == D.Ask(0, D.NO_BLOCK, WOOD) and seat.deal_ask() is ask
    assert len(seat.deal_log) == 1 and seat.deal_log[0].resource == WOOD
    assert Seat(observe(a_robber_move(), 0), 0, object(), Gate()).deal_ask() is None   # a gate that does not ask


def test_a_hosted_mover_counts_promises_its_view_allows_and_spares_its_pick():
    game = a_robber_move()
    clear_hand(game._state, 3)
    game.ledger.seats[3].known = [0] * 5          # the record: seat 3 holds nothing
    game.ledger.seats[3].unknown = 0
    seat = a_mover(game)
    seat.deal_ask()
    assert seat.deal_promised((3, 2, 1)) == (2,)
    (deal,) = seat.deal_log
    assert deal.promised == (3, 2, 1) and deal.void == (3,) and deal.counted == (2, 1)
    assert seat.game.robber_promised == ((2, D.NO_BLOCK),)


def test_a_hosted_mover_ignores_a_blacklisted_seat():
    seat = a_mover()
    seat.deal_ask()
    seat._deal_log.insert(0, D.Deal(0, 0, D.NO_BLOCK, WOOD, promised=(1,), kept=(1,), rejected=(1,)))
    assert seat.deal_promised((1, 2)) == (2,)
    assert seat.deal_log[-1].ignored == (1,)


@pytest.mark.parametrize("withdrawn", [(), (3,)])
def test_withdrawn_or_void_promises_clear_the_previous_acceptance(withdrawn):
    game = a_robber_move()
    clear_hand(game._state, 3)
    game.ledger.seats[3].known = [0] * 5
    game.ledger.seats[3].unknown = 0
    seat = a_mover(game)
    seat.deal_ask()
    assert seat.deal_promised((1, 2)) == (1,)
    assert seat.deal_promised(withdrawn) == ()
    assert seat.deal_log[-1].accepted == ()
    assert seat.game.robber_promised == ()


def a_spared_deal():
    """Seat 0 asked for wood; seats 1 and 2 promised; it spared seat 1 and robbed seat 2."""
    seat = a_mover()
    seat.deal_ask()
    assert seat.deal_promised((1, 2)) == (1,)
    state = seat.game._state
    hex_ = next(h for h in range(state.board.num_hexes)
                if 2 in occupants(state, h) and 1 not in occupants(state, h) and h != state.robber)
    seat.play(0, Action(ActionType.MOVE_ROBBER, hex_, 2), stolen=WOOD)
    return seat


def test_a_hosted_mover_s_robber_move_settles_whom_it_spared():
    seat = a_spared_deal()
    (deal,) = seat.deal_log
    assert deal.kept == (1,) and deal.broken == (2,) and deal.victim == 2
    assert seat.game.phase is Phase.MAIN and seat.game.robber_promised == ()


def test_the_deal_s_exchange_is_the_mover_s_first_offer_and_any_seat_may_take_it():
    seat = a_spared_deal()
    offer = seat.offer()
    assert offer == Offer(0, (1, 0, 0, 0, -1))           # wood for ore
    taken = seat.pick([offer_answer(1, RESPONSE_PASS), offer_answer(3, RESPONSE_ACCEPT, offer.received)])
    assert taken.seat == 3
    trade = seat.traded(0, 3, offer.received)
    seat.close_round(trade)
    (deal,) = seat.deal_log
    assert deal.offered == ORE and deal.taker == 3 and deal.paid == () and deal.rejected == (1,)
    assert D.blacklist(seat.game, 0) == {1}
    assert seat.offer() is None                           # the deal's offer goes once


def test_a_spared_promiser_that_takes_it_is_paid_and_not_blacklisted():
    seat = a_spared_deal()
    offer = seat.offer()
    seat.pick([offer_answer(1, RESPONSE_ACCEPT, offer.received)])
    seat.close_round(seat.traded(0, 1, offer.received))
    (deal,) = seat.deal_log
    assert deal.paid == ((1, ORE),) and deal.rejected == () and D.blacklist(seat.game, 0) == set()


def test_the_deal_log_survives_a_rebuilt_game():
    seat = a_spared_deal()
    offer = seat.offer()
    seat.pick([])
    seat.close_round(None)
    rebuilt = observe(seat.game, 0)
    seat.resync(rebuilt)
    assert D.blacklist(seat.game, 0) == {1}


def offer_answer(seat, kind, bundle=None):
    from hexset.trading import Response
    return Response(seat, kind, bundle)
