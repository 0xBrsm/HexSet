# SPDX-License-Identifier: GPL-3.0-only
"""The trade round: the served table's propose-and-respond protocol, over the same primitives the clearing house uses."""

from __future__ import annotations

import random

import pytest

from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.game import Phase, end_turn, start
from hexset.server._webplay import PendingGate
from hexset.trading import (
    RESPONSE_ACCEPT,
    RESPONSE_COUNTER,
    RESPONSE_PASS,
    Offer,
    Response,
    Trade,
    bundle,
    choose_and_execute,
    default_pick,
    default_offer,
    default_respond,
)
from hexset.trading import ENUMERATION_CARDS, TradeParams
from hexset.trading._engine import _candidates as candidates_of
from hexset.trading import trade_round
from helpers import deal, give

WOOD, BRICK, SHEEP, WHEAT, ORE = (int(r) for r in Resource)


def a_game(players: int = 4):
    rng = random.Random(0)
    game = start(random_base_board(rng), players, rng)
    game.phase = Phase.MAIN
    game.current_player = 0
    return game


def stocked(*hands: tuple[int, Resource, int]):
    game = a_game()
    for player, resource, count in hands:
        deal(game, player, resource, count)
    return game


class Gate:
    """A gate returning a caller-supplied gain per candidate, and an estimate of
    the other side's gain only when `estimate_fn` is given -- so a `Gate`
    without one exercises the "own gain stands in" fallback.
    """

    trade_floor = 0.0

    def __init__(self, gain_fn, estimate_fn=None):
        self.gain_fn = gain_fn
        self.gains_calls = 0
        if estimate_fn is not None:
            def estimate_many(view, candidates):
                self.estimate_calls += 1
                return [estimate_fn(c, tuple(b)) for c, b in candidates]

            self.estimate_calls = 0
            self.estimate_many = estimate_many

    def gains_many(self, view, received, counterparties):
        self.gains_calls += 1
        return [self.gain_fn(tuple(r), c) for r, c in zip(received, counterparties)]


def test_default_offer_maximises_own_gain_among_estimate_clearing_candidates():
    """The highest own gain is excluded: its *estimated* counterparty gain never clears."""
    game = stocked(
        (0, Resource.WOOD, 1),
        (1, Resource.ORE, 1),
        (2, Resource.SHEEP, 1),
        (3, Resource.WHEAT, 1),
    )
    candidates = list(candidates_of(game._state, 0, game.locked, ENUMERATION_CARDS))
    assert {them for them, _ in candidates} == {1, 2, 3}

    own_gain = {1: 5.0, 2: 10.0, 3: 3.0}
    estimate = {1: 1.0, 2: -1.0, 3: 1.0}
    gate = Gate(
        lambda received, counterparty: own_gain[counterparty],
        lambda counterparty, received: estimate[counterparty],
    )

    index = default_offer(gate, game.state(0), candidates)

    assert index is not None
    them, _received = candidates[index]
    assert them == 1


def test_default_respond_counters_with_its_best_estimate_clearing_bundle():
    """Seat 1 prices two sheep higher, but only one sheep clears the actor's floor."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    deal(game, 0, Resource.SHEEP, 2)
    offer = Offer(actor=0, received=bundle(wood=-1, ore=1))

    def gain_fn(received, counterparty):
        if received[SHEEP] == 0:
            return -1.0
        return {1: 3.0, 2: 8.0}[received[SHEEP]]

    def estimate_fn(counterparty, received):
        return 1.0 if received[SHEEP] == 1 else -1.0

    gate = Gate(gain_fn, estimate_fn)

    response = default_respond(gate, game.state(1), offer)

    assert response.seat == 1 and response.kind == RESPONSE_COUNTER
    assert response.bundle[ORE] == 1
    assert response.bundle[SHEEP] == -1


def test_default_pick_prices_each_acceptance_by_its_taker():
    """An acceptance is priced by who took the offer: the best taker above
    the actor's floor is picked, and none below it."""
    game = a_game()
    offer = bundle(wood=-1, ore=1)
    responses = [
        Response(1, RESPONSE_ACCEPT, offer),
        Response(2, RESPONSE_ACCEPT, offer),
        Response(3, RESPONSE_PASS, None),
    ]
    gate = Gate(lambda received, counterparty: {1: -3.0, 2: 2.0}.get(counterparty, -9.0))
    assert default_pick(gate, game.state(0), responses) == 1   # seat 2, the one it deals with
    gate = Gate(lambda received, counterparty: {1: -3.0, 2: -1.0}.get(counterparty, -9.0))
    assert default_pick(gate, game.state(0), responses) is None


def test_default_pick_declines_a_counter_that_clears_nothing():
    """A counter is somebody else's proposal, and the actor judges it."""
    game = a_game()
    responses = [Response(1, RESPONSE_COUNTER, bundle(wood=-1, sheep=1))]
    gate = Gate(lambda received, counterparty: -1.0)

    assert default_pick(gate, game.state(0), responses) is None


def test_trade_round_end_to_end_two_bots():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    actor_gate = Gate(lambda r, c: 5.0 if r[ORE] > 0 else -1.0)
    responder_gate = Gate(lambda r, c: 3.0 if r[WOOD] > 0 else -1.0)
    gates = (actor_gate, responder_gate, None, None)

    trades = trade_round(game, gates)

    assert len(trades) == 1
    trade = trades[0]
    assert (trade.a, trade.b, trade.received) == (0, 1, bundle(wood=-1, ore=1))
    assert trade.gain_a == 5.0 and trade.gain_b == 3.0
    assert game._state.hands[0][ORE] == 1
    assert game._state.hands[1][WOOD] == 1
    assert game.trades == [trade]
    assert game.trades_made == 1


def test_an_accept_binds_the_responder_whatever_it_would_price_now():
    """A responder judges an offer when it answers it. Its `respond` is its
    word, so execution does not ask it again: a gate whose answer and whose
    valuation disagree is held to its answer, not rescued from it."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    actor_gate = Gate(lambda r, c: 5.0 if r[ORE] > 0 else -1.0)

    class AnswersYesPricesNo:
        trade_floor = 0.0

        def respond(self, view, offer):
            return Response(1, RESPONSE_ACCEPT, offer.received)

        def gains_many(self, view, received, counterparties):
            return [0.0] * len(received)

    gates = (actor_gate, AnswersYesPricesNo(), None, None)

    trades = trade_round(game, gates)
    assert [(t.a, t.b, t.received) for t in trades] == [(0, 1, bundle(wood=-1, ore=1))]
    # Priced for the record, never asked: the census still reads each side.
    assert (trades[0].gain_a, trades[0].gain_b) == (5.0, 0.0)
    assert game._state.hands[0][ORE] == 1 and game._state.hands[1][WOOD] == 1


def test_an_acceptance_is_priced_by_who_took_the_offer():
    """An offer goes out with some seat in mind. Two seats take it: the actor
    deals with the one it prices above its floor, and a seat it prices below
    -- the same cards, from a seat it would never have proposed them to --
    is declined, alone or not."""
    def table(takers):
        game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1), (2, Resource.ORE, 1))
        # Seat 0 wants ore from seat 1, never from seat 2.
        actor_gate = Gate(lambda r, c: (5.0 if c == 1 else -5.0) if r[ORE] > 0 else -1.0)
        taker = Gate(lambda r, c: 3.0 if r[WOOD] > 0 else -1.0)
        refuser = Gate(lambda r, c: -1.0)
        gates = (actor_gate,) + tuple(taker if s in takers else refuser for s in (1, 2)) + (None,)
        return game, gates

    game, gates = table({1, 2})
    trades = trade_round(game, gates)
    assert [(t.a, t.b) for t in trades] == [(0, 1)]
    game, gates = table({2})
    assert trade_round(game, gates) == []
    assert game._state.hands[2][ORE] == 1 and game._state.hands[0][WOOD] == 1


def test_an_actors_own_give_cap_blocks_a_wider_counter_both_gates_would_clear():
    """Nothing at the table refuses a wide counter -- the actor's own
    `max_give_cards` does, at `pick`."""
    from dataclasses import replace
    from hexset.trading import UNLIMITED, TradeProtocol, install

    game = stocked((0, Resource.WOOD, 3), (1, Resource.ORE, 5))
    actor_gate = Gate(lambda r, c: 5.0 if r[ORE] > 0 else -1.0)
    actor_gate.trade_params = replace(UNLIMITED, max_give_cards=1)
    install(actor_gate, TradeProtocol(actor_gate, actor_gate.trade_params, seed=0))
    over_cap = [0, 0, 0, 0, 0]
    over_cap[WOOD] = -2  # two out, past what the actor says it will give
    over_cap[ORE] = 1

    class Wide:
        trade_floor = 0.0

        def respond(self, view, offer):
            return Response(1, RESPONSE_COUNTER, tuple(over_cap))

        def gains_many(self, view, received, counterparties):
            return [100.0] * len(received)

    gates = (actor_gate, Wide(), None, None)

    assert trade_round(game, gates) == []
    assert game._state.hands[1][ORE] == 5
    assert game._state.hands[0][WOOD] == 3


def test_choose_and_execute_resolves_responses_collected_across_two_calls():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    actor_gate = Gate(lambda r, c: 5.0 if r[ORE] > 0 else -1.0)

    early = [Response(2, RESPONSE_PASS, None)]
    assert choose_and_execute(game, (actor_gate, None, None, None), 0, early) is None
    assert game.trades == []

    responder_gate = Gate(lambda r, c: 3.0 if r[WOOD] > 0 else -1.0)
    gates = (actor_gate, responder_gate, None, None)
    later = early + [Response(1, RESPONSE_ACCEPT, bundle(wood=-1, ore=1))]

    trade = choose_and_execute(game, gates, 0, later)

    assert trade == Trade(0, 1, bundle(wood=-1, ore=1), gain_a=5.0, gain_b=3.0)
    assert game.trades == [trade]
    assert game._state.hands[0][ORE] == 1
    assert game._state.hands[1][WOOD] == 1


def test_trade_round_manual_seats_offer_lands_in_pending_and_stays_open():
    """A second broadcast adds to `game.pending`; only `end_turn` clears it."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    actor_gate = Gate(lambda r, c: 5.0 if r[ORE] > 0 else -1.0)
    pending_gate = PendingGate(game, seat=1)
    gates = (actor_gate, pending_gate, None, None)

    assert trade_round(game, gates) == []
    assert game.pending == [Trade(0, 1, bundle(wood=-1, ore=1))]

    assert trade_round(game, gates) == []
    assert len(game.pending) == 2

    end_turn(game)
    assert game.pending == []


def test_trade_round_manual_actor_never_auto_broadcasts():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    pending_gate = PendingGate(game, seat=0)
    responder_gate = Gate(lambda r, c: 5.0)
    gates = (pending_gate, responder_gate, None, None)

    assert trade_round(game, gates) == []
    assert game.pending == []
    assert game._state.hands[0][WOOD] == 1


@pytest.mark.parametrize("budget,expected_trades", [(1, 0), (None, 1)])
def test_engine_rounds_try_a_distinct_offer_after_refusal(budget, expected_trades):
    from hexset.game import run_trade_event

    game = stocked((1, Resource.ORE, 1), (2, Resource.SHEEP, 1))
    # The actor's wood is held but not certified, so nobody can counter for
    # it: round one is refused outright and round two must try a new offer.
    give(game._state, 0, Resource.WOOD, 1)
    offers = []

    def one_card(r, c):
        # The actor wants one ore, else one sheep, and nothing wider: the
        # menu asks for any cards, so the fixture says which it is after.
        if sum(max(n, 0) for n in r) != 1:
            return -1.0
        return 10.0 * r[ORE] + 5.0 * r[SHEEP]

    class Actor(Gate):
        def offer(self, view, candidates):
            index = default_offer(self, view, candidates)
            if index is not None:
                offers.append(candidates[index][1])
            return index

    actor = Actor(one_card)
    if budget is not None:
        actor.trade_params = TradeParams(max_offers=budget)
    game.gates = (actor, Gate(lambda r, c: -1.0), Gate(lambda r, c: float(r[WOOD])), None)
    run_trade_event(game)
    assert offers[0] == bundle(wood=-1, ore=1)
    assert len(game.trades) == expected_trades
    if expected_trades:
        assert offers[1] == bundle(wood=-1, sheep=1)
        assert game.trades[0].b == 2
    assert len(offers) == len(set(offers))


def test_trade_round_is_its_three_stages_run_at_once():
    """`offer` + `respond` for every seat + `choose_and_execute` is exactly
    `trade_round`: the served table drives those stages around a pause, so
    the two tables play one round."""
    from hexset import trading
    from hexset.game import imagine

    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1), (2, Resource.SHEEP, 1))
    gates = (
        Gate(lambda r, c: 5.0 if r[ORE] > 0 else -1.0),
        Gate(lambda r, c: 3.0 if r[WOOD] > 0 else -1.0),
        Gate(lambda r, c: -1.0),
        None,
    )
    staged = imagine(game, random.Random(0), randomize_deck=False)

    offered = trading.offer(staged, gates)
    assert offered == Offer(0, bundle(wood=-1, ore=1))
    responses = [trading.respond(staged, gates[s], s, offered) for s in (1, 2)]
    trade = choose_and_execute(staged, gates, 0, responses)

    assert trade_round(game, gates) == [trade]
    assert staged._state.hands == game._state.hands
    assert staged.trades == game.trades


def test_the_actor_offers_for_a_card_no_seat_is_certified_to_hold():
    """An offer is broadcast, so the actor asks for the card it needs whether
    or not the record shows anyone holding it -- and a seat that does hold it
    takes it."""
    from hexset import trading
    from hexset.trading import open_candidates

    game = a_game()
    deal(game, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)  # seat 1 holds ore the record cannot see
    view = game.state(0)
    assert all(view.known[s][ORE] == 0 for s in (1, 2, 3))
    gates = (Gate(lambda r, c: 5.0 if r[ORE] > 0 else -1.0), Gate(lambda r, c: 1.0), None, None)

    assert (1, bundle(wood=-1, ore=1)) in open_candidates(view, [1, 2, 3])
    assert trading.offer(game, gates) == Offer(0, bundle(wood=-1, ore=1))
    [trade] = trade_round(game, gates)
    assert (trade.a, trade.b, trade.received) == (0, 1, bundle(wood=-1, ore=1))
    assert game._state.hands[0][ORE] == 1 and game._state.hands[1][WOOD] == 1


def test_the_offer_menu_asks_for_what_the_table_holds_and_no_more():
    """The menu is the actor's own hand against any cards the other seats
    hold between them -- read off the bank, since every card is in the bank or
    a hand -- and nothing about which seat holds them: moving cards between
    the other seats, seen or not, changes nothing in it."""
    from hexset.trading import open_candidates

    game = a_game()
    deal(game, 0, Resource.WOOD, 2)
    others = [1, 2, 3]
    assert open_candidates(game.state(0), others) == []  # the table holds nothing

    deal(game, 1, Resource.ORE, 2)
    give(game._state, 2, Resource.SHEEP, 3)
    game.ledger.gain_unknown(2, 3)
    menu = open_candidates(game.state(0), others)
    wanted = {tuple(b) for _, b in menu}
    assert bundle(wood=-1, ore=1) in wanted and bundle(wood=-2, ore=2) in wanted
    assert bundle(wood=-1, sheep=3) in wanted
    assert all(b[ORE] <= 2 and b[SHEEP] <= 3 and b[WHEAT] <= 0 for b in wanted), \
        "no ask past the table's total"
    assert all(b[WOOD] < 0 for b in wanted), "only wood is the actor's to give"
    assert all(
        sum(max(n, 0) for n in b) <= ENUMERATION_CARDS
        and sum(max(-n, 0) for n in b) <= ENUMERATION_CARDS
        for b in wanted
    )
    assert {them for them, _ in menu} == set(others)

    # Seat 1's ore moves to seat 3, where the record cannot see it.
    game._state.hands[1][ORE] -= 2
    game._state.hands[3][ORE] += 2
    assert open_candidates(game.state(0), others) == menu
    assert open_candidates(game.state(0), []) == []


def test_an_offer_nobody_can_cover_goes_unanswered():
    """Asking is free; covering is not. The seats that want the trade do not
    hold the cards and cannot accept, the one that holds them does not want
    it, and the round moves nothing."""
    from hexset import trading

    game = a_game()
    deal(game, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)  # held where the record cannot see it
    eager = Gate(lambda r, c: 1.0)
    gates = (Gate(lambda r, c: 5.0 if r[ORE] > 0 else -1.0), Gate(lambda r, c: -1.0), eager, eager)
    before = [hand[:] for hand in game._state.hands]

    offered = trading.offer(game, gates)
    assert offered == Offer(0, bundle(wood=-1, ore=1))
    for seat in (2, 3):
        response = default_respond(eager, game.state(seat), offered)
        assert response.kind != RESPONSE_ACCEPT
    assert trade_round(game, gates) == []
    assert game._state.hands == before


def test_the_engine_refuses_an_acceptance_the_responder_cannot_cover():
    """A gate that accepts what it does not hold is stopped at execution: the
    referee's coverage check reads the true hands. That is a rule, not a
    price, and it is the one thing execution still checks."""

    class Blind(Gate):
        def respond(self, view, offer):
            return Response(view.perspective, RESPONSE_ACCEPT, offer.received)

    game = a_game()
    deal(game, 0, Resource.WOOD, 1)
    blind = Blind(lambda r, c: 1.0)
    gates = (Gate(lambda r, c: 5.0 if r[ORE] > 0 else -1.0), blind, None, None)
    before = [hand[:] for hand in game._state.hands]

    assert trade_round(game, gates) == []
    assert game._state.hands == before and game.trades == []


def _untyped_ore_table():
    """Seat 0 holds a certified wood; seat 1 holds one card the record cannot
    type -- and the pool says it can only be ore."""
    game = a_game()
    deal(game, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)
    game.ledger.gain_unknown(1, 1)
    view = game.state(0)
    assert view.known[1][ORE] == 0 and view.unknown[1] == 1 and view.pool[ORE] == 1
    return game, view


def test_a_counter_asks_only_for_what_the_record_certifies_the_actor_holds():
    """A counter goes to the actor alone, so its menu stays the known one:
    seat 1 may counter seat 0 for the certified wood, and seat 0 may not
    counter seat 1 for the ore it cannot see."""
    from hexset.trading import counter_menu, known_candidates

    game, view = _untyped_ore_table()
    eager = Gate(lambda r, c: 1.0)
    assert counter_menu(eager, view, 1) == known_candidates(view, [1]) == []
    seat_one = game.state(1)
    assert counter_menu(eager, seat_one, 0) == known_candidates(seat_one, [0])
    assert (0, bundle(ore=-1, wood=1)) in counter_menu(eager, seat_one, 0)


def test_a_gate_supplies_its_own_menu_for_offers_and_counters():
    """`menu` and `counter_menu` read a gate's `candidates(view,
    counterparties)`; the offer stage and `default_respond` both go through
    it, the counter with `turn=None`."""
    from hexset import trading

    game, _view = _untyped_ore_table()

    class Sampler(Gate):
        def __init__(self, gain_fn, extra):
            super().__init__(gain_fn)
            self.extra = extra
            self.asked = []

        def candidates(self, view, counterparties, *, turn=None, already_offered=()):
            self.asked.append((list(counterparties), turn))
            return [(c, self.extra) for c in counterparties]

    actor = Sampler(lambda r, c: 5.0 if r[ORE] > 0 else -1.0, bundle(wood=-1, ore=1))
    gates = (actor, Gate(lambda r, c: 1.0), None, None)

    assert trading.offer(game, gates) == Offer(0, bundle(wood=-1, ore=1))
    assert actor.asked == [([1, 2, 3], game.turns)]

    responder = Sampler(lambda r, c: 1.0, bundle(ore=-1, wood=1))  # signed towards seat 1
    response = default_respond(responder, game.state(1), Offer(0, bundle(wood=-1, sheep=1)))
    assert responder.asked == [([0], None)]
    assert response.kind == RESPONSE_COUNTER and response.bundle == bundle(ore=1, wood=-1)


class Scripted:
    """A gate that offers one bundle, picks nothing, and answers each offer
    put to it from `answers`, keyed by `(actor, received)` with `received`
    signed towards the actor; an offer it has no answer for is a pass."""

    trade_floor = 0.0

    def __init__(self, offers=None, answers=None):
        self.offers, self.answers, self.asked = offers, answers or {}, []

    def gains_many(self, view, received, counterparties):
        return [1.0] * len(received)

    def candidates(self, view, counterparties, *, turn=None, already_offered=()):
        return [] if turn is None or self.offers is None else [self.offers]

    def offer(self, view, candidates):
        return 0 if candidates else None

    def respond(self, view, offer):
        self.asked.append((offer.actor, tuple(offer.received)))
        kind, answer = self.answers.get((offer.actor, tuple(offer.received)), (RESPONSE_PASS, None))
        return Response(view.perspective, kind, answer)

    def pick(self, view, responses):
        return None


def _chain_table(mover_answers, seat_answers, other=None):
    """Seat 0 offers a wood for seat 1's ore; seat 1 counters with two wood
    for it; what follows is the two scripts'."""
    game = stocked((0, Resource.WOOD, 2), (0, Resource.BRICK, 1), (1, Resource.ORE, 2), (2, Resource.ORE, 1))
    seat1 = {(0, bundle(wood=-1, ore=1)): (RESPONSE_COUNTER, bundle(wood=-2, ore=1)), **seat_answers}
    gates = (Scripted((1, bundle(wood=-1, ore=1)), mover_answers), Scripted(answers=seat1), other, None)
    return game, gates


MOVER_COUNTERS = {(1, bundle(wood=2, ore=-1)): (RESPONSE_COUNTER, bundle(wood=1, brick=1, ore=-1))}


def test_one_counter_by_default_the_actor_takes_or_leaves():
    game, gates = _chain_table(MOVER_COUNTERS, {(0, bundle(wood=-1, brick=-1, ore=1)): (RESPONSE_ACCEPT, bundle(wood=-1, brick=-1, ore=1))})
    assert trade_round(game, gates) == []
    assert gates[0].asked == []                       # never asked to answer the counter


def test_the_mover_counters_a_counter_and_the_seat_takes_it():
    """mover -> seat -> mover, then the seat accepts: a wood and a brick for its ore."""
    game, gates = _chain_table(MOVER_COUNTERS, {(0, bundle(wood=-1, brick=-1, ore=1)): (RESPONSE_ACCEPT, bundle(wood=-1, brick=-1, ore=1))})
    [trade] = trade_round(game, gates, counter_steps=2)
    assert (trade.a, trade.b, trade.received) == (0, 1, bundle(wood=-1, brick=-1, ore=1))
    assert gates[0].asked == [(1, bundle(wood=2, ore=-1))]
    assert game._state.hands[0][ORE] == 1 and game._state.hands[0][WOOD] == 1 and game._state.hands[0][BRICK] == 0
    assert (1, 1, bundle(wood=1, brick=1, ore=-1)) not in game.shown   # one exchange later, by seat 0
    assert game.shown[-2:] == [(0, 0, bundle(wood=-1, brick=-1, ore=1)), (0, 1, bundle(wood=1, brick=1, ore=-1))]


@pytest.mark.parametrize("steps,trades", [(2, 0), (3, 1)])
def test_a_thread_holds_at_most_counter_steps_counters(steps, trades):
    """mover -> seat -> mover -> seat: the seat's second counter is the thread's
    third, a pass at two steps and taken by the mover at three."""
    mover = {**MOVER_COUNTERS, (1, bundle(brick=1, ore=-1)): (RESPONSE_ACCEPT, bundle(brick=1, ore=-1))}
    seat = {(0, bundle(wood=-1, brick=-1, ore=1)): (RESPONSE_COUNTER, bundle(brick=-1, ore=1))}
    game, gates = _chain_table(mover, seat)
    assert len(trade_round(game, gates, counter_steps=steps)) == trades
    assert len(gates[0].asked) == (2 if steps == 3 else 1)


def test_a_counter_repeating_a_bundle_already_put_ends_the_thread():
    """The seat answers the mover's counter with its own first counter again."""
    game, gates = _chain_table(MOVER_COUNTERS, {(0, bundle(wood=-1, brick=-1, ore=1)): (RESPONSE_COUNTER, bundle(wood=-2, ore=1))})
    assert trade_round(game, gates, counter_steps=5) == []
    assert len(gates[0].asked) == 1


def test_each_counter_is_its_own_thread_between_the_mover_and_that_seat():
    """Seat 1's thread ends in a pass; seat 2's counter, taken back and forth
    with the mover, is the one that executes."""
    seat2 = Scripted(answers={
        (0, bundle(wood=-1, ore=1)): (RESPONSE_COUNTER, bundle(wood=-2, ore=1)),
        (0, bundle(wood=-1, brick=-1, ore=1)): (RESPONSE_ACCEPT, bundle(wood=-1, brick=-1, ore=1)),
    })
    mover = {(1, bundle(wood=2, ore=-1)): (RESPONSE_PASS, None), (2, bundle(wood=2, ore=-1)): MOVER_COUNTERS[(1, bundle(wood=2, ore=-1))]}
    game, gates = _chain_table(mover, {}, other=seat2)
    [trade] = trade_round(game, gates, counter_steps=2)
    assert (trade.a, trade.b) == (0, 2)
    assert gates[0].asked == [(1, bundle(wood=2, ore=-1)), (2, bundle(wood=2, ore=-1))]


def test_a_games_counter_steps_carries_into_its_trade_rounds():
    from hexset.game import _run_trade_rounds, imagine

    game, gates = _chain_table(MOVER_COUNTERS, {(0, bundle(wood=-1, brick=-1, ore=1)): (RESPONSE_ACCEPT, bundle(wood=-1, brick=-1, ore=1))})
    game.counter_steps = 2
    assert imagine(game, random.Random(0), randomize_deck=False).counter_steps == 2
    assert len(_run_trade_rounds(game, gates)) == 1




# -- the referee admits a gate's answer only as the rules allow --------------


class Answers(Gate):
    """A gate answering every offer with `answer(view, offer)` as it stands."""

    def __init__(self, answer):
        super().__init__(lambda r, c: 1.0)
        self.answer = answer

    def respond(self, view, offer):
        return self.answer(view, offer)


def _one_ore_table():
    """Seat 0 offers a wood for an ore; seat 1 holds the ore."""
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1))
    return game, Offer(0, bundle(wood=-1, ore=1))


def test_an_answer_is_always_the_asked_seats():
    from hexset import trading

    game, offer = _one_ore_table()
    spoof = Answers(lambda view, offer: Response(3, RESPONSE_ACCEPT, offer.received))
    assert trading.respond(game, spoof, 1, offer) == Response(1, RESPONSE_ACCEPT, offer.received)


def test_an_acceptance_is_of_the_offer_as_it_was_made():
    from hexset import trading

    game, offer = _one_ore_table()
    greedy = Answers(lambda view, offer: Response(1, RESPONSE_ACCEPT, bundle(wood=-2, ore=1)))
    assert trading.respond(game, greedy, 1, offer) == Response(1, RESPONSE_ACCEPT, offer.received)
    [trade] = trading.resolve_offer(game, (Gate(lambda r, c: 5.0), greedy, None, None), offer)
    assert trade.received == offer.received and game._state.hands[0][WOOD] == 1


def test_an_acceptance_the_seats_own_hand_cannot_cover_is_a_pass_and_shows_nothing():
    from hexset import trading

    game = stocked((0, Resource.WOOD, 1))
    offer = Offer(0, bundle(wood=-1, ore=1))
    blind = Answers(lambda view, offer: Response(1, RESPONSE_ACCEPT, offer.received))
    assert trading.respond(game, blind, 1, offer) == Response(1, RESPONSE_PASS)
    trading.resolve_offer(game, (Gate(lambda r, c: 5.0), blind, None, None), offer)
    assert [seat for _, seat, _ in game.shown] == [0], "only the offer went to the table"


@pytest.mark.parametrize("counter", [
    bundle(wood=-1),                 # one-sided
    (0, 0, 0, 0, 0),                 # empty
    (1, 0, 0, -1),                   # the wrong width
    (True, 0, 0, 0, -1),             # not counts
    None,
])
def test_a_counter_that_is_not_an_exchange_is_a_pass(counter):
    from hexset import trading

    game, offer = _one_ore_table()
    odd = Answers(lambda view, offer: Response(1, RESPONSE_COUNTER, counter))
    assert trading.respond(game, odd, 1, offer) == Response(1, RESPONSE_PASS)


@pytest.mark.parametrize("a,b,received,match", [
    (0, 4, bundle(wood=-1, ore=1), "not a seat"),
    (0, 1, bundle(wood=-1), "not an exchange"),
    (0, 1, (0, 0, 0, 0, 0), "not an exchange"),
    (0, 2, bundle(wood=-1, ore=1), "locked"),
])
def test_the_referee_refuses_an_exchange_outside_the_table(a, b, received, match):
    from hexset.game import lock_seat
    from hexset.trading import execute_agreed

    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1), (2, Resource.ORE, 1))
    lock_seat(game, 2)
    with pytest.raises(ValueError, match=match):
        execute_agreed(game, a, b, received, ask_actor=False, ask_counterparty=False)
    assert game.trades == []


# -- a gate's own limits bind what it offers and what it signs ---------------


def test_a_caps_only_gate_never_offers_more_than_it_will_give():
    """Its menu is the engine's, enumerated at three cards a side; what it
    would refuse to sign never reaches the table."""
    from dataclasses import replace
    from hexset import trading
    from hexset.trading import UNLIMITED, TradeProtocol, install

    game = stocked((0, Resource.WOOD, 3), (1, Resource.ORE, 3))
    dumping = Gate(lambda r, c: r[ORE] - r[WOOD])      # rid of wood, after ore
    gates = (dumping, Gate(lambda r, c: 1.0), None, None)
    assert trading.offer(game, gates).received[WOOD] == -3
    capped = Gate(lambda r, c: r[ORE] - r[WOOD])
    capped.trade_params = replace(UNLIMITED, max_give_cards=2)
    install(capped, TradeProtocol(capped, capped.trade_params, seed=0))
    offered = trading.offer(game, (capped,) + gates[1:])
    assert offered is not None and offered.received[WOOD] == -2


def test_a_gate_that_does_not_trade_signs_nothing():
    """`max_offers=0` opens nothing and answers every offer with a pass,
    however much it would gain."""
    from hexset import trading

    game, offer = _one_ore_table()
    keen = Gate(lambda r, c: 5.0)
    assert trading.respond(game, keen, 1, offer).kind == RESPONSE_ACCEPT
    keen.trade_params = TradeParams(max_offers=0)
    assert trading.respond(game, keen, 1, offer) == Response(1, RESPONSE_PASS)
    assert trading.resolve_offer(game, (Gate(lambda r, c: 5.0), keen, None, None), offer) == []


def test_retuning_a_plain_gates_offers_reaches_the_engine():
    """A gate with no `TradeParams` is retuned onto its loose budget, which
    `params_of` reads, so the driver stops where the run asked."""
    from hexset.game import run_trade_event
    from hexset.trading import params_of, retune

    game = stocked((1, Resource.ORE, 1), (2, Resource.SHEEP, 1))
    give(game._state, 0, Resource.WOOD, 1)  # uncertified: nobody can counter for it
    actor = Gate(lambda r, c: -1.0 if sum(max(n, 0) for n in r) != 1 else 10.0 * r[ORE] + 5.0 * r[SHEEP])
    retune(actor, max_offers=1)
    assert params_of(actor).max_offers == 1
    game.gates = (actor, Gate(lambda r, c: -1.0), Gate(lambda r, c: float(r[WOOD])), None)
    run_trade_event(game)
    assert game.trades == [], "one offer, refused, and no second"
    retune(actor, max_offers=None)
    assert params_of(actor).max_offers is None
