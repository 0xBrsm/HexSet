# SPDX-License-Identifier: GPL-3.0-only
"""The trade round (`hexset.trading.trade_round`): the served table's own
protocol, propose-and-respond over the same primitives the clearing house
(`trade_event`, wholly unaffected by anything here) uses -- see
`hexset.trading`'s module docstring, "The trade round".
"""

from __future__ import annotations

import random

import pytest

from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.game import Phase, end_turn, start
from hexset.server.webplay import PendingGate
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
from hexset.trading import _candidates as candidates_of
from hexset.trading import trade_round
from helpers import give

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
        give(game._state, player, resource, count)
    return game


class Gate:
    """A seat whose gate returns a caller-supplied gain per candidate
    (`gain_fn(received, counterparty) -> float`) and, only when `estimate_fn`
    is given, a caller-supplied estimate of the *other* side's gain
    (`estimate_fn(counterparty, received) -> float`) -- `estimate_many` is
    an instance attribute set only in that case, so a `Gate` built without
    one exercises `hexset.trading`'s "own gain stands in for the estimate"
    fallback (`_estimate_many`) exactly as a plain `gains_many`-only bot
    would.
    """

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


# --- default_offer --------------------------------------------------------


def test_default_offer_maximises_own_gain_among_estimate_clearing_candidates():
    """Three counterparties offer the same shaped 1-for-1 bundle; the
    highest own gain (seat 2, sheep) is excluded because its *estimated*
    counterparty gain never clears the floor, so the winner is the highest
    own gain among the two candidates whose estimate does."""
    game = stocked(
        (0, Resource.WOOD, 1),
        (1, Resource.ORE, 1),
        (2, Resource.SHEEP, 1),
        (3, Resource.WHEAT, 1),
    )
    candidates = list(candidates_of(game._state, 0, game.locked))
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
    assert them == 1  # 5.0 beats 3.0 among {1, 3}; 2's 10.0 never qualifies


def test_default_offer_falls_back_to_its_own_gain_as_the_estimate():
    """A gate with no `estimate_many` at all offers "what is best for
    itself" -- its own gain stands in for the estimate too."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    candidates = list(candidates_of(game._state, 0, game.locked))
    gate = Gate(lambda received, counterparty: 5.0)

    assert default_offer(gate, game.state(0), candidates) == 0


def test_default_offer_passes_when_no_estimate_clears():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    candidates = list(candidates_of(game._state, 0, game.locked))
    gate = Gate(lambda received, counterparty: 5.0, lambda counterparty, received: -1.0)

    assert default_offer(gate, game.state(0), candidates) is None


# --- default_respond --------------------------------------------------------


def test_default_respond_accepts_when_its_own_gain_clears_the_floor():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    offer = Offer(actor=0, received=bundle(wood=-1, ore=1))
    gate = Gate(lambda received, counterparty: 1.0)

    response = default_respond(gate, game.state(1), offer)

    assert response == Response(1, RESPONSE_ACCEPT, offer.received)


def test_default_respond_never_accepts_what_it_cannot_cover():
    """Seat 1 holds sheep, not the ore the offer asks of it. However much
    its gate likes the exchange, an accept it cannot cover would only fail
    at the actor's `execute_round_choice` -- so it must not accept."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.SHEEP, 1))
    offer = Offer(actor=0, received=bundle(wood=-1, ore=1))
    gate = Gate(lambda received, counterparty: 1.0)

    response = default_respond(gate, game.state(1), offer)

    assert response.kind != RESPONSE_ACCEPT
    assert response == Response(1, RESPONSE_PASS, None)


def test_default_respond_counters_with_its_best_estimate_clearing_bundle():
    """Seat 1 refuses the offer outright, but can infer (from the ledger's
    certified lower bound on seat 0's hand) that the actor holds sheep --
    two counter-candidates arise (one sheep, two sheep), the two-sheep one
    priced higher by seat 1's own gate, but only the one-sheep counter's
    *estimated* actor gain clears the floor."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    give(game._state, 0, Resource.SHEEP, 2)
    game.ledger.receive(0, Resource.SHEEP, 2)  # certified to seat 1's belief
    offer = Offer(actor=0, received=bundle(wood=-1, ore=1))

    def gain_fn(received, counterparty):
        if received[SHEEP] == 0:
            return -1.0  # the original offer: not worth it
        return {1: 3.0, 2: 8.0}[received[SHEEP]]

    def estimate_fn(counterparty, received):
        return 1.0 if received[SHEEP] == 1 else -1.0

    gate = Gate(gain_fn, estimate_fn)

    response = default_respond(gate, game.state(1), offer)

    assert response.seat == 1 and response.kind == RESPONSE_COUNTER
    # Signed towards the actor: seat 1 still gives its one ore, and takes
    # exactly one sheep (never two, despite pricing it higher itself).
    assert response.bundle[ORE] == 1
    assert response.bundle[SHEEP] == -1


def test_default_respond_never_counters_with_a_deal_it_would_refuse():
    """The actor's estimated gain clears on both counter-candidates, but
    seat 1's own gain clears on neither -- so it passes rather than
    countering. A counter it would not itself honour is a deal that fails
    at `execute_agreed`, which asks the responder's gate again and applies
    the same floor, in an error the actor could do nothing about."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    give(game._state, 0, Resource.SHEEP, 2)
    game.ledger.receive(0, Resource.SHEEP, 2)
    offer = Offer(actor=0, received=bundle(wood=-1, ore=1))

    gate = Gate(
        lambda received, counterparty: -1.0,  # wants nothing on offer
        lambda counterparty, received: 1.0,  # but thinks the actor wants it all
    )

    response = default_respond(gate, game.state(1), offer)

    assert response == Response(1, RESPONSE_PASS, None)


def test_default_respond_passes_when_nothing_clears():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    offer = Offer(actor=0, received=bundle(wood=-1, ore=1))
    gate = Gate(lambda received, counterparty: -1.0)

    response = default_respond(gate, game.state(1), offer)

    assert response == Response(1, RESPONSE_PASS, None)


# --- default_pick ----------------------------------------------------------


def test_default_pick_picks_the_response_with_the_highest_clearing_gain():
    game = a_game()
    responses = [
        Response(1, RESPONSE_ACCEPT, bundle(wood=-1, ore=1)),
        Response(2, RESPONSE_COUNTER, bundle(wood=-1, sheep=2)),
        Response(3, RESPONSE_PASS, None),
    ]
    gate = Gate(lambda received, counterparty: {1: 2.0, 2: 9.0}.get(counterparty, -1.0))

    chosen = default_pick(gate, game.state(0), responses)

    assert chosen == 1  # seat 2's counter (9.0) beats seat 1's accept (2.0)


def test_default_pick_returns_none_when_nothing_clears():
    game = a_game()
    responses = [Response(1, RESPONSE_ACCEPT, bundle(wood=-1, ore=1))]
    gate = Gate(lambda received, counterparty: -1.0)

    assert default_pick(gate, game.state(0), responses) is None


# --- trade_round: end to end ---------------------------------------------------


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


def test_trade_round_pass_all_executes_nothing():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    actor_gate = Gate(lambda r, c: 5.0 if r[ORE] > 0 else -1.0)
    responder_gate = Gate(lambda r, c: -1.0)  # refuses everything, no belief to counter with
    gates = (actor_gate, responder_gate, None, None)

    assert trade_round(game, gates) == []
    assert game.trades == []
    assert game._state.hands[0][WOOD] == 1
    assert game._state.hands[1][ORE] == 1


def test_trade_round_is_a_noop_outside_main():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.phase = Phase.ROLL
    gates = (Gate(lambda r, c: 5.0), Gate(lambda r, c: 5.0), None, None)

    assert trade_round(game, gates) == []


def test_trade_round_executes_even_when_max_trades_is_zero():
    """`max_trades = 0` is the existing off switch for the automatic
    clearing house (`run_trade_event`'s own guard) -- a served game sets it
    for exactly that reason, and `trade_round` is unaffected by it: it is a
    separate call the session makes itself, not something the switch was
    ever meant to gate."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.max_trades = 0
    actor_gate = Gate(lambda r, c: 5.0 if r[ORE] > 0 else -1.0)
    responder_gate = Gate(lambda r, c: 3.0 if r[WOOD] > 0 else -1.0)
    gates = (actor_gate, responder_gate, None, None)

    trades = trade_round(game, gates)

    assert len(trades) == 1
    assert game.trades == trades
    assert game.trades_made == 1
    assert game._state.hands[0][ORE] == 1


# --- the engine re-checks at execution, trusting neither side's report -------


def test_trade_round_floor_blocks_execution_even_if_a_gate_misbehaves():
    """A gate whose custom `respond` claims acceptance regardless, but whose
    `gains_many` -- what `_execute_round` re-asks fresh, at the moment cards
    would actually move -- never clears the floor, must not let the trade
    through."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    actor_gate = Gate(lambda r, c: 5.0 if r[ORE] > 0 else -1.0)

    class Misbehaving:
        def respond(self, view, offer):
            return Response(1, RESPONSE_ACCEPT, offer.received)

        def gains_many(self, view, received, counterparties):
            return [0.0] * len(received)  # at the floor, never above it

    gates = (actor_gate, Misbehaving(), None, None)

    assert trade_round(game, gates) == []
    assert game.trades == []
    assert game._state.hands[0][WOOD] == 1
    assert game._state.hands[1][ORE] == 1


def test_trade_round_cap_blocks_an_over_cap_counter_even_if_both_gates_would_clear():
    """A misbehaving `respond` can hand back a `Response` with any bundle at
    all -- unlike an `offer`'s index, which is always one of the engine's
    own capped `_candidates` -- so the cap has to be enforced again at
    execution, the same as it is for `execute_trade`."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 5))
    actor_gate = Gate(lambda r, c: 5.0 if r[ORE] > 0 else -1.0)
    over_cap = [0, 0, 0, 0, 0]
    over_cap[WOOD] = -1
    over_cap[ORE] = 4  # exceeds MAX_TRADE_CARDS on the actor's take side

    class OverCap:
        def respond(self, view, offer):
            return Response(1, RESPONSE_COUNTER, tuple(over_cap))

        def gains_many(self, view, received, counterparties):
            return [100.0] * len(received)  # would clear happily, if it were even asked

    gates = (actor_gate, OverCap(), None, None)

    assert trade_round(game, gates) == []
    assert game._state.hands[1][ORE] == 5
    assert game._state.hands[0][WOOD] == 1


# --- choose_and_execute: resolving a round assembled across more than one call ---


def test_choose_and_execute_resolves_responses_collected_across_two_calls():
    """A served table keeps a round open across a manual seat's late answer
    (`hexset.server.webplay.GameSession`) rather than calling `trade_round`
    itself, so it builds its own `responses` list one seat at a time --
    exactly what this test does by hand -- and calls `choose_and_execute`
    once enough of them are in, the same way `trade_round`'s own tail
    would from one synchronous batch."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    actor_gate = Gate(lambda r, c: 5.0 if r[ORE] > 0 else -1.0)

    # First call: only seat 2's (indifferent) answer is in hand yet -- no
    # seat 1 response at all, so nothing can clear.
    early = [Response(2, RESPONSE_PASS, None)]
    assert choose_and_execute(game, (actor_gate, None, None, None), 0, early) is None
    assert game.trades == []

    # Second call: seat 1's real answer has since arrived and is appended to
    # the same list -- this is the "late manual answer" shape.
    responder_gate = Gate(lambda r, c: 3.0 if r[WOOD] > 0 else -1.0)
    gates = (actor_gate, responder_gate, None, None)
    later = early + [Response(1, RESPONSE_ACCEPT, bundle(wood=-1, ore=1))]

    trade = choose_and_execute(game, gates, 0, later)

    assert trade == Trade(0, 1, bundle(wood=-1, ore=1), gain_a=5.0, gain_b=3.0)
    assert game.trades == [trade]
    assert game._state.hands[0][ORE] == 1
    assert game._state.hands[1][WOOD] == 1


def test_choose_and_execute_returns_none_with_no_responses():
    game = a_game()
    assert choose_and_execute(game, (Gate(lambda r, c: 5.0), None, None, None), 0, []) is None


# --- manual seats (`PendingGate`) --------------------------------------------


def test_trade_round_manual_seats_offer_lands_in_pending_and_stays_open():
    """A `PendingGate` responder never clears a round on its own -- it
    records the broadcast offer to `game.pending` and answers pass for the
    round's own purposes. The round stays open for the rest of the seat's
    turn: a second broadcast the same turn adds to `game.pending` rather
    than replacing it, and only `end_turn` (unchanged) clears it."""
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
    """A `PendingGate` actor's `offer` always passes -- a human or LLM
    composes an offer through a separate, explicit server call, not
    automatically at every `trade_round`."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    pending_gate = PendingGate(game, seat=0)
    responder_gate = Gate(lambda r, c: 5.0)
    gates = (pending_gate, responder_gate, None, None)

    assert trade_round(game, gates) == []
    assert game.pending == []
    assert game._state.hands[0][WOOD] == 1


# --- heximax/search2: `estimate_many` is wired to the counterparty's row -----


def test_heximax_estimate_many_reads_the_counterpartys_row_through_belief():
    from hexset.bots.heximax import Heximax, HonestEvaluator

    game = a_game()
    give(game._state, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)
    game.ledger.receive(0, Resource.WOOD, 1)
    game.ledger.receive(1, Resource.ORE, 1)
    bot = Heximax(HonestEvaluator(game._state.board), rng=random.Random(0))
    view = game.state(0)
    candidates = [(1, bundle(wood=-1, ore=1))]

    estimates = bot.estimate_many(view, candidates)

    assert len(estimates) == 1
    expected = bot._delta(view, 0, 1, bundle(ore=-1, wood=1), 0, bot._rank)
    assert estimates[0] == pytest.approx(expected)


def test_heximax_estimate_many_off_switch():
    from hexset.bots.heximax import Heximax, HonestEvaluator

    game = a_game()
    bot = Heximax(HonestEvaluator(game._state.board), rng=random.Random(0), max_trades=0)
    view = game.state(0)

    assert bot.estimate_many(view, [(1, bundle(wood=-1, ore=1))]) == [-1.0]


def test_search_bot_estimate_many_reads_the_counterpartys_row():
    from hexset.bots import SearchBot
    from hexset.bots.evaluate import Evaluator

    game = a_game()
    give(game._state, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)
    bot = SearchBot(Evaluator(game._state.board), rng=random.Random(0))
    view = game.state(0)
    candidates = [(1, bundle(wood=-1, ore=1))]

    estimates = bot.estimate_many(view, candidates)

    expected = bot._delta_for(view, 0, 1, bundle(wood=-1, ore=1), 1)
    assert estimates == [pytest.approx(expected)]


def test_search_bot_estimate_many_off_switch():
    from hexset.bots import SearchBot
    from hexset.bots.evaluate import Evaluator

    game = a_game()
    bot = SearchBot(Evaluator(game._state.board), rng=random.Random(0), max_trades=0)
    view = game.state(0)

    assert bot.estimate_many(view, [(1, bundle(wood=-1, ore=1))]) == [-1.0]
