# SPDX-License-Identifier: GPL-3.0-only
"""Open offers: an offer with "any" cards on one side, an invitation to
counter. It is never accepted as it stands, only countered, and what
executes is always concrete."""

from __future__ import annotations

import random

import pytest

from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.game import Phase, start
from hexset.server._webplay import PendingGate
from hexset.trading import (
    RESPONSE_ACCEPT,
    RESPONSE_COUNTER,
    RESPONSE_PASS,
    Offer,
    Response,
    default_respond,
    is_open,
    offer,
    resolve_offer,
    respond,
)
from helpers import deal

WOOD, BRICK, SHEEP, WHEAT, ORE = (int(r) for r in Resource)


def _game(*hands):
    rng = random.Random(0)
    game = start(random_base_board(rng), 4, rng)
    game.phase = Phase.MAIN
    game.current_player = 0
    for player, resource, count in hands:
        deal(game, player, resource, count)
    return game


class _Values:
    """A gate valuing each card at a fixed price: receiving one gains it,
    giving one costs it."""

    trade_floor = 0.0

    def __init__(self, **price):
        self.price = [price.get(r.name.lower(), 0.0) for r in Resource]

    def gains_many(self, view, received, counterparties):
        return [sum(p * n for p, n in zip(self.price, r)) for r in received]


def test_an_offer_is_open_when_it_holds_any_cards():
    assert is_open(Offer(0, (-1, 0, 0, 0, 0), any=1))
    assert not is_open(Offer(0, (-1, 0, 0, 0, 1)))


def test_an_open_offer_names_a_card_on_each_side_and_any_cards_on_one():
    from hexset.server.wire import round_offer_from_wire

    none, wood, ore = [0] * 5, [1, 0, 0, 0, 0], [0, 0, 0, 0, 1]
    assert round_offer_from_wire(none, ore, give_any=1) == ((0, 0, 0, 0, 1), -1)   # any card for ore
    assert round_offer_from_wire(wood, none, want_any=2) == ((-1, 0, 0, 0, 0), 2)  # wood for any two
    assert round_offer_from_wire(wood, ore) == ((-1, 0, 0, 0, 1), 0)
    for give, want, kwargs in ((wood, none, {"give_any": 1, "want_any": 1}),   # both sides
                               (none, none, {"give_any": 1}),                  # nothing asked for
                               (none, ore, {})):                               # nothing given
        try:
            round_offer_from_wire(give, want, **kwargs)
        except ValueError:
            continue
        raise AssertionError((give, want, kwargs))


def test_an_open_offer_skips_acceptance_and_is_countered_as_usual():
    """Any card for your ore, from a seat that would take wood for it as it
    stands: it answers with the counter it would send any offer, never an
    accept."""
    game = _game((0, Resource.WOOD, 1), (0, Resource.WHEAT, 2), (1, Resource.ORE, 1))
    gate = _Values(wheat=2.0, wood=1.0, ore=0.5)
    open_offer = Offer(0, (0, 0, 0, 0, 1), any=-1)
    assert default_respond(gate, game.state(1), Offer(0, (-1, 0, 0, 0, 1))).kind == RESPONSE_ACCEPT
    answer = respond(game, gate, 1, open_offer)
    assert answer.kind == RESPONSE_COUNTER and answer.bundle[ORE] == 1
    assert respond(game, _Values(ore=10.0), 1, open_offer).kind == RESPONSE_PASS


def test_an_answerer_chooses_the_card_it_gives():
    """My wood for any card: the responder gives the card it minds least,
    among those it holds."""
    game = _game((0, Resource.WOOD, 1), (1, Resource.SHEEP, 1), (1, Resource.ORE, 1))
    offer = Offer(0, (-1, 0, 0, 0, 0), any=1)
    answer = respond(game, _Values(wood=2.0, sheep=1.0, ore=3.0), 1, offer)
    assert answer == Response(1, RESPONSE_COUNTER, (-1, 0, 1, 0, 0))  # their sheep, never ore


def test_a_gate_may_answer_open_offers_itself_but_never_accepts_one():
    game = _game((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    offer = Offer(0, (0, 0, 0, 0, 1), any=-1)

    class _OwnAnswer(_Values):
        def respond_any(self, view, offer):
            self.asked = offer
            return Response(view.perspective, RESPONSE_ACCEPT, offer.received)

    gate = _OwnAnswer()
    assert respond(game, gate, 1, offer) == Response(1, RESPONSE_PASS)
    assert gate.asked == offer


def test_an_open_round_executes_the_counter_its_actor_picks():
    game = _game((0, Resource.WHEAT, 1), (1, Resource.ORE, 1))
    state = game.state(0, hidden=False)
    gates = [_Values(ore=3.0, wheat=1.0), _Values(wheat=2.0, ore=1.0), None, None]
    trades = resolve_offer(game, gates, Offer(0, (0, 0, 0, 0, 1), any=-1))
    assert [t.received for t in trades] == [(0, 0, 0, -1, 1)]
    assert state.hands[0][ORE] == 1 and state.hands[0][WHEAT] == 0
    assert state.hands[1][WHEAT] == 1 and state.hands[1][ORE] == 0


def test_a_manual_seat_is_asked_an_open_offer_once():
    game = _game((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    gate = PendingGate(game, 1)
    offer = Offer(0, (-1, 0, 0, 0, 0), any=1)
    assert respond(game, gate, 1, offer).kind == RESPONSE_PASS
    assert len(game.pending) == 1 and game.pending[0].received == offer.received


def test_an_open_offer_is_journalled_with_its_any_cards(tmp_path):
    import json

    from hexset.server._journal import Journal, notes_of
    from hexset.server._webplay import RoundNote

    journal = Journal(str(tmp_path), "g1")
    journal.note(step=3, round_num=2, note=RoundNote("offer", 0, 0, (0, 0, 0, 0, 1), -1))
    journal.note(step=3, round_num=2, note=RoundNote("counter", 1, 0, (-1, 0, 0, 0, 1)))
    lines = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert lines[0]["any"] == -1 and "any" not in lines[1]     # written only when there are any
    notes = notes_of(lines)
    assert [note for _, note in notes[3]] == [
        RoundNote("offer", 0, 0, (0, 0, 0, 0, 1), -1), RoundNote("counter", 1, 0, (-1, 0, 0, 0, 1), 0)]


def test_the_mcp_view_names_an_open_offers_any_cards_and_never_offers_accept():
    from hexset.server.mcptools import _translate_trades

    raw = {
        "seat": 1,
        "players": [{"seat": 1, "hand": {"Wood": 1, "Brick": 0, "Sheep": 0, "Wheat": 0, "Ore": 1}}],
        "pending": [{"actor": 0, "bundle": [0, 0, 0, 0, 1], "any": -1},     # any card for your ore
                    {"actor": 0, "bundle": [-1, 0, 0, 0, 1]}],
        "trade_round": None,
    }
    open_offer, plain = _translate_trades(raw)["pending"]
    assert open_offer["you_receive_any"] == 1 and open_offer["can_accept"] is False
    assert open_offer["you_give"] == {"Ore": 1}
    assert "you_receive_any" not in plain and plain["can_accept"] is True


def _protocol_gate(params, **price):
    """A `_Values` gate carrying `params`, with the protocol installed."""
    from hexset.trading import TradeProtocol, install

    gate = _Values(**price)
    gate.trade_params = params
    install(gate, TradeProtocol(gate, params, seed=0))
    return gate


def _dumping_sheep():
    """Seat 0 holds two ore and offers one for any card; seat 1 holds two
    sheep it would rather be rid of, so its best counter by value is both
    sheep for both ore."""
    game = _game((0, Resource.ORE, 2), (1, Resource.SHEEP, 2))
    return game, Offer(0, (0, 0, 0, 0, -1), any=1)


def test_a_counter_to_an_open_offer_keeps_the_gates_own_card_cap():
    from hexset.trading import TradeParams

    game, offer = _dumping_sheep()
    unbounded = respond(game, _Values(ore=3.0, sheep=-1.0), 1, offer)
    assert unbounded.kind == RESPONSE_COUNTER and unbounded.bundle[SHEEP] == 2
    capped = _protocol_gate(TradeParams(max_give_cards=1), ore=3.0, sheep=-1.0)
    answer = respond(game, capped, 1, offer)
    assert answer.kind == RESPONSE_COUNTER
    assert answer.bundle[SHEEP] == 1, "one card out, at most, as the gate declares"


def test_a_planning_gate_counters_an_open_offer_with_a_fragment():
    from hexset.trading import legal_fragment
    from traders import FRAGMENTED

    game, offer = _dumping_sheep()
    gate = _protocol_gate(FRAGMENTED, ore=3.0, sheep=-1.0)
    answer = respond(game, gate, 1, offer)
    assert answer.kind == RESPONSE_COUNTER
    assert legal_fragment(answer.bundle, max_cards=FRAGMENTED.fragment_width)


class _Asker:
    """A gate whose asks, best first, are `asks` (signed towards it), one a
    call -- the next not already offered -- declaring `open_asks`."""

    def __init__(self, asks, look=3, budget=3):
        from hexset.trading import TradeParams
        self.asks = [tuple(a) for a in asks]
        self.trade_params = TradeParams(max_offers=budget, open_asks=look)
        self.answers = {}

    def candidates(self, view, counterparties, *, turn=None, already_offered=()):
        struck = {tuple(b) for b in already_offered}
        ask = next((a for a in self.asks if a not in struck), None)
        return [] if ask is None else [(c, ask) for c in counterparties]

    def offer(self, view, candidates):
        return 0

    def respond(self, view, offer):
        kind = self.answers.get(tuple(offer.received), RESPONSE_PASS)
        return Response(view.perspective, kind, offer.received if kind != RESPONSE_PASS else None)


WOOD_ORE, SHEEP_ORE, BRICK_ORE, WOOD_WHEAT = (-1, 0, 0, 0, 1), (0, 0, -1, 0, 1), (0, -1, 0, 0, 1), (-1, 0, 0, 1, 0)


def _asking():
    return _game((0, Resource.WOOD, 2), (0, Resource.SHEEP, 1), (0, Resource.BRICK, 1), (1, Resource.ORE, 1))


@pytest.mark.parametrize("asks,slots", [((WOOD_ORE, SHEEP_ORE, BRICK_ORE), 3), ((WOOD_ORE, WOOD_WHEAT, SHEEP_ORE), 2)],
                         ids=["three-for-ore", "two-for-ore"])
def test_asks_for_one_card_go_up_as_one_open_ask_counting_as_many_offers(asks, slots):
    from hexset.trading import offer_slots
    game, offered = _asking(), set()
    gates = [_Asker(asks), None, None, None]
    made = offer(game, gates, offered, slots_left=3)
    assert made == Offer(0, (0, 0, 0, 0, 1), any=-1) and offer_slots(game, made) == slots
    following = offer(game, gates, offered, slots_left=3 - slots)
    # nothing more for the ore this turn: the wheat ask, where one is left, is next
    assert following == (Offer(0, WOOD_WHEAT) if WOOD_WHEAT in asks else None)


@pytest.mark.parametrize("asks,look,left", [((WOOD_ORE, WOOD_WHEAT), 3, 3), ((WOOD_ORE, SHEEP_ORE), 0, 3),
                                            ((WOOD_ORE, SHEEP_ORE), 3, 1)],
                         ids=["one-for-ore", "not-declared", "one-offer-left"])
def test_a_lone_ask_or_an_undeclared_gate_or_no_offers_left_asks_as_it_stands(asks, look, left):
    from hexset.trading import offer_slots
    game = _asking()
    made = offer(game, [_Asker(asks, look=look), None, None, None], set(), slots_left=left)
    assert made == Offer(0, asks[0]) and offer_slots(game, made) == 1


def test_the_same_ask_listed_twice_is_one_ask():
    game = _asking()
    assert offer(game, [_Asker((WOOD_ORE, WOOD_ORE)), None, None, None], set(), slots_left=3) == Offer(0, WOOD_ORE)


def test_a_counter_bringing_the_card_for_one_it_keeps_is_countered_back_with_one_it_offered():
    """After its open "any card for your ore": seat 1 counters, its ore for
    our wheat (we hold none to spare); our gate takes sheep-for-ore, so that
    is the counter back -- the first card it offered that it still holds and
    would trade."""
    game = _asking()
    gate = _Asker((WOOD_ORE, SHEEP_ORE, BRICK_ORE))
    gate.answers = {(0, 0, 1, 0, -1): RESPONSE_ACCEPT}       # seat 1 offering its ore for our sheep
    offer(game, [gate, None, None, None], set(), slots_left=3)
    back = respond(game, gate, 0, Offer(1, (0, 0, 0, 1, -1)))  # seat 1: my ore for your wheat
    assert back == Response(0, RESPONSE_COUNTER, (0, 0, 1, 0, -1))
    game.turns += 1                                           # not on the turn of its open ask: no counter back
    assert respond(game, gate, 0, Offer(1, (0, 0, 0, 1, -1))).kind == RESPONSE_PASS
