# SPDX-License-Identifier: GPL-3.0-only
"""A hosted seat (`hexset.seat.Seat`) decides what the engine's own tables
would decide for it, from the game it observes -- and lets a client carry
those decisions without making any.

The strongest form is equivalence: a seat's view of an observed game is the
same information set as its view of the true game, so every gate verb asked
through `Seat` must return what the engine's own stage returns on the true
game. The rest pins the bookkeeping the engine's drivers do around those
verbs -- the offer budget, the repeat rule, `trade_now`, the round's close,
the once-a-turn publication -- because a client driving a foreign table has
no other place to get them right.
"""

from __future__ import annotations

import random

import pytest

from hexset.actions import Action, ActionType, apply, legal_actions
from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.chance import UNSEEN
from hexset.game import Phase, is_over, observe, players_owing_discards, start, to_move
from hexset.seat import OutOfStep, Seat
from hexset import trading
from hexset.trading import (
    RESPONSE_ACCEPT,
    RESPONSE_COUNTER,
    RESPONSE_PASS,
    Offer,
    Response,
    TradeParams,
    bundle,
)

WOOD, BRICK, SHEEP, WHEAT, ORE = (int(r) for r in Resource)


class Wants:
    """Wants `resource`, and nothing else; offers up to `budget` a turn."""

    trade_floor = 0.0

    def __init__(self, resource: int, budget: int | None = None):
        self.resource = resource
        self.trade_params = TradeParams(max_offers=budget)
        self.finished: list = []
        self.observed: list = []

    @property
    def trade_offer_budget(self):
        return self.trade_params.trade_offer_budget

    def gains_many(self, view, received, counterparties):
        return [1.0 if r[self.resource] > 0 else -1.0 for r in received]

    def choose(self, game):
        return next(a for a in legal_actions(game) if a.type is not ActionType.END_TURN) \
            if len(legal_actions(game)) > 1 else legal_actions(game)[0]

    def trade_round_finished(self, view, offer, responses, trade, *, turn):
        self.finished.append((offer, tuple(responses), trade, turn))

    def observe_trade(self, **event):
        self.observed.append(event)


def _main(players=4, seed=0):
    """A true game in seat 0's main phase with some cards in every hand,
    and the game seat 0 observes of it."""
    rng = random.Random(seed)
    true = start(random_base_board(rng), players, rng)
    true.phase = Phase.MAIN
    true.current_player = 0
    true.turns = 5
    for seat, (r, n) in enumerate([(WOOD, 2), (ORE, 2), (ORE, 1), (SHEEP, 3)][:players]):
        true._state.hands[seat][r] += n
        true._state.bank[r] -= n
        true.ledger.receive(seat, r, n)
    return true


def _seat(true, gate, seat=0):
    return Seat(observe(true, seat), seat, gate)


def test_the_offer_budget_is_the_gates_own_and_repeats_are_refused():
    true = _main()
    gate = Wants(ORE, budget=2)
    seat = _seat(true, gate)
    first = seat.offer()
    assert first is not None
    assert seat.offer() is None, "a round of this seat's is still open"
    seat.pick([Response(1, RESPONSE_PASS), Response(2, RESPONSE_PASS), Response(3, RESPONSE_PASS)])
    seat.close_round(None)
    second = seat.offer()
    assert second is not None and second != first, "a refused offer is not made again"
    seat.close_round(None)
    assert seat.offer() is None, "two offers were this gate's budget"
    assert [f[0] for f in gate.finished] == [first, second]


def test_an_acceptance_is_priced_by_who_took_it():
    """The seat's own offer, taken by two seats: `pick` deals with the one its
    gate prices above its floor, and not with the other, alone or not."""

    class NotWithSeatTwo(Wants):
        def gains_many(self, view, received, counterparties):
            return [-1.0 if c == 2 else g
                    for g, c in zip(super().gains_many(view, received, counterparties), counterparties)]

    true = _main()
    seat = _seat(true, NotWithSeatTwo(ORE, budget=1))
    offer = seat.offer()
    one, two = (Response(s, RESPONSE_ACCEPT, offer.received) for s in (1, 2))
    assert seat.pick([two, one, Response(3, RESPONSE_PASS)]) == one
    assert seat.pick([two, Response(1, RESPONSE_PASS), Response(3, RESPONSE_PASS)]) is None


def test_a_closed_round_tells_the_gate_and_the_turn_is_published_once():
    true = _main()
    gate = Wants(ORE, budget=1)
    seat = _seat(true, gate)
    offer = seat.offer()
    took = Response(1, RESPONSE_ACCEPT, offer.received)
    chosen = seat.pick([took])
    trade = seat.traded(0, chosen.seat, offer.received)
    seat.close_round(trade)
    assert gate.finished == [(offer, (took,), trade, 5)]
    assert seat.game._state.hands[0][ORE] == 1, "the trade went into the game"
    assert seat.offer() is None, "the budget is spent: the event closes and publishes"
    assert gate.observed == [{"turn": 5, "actor": 0, "hand_sizes": (2, 2, 1, 3),
                              "trade_participants": ((0, 1),)}]
    seat.play(0, Action(ActionType.END_TURN))
    assert len(gate.observed) == 1, "published once a turn"


def test_a_move_after_the_event_opened_ends_it():
    """The engine runs a turn's rounds back to back at the decision point
    the event opened at; a seat that has moved on since does not spend the
    rest of its budget later in the turn."""
    true = _main()
    true._state.hands[0][WOOD] += 2
    true._state.bank[WOOD] -= 2
    true.ledger.receive(0, WOOD, 2)
    gate = Wants(ORE, budget=3)
    seat = _seat(true, gate)
    assert seat.offer() is not None
    seat.close_round(None)
    move = next(a for a in legal_actions(seat.game, 0) if a.type is ActionType.BANK_TRADE)
    seat.play(0, move)
    assert seat.offer() is None
    assert len(gate.observed) == 1, "the event closed, and was published"


def test_trade_now_opens_the_window_when_the_gate_says():
    true = _main()

    class Later(Wants):
        def __init__(self):
            super().__init__(ORE, budget=1)
            self.asked = 0

        def trade_now(self, game):
            self.asked += 1
            return self.asked > 1

    gate = Later()
    seat = _seat(true, gate)
    assert seat.offer() is None and gate.asked == 1, "not yet: the window stays open"
    assert seat.offer() is not None and gate.asked == 2


def test_no_offer_between_a_road_building_cards_roads():
    true = _main()
    seat = _seat(true, Wants(ORE, budget=1))
    seat.game.free_roads = 2
    assert seat.offer() is None
    seat.game.free_roads = 0
    assert seat.offer() is not None


def test_an_answer_is_the_gates_respond():
    true = _main()
    true.current_player = 1
    seat = _seat(true, Wants(ORE))
    offered = Offer(1, bundle(ore=-1, wood=1))      # seat 1 gives ore for wood
    assert seat.answer(offered) == trading.respond(true, Wants(ORE), 0, offered)
    assert seat.answer(offered).kind == RESPONSE_ACCEPT
    assert Seat(observe(true, 0), 0, object()).answer(offered).kind == RESPONSE_PASS


def test_a_whole_discard_is_the_bots_card_by_card():
    rng = random.Random(0)
    true = start(random_base_board(rng), 3, rng)
    true.phase = Phase.DISCARD
    true.current_player = 1
    for r in (WOOD, WOOD, BRICK, ORE, ORE, ORE, SHEEP, SHEEP, WHEAT):
        true._state.hands[0][r] += 1
        true._state.bank[r] -= 1
    true.discard_quota = [4, 0, 5]    # another seat owes too, and is not in the way
    seat = _seat(true, None)

    class Keeps:
        """Discards ore first, then whatever is left, in index order."""

        def choose(self, game):
            options = legal_actions(game)
            ore = [a for a in options if a.a == ORE]
            return (ore or options)[0]

    seat.bot = Keeps()
    assert seat.discard() == [ORE, ORE, ORE, WOOD]
    assert seat.game.discard_quota == [4, 0, 5], "nothing left the real hand"


def test_a_discard_the_bot_has_not_answered_yet_is_none_not_a_refusal():
    """A seat driven by a person's calls has no answer until they make
    them; that is not a bot giving up half way, which still raises."""
    rng = random.Random(0)
    true = start(random_base_board(rng), 3, rng)
    true.phase = Phase.DISCARD
    for r in (WOOD, WOOD, BRICK, ORE, ORE, ORE, SHEEP, SHEEP, WHEAT):
        true._state.hands[0][r] += 1
        true._state.bank[r] -= 1
    true.discard_quota = [4, 0, 0]
    seat = _seat(true, None)
    seat.bot = type("Later", (), {"choose": staticmethod(lambda game: None)})()
    assert seat.discard() is None

    asked = []

    def once(game):
        asked.append(game)
        return next(a for a in legal_actions(game) if a.type is ActionType.DISCARD) \
            if len(asked) == 1 else None

    seat.bot = type("Quits", (), {"choose": staticmethod(once)})()
    with pytest.raises(ValueError, match="stopped discarding"):
        seat.discard()


def test_a_finished_observed_game_renders_through_the_served_tables_view():
    """The served table's `state_view` discloses every seat once the game is
    over; on an observed game another seat's cards are counts, so it
    discloses the count and not a card it does not have, rather than
    raising at the end of every game a hosted seat plays."""
    from hexset.server._webplay import GameSession

    true = _main()
    seen = observe(true, 0)
    seen.phase = Phase.GAME_OVER
    seen.won_by = 0
    view = GameSession(game=seen, claimed_seats={0}).state_view(0)

    assert view["game_over"] is True
    ours = next(p for p in view["players"] if p["seat"] == 0)
    theirs = [p for p in view["players"] if p["seat"] != 0]
    assert "hand" in ours and all("hand" not in p for p in theirs)
    assert [p["hand_size"] for p in theirs] == [2, 1, 3]


def test_the_table_revealing_what_the_engine_did_not_take_is_out_of_step():
    """A roll revealed alongside an end of turn is a mirror that has lost its
    place: it is told so, and the stale outcome is not left queued to be
    taken by the next roll, where it would be wrong in silence."""
    true = _main()
    seat = _seat(true, None)
    with pytest.raises(OutOfStep):
        seat.play(0, Action(ActionType.END_TURN), roll=8)
    assert not seat.game.chance.pending
    with pytest.raises(OutOfStep):
        seat.play(1, Action(ActionType.ROLL))       # and a roll nobody revealed


def test_a_hosted_seat_plays_a_whole_game_it_only_observes():
    """Driven end to end: the true game plays itself, and the seat -- fed
    only what seat 2 is shown -- chooses every one of seat 2's moves exactly
    as the same bot does on the true game."""
    true, seat = _drive(6)
    assert is_over(true) or true.turns > 50
    assert seat.game.turns == true.turns


def _drive(seed=5, players=3, me=2, steps=3000):
    from hexset.bots.base import RandomBot

    rng = random.Random(seed)
    true = start(random_base_board(rng), players, rng)
    seat = Seat(observe(true, me), me, RandomBot(random.Random(1)))
    twin = RandomBot(random.Random(1))
    others = random.Random(seed + 1)
    for _ in range(steps):
        if is_over(true):
            break
        actor = (sorted(players_owing_discards(true))[0] if true.phase is Phase.DISCARD
                 else to_move(true))
        if actor == me:
            want = twin.choose(true)
            got = seat.choose()
            assert got == want
        else:
            want = others.choice(legal_actions(true, actor))
        before = [h[:] for h in true._state.hands]
        fresh = true._state.new_dev_cards[actor][:]
        apply(true, want, seat=actor)
        kw = {}
        if want.type is ActionType.ROLL:
            kw["roll"] = true.last_roll
        elif want.type is ActionType.MOVE_ROBBER and want.b < players:
            lost = [b - a for a, b in zip(true._state.hands[want.b], before[want.b])]
            if any(lost):
                kw["stolen"] = lost.index(1) if me in (actor, want.b) else UNSEEN
        elif want.type is ActionType.BUY_DEV_CARD and actor == me:
            kw["drawn"] = [b - a for a, b in zip(fresh, true._state.new_dev_cards[me])].index(1)
        elif want.type is ActionType.PLAY_MONOPOLY:
            kw["surrendered"] = [before[o][want.a] for o in range(players)
                                 if o not in (actor, me)]
        seat.play(actor, want, **kw)
    return true, seat


class Script:
    """A gate that offers one bundle and answers each offer put to it from
    `answers`, keyed by `(actor, received)` with `received` signed towards
    the actor; an offer it has no answer for is a pass."""

    trade_floor = 0.0

    def __init__(self, offers=None, answers=None):
        self.offers, self.answers = offers, answers or {}

    def gains_many(self, view, received, counterparties):
        return [1.0] * len(received)

    def candidates(self, view, counterparties, *, turn=None, already_offered=()):
        return [] if turn is None or self.offers is None else [self.offers]

    def offer(self, view, candidates):
        return 0 if candidates else None

    def respond(self, view, offer):
        kind, answer = self.answers.get((offer.actor, tuple(offer.received)), (RESPONSE_PASS, None))
        return Response(view.perspective, kind, answer)


def _moving(counter_steps, answers):
    """Seat 0 on turn offers a wood for an ore; the seat's answers are `answers`."""
    true = _main()
    seat = Seat(observe(true, 0), 0, Script((1, bundle(wood=-1, ore=1)), answers),
                counter_steps=counter_steps)
    assert seat.offer() == Offer(0, bundle(wood=-1, ore=1))
    return seat


# Seat 1's counter to seat 0's offer, two wood for its ore, and seat 0's back: a wood and a brick.
THEIRS = bundle(wood=2, ore=-1)
BACK = {(1, THEIRS): (RESPONSE_COUNTER, bundle(wood=1, brick=1, ore=-1))}


def test_on_turn_it_counters_a_counter_to_its_offer_and_takes_the_next():
    seat = _moving(3, {**BACK, (1, bundle(brick=1, ore=-1)): (RESPONSE_ACCEPT, bundle(brick=1, ore=-1))})
    assert seat.counter(1, THEIRS) == Response(0, RESPONSE_COUNTER, bundle(wood=1, brick=1, ore=-1))
    assert seat.counter(1, bundle(brick=1, ore=-1)).kind == RESPONSE_ACCEPT


@pytest.mark.parametrize("steps,kind", [(1, RESPONSE_PASS), (2, RESPONSE_COUNTER)])
def test_a_counter_back_needs_room_in_the_thread(steps, kind):
    assert _moving(steps, BACK).counter(1, THEIRS).kind == kind


def test_a_counter_past_the_thread_or_repeating_one_is_passed():
    """At two steps seat 1's third counter is out of the thread; at five, a
    counter repeating its first is."""
    capped = _moving(2, {**BACK, (1, bundle(brick=1, ore=-1)): (RESPONSE_ACCEPT, bundle(brick=1, ore=-1))})
    capped.counter(1, THEIRS)
    assert capped.counter(1, bundle(brick=1, ore=-1)).kind == RESPONSE_PASS
    repeated = _moving(5, {**BACK, (1, THEIRS): BACK[(1, THEIRS)]})
    repeated.counter(1, THEIRS)
    assert repeated.counter(1, THEIRS).kind == RESPONSE_PASS


def test_off_turn_it_answers_the_movers_counter_to_its_own_counter():
    """Seat 1 on turn offers an ore for a wood; seat 0 counters two ore for
    it, and seat 1 counters that with an ore and a brick, which seat 0 takes."""
    true = _main()
    true.current_player = 1
    offered = Offer(1, bundle(ore=-1, wood=1))
    gate = Script(answers={
        (1, bundle(ore=-1, wood=1)): (RESPONSE_COUNTER, bundle(ore=-2, wood=1)),
        (1, bundle(ore=-1, brick=-1, wood=1)): (RESPONSE_ACCEPT, bundle(ore=-1, brick=-1, wood=1)),
    })
    seat = Seat(observe(true, 0), 0, gate, counter_steps=3)
    assert seat.counter(1, bundle(ore=-1, brick=-1, wood=1)).kind == RESPONSE_PASS   # it never countered
    assert seat.answer(offered).kind == RESPONSE_COUNTER
    assert seat.counter(1, bundle(ore=-1, brick=-1, wood=1)).kind == RESPONSE_ACCEPT
    assert seat.counter(2, bundle(ore=-1, wood=1)).kind == RESPONSE_PASS             # seat 2 is not on turn
