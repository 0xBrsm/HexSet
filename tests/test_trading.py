# SPDX-License-Identifier: GPL-3.0-only
"""The one-event trade mechanic (`hexset.trading`), registered
`agents/reference/trading-final.md`: gates return magnitudes, the table
clears the deal `Game.trade_rule` ranks highest, and nothing is published.
"""

from __future__ import annotations

import random

import pytest

import hexset.trading as trading_mod
from hexset.actions import Action, ActionType
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.game import Phase, enter_main, end_turn, imagine, start
from hexset.trading import (
    Trade,
    _best_clearing,
    _candidates,
    bundle,
    exchange,
    execute_trade,
    holds,
    one_for_one,
    trade_event,
    valued,
    valued_many,
)
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


class Trader:
    """A seat whose gate returns a caller-supplied gain per candidate.

    `gain(received, counterparty) -> float` decides what this seat prices
    every candidate it is asked about at; the default never trades. Every
    candidate asked is recorded to `asked`, in the order the batched call
    received it.
    """

    def __init__(self, gain=lambda received, counterparty: -1.0):
        self.gain = gain
        self.asked: list[tuple[tuple[int, ...], int]] = []
        self.calls = 0

    def gains_many(self, view, received, counterparties):
        self.calls += 1
        out = []
        for r, c in zip(received, counterparties):
            r = tuple(r)
            self.asked.append((r, c))
            out.append(self.gain(r, c))
        return out


def wants(resource: int, magnitude: float = 1.0):
    """A gain function positive only when this candidate hands the seat
    more of `resource` than it had, negative otherwise (magnitude ignored
    on the negative side).

    A gate that is direction-blind (a flat `lambda r, c: 1.0`) prices the
    reverse of a trade it just took the same as the trade itself, and the
    engine (correctly) ping-pongs it back and forth until it trips the
    cards-on-the-table assertion -- exactly the "broken gate" case that
    assertion exists to catch. Every test below that actually clears a
    trade uses this instead, so the fixed test double behaves like the
    real gates it stands in for: strictly better in the direction it wants,
    never in reverse.
    """
    return lambda received, counterparty: magnitude if received[resource] > 0 else -1.0


def _unused_gate(seat, view, received, other):
    raise AssertionError("the single-ask gate must not be used when game.gates is seated")


def run(game, traders) -> list[Trade]:
    """Seat `traders` as the game's gates and run one trade event."""
    game.gates = tuple(traders)
    return trade_event(game, _unused_gate)


def ab_received(trades: list[Trade]) -> list[tuple[int, int, tuple[int, ...]]]:
    """A trade list, ignoring the recorded gains -- for tests that only care
    which exchange cleared."""
    return [(t.a, t.b, t.received) for t in trades]


# --- the bundle helpers -------------------------------------------------------


def test_a_bundle_reads_by_resource_name():
    assert bundle(wood=2, ore=1) == (2, 0, 0, 0, 1)


def test_holds_checks_every_resource():
    game = stocked((0, Resource.WOOD, 1))
    assert holds(game._state, 0, bundle(wood=1))
    assert not holds(game._state, 0, bundle(wood=1, ore=1))


def test_one_for_one_is_signed_towards_the_receiver():
    assert one_for_one(WOOD, ORE) == (-1, 0, 0, 0, 1)


def test_exchange_moves_both_sides_and_conserves_the_cards():
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 1))
    before = sum(sum(hand) for hand in game._state.hands)
    exchange(game._state, 0, 1, one_for_one(WOOD, ORE))
    assert game._state.hands[0][WOOD] == 1
    assert game._state.hands[0][ORE] == 1
    assert game._state.hands[1][WOOD] == 1
    assert game._state.hands[1][ORE] == 0
    assert sum(sum(hand) for hand in game._state.hands) == before


def test_disjoint_sides_forbids_a_resource_on_both_sides_of_one_bundle():
    game = stocked(
        (0, Resource.WOOD, 1),
        (0, Resource.ORE, 1),
        (1, Resource.WOOD, 1),
        (1, Resource.ORE, 1),
    )
    candidates = list(_candidates(game._state, 0, game.locked))
    assert set(candidates) == {
        (1, bundle(wood=-1, ore=1)),
        (1, bundle(ore=-1, wood=1)),
    }


def test_candidates_are_coverable_from_the_true_hands():
    game = stocked(
        (0, Resource.WOOD, 2),
        (0, Resource.BRICK, 1),
        (1, Resource.ORE, 2),
        (1, Resource.SHEEP, 3),
    )
    state = game._state
    candidates = list(_candidates(state, 0, game.locked))
    assert candidates
    for them, received in candidates:
        for r in range(NUM_RESOURCES):
            if received[r] < 0:
                assert -received[r] <= state.hands[0][r]
            elif received[r] > 0:
                assert received[r] <= state.hands[them][r]


# --- both gates strictly positive required -------------------------------------


def test_a_deal_both_sides_gain_from_clears():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [
        Trader(wants(ORE)),
        Trader(wants(WOOD)),
        Trader(),
        Trader(),
    ]
    done = run(game, traders)
    assert ab_received(done) == [(0, 1, one_for_one(WOOD, ORE))]
    assert done[0].gain_a == 1.0 and done[0].gain_b == 1.0
    assert game._state.hands[0][ORE] == 1
    assert game._state.hands[1][WOOD] == 1
    assert game.trades_made == 1
    assert game.trades == done


@pytest.mark.parametrize("zeroed", [0, 1])
def test_either_side_priced_at_zero_or_below_vetoes_the_deal(zeroed):
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    gains = [1.0, 1.0]
    gains[zeroed] = 0.0
    traders = [
        Trader(lambda r, c: gains[0]),
        Trader(lambda r, c: gains[1]),
        Trader(),
        Trader(),
    ]
    assert run(game, traders) == []


def test_a_gain_at_or_below_the_floor_does_not_clear_but_above_it_does(monkeypatch):
    """`TRADE_FLOOR` (`hexset.trading.TRADE_FLOOR`) is the measured 0.0197;
    the floor is pinned at a round value by monkeypatching it -- the same
    admission point (`clears_floor`, read by `_best_clearing`'s "mine"
    subset) that a shipped measurement will later set for real."""
    monkeypatch.setattr(trading_mod, "TRADE_FLOOR", 1.0)

    at_floor = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [Trader(wants(ORE, 1.0)), Trader(wants(WOOD, 5.0)), Trader(), Trader()]
    assert run(at_floor, traders) == [], "a gain in (0, floor] must not clear"

    above_floor = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [Trader(wants(ORE, 1.5)), Trader(wants(WOOD, 5.0)), Trader(), Trader()]
    done = run(above_floor, traders)
    assert len(done) == 1 and done[0].gain_a == 1.5


def test_only_the_current_player_trades():
    """Seats 1 and 2 would both love the swap; it is not their turn."""
    game = stocked((1, Resource.WOOD, 1), (2, Resource.ORE, 1))
    traders = [
        Trader(),
        Trader(lambda r, c: 1.0),
        Trader(lambda r, c: 1.0),
        Trader(),
    ]
    assert run(game, traders) == []


def test_a_locked_seat_is_never_a_counterparty():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.locked = frozenset({1})
    traders = [
        Trader(lambda r, c: 1.0),
        Trader(lambda r, c: 1.0),
        Trader(),
        Trader(),
    ]
    assert run(game, traders) == []


def test_the_engine_checks_coverage_itself():
    """Seat 1 has nothing to give, so no candidate against it exists at all."""
    game = stocked((0, Resource.WOOD, 1))
    traders = [
        Trader(lambda r, c: 1.0),
        Trader(lambda r, c: 1.0),
        Trader(),
        Trader(),
    ]
    assert run(game, traders) == []


def test_the_ledger_certifies_what_a_trade_moved():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [Trader(wants(ORE)), Trader(wants(WOOD)), Trader(), Trader()]
    run(game, traders)
    assert game.ledger.seats[1].known[WOOD] == 1


def test_both_sides_are_handed_their_own_view_and_nothing_else():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [Trader(wants(ORE)), Trader(wants(WOOD)), Trader(), Trader()]
    run(game, traders)
    assert traders[0].asked and traders[1].asked


# --- the loop -----------------------------------------------------------------


def test_the_event_keeps_going_while_anything_clears():
    """Wood clears against seat 1 in one bundle; brick, still held, only
    then clears against seat 2 -- a second, unrelated deal with a different
    counterparty is what makes the loop necessary."""
    game = stocked(
        (0, Resource.WOOD, 3), (0, Resource.BRICK, 3), (1, Resource.ORE, 3), (2, Resource.WHEAT, 3)
    )

    def seat0_gain(received, counterparty):
        # Wants ore from 1, wheat from 2, in full-hand bundles only.
        if counterparty == 1:
            return 1.0 if received == bundle(wood=-3, ore=3) else -1.0
        if counterparty == 2:
            return 1.0 if received == bundle(brick=-3, wheat=3) else -1.0
        return -1.0

    traders = [
        Trader(seat0_gain),
        Trader(lambda r, c: 1.0 if r == bundle(ore=-3, wood=3) else -1.0),
        Trader(lambda r, c: 1.0 if r == bundle(wheat=-3, brick=3) else -1.0),
        Trader(),
    ]
    done = run(game, traders)
    assert ab_received(done) == [
        (0, 1, bundle(wood=-3, ore=3)),
        (0, 2, bundle(brick=-3, wheat=3)),
    ]
    assert game._state.hands[0] == [0, 0, 0, 3, 3]
    assert game._state.hands[1] == [3, 0, 0, 0, 0]
    assert game._state.hands[2] == [0, 3, 0, 0, 0]


class _Fussy:
    """Only ever wants one more ore than it currently holds -- reads the
    live view, unlike `Trader`'s fixed-gain stub, so the gate genuinely
    stops saying yes once that ore has moved."""

    def gains_many(self, view, received, counterparties):
        seat = view.perspective
        return [
            1.0 if view.state.hands[seat][ORE] < 1 else -1.0
            for _ in zip(received, counterparties)
        ]


def test_a_gate_that_stops_saying_yes_stops_the_event():
    game = stocked((0, Resource.WOOD, 3), (1, Resource.ORE, 3))
    traders = [
        _Fussy(),
        Trader(lambda r, c: 1.0),
        Trader(),
        Trader(),
    ]
    assert len(run(game, traders)) == 1


def test_max_trades_zero_is_the_off_switch():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.max_trades = 0
    traders = [
        Trader(lambda r, c: 1.0),
        Trader(lambda r, c: 1.0),
        Trader(),
        Trader(),
    ]
    assert run(game, traders) == []
    assert not traders[0].asked, "a switched-off event should not even ask a gate"


def test_an_event_never_outruns_the_cards_on_the_table():
    game = stocked((0, Resource.WOOD, 4), (1, Resource.ORE, 4))
    cards = sum(sum(hand) for hand in game._state.hands)
    traders = [
        Trader(wants(ORE)),
        Trader(wants(WOOD)),
        Trader(),
        Trader(),
    ]
    done = run(game, traders)
    assert 0 < len(done) <= cards


def test_the_count_and_the_log_reset_with_the_turn():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    run(game, [Trader(wants(ORE)), Trader(wants(WOOD)), Trader(), Trader()])
    assert game.trades_made == 1
    end_turn(game)
    assert game.trades_made == 0
    assert game.trades == []


def test_trading_is_not_in_the_action_space():
    names = {kind.name for kind in ActionType}
    assert names & {"PROPOSE_TRADE", "ACCEPT_TRADE", "DECLINE_TRADE"} == set()
    assert "BANK_TRADE" in names
    assert Action(ActionType.BANK_TRADE, 0, 1)._fields == ("type", "a", "b")


# --- the trade rule: egalitarian (default), nash, actor ------------------------
#
# One fixture, three counterparties, each offering the same shaped bundle
# (seat 0 gives its one wood for one ore), so the *only* thing that decides
# which one clears is the selection rule over (gain_me, gain_them):
#
#   counterparty 1: (mine=5,    theirs=5)    egalitarian key 5,   nash 25,   actor 5
#   counterparty 2: (mine=20,   theirs=0.1)  egalitarian key 0.1, nash 2,    actor 20
#   counterparty 3: (mine=4.9,  theirs=100)  egalitarian key 4.9, nash 490,  actor 4.9
#
# egalitarian picks 1 (5 beats 4.9 and 0.1); actor picks 2 (20 is the
# largest own gain); nash picks 3 (490 is the largest product) -- three
# different winners from the same candidate set.


def _rule_fixture():
    game = stocked(
        (0, Resource.WOOD, 1), (1, Resource.ORE, 1), (2, Resource.ORE, 1), (3, Resource.ORE, 1)
    )
    seat0_gain_by_counterparty = {1: 5.0, 2: 20.0, 3: 4.9}
    counterparty_gain = {1: 5.0, 2: 0.1, 3: 100.0}

    def seat0_gain(received, counterparty):
        # Only wants ore, so the reverse of whichever trade clears (giving
        # the ore back) is never priced positively -- no ping-pong.
        return seat0_gain_by_counterparty[counterparty] if received[ORE] > 0 else -1.0

    traders = [
        Trader(seat0_gain),
        Trader(wants(WOOD, counterparty_gain[1])),
        Trader(wants(WOOD, counterparty_gain[2])),
        Trader(wants(WOOD, counterparty_gain[3])),
    ]
    return game, traders


def test_egalitarian_is_the_default_and_maximises_the_smaller_gain():
    game, traders = _rule_fixture()
    assert game.trade_rule == "egalitarian"
    done = run(game, traders)
    assert len(done) == 1
    assert done[0].b == 1
    assert done[0].gain_a == 5.0 and done[0].gain_b == 5.0


def test_actor_rule_maximises_the_current_players_own_gain():
    game, traders = _rule_fixture()
    game.trade_rule = "actor"
    done = run(game, traders)
    assert len(done) == 1
    assert done[0].b == 2
    assert done[0].gain_a == 20.0


def test_nash_rule_maximises_the_product_of_both_gains():
    game, traders = _rule_fixture()
    game.trade_rule = "nash"
    done = run(game, traders)
    assert len(done) == 1
    assert done[0].b == 3
    assert done[0].gain_a == pytest.approx(4.9) and done[0].gain_b == pytest.approx(100.0)


def test_an_unknown_trade_rule_is_rejected_at_start():
    with pytest.raises(ValueError, match="unknown trade rule"):
        start(random_base_board(random.Random(0)), 4, random.Random(0), trade_rule="bogus")


# --- the engine fires the event eagerly, not lazily ----------------------------


def test_entering_main_clears_the_turn_s_first_event_immediately():
    """There is no lazy trigger any more: `enter_main` -- reached from
    `roll_dice`/`move_robber_to` -- runs the event synchronously, before
    anything else observes the game."""
    game = a_game()
    game.phase = Phase.ROLL
    give(game._state, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)
    game.gates = (
        Trader(wants(ORE)),
        Trader(wants(WOOD)),
        Trader(),
        Trader(),
    )
    enter_main(game)
    assert game.phase is Phase.MAIN
    assert game.trades_made == 1


def test_a_game_with_nobody_seated_simply_does_not_trade():
    game = a_game()
    game.phase = Phase.ROLL
    give(game._state, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)
    enter_main(game)
    assert game.trades == []


# --- default gate dispatch (`valued`/`valued_many`) ----------------------------


class _AcceptsOnly:
    """A trader with plain `accepts` and no `accepts_many`/`gains_many` --
    exercises the structural default chain: `accepts_many` loops over
    `accepts`, and `gains_many` maps that to +1.0/-1.0."""

    def __init__(self):
        self.asked: list[tuple[tuple[int, ...], int]] = []

    def accepts(self, view, received, counterparty):
        self.asked.append((tuple(received), counterparty))
        return received[WOOD] > 0


def test_valued_many_default_loops_over_accepts_in_order():
    trader = _AcceptsOnly()
    view = a_game().state(0)
    received = [
        one_for_one(ORE, WOOD),  # gives wood -> True -> +1.0
        one_for_one(WOOD, ORE),  # takes wood -> False -> -1.0
    ]
    counterparties = [1, 2]

    many = valued_many(trader, view, received, counterparties)
    assert many == [1.0, -1.0]
    assert trader.asked == list(zip(received, counterparties))
    assert valued(trader, view, received[0], counterparties[0]) == 1.0


def test_a_bot_with_no_trading_surface_never_trades():
    class JustChoose:
        def choose(self, game):  # pragma: no cover -- not exercised here
            raise NotImplementedError

    view = a_game().state(0)
    assert valued(JustChoose(), view, one_for_one(WOOD, ORE), 1) == -1.0

    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [JustChoose(), Trader(lambda r, c: 1.0), Trader(), Trader()]
    assert run(game, traders) == []


# --- batching: one call to the acting seat, at most one per counterparty ------


def test_the_acting_seats_gate_is_asked_at_most_once_per_call():
    """One batched call to `_best_clearing` asks the acting seat's gate
    once, over every coverable candidate, and each counterparty whose
    candidates priced above zero once more -- never once per candidate."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1), (2, Resource.SHEEP, 1))
    traders = [
        # Blanket-positive on purpose, so both counterparties' candidates
        # reach this seat's gate above zero and both get asked -- seat 1's
        # own direction-aware gate (`wants`) is what stops any ping-pong in
        # the full-event tests elsewhere; this test calls `_best_clearing`
        # only once, so that concern does not apply here.
        Trader(lambda r, c: 1.0),
        Trader(wants(WOOD)),
        Trader(lambda r, c: -1.0),
        Trader(),
    ]
    game.gates = tuple(traders)
    result = _best_clearing(game, 0, _unused_gate, game.state)
    assert result is not None
    them, received, gain_me, gain_them = result
    assert them == 1 and received == one_for_one(WOOD, ORE)
    assert traders[0].calls == 1
    assert len(traders[0].asked) == 2, "one candidate per counterparty"
    assert traders[1].calls == 1  # accepted -- asked once
    assert traders[2].calls == 1  # asked once, even though it then declines


# --- imagine ------------------------------------------------------------------


def test_an_imagined_game_does_not_carry_the_seated_gates():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.gates = tuple(Trader(lambda r, c: 1.0) for _ in range(4))
    child = imagine(game, random.Random(1))
    assert child.gates is None
    child.phase = Phase.MAIN
    enter_main(child)
    assert child.trades == []


def test_an_imagined_game_carries_the_trade_rule():
    game = a_game()
    game.trade_rule = "nash"
    child = imagine(game, random.Random(1))
    assert child.trade_rule == "nash"


def test_an_imagined_game_carries_the_trade_switch_and_the_log():
    game = stocked((0, Resource.WOOD, 1))
    game.max_trades = 0
    game.trades.append(Trade(0, 1, one_for_one(WOOD, ORE)))
    game.trades_made = 1
    child = imagine(game, random.Random(1))
    assert child.max_trades == 0
    assert child.trades_made == 1
    child.trades.append(Trade(0, 2, one_for_one(WOOD, ORE)))
    assert len(game.trades) == 1


# --- Game.pending (a snapshot of the last event) -------------------------------


def test_pending_is_cleared_at_the_start_of_every_trade_event():
    game = stocked((0, Resource.WOOD, 1))
    game.pending.append(Trade(1, 0, one_for_one(WOOD, ORE)))
    game.gates = tuple(Trader() for _ in range(4))
    trade_event(game, _unused_gate)
    assert game.pending == []


def test_pending_is_cleared_by_end_turn():
    game = a_game()
    game.pending.append(Trade(1, 0, one_for_one(WOOD, ORE)))
    end_turn(game)
    assert game.pending == []


def test_pending_is_not_copied_by_imagine():
    game = a_game()
    game.pending.append(Trade(1, 0, one_for_one(WOOD, ORE)))
    child = imagine(game, random.Random(1))
    assert child.pending == []


# --- execute_trade (the negotiation interface, docs/negotiation-interface.md) -


def _seated(game, traders):
    game.gates = tuple(traders)


def test_execute_trade_clears_on_coverage_and_the_counterpartys_gain():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    _seated(game, [Trader(), Trader(lambda r, c: 1.0), Trader(), Trader()])
    received = one_for_one(WOOD, ORE)  # proposer gives wood, receives ore
    trade = execute_trade(game, 0, 1, received)
    assert trade.a == 0 and trade.b == 1 and trade.received == received
    assert trade.gain_b == 1.0
    assert game._state.hands[0] == [0, 0, 0, 0, 1]
    assert game._state.hands[1] == [1, 0, 0, 0, 0]
    assert game.trades == [trade]
    assert game.trades_made == 1


def test_execute_trade_rejects_whichever_side_cannot_cover_it():
    proposer_short = stocked((1, Resource.ORE, 1))
    _seated(proposer_short, [Trader(), Trader(lambda r, c: 1.0), Trader(), Trader()])
    with pytest.raises(ValueError, match="seat 0 cannot cover"):
        execute_trade(proposer_short, 0, 1, one_for_one(WOOD, ORE))

    counterparty_short = stocked((0, Resource.WOOD, 1))
    _seated(counterparty_short, [Trader(), Trader(lambda r, c: 1.0), Trader(), Trader()])
    with pytest.raises(ValueError, match="seat 1 cannot cover"):
        execute_trade(counterparty_short, 0, 1, one_for_one(WOOD, ORE))


def test_execute_trade_refuses_a_counterparty_priced_at_zero_or_below():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    _seated(game, [Trader(), Trader(lambda r, c: 0.0), Trader(), Trader()])
    with pytest.raises(ValueError, match="does not want"):
        execute_trade(game, 0, 1, one_for_one(WOOD, ORE))


def test_execute_trade_refuses_a_counterparty_gain_under_a_nonzero_floor(monkeypatch):
    """A gain that is positive but at or below `TRADE_FLOOR` still refuses --
    the same `clears_floor` predicate `_best_clearing` reads, not a bare
    `> 0.0` check."""
    monkeypatch.setattr(trading_mod, "TRADE_FLOOR", 2.0)
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    _seated(game, [Trader(), Trader(lambda r, c: 1.0), Trader(), Trader()])
    with pytest.raises(ValueError, match="does not want"):
        execute_trade(game, 0, 1, one_for_one(WOOD, ORE))


def test_execute_trade_never_asks_the_proposers_own_gate():
    """A gate that raises if ever called, seated on the proposer, still lets
    the trade clear -- submitting is consent, so the proposer's own gate is
    never consulted."""

    class Boom:
        def gains_many(self, view, received, counterparties):
            raise AssertionError("the proposer's own gate must never be asked")

    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    _seated(game, [Boom(), Trader(lambda r, c: 1.0), Trader(), Trader()])
    trade = execute_trade(game, 0, 1, one_for_one(WOOD, ORE))
    assert trade.a == 0 and trade.b == 1


def test_execute_trade_rejects_a_seat_that_is_neither_proposer_nor_current_player():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.current_player = 2
    _seated(game, [Trader(), Trader(lambda r, c: 1.0), Trader(), Trader()])
    with pytest.raises(ValueError, match="neither seat"):
        execute_trade(game, 0, 1, one_for_one(WOOD, ORE))


def test_execute_trade_allows_the_current_player_to_answer_another_seats_offer():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.current_player = 1
    _seated(game, [Trader(), Trader(lambda r, c: 1.0), Trader(), Trader()])
    trade = execute_trade(game, 0, 1, one_for_one(WOOD, ORE))
    assert trade.a == 0 and trade.b == 1


def test_execute_trade_requires_main_phase():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.phase = Phase.ROLL
    _seated(game, [Trader(), Trader(lambda r, c: 1.0), Trader(), Trader()])
    with pytest.raises(ValueError, match="MAIN"):
        execute_trade(game, 0, 1, one_for_one(WOOD, ORE))


def test_execute_trade_rejects_a_seat_trading_with_itself():
    game = stocked((0, Resource.WOOD, 1))
    _seated(game, [Trader(), Trader(), Trader(), Trader()])
    with pytest.raises(ValueError, match="itself"):
        execute_trade(game, 0, 0, one_for_one(WOOD, ORE))


def test_execute_trade_bypasses_candidates_any_coverable_bundle_is_legal():
    """`_candidates` still enumerates single-resource-per-side multisets; a
    manual trade is not limited to what it would have found."""
    game = stocked((0, Resource.WOOD, 2), (0, Resource.BRICK, 1), (1, Resource.ORE, 3))
    _seated(game, [Trader(), Trader(lambda r, c: 1.0), Trader(), Trader()])
    received = [0, 0, 0, 0, 0]
    received[ORE] = 3
    received[WOOD] = -2
    received[BRICK] = -1
    trade = execute_trade(game, 0, 1, tuple(received))
    assert trade.received == tuple(received)
    assert game._state.hands[0][ORE] == 3
    assert game._state.hands[1][WOOD] == 2 and game._state.hands[1][BRICK] == 1
