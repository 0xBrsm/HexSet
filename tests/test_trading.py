# SPDX-License-Identifier: GPL-3.0-only
"""The one-event trade mechanic (`hexset.trading`), gate (i) of the trading
design's registration: clearing, the veto, termination, the tie-break, and
what `imagine` carries."""

from __future__ import annotations

import random

import pytest

from hexset.actions import Action, ActionType
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.game import Phase, end_turn, enter_main, imagine, move_robber_to, roll_dice, start
from hexset.trading import (
    NO_VALUATION,
    Trade,
    bundle,
    exchange,
    holds,
    one_for_one,
    trade_event,
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


def vector(**amounts: float) -> tuple[float, ...]:
    out = [0.0] * NUM_RESOURCES
    for name, value in amounts.items():
        out[Resource[name.upper()]] = value
    return tuple(out)


class Trader:
    """A seat that publishes a fixed vector and answers a fixed gate."""

    def __init__(self, vec=NO_VALUATION, gate=True):
        self.vec = vec
        self.gate = gate
        self.asked: list[tuple[tuple[int, ...], int]] = []
        self.views: list = []

    def valuation(self, view):
        self.views.append(view)
        return self.vec

    def accepts(self, view, received, counterparty):
        self.asked.append((tuple(received), counterparty))
        return self.gate


def run(game, traders):
    """Seat `traders` as the game's gates, publish each one's vector exactly
    as a driver would at that seat's own decision, then run the event.

    `trade_event` itself no longer takes a valuation callback -- it only
    reads `game.valuations` -- so the harness does what `arena.play`'s loop
    (and every other driver) does: ask, then `Game.publish`, once per seat,
    before the event that reads them.
    """
    game.gates = tuple(traders)
    for seat, trader in enumerate(traders):
        game.publish(seat, trader.valuation(game.state(seat)))
    return trade_event(
        game,
        lambda seat, view, received, other: traders[seat].accepts(view, received, other),
    )


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


# --- clearing -----------------------------------------------------------------


def test_a_deal_both_sides_want_clears():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    ]
    done = run(game, traders)
    assert done == [Trade(0, 1, one_for_one(WOOD, ORE))]
    assert game._state.hands[0][ORE] == 1
    assert game._state.hands[1][WOOD] == 1
    assert game.trades_made == 1
    assert game.trades == done


def test_the_ledger_certifies_what_a_trade_moved():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    ]
    run(game, traders)
    assert game.ledger.seats[1].known[WOOD] == 1


def test_a_seat_that_publishes_nothing_never_trades():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [Trader(vector(ore=1.0, wood=-1.0)), Trader(), Trader(), Trader()]
    assert run(game, traders) == []


def test_one_positive_surplus_is_not_enough():
    """The counterparty has to want it too, not merely not mind."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=1.0)),  # values them equally: surplus 0
        Trader(),
        Trader(),
    ]
    assert run(game, traders) == []


def test_ties_do_not_clear():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [
        Trader(vector(ore=0.5, wood=0.5)),
        Trader(vector(wood=0.5, ore=0.5)),
        Trader(),
        Trader(),
    ]
    assert run(game, traders) == []


def test_either_private_gate_vetoes_a_deal_both_vectors_advertise():
    """§8.3's safety property: no vector anyone posts can force a trade a
    seat's own gate rejects."""
    for refuser in (0, 1):
        game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
        traders = [
            Trader(vector(ore=1.0, wood=-1.0)),
            Trader(vector(wood=1.0, ore=-1.0)),
            Trader(),
            Trader(),
        ]
        traders[refuser].gate = False
        assert run(game, traders) == [], f"seat {refuser}'s veto was ignored"


def test_the_gate_is_asked_about_the_bundle_from_each_seat_s_own_side():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    ]
    run(game, traders)
    assert traders[0].asked[0] == (one_for_one(WOOD, ORE), 1)
    assert traders[1].asked[0] == (one_for_one(ORE, WOOD), 0)


def test_both_sides_are_handed_their_own_view_and_nothing_else():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [Trader(vector(ore=1.0, wood=-1.0)) for _ in range(4)]
    run(game, traders)
    for seat, trader in enumerate(traders):
        assert trader.views, f"seat {seat} was never asked to publish"
        assert all(view.perspective == seat for view in trader.views)
        assert all(not view.omniscient for view in trader.views)


def test_only_the_current_player_trades():
    """Seats 1 and 2 would both love the swap; it is not their turn."""
    game = stocked((1, Resource.WOOD, 1), (2, Resource.ORE, 1))
    traders = [
        Trader(),
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
    ]
    assert run(game, traders) == []


def test_a_locked_seat_is_never_a_counterparty():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.locked = frozenset({1})
    traders = [
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    ]
    assert run(game, traders) == []


def test_the_engine_checks_coverage_itself():
    """Seat 1 wants wood and would give ore, but holds none to give."""
    game = stocked((0, Resource.WOOD, 1))
    traders = [
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    ]
    assert run(game, traders) == []


# --- the loop -----------------------------------------------------------------


def test_the_event_keeps_going_while_anything_clears():
    game = stocked((0, Resource.WOOD, 3), (1, Resource.ORE, 3))
    traders = [
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    ]
    done = run(game, traders)
    assert len(done) == 3
    assert game._state.hands[0][ORE] == 3
    assert game._state.hands[0][WOOD] == 0


def test_a_gate_that_stops_saying_yes_stops_the_event():
    game = stocked((0, Resource.WOOD, 3), (1, Resource.ORE, 3))

    class Fussy(Trader):
        def accepts(self, view, received, counterparty):
            # Only ever wants one more ore.
            return view.state.hands[view.perspective][ORE] < 1

    traders = [
        Fussy(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    ]
    assert len(run(game, traders)) == 1


def test_max_trades_zero_is_the_off_switch():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.max_trades = 0
    traders = [
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    ]
    assert run(game, traders) == []
    assert not traders[0].asked, "a switched-off event should not even ask a gate"


def test_the_published_vectors_are_recorded_on_the_game():
    game = stocked((0, Resource.WOOD, 1))
    traders = [Trader(vector(ore=0.25)) for _ in range(4)]
    run(game, traders)
    assert game.valuations[0] == vector(ore=0.25)
    assert len(game.valuations) == 4


def test_a_published_vector_must_be_five_numbers_in_range():
    game = stocked((0, Resource.WOOD, 1))
    for bad in ((1.0, 1.0), (0.0, 0.0, 0.0, 0.0, 2.0)):
        traders = [Trader(bad)] + [Trader() for _ in range(3)]
        with pytest.raises(ValueError):
            run(game, traders)


def test_the_vectors_are_fixed_for_the_whole_event():
    """Published at the last decision; only the private gates move as cards
    change hands, which is what makes the second ore worth less."""
    game = stocked((0, Resource.WOOD, 2), (1, Resource.ORE, 2))
    traders = [
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    ]
    run(game, traders)
    assert len(traders[0].views) == 1
    assert len(traders[1].views) == 1


# --- the tie-break ------------------------------------------------------------


def test_the_best_deal_goes_first():
    """Two clearing swaps; the one with the larger smaller-surplus wins."""
    game = stocked((0, Resource.WOOD, 1), (0, Resource.BRICK, 1), (1, Resource.ORE, 1))
    traders = [
        Trader(vector(ore=1.0, wood=-0.1, brick=-0.9)),
        Trader(vector(brick=0.9, wood=0.1, ore=-1.0)),
        Trader(),
        Trader(),
    ]
    done = run(game, traders)
    # brick->ore: min(1.0 - -0.9, 0.9 - -1.0) = 1.9
    # wood->ore: min(1.0 - -0.1, 0.1 - -1.0) = 1.1
    assert done[0].received == one_for_one(BRICK, ORE)


def test_equal_surpluses_break_by_the_canonical_trade_index():
    """Wood(0) and brick(1) are worth the same to both sides, so the pair
    ordered first by `given * NUM_RESOURCES + wanted` clears first."""
    game = stocked((0, Resource.WOOD, 1), (0, Resource.BRICK, 1), (1, Resource.ORE, 1))
    traders = [
        Trader(vector(ore=1.0, wood=-1.0, brick=-1.0)),
        Trader(vector(wood=1.0, brick=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    ]
    done = run(game, traders)
    assert done[0].received == one_for_one(WOOD, ORE)


def test_equal_deals_break_by_the_lower_counterparty():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1), (2, Resource.ORE, 1))
    partner = vector(wood=1.0, ore=-1.0)
    traders = [
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(partner),
        Trader(partner),
        Trader(),
    ]
    assert run(game, traders)[0].b == 1


# --- where the event runs -----------------------------------------------------


def _seat_and_publish(game, traders):
    """Seat `traders` as the game's gates and publish each one's vector --
    what a driver has already done by the time a roll or a robber move
    reaches `enter_main`, since publishing rides on a seat's own decision,
    not on the trade event."""
    game.gates = tuple(traders)
    for seat, trader in enumerate(traders):
        game.publish(seat, trader.vec)


def test_the_event_runs_on_the_way_into_main_from_a_roll():
    game = a_game()
    game.phase = Phase.ROLL
    give(game._state, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)
    _seat_and_publish(
        game,
        (
            Trader(vector(ore=1.0, wood=-1.0)),
            Trader(vector(wood=1.0, ore=-1.0)),
            Trader(),
            Trader(),
        ),
    )
    roll_dice(game, 8)
    assert game.phase is Phase.MAIN
    assert game.trades_made == 1


def test_the_event_runs_on_the_way_into_main_from_the_robber():
    game = a_game()
    game.phase = Phase.ROBBER
    give(game._state, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)
    _seat_and_publish(
        game,
        (
            Trader(vector(ore=1.0, wood=-1.0)),
            Trader(vector(wood=1.0, ore=-1.0)),
            Trader(),
            Trader(),
        ),
    )
    move_robber_to(game, 3)
    assert game.phase is Phase.MAIN
    assert game.trades_made == 1


def test_a_game_with_nobody_seated_simply_does_not_trade():
    game = a_game()
    game.phase = Phase.ROLL
    give(game._state, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)
    roll_dice(game, 8)
    assert game.trades == []


def test_the_count_and_the_log_reset_with_the_turn():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    run(
        game,
        [
            Trader(vector(ore=1.0, wood=-1.0)),
            Trader(vector(wood=1.0, ore=-1.0)),
            Trader(),
            Trader(),
        ],
    )
    assert game.trades_made == 1
    end_turn(game)
    assert game.trades_made == 0
    assert game.trades == []


def test_trading_is_not_in_the_action_space():
    names = {kind.name for kind in ActionType}
    assert names & {"PROPOSE_TRADE", "ACCEPT_TRADE", "DECLINE_TRADE"} == set()
    assert "BANK_TRADE" in names
    assert Action(ActionType.BANK_TRADE, 0, 1)._fields == ("type", "a", "b")


# --- imagine ------------------------------------------------------------------


def test_an_imagined_game_carries_the_published_vectors():
    game = stocked((0, Resource.WOOD, 1))
    game.valuations[0] = vector(ore=0.5)
    child = imagine(game, random.Random(1))
    assert child.valuations[0] == vector(ore=0.5)
    child.valuations[0] = vector(ore=-0.5)
    assert game.valuations[0] == vector(ore=0.5)


def test_an_imagined_game_does_not_carry_the_seated_gates():
    """A hypothetical must not reach the real opponents' private gates."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.gates = tuple(Trader(vector(ore=1.0, wood=-1.0)) for _ in range(4))
    child = imagine(game, random.Random(1))
    assert child.gates is None
    child.phase = Phase.MAIN
    enter_main(child)
    assert child.trades == []


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


# --- the assertion ------------------------------------------------------------


def test_an_event_never_outruns_the_cards_on_the_table():
    """The one engine limit (`trade_event`'s own assertion), exercised on a
    position that trades as hard as the mechanic allows: a gate that always
    says yes and vectors that make every swap advertised."""
    game = stocked((0, Resource.WOOD, 4), (1, Resource.ORE, 4))
    cards = sum(sum(hand) for hand in game._state.hands)
    traders = [
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    ]
    done = run(game, traders)
    assert 0 < len(done) <= cards
