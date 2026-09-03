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
    GATE_BUDGET,
    NO_VALUATION,
    Trade,
    _candidates,
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
    """No bundle-size cap (owner review, 2026-09-03), so a single bundle
    with the counterparty who wants it can move a whole three-card holding
    at once -- what makes the loop necessary now is a *second*, unrelated
    deal with a different counterparty, not a size limit. Wood clears
    against seat 1 in one bundle; brick, still held, only then clears
    against seat 2."""
    game = stocked(
        (0, Resource.WOOD, 3), (0, Resource.BRICK, 3), (1, Resource.ORE, 3), (2, Resource.WHEAT, 3)
    )
    traders = [
        Trader(vector(ore=1.0, wood=-1.0, wheat=1.0, brick=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0, brick=-0.5)),
        Trader(vector(brick=1.0, wheat=-1.0, wood=-0.5)),
        Trader(),
    ]
    done = run(game, traders)
    assert done == [
        Trade(0, 1, bundle(wood=-3, ore=3)),
        Trade(0, 2, bundle(brick=-3, wheat=3)),
    ]
    assert game._state.hands[0] == [0, 0, 0, 3, 3]
    assert game._state.hands[1] == [3, 0, 0, 0, 0]
    assert game._state.hands[2] == [0, 3, 0, 0, 0]


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
    """Three clearing candidates against the same counterparty: wood alone,
    brick alone, and the bundle of both. The bundle strictly dominates
    either single-resource swap here (giving up more of something both
    sides dislike-in-my-hand/want-in-theirs only adds surplus), so it wins
    -- a 2-for-1 that no sequence of 1-for-1 steps could reach, since a
    1-for-1 step re-asks both gates on its own."""
    game = stocked((0, Resource.WOOD, 1), (0, Resource.BRICK, 1), (1, Resource.ORE, 1))
    traders = [
        Trader(vector(ore=1.0, wood=-0.1, brick=-0.9)),
        Trader(vector(brick=0.9, wood=0.1, ore=-1.0)),
        Trader(),
        Trader(),
    ]
    done = run(game, traders)
    # wood+brick->ore: min(1.0+0.1+0.9, 0.1+0.9+1.0) = 2.0
    # brick->ore alone: min(1.0+0.9, 0.9+1.0) = 1.9
    # wood->ore alone: min(1.0+0.1, 0.1+1.0) = 1.1
    assert done[0].received == bundle(wood=-1, brick=-1, ore=1)
    # And the same position clears nothing for either single-card swap
    # between these two seats once the bundle has already gone through.
    assert len(done) == 1


def test_a_tie_on_the_smaller_surplus_goes_to_the_actor_s_bigger_own_surplus():
    """Owner review (2026-09-03), "the tie-break": real-valued surpluses
    essentially never tie, but when they do the rulebook gives the current
    player the choice among equally fair deals, so it takes the better one
    for itself.

    Giving brick alone and giving wood+brick together both leave the
    counterparty (who does not value wood either way) at exactly the same
    surplus, 1.1 -- engineered so `min(mine, theirs)` ties at 1.1 for both
    (brick-alone: mine 1.9, theirs 1.0, min 1.0 is *not* one of the tied
    pair; wood-alone: mine 1.1, theirs 1.1, min 1.1; wood+brick: mine 2.0,
    theirs 1.1, min 1.1) -- while the actor's own surplus does not: 2.0 for
    the bundle that also gives away the unwanted wood, 1.1 for wood alone.
    The bigger-owned-surplus bundle wins."""
    game = stocked((0, Resource.WOOD, 1), (0, Resource.BRICK, 1), (1, Resource.ORE, 1))
    v_me = vector(ore=1.0, wood=-0.1, brick=-0.9)
    v_them = vector(wood=0.1, ore=-1.0)  # brick left at 0: indifferent to it
    traders = [Trader(v_me), Trader(v_them), Trader(), Trader()]
    done = run(game, traders)
    assert done[0].received == bundle(wood=-1, brick=-1, ore=1)


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


# --- bundles: the owner's 2026-09-03 correction --------------------------------
#
# `agents/reference/trading-design.md`'s post-data note found the shipped
# `_candidates` wrong: it enumerated only coverable one-for-one swaps and
# claimed a 2-for-1 "arises as a sequence" of those, which is false because
# each step in such a sequence would have to clear both gates entirely on
# its own. These tests exercise the fix directly: real multi-card bundles,
# the disjoint-sides rule, coverability from the true hands, ranking by the
# smaller surplus rather than the total, and the gate budget the owner's
# correction adds as a cost bound.


def test_a_two_for_one_clears_when_no_one_for_one_between_the_same_seats_does():
    """The gate is what makes this the load-bearing case: `HoldsOutForTwo`
    refuses anything under three cards, so neither of the two candidate
    one-for-one swaps between these seats can ever clear on its own -- only
    the bundle that gives both of my resources for the counterparty's one
    card reaches three cards and clears."""
    game = stocked((0, Resource.WOOD, 1), (0, Resource.BRICK, 1), (1, Resource.ORE, 1))

    class HoldsOutForTwo(Trader):
        def accepts(self, view, received, counterparty):
            self.asked.append((tuple(received), counterparty))
            return sum(abs(n) for n in received) >= 3

    v_me = vector(ore=1.0, wood=-1.0, brick=-1.0)
    v_them = vector(wood=1.0, brick=1.0, ore=-1.0)
    traders = [HoldsOutForTwo(v_me), HoldsOutForTwo(v_them), Trader(), Trader()]

    # Neither one-for-one candidate this position could offer would have
    # cleared on its own -- confirmed directly against the same gate.
    view0, view1 = game.state(0), game.state(1)
    assert not traders[0].accepts(view0, one_for_one(WOOD, ORE), 1)
    assert not traders[0].accepts(view0, one_for_one(BRICK, ORE), 1)
    assert not traders[1].accepts(view1, one_for_one(ORE, WOOD), 0)
    traders[0].asked.clear()
    traders[1].asked.clear()

    done = run(game, traders)
    assert done == [Trade(0, 1, bundle(wood=-1, brick=-1, ore=1))]


def test_disjoint_sides_forbids_a_resource_on_both_sides_of_one_bundle():
    """Both seats hold both resources on the table, so every combination
    that repeats a resource across the give and receive sides -- "give wood,
    receive wood back" and its mirror -- is a candidate only the
    disjoint-resource-sets rule rules out; what is left is exactly the two
    genuine one-for-one swaps."""
    game = stocked(
        (0, Resource.WOOD, 1),
        (0, Resource.ORE, 1),
        (1, Resource.WOOD, 1),
        (1, Resource.ORE, 1),
    )
    candidates = _candidates(game._state, 0, game.locked)
    assert set(candidates) == {
        (1, bundle(wood=-1, ore=1)),
        (1, bundle(ore=-1, wood=1)),
    }


def test_candidates_are_coverable_from_the_true_hands():
    """No candidate ever asks `me` to give more than its hand holds, or the
    counterparty to give more than theirs -- the engine is the referee, so
    nothing downstream has to check this again."""
    game = stocked(
        (0, Resource.WOOD, 2),
        (0, Resource.BRICK, 1),
        (1, Resource.ORE, 2),
        (1, Resource.SHEEP, 3),
    )
    state = game._state
    candidates = list(_candidates(state, 0, game.locked))  # a generator now
    assert candidates  # the position actually has something to check
    for them, received in candidates:
        for r in range(NUM_RESOURCES):
            if received[r] < 0:
                assert -received[r] <= state.hands[0][r]
            elif received[r] > 0:
                assert received[r] <= state.hands[them][r]


def test_ranking_prefers_the_fairer_bundle_over_a_bigger_lopsided_one():
    """Two candidates against two different counterparties (so they cannot
    merge into one bigger bundle): one lopsided (mine 1.01, theirs 0.02 --
    total 1.03, but the counterparty barely wants it), one fair (mine 0.3,
    theirs 0.1 -- total 0.4, and its smaller half, 0.1, still beats the
    lopsided candidate's smaller half, 0.02). Ranking by the smaller surplus
    picks the fair one despite its far smaller total. Every vector component
    is given explicitly (no bundle-size cap any more, so an unlisted "0"
    component is a free resource either side would happily add for nothing
    -- that would let a bigger combined bundle win on the actor's-own-surplus
    key instead, which is not what this test is about)."""
    game = stocked(
        (0, Resource.WOOD, 1),
        (0, Resource.ORE, 1),
        (1, Resource.BRICK, 1),
        (2, Resource.SHEEP, 1),
    )
    v_me = vector(wood=-0.01, ore=-0.1, brick=1.0, sheep=0.2, wheat=0.0)
    v_lopsided_partner = vector(wood=0.01, brick=-0.01, ore=-0.02, sheep=-0.02, wheat=0.0)
    v_fair_partner = vector(ore=0.05, sheep=-0.05, wood=-0.02, brick=-0.02, wheat=0.0)
    traders = [
        Trader(v_me),
        Trader(v_lopsided_partner),
        Trader(v_fair_partner),
        Trader(),
    ]
    done = run(game, traders)
    assert done[0].received == bundle(ore=-1, sheep=1)


def test_the_gate_budget_binds_and_is_counted():
    """Fourteen candidates advertise (every subset of my four resources
    against every size of the counterparty's ore pile that receives more
    cards than it gives); both gates refuse everything. `GATE_BUDGET` (8)
    candidate pairs get asked, no more -- without the cap this hand would
    price all fourteen against two position evaluations every time it
    looked for a deal."""
    game = stocked(
        (0, Resource.WOOD, 1),
        (0, Resource.BRICK, 1),
        (0, Resource.SHEEP, 1),
        (0, Resource.WHEAT, 1),
        (1, Resource.ORE, 3),
    )
    # `theirs = dot(v_them, -b)`, so an identical vector on both seats would
    # make `theirs == -mine` always -- never both positive. `wants_more`
    # (receiving is worth more than giving, uniformly) on one seat and
    # `wants_less` (the mirror: giving is worth more than receiving) on the
    # other both read as "receiving beats giving" from each seat's own side,
    # so `mine == theirs == received_cards - given_cards`.
    wants_more = vector(wood=1.0, brick=1.0, sheep=1.0, wheat=1.0, ore=1.0)
    wants_less = vector(wood=-1.0, brick=-1.0, sheep=-1.0, wheat=-1.0, ore=-1.0)
    traders = [
        Trader(wants_more, gate=False),
        Trader(wants_less, gate=False),
        Trader(),
        Trader(),
    ]
    assert GATE_BUDGET == 8
    assert run(game, traders) == []
    assert game.budget_binds == 1


def test_the_budget_does_not_bind_when_there_are_few_enough_candidates():
    """The ordinary one-for-one position from `test_a_deal_both_sides_want_clears`
    never comes close to the budget, so it must never be counted as binding."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    traders = [
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    ]
    run(game, traders)
    assert game.budget_binds == 0


def test_the_bundle_engine_is_deterministic():
    """The same position, replayed from scratch, clears the same trades in
    the same order -- nothing here depends on dict, set, or hash-order
    iteration."""

    def once():
        game = stocked(
            (0, Resource.WOOD, 3), (0, Resource.BRICK, 2), (1, Resource.ORE, 4)
        )
        traders = [
            Trader(vector(ore=1.0, wood=-1.0, brick=-0.5)),
            Trader(vector(wood=1.0, brick=0.5, ore=-1.0)),
            Trader(),
            Trader(),
        ]
        return run(game, traders)

    first = once()
    second = once()
    assert first == second
    assert len(first) > 0


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
    game.budget_binds = 3
    end_turn(game)
    assert game.trades_made == 0
    assert game.trades == []
    assert game.budget_binds == 0


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
    game.budget_binds = 2
    child = imagine(game, random.Random(1))
    assert child.max_trades == 0
    assert child.trades_made == 1
    assert child.budget_binds == 2
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
