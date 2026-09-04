# SPDX-License-Identifier: GPL-3.0-only
"""The one-event trade mechanic (`hexset.trading`), gate (i) of the trading
design's registration: clearing, the veto, termination, the tie-break, and
what `imagine` carries."""

from __future__ import annotations

import random

import pytest

from hexset.actions import Action, ActionType, legal_actions
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES, Resource
from hexset.game import (
    Phase,
    end_turn,
    enter_main,
    imagine,
    move_robber_to,
    roll_dice,
    run_trade_event,
    start,
)
from hexset.trading import (
    NO_VALUATION,
    Trade,
    _best_clearing,
    _candidates,
    _rank_candidates_loop,
    _rank_candidates_vectorized,
    _VECTORIZE_ABOVE,
    bundle,
    judged,
    judged_many,
    exchange,
    execute_trade,
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


def test_zero_valuation_seats_are_never_enumerated():
    """A seat that has not published (`NO_VALUATION`, all zero) is skipped
    before any hand is walked, in either role: as `me` (nothing enumerated
    at all) and as a counterparty (that seat's hand is never passed to
    `_hand_multisets`) -- the actual expensive part `_candidates` exists to
    avoid doing needlessly."""
    import hexset.trading as trading_mod

    game = stocked(
        (0, Resource.WOOD, 2),
        (0, Resource.ORE, 1),
        (1, Resource.BRICK, 2),
        (2, Resource.SHEEP, 2),  # seat 2 never publishes
    )
    state = game._state
    vectors = [
        vector(wood=-0.1, ore=-0.1, brick=1.0, sheep=0.5),
        vector(wood=1.0, ore=1.0, brick=-0.2, sheep=-0.1),
        NO_VALUATION,
        NO_VALUATION,
    ]

    real = trading_mod._hand_multisets
    walked: list[list[int]] = []

    def spy(hand):
        walked.append(list(hand))
        return real(hand)

    orig = trading_mod._hand_multisets
    trading_mod._hand_multisets = spy
    try:
        # `me` non-zero, one counterparty non-zero, one zero: seat 2's hand
        # (`sheep`) is never walked.
        list(_candidates(state, 0, game.locked, vectors))
        assert walked == [list(state.hands[0]), list(state.hands[1])]

        # `me` itself zero: nothing is walked at all, not even `me`'s own
        # hand for `give_options`.
        walked.clear()
        list(_candidates(state, 2, game.locked, vectors))
        assert walked == []
    finally:
        trading_mod._hand_multisets = orig


def test_trade_event_chooses_the_same_trade_with_a_zero_valuation_bystander():
    """End-to-end: a mixed table (two publishing seats, one that has never
    published) clears the same deal the two publishing seats would have
    cleared alone -- the bystander's presence, and the skip that ignores
    it, changes nothing about which trade executes."""
    game = stocked(
        (0, Resource.WOOD, 1),
        (1, Resource.BRICK, 1),
        (2, Resource.SHEEP, 3),  # never publishes; not a party to anything
    )
    v0 = vector(wood=-1.0, brick=1.0)
    v1 = vector(brick=-1.0, wood=1.0)
    traders = [Trader(v0), Trader(v1), Trader()]  # seat 2: default NO_VALUATION
    done = run(game, traders)
    assert done == [Trade(0, 1, bundle(wood=-1, brick=1))]

    # Seat 2 was never asked anything: its gate has no record of a call,
    # confirming it was skipped as a counterparty candidate, not merely
    # never chosen among several.
    assert traders[2].asked == []


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


def test_every_candidate_is_asked_in_rank_order_until_the_last_one_clears():
    """No budget (`agents/reference/trading-design.md`'s post-data note "the
    gate budget goes away"): the private gates are asked in rank order until
    one clears or candidates run out, however many candidates there are.
    Fourteen candidates advertise here -- every subset of my four resources
    against every size of the counterparty's ore pile that receives more
    cards than it gives (the same hand the deleted gate-budget tests used,
    back when a cap of 8 would have stopped short of all fourteen) -- and
    both seats' gates refuse every one of them except the bundle ranked dead
    last. That bundle must still be reached and clear.
    """
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

    candidates = list(_candidates(game._state, 0, game.locked))
    vectors = [wants_more, wants_less, NO_VALUATION, NO_VALUATION]
    ranked = _rank_candidates_loop(0, vectors, candidates)
    assert len(ranked) == 14
    last_received, last_them = ranked[-1]
    mirror = tuple(-n for n in last_received)

    class ExactGate:
        """Accepts one exact bundle from its own side, refuses every other."""

        def __init__(self, vec, target):
            self.vec = vec
            self.target = target

        def valuation(self, view):
            return self.vec

        def accepts(self, view, received, counterparty):
            return tuple(received) == self.target

    traders = [
        ExactGate(wants_more, last_received),
        ExactGate(wants_less, mirror),
        Trader(),
        Trader(),
    ]
    done = run(game, traders)
    assert done == [Trade(0, last_them, last_received)]


# --- where the event runs -----------------------------------------------------


def _seat_and_publish(game, traders):
    """Seat `traders` as the game's gates and publish each one's vector --
    what a driver has already done by the time a roll or a robber move
    reaches `enter_main`, since publishing rides on a seat's own decision,
    not on the trade event."""
    game.gates = tuple(traders)
    for seat, trader in enumerate(traders):
        game.publish(seat, trader.vec)


def test_entering_main_from_a_roll_arms_the_event_rather_than_running_it():
    """`enter_main` no longer runs the turn's first event itself (the PI
    amendment "publish points and the event trigger") -- it only arms
    `event_pending`; the event fires lazily, the first time anything
    observes or publishes for the current player."""
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
    assert game.event_pending is True
    assert game.trades_made == 0
    legal_actions(game)
    assert game.trades_made == 1
    assert game.event_pending is False


def test_entering_main_from_the_robber_arms_the_event_rather_than_running_it():
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
    assert game.event_pending is True
    assert game.trades_made == 0
    legal_actions(game)
    assert game.trades_made == 1
    assert game.event_pending is False


def test_the_event_fires_on_legal_actions_the_observation_first_path():
    """One of the three trigger points (`Game.event_pending`'s docstring):
    the current player's own `legal_actions(game)` fires the pending event,
    using whatever vector is already standing -- the seat has not
    published this turn yet."""
    game = a_game()
    give(game._state, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)
    game.gates = (
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    )
    for seat, trader in enumerate(game.gates):
        game.valuations[seat] = trader.vec  # a standing vector, not published now
    game.event_pending = True
    assert game.trades_made == 0
    legal_actions(game)
    assert game.trades_made == 1
    assert game.event_pending is False


def test_the_event_fires_on_state_the_observation_first_path():
    """The second trigger point: `game.state(seat)` for the current seat."""
    game = a_game()
    give(game._state, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)
    game.gates = (
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    )
    for seat, trader in enumerate(game.gates):
        game.valuations[seat] = trader.vec
    game.event_pending = True
    game.state(game.current_player)
    assert game.trades_made == 1
    assert game.event_pending is False


def test_a_hidden_false_read_of_the_current_player_s_state_does_not_fire_it():
    """`game.state(seat, hidden=False)` is the true-state access path used
    for reasons unrelated to that seat's own turn -- omniscient search,
    final scoring, a value function hashing a hypothetical position -- and
    must not double as an event trigger: `hexset.bench.aivat.
    chance_outcomes` re-seats a hypothetical child with the real seated
    bots' gates for scoring, and a value function's `child.state(0,
    hidden=False)` firing a live trade event there, using those bots' real
    judgement, measurably diverged the real game from what it would
    otherwise have played (`test_aivat.py`'s exact-replay gate). Only
    `hidden=True` triggers; `legal_actions` and `Game.publish` are
    unaffected and still reach the same seat's pending event."""
    game = a_game()
    give(game._state, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)
    game.gates = (
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    )
    for seat, trader in enumerate(game.gates):
        game.valuations[seat] = trader.vec
    game.event_pending = True
    game.state(game.current_player, hidden=False)
    assert game.trades_made == 0
    assert game.event_pending is True
    # The pending event is still reachable normally afterwards.
    game.state(game.current_player)
    assert game.trades_made == 1
    assert game.event_pending is False


def test_the_event_fires_on_publish_before_any_observation():
    """The third trigger point, and the other order: the current player's
    own `Game.publish`, reached before anything has observed the game this
    turn, fires the event on the vector *just* published -- not the one
    standing from its last turn."""
    game = a_game()
    give(game._state, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)
    game.gates = (
        Trader(),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    )
    # Seat 1 published on its own, earlier turn; seat 0 (the current
    # player) has not yet -- `game.publish` for a seat that is not the
    # current player never fires the trigger, so this alone sets up the
    # counterparty's side without spending the event.
    game.publish(1, vector(wood=1.0, ore=-1.0))
    game.event_pending = True
    assert game.trades_made == 0
    # Nothing has called `legal_actions` or `game.state` for seat 0 yet.
    game.publish(0, vector(ore=1.0, wood=-1.0))
    assert game.trades_made == 1
    assert game.event_pending is False


def test_the_event_fires_on_observation_even_if_nobody_ever_publishes():
    """The idle-human path: a seat that never calls `Game.publish` still
    gets its event, on whatever vector is already standing, the first time
    anything observes the game for it."""
    game = a_game()
    give(game._state, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)
    game.gates = (
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    )
    for seat, trader in enumerate(game.gates):
        game.valuations[seat] = trader.vec
    game.event_pending = True
    assert game.trades_made == 0
    view = game.state(game.current_player)  # e.g. a server rendering the human's own view
    assert game.trades_made == 1
    assert view is not None


def test_the_event_never_fires_twice_for_one_turn():
    game = a_game()
    give(game._state, 0, Resource.WOOD, 4)
    give(game._state, 1, Resource.ORE, 4)
    game.gates = (
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    )
    for seat, trader in enumerate(game.gates):
        game.valuations[seat] = trader.vec
    game.event_pending = True
    legal_actions(game)
    made = game.trades_made
    assert made > 0
    game.state(game.current_player)
    game.publish(0, vector(ore=1.0, wood=-1.0))
    legal_actions(game)
    assert game.trades_made == made


def test_publish_due_is_true_once_per_turn_for_the_current_player_in_main():
    """`publish_due` is keyed off `awaiting_publish` (set by `enter_main`,
    cleared by this seat's own `Game.publish`), not `event_pending` -- the
    regression fix. An observation can consume `event_pending` without ever
    making a still-unpublished current player's own publish moot."""
    game = a_game()
    game.gates = tuple(Trader() for _ in range(4))
    game.phase = Phase.ROLL
    enter_main(game)
    assert game.publish_due(0) is True
    assert game.publish_due(1) is False  # not the current player
    game.phase = Phase.ROLL
    assert game.publish_due(0) is False  # not MAIN
    game.phase = Phase.MAIN
    # An observation fires (and so consumes) `event_pending` -- but
    # `publish_due` must not go false as a side effect of that.
    legal_actions(game)
    assert game.event_pending is False
    assert game.publish_due(0) is True
    game.publish(0, NO_VALUATION)
    assert game.publish_due(0) is False  # published this turn


def test_an_observation_before_the_current_players_publish_still_lets_it_publish():
    """The regression this decoupling fixes, reproduced directly:
    `hexset.server.webplay.state_view` polls the current player through one
    of the three trigger points on *every* viewer's request -- a
    spectator's, or an acting bot's own runner checking whose turn it is
    before it ever calls `decide`. That observation still fires the turn's
    first event early, on whatever vector is already standing for the
    current player (`run_pending_event`'s docstring on why that is not
    changed) -- but it must not stop the current player from publishing a
    moment later, the way `publish_due` used to (the actual regression)."""
    game = a_game()
    give(game._state, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)
    game.gates = (
        Trader(vector(ore=1.0, wood=-1.0)),
        Trader(vector(wood=1.0, ore=-1.0)),
        Trader(),
        Trader(),
    )
    game.valuations[1] = game.gates[1].vec
    # Seat 0's own vector is nothing yet -- e.g. its very first turn ever.
    game.phase = Phase.ROLL
    enter_main(game)
    assert game.event_pending is True
    # An observation of the current player -- exactly what `state_view` and
    # a bot runner's own poll do -- fires the event early, on seat 0's
    # standing (here: all-zero) vector, so nothing clears yet.
    game.state(0)
    assert game.event_pending is False
    assert game.trades_made == 0
    # The regression: this used to make `publish_due` false too, so the
    # publish below never happened. It still works.
    assert game.publish_due(0) is True
    game.publish(0, vector(ore=1.0, wood=-1.0))
    assert game.publish_due(0) is False
    # The published vector reaches the turn's *next* event -- every
    # interleaved one after a MAIN action -- even though the first one
    # already ran on the stale vector above.
    game.trades = []
    run_trade_event(game)
    assert game.trades_made == 1


def test_a_game_with_nobody_seated_simply_does_not_trade():
    game = a_game()
    game.phase = Phase.ROLL
    give(game._state, 0, Resource.WOOD, 1)
    give(game._state, 1, Resource.ORE, 1)
    roll_dice(game, 8)
    legal_actions(game)
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


def test_an_imagined_game_carries_event_pending():
    """`imagine` copies `event_pending` -- a chain of simulated turns arms
    and consumes the flag the same way a real game does."""
    game = a_game()
    game.event_pending = True
    child = imagine(game, random.Random(1))
    assert child.event_pending is True
    child.event_pending = False
    assert game.event_pending is True  # independent copies, not aliased


def test_an_imagined_game_carries_awaiting_publish():
    """`imagine` copies `awaiting_publish` alongside `event_pending`, for the
    same reason: a chain of simulated turns arms and consumes both flags the
    same way a real game does."""
    game = a_game()
    game.awaiting_publish = True
    child = imagine(game, random.Random(1))
    assert child.awaiting_publish is True
    child.awaiting_publish = False
    assert game.awaiting_publish is True  # independent copies, not aliased


def test_a_search_stepping_its_own_copy_never_re_runs_a_live_event():
    """A search's `imagine`d copy carries `gates=None` (the test above), and
    stepping that copy through its own `legal_actions` -- exactly what a
    search does turn after simulated turn -- must not re-run a live event
    with the copy's stand-in gates. The trigger is a true no-op then: it
    does not even consume `event_pending`, so a later real handoff of gates
    to the copy (which never happens today, but the invariant should not
    depend on that) would not have silently missed an event."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.gates = (Trader(vector(ore=1.0, wood=-1.0)), Trader(), Trader(), Trader())
    game.event_pending = True
    child = imagine(game, random.Random(1))
    assert child.gates is None
    assert child.event_pending is True
    legal_actions(child)
    assert child.trades == []
    assert child.event_pending is True


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


# --- execute_trade (the negotiation interface, docs/negotiation-interface.md) -


def _seated(game, traders):
    """Seats `traders` as `game`'s gates, publishing nothing -- `execute_trade`
    tests set `game.valuations` directly, since the interface under test is
    the manual bundle, not the automatic candidate search."""
    game.gates = tuple(traders)


def test_execute_trade_clears_on_coverage_surplus_and_gate():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.valuations[1] = vector(wood=1.0, ore=-1.0)
    _seated(game, [Trader(), Trader(gate=True), Trader(), Trader()])
    received = one_for_one(WOOD, ORE)  # proposer gives wood, receives ore
    trade = execute_trade(game, 0, 1, received)
    assert trade == Trade(0, 1, received)
    assert game._state.hands[0] == [0, 0, 0, 0, 1]
    assert game._state.hands[1] == [1, 0, 0, 0, 0]
    assert game.trades == [trade]
    assert game.trades_made == 1


def test_execute_trade_rejects_whichever_side_cannot_cover_it():
    """The same coverage rule, checked mirrored on each side of the trade."""
    proposer_short = stocked((1, Resource.ORE, 1))
    proposer_short.valuations[1] = vector(wood=1.0, ore=-1.0)
    _seated(proposer_short, [Trader(), Trader(), Trader(), Trader()])
    with pytest.raises(ValueError, match="seat 0 cannot cover"):
        execute_trade(proposer_short, 0, 1, one_for_one(WOOD, ORE))

    counterparty_short = stocked((0, Resource.WOOD, 1))
    counterparty_short.valuations[1] = vector(wood=1.0, ore=-1.0)
    _seated(counterparty_short, [Trader(), Trader(), Trader(), Trader()])
    with pytest.raises(ValueError, match="seat 1 cannot cover"):
        execute_trade(counterparty_short, 0, 1, one_for_one(WOOD, ORE))


def test_execute_trade_rejects_a_non_clearing_counterparty_surplus():
    """The counterparty's public surplus is a hard rule (PI ratification
    decision 4): a bundle its own vector doesn't call an improvement never
    clears, whatever the proposer thinks of it."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    # seat 1 has published nothing -- NO_VALUATION, so no bundle can clear it.
    _seated(game, [Trader(), Trader(), Trader(), Trader()])
    with pytest.raises(ValueError, match="not advertised wanting"):
        execute_trade(game, 0, 1, one_for_one(WOOD, ORE))


def test_execute_trade_rejects_a_declining_gate():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.valuations[1] = vector(wood=1.0, ore=-1.0)
    _seated(game, [Trader(), Trader(gate=False), Trader(), Trader()])
    with pytest.raises(ValueError, match="declined"):
        execute_trade(game, 0, 1, one_for_one(WOOD, ORE))


def test_execute_trade_ignores_the_proposers_own_surplus():
    """Ratification decision 4: submitting is the proposer's own consent, so
    a bundle the proposer's own vector calls bad for itself still clears."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.valuations[0] = vector(wood=1.0, ore=-1.0)  # "giving up wood is bad for me"
    game.valuations[1] = vector(wood=1.0, ore=-1.0)
    _seated(game, [Trader(), Trader(), Trader(), Trader()])
    trade = execute_trade(game, 0, 1, one_for_one(WOOD, ORE))
    assert trade == Trade(0, 1, one_for_one(WOOD, ORE))


def test_execute_trade_never_asks_the_proposers_own_gate():
    """A gate that raises if ever called, seated on the proposer, still lets
    the trade clear -- submitting is consent, so the proposer's own gate is
    never consulted (`docs/negotiation-interface.md` §1)."""

    class Boom:
        def valuation(self, view):
            return NO_VALUATION

        def accepts(self, view, received, counterparty):
            raise AssertionError("the proposer's own gate must never be asked")

    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.valuations[1] = vector(wood=1.0, ore=-1.0)
    _seated(game, [Boom(), Trader(), Trader(), Trader()])
    trade = execute_trade(game, 0, 1, one_for_one(WOOD, ORE))
    assert trade == Trade(0, 1, one_for_one(WOOD, ORE))


def test_execute_trade_rejects_a_seat_that_is_neither_proposer_nor_current_player():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.valuations[1] = vector(wood=1.0, ore=-1.0)
    game.current_player = 2
    _seated(game, [Trader(), Trader(), Trader(), Trader()])
    with pytest.raises(ValueError, match="neither seat"):
        execute_trade(game, 0, 1, one_for_one(WOOD, ORE))


def test_execute_trade_allows_the_current_player_to_answer_another_seats_offer():
    """The turn-timing rule's other half: a seat may also propose during
    another seat's turn, naming only that seat -- so the counterparty being
    `current_player` is just as legal as the proposer being it."""
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.valuations[1] = vector(wood=1.0, ore=-1.0)
    game.current_player = 1
    _seated(game, [Trader(), Trader(), Trader(), Trader()])
    trade = execute_trade(game, 0, 1, one_for_one(WOOD, ORE))
    assert trade == Trade(0, 1, one_for_one(WOOD, ORE))


def test_execute_trade_requires_main_phase():
    game = stocked((0, Resource.WOOD, 1), (1, Resource.ORE, 1))
    game.valuations[1] = vector(wood=1.0, ore=-1.0)
    game.phase = Phase.ROLL
    _seated(game, [Trader(), Trader(), Trader(), Trader()])
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
    v = vector(wood=1.0, brick=1.0, ore=-1.0)
    game.valuations[1] = v
    _seated(game, [Trader(), Trader(), Trader(), Trader()])
    received = [0, 0, 0, 0, 0]
    received[ORE] = 3
    received[WOOD] = -2
    received[BRICK] = -1
    trade = execute_trade(game, 0, 1, tuple(received))
    assert trade.received == tuple(received)
    assert game._state.hands[0][ORE] == 3
    assert game._state.hands[1][WOOD] == 2 and game._state.hands[1][BRICK] == 1


# --- Game.pending (a snapshot of the last event) -------------------------------


def test_pending_is_cleared_at_the_start_of_every_trade_event():
    game = stocked((0, Resource.WOOD, 1))
    game.pending.append(Trade(1, 0, one_for_one(WOOD, ORE)))
    game.gates = tuple(Trader() for _ in range(4))
    trade_event(game, lambda seat, view, received, other: True)
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


# --- Batched gates (`agents/reference/trading-design.md`'s post-data note, ---
# --- "the collector cost gate fails at 2.9-3.6x") ----------------------------


class _AcceptsOnly:
    """A trader with `accepts` and no `accepts_many` -- exercises
    `judged_many`'s structural default (`Bot.accepts_many`'s documented
    default is "loop over `accepts`", implemented here rather than by
    inheritance, exactly as `judged` already implements `accepts`'s own
    default)."""

    def __init__(self):
        self.asked: list[tuple[tuple[int, ...], int]] = []

    def accepts(self, view, received, counterparty):
        self.asked.append((received, counterparty))
        return received[WOOD] > 0


def test_judged_many_default_loops_over_accepts_in_order():
    trader = _AcceptsOnly()
    view = a_game().state(0)
    received = [
        one_for_one(ORE, WOOD),  # gives wood -> True
        one_for_one(WOOD, ORE),  # takes wood -> False
        one_for_one(WHEAT, SHEEP),  # no wood either way -> False
    ]
    counterparties = [1, 2, 3]

    many = judged_many(trader, view, received, counterparties)
    asked_via_many = list(trader.asked)

    trader.asked.clear()
    looped = [judged(trader, view, r, c) for r, c in zip(received, counterparties)]

    assert many == looped == [True, False, False]
    assert asked_via_many == list(zip(received, counterparties))


class BatchTrader:
    """A seat that answers a fixed, deterministic per-candidate rule --
    accepting some candidates and refusing others, so the winner is not
    simply "whatever ranks first" -- and only ever exposes `accepts_many`.

    `accepts` raises if called at all: `_best_clearing`, once `game.gates`
    is seated, must never fall back to asking this trader one candidate at
    a time.
    """

    def __init__(self, seat: int, vec: tuple[float, ...]):
        self.seat = seat
        self.vec = vec
        self.batch_calls = 0
        self.asked: list[tuple[tuple[int, ...], int]] = []

    def valuation(self, view):
        return self.vec

    def accepts(self, view, received, counterparty):  # pragma: no cover
        raise AssertionError(
            "accepts was called; the batched gate should have used accepts_many"
        )

    def accepts_many(self, view, received, counterparties):
        self.batch_calls += 1
        self.asked.extend(zip(received, counterparties))
        return [_verdict(self.seat, r, c) for r, c in zip(received, counterparties)]


def _verdict(seat: int, received: tuple[int, ...], counterparty: int) -> bool:
    """A fixed, reproducible per-candidate rule with no Python-hash
    dependence: some candidates pass, some don't, for every seat."""
    return (sum(abs(n) for n in received) + counterparty + seat) % 3 != 0


def _reference_best_clearing(game, me: int, vectors):
    """The pre-batching `_best_clearing`, verbatim: candidates ranked once,
    then `_verdict` asked one candidate at a time, stopping at the first
    the acting seat and the counterparty both accept. The ground truth this
    module's batched `_best_clearing` must keep matching."""
    candidates = list(_candidates(game._state, me, game.locked))
    if not candidates:
        return None
    if len(candidates) < _VECTORIZE_ABOVE:
        ranked = _rank_candidates_loop(me, vectors, candidates)
    else:
        ranked = _rank_candidates_vectorized(me, vectors, candidates)
    for received, them in ranked:
        mirror = tuple(-n for n in received)
        if _verdict(me, received, them) and _verdict(them, mirror, me):
            return them, received
    return None


def _random_trade_ready_game(seed: int, players: int = 4):
    rng = random.Random(seed)
    game = a_game(players=players)
    for seat in range(players):
        for resource in range(NUM_RESOURCES):
            count = rng.choice([0, 0, 1, 1, 2, 3])
            if count:
                give(game._state, seat, Resource(resource), count)
    vectors = [tuple(rng.uniform(-1.0, 1.0) for _ in range(NUM_RESOURCES)) for _ in range(players)]
    return game, vectors


def test_best_clearing_asks_each_seats_gate_at_most_once():
    """The call-count bound: one batched ask for the acting seat, plus at
    most one more per counterparty whose candidates it accepted -- never one
    ask per candidate, however many candidates are on the table."""
    game, vectors = _random_trade_ready_game(seed=7)
    traders = [BatchTrader(seat, vectors[seat]) for seat in range(game.num_players)]
    game.gates = tuple(traders)
    for seat, trader in enumerate(traders):
        game.publish(seat, trader.valuation(game.state(seat)))

    def view(seat):
        return game.state(seat)

    candidates = list(_candidates(game._state, game.current_player, game.locked))
    assert len(candidates) > 0, "the seeded position should offer at least one candidate"

    result = _best_clearing(
        game,
        game.current_player,
        game.valuations,
        lambda seat, v, received, other: (_ for _ in ()).throw(
            AssertionError("the single-ask gate must not be used when game.gates is seated")
        ),
        view,
    )

    me = traders[game.current_player]
    assert me.batch_calls == 1
    counterparties_asked = {t.seat for t in traders if t is not me and t.batch_calls}
    for t in traders:
        assert t.batch_calls <= 1
    total_calls = sum(t.batch_calls for t in traders)
    assert total_calls <= 1 + max(0, game.num_players - 1)
    assert counterparties_asked <= {t.seat for t in traders if t is not me}
    # The reference (pre-batching) algorithm must agree on the winner.
    assert result == _reference_best_clearing(game, game.current_player, game.valuations)


def test_trade_event_executes_the_same_trades_as_the_sequential_reference():
    """20 seeded random trade-ready positions: `trade_event`, run against
    batched-only traders (`BatchTrader`, `accepts` disabled), must execute
    exactly the sequence of trades the pre-batching, one-candidate-at-a-time
    algorithm (`_reference_best_clearing`) would have found on the same
    starting position -- the batching changes how the booleans are
    gathered, never which trade wins."""
    for seed in range(20):
        game, vectors = _random_trade_ready_game(seed)
        traders = [BatchTrader(seat, vectors[seat]) for seat in range(game.num_players)]
        game.gates = tuple(traders)
        for seat, trader in enumerate(traders):
            game.publish(seat, trader.valuation(game.state(seat)))

        # The reference walk, over an independent copy of the same starting
        # state, using the same deterministic verdict but the old
        # sequential, one-candidate-at-a-time algorithm.
        ref_game, ref_vectors = _random_trade_ready_game(seed)
        expected: list[Trade] = []
        me = ref_game.current_player
        while True:
            best = _reference_best_clearing(ref_game, me, ref_vectors)
            if best is None:
                break
            them, received = best
            exchange(ref_game._state, me, them, received)
            expected.append(Trade(me, them, received))

        def gate(seat, view, received, other):  # pragma: no cover
            raise AssertionError("single-ask gate must not be used when game.gates is seated")

        done = trade_event(game, gate)

        assert done == expected, f"seed {seed}: batched execution diverged from the reference"
        assert game.trades == expected
