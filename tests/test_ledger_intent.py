# SPDX-License-Identifier: GPL-3.0-only
"""What a seat has shown it wants and will give up (`PublicLedger.show`), how
the cards clear it, and the offers that read it (`TradeParams.fit_offers`)."""

from __future__ import annotations

import random

import pytest

from hexset.actions import apply
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES, Resource
from traders import ScarcityTrader
from hexset.game import Phase, is_over, start, to_move
from hexset.ledger import PublicLedger
from hexset.seat import Seat
from hexset.trading import (
    RESPONSE_COUNTER,
    RESPONSE_PASS,
    Offer,
    Response,
    bundle,
    choose_initial,
    choose_remainder,
    fit,
    resolve_offer,
    show,
)
from helpers import deal

WOOD, BRICK, SHEEP, WHEAT, ORE = (int(r) for r in Resource)


def a_game(players: int = 4):
    rng = random.Random(0)
    game = start(random_base_board(rng), players, rng)
    game.phase = Phase.MAIN
    game.current_player = 0
    return game


def test_an_ask_is_a_want_and_a_give_is_a_waste():
    ledger = PublicLedger.new(2)
    ledger.show(1, bundle(brick=1, sheep=-2), holds=False)
    assert ledger.seats[1].want == [0, 1, 0, 0, 0]
    assert ledger.seats[1].waste == [0, 0, 2, 0, 0]
    # The same offer heard twice changes nothing; a larger one keeps the most.
    ledger.show(1, bundle(brick=1, sheep=-2), holds=False)
    ledger.show(1, bundle(brick=2, sheep=-1), holds=False)
    assert ledger.seats[1].want == [0, 2, 0, 0, 0]
    assert ledger.seats[1].waste == [0, 0, 2, 0, 0]


def test_asking_for_a_resource_withdraws_it_as_waste_and_back():
    ledger = PublicLedger.new(2)
    ledger.show(1, bundle(brick=1, sheep=-1), holds=False)
    ledger.show(1, bundle(sheep=1, ore=-1), holds=False)
    assert ledger.seats[1].want[SHEEP] == 1 and ledger.seats[1].waste[SHEEP] == 0
    ledger.show(1, bundle(wood=1, sheep=-1), holds=False)
    assert ledger.seats[1].want[SHEEP] == 0 and ledger.seats[1].waste[SHEEP] == 1


def test_receiving_meets_a_want_and_parting_with_a_card_uses_up_its_waste():
    ledger = PublicLedger.new(2)
    ledger.receive(1, SHEEP, 3)
    ledger.show(1, bundle(brick=2, sheep=-3))
    ledger.receive(1, BRICK, 1)
    assert ledger.seats[1].want[BRICK] == 1
    ledger.receive(1, BRICK, 1)
    assert ledger.seats[1].want[BRICK] == 0
    ledger.spend(1, SHEEP, 2)
    assert ledger.seats[1].waste[SHEEP] == 1


def test_a_steal_moves_neither_want_nor_waste():
    ledger = PublicLedger.new(2)
    ledger.receive(1, SHEEP, 2)
    ledger.show(1, bundle(brick=1, sheep=-2))
    ledger.steal(thief=0, victim=1)
    assert ledger.seats[1].want == [0, 1, 0, 0, 0]
    assert ledger.seats[1].waste == [0, 0, 2, 0, 0]
    assert ledger.seats[0].want == [0] * NUM_RESOURCES


def test_an_offer_the_referee_checked_certifies_the_cards_it_gives():
    ledger = PublicLedger.new(2)
    ledger.gain_unknown(1, 3)
    ledger.show(1, bundle(brick=1, sheep=-2))
    assert ledger.seats[1].known == [0, 0, 2, 0, 0]
    assert ledger.seats[1].unknown == 1
    # Unchecked, nothing is certified.
    unchecked = PublicLedger.new(2)
    unchecked.gain_unknown(1, 3)
    unchecked.show(1, bundle(brick=1, sheep=-2), holds=False)
    assert unchecked.seats[1].known == [0] * NUM_RESOURCES
    assert unchecked.seats[1].unknown == 3


def test_copy_carries_want_and_waste_independently():
    ledger = PublicLedger.new(2)
    ledger.show(1, bundle(brick=1, sheep=-1), holds=False)
    copied = ledger.copy()
    copied.show(1, bundle(ore=1, wood=-1), holds=False)
    assert ledger.seats[1].want == [0, 1, 0, 0, 0]
    assert copied.seats[1].want == [0, 1, 0, 0, 1]


def test_show_certifies_only_what_the_true_hand_covers():
    game = a_game()
    deal(game, 1, Resource.SHEEP, 1)
    show(game, 1, bundle(brick=1, sheep=-2))
    # Seat 1 holds one sheep: an offer of two is not certified, but it is
    # still what the seat showed.
    assert game.ledger.seats[1].known[SHEEP] == 1
    assert game.ledger.seats[1].waste[SHEEP] == 2


class _Fixed:
    """A gate that answers every offer with one fixed response."""

    trade_floor = 0.0

    def __init__(self, response):
        self.response = response

    def gains_many(self, view, received, counterparties):
        return [-1.0] * len(received)

    def respond(self, view, offer):
        return self.response(view.perspective)


def test_a_round_puts_the_offer_and_its_answers_on_the_ledger():
    game = a_game()
    deal(game, 0, Resource.SHEEP, 2)
    deal(game, 2, Resource.ORE, 1)
    passing = _Fixed(lambda seat: Response(seat, RESPONSE_PASS))
    # Seat 2 counters "your sheep for my ore"; signed towards the actor.
    countering = _Fixed(lambda seat: Response(seat, RESPONSE_COUNTER, bundle(ore=1, sheep=-1)))
    actor = _Fixed(lambda seat: Response(seat, RESPONSE_PASS))
    gates = [actor, passing, countering, passing]
    resolve_offer(game, gates, Offer(0, bundle(brick=1, sheep=-2)))
    assert game.ledger.seats[0].want[BRICK] == 1
    assert game.ledger.seats[0].waste[SHEEP] == 2
    assert game.ledger.seats[2].want[SHEEP] == 1
    assert game.ledger.seats[2].waste[ORE] == 1
    assert game.ledger.seats[1].want == [0] * NUM_RESOURCES


def test_fit_scores_giving_a_want_and_asking_for_waste():
    game = a_game()
    game.ledger.show(1, bundle(brick=1, sheep=-2), holds=False)
    view = game.state(0)
    # Signed towards seat 0: seat 1 gives the positive counts.
    assert fit(view, 1, bundle(sheep=1, brick=-1)) == 2
    assert fit(view, 1, bundle(sheep=2, brick=-1)) == 2
    assert fit(view, 1, bundle(sheep=3, brick=-1)) == 1   # more than it offered
    assert fit(view, 1, bundle(ore=1, brick=-1)) == 1
    assert fit(view, 1, bundle(sheep=1, wood=-1)) == 1
    assert fit(view, 1, bundle(ore=1, wood=-1)) == 0
    assert fit(view, 2, bundle(sheep=1, brick=-1)) == 0


def _rows():
    # (partner, bundle signed towards the actor, raw gain, estimate)
    return [
        (1, bundle(ore=1, wood=-1), 0.05, 0.01),
        (2, bundle(sheep=1, brick=-1), 0.02, 0.01),
    ]


def test_the_fitting_offer_goes_first():
    rows = _rows()
    plain = choose_initial(rows, cutoff=0.0, draw_key=(0, 0, 0, 0), max_cards=2)
    assert plain.bundle == bundle(ore=1, wood=-1)
    fits = {(2, bundle(sheep=1, brick=-1)): 2}
    chosen = choose_initial(rows, cutoff=0.0, draw_key=(0, 0, 0, 0), max_cards=2,
                            fit=lambda p, b: fits.get((p, tuple(b)), 0))
    assert chosen.bundle == bundle(sheep=1, brick=-1)
    assert chosen.partner == 2


def test_nothing_fitting_is_the_choice_without_fit():
    rows = _rows()
    plain = choose_initial(rows, cutoff=0.0, draw_key=(0, 0, 0, 0), max_cards=2)
    zero = choose_initial(rows, cutoff=0.0, draw_key=(0, 0, 0, 0), max_cards=2,
                          fit=lambda p, b: 0)
    assert zero == plain


def test_a_fitting_offer_under_the_cutoff_leaves_the_choice_without_it():
    rows = _rows()
    fits = {(2, bundle(sheep=1, brick=-1)): 2}
    chosen = choose_initial(rows, cutoff=0.03, draw_key=(0, 0, 0, 0), max_cards=2,
                            fit=lambda p, b: fits.get((p, tuple(b)), 0))
    assert chosen.bundle == bundle(ore=1, wood=-1)


def test_a_remainder_goes_to_the_partner_that_fits():
    remainder = bundle(sheep=1, brick=-1)
    rows = [(1, remainder, 0.02, 0.01), (2, remainder, 0.02, 0.01)]
    assert choose_remainder(rows, remainder, max_cards=2)[0] == 1
    chosen = choose_remainder(rows, remainder, max_cards=2,
                              fit=lambda p, b: 1 if p == 2 else 0)
    assert chosen[0] == 2


def test_a_hosted_seat_notes_what_the_table_shows():
    rng = random.Random(0)
    board = random_base_board(rng)
    seat = Seat.sit(board, 4, 0, ScarcityTrader())
    seat.shown(2, bundle(wood=1))  # "need wood"
    assert seat.game.ledger.seats[2].want[WOOD] == 1
    assert seat.game.ledger.seats[2].waste == [0] * NUM_RESOURCES


def _assert_invariant(game) -> None:
    for s, row in enumerate(game.ledger.seats):
        hand = game._state.hands[s]
        assert row.total() == sum(hand)
        assert all(k <= h for k, h in zip(row.known, hand))


@pytest.mark.parametrize("seed", [3])
def test_certified_offers_never_overclaim_over_a_traded_game(seed):
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    bots = [ScarcityTrader(random.Random(seed + s)) for s in range(4)]
    game.gates = tuple(bots)
    shown = False
    for _ in range(3000):
        if is_over(game):
            break
        seat = to_move(game)
        apply(game, bots[seat].choose(game))
        _assert_invariant(game)
        shown = shown or any(any(r.want) or any(r.waste) for r in game.ledger.seats)
    assert shown
