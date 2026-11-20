# SPDX-License-Identifier: GPL-3.0-only
"""The untyped resource cards a sampled world deals, weighed by what each seat
has shown the table (`HoldModel.shown`, `View.shown_weights`): exactly in
proportion to the weights where a reading declares them, and draw for draw
the unweighted deal where none applies. The cards a seat offers are certified
and held in every world."""

from __future__ import annotations

import dataclasses
import random

import pytest

from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.clients._modelmeta import hold_config
from hexset.game import Phase, start
from hexset.holding import HoldModel
from hexset.ledger import SeatLedger
from hexset.state import BANK_PER_RESOURCE
from hexset.trading import show
from hexset.view import View

WOOD, BRICK, SHEEP, WHEAT, ORE = (int(r) for r in Resource)
#: Illustrative numbers: the engine carries none of its own, a checkpoint
#: declares them.
PLAIN = HoldModel(shape=1.0, scale=(0.5, 1.0, 0.6, 0.6, 0.7))
WEIGHED = dataclasses.replace(PLAIN, asked=0.25, offered=3.0)


def untyped_game():
    """Three seats: seat 0 sees seat 1 hold two untyped cards and seat 2 two
    beside a known wheat; the four untyped cards are a wood, a brick, a sheep
    and an ore, the whole residual pool."""
    rng = random.Random(0)
    game = start(random_base_board(rng), 3, rng)
    game.phase = Phase.MAIN
    state = game._state
    state.hands = [[1, 0, 0, 1, 0], [1, 1, 0, 0, 0], [0, 0, 1, 1, 1]]
    state.bank = [BANK_PER_RESOURCE - sum(h[r] for h in state.hands) for r in range(5)]
    game.ledger.seats = [
        SeatLedger(known=state.hands[0][:]),
        SeatLedger(known=[0, 0, 0, 0, 0], unknown=2),
        SeatLedger(known=[0, 0, 0, 1, 0], unknown=2),
    ]
    return game


def worlds(view, hold, n, seed=1):
    rng = random.Random(seed)
    return [view.sample(rng, hold) for _ in range(n)]


def test_a_hold_models_shown_weights_round_trip_and_stay_out_of_metadata_unset():
    assert hold_config(WEIGHED.meta()) == WEIGHED
    assert set(PLAIN.meta()) == {"hold_shape", "hold_scale"}
    assert hold_config(PLAIN.meta()) == PLAIN
    assert not PLAIN.reads_shown() and WEIGHED.reads_shown()
    assert WEIGHED.shown(WOOD, 1, 0) == 0.25
    assert WEIGHED.shown(WOOD, 0, 2) == 3.0
    assert WEIGHED.shown(WOOD, 0, 0) == 1.0


@pytest.mark.parametrize("extra", [
    {"hold_asked": "0.5"},
    {"hold_offered": "2"},
    {"hold_asked": "0", "hold_offered": "2"},
    {"hold_asked": "nan", "hold_offered": "2"},
    {"hold_asked": "x", "hold_offered": "2"},
])
def test_a_declared_shown_reading_that_cannot_be_read_fails_the_load(extra):
    with pytest.raises(ValueError):
        hold_config({**PLAIN.meta(), **extra})


def test_shown_weights_without_the_hold_model_fail_the_load():
    with pytest.raises(ValueError):
        hold_config({"hold_asked": "0.5", "hold_offered": "2"})


def test_an_asked_resource_is_dealt_exactly_in_proportion_to_its_weight():
    game = untyped_game()
    game.ledger.show(1, [1, 0, 0, 0, 0])            # seat 1 asks for wood
    view = View.from_game(game, 0)
    assert view.shown_weights(WEIGHED) == {1: [0.25, 1.0, 1.0, 1.0, 1.0]}
    # Seat 1 is dealt two of the four pool cards, each pair alike but for
    # the weight: the three pairs with wood weigh 0.25 against three without.
    exact = 3 * 0.25 / (3 * 0.25 + 3)
    n = 6000
    share = sum(w.hands[1][WOOD] for w in worlds(view, WEIGHED, n)) / n
    assert abs(share - exact) < 0.025
    plain = sum(w.hands[1][WOOD] for w in worlds(view, PLAIN, n)) / n
    assert abs(plain - 0.5) < 0.025
    assert abs(view.p_holds(1, [1, 0, 0, 0, 0], draws=n, hold=WEIGHED) - exact) < 0.025


def test_an_offered_resource_is_dealt_more_often_and_bounds_hold_every_world():
    game = untyped_game()
    game.ledger.show(2, [0, 0, -1, 0, 0], holds=False)   # seat 2 offered a sheep away, uncertified
    view = View.from_game(game, 0)
    # Seat 2 is dealt two of the four pool cards beside its known wheat: the
    # three pairs with the sheep weigh 3 against three without.
    exact = 3 * 3.0 / (3 * 3.0 + 3)
    n = 6000
    dealt = worlds(view, WEIGHED, n)
    share = sum(w.hands[2][SHEEP] for w in dealt) / n
    assert abs(share - exact) < 0.025
    for w in dealt:
        assert w.hands[2][WHEAT] >= 1
        assert sum(w.hands[2]) == 3 and sum(w.hands[1]) == 2
        assert all(sum(h[r] for h in w.hands) + w.bank[r] == BANK_PER_RESOURCE for r in range(5))


def test_the_cards_a_seat_offers_are_held_in_every_world():
    game = untyped_game()
    state = game._state
    state.hands[1] = [2, 1, 0, 0, 0]
    state.hands[2] = [0, 0, 1, 1, 1]
    state.bank = [BANK_PER_RESOURCE - sum(h[r] for h in state.hands) for r in range(5)]
    game.ledger.seats[1] = SeatLedger(known=[0, 0, 0, 0, 0], unknown=3)
    show(game, 1, [0, 0, 0, 1, 0])          # wheat for two wood and a brick: no certification yet
    show(game, 1, [-2, -1, 0, 1, 0])
    view = View.from_game(game, 0)
    assert view.known[1] == [2, 1, 0, 0, 0] and view.unknown[1] == 0
    for hold in (None, PLAIN, WEIGHED):
        for w in worlds(view, hold, 200):
            assert w.hands[1] == [2, 1, 0, 0, 0]


def test_without_a_weight_that_applies_every_deal_is_the_unweighted_one():
    game = untyped_game()
    view = View.from_game(game, 0)
    # A reading with weights, but no seat holding untyped cards has shown anything.
    assert view.shown_weights(WEIGHED) is None
    neutral = dataclasses.replace(PLAIN, asked=1.0, offered=1.0)
    for hold in (WEIGHED, neutral):
        for a, b in zip(worlds(view, PLAIN, 50), worlds(view, hold, 50)):
            assert a.hands == b.hands and a.deck == b.deck and a.dev_cards == b.dev_cards
    # A seat that has shown something, read by a reading without weights.
    game.ledger.show(1, [1, 0, 0, 0, 0])
    view = View.from_game(game, 0)
    assert view.shown_weights(PLAIN) is None and view.shown_weights(None) is None
    for a, b in zip(worlds(view, None, 50), worlds(view, neutral, 50)):
        assert a.hands == b.hands


def test_a_belief_key_reads_what_was_shown_only_where_a_weight_applies():
    game = untyped_game()
    before = View.from_game(game, 0)
    game.ledger.show(1, [1, 0, 0, 0, 0])
    after = View.from_game(game, 0)
    assert before.signature() == after.signature()
    assert before.belief_key(PLAIN) == after.belief_key(PLAIN) == after.signature()
    assert before.belief_key(WEIGHED) != after.belief_key(WEIGHED)
