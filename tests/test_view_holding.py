# SPDX-License-Identifier: GPL-3.0-only
"""The development cards a sampled world deals (`View.sample`): by their ages
through a hold model (`hexset.holding.HoldModel`), with a seat's cards bought
this turn kept unplayable and no seat dealt the victory point cards it would
already have won with (`hexset.view.vp_room`), and nothing the perspective
cannot see read."""

from __future__ import annotations

import dataclasses
import math
import random

import pytest

from hexset.actions import apply, legal_actions
from hexset.board.board import random_base_board
from hexset.cards import DevCard
from hexset.game import Phase, start
from hexset.clients._modelmeta import hold_config
from hexset.holding import HoldModel, chances
from hexset.view import View, vp_room

VP, KNIGHT, MONOPOLY = DevCard.VICTORY_POINT, DevCard.KNIGHT, DevCard.MONOPOLY
#: A hold model for the tests, its numbers illustrative: the engine carries
#: none of its own, a checkpoint declares them.
HOLD = HoldModel(shape=1.0, scale=(0.5, 1.0, 0.6, 0.6, 0.7))
HOLDS = (None, HOLD)


def after_setup(seed: int = 0):
    rng = random.Random(seed)
    game = start(random_base_board(rng), 4, rng)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, legal_actions(game)[0])
    return game


def hold(state, seat: int, card: DevCard, age: int, *, fresh: bool = False) -> None:
    """`seat` holds `card`, kept `age` of its turns (bought this turn when
    `fresh`), taken out of the deck so the position stays one a game could
    reach. A fresh card is the youngest, so it is added last."""
    state.deck.remove(int(card))
    (state.new_dev_cards if fresh else state.dev_cards)[seat][card] += 1
    state.dev_ages[seat].append(0 if fresh else age)


def dealt_once(view: View, world) -> bool:
    """Every unseen card dealt exactly once, to the deck or to a holding."""
    others = [s for s in range(view.num_players) if s != view.perspective]
    dealt = [world.deck.count(card) + sum(world.dev_cards[s][card] + world.new_dev_cards[s][card] for s in others)
             for card in range(len(DevCard))]
    return dealt == view.unseen_dev_cards()


def test_a_hold_model_round_trips_through_checkpoint_metadata():
    for model in (HOLD, HoldModel(shape=0.123456789, scale=(0.1, 1.0, 1 / 3, 0.25, 0.999))):
        assert hold_config(model.meta()) == model
    assert hold_config({}) is None
    assert hold_config({"hold_shape": "", "hold_scale": " "}) is None


@pytest.mark.parametrize("meta", [
    {"hold_shape": "1.0"},
    {"hold_scale": "0.5,1,0.5,0.5,0.5"},
    {"hold_shape": "0", "hold_scale": "0.5,1,0.5,0.5,0.5"},
    {"hold_shape": "nan", "hold_scale": "0.5,1,0.5,0.5,0.5"},
    {"hold_shape": "1.0", "hold_scale": "0.5,1,0.5,0.5"},
    {"hold_shape": "1.0", "hold_scale": "0.5,1.5,0.5,0.5,0.5"},
    {"hold_shape": "1.0", "hold_scale": "0.5,0,0.5,0.5,0.5"},
    {"hold_shape": "x", "hold_scale": "0.5,1,0.5,0.5,0.5"},
])
def test_a_declared_hold_model_that_cannot_be_read_fails_the_load(meta):
    with pytest.raises(ValueError):
        hold_config(meta)


def test_a_victory_point_card_is_always_kept_and_the_others_less_so_with_each_turn():
    for card in DevCard:
        assert HOLD.kept(card, 0) == 1.0
        assert HOLD.weight(card, 0) == HOLD.weight(card, 1) == 1.0
    assert all(HOLD.kept(VP, c) == 1.0 for c in range(10))
    for card in (KNIGHT, DevCard.ROAD_BUILDING, DevCard.YEAR_OF_PLENTY, MONOPOLY):
        kept = [HOLD.kept(card, c) for c in range(8)]
        assert kept == sorted(kept, reverse=True) and kept[-1] < 0.25
        assert HOLD.weight(card, 5) == HOLD.scale[card] ** (4 ** HOLD.shape)
    assert chances(0) == chances(1) == 0 and chances(4) == 3
    assert HOLD.odds([0, 1, 0, 0, 0], 3) == (0.0, 1.0, 0.0, 0.0, 0.0)


def test_a_card_kept_long_is_dealt_as_a_victory_point_card_more_often():
    game = after_setup()
    state = game._state
    hold(state, 1, KNIGHT, 1)
    hold(state, 2, KNIGHT, 8)
    view = View.from_game(game, 0)
    rng = random.Random(0)
    vp = {1: 0, 2: 0}
    for _ in range(400):
        world = view.sample(rng, HOLD)
        assert dealt_once(view, world)
        for seat in vp:
            vp[seat] += world.dev_cards[seat][VP]
    # A card bought last turn has had no turn to be played in: the unseen odds, 5 in 25.
    assert 400 * 5 / 25 * 0.7 < vp[1] < 400 * 5 / 25 * 1.3
    assert vp[2] > 2.5 * vp[1]


@pytest.mark.parametrize("model", HOLDS)
def test_a_card_bought_this_turn_stays_unplayable_in_every_world(model):
    game = after_setup()
    state = game._state
    hold(state, 1, KNIGHT, 3)
    hold(state, 1, MONOPOLY, 0, fresh=True)
    hold(state, 2, KNIGHT, 0, fresh=True)
    hold(state, 2, KNIGHT, 0, fresh=True)
    view = View.from_game(game, 0)
    rng = random.Random(1)
    for _ in range(200):
        world = view.sample(rng, model)
        assert dealt_once(view, world)
        assert (sum(world.dev_cards[1]), sum(world.new_dev_cards[1])) == (1, 1)
        assert (sum(world.dev_cards[2]), sum(world.new_dev_cards[2])) == (0, 2)
        assert sum(world.dev_cards[3]) + sum(world.new_dev_cards[3]) == 0
        assert world.dev_ages == state.dev_ages


def test_the_victory_point_room_is_the_points_short_of_a_win():
    game = after_setup()
    state = game._state
    assert vp_room(state, 1) == state.rules.winning_points - 1 - 2
    state.rules = dataclasses.replace(state.rules, winning_points=3)
    assert vp_room(state, 1) == 0
    # Points that reach the target can only have come off the seat's turn: no bound.
    state.largest_army_holder = 1
    assert vp_room(state, 1) is None


@pytest.mark.parametrize("model", HOLDS)
@pytest.mark.parametrize("target, most", [(3, 0), (4, 1)])
def test_no_seat_is_dealt_the_victory_point_cards_it_would_have_won_with(model, target, most):
    game = after_setup()
    state = game._state
    for seat in (1, 2):
        for age in (9, 8, 7, 6):
            hold(state, seat, KNIGHT, age)
    state.rules = dataclasses.replace(state.rules, winning_points=target)
    view = View.from_game(game, 0)
    rng = random.Random(2)
    seen = set()
    for _ in range(300):
        world = view.sample(rng, model)
        assert dealt_once(view, world)
        for seat in (1, 2):
            assert sum(world.dev_cards[seat]) == 4
            seen.add(world.dev_cards[seat][VP])
    assert max(seen) == most


def test_the_victory_point_bound_deals_the_exact_conditional():
    game = after_setup()
    state = game._state
    for age in (2, 2, 2):
        hold(state, 1, KNIGHT, age)
    state.rules = dataclasses.replace(state.rules, winning_points=4)   # room for one
    view = View.from_game(game, 0)
    unseen = view.unseen_dev_cards()
    vps, total = unseen[VP], sum(unseen)
    weights = [math.comb(3, v) * math.comb(total - 3, vps - v) for v in (0, 1)]
    expected = weights[1] / sum(weights)
    rng = random.Random(3)
    draws = 4000
    ones = sum(view.sample(rng).dev_cards[1][VP] for _ in range(draws))
    assert ones / draws == pytest.approx(expected, abs=0.03)


@pytest.mark.parametrize("model", HOLDS)
def test_a_world_reads_nothing_the_perspective_cannot_see(model):
    """Two tables that differ only in what seat 0 cannot see -- seat 1's card
    kinds, and the deck's order and composition between them -- deal the
    same worlds from the same stream."""
    one = after_setup()
    two = after_setup()
    for game, kinds in ((one, (KNIGHT, MONOPOLY)), (two, (VP, KNIGHT))):
        state = game._state
        hold(state, 1, kinds[0], 4)
        hold(state, 1, kinds[1], 0, fresh=True)
        hold(state, 2, KNIGHT, 2)
        state.rules = dataclasses.replace(state.rules, winning_points=4)   # room for one
    random.Random(4).shuffle(two._state.deck)
    for seed in range(20):
        a = View.from_game(one, 0).sample(random.Random(seed), model)
        b = View.from_game(two, 0).sample(random.Random(seed), model)
        assert a.hands == b.hands
        assert a.dev_cards == b.dev_cards and a.new_dev_cards == b.new_dev_cards
        assert a.deck == b.deck
        assert a.dev_cards[1][VP] + a.new_dev_cards[1][VP] <= 1
