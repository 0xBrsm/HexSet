# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.state.Hidden` lets `GameState` say "n cards, types unknown" for the
seats its author cannot see. Two halves: every size-only path reads the same
off an observed state as off the truth, and every identity path fails loudly.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from hexset.actions import legal_actions, options_for, space_for
from hexset.board.board import random_base_board
from hexset.devcards import dev_count, holdings
from hexset.economy import hand_size
from hexset.encoding import encode
from hexset.game import Phase, imagine, is_over, start, to_move
from hexset.onnx_record import record_from_game
from hexset.play import step_randomly
from hexset.state import (
    Hidden,
    HiddenCards,
    HiddenDeck,
    HiddenHand,
    HiddenRead,
    copy_state,
    new_game,
    observed_by,
)
from hexset.victory import card_points, victory_points
from hexset.view import View


def positions(seed: int = 3, wanted: int = 3, players: int = 4):
    """Two `imagine` copies of one position, differing only in the piles `seat`
    cannot see.
    """
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    found = []
    for step in range(600):
        if is_over(game) or len(found) == wanted:
            break
        seat = to_move(game)
        if (
            game.phase is Phase.MAIN
            and step > 40
            and step % 7 == 0
            and len(options_for(game)) > 1
            and sum(View.from_game(game, seat).unknown)
        ):
            truth = imagine(game, random.Random(1), randomize_deck=False)
            seen = imagine(game, random.Random(1), randomize_deck=False)
            seen.set_state(observed_by(seen.state(seat, hidden=False), seat))
            found.append((truth, seen, seat))
        step_randomly(game, rng)
    assert len(found) == wanted, f"only {len(found)} usable positions"
    return found


def test_a_hidden_pile_counts_copies_and_refuses_identities():
    rng = random.Random(0)
    state = new_game(random_base_board(rng), 4, rng)
    hand = HiddenHand(4)
    state.hands[1] = hand
    state.dev_cards[1] = HiddenCards(2)
    state.new_dev_cards[1] = HiddenCards(1)
    state.deck = HiddenDeck(20)
    assert len(hand) == 4
    assert hand_size(state, 1) == 4
    assert dev_count(state, 1) == 3
    assert len(state.deck) == 20
    assert HiddenHand(0)
    assert hand[:] == hand and isinstance(hand[:], HiddenHand)
    assert HiddenHand(4) != HiddenCards(4)
    for read in (lambda: hand[0], lambda: sum(hand), lambda: list(hand),
                 lambda: random.Random(0).shuffle(HiddenDeck(3))):
        with pytest.raises(HiddenRead):
            read()
    with pytest.raises(HiddenRead):
        hand[0] = 1
    with pytest.raises(ValueError):
        HiddenHand(-1)


def test_a_view_of_an_observed_state_is_the_view_of_the_truth():
    for truth, seen, seat in positions():
        true_view = View.from_game(truth, seat)
        seen_view = View.from_game(seen, seat)
        assert seen_view.sizes == true_view.sizes
        assert seen_view.signature() == true_view.signature()
        assert seen_view == true_view
        assert seen_view.unseen_dev_cards() == true_view.unseen_dev_cards()


def test_sampling_an_observed_view_gives_a_fully_concrete_state():
    for truth, seen, seat in positions():
        true_state = truth.state(seat, hidden=False)
        world = View.from_game(seen, seat).sample(random.Random(7))
        assert not any(
            isinstance(pile, Hidden)
            for pile in (*world.hands, *world.dev_cards, *world.new_dev_cards, world.deck)
        )
        assert len(world.deck) == len(true_state.deck)
        for other in range(world.num_players):
            assert hand_size(world, other) == hand_size(true_state, other)
            assert dev_count(world, other) == dev_count(true_state, other)


def test_the_encoding_and_the_record_match_the_truth():
    for truth, seen, seat in positions():
        true_obs, seen_obs = encode(truth, seat), encode(seen, seat)
        for field in ("hexes", "vertices", "edges", "globals"):
            assert np.array_equal(
                getattr(true_obs, field), getattr(seen_obs, field)
            ), field
        space = space_for(truth)
        true_row = record_from_game(truth, seat, space)
        seen_row = record_from_game(seen, seat, space)
        assert set(true_row) == set(seen_row)
        for field, value in true_row.items():
            assert np.array_equal(value, seen_row[field]), field


def test_reading_a_hidden_seats_cards_fails_loudly():
    truth, _, mover = positions(wanted=1)[0]
    state = truth.state(mover, hidden=False)
    bystander = (mover + 1) % state.num_players
    seen = imagine(truth, random.Random(1), randomize_deck=False)
    seen.set_state(observed_by(state, bystander))
    hidden_state = seen.state(bystander, hidden=False)

    with pytest.raises(HiddenRead):
        legal_actions(seen)
    with pytest.raises(HiddenRead):
        hidden_state.hands[mover][0]
    with pytest.raises(HiddenRead):
        holdings(hidden_state, mover)
    with pytest.raises(HiddenRead):
        card_points(hidden_state, mover)
    with pytest.raises(HiddenRead):
        victory_points(hidden_state, mover)
    with pytest.raises(HiddenRead):
        View.from_game(seen, mover)


def test_an_empty_hidden_deck_does_not_allow_a_purchase():
    from hexset.actions import ActionType
    from hexset.devcards import can_buy
    from helpers import give

    game = start(random_base_board(random.Random(0)), 4, random.Random(0))
    game.phase = Phase.MAIN
    for resource in (2, 3, 4):
        give(game._state, 0, resource, 1)
    game._state.deck = HiddenDeck(0)
    assert not game._state.deck
    assert not can_buy(game._state, 0)
    assert all(action.type is not ActionType.BUY_DEV_CARD for action in legal_actions(game))


def test_buying_an_unknown_card_fails_before_spending_resources():
    from hexset.devcards import buy
    from helpers import give

    game = start(random_base_board(random.Random(0)), 4, random.Random(0))
    for resource in (2, 3, 4):
        give(game._state, 0, resource, 1)
    game._state.deck = HiddenDeck(1)
    before = copy_state(game._state)
    with pytest.raises(HiddenRead):
        buy(game._state, 0)
    assert game._state == before


def test_a_view_rejects_hidden_own_development_cards():
    game = start(random_base_board(random.Random(0)), 4, random.Random(0))
    game._state.dev_cards[0] = HiddenCards(1)
    with pytest.raises(HiddenRead):
        View.from_game(game, 0)
