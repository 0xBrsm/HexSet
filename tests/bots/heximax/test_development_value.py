"""Optional development-card value must not reveal hidden card identities."""
import random

import numpy as np
import pytest

from hexset.board.board import random_base_board
from hexset.bots.heximax import heximax
from hexset.cards import DevCard
from hexset.game import start


def fixture():
    rng = random.Random(17)
    board = random_base_board(rng)
    return start(board, 4, rng), heximax(board, random.Random(18), development_value=.2)


def take_card(state, seat, card, *, fresh=False):
    state.deck.remove(card)
    (state.new_dev_cards if fresh else state.dev_cards)[seat][card] += 1


def test_counts_own_fresh_cards_but_never_counts_vp_twice():
    game, bot = fixture()
    state = game._state
    take_card(state, 0, DevCard.KNIGHT, fresh=True)
    take_card(state, 0, DevCard.VICTORY_POINT)
    assert bot.evaluator.development_bonus(state, 0, 0) == pytest.approx(.2)
    state.new_dev_cards[0][DevCard.KNIGHT] -= 1
    state.knights_played[0] += 1
    assert bot.evaluator.development_bonus(state, 0, 0) == 0


def test_hidden_opponent_card_type_does_not_change_bonus_or_score():
    game, bot = fixture()
    state = game._state
    take_card(state, 1, DevCard.KNIGHT)
    before = bot.evaluator.score(state, 1, [0]*5, knower=0)
    # Exchange a hidden opponent card with the hidden deck, preserving the
    # entire information set, public totals and physical card conservation.
    state.dev_cards[1][DevCard.KNIGHT] -= 1
    state.dev_cards[1][DevCard.VICTORY_POINT] += 1
    state.deck.remove(DevCard.VICTORY_POINT)
    state.deck.append(DevCard.KNIGHT)
    assert bot.evaluator.score(state, 1, [0]*5, knower=0) == before


def test_batched_trade_values_include_the_same_bonus_as_search():
    game, bot = fixture()
    state = game._state
    take_card(state, 0, DevCard.ROAD_BUILDING)
    take_card(state, 1, DevCard.VICTORY_POINT, fresh=True)
    hands = np.array([[[0,0,0,0,0]]*4, [[1,2,3,0,1]]*4], dtype=float)
    batch = bot.evaluator.score_many(state, 0, hands)
    scalar = [[bot.evaluator.score(state, seat, row[seat], knower=0)
               for seat in range(4)] for row in hands]
    np.testing.assert_allclose(batch, scalar, rtol=0, atol=1e-12)


def test_default_remains_zero_and_factory_exposes_effective_parameter():
    game, bot = fixture()
    assert bot.development_value == .2
    assert heximax(game._state.board).development_value == 0
