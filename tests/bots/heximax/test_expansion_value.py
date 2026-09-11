"""Bounded expansion credit rewards useful roads without outweighing a build."""
import random

import numpy as np
import pytest

from hexset.board.board import random_base_board
from hexset.bots.heximax import heximax
from hexset.game import start
from hexset.state import (can_place_road, can_place_settlement, place_road,
                          place_settlement, MAX_SETTLEMENTS, Building)


def position():
    board = random_base_board(random.Random(3))
    game = start(board, 4, random.Random(4))
    state = game._state
    place_settlement(state, 0, 0, connected=False)
    # Grow an actual legal path until there is a buildable site.
    for _ in range(6):
        places = [v for v in range(len(state.vertex_owner))
                  if can_place_settlement(state, 0, v)]
        if places:
            return game, places
        edge = next(e for e in range(len(state.edge_owner)) if can_place_road(state, 0, e))
        place_road(state, 0, edge)
    raise AssertionError('fixture failed to open a settlement site')


def test_option_is_bounded_and_building_cannot_lose_to_its_option_credit():
    game, places = position()
    state = game._state
    bot = heximax(state.board, expansion_value=.5)
    before = bot.evaluator.score(state, 0, [0]*5, knower=0)
    assert 0 < bot.evaluator.expansion_bonus(state, 0) <= .5
    place_settlement(state, 0, places[0])
    after = bot.evaluator.score(state, 0, [0]*5, knower=0)
    assert after > before + .5


def test_no_piece_supply_means_no_expansion_credit():
    game, _ = position()
    state = game._state
    # The supply gate is a public board property; no hand information enters.
    owned = [v for v,x in enumerate(state.vertex_owner) if x==0]
    for v in [v for v,x in enumerate(state.vertex_owner) if x<0][:MAX_SETTLEMENTS-len(owned)]:
        state.vertex_owner[v]=0
        state.vertex_building[v]=Building.SETTLEMENT
    assert heximax(state.board, expansion_value=.5).evaluator.expansion_bonus(state, 0)==0


def test_scalar_batch_and_hidden_hand_invariance():
    game, _ = position()
    state = game._state
    bot = heximax(state.board, expansion_value=.5)
    bonus=bot.evaluator.expansion_bonus(state,0)
    state.hands[1]=[5,0,0,0,0]
    assert bot.evaluator.expansion_bonus(state,0)==bonus
    hands=np.zeros((1,4,5))
    scalar=[bot.evaluator.score(state,s,hands[0,s],knower=0) for s in range(4)]
    np.testing.assert_allclose(bot.evaluator.score_many(state,0,hands)[0],scalar,rtol=0,atol=1e-12)
    assert bot.expansion_value == .5
    assert heximax(state.board).expansion_value == 0


def test_adopted_notrade_defaults_and_explicit_legacy_override():
    from dataclasses import replace
    from hexset.bots.heximax.evaluate import NO_TRADE_WEIGHTS
    from hexset.arena import PRESETS, spawn

    board = random_base_board(random.Random(11))
    adopted = spawn(PRESETS["heximax-notrade"], board, random.Random(12))
    assert adopted.expansion_value == .25
    assert adopted.evaluator.weights.road == 0
    assert adopted.max_trades == 0
    legacy = heximax(board, mode="notrade", expansion_value=0,
                     weights=replace(NO_TRADE_WEIGHTS, road=.1237))
    assert legacy.expansion_value == 0
    assert legacy.evaluator.weights.road == .1237
    assert heximax(board, mode="honest").expansion_value == 0
