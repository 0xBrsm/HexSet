# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

from hexset.actions import Action, ActionType, apply, legal_actions
from hexset.board.board import random_base_board
from hexset.board.terrain import Resource
from hexset.game import Phase, start
from hexset.placement import PlacementBot, scarce_resources


def test_scarcity_is_read_off_the_board_not_hardcoded():
    board = random_base_board(random.Random(0))
    assert scarce_resources(board) == {Resource.BRICK, Resource.ORE}


def test_the_wrapper_only_intercepts_setup_settlements():
    board = random_base_board(random.Random(1))
    game = start(board, 4, rng=random.Random(2))
    sentinel = Action(ActionType.END_TURN)

    class Inner:
        def __init__(self):
            self.calls = 0

        def choose(self, _game):
            self.calls += 1
            return sentinel

    inner = Inner()
    bot = PlacementBot(inner)

    assert game.phase is Phase.SETUP_SETTLEMENT
    chosen = bot.choose(game)
    assert chosen.type is ActionType.SETUP_SETTLEMENT
    assert inner.calls == 0

    game.phase = Phase.SETUP_ROAD
    assert bot.choose(game) is sentinel
    assert inner.calls == 1


def test_the_wrapper_picks_a_legal_vertex_throughout_setup():
    board = random_base_board(random.Random(4))
    game = start(board, 4, rng=random.Random(6))
    bot = PlacementBot(None)

    placed: list[int] = []
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        options = legal_actions(game)
        if game.phase is Phase.SETUP_ROAD:
            apply(game, options[0])
            continue
        action = bot.choose(game)
        assert action in options
        placed.append(action.a)
        apply(game, action)

    assert len(placed) == 8
    assert len(set(placed)) == 8
