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


def test_heximax_can_test_more_opening_variety_without_changing_default():
    from hexset.board.terrain import TERRAIN_RESOURCE
    from hexset.bots.heximax import heximax

    board = random_base_board(random.Random(7))
    game = start(board, 4, random.Random(7))
    default = heximax(board).choose(game)
    explicit = heximax(board, placement_resource_weight=1.19).choose(game)
    diverse = heximax(board, placement_resource_weight=8).choose(game)
    assert default == explicit
    assert default in legal_actions(game) and diverse in legal_actions(game)

    def resources(action):
        return {TERRAIN_RESOURCE[board.terrain[h]]
                for h in board.topology.vertex_hexes[action.a]} - {None}

    assert len(resources(diverse)) > len(resources(default))
