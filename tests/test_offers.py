"""A bundle the table has already turned down this turn leaves the sample of
offers `legal_actions` enumerates, without leaving the set of legal moves —
the same sample the training engine draws, so the mask a checkpoint sees here
is the one it was trained under."""

from __future__ import annotations

import random

from hexset_ui.actions import Action, ActionType, is_legal, legal_actions
from hexset_ui.board.board import random_base_board
from hexset_ui.board.terrain import Resource
from hexset_ui.game import Phase, decline_trade, end_turn, imagine, propose_trade, start
from hexset_ui.trading import bundle


def a_main_phase_game():
    rng = random.Random(0)
    game = start(random_base_board(rng), 4, rng)
    game.phase = Phase.MAIN
    game.current_player = 0
    for hand in game.state.hands:
        for r in range(len(hand)):
            hand[r] = 0
    game.state.hands[0][Resource.WOOD] = 2
    game.state.hands[1][Resource.ORE] = 1
    game.state.hands[2][Resource.BRICK] = 1
    return game


def proposals(game) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    return {
        (a.give, a.want)
        for a in legal_actions(game)
        if a.type is ActionType.PROPOSE_TRADE
    }


def test_a_declined_bundle_leaves_the_sample_for_the_rest_of_the_turn():
    game = a_main_phase_game()
    wood_for_ore = (bundle(wood=1), bundle(ore=1))
    wood_for_brick = (bundle(wood=1), bundle(brick=1))
    assert {wood_for_ore, wood_for_brick} <= proposals(game)

    propose_trade(game, *wood_for_ore)
    assert game.pending_responders == [1]
    decline_trade(game, 1)
    assert game.phase is Phase.MAIN

    after = proposals(game)
    assert wood_for_ore not in after
    assert wood_for_brick in after


def test_the_declined_bundle_is_still_a_legal_move():
    game = a_main_phase_game()
    propose_trade(game, bundle(wood=1), bundle(ore=1))
    decline_trade(game, 1)
    again = Action(ActionType.PROPOSE_TRADE, give=bundle(wood=1), want=bundle(ore=1))
    options = legal_actions(game)
    assert again not in options
    assert is_legal(game, again, options)


def test_imagining_a_game_copies_what_was_offered():
    game = a_main_phase_game()
    propose_trade(game, bundle(wood=1), bundle(ore=1))
    decline_trade(game, 1)
    copy = imagine(game, random.Random(1))
    assert copy.offered == game.offered
    copy.offered.add((bundle(wood=1), bundle(brick=1)))
    assert (bundle(wood=1), bundle(brick=1)) not in game.offered


def test_the_next_turn_starts_with_nothing_offered():
    game = a_main_phase_game()
    propose_trade(game, bundle(wood=1), bundle(ore=1))
    decline_trade(game, 1)
    assert game.offered
    end_turn(game)
    assert game.offered == set()
