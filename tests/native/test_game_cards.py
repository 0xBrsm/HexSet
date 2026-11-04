# SPDX-License-Identifier: GPL-3.0-only
"""Constructed single card actions, no arena, saved bank or game run."""
import dataclasses
import unittest

from hexset.native import available
if not available():
    raise unittest.SkipTest("optional native state extension is unavailable")

from hexset._native_core import ArenaState
from hexset.native.state import capture_referee, NativeState
from .test_native_state import position
from hexset import victory as V
from hexset.actions import Action, ActionType, apply
from hexset.cards import DevCard
from hexset.game import Phase


def prepared(players, phase=Phase.MAIN):
    game = position(players)
    game.phase = phase
    game._state.hands[0] = [6]*5
    game._state.dev_cards[0] = [2, 1, 1, 1, 1]
    game._state.new_dev_cards[0] = [0, 1, 0, 0, 0]
    game._state.dev_ages[0] = [6, 5, 4, 3, 2, 1, 0]
    game._state.rules = dataclasses.replace(game._state.rules, winning_points=100)
    V.update_longest_road(game._state)
    V.update_largest_army(game._state)
    return game


class NativeCardActions(unittest.TestCase):
    def assert_transition(self, game, action):
        native = NativeState(game)
        core = native.core
        apply(game, action)
        native.apply(action)
        actual = core.referee_snapshot()
        expected = capture_referee(game)
        expected["epoch"] = actual["epoch"]
        self.assertEqual(actual, expected)
        return core

    def test_all_playable_kinds_before_roll_and_main_and_age_history(self):
        for players in (2, 4):
            for phase in (Phase.ROLL, Phase.MAIN):
                for kind, argument in ((ActionType.PLAY_KNIGHT, 0), (ActionType.PLAY_ROAD_BUILDING, 0),
                                       (ActionType.PLAY_MONOPOLY, 1), (ActionType.PLAY_YEAR_OF_PLENTY, 7)):
                    for unknown_ages in (False, True):
                        with self.subTest(players=players, phase=phase, kind=kind, unknown=unknown_ages):
                            game = prepared(players, phase)
                            if unknown_ages: game._state.dev_ages[0] = None
                            self.assert_transition(game, Action(kind, argument))

    def test_every_year_of_plenty_pair_and_public_monopoly_bounds(self):
        for pair in range(15):
            self.assert_transition(prepared(4), Action(ActionType.PLAY_YEAR_OF_PLENTY, pair))
        game = prepared(4)
        self.assert_transition(game, Action(ActionType.PLAY_MONOPOLY, 1))
        for seat in range(1, 4):
            self.assertEqual(game._state.hands[seat][1], 0)
            for perspective in range(4): self.assertEqual(game.ledger.for_seat(perspective).seats[seat].most[1], 0)

    def test_buying_held_aside_card_deck_order_and_immediate_vp_win(self):
        for players in (2, 4):
            for card in DevCard:
                game = prepared(players)
                game._state.deck[-1] = card
                self.assert_transition(game, Action(ActionType.BUY_DEV_CARD))
                self.assertGreater(game._state.new_dev_cards[0][card], 0)
            game = prepared(players)
            game._state.deck[-1] = DevCard.VICTORY_POINT
            game._state.rules = dataclasses.replace(game._state.rules, winning_points=4)
            self.assert_transition(game, Action(ActionType.BUY_DEV_CARD))
            self.assertEqual((game.phase, game.won_by), (Phase.GAME_OVER, 0))

    def test_knight_can_end_game_before_robber_or_chance(self):
        game = prepared(2, Phase.ROLL)
        game._state.knights_played[0] = 2
        game._state.rules = dataclasses.replace(game._state.rules, winning_points=5)
        self.assert_transition(game, Action(ActionType.PLAY_KNIGHT))
        self.assertEqual((game.phase, game.won_by), (Phase.GAME_OVER, 0))
        self.assertEqual(game.chance.events, [])

    def test_rejected_cards_preserve_native_state_ledger_deck_and_epoch(self):
        game = prepared(4)
        game._state.bank[0] = 0
        core = ArenaState(capture_referee(game))
        before = core.referee_snapshot()
        for kind, argument in ((ActionType.PLAY_YEAR_OF_PLENTY, 0), (ActionType.PLAY_MONOPOLY, -1)):
            with self.assertRaises(ValueError): core.apply_cards(int(kind), argument)
            self.assertEqual(core.referee_snapshot(), before)
        game.dev_card_played = True
        core = ArenaState(capture_referee(game))
        before = core.referee_snapshot()
        with self.assertRaises(ValueError): core.apply_cards(int(ActionType.PLAY_KNIGHT))
        self.assertEqual(core.referee_snapshot(), before)
        game.dev_card_played = False
        game._state.dev_cards[0][int(DevCard.ROAD_BUILDING)] = 0
        core = ArenaState(capture_referee(game))
        before = core.referee_snapshot()
        with self.assertRaises(ValueError): core.apply_cards(int(ActionType.PLAY_ROAD_BUILDING))
        self.assertEqual(core.referee_snapshot(), before)


if __name__ == "__main__":
    unittest.main()
