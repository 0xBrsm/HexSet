# SPDX-License-Identifier: GPL-3.0-only
"""Constructed single-action transitions; never a full game or benchmark."""
import dataclasses
import unittest

from hexset.native import available
if not available():
    raise unittest.SkipTest("optional native state extension is unavailable")

from hexset._native_core import ArenaState
from hexset.native.state import capture_referee, NativeState
from .test_native_state import position
from hexset.actions import Action, ActionType, apply
from hexset.cards import DevCard
from hexset.game import Phase


class SimpleGameActions(unittest.TestCase):
    def assert_transition(self, game, action):
        native = NativeState(game)
        core = native.core
        apply(game, action)
        native.apply(action)
        actual = core.referee_snapshot()
        expected = capture_referee(game)
        expected["epoch"] = actual["epoch"]
        self.assertEqual(actual, expected)
        self.assertEqual(native.restore_referee()._state, game._state)
        self.assertEqual(native.restore_referee().ledger, game.ledger)
        return core

    def test_city_payment_private_ledger_win_and_trade_event(self):
        for players in (2, 4):
            for wins in (False, True):
                game = position(players)
                game._state.hands[0] = [6]*5
                game._state.rules = dataclasses.replace(game._state.rules, winning_points=5 if wins else 15)
                self.assert_transition(game, Action(ActionType.BUILD_CITY, 0))
                self.assertEqual(game.phase, Phase.GAME_OVER if wins else Phase.MAIN)
                if wins: self.assertEqual(game.won_by, 0)
                else: self.assertEqual(game.trade_event_turn, game.turns)

    def test_bank_trade_updates_only_public_knowledge_and_bookkeeping(self):
        for players in (2, 4):
            game = position(players)
            game._state.hands[0][0] = 5
            self.assert_transition(game, Action(ActionType.BANK_TRADE, 0, 4))
            self.assertEqual(game.trade_event_turn, game.turns)

    def test_end_turn_matures_new_cards_and_ages_and_rotates_past_locked_seat(self):
        for players in (2, 4):
            game = position(players)
            game._state.dev_cards[0] = [1, 0, 0, 0, 0]
            game._state.new_dev_cards[0] = [0, 1, 0, 0, 1]
            game._state.dev_ages[0] = [4, 0, 0]
            game.dev_card_played = True
            game.robber_allowed = frozenset({0})
            game.locked = frozenset({1}) if players == 4 else frozenset()
            self.assert_transition(game, Action(ActionType.END_TURN))
            self.assertEqual(game.current_player, 2 if players == 4 else 1)
            self.assertEqual(game._state.dev_cards[0], [1, 1, 0, 0, 1])
            self.assertEqual(game._state.new_dev_cards[0], [0]*5)
            self.assertEqual(game._state.dev_ages[0], [5, 1, 1])
            self.assertFalse(game.dev_card_played)

    def test_end_turn_checks_next_seat_win_and_turn_cap(self):
        for players in (2, 4):
            game = position(players)
            game._state.new_dev_cards[1][int(DevCard.VICTORY_POINT)] = 15
            self.assert_transition(game, Action(ActionType.END_TURN))
            self.assertEqual((game.phase, game.won_by), (Phase.GAME_OVER, 1))
            game = position(players)
            game.turn_cap = game.turns + 1
            self.assert_transition(game, Action(ActionType.END_TURN))
            self.assertEqual(game.phase, Phase.GAME_OVER)
            self.assertEqual(game.current_player, 0)
            self.assertIsNone(game.won_by)

    def test_unplaceable_free_credit_does_not_block_main_action(self):
        game = position(4)
        game.free_roads = 1
        game._state.edge_owner = [1]*len(game._state.edge_owner)
        self.assert_transition(game, Action(ActionType.END_TURN))

    def test_illegal_and_unimplemented_actions_are_atomic(self):
        game = position(4)
        core = ArenaState(capture_referee(game))
        before = core.referee_snapshot()
        for kind, a, b in ((ActionType.BUILD_CITY, 0, 0), (ActionType.BANK_TRADE, 0, 0)):
            with self.assertRaises(ValueError): core.apply_simple(int(kind), a, b)
            self.assertEqual(core.referee_snapshot(), before)
        for kind in ActionType:
            if kind in (ActionType.END_TURN, ActionType.BUILD_CITY, ActionType.BANK_TRADE): continue
            with self.assertRaises(NotImplementedError): core.apply_simple(int(kind), 0, 0)
            self.assertEqual(core.referee_snapshot(), before)
        game.phase = Phase.ROLL
        core = ArenaState(capture_referee(game))
        before = core.referee_snapshot()
        with self.assertRaises(ValueError): core.apply_simple(int(ActionType.END_TURN))
        self.assertEqual(core.referee_snapshot(), before)
        game.phase = Phase.MAIN
        game.free_roads = 1
        core = ArenaState(capture_referee(game))
        before = core.referee_snapshot()
        with self.assertRaises(ValueError): core.apply_simple(int(ActionType.END_TURN))
        self.assertEqual(core.referee_snapshot(), before)


if __name__ == "__main__":
    unittest.main()
