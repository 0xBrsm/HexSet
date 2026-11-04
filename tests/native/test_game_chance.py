# SPDX-License-Identifier: GPL-3.0-only
"""Single constructed chance/discard actions; no games, banks or timing."""
import dataclasses
import random
import unittest

from hexset.native import available
if not available():
    raise unittest.SkipTest("optional native state extension is unavailable")

from hexset.native.state import NativeState, capture_referee
from .test_native_state import position
from hexset import robber as R
from hexset.actions import Action, ActionType, apply
from hexset.chance import Balanced, DiceDeck, Live, Recording, Scripted
from hexset.game import Phase
from hexset.state import Building


def roll_position(players, total):
    game = position(players)
    game.phase = Phase.ROLL
    game.chance = Recording(Scripted((("roll", total),)))
    return game


def robber_position(players, resume=Phase.MAIN):
    game = position(players)
    game.phase = Phase.ROBBER
    game.resume_phase = resume
    game._state.rules = dataclasses.replace(game._state.rules, friendly_robber=False)
    vertex = game._state.board.topology.hex_vertices[0][-1]
    game._state.vertex_owner[vertex] = 1
    game._state.vertex_building[vertex] = Building.SETTLEMENT
    game.chance = Recording(Scripted((("steal", 1),)))
    return game


class NativeChanceActions(unittest.TestCase):
    def assert_transition(self, game, action, seat=None):
        native = NativeState(game)
        apply(game, action, seat)
        native.apply(action, seat)
        actual = native.core.referee_snapshot()
        expected = capture_referee(game)
        expected["epoch"] = actual["epoch"]
        self.assertEqual(actual, expected)
        restored = native.restore_referee()
        self.assertEqual(restored.chance.events, game.chance.events)
        return native

    def test_every_roll_for_both_formats_and_public_dice_reshuffle(self):
        for players in (2, 4):
            for total in range(2, 13):
                for near_reshuffle in (False, True):
                    with self.subTest(players=players, total=total, reshuffle=near_reshuffle):
                        game = roll_position(players, total)
                        # Exercise history, sorted insertion of a new roller,
                        # seven-run continuation/change and public reshuffle.
                        game.current_player = 1
                        game._state.dice = DiceDeck(
                            (1, 1, 1, 1, 2, 3, 1, 1, 1, 1, 1) if near_reshuffle else DiceDeck().left,
                            13 if near_reshuffle else 36,
                            (2, 8, 7, 6, 9), ((0, 2),), (1 if near_reshuffle else 0, 2))
                        self.assert_transition(game, Action(ActionType.ROLL))

    def test_seven_quotes_large_hands_and_excludes_locked_seats(self):
        game = roll_position(4, 7)
        game._state.hands = [[5]*5, [0]*5, [9]*5, [2]*5]
        game.locked = frozenset({2})
        self.assert_transition(game, Action(ActionType.ROLL))
        self.assertEqual(game.discard_quota, [12, 0, 0, 5])
        self.assertEqual(game.phase, Phase.DISCARD)
        game = roll_position(2, 7)
        game._state.hands = [[0]*5 for _ in range(2)]
        self.assert_transition(game, Action(ActionType.ROLL))
        self.assertEqual(game.phase, Phase.ROBBER)

    def test_original_live_balanced_rng_and_recording_are_owned(self):
        for source in (Live, Balanced):
            for split in (False, True):
                game = roll_position(2, 7)
                game.chance = Recording(source(random.Random(), split=split))
                self.assert_transition(game, Action(ActionType.ROLL))

    def test_robber_steal_private_ledger_and_both_resume_phases(self):
        for players in (2, 4):
            for resume in (Phase.ROLL, Phase.MAIN):
                with self.subTest(players=players, resume=resume):
                    game = robber_position(players, resume)
                    native = self.assert_transition(game, Action(ActionType.MOVE_ROBBER, 0, 1))
                    self.assertEqual(native.restore_referee().chance.events, [("steal", 1)])
                    # Every perspective agrees with the original ledger, not
                    # just the thief's private identity-bearing rows.
                    for seat in range(players):
                        self.assertEqual(native.visible_card_facts(seat).ledger,
                                         native.core.visible_ledger(seat))

    def test_robber_no_victim_and_scripted_no_card_consume_exactly(self):
        game = position(4)
        game.phase = Phase.ROBBER
        game._state.rules = dataclasses.replace(game._state.rules, friendly_robber=False)
        native = self.assert_transition(game, Action(ActionType.MOVE_ROBBER, 1, 4))
        self.assertEqual(native.restore_referee().chance.events, [])
        game = robber_position(4)
        game.chance = Recording(Scripted((("steal", None),)))
        self.assert_transition(game, Action(ActionType.MOVE_ROBBER, 0, 1))

    def test_friendly_and_host_target_rules_use_public_points(self):
        game = robber_position(4)
        game._state.rules = dataclasses.replace(game._state.rules, friendly_robber=True)
        # Opponent VP cards do not lift the public shield.
        game._state.dev_cards[1][1] = 10
        allowed = R.allowed_targets(game._state, 0, None)
        targets = [h for h in range(game._state.board.num_hexes) if R.may_stand(game._state, h)
                   and (allowed is None or h in allowed)]
        for target in targets:
            clone = NativeState(game).restore_referee()
            victims = R.victims(clone._state, target, 0)
            self.assert_transition(clone, Action(ActionType.MOVE_ROBBER, target, victims[0] if victims else 4))
        game.robber_allowed = frozenset({0})
        self.assert_transition(game, Action(ActionType.MOVE_ROBBER, 0, 1))

    def test_discard_named_and_default_seat_partial_and_completion(self):
        for players in (2, 4):
            for named in (False, True):
                for last in (False, True):
                    game = position(players)
                    game.phase = Phase.DISCARD
                    game.discard_quota = [0]*players
                    game.discard_quota[1] = 1 if last else 2
                    self.assert_transition(game, Action(ActionType.DISCARD, 0), 1 if named else None)
                    self.assertEqual(game.phase, Phase.ROBBER if last else Phase.DISCARD)

    def test_last_discard_closes_retired_rollers_turn_and_matures_cards(self):
        game = position(4)
        game.phase = Phase.DISCARD
        game.locked = frozenset({0})
        game.discard_quota = [0, 1, 0, 0]
        game._state.new_dev_cards[0][0] = 1
        game._state.dev_ages[0].append(0)
        self.assert_transition(game, Action(ActionType.DISCARD, 0), 1)
        self.assertEqual((game.phase, game.current_player, game.turns), (Phase.ROLL, 1, 4))
        self.assertEqual(game._state.dev_cards[0][0], 1)

    def test_chance_source_refusal_preserves_native_state_and_events(self):
        for action, game in ((Action(ActionType.ROLL), roll_position(2, 8)),
                             (Action(ActionType.MOVE_ROBBER, 0, 1), robber_position(4))):
            game.chance = Recording(Scripted(()))
            native = NativeState(game)
            before = native.core.referee_snapshot()
            with self.assertRaises(Exception): native.apply(action)
            self.assertEqual(native.core.referee_snapshot(), before)
            self.assertEqual(native.restore_referee().chance.events, [])
            self.assertEqual(native.core.control(), before["arrays"]["game_control"])

    def test_rejected_moves_consume_no_chance_and_leave_native_state_exact(self):
        game = robber_position(4)
        for target, victim in ((-1, 1), (2, 1), (0, 4), (0, -1)):
            # The public state becomes unusable after any apply exception.
            # Check each rejection on an independently owned position.
            native = NativeState(game)
            before = native.core.referee_snapshot()
            with self.assertRaises(ValueError): native.apply(Action(ActionType.MOVE_ROBBER, target, victim))
            self.assertEqual(native.core.referee_snapshot(), before)
            self.assertEqual(native.restore_referee().chance.events, [])
        game.robber_allowed = frozenset()
        native = NativeState(game)
        with self.assertRaises(ValueError): native.apply(Action(ActionType.MOVE_ROBBER, 0, 1))
        self.assertEqual(native.restore_referee().chance.events, [])
        game = roll_position(2, 8)
        game.free_roads = 1
        native = NativeState(game)
        before = native.core.referee_snapshot()
        with self.assertRaises(ValueError): native.apply(Action(ActionType.ROLL))
        self.assertEqual(native.core.referee_snapshot(), before)
        self.assertEqual(native.restore_referee().chance.events, [])
        game = position(2)
        game.phase = Phase.DISCARD
        game.discard_quota = [0, 1]
        native = NativeState(game)
        before = native.core.referee_snapshot()
        with self.assertRaises(ValueError): native.apply(Action(ActionType.DISCARD, 0), 0)
        self.assertEqual(native.core.referee_snapshot(), before)


if __name__ == "__main__":
    unittest.main()
