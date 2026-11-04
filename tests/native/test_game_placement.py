# SPDX-License-Identifier: GPL-3.0-only
"""Constructed placement action parity, no complete games or timing runs."""
import copy
import unittest

from hexset.native import available
if not available():
    raise unittest.SkipTest("optional native state extension is unavailable")

from hexset._native_core import ArenaState
from hexset.native.state import capture_referee, NativeState
from .test_native_state import position
from hexset import state as S, victory as V
from hexset.actions import Action, ActionType, apply
from hexset.game import Phase


def prepared(players):
    game = position(players)
    game._state.hands[0] = [6]*5
    V.update_longest_road(game._state)
    V.update_largest_army(game._state)
    return game


class NativePlacementActions(unittest.TestCase):
    def assert_transition(self, game, action):
        native = NativeState(game)
        core = native.core
        # Populate generation caches before mutating, to detect stale answers.
        core.road_queries(game.current_player, 1)
        apply(game, action)
        native.apply(action)
        actual = core.referee_snapshot()
        expected = capture_referee(game)
        expected["epoch"] = actual["epoch"]
        self.assertEqual(actual, expected)
        self.assertEqual(core.road_queries(game.current_player, 1)["lengths"], tuple(V.road_lengths(game._state)))
        return core

    def test_first_and_second_setup_settlements_include_resource_grants(self):
        for players in (2, 4):
            for second in (False, True):
                game = prepared(players)
                game.phase = Phase.SETUP_SETTLEMENT
                game.setup_step = players if second else 0
                vertex = next(v for v in range(len(game._state.vertex_owner))
                              if S.can_place_settlement(game._state, 0, v, connected=False))
                before = sum(game._state.hands[0])
                self.assert_transition(game, Action(ActionType.SETUP_SETTLEMENT, vertex))
                self.assertEqual(game.phase, Phase.SETUP_ROAD)
                if not second: self.assertEqual(sum(game._state.hands[0]), before)
                else: self.assertGreater(sum(game._state.hands[0]), before)

    def test_setup_road_advances_snake_and_skips_locked_entries(self):
        for players in (2, 4):
            for last in (False, True):
                game = prepared(players)
                game.phase = Phase.SETUP_ROAD
                game.last_settlement = 0
                game.setup_step = len(game.setup_queue)-1 if last else 0
                if players == 4 and not last: game.locked = frozenset({1})
                edge = game._state.board.topology.vertex_edges[0][0]
                self.assert_transition(game, Action(ActionType.SETUP_ROAD, edge))
                self.assertEqual(game.phase, Phase.ROLL if last else Phase.SETUP_SETTLEMENT)
                self.assertEqual(game.current_player, 0 if last else (2 if players == 4 else 1))

    def test_paid_and_before_roll_free_road_updates_are_complete(self):
        for players in (2, 4):
            for free in (False, True):
                game = prepared(players)
                if free: game.phase, game.free_roads = Phase.ROLL, 2
                edge = S.placeable_roads(game._state, 0)[0]
                hand_before = copy.deepcopy(game._state.hands)
                self.assert_transition(game, Action(ActionType.BUILD_ROAD, edge))
                self.assertEqual(game.phase, Phase.ROLL if free else Phase.MAIN)
                if free:
                    self.assertEqual(game._state.hands, hand_before)
                    self.assertEqual(game.free_roads, 1)

    def test_connected_settlement_pays_and_updates_road_awards(self):
        for players in (2, 4):
            game = prepared(players)
            # Construct one connected buildable vertex far enough from vertex0.
            target = next(v for v in range(len(game._state.vertex_owner))
                          if S.can_place_settlement(game._state, 0, v, connected=False))
            edge = game._state.board.topology.vertex_edges[target][0]
            game._state.edge_owner[edge] = 0
            V.update_longest_road(game._state)
            self.assertTrue(S.can_place_settlement(game._state, 0, target))
            self.assert_transition(game, Action(ActionType.BUILD_SETTLEMENT, target))
            self.assertEqual(game._state.vertex_owner[target], 0)

    def test_free_roads_in_main_keep_trade_window_closed_until_credit_resolves(self):
        game = prepared(4)
        game.free_roads = 2
        game.trade_event_turn = -1
        edge = S.placeable_roads(game._state, 0)[0]
        self.assert_transition(game, Action(ActionType.BUILD_ROAD, edge))
        self.assertEqual(game.trade_event_turn, -1)

    def test_illegal_placement_and_stale_award_cache_preserve_everything(self):
        game = prepared(4)
        core = ArenaState(capture_referee(game))
        before = core.referee_snapshot()
        for kind, index in ((ActionType.SETUP_SETTLEMENT, 0), (ActionType.SETUP_ROAD, 0),
                            (ActionType.BUILD_ROAD, -1), (ActionType.BUILD_SETTLEMENT, 0)):
            with self.assertRaises(ValueError): core.apply_placement(int(kind), index)
            self.assertEqual(core.referee_snapshot(), before)
        with self.assertRaises(NotImplementedError): core.apply_placement(int(ActionType.ROLL), 0)
        self.assertEqual(core.referee_snapshot(), before)
        game._state.edge_owner[-1] = 1
        game._state.road_lengths[1] = 0
        core = ArenaState(capture_referee(game))
        before = core.referee_snapshot()
        edge = S.placeable_roads(game._state, 0)[0]
        with self.assertRaises(ValueError): core.apply_placement(int(ActionType.BUILD_ROAD), edge)
        self.assertEqual(core.referee_snapshot(), before)


if __name__ == "__main__":
    unittest.main()
