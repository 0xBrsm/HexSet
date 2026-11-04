# SPDX-License-Identifier: GPL-3.0-only
"""Constructed graph-query parity; no game, position bank or timing run."""
import copy
import unittest

from hexset.native import available
if not available():
    raise unittest.SkipTest("optional native state extension is unavailable")

from hexset._native_core import ArenaState
from hexset.native.state import capture_referee
from .test_native_state import position
from hexset import roads as R, victory as V
from hexset.state import Building


class PersistentNativeRoads(unittest.TestCase):
    def assert_queries(self, game, core):
        state = game._state
        for seat in range(state.num_players):
            paths = []
            best = R.longest_with_new_roads(state, seat, 2, paths)
            actual = core.road_queries(seat, 2)
            self.assertEqual(actual, dict(lengths=tuple(R.road_lengths(state)), with_new=tuple(best),
                paths=tuple(map(tuple, paths)), to_take=R.roads_to_take_all(state),
                reach=tuple(v for row in R.reach_all(state) for v in row)))
            self.assertEqual(core.road_queries(seat, 2), actual)  # cached query identity in values

    def test_queries_paths_and_cache_follow_primitive_occupancy_changes(self):
        for players in (2, 4):
            game = position(players)
            V.update_longest_road(game._state)
            V.update_largest_army(game._state)
            core = ArenaState(capture_referee(game))
            self.assert_queries(game, core)
            edge = game._state.board.topology.vertex_edges[0][0]
            from hexset.state import place_road
            place_road(game._state, 0, edge)
            core.place_road(0, edge)
            # Same stale cache values are read by both primitives until refresh.
            self.assert_queries(game, core)
            V.update_longest_road(game._state, road_owner=0)
            V.update_largest_army(game._state)
            core.refresh_awards(0)
            self.assert_queries(game, core)
            actual = core.referee_snapshot()
            expected = capture_referee(game)
            expected["epoch"] = actual["epoch"]
            self.assertEqual(actual, expected)

    def test_loop_branch_and_opponent_blocker_match_python_search_ties(self):
        for players in (2, 4):
            game = position(players)
            topology = game._state.board.topology
            for edge in topology.hex_edges[0]: game._state.edge_owner[edge] = 0
            vertex = topology.hex_vertices[0][2]
            game._state.vertex_owner[vertex] = 1
            game._state.vertex_building[vertex] = Building.SETTLEMENT
            game._state.knights_played = [3] + [0]*(players-1)
            V.update_longest_road(game._state)
            V.update_largest_army(game._state)
            core = ArenaState(capture_referee(game))
            self.assert_queries(game, core)
            core.refresh_awards()
            actual = core.referee_snapshot()
            expected = capture_referee(game)
            expected["epoch"] = actual["epoch"]
            self.assertEqual(actual, expected)

    def test_incremental_refresh_refuses_unfilled_road_cache_without_changes(self):
        game = position(4)
        game._state.edge_owner[0] = 1
        core = ArenaState(capture_referee(game))
        before = core.referee_snapshot()
        with self.assertRaises(ValueError): core.refresh_awards(0)
        self.assertEqual(core.referee_snapshot(), before)


if __name__ == "__main__":
    unittest.main()
