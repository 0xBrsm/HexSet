# SPDX-License-Identifier: GPL-3.0-only
"""Native state/economy primitives against constructed Python reference state.

No Game action is applied, no arena is created and no game is played.
"""
import copy
import dataclasses
import unittest

from hexset.native import available
if not available():
    raise unittest.SkipTest("optional native state extension is unavailable")

from hexset._native_core import ArenaState
from hexset.native.state import capture_referee
from .test_native_state import position
from hexset import economy as E, state as S
from hexset.board.ports import Port
from hexset.board.terrain import Resource, Terrain


def flat(rows):
    return tuple(int(value) for row in rows for value in row)


class StatePrimitives(unittest.TestCase):
    def assert_state(self, game, core):
        actual = core.referee_snapshot()
        expected = capture_referee(game)
        expected["epoch"] = actual["epoch"]
        self.assertEqual(actual, expected)
        for seat in range(game._state.num_players):
            state = game._state
            expected_queries = dict(
                setup=tuple(v for v in range(len(state.vertex_owner))
                            if S.can_place_settlement(state, seat, v, connected=False)),
                settlements=tuple(v for v in range(len(state.vertex_owner))
                                  if S.can_place_settlement(state, seat, v)),
                cities=tuple(v for v in range(len(state.vertex_owner))
                             if S.can_upgrade_to_city(state, seat, v)),
                roads=tuple(S.placeable_roads(state, seat)),
                counts=(S.settlement_count(state, seat), S.city_count(state, seat), S.road_count(state, seat)))
            self.assertEqual(core.placement_queries(seat), expected_queries)

    def test_cached_piece_counts_and_all_placement_candidates_follow_mutations(self):
        for players in (2, 4):
            game = position(players)
            core = ArenaState(capture_referee(game))
            self.assert_state(game, core)
            road = S.placeable_roads(game._state, 0)[0]
            S.place_road(game._state, 0, road)
            core.place_road(0, road)
            self.assert_state(game, core)
            S.upgrade_to_city(game._state, 0, 0)
            core.upgrade_to_city(0, 0)
            self.assert_state(game, core)
            vertex = next(v for v in range(len(game._state.vertex_owner))
                          if S.can_place_settlement(game._state, 1, v, connected=False))
            S.place_settlement(game._state, 1, vertex, connected=False)
            core.place_settlement(1, vertex, False)
            self.assert_state(game, core)

    def test_opponent_building_stops_road_extension_and_piece_limits_apply(self):
        game = position(4)
        edge = game._state.board.topology.vertex_edges[0][0]
        game._state.edge_owner[edge] = 0
        other = next(v for v in game._state.board.topology.edges[edge] if v != 0)
        game._state.vertex_owner[other] = 1
        game._state.vertex_building[other] = S.Building.SETTLEMENT
        core = ArenaState(capture_referee(game))
        self.assert_state(game, core)
        # Constructed supply exhaustion, independent of the placement history.
        for vertex in range(5):
            game._state.vertex_owner[vertex] = 0
            game._state.vertex_building[vertex] = S.Building.SETTLEMENT
        core = ArenaState(capture_referee(game))
        self.assert_state(game, core)
        self.assertEqual(core.placement_queries(0)["setup"], ())

    def test_affordability_payment_and_public_diff_are_separate_exact_primitives(self):
        for players in (2, 4):
            for purchase in E.Purchase:
                game = position(players)
                game._state.hands[0] = [6]*5
                core = ArenaState(capture_referee(game))
                for candidate in E.Purchase:
                    self.assertEqual(core.can_afford(0, int(candidate)), E.can_afford(game._state, 0, candidate))
                before = copy.deepcopy(game._state.hands)
                E.pay(game._state, 0, purchase)
                core.pay(0, int(purchase))
                self.assert_state(game, core)
                game.ledger.apply_hand_diff(before, game._state.hands)
                core.public_hand_diff(flat(before))
                self.assert_state(game, core)

    def test_port_ratios_bank_exchange_and_public_ledger_update(self):
        game = position(4)
        topology = game._state.board.topology
        edge = topology.vertex_edges[0][0]
        ports = (Port(edge, topology.edges[edge], None, 3),
                 Port(edge, topology.edges[edge], Resource.WOOD, 2))
        game._state.board = dataclasses.replace(game._state.board, ports=ports)
        game._state.hands[0][0] = 6
        core = ArenaState(capture_referee(game))
        for seat in range(4):
            self.assertEqual(core.trade_ratios(seat), tuple(E.trade_ratios(game._state, seat)))
        before = copy.deepcopy(game._state.hands)
        E.bank_trade(game._state, 0, Resource.WOOD, Resource.SHEEP)
        core.bank_trade(0, int(Resource.WOOD), int(Resource.SHEEP))
        self.assert_state(game, core)
        game.ledger.apply_hand_diff(before, game._state.hands)
        core.public_hand_diff(flat(before))
        self.assert_state(game, core)

    def test_gross_production_robber_gold_and_bank_shortage(self):
        for players in (2, 4):
            for claimants in (1, 2):
                game = position(players)
                vertices = game._state.board.topology.hex_vertices[0]
                game._state.vertex_owner = [-1]*len(game._state.vertex_owner)
                game._state.vertex_building = [S.Building.NONE]*len(game._state.vertex_building)
                for seat in range(claimants):
                    game._state.vertex_owner[vertices[seat]] = seat
                    game._state.vertex_building[vertices[seat]] = S.Building.CITY
                game._state.bank[0] = 1
                core = ArenaState(capture_referee(game))
                for roll in range(13):
                    self.assertEqual(core.production(roll), flat(S.production(game._state, roll)))
                before = copy.deepcopy(game._state.hands)
                grants = E.distribute(game._state, 5)
                self.assertEqual(core.distribute(5), flat(grants))
                self.assertEqual(grants[0][0], 1 if claimants == 1 else 0)
                game.ledger.apply_hand_diff(before, game._state.hands)
                core.public_hand_diff(flat(before))
                self.assert_state(game, core)
                game._state.robber = 0
                core = ArenaState(capture_referee(game))
                self.assertEqual(core.production(5), flat(S.production(game._state, 5)))
            game = position(players)
            game._state.board = dataclasses.replace(game._state.board, terrain=(Terrain.GOLD, Terrain.FIELDS))
            core = ArenaState(capture_referee(game))
            self.assertEqual(core.production(5, True), tuple(S.gold_claims(game._state, 5)))
            self.assertEqual(core.production(5), flat(S.production(game._state, 5)))

    def test_invalid_primitive_calls_leave_state_ledger_and_epoch_untouched(self):
        core = ArenaState(capture_referee(position(2)))
        before = core.referee_snapshot()
        for operation in (lambda: core.place_road(0, -1), lambda: core.place_settlement(0, 0, False),
                          lambda: core.upgrade_to_city(1, 0), lambda: core.pay(0, int(E.Purchase.CITY)),
                          lambda: core.bank_trade(0, 0, 0), lambda: core.distribute(13),
                          lambda: core.public_hand_diff([0])):
            with self.assertRaises(ValueError): operation()
            self.assertEqual(core.referee_snapshot(), before)


if __name__ == "__main__":
    unittest.main()
