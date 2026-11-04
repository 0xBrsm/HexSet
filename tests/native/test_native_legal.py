# SPDX-License-Identifier: GPL-3.0-only
"""Constructed legality/candidate parity; no games, banks or timing loops."""
import copy
import dataclasses
import unittest

from hexset.native import available
if not available():
    raise unittest.SkipTest("optional native state extension is unavailable")

from hexset._native_core import ArenaState
from hexset.native.state import capture_referee
from .test_native_state import position
from hexset import actions as A, decisions as D, state as S, victory as V
from hexset.board.board import base_board
from hexset.cards import DevCard
from hexset.game import Phase


class NativeLegality(unittest.TestCase):
    def assert_legal(self, game):
        core = ArenaState(capture_referee(game))
        before = core.referee_snapshot()
        for seat in (None, *range(game._state.num_players)):
            expected = [tuple(map(int, action)) for action in A.legal_actions(game, seat)]
            actual = core.legal_actions() if seat is None else core.legal_actions(seat)
            self.assertEqual(actual, expected, (game.phase, seat))
        self.assertEqual(core.referee_snapshot(), before)
        return core

    def assert_decision(self, game, seat=None):
        seat = game.current_player if seat is None else seat
        core = self.assert_legal(game)
        expected = D.tensors(game, seat, A.space_for(game), double_turn=False)
        actual = core.decision(seat)
        self.assertEqual(actual['flat'], expected.flat.tobytes())
        self.assertEqual(actual['setup'], expected.setup.tobytes())
        self.assertEqual(actual['roads'], expected.road_building.tobytes())
        self.assertEqual(actual['discard'], expected.discard.tobytes())
        return actual

    def test_every_phase_and_named_discard_actor_order(self):
        for players in (2, 4):
            for phase in Phase:
                with self.subTest(players=players, phase=phase):
                    game = position(players)
                    game.phase, game.last_settlement = phase, 0
                    game.discard_quota = [2, 1] + [0]*(players-2)
                    self.assert_legal(game)

    def test_main_cards_affordability_bank_and_piece_masks(self):
        for players in (2, 4):
            for played in (False, True):
                for rich in (False, True):
                    game = position(players)
                    game.dev_card_played = played
                    game._state.hands[0] = [6]*5 if rich else [0]*5
                    game._state.dev_cards[0] = [1]*5
                    game._state.dev_ages[0] = [0]*5
                    self.assert_decision(game)

    def test_before_roll_filter_and_free_road_partial_refusal(self):
        for players in (2, 4):
            for permitted in (None, frozenset(), frozenset({DevCard.KNIGHT, DevCard.MONOPOLY})):
                game = position(players)
                game.phase, game.before_roll = Phase.ROLL, permitted
                game._state.dev_cards[0] = [1]*5
                game._state.dev_ages[0] = [0]*5
                self.assert_decision(game)
                game.free_roads = 2
                core = self.assert_legal(game)
                with self.assertRaises(ValueError): core.decision(0)
                with self.assertRaises(ValueError): D.tensors(game, 0, A.space_for(game), double_turn=False)

    def test_setup_incidence_order_and_partial_setup_refusal(self):
        for players in (2, 4):
            game = position(players)
            game.phase = Phase.SETUP_SETTLEMENT
            self.assert_decision(game)
            game.phase, game.last_settlement = Phase.SETUP_ROAD, 0
            core = self.assert_legal(game)
            with self.assertRaises(ValueError): core.decision(0)

    def test_discard_vectors_keep_exact_lexical_order_and_zero_quota(self):
        for players in (2, 4):
            game = position(players)
            game.phase = Phase.DISCARD
            game._state.hands[0] = [2, 1, 2, 0, 1]
            game.discard_quota = [2, 0] + [0]*(players-2)
            self.assert_decision(game, 0)
            self.assert_decision(game, 1)

    def test_robber_friendly_and_explicit_host_targets(self):
        for players in (2, 4):
            for allowed in (None, frozenset(), frozenset({0}), frozenset({1})):
                game = position(players)
                game.phase, game.robber_allowed = Phase.ROBBER, allowed
                game._state.rules = dataclasses.replace(game._state.rules, friendly_robber=True)
                self.assert_decision(game)

    def test_compound_road_pairs_preserve_order_on_loops_and_blockers(self):
        for players in (2, 4):
            for blocked in (False, True):
                game = position(players)
                topology = game._state.board.topology
                for edge in topology.hex_edges[0]: game._state.edge_owner[edge] = 0
                if blocked:
                    vertex = topology.hex_vertices[0][2]
                    game._state.vertex_owner[vertex], game._state.vertex_building[vertex] = 1, S.Building.SETTLEMENT
                game._state.dev_cards[0][DevCard.ROAD_BUILDING] = 1
                game._state.dev_ages[0] = None
                V.update_longest_road(game._state)
                game._state.rules = dataclasses.replace(game._state.rules, winning_points=6)
                self.assert_decision(game)

    def test_standard_board_all_deciding_seats_and_relative_robber_columns(self):
        for players in (2, 4):
            for actor in range(players):
                game = position(players)
                board = base_board()
                topology = board.topology
                game._state.board = board
                game._state.vertex_owner = [-1]*topology.num_vertices
                game._state.vertex_building = [S.Building.NONE]*topology.num_vertices
                game._state.edge_owner = [-1]*topology.num_edges
                game._state.robber = -1
                occupied = []
                opponents = [seat for seat in range(players) if seat != actor]
                for seat, vertex in zip(opponents, topology.hex_vertices[0][::2]):
                    game._state.vertex_owner[vertex] = seat
                    game._state.vertex_building[vertex] = S.Building.SETTLEMENT
                    occupied.append(vertex)
                own = next(vertex for vertex in range(topology.num_vertices)
                           if vertex not in topology.hex_vertices[0] and vertex not in occupied
                           and not set(topology.vertex_neighbors[vertex]).intersection(occupied))
                game._state.vertex_owner[own] = actor
                game._state.vertex_building[own] = S.Building.SETTLEMENT
                game.current_player = actor
                game._state.hands[actor] = [6]*5
                game._state.dev_cards[actor] = [1]*5
                game._state.dev_ages[actor] = None
                game._state.rules = dataclasses.replace(game._state.rules, friendly_robber=False)
                V.update_longest_road(game._state)
                V.update_largest_army(game._state)
                for phase in (Phase.ROLL, Phase.MAIN, Phase.ROBBER, Phase.SETUP_SETTLEMENT):
                    with self.subTest(players=players, actor=actor, phase=phase):
                        game.phase = phase
                        self.assert_decision(game, actor)

    def test_hidden_opponent_composition_never_changes_legal_requests(self):
        game = position(4)
        game.phase = Phase.ROBBER
        first = self.assert_decision(game)
        changed = copy.deepcopy(game)
        for seat in range(1, 4):
            changed._state.hands[seat] = [0, 0, 0, 0, sum(game._state.hands[seat])]
            changed._state.dev_cards[seat] = [1, 0, 0, 0, 0]
            changed._state.dev_ages[seat] = None
        self.assertEqual(first, self.assert_decision(changed))


if __name__ == '__main__':
    unittest.main()
