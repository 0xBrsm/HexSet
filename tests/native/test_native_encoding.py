# SPDX-License-Identifier: GPL-3.0-only
"""Constructed observation parity; no saved positions, games or timing loop."""
import copy
from unittest import mock
import unittest

from hexset.native import available
if not available():
    raise unittest.SkipTest("optional native state extension is unavailable")

import numpy as np

from hexset.native import encode9
from hexset.native.state import NativeState
from .test_native_state import position
from hexset import contract9
from hexset.board.board import base_board
from hexset.cards import DevCard
from hexset.chance import DiceDeck
from hexset.game import Phase
from hexset.state import Building


class NativeEncoding(unittest.TestCase):
    def assert_row(self, game, native, seat, contract):
        expected = contract9.flatten(contract9.encode9(game, seat, contract))
        actual = encode9(native, seat, contract)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(actual.flags.writeable and actual.flags.c_contiguous)
        if actual.tobytes() != expected.tobytes():
            indices = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))[:8]
            self.fail(f"{contract}, seat{seat}: "+repr([(int(i),float(actual[i]),float(expected[i]),
                int(actual.view(np.uint32)[i]),int(expected.view(np.uint32)[i])) for i in indices]))

    def contracts(self, players):
        return [contract for contract in contract9.CONTRACTS if contract not in contract9.DUEL_CONTRACTS or players == 2]

    def test_original_contracts_all_seats_and_independent_writable_rows(self):
        for players in (2, 4):
            game = position(players)
            native = NativeState(game)
            before = native.core.referee_snapshot()
            for contract in self.contracts(players):
                for seat in range(players):
                    with self.subTest(players=players, contract=contract, seat=seat):
                        self.assert_row(game, native, seat, contract)
                        row = encode9(native, seat, contract)
                        row[:] = -7
                        self.assert_row(game, native, seat, contract)
            self.assertEqual(before, native.core.referee_snapshot())

    def test_private_certainty_bounds_overclaims_and_residual_padding(self):
        for players in (2, 4):
            for variant in ("bounded", "contradictory", "overclaimed", "empty-pool", "weighted-padding"):
                game = position(players)
                for row in game.ledger.seats:
                    row.known = [0]*5
                    row.unknown = 4
                    row.most = [1, 1, 1, 1, 1] if variant == "bounded" else [0]*5
                if variant == "overclaimed":
                    for row in game.ledger.seats:
                        row.known = [9, 8, 7, 0, 0]
                        row.most = [9, 8, 7, 0, 0]
                elif variant == "empty-pool":
                    game._state.bank = [19]*5
                elif variant == "weighted-padding":
                    game._state.bank = [16, 15, 19, 19, 19]
                # A private steal witness has its own ledger, separate from
                # the public bounds and other perspectives' private ledgers.
                game.ledger.private = [None]*players
                game.ledger.private[1] = game.ledger.knowledge(1)
                game.ledger.private[1].seats[0].known = [1, 0, 0, 0, 0]
                native = NativeState(game)
                for contract in self.contracts(players):
                    for seat in range(players):
                        with self.subTest(players=players, variant=variant, contract=contract, seat=seat):
                            self.assert_row(game, native, seat, contract)

    def test_public_balanced_dice_phases_resume_and_next_unlocked_roller(self):
        for players in (2, 4):
            for phase, resume in ((Phase.ROLL, Phase.ROLL), (Phase.MAIN, Phase.MAIN),
                                  (Phase.ROBBER, Phase.ROLL), (Phase.ROBBER, Phase.MAIN),
                                  (Phase.DISCARD, Phase.MAIN)):
                for zero_mass in (False, True):
                    game = position(players)
                    game.phase, game.resume_phase = phase, resume
                    game.current_player, game.first = 1, players-1
                    game.locked = frozenset({0})
                    game.dev_card_played = True
                    game.discard_quota = [2]*players
                    left = (0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0) if zero_mass else (1, 0, 3, 4, 0, 6, 5, 4, 0, 2, 1)
                    game._state.dice = DiceDeck(left=left, cards=sum(left), recent=(7,)*5,
                                                sevens=((0, 8), (1, 1)), run=(0, 3))
                    native = NativeState(game)
                    for contract in self.contracts(players):
                        for seat in range(players):
                            with self.subTest(players=players, phase=phase, zero=zero_mass, contract=contract, seat=seat):
                                self.assert_row(game, native, seat, contract)

    def test_standard_board_rows_and_native_road_cache_after_placement(self):
        for players in (2, 4):
            game = position(players)
            board = base_board()
            game._state.board = board
            game._state.vertex_owner = [0]+[-1]*(board.topology.num_vertices-1)
            game._state.vertex_building = [Building.SETTLEMENT]+[Building.NONE]*(board.topology.num_vertices-1)
            game._state.edge_owner = [-1]*board.topology.num_edges
            game._state.robber = 0
            native = NativeState(game)
            for contract in self.contracts(players):
                for seat in range(players):
                    self.assert_row(game, native, seat, contract)
                    if contract == "9d":
                        self.assertEqual(len(encode9(native, seat, contract)), 828)
                    if contract == "11d":
                        self.assertEqual(len(encode9(native, seat, contract)), 1000)
            edge = native.core.placement_queries(0)["roads"][0]
            native.core.place_road(0, edge)
            updated = native.restore_referee()
            for contract in self.contracts(players):
                for seat in range(players):
                    self.assert_row(updated, native, seat, contract)

    def test_nonzero_seven_balance_and_next_roller_have_exact_odds(self):
        for players in (2, 4):
            for phase in (Phase.ROLL, Phase.MAIN):
                for locked in (frozenset(), frozenset({1})):
                    game = position(players)
                    game.current_player, game.phase, game.locked = 0, phase, locked
                    game._state.dice = DiceDeck(recent=(5, 9, 5, 7, 3),
                                                sevens=((0, 1), (1, 2)), run=(None, 0))
                    native = NativeState(game)
                    for contract in self.contracts(players):
                        for seat in range(players):
                            self.assert_row(game, native, seat, contract)

    def test_exact_age_sorting_fills_and_card_playability(self):
        for players in (2, 4):
            game = position(players)
            game.dev_card_played = True
            game.current_player = 1
            for seat in range(players):
                game._state.dev_cards[seat] = [2, 1, 1, 1, 1]
                game._state.new_dev_cards[seat] = [0, 1, 0, 1, 0]
                game._state.dev_ages[seat] = None if seat%2 else [0, 6, 2, 6, 9, 1, 4]
            native = NativeState(game)
            for contract in self.contracts(players):
                for seat in range(players):
                    self.assert_row(game, native, seat, contract)

    def test_opponent_identity_and_deck_order_cannot_change_visible_row(self):
        for players in (2, 4):
            game = position(players)
            altered = copy.deepcopy(game)
            for seat in range(1, players):
                altered._state.hands[seat] = [0, 0, 0, 0, sum(game._state.hands[seat])]
                altered._state.dev_cards[seat] = [sum(game._state.dev_cards[seat]), 0, 0, 0, 0]
                altered._state.new_dev_cards[seat] = [0, 0, 0, 0, sum(game._state.new_dev_cards[seat])]
            altered._state.deck.reverse()
            native, changed = NativeState(game), NativeState(altered)
            for contract in self.contracts(players):
                self.assertEqual(encode9(native, 0, contract).tobytes(), encode9(changed, 0, contract).tobytes())

    def test_no_python_view_feature_or_restore_callback_per_row(self):
        game = position(2)
        native = NativeState(game)
        expected = {contract:contract9.flatten(contract9.encode9(game, 0, contract)).tobytes()
                    for contract in self.contracts(2)}
        with mock.patch.object(contract9.View, "from_game", side_effect=AssertionError("Python View called")), \
             mock.patch.object(native, "restore_referee", side_effect=AssertionError("referee restored")), \
             mock.patch.object(contract9, "roads_to_take_all", side_effect=AssertionError("Python road query")), \
             mock.patch.object(contract9, "reach_all", side_effect=AssertionError("Python reach query")), \
             mock.patch.object(contract9, "knights_needed_all", side_effect=AssertionError("Python knight query")), \
             mock.patch.object(contract9, "_ages", side_effect=AssertionError("Python ages called")), \
             mock.patch.object(contract9, "_template", side_effect=AssertionError("Python template called")):
            for contract, row in expected.items():
                self.assertEqual(encode9(native, 0, contract).tobytes(), row)

    def test_native_mutations_and_clone_reuse_only_static_templates(self):
        native = NativeState(position(4))
        encode9(native, 0)
        child = native.fork()
        child.core.upgrade_to_city(0, 0)
        child.core.ledger_receive(1, 3, 1)
        self.assertNotEqual(encode9(native, 0).tobytes(), encode9(child, 0).tobytes())
        for state in (native, child):
            oracle = state.restore_referee()
            for contract in self.contracts(4):
                for seat in range(4):
                    self.assert_row(oracle, state, seat, contract)

    def test_invalid_perspective_contract_and_9d_player_count_are_refused(self):
        native = NativeState(position(4))
        for seat in (-1, 4):
            with self.assertRaises(ValueError): encode9(native, seat)
        with self.assertRaises(ValueError): encode9(native, 0, "unknown")
        with self.assertRaises(ValueError): encode9(native, 0, "9d")
        with self.assertRaises(ValueError): encode9(native, 0, "11d")


if __name__ == "__main__":
    unittest.main()
