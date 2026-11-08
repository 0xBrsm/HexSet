# SPDX-License-Identifier: GPL-3.0-only
"""Native contract-8 rows (`hexset.native.encode8`) against `hexset.encoding`, and native information-set records
(`NativeCopy.record`) against `hexset.onnx_record`: the same bytes, on constructed ledgers that bind and on every
position of played games, for every seat."""
import random
import unittest

from hexset.native import available
if not available():
    raise unittest.SkipTest("optional native state extension is unavailable")

import numpy as np

from hexset import encoding, onnx_record
from hexset.actions import space_for
from hexset.native.playout import NativeCopy
from hexset.gym.driver import Driver
from hexset.gym.lanes import LaneEnv
from hexset.native import encode8
from hexset.native.state import NativeState
from hexset.view import View
from .test_native_state import position

CONTRACTS = ("8", "8.1")


def packed(game, seat, contract):
    o = encoding.encode(game, seat, contract)
    return np.concatenate([o.hexes.ravel(), o.vertices.ravel(), o.edges.ravel(), o.globals])


class Checker:
    """An atomic seat that checks every seat's row at each position it is asked, then plays a keyed random move."""

    def __init__(self, case):
        self.case, self.rows, self.bounded = case, 0, 0

    def choose_many(self, requests):
        out = []
        for request in requests:
            game = request.game
            native = NativeState(game)
            view = NativeCopy.viewing(native, game._state.board.topology, game.num_players, game.locked)
            space = space_for(game)
            for seat in range(game.num_players):
                self.bounded += any(View.from_game(game, seat).bounded)
                for contract in CONTRACTS:
                    self.case.assert_row(game, native, seat, contract)
                    self.rows += 1
                self.case.assert_record(game, view, seat, space, request.options if seat == request.seat else None)
            out.append(_Choice(random.Random(request.draw).choice(request.options)))
        return out


class _Choice:
    def __init__(self, action):
        self.action, self.decision = action, None


class NativeEncoding8(unittest.TestCase):
    def assert_row(self, game, native, seat, contract):
        expected = packed(game, seat, contract)
        actual = encode8(native, seat, contract)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, expected.shape)
        if actual.tobytes() != expected.tobytes():
            indices = np.flatnonzero(actual.view(np.uint32) != expected.view(np.uint32))[:8]
            self.fail(f"{contract}, seat {seat}: " + repr([(int(i), float(actual[i]), float(expected[i]))
                                                          for i in indices]))

    def assert_record(self, game, view, seat, space, options):
        expected = onnx_record.record_rows([(game, seat, options)], space)
        actual = onnx_record.record_rows([(view, seat, options)], space)
        self.assertEqual(list(actual), list(expected))
        for name in expected:
            self.assertEqual(actual[name].dtype, expected[name].dtype, name)
            self.assertEqual(actual[name].shape, expected[name].shape, name)
            self.assertEqual(actual[name].tobytes(), expected[name].tobytes(), (name, seat))

    def test_constructed_ledgers_bounded_contradictory_overclaimed_and_padded(self):
        for players in (2, 4):
            for variant in ("plain", "bounded", "tight", "contradictory", "overclaimed", "empty-pool",
                            "weighted-padding"):
                game = position(players)
                for row in game.ledger.seats:
                    row.known = [0]*5
                    row.unknown = 4
                    row.most = {"bounded": [1, 1, 1, 1, 1], "tight": [3, 0, 1, 2, 0]}.get(variant, [0]*5)
                if variant == "plain":
                    for row in game.ledger.seats:
                        row.most = [99]*5
                elif variant == "overclaimed":
                    for row in game.ledger.seats:
                        row.known = [9, 8, 7, 0, 0]
                        row.most = [9, 8, 7, 0, 0]
                elif variant == "empty-pool":
                    game._state.bank = [19]*5
                elif variant == "weighted-padding":
                    game._state.bank = [16, 15, 19, 19, 19]
                game.ledger.private = [None]*players
                game.ledger.private[1] = game.ledger.knowledge(1)
                game.ledger.private[1].seats[0].known = [1, 0, 0, 0, 0]
                native = NativeState(game)
                view = NativeCopy.viewing(native, game._state.board.topology, players, game.locked)
                for seat in range(players):
                    with self.subTest(players=players, variant=variant, seat=seat):
                        self.assert_record(game, view, seat, space_for(game), None)
                    for contract in CONTRACTS:
                        with self.subTest(players=players, variant=variant, contract=contract, seat=seat):
                            self.assert_row(game, native, seat, contract)

    def test_every_position_of_played_games(self):
        bounded = 0
        for players, games in ((4, 4), (2, 4)):
            checker = Checker(self)
            env = LaneEnv(players, 11, 2, deal=games, action_cap=500, discard_order="random")
            Driver(env, {0: checker}, keep=lambda episode: None).play()
            self.assertGreater(checker.rows, 1000)
            bounded += checker.bounded
        # Some positions read a ledger whose bounds bind: the exact hypergeometric path.
        self.assertGreater(bounded, 0)

    def test_exact_division_rounds_as_python(self):
        # A ledger whose hypergeometric weights are large: a seat holding many unknown cards under a bound.
        game = position(4)
        for row in game.ledger.seats:
            row.known = [0]*5
            row.unknown = 0
        hands = [[0]*5, [3, 4, 2, 5, 3], [4, 2, 6, 1, 4], [2, 3, 3, 3, 2]]
        game._state.hands = hands
        game._state.bank = [19 - sum(h[r] for h in hands) for r in range(5)]
        for seat in (1, 2, 3):
            game.ledger.seats[seat].unknown = sum(hands[seat])
            game.ledger.seats[seat].most = [6, 6, 6, 6, 6]
        native = NativeState(game)
        for contract in CONTRACTS:
            for seat in range(4):
                self.assert_row(game, native, seat, contract)


if __name__ == "__main__":
    unittest.main()
