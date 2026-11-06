# SPDX-License-Identifier: GPL-3.0-only
"""A two-seat (contract 9d) policy at a table dealt four with two seats
retired: native playout copies state its observation, decisions and seat
frame exactly as the Python engine does, at positions from played games."""
import os
import random
import unittest
from unittest import mock

from hexset.native import available
if not available():
    raise unittest.SkipTest("optional native state extension is unavailable")

import numpy as np

from hexset import contract9, decisions as D, rollout
from hexset.actions import ActionType, apply, build_space, legal_actions
from hexset.board.board import base_board
from hexset.game import Phase, is_over, start, to_move
from hexset.native.playout import SWITCH, NativeCopy, copyable, held
from hexset.native.state import NativeState
from hexset.rollout import RolloutBatch, paired_copy
from hexset.rules import DUEL_VARIANT_GAME, STANDARD_GAME

from .test_native_playout import Probe

DUEL = contract9.DUEL_CONTRACTS
#: Which seats are retired at each table played: the live pair adjacent,
#: apart, and with seat 0 retired.
RETIRED = (frozenset({1, 3}), frozenset({2, 3}), frozenset({0, 2}))
#: Build and card moves first, so a game reaches dev cards, robbers and its end.
_EAGER = (ActionType.BUILD_CITY, ActionType.BUILD_SETTLEMENT, ActionType.PLAY_KNIGHT,
          ActionType.PLAY_ROAD_BUILDING, ActionType.PLAY_MONOPOLY, ActionType.PLAY_YEAR_OF_PLENTY,
          ActionType.BUY_DEV_CARD, ActionType.BUILD_ROAD)


def duel_four(retired, seed, first=0):
    """A duel-variant game dealt four seats with `retired` retired."""
    rng = random.Random(seed)
    return start(base_board(rng=rng), 4, rng, game_type=DUEL_VARIANT_GAME, locked=retired, first=first)


def two_seat_space(game):
    topology = game._state.board.topology
    return build_space(topology.num_vertices, topology.num_edges, topology.num_hexes, 2)


def choose(legal, rng):
    """A move that plays a game through: an eager move most of the time."""
    eager = [a for a in legal if a.type in _EAGER]
    if eager and rng.random() < 0.85:
        return rng.choice(eager)
    return rng.choice(legal)


class DuelFourParity(unittest.TestCase):
    def assert_same(self, native, python, space, seen):
        live = [s for s in range(4) if s not in python.locked]
        for seat in live:
            self.assertEqual(native.frame(seat, 2), D.frame_for(python, seat, space))
            for contract in DUEL:
                a = contract9.observation_row(native, seat, contract)
                b = contract9.observation_row(python, seat, contract)
                self.assertEqual(a.shape, b.shape)
                if a.tobytes() != b.tobytes():
                    at = np.flatnonzero(a.view(np.uint32) != b.view(np.uint32))[:8]
                    self.fail(f"{contract} seat {seat} {python.phase}: "
                              f"{[(int(i), float(a[i]), float(b[i])) for i in at]}")
        seats = live if python.phase is Phase.DISCARD else [to_move(python)]
        for seat in seats:
            if python.phase is Phase.DISCARD and not python.discard_quota[seat]:
                continue
            a = D.decision_at(native, seat, space, double_turn=False)
            b = D.decision_at(python, seat, space, double_turn=False)
            self.assertEqual(len(a.flat), space.size)
            for name in ("flat", "setup", "road_building", "discard"):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=f"{name} {python.phase}")
            self.assertEqual(a.double, b.double)
            if python.phase is Phase.ROBBER:
                seen.add("victim" if any(m.b < 4 for m in legal_actions(python)) else "no victim")
        seen.add(python.phase)

    def test_native_copies_state_every_read_as_the_python_engine(self):
        seen, played = set(), set()
        for g, retired in enumerate(RETIRED):
            for seed in (1, 2):
                with self.subTest(retired=sorted(retired), seed=seed):
                    game = duel_four(retired, 100 * g + seed, first=seed % 4)
                    space = two_seat_space(game)
                    seat = to_move(game)
                    python = paired_copy(game, 7 + seed, seat)
                    native = held(paired_copy(game, 7 + seed, seat))
                    self.assertIsInstance(native, NativeCopy)
                    rng = random.Random(seed)
                    for _ in range(4000):
                        self.assertEqual((native.phase, native.turns, native.won_by),
                                         (python.phase, python.turns, python.won_by))
                        self.assert_same(native, python, space, seen)
                        if is_over(python):
                            break
                        mover = to_move(python)
                        self.assertNotIn(mover, retired)
                        self.assertEqual(native.to_move(), mover)
                        legal = tuple(legal_actions(python))
                        self.assertEqual(native.legal(), legal)
                        self.assertEqual(native.inside(), D.inside(python))
                        action = choose(legal, rng)
                        played.add(action.type)
                        apply(python, action)
                        native.apply(action)
                    self.assertTrue(is_over(python), "a game played through to its end")
        self.assertLessEqual({Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD, Phase.ROLL, Phase.MAIN,
                              Phase.DISCARD, Phase.ROBBER, Phase.GAME_OVER, "victim", "no victim"}, seen)
        self.assertLessEqual({ActionType.BUY_DEV_CARD, ActionType.PLAY_KNIGHT, ActionType.PLAY_ROAD_BUILDING,
                              ActionType.BUILD_CITY, ActionType.DISCARD, ActionType.MOVE_ROBBER}, played)

    def test_a_batch_held_natively_plays_out_as_the_python_one(self):
        for retired in RETIRED:
            for contract in DUEL:
                with self.subTest(retired=sorted(retired), contract=contract):
                    game = duel_four(retired, 31)
                    rng = random.Random(5)
                    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD) or game.turns < 6:
                        apply(game, choose(tuple(legal_actions(game)), rng))
                    space = two_seat_space(game)
                    policy = Probe(space, contract)
                    seat = to_move(game)
                    moves = tuple(D.options(game, seat, double_turn=False))[:3]
                    tasks = [(game, 19), (game, 29), (game, 39)]
                    real = rollout.copyable

                    def lanes(native):
                        with mock.patch.object(rollout, "copyable", lambda *a: native and real(*a)):
                            batch = RolloutBatch.from_tasks(policy, tasks, moves, seat, game.turns, (2, 4))
                            return batch.advance(8), {type(g) for g in batch.games}

                    (python, kinds_p), (native, kinds_n) = lanes(False), lanes(True)
                    self.assertEqual(kinds_p, {type(game)})
                    self.assertEqual(kinds_n, {NativeCopy})
                    np.testing.assert_array_equal(native.outcomes, python.outcomes)
                    np.testing.assert_array_equal(native.after, python.after)
                    np.testing.assert_array_equal(native.reads, python.reads)
                    self.assertEqual((native.steps, native.row_steps, native.truncated),
                                     (python.steps, python.row_steps, python.truncated))


class DuelFourShape(unittest.TestCase):
    def test_copyable_opens_only_a_two_seat_policy_at_two_live_seats(self):
        two = duel_four(frozenset({1, 3}), 3)
        space2 = two_seat_space(two)
        space4 = build_space(space2.num_vertices, space2.num_edges, space2.num_hexes, 4)
        with mock.patch.dict(os.environ, {SWITCH: "1"}):
            for contract in DUEL:
                self.assertTrue(copyable(two, contract, space2))
                self.assertFalse(copyable(two, contract, space4))
            self.assertFalse(copyable(two, "9", space2))
            self.assertTrue(copyable(two, "9.1", space4))
            rng = random.Random(4)
            three = start(base_board(rng=rng), 4, rng, game_type=STANDARD_GAME, locked=frozenset({2}))
            self.assertFalse(copyable(three, "9d", space2))
            full = start(base_board(rng=rng), 4, rng, game_type=STANDARD_GAME)
            self.assertFalse(copyable(full, "9d", space2))
            self.assertTrue(copyable(full, "9", space4))

    def test_the_switch_keeps_every_copy_python(self):
        game = duel_four(frozenset({1, 3}), 3)
        space = two_seat_space(game)
        with mock.patch.dict(os.environ, {SWITCH: "0"}):
            self.assertFalse(copyable(game, "9d.1", space))
            policy = Probe(space, "9d.1")
            seat = to_move(game)
            moves = tuple(D.options(game, seat, double_turn=False))[:2]
            batch = RolloutBatch.from_tasks(policy, [(game, 19)], moves, seat, game.turns, ())
            self.assertEqual({type(g) for g in batch.games}, {type(game)})
        with mock.patch.dict(os.environ, {SWITCH: "1"}):
            self.assertTrue(copyable(game, "9d.1", space))
        os.environ.pop(SWITCH, None)
        self.assertTrue(copyable(game, "9d.1", space))

    def test_a_retired_perspective_and_other_counts_are_refused(self):
        game = duel_four(frozenset({1, 3}), 3)
        native = held(paired_copy(game, 3, 0))
        for seat in (1, 3):
            with self.assertRaises(ValueError):
                native.observation(seat, "9d")
            with self.assertRaises(ValueError):
                native.frame(seat, 2)
            with self.assertRaises(ValueError):
                native.state.core.decision(seat, 2)
        with self.assertRaises(ValueError):
            native.frame(0, 3)
        with self.assertRaises(ValueError):
            native.state.core.decision(0, 3)
        self.assertEqual(native.frame(2, 2).seats, (2, 0))
        self.assertEqual(native.frame(2, 4).seats, (2, 3, 0, 1))

    def test_request_batches_keep_contract_9d_at_two_seat_tables(self):
        from hexset._native_core import RequestBatch
        game = duel_four(frozenset({1, 3}), 3)
        with self.assertRaises(ValueError):
            RequestBatch([NativeState(game).core], contract="9d")


if __name__ == "__main__":
    unittest.main()
