# SPDX-License-Identifier: GPL-3.0-only
"""Native playout copies answer exactly as the Python copies they stand in for."""
import hashlib
import random
import unittest
from unittest import mock

from hexset.native import available
if not available():
    raise unittest.SkipTest("optional native state extension is unavailable")

import numpy as np

from hexset import contract9, decisions as D, rollout
from hexset.actions import apply, build_space, legal_actions
from hexset.board.board import base_board
from hexset.game import Phase, is_over, start, to_move
from hexset.native.playout import NativeCopy, held
from hexset.rollout import RolloutBatch, paired_copy
from hexset.rules import DUEL_VARIANT_GAME, STANDARD_GAME


def opened(players, seed):
    """A game past its opening, every setup move drawn from `seed`."""
    rng = random.Random(seed)
    game = start(base_board(rng=rng), players, rng,
                 game_type=DUEL_VARIANT_GAME if players == 2 else STANDARD_GAME)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, rng.choice(legal_actions(game)))
    return game


def space_of(game):
    topology = game._state.board.topology
    return build_space(topology.num_vertices, topology.num_edges, topology.num_hexes, game._state.num_players)


def contract_of(players):
    return "9d.1" if players == 2 else "9"


class Probe:
    """A contract-9 policy for tests: each choice and value a pure function of
    the row's observation bytes and decision masks, as a net's would be."""

    def __init__(self, space, contract):
        self.space, self.native_contract = space, contract

    @staticmethod
    def _draw(*arrays):
        digest = hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)).digest()
        return int.from_bytes(digest[:8], "little")

    def act_rows(self, rows):
        out = []
        for game, seat, options in rows:
            row = contract9.observation_row(game, seat, self.native_contract)
            decision = D.decision_at(game, seat, self.space, double_turn=False)
            out.append(options[self._draw(row, decision.flat, decision.setup, decision.road_building,
                                          decision.discard) % len(options)])
        return out

    def value_rows(self, rows):
        out = []
        for game, seat in rows:
            row = contract9.observation_row(game, seat, self.native_contract)
            frame = D.frame_for(game, seat, self.space)
            players = len(frame.seats)
            values = [((self._draw(row, np.int64(k)) % 1000) / 1000.0) for k in range(players)]
            out.append(frame.to_board(values))
        return out


class NativePlayout(unittest.TestCase):
    def test_a_copy_answers_every_read_as_its_python_twin(self):
        for players in (2, 4):
            for seed in (1, 2):
                with self.subTest(players=players, seed=seed):
                    game = opened(players, seed)
                    space = space_of(game)
                    python = paired_copy(game, 11, 0)
                    native = held(paired_copy(game, 11, 0))
                    self.assertIsInstance(native, NativeCopy)
                    contract, rng = contract_of(players), random.Random(seed)
                    for _ in range(600):
                        self.assertEqual((native.phase, native.turns, native.won_by),
                                         (python.phase, python.turns, python.won_by))
                        if is_over(python):
                            break
                        seat = to_move(python)
                        self.assertEqual(native.to_move(), seat)
                        legal = tuple(legal_actions(python))
                        self.assertEqual(native.legal(), legal)
                        self.assertEqual(native.inside(), D.inside(python))
                        self.assertEqual(contract9.observation_row(native, seat, contract).tobytes(),
                                         contract9.observation_row(python, seat, contract).tobytes())
                        a = D.decision_at(native, seat, space, double_turn=False)
                        b = D.decision_at(python, seat, space, double_turn=False)
                        for name in ("flat", "setup", "road_building", "discard"):
                            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
                        self.assertEqual(a.double, b.double)
                        action = rng.choice(legal)
                        apply(python, action)
                        native.apply(action)

    def test_a_fork_plays_on_as_its_parent_would(self):
        game = opened(2, 3)
        base = held(paired_copy(game, 5, 0))
        first, second = base.fork(), base.fork()
        rng_a, rng_b = random.Random(9), random.Random(9)
        for _ in range(200):
            if first.over():
                break
            self.assertEqual(first.legal(), second.legal())
            first.apply(rng_a.choice(first.legal()))
            second.apply(rng_b.choice(second.legal()))
            self.assertEqual(first._control, second._control)

    def test_a_batch_held_natively_plays_out_as_the_python_one(self):
        for players in (2, 4):
            with self.subTest(players=players):
                game = opened(players, 4)
                space = space_of(game)
                policy = Probe(space, contract_of(players))
                seat = to_move(game)
                moves = tuple(legal_actions(game))[:3]
                tasks = [(game, 19), (game, 29), (game, 39)]

                def lanes(native):
                    with mock.patch.object(rollout, "copyable", lambda *a: native and real(*a)):
                        batch = RolloutBatch.from_tasks(policy, tasks, moves, seat, game.turns, (2, 4))
                        kinds = {type(g) for g in batch.games}
                        return batch.advance(6), kinds

                real = rollout.copyable
                (python, kinds_p), (native, kinds_n) = lanes(False), lanes(True)
                self.assertEqual(kinds_p, {type(game)})
                self.assertEqual(kinds_n, {NativeCopy})
                np.testing.assert_array_equal(native.outcomes, python.outcomes)
                np.testing.assert_array_equal(native.after, python.after)
                np.testing.assert_array_equal(native.reads, python.reads)
                self.assertEqual((native.steps, native.row_steps, native.mark_row_steps, native.truncated),
                                 (python.steps, python.row_steps, python.mark_row_steps, python.truncated))


if __name__ == "__main__":
    unittest.main()


def discarding(players):
    """A game past its opening with the seat to roll holding ten cards, two
    of each, and owing half of them to a seven; nobody else owing."""
    from hexset.board.terrain import Resource
    from hexset.game import roll_dice

    game = opened(players, 7)
    state = game._state
    seat = game.current_player
    for s in range(players):
        for r, n in enumerate(state.hands[s]):
            state.bank[r] += n
            state.hands[s][r] = 0
    for r in Resource:
        state.bank[r] -= 2
        state.hands[seat][r] += 2
    game.phase = Phase.ROLL
    roll_dice(game, 7)
    assert game.phase is Phase.DISCARD and game.discard_quota[seat] == 5
    return game


class WholeCandidates(unittest.TestCase):
    def test_whole_decisions_applied_natively_play_out_as_the_python_ones(self):
        """Candidates that are whole decisions -- setup turns and a discard,
        their parts applied in order -- held natively, as their Python copies."""
        for players in (2, 4):
            for name, game in (("setup", start(base_board(rng=random.Random(players)), players,
                                               random.Random(players),
                                               game_type=DUEL_VARIANT_GAME if players == 2 else STANDARD_GAME)),
                               ("discard", discarding(players))):
                with self.subTest(players=players, kind=name):
                    space = space_of(game)
                    policy = Probe(space, contract_of(players))
                    seat = to_move(game)
                    moves = tuple(D.options(game, seat, double_turn=False))[:3]
                    self.assertTrue(all(isinstance(m, D.Compound) for m in moves))
                    tasks = [(game, 19), (game, 29)]

                    def lanes(native):
                        with mock.patch.object(rollout, "copyable", lambda *a: native and real(*a)):
                            batch = RolloutBatch.from_tasks(policy, tasks, moves, seat, game.turns, (2, 4))
                            return batch.advance(6), {type(g) for g in batch.games}

                    real = rollout.copyable
                    (python, kinds_p), (native, kinds_n) = lanes(False), lanes(True)
                    self.assertEqual(kinds_p, {type(game)})
                    self.assertEqual(kinds_n, {NativeCopy})
                    np.testing.assert_array_equal(native.outcomes, python.outcomes)
                    np.testing.assert_array_equal(native.after, python.after)
