# SPDX-License-Identifier: GPL-3.0-only
"""Constructed native-lane protocols, without a deal or complete game loop."""
import copy
import dataclasses
import pickle
import random
import unittest
from unittest import mock

import numpy as np

from hexset.native import available, supported
if not available():
    raise unittest.SkipTest("optional native state extension is unavailable")

from hexset import contract9, decisions as D
from hexset.actions import Action, ActionType, apply, space_for
from hexset.arena import game_key
from hexset.chance import Recording, Scripted
from hexset.game import Phase
from hexset.gym.native import NativeAnswer, NativeBatch, NativeLane
from hexset.native import NativeState
from hexset.record import Tape
from .test_native_state import position


def lane(game, index=0, cap=100):
    return NativeLane.import_game(game, index=index, seed=0, tape=Tape(),
                                  action_cap=cap, cast=tuple(range(game.num_players)))


def batch(games, *, cap=100, keep=lambda completion: None, **kwargs):
    lanes = [lane(game, i, cap) for i, game in enumerate(games)]
    return NativeBatch(lanes, space=space_for(games[0]),
        pairs=contract9.incidences(games[0]._state.board.topology),
        contract="9d" if games[0].num_players == 2 else "9",
        planned_indices=range(len(games)+1), keep=keep, **kwargs)


def answer(group, request, kind=None, at=None):
    decision = request.decision
    blocks = ((D.DecisionKind.FLAT, np.flatnonzero(decision.flat)),
              (D.DecisionKind.SETUP, np.flatnonzero(decision.setup)),
              (D.DecisionKind.ROAD_BUILDING, range(len(decision.road_building))),
              (D.DecisionKind.DISCARD, range(len(decision.discard))))
    if at is None:
        kind, indices = next((k, values) for k, values in blocks
                             if len(values) and (kind is None or k == kind))
        at = int(indices[0])
    option = D.option_of(decision, kind, at, group.pairs, request.seat,
                         group.space, group.space.num_players)
    return NativeAnswer(option, (int(kind), int(at)))


class NativeBatchProtocols(unittest.TestCase):
    def test_visible_bytes_candidates_and_separate_labels(self):
        for players in (2, 4):
            for phase in (Phase.SETUP_SETTLEMENT, Phase.ROLL, Phase.MAIN, Phase.DISCARD):
                games = [position(players), position(players)]
                for game in games:
                    game.phase = phase
                    game.discard_quota = [2]+[0]*(players-1)
                group = batch(games)
                requests = group.requests()
                self.assertIs(requests, group.requests())
                targets = group.targets()
                for row, (request, game) in enumerate(zip(requests, games)):
                    contract = "9d" if players == 2 else "9"
                    expected = contract9.flatten(contract9.encode9(game, request.seat, contract))
                    self.assertEqual(request.observation.tobytes(), expected.tobytes())
                    decision = D.tensors(game, request.seat, group.space, double_turn=False)
                    for name in ("flat", "setup", "road_building", "discard"):
                        actual = getattr(request.decision, name)
                        self.assertEqual(actual.tobytes(), getattr(decision, name).tobytes())
                        self.assertFalse(actual.flags.writeable)
                        with self.assertRaises(ValueError):
                            actual.flags.writeable = True
                    self.assertEqual(request.ids, (row, row, 0, 0, 0, 0, 0))
                    self.assertFalse(request.observation.flags.writeable)
                    names = {f.name for f in dataclasses.fields(request)}
                    self.assertTrue(names.isdisjoint({"game", "targets", "points", "ledger", "state"}))
                    self.assertEqual(tuple(targets.points[row]), group.lanes[row].native.referee_points().true_points)
                self.assertFalse(targets.points.flags.writeable)
                self.assertFalse(targets.turns.flags.writeable)

    def test_all_answers_validate_before_any_state_or_tape_changes(self):
        group = batch([position(2), position(2)])
        asks = group.requests()
        before = [item.native.core.referee_snapshot() for item in group.lanes]
        answers = [answer(group, request) for request in asks]
        answers[1] = NativeAnswer(answers[1].action, (999, 0))
        with self.assertRaises(ValueError):
            group.advance(answers)
        self.assertEqual(before, [item.native.core.referee_snapshot() for item in group.lanes])
        self.assertTrue(all(not item.tape.actions for item in group.lanes))
        self.assertIs(group.requests(), asks)

    def test_compound_cap_callback_and_keep_happen_in_order(self):
        for players in (2, 4):
            game = position(players)
            game.phase = Phase.SETUP_SETTLEMENT
            calls = []
            group = None
            def decision(request, chosen):
                item = group.lanes[request.lane]
                self.assertEqual(len(item.tape.actions), 1)
                self.assertEqual(item.actions, 1)
                self.assertFalse(item.finished)
                calls.append(("decision", chosen))
            def keep(completion):
                self.assertFalse(group.lanes[0].finished)
                self.assertEqual(len(completion.record.actions), 1)
                self.assertTrue(completion.outcome.truncated)
                self.assertEqual(completion.cast, tuple(range(players)))
                calls.append(("keep", completion))
                return "retained"
            group = batch([game], cap=1, keep=keep, on_decision=decision)
            request = group.requests()[0]
            chosen = answer(group, request, D.DecisionKind.SETUP)
            self.assertEqual(group.advance([chosen]), ("retained",))
            self.assertEqual([call[0] for call in calls], ["decision", "keep"])
            self.assertIs(calls[0][1], chosen)
            self.assertFalse(group.running)
            self.assertTrue(group.lanes[0].finished)
            first = chosen.action.parts[0]
            self.assertEqual(group.lanes[0].tape.actions, [(int(first.type), first.a, first.b)])

    def test_keep_failure_poison_prevents_retirement_and_retry(self):
        def failed_keep(completion):
            raise OSError("retention failed")
        group = batch([position(4)], cap=1, keep=failed_keep)
        chosen = answer(group, group.requests()[0])
        with self.assertRaisesRegex(OSError, "retention"):
            group.advance([chosen])
        self.assertFalse(group.lanes[0].finished)
        self.assertEqual(group.lanes[0].actions, 1)
        for operation in (group.requests, lambda: group.advance([chosen])):
            with self.assertRaisesRegex(RuntimeError, "do not retry"):
                operation()
        self.assertEqual(group.lanes[0].actions, 1)

    def test_completed_setup_scalar_types_record_and_refill(self):
        for players in (2, 4):
            game = position(players)
            game.phase = Phase.SETUP_SETTLEMENT
            group = batch([game], cap=2)
            request = group.requests()[0]
            saved_bytes = request.observation.tobytes()
            chosen = answer(group, request, D.DecisionKind.SETUP)
            self.assertTrue(all(type(part.a) is int and type(part.b) is int for part in chosen.action.parts))
            expected = copy.deepcopy(game)
            tape = Tape()
            for part in chosen.action.parts:
                apply(expected, part)
                tape.step(part, ())
            (completion,) = group.advance([chosen])
            reference = tape.sealed(expected, seed=game_key(0, 0))
            self.assertEqual(pickle.dumps(completion.record), pickle.dumps(reference))
            replacement = lane(position(players), 1, 1)
            group.replace_lane(0, replacement)
            next_request = group.requests()[0]
            self.assertGreater(next_request.epoch, request.epoch)
            self.assertEqual(next_request.index, 1)
            self.assertEqual(request.observation.tobytes(), saved_bytes)
            self.assertTrue(group.running)
            with self.assertRaises(ValueError):
                group.replace_lane(0, lane(position(players), 2))

    def test_random_single_discard_consumes_original_rng_and_sparse_actor(self):
        game = position(4)
        game.phase, game.discard_quota = Phase.DISCARD, [0, 0, 1, 0]
        group = batch([game], cap=1, discard_order="random")
        stream = random.Random(f"discard-order:{game_key(0, 0)}")
        self.assertEqual(stream.choice((2,)), 2)
        request = group.requests()[0]
        self.assertEqual(request.seat, 2)
        self.assertEqual(group.lanes[0].discard_rng.getstate(), stream.getstate())
        chosen = answer(group, request, D.DecisionKind.DISCARD)
        (completion,) = group.advance([chosen])
        self.assertEqual(completion.record.actors, ())  # lowest owing seat is implicit

    def test_request_change_is_rejected_before_apply(self):
        group = batch([position(2)])
        request = group.requests()[0]
        chosen = answer(group, request)
        group.lanes[0].native.core.ledger_receive(0, 0, 1)
        with self.assertRaisesRegex(ValueError, "changed"):
            group.advance([chosen])
        self.assertFalse(group.lanes[0].tape.actions)

    def test_preflight_fallback_before_native_import(self):
        game = position(2)
        self.assertTrue(supported(game, "9d"))
        game.gates = (object(), None)
        self.assertFalse(supported(game, "9d"))
        game.gates = (None, None)
        with mock.patch("hexset.native.state.ArenaState", None):
            self.assertFalse(supported(game, "9d"))

    def test_owned_chance_failure_poisons_state_and_vetoes_retry(self):
        game = position(2)
        game.phase = Phase.ROLL
        game.chance = Recording(Scripted(()))
        native = NativeState(game)
        before = native.core.referee_snapshot()
        with self.assertRaises(Exception):
            native.apply(Action(ActionType.ROLL))
        self.assertEqual(native.core.referee_snapshot(), before)
        with self.assertRaisesRegex(RuntimeError, "do not retry"):
            native.apply(Action(ActionType.ROLL))


if __name__ == "__main__":
    unittest.main()
