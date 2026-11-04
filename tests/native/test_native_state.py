# SPDX-License-Identifier: GPL-3.0-only
"""Constructed states and ledger-event transcripts: no game loop."""
import copy
import dataclasses
import random
import unittest

from hexset.native import available
if not available():
    raise unittest.SkipTest("optional native state extension is unavailable")

from hexset._native_core import ArenaState
from .helpers import canonical_bytes, debug_state
from hexset.native.state import NativeState, capture_referee
from hexset.board.board import make_board
from hexset.board.coords import Hex
from hexset.board.terrain import Terrain
from hexset.board.topology import build
from hexset.cards import DevCard
from hexset.chance import Balanced, DiceDeck, Live, Recording, Scripted
from hexset.game import Game, Phase
from hexset.ledger import PublicLedger, SeatLedger
from hexset.rules import DUEL_VARIANT, STANDARD
from hexset.state import Building, GameState, HiddenRead, observed_by
from hexset.victory import victory_points


def position(players):
    """Direct construction, with no dealing, seed, arena or game loop."""
    topology = build((Hex(0, 0, 0), Hex(1, -1, 0)))
    board = make_board(topology, (Terrain.FOREST, Terrain.FIELDS), (5, 9))
    v, e = topology.num_vertices, topology.num_edges
    hands = [[1, 2, 0, 1, 0] for _ in range(players)]
    rules = DUEL_VARIANT if players == 2 else STANDARD
    state = GameState(board=board, num_players=players,
        vertex_owner=[0] + [-1]*(v-1), vertex_building=[Building.SETTLEMENT] + [Building.NONE]*(v-1),
        edge_owner=[-1]*e, robber=-1, hands=hands, bank=[19-players, 19-players*2, 19, 19-players, 19],
        deck=[DevCard.KNIGHT, 1, DevCard.MONOPOLY],
        dev_cards=[[0, 1, 0, 0, 0] for _ in range(players)],
        new_dev_cards=[[0, 0, 0, 0, 0] for _ in range(players)],
        knights_played=[0]*players, dev_cards_played=[0]*5,
        dev_ages=[None if seat%2 else [4] for seat in range(players)],
        road_lengths=[0]*players, longest_road_holder=0, largest_army_holder=1,
        rules=rules, dice=DiceDeck() if rules.balanced_dice else None)
    ledger = PublicLedger.new(players)
    for seat, hand in enumerate(hands):
        for resource, n in enumerate(hand): ledger.receive(seat, resource, n)
    ledger.steal(0, 1, 1)
    state.hands[0][1] += 1
    state.hands[1][1] -= 1
    chance = Recording(Scripted((("roll", 8), ("steal", 1))))
    return Game(state, random.Random(), chance, ledger, phase=Phase.MAIN,
                setup_queue=list(range(players)) + list(reversed(range(players))),
                discard_quota=[0]*players, last_roll=6, turns=3,
                before_roll=frozenset({DevCard.KNIGHT, DevCard.MONOPOLY}), gates=(None,)*players)


def ledger_values(ledger):
    def rows(value):
        return tuple((*row.known, row.unknown, *row.want, *row.waste, *row.most) for row in value.seats)
    return dict(common=rows(ledger), private=None if ledger.private is None else
                tuple(None if own is None else rows(own) for own in ledger.private))


class NativeOwnership(unittest.TestCase):
    def test_exact_debug_roundtrip_for_both_formats(self):
        for players in (2, 4):
            with self.subTest(players=players):
                game = position(players)
                native = NativeState(game)
                restored = native.restore_referee()
                self.assertEqual(debug_state(game), debug_state(restored))
                self.assertEqual(capture_referee(game), native.core.referee_snapshot())
                self.assertEqual(native.referee_points().true_points,
                                 tuple(victory_points(game._state, seat) for seat in range(players)))

    def test_no_native_python_snapshot_or_fork_aliases(self):
        game = position(4)
        payload = capture_referee(game)
        core = ArenaState(payload)
        original = core.referee_snapshot()
        payload["arrays"]["hands"] = (0,)*20
        payload["ledger"]["common"] = ()
        game._state.hands[0][0] += 7
        game.ledger.seats[0].known[0] += 7
        self.assertEqual(core.referee_snapshot(), original)
        exported = core.referee_snapshot()
        exported["arrays"]["hands"] = (0,)*20
        exported["ledger"]["private"] = None
        self.assertEqual(core.referee_snapshot(), original)
        child = core.clone()
        child.ledger_receive(0, 0, 1)
        self.assertEqual(core.referee_snapshot(), original)
        self.assertNotEqual(child.referee_snapshot()["ledger"], original["ledger"])

    def test_every_restore_and_chance_stream_is_independent(self):
        original = position(2)
        native = NativeState(original)
        fork = native.fork()
        restored = native.restore_referee()
        restored._state.hands[0][0] = 99
        restored.ledger.private[0].seats[0].known[0] = 99
        self.assertEqual(native.chance_event("roll", 0), 8)
        self.assertEqual(fork.chance_event("roll", 0), 8)
        self.assertEqual(original.chance.roll_by(0), 8)
        self.assertEqual(native.chance_event("steal", [0, 1, 0, 0, 0]), 1)
        self.assertEqual(native.restore_referee().chance.events, [("roll", 8), ("steal", 1)])
        self.assertEqual(fork.restore_referee().chance.events, [("roll", 8)])
        self.assertNotEqual(native.restore_referee()._state.hands[0][0], 99)
        self.assertNotEqual(native.restore_referee().ledger.private[0].seats[0].known[0], 99)

    def test_visible_cards_hide_opponent_composition_and_keep_own_identities(self):
        game = position(4)
        first = NativeState(game)
        own = first.visible_card_facts(0)
        changed = copy.deepcopy(game)
        changed._state.hands[2] = [0, 0, 0, 0, sum(changed._state.hands[2])]
        changed._state.dev_cards[2] = [1, 0, 0, 0, 0]
        changed._state.deck = list(reversed(changed._state.deck))
        second = NativeState(changed)
        self.assertEqual(own, second.visible_card_facts(0))
        self.assertEqual(own.own_hand, tuple(game._state.hands[0]))
        self.assertEqual(own.own_dev, tuple(game._state.dev_cards[0]))
        self.assertEqual(own.ledger, ledger_values(game.ledger.private[0])["common"])
        self.assertNotEqual(own.ledger, first.visible_card_facts(2).ledger)
        self.assertNotIn("true_points", {f.name for f in dataclasses.fields(own)})
        observed = copy.deepcopy(game)
        observed._state = observed_by(game._state, 0)
        with self.assertRaises(HiddenRead): NativeState(observed)

    def test_actual_live_and_balanced_rng_graphs_are_owned_and_preserved(self):
        for source_type in (Live, Balanced):
            for split in (False, True):
                with self.subTest(source=source_type.__name__, split=split):
                    game = position(2 if source_type is Balanced else 4)
                    game.chance = Recording(source_type(game.rng, split=split))
                    native = NativeState(game)
                    fork = native.fork()
                    restored = native.restore_referee()
                    self.assertIs(restored.rng, restored.chance.inner.rng)
                    self.assertIsNot(restored.rng, game.rng)
                    if split:
                        self.assertIsNot(restored.chance.inner.dice, restored.rng)
                        self.assertIsNot(restored.chance.inner.steals, restored.rng)
                    else:
                        self.assertIs(restored.chance.inner.dice, restored.rng)
                        self.assertIs(restored.chance.inner.steals, restored.rng)
                    # Chance-event unit calls only: no action, arena or game loop.
                    root_before = game.rng.getstate()
                    first = native.chance_event("roll", 0)
                    self.assertEqual(game.rng.getstate(), root_before)
                    self.assertEqual(first, fork.chance_event("roll", 0))
                    self.assertEqual(first, restored.chance.roll_by(0))
                    self.assertEqual(first, game.chance.roll_by(0))
                    self.assertEqual(native.restore_referee().chance.events,
                                     fork.restore_referee().chance.events)

    def test_unsupported_state_and_schema_drift_fail_before_native_use(self):
        game = position(4)
        game.gates = (object(), None, None, None)
        with self.assertRaisesRegex(ValueError, "gates"): NativeState(game)
        game.gates = None
        game.robber_promised = ((1, "spare"),)
        with self.assertRaisesRegex(ValueError, "deal"): NativeState(game)
        payload = capture_referee(position(2))
        payload["arrays"]["new_unknown_field"] = ()
        with self.assertRaisesRegex(ValueError, "arrays"): ArenaState(payload)


class NativeLedger(unittest.TestCase):
    def assert_ledger(self, core, oracle):
        self.assertEqual(core.referee_snapshot()["ledger"], ledger_values(oracle))
        for seat in range(len(oracle.seats)):
            self.assertEqual(core.visible_ledger(seat), ledger_values(oracle.for_seat(seat))["common"])

    def test_native_public_and_private_event_transcript_matches_original_after_each_event(self):
        for players in (2, 4):
            game = position(players)
            core = ArenaState(capture_referee(game))
            oracle = game.ledger.copy()
            events = [
                ("receive", (1, 2, 3)), ("spend", (0, 0, 1)), ("gain_unknown", (1, 2)),
                ("show", (1, [-1, 2, 0, 0, 0], False)), ("steal", (0, 1, 2)),
                ("steal", (1, 0, None)), ("emptied", (1, 2)),
                ("show", (0, [0, -2, 1, 0, 0], True)), ("receive", (0, 1, 0)),
                ("spend", (1, 0, -1)),
            ]
            if players == 4: events += [("steal", (2, 3, 1)), ("steal", (0, 2, 1))]
            for name, args in events:
                with self.subTest(players=players, event=name, args=args):
                    if name == "show":
                        getattr(oracle, name)(*args[:2], holds=args[2])
                    else: getattr(oracle, name)(*args)
                    getattr(core, "ledger_" + name)(*args)
                    self.assert_ledger(core, oracle)

    def test_settlement_forced_bounds_and_contradictions_match_source(self):
        fixtures = (
            SeatLedger(known=[1, 0, 0, 0, 0], unknown=3, most=[1, 1, 1, 1, 0]),
            SeatLedger(known=[2, 0, 0, 0, 0], unknown=4, most=[1, 0, 0, 0, 0]),
            SeatLedger(known=[0]*5, unknown=2, most=[0, 0, 2, 0, 0]),
        )
        for row in fixtures:
            game = position(2)
            game.ledger = PublicLedger([row.copy(), row.copy()])
            core = ArenaState(capture_referee(game))
            game.ledger.show(0, [0]*5, holds=False)
            core.ledger_show(0, [0]*5, False)
            self.assert_ledger(core, game.ledger)
            game.ledger.steal(1, 0, None)
            core.ledger_steal(1, 0, None)
            self.assert_ledger(core, game.ledger)

    def test_failed_operation_is_atomic_and_checked(self):
        core = ArenaState(capture_referee(position(4)))
        before = core.referee_snapshot()
        for operation in (lambda: core.ledger_receive(4, 0, 1),
                          lambda: core.ledger_spend(0, 5, 1),
                          lambda: core.ledger_steal(0, 1, 5),
                          lambda: core.ledger_show(0, [1, 2]),
                          lambda: core.ledger_receive(0, 0, 2**63-1)):
            with self.assertRaises((ValueError, OverflowError)): operation()
            self.assertEqual(core.referee_snapshot(), before)


if __name__ == "__main__":
    unittest.main()
