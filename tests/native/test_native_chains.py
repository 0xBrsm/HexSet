# SPDX-License-Identifier: GPL-3.0-only
"""Constructed action transcripts, never complete games or timing loops."""
import dataclasses
import random
import unittest

from hexset.native import available
if not available():
    raise unittest.SkipTest("optional native state extension is unavailable")

from hexset.native import encode9
from hexset.native.state import GAME_COLUMNS, NativeState, capture_referee
from .test_native_state import position
from hexset import contract9
from hexset.actions import Action, ActionType, apply, legal_actions
from hexset.cards import DevCard
from hexset.chance import Balanced, DiceDeck, Recording
from hexset.game import Phase
from hexset.ledger import PublicLedger
from hexset.state import Building


class NativeActionChains(unittest.TestCase):
    def assert_state(self, game, native):
        actual = native.core.referee_snapshot()
        expected = capture_referee(game)
        expected["epoch"] = actual["epoch"]
        self.assertEqual(actual, expected)
        self.assertEqual(native._chance.events, game.chance.events)
        for contract in contract9.CONTRACTS:
            if contract in contract9.DUEL_CONTRACTS and game.num_players != 2:
                continue
            for seat in range(game.num_players):
                expected_row = contract9.flatten(contract9.encode9(game, seat, contract))
                self.assertEqual(encode9(native, seat, contract).tobytes(), expected_row.tobytes())

    def condition(self, game, native, *, phase, actor, clear_robber=False):
        """Condition the next isolated action, preserving each side's OWN state.

        These transcripts deliberately do not play intervening turns/discards.
        Only named control fields change; no oracle state is copied into the
        native implementation, and its owned Chance object is never rebuilt.
        """
        game.phase, game.current_player = phase, actor
        game.discard_quota = [0]*game.num_players
        game.free_roads, game.resume_phase = 0, Phase.MAIN
        snapshot = native.core.referee_snapshot()
        arrays = snapshot["arrays"]
        control = list(arrays["game_control"])
        for name, value in (("phase", phase), ("current_player", actor),
                            ("free_roads", 0), ("resume_phase", Phase.MAIN)):
            control[GAME_COLUMNS.index(name)] = int(value)
        arrays["game_control"] = tuple(control)
        arrays["discard_quota"] = (0,)*game.num_players
        if clear_robber:
            game._state.robber = -1
            state_control = list(arrays["state_control"])
            state_control[0] = -1
            arrays["state_control"] = tuple(state_control)
        native.core.__init__(snapshot)
        self.assert_state(game, native)

    def step(self, game, native, action):
        apply(game, action)
        native.apply(action)
        self.assert_state(game, native)

    def assert_legal(self, game, native):
        expected = tuple((int(action.type), int(action.a), int(action.b))
                         for action in legal_actions(game))
        self.assertEqual(tuple(native.core.legal_actions()), expected)

    def test_actual_owned_balanced_roll_chain_crosses_refill_with_interleaved_steals(self):
        for players in (2, 4):
            for split in (False, True):
                with self.subTest(players=players, split=split):
                    game = position(players)
                    game._state.rules = dataclasses.replace(game._state.rules,
                        balanced_dice=True, friendly_robber=False, winning_points=100)
                    game._state.dice = DiceDeck()
                    game._state.hands = [[3]*5 for _ in range(players)]
                    game._state.bank = [19-3*players]*5
                    game.ledger = PublicLedger.new(players)
                    for seat, hand in enumerate(game._state.hands):
                        for resource, count in enumerate(hand):
                            game.ledger.receive(seat, resource, count)
                    # Both robber participants have buildings on hex0. Their
                    # separation is legal; no search or game deal is involved.
                    topology = game._state.board.topology
                    other = next(vertex for vertex in topology.hex_vertices[0]
                        if vertex != 0 and vertex not in topology.vertex_neighbors[0])
                    game._state.vertex_owner[other] = 1
                    game._state.vertex_building[other] = Building.SETTLEMENT
                    game._state.robber = -1
                    game.rng = random.Random(0)
                    game.chance = Recording(Balanced(game.rng, split=split))
                    native = NativeState(game)
                    self.assertIsNot(native._chance.inner, game.chance.inner)
                    self.assertIsNot(native._chance.inner.dice, game.chance.inner.dice)
                    self.assertIs(native._chance.inner.rng, native._rng)
                    rolls = 0
                    for roll_number in range(1, 29):
                        self.condition(game, native, phase=Phase.ROLL, actor=(roll_number-1) % players)
                        self.step(game, native, Action(ActionType.ROLL))
                        rolls += 1
                        self.assertEqual(native._chance.inner.cards, game.chance.inner.cards)
                        self.assertEqual(native._chance.inner.left, game.chance.inner.left)
                        self.assertEqual(native._chance.inner.sevens, game.chance.inner.sevens)
                        self.assertEqual(native._chance.inner.run, game.chance.inner.run)
                        self.assertEqual(native._chance.inner.dice.getstate(), game.chance.inner.dice.getstate())
                        self.assertEqual(native._rng.getstate(), game.rng.getstate())
                        if roll_number == 24:
                            # The public projection refills after this draw;
                            # Balanced refills immediately before the next.
                            self.assertEqual(game._state.dice.cards, 36)
                            self.assertEqual(game.chance.inner.cards, 12)
                        if roll_number == 25:
                            self.assertEqual(game._state.dice.cards, 35)
                            self.assertEqual(game.chance.inner.cards, 35)
                        if roll_number in (2, 7, 13, 18, 22, 26):
                            self.condition(game, native, phase=Phase.ROBBER, actor=0, clear_robber=True)
                            self.step(game, native, Action(ActionType.MOVE_ROBBER, 0, 1))
                            self.assertEqual(native._chance.inner.steals.getstate(), game.chance.inner.steals.getstate())
                    self.assertEqual(rolls, 28)
                    self.assertEqual([kind for kind, _ in game.chance.events].count("roll"), 28)
                    self.assertEqual([kind for kind, _ in game.chance.events].count("steal"), 6)
                    self.assertIsNotNone(native.core.referee_snapshot()["ledger"]["private"])

    def test_last_card_purchase_maturation_and_youngest_play_preserve_private_ledgers(self):
        for players in (2, 4):
            for held_old in (False, True):
                with self.subTest(players=players, held_old=held_old):
                    game = position(players)
                    game._state.rules = dataclasses.replace(game._state.rules, winning_points=100)
                    game._state.hands[0] = [6]*5
                    game._state.deck = [DevCard.MONOPOLY]
                    game._state.dev_cards[0] = [0, 0, 0, 0, int(held_old)]
                    game._state.new_dev_cards[0] = [0]*5
                    game._state.dev_ages[0] = [5] if held_old else []
                    if players == 4:
                        game.ledger.steal(2, 3, 1)
                        game._state.hands[3][1] -= 1
                        game._state.hands[2][1] += 1
                    native = NativeState(game)
                    self.step(game, native, Action(ActionType.BUY_DEV_CARD))
                    self.assertEqual(game._state.deck, [])
                    self.assertEqual(game._state.new_dev_cards[0][DevCard.MONOPOLY], 1)
                    self.assert_legal(game, native)
                    monop = [a for a in legal_actions(game) if a.type is ActionType.PLAY_MONOPOLY]
                    self.assertEqual(bool(monop), held_old)
                    self.assertFalse(any(a.type is ActionType.BUY_DEV_CARD for a in legal_actions(game)))
                    self.step(game, native, Action(ActionType.END_TURN))
                    self.assertEqual(game._state.new_dev_cards[0][DevCard.MONOPOLY], 0)
                    self.assertEqual(game._state.dev_ages[0], [6, 1] if held_old else [1])
                    self.condition(game, native, phase=Phase.MAIN, actor=0)
                    self.assert_legal(game, native)
                    self.step(game, native, Action(ActionType.PLAY_MONOPOLY, 1))
                    self.assertEqual(game._state.dev_ages[0], [6] if held_old else [])
                    self.assertEqual(game._state.dev_cards[0][DevCard.MONOPOLY], int(held_old))
                    for victim in range(1, players):
                        self.assertEqual(game._state.hands[victim][1], 0)
                        for perspective in range(players):
                            self.assertEqual(game.ledger.for_seat(perspective).seats[victim].most[1], 0)
                    self.assertEqual(game.chance.events, [])


if __name__ == "__main__":
    unittest.main()
