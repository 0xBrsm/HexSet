# SPDX-License-Identifier: GPL-3.0-only
"""A tree held natively searches exactly as the Python tree it stands in for."""
import hashlib
import os
import random
import unittest
from unittest import mock

from hexset.native import available
if not available():
    raise unittest.SkipTest("optional native state extension is unavailable")

import numpy as np

from hexset import contract9, decisions as D
from hexset.actions import ActionType, apply, build_space, legal_actions
from hexset.board.board import base_board
from hexset.game import Phase, imagine, is_over, start
from hexset.mcts import Search, _Chance
from hexset.native.playout import SWITCH, NativeCopy, held
from hexset.rules import DUEL_VARIANT_GAME, STANDARD_GAME


def played(players, seed, moves):
    """A game `moves` random moves past its opening, every move drawn from `seed`."""
    rng = random.Random(seed)
    game = start(base_board(rng=rng), players, rng,
                 game_type=DUEL_VARIANT_GAME if players == 2 else STANDARD_GAME)
    while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
        apply(game, rng.choice(legal_actions(game)))
    for _ in range(moves):
        if is_over(game):
            break
        apply(game, rng.choice(legal_actions(game)))
    return game


def buying(players, seed):
    """A game played on at random from `seed` to the first position where the
    seat to move may buy a development card."""
    rng = random.Random(seed)
    game = played(players, seed, 0)
    while not is_over(game):
        options = legal_actions(game)
        if any(option.type is ActionType.BUY_DEV_CARD for option in options):
            return game
        apply(game, rng.choice(options))
    return None


def space_of(game):
    topology = game._state.board.topology
    return build_space(topology.num_vertices, topology.num_edges, topology.num_hexes, game._state.num_players)


def _draw(*arrays):
    digest = hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)).digest()
    return int.from_bytes(digest[:8], "little")


class ProbeEvaluator:
    """A `hexset.mcts.Evaluator` for tests: each leaf's prior and value a pure
    function of its observation bytes and options, as a net's would be, so a
    Python and a native position that encode alike score alike."""

    def __init__(self, space, contract):
        self.space, self.native_contract = space, contract

    def evaluate(self, leaves):
        out = []
        for leaf in leaves:
            row = contract9.observation_row(leaf.game, leaf.seat, self.native_contract)
            weights = np.array([1 + _draw(row, np.array(tuple(option), dtype=np.int64)) % 97
                                for option in leaf.options], dtype=np.float64)
            frame = D.frame_for(leaf.game, leaf.seat, self.space)
            values = [(_draw(row, np.int64(k)) % 1000) / 1000.0 for k in range(len(frame.seats))]
            out.append((weights / weights.sum(), frame.to_board(values)))
        return out

    def terminal(self, game):
        seats = game.dealt if type(game) is NativeCopy else game._state.num_players
        return tuple(1.0 if seat == game.won_by else 0.0 for seat in range(seats))


def edges(root):
    """Every expanded edge of a tree, as `(kind, action type)` counts: `kind`
    is "roll", "draw" (a steal or a bought card) or "plain"."""
    counts, stack = {}, [root]
    while stack:
        node = stack.pop()
        for action, child in zip(node.options, node.children):
            if child is None:
                continue
            if isinstance(child, _Chance):
                kind = "roll" if action.type is ActionType.ROLL else "draw"
                stack.extend(child.outcomes.values())
            else:
                kind = "plain"
                stack.append(child)
            key = (kind, action.type.name)
            counts[key] = counts.get(key, 0) + 1
    return counts


def searched(game, *, native, seed, simulations=256):
    space = space_of(game)
    contract = "9d.1" if game._state.num_players == 2 else "9"
    search = Search(ProbeEvaluator(space, contract), simulations=simulations, wave=16,
                    rng=random.Random(seed))
    with mock.patch.dict(os.environ, {} if native else {SWITCH: "0"}):
        if native:
            os.environ.pop(SWITCH, None)
        root, options, visits = search.run(game)
    return search, root, options, visits


class NativeTree(unittest.TestCase):
    def test_a_native_deck_shuffles_as_a_python_copy(self):
        for players, seed in ((2, 7), (4, 8)):
            with self.subTest(players=players):
                game = played(players, seed, 20)
                python = imagine(game, random.Random(3), randomize_deck=True)
                native = held(game)
                native.shuffle_deck(random.Random(3))
                deck, flags = native.state.core.deck()
                self.assertEqual(list(deck), [int(card) for card in python._state.deck])
                with self.assertRaises(Exception):
                    native.state.core.set_deck([0] * len(deck), list(flags))

    def test_a_native_tree_searches_as_the_python_tree(self):
        seen = {}
        positions = [(players, seed, moves, played(players, seed, moves))
                     for players in (2, 4) for seed, moves in ((1, 0), (2, 40), (3, 90), (4, 160))]
        positions += [(players, seed, "buy", buying(players, seed)) for players in (2, 4) for seed in (5, 6)]
        for players, seed, moves, game in positions:
                if game is None or is_over(game):
                    continue
                with self.subTest(players=players, seed=seed, moves=moves):
                    py, py_root, py_options, py_visits = searched(game, native=False, seed=seed)
                    nat, nat_root, nat_options, nat_visits = searched(game, native=True, seed=seed)
                    self.assertIsNot(type(py_root.game), NativeCopy)
                    self.assertIs(type(nat_root.game), NativeCopy)
                    self.assertEqual(py_options, nat_options)
                    np.testing.assert_array_equal(py_visits, nat_visits)
                    np.testing.assert_array_equal(py_root.totals, nat_root.totals)
                    np.testing.assert_array_equal(py_root.ranked, nat_root.ranked)
                    self.assertEqual(py_root.value, nat_root.value)
                    # The same draws in the same order: the streams end alike.
                    self.assertEqual(py.rng.getstate(), nat.rng.getstate())
                    py_edges, nat_edges = edges(py_root), edges(nat_root)
                    self.assertEqual(py_edges, nat_edges)
                    for key, count in nat_edges.items():
                        seen[key] = seen.get(key, 0) + count
        # The positions reach every kind of edge the native tree handles itself.
        kinds = {key for key in seen}
        self.assertIn(("roll", "ROLL"), kinds)
        self.assertIn(("draw", "BUY_DEV_CARD"), kinds)
        self.assertTrue(any(kind == "draw" and name != "BUY_DEV_CARD" for kind, name in kinds),
                        f"no steal edge reached: {sorted(kinds)}")


class ProbePolicy:
    """A policy for `hexset.clients.netbot.LeafEvaluator`, scoring as
    `ProbeEvaluator` does: the mover's prior and value from `score_rows`, every
    other observer's value from `value_rows`, each read in its own seat frame."""

    def __init__(self, space, contract):
        self.space, self.native_contract = space, contract

    def _value(self, game, seat):
        row = contract9.observation_row(game, seat, self.native_contract)
        frame = D.frame_for(game, seat, self.space)
        return row, frame.to_board([(_draw(row, np.int64(k)) % 1000) / 1000.0 for k in range(len(frame.seats))])

    def score_rows(self, rows):
        out = []
        for game, seat, options in rows:
            row, value = self._value(game, seat)
            weights = np.array([1 + _draw(row, np.array(tuple(option), dtype=np.int64)) % 97
                                for option in options], dtype=np.float64)
            out.append((weights / weights.sum(), value))
        return out

    def value_rows(self, rows):
        return [self._value(game, observer)[1] for game, observer in rows]


class NativeDuelFourTree(unittest.TestCase):
    """At a duel dealt four with two seats retired, the leaf evaluator a net's
    MCTS seat uses searches a native tree exactly as the Python tree."""

    def test_a_native_duel_four_tree_searches_as_the_python_tree(self):
        from hexset.clients.netbot import LeafEvaluator

        from .test_native_duel_four import RETIRED, duel_four, two_seat_space

        searched_positions = 0
        for retired in RETIRED:
            for seed, moves in ((1, 0), (2, 40), (3, 120)):
                rng = random.Random(seed)
                game = duel_four(retired, seed)
                while game.phase in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD) or moves > 0:
                    if is_over(game):
                        break
                    if game.phase not in (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD):
                        moves -= 1
                    apply(game, rng.choice(legal_actions(game)))
                if is_over(game):
                    continue
                space = two_seat_space(game)
                results = []
                for native in (False, True):
                    search = Search(LeafEvaluator(ProbePolicy(space, "9d.1")), simulations=256, wave=16,
                                    rng=random.Random(seed))
                    with mock.patch.dict(os.environ, {} if native else {SWITCH: "0"}):
                        if native:
                            os.environ.pop(SWITCH, None)
                        results.append(search.run(game) + (search.rng.getstate(),))
                (py_root, py_options, py_visits, py_rng), (nat_root, nat_options, nat_visits, nat_rng) = results
                with self.subTest(retired=sorted(retired), seed=seed):
                    self.assertIsNot(type(py_root.game), NativeCopy)
                    self.assertIs(type(nat_root.game), NativeCopy)
                    self.assertEqual(py_options, nat_options)
                    np.testing.assert_array_equal(py_visits, nat_visits)
                    np.testing.assert_array_equal(py_root.totals, nat_root.totals)
                    np.testing.assert_array_equal(py_root.ranked, nat_root.ranked)
                    self.assertEqual(py_root.value, nat_root.value)
                    self.assertEqual(py_rng, nat_rng)
                    searched_positions += 1
        self.assertGreaterEqual(searched_positions, 6)


if __name__ == "__main__":
    unittest.main()
