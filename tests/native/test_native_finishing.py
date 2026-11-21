# SPDX-License-Identifier: GPL-3.0-only
"""A native playout copy is finished exactly as the Python game it stands for."""
import random
import unittest

from hexset.native import available
if not available():
    raise unittest.SkipTest("optional native state extension is unavailable")

import numpy as np

from hexset import contract9
from hexset.actions import ActionType, apply, legal_actions
from hexset.board.board import base_board
from hexset.clients.finishing import Finishing, finished, inside
from hexset.game import Game, Phase, is_over, start, to_move
from hexset.native.playout import NativeCopy

CONTRACT = "9"


def _row(game, seat):
    if isinstance(game, Game):
        return contract9.flatten(contract9.encode9(game, seat, CONTRACT))
    return game.observation(seat, CONTRACT)


class LastOption:
    """Favours each row's last option, so a finish that read the options in
    another order would place another road; its value is a digest of the
    position's contract-9 row for seat 0."""

    has_prior = True

    def _value(self, game):
        row = _row(game, 0)
        return (float(np.dot(row, np.arange(len(row)) % 7)), 0.0, 0.0, 0.0)

    def score_rows(self, rows):
        out = []
        for game, _seat, options in rows:
            prior = np.zeros(len(options))
            prior[-1] = 1.0
            out.append((prior, self._value(game)))
        return out

    def value_rows(self, rows):
        return [self._value(game) for game, _seat in rows]


def _setup_road(seed):
    rng = random.Random(seed)
    game = start(base_board(rng=rng), 4, rng)
    apply(game, rng.choice([a for a in legal_actions(game) if a.type is ActionType.SETUP_SETTLEMENT]))
    assert game.phase is Phase.SETUP_ROAD
    return game


def _road_building(seed):
    """A game played on at random from `seed` until a Road Building card
    leaves two free roads owed, or `None`."""
    rng = random.Random(seed)
    game = start(base_board(rng=rng), 4, rng)
    for _ in range(3000):
        if is_over(game):
            return None
        options = legal_actions(game)
        card = [a for a in options if a.type is ActionType.PLAY_ROAD_BUILDING]
        apply(game, card[0] if card else rng.choice(options))
        if game.free_roads == 2 and inside(game):
            return game
    return None


class NativeFinishing(unittest.TestCase):
    def assert_same_finish(self, game):
        seat = to_move(game)
        native = NativeCopy(game)
        self.assertTrue(inside(native))
        done, native_done = finished(LastOption(), game), finished(LastOption(), native)
        self.assertFalse(inside(native_done))
        np.testing.assert_array_equal(_row(native_done, seat), _row(done, seat))
        self.assertTrue(inside(native))  # the copy asked about is untouched
        wrapped = Finishing(LastOption())
        self.assertEqual(wrapped.value_rows([(native, seat)]), wrapped.value_rows([(game, seat)]))
        options = tuple(legal_actions(game))
        ((_, value),) = wrapped.score_rows([(native, seat, native.legal())])
        ((_, expected),) = wrapped.score_rows([(game, seat, options)])
        self.assertEqual(value, expected)

    def test_a_setup_road_is_finished_as_the_python_game_finishes_it(self):
        for seed in range(4):
            with self.subTest(seed=seed):
                self.assert_same_finish(_setup_road(seed))

    def test_a_road_building_card_is_finished_as_the_python_game_finishes_it(self):
        games = [g for g in (_road_building(seed) for seed in range(40)) if g is not None][:3]
        self.assertTrue(games, "no seed reached a Road Building card")
        for game in games:
            with self.subTest(turns=game.turns):
                self.assert_same_finish(game)


if __name__ == "__main__":
    unittest.main()
