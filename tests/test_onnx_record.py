# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.onnx_record` is the torch-free half of the information-set record: no
`pytest.importorskip("torch")` anywhere in this file, on purpose.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from hexset.actions import space_for
from hexset.board.board import random_base_board
from hexset.game import is_over, start
from hexset.onnx_record import RECORD_FIELDS, record_batch, record_from_game
from hexset.play import step_randomly


def a_game(players: int = 4, seed: int = 0, steps: int = 120):
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    for _ in range(steps):
        if is_over(game):
            break
        step_randomly(game, rng)
    return game


def test_onnx_record_module_is_torch_free():
    """Whether or not torch is installed on the box running this test."""
    import sys

    class _BlockTorch:
        def find_spec(self, name, path=None, target=None):
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch blocked for this check")
            return None

    blocker = _BlockTorch()
    sys.meta_path.insert(0, blocker)
    try:
        for name in list(sys.modules):
            if name == "hexset.onnx_record" or name.startswith("hexset.onnx_record."):
                del sys.modules[name]
        import hexset.onnx_record as reloaded

        game = a_game(seed=0, steps=20)
        space = space_for(game)
        row = reloaded.record_from_game(game, None, space)
        assert set(row) == set(RECORD_FIELDS)
    finally:
        sys.meta_path.remove(blocker)


def test_the_record_carries_the_ledger_in_board_seat_order():
    """`RecordEncoder` alone rotates and drops the perspective seat's row."""
    from hexset.ledger import SeatLedger

    game = a_game(seed=6, steps=60)
    game.ledger.seats[0] = SeatLedger(known=[1, 0, 0, 0, 0], unknown=2)
    game.ledger.seats[1] = SeatLedger(known=[0, 3, 0, 0, 1], unknown=0)
    space = space_for(game)

    row = record_from_game(game, 0, space)

    assert row["ledger_known"].shape == (game._state.num_players, 5)
    assert row["ledger_unknown"].shape == (game._state.num_players,)
    for seat in range(game._state.num_players):
        assert list(row["ledger_known"][seat]) == game.ledger.seats[seat].known
        assert int(row["ledger_unknown"][seat]) == game.ledger.seats[seat].unknown


def test_record_batch_stacks_each_positions_own_record():
    games = [a_game(seed=seed, steps=40) for seed in (0, 1)]
    space = space_for(games[0])
    batch = record_batch([(game, None) for game in games], space)
    assert set(batch) == set(RECORD_FIELDS)
    for row, game in enumerate(games):
        single = record_from_game(game, None, space)
        for name in RECORD_FIELDS:
            np.testing.assert_array_equal(batch[name][row], single[name])
    with pytest.raises(ValueError, match="at least one"):
        record_batch([], space)
