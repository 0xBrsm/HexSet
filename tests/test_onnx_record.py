# SPDX-License-Identifier: GPL-3.0-only
"""`hexset.onnx_record` is the torch-free half of the information-set record
-- these tests pin that it stays that way (no `pytest.importorskip("torch")`
anywhere in this file, on purpose) and check the record's own construction.

The traced encoder that reads this record (`hexnet.export_onnx.RecordEncoder`)
needs torch and lives in `tests/hexnet/test_export_onnx.py` instead.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from hexset.actions import space_for
from hexset.board.board import random_base_board
from hexset.game import is_over, start, to_move
from hexset.onnx_record import RECORD_FIELDS, record_from_game
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
    """`hexset.onnx_record` is the contract module a torch-free consumer (the
    gym, or any other information-set-only reader) builds a request from --
    it must import and run with torch absent, whether or not torch happens
    to be installed on the box actually running this test."""
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


def test_record_from_game_defaults_perspective_to_the_mover():
    game = a_game(seed=3, steps=50)
    space = space_for(game)
    default = record_from_game(game, None, space)
    explicit = record_from_game(game, to_move(game), space)
    for name in RECORD_FIELDS:
        assert np.array_equal(default[name], explicit[name])


def test_record_rejects_an_out_of_range_perspective():
    game = a_game(seed=4, steps=10)
    space = space_for(game)
    with pytest.raises(ValueError):
        record_from_game(game, 99, space)


def test_record_from_game_accepts_precomputed_options():
    """A caller that already enumerated its legal options (a gym step, a
    search leaf) can pass them through instead of paying for a second
    `legal_actions` walk -- and must get exactly the same record back."""
    from hexset.actions import legal_actions

    game = a_game(seed=5, steps=30)
    space = space_for(game)
    options = legal_actions(game)

    recomputed = record_from_game(game, None, space)
    passed_through = record_from_game(game, None, space, options)
    for name in RECORD_FIELDS:
        assert np.array_equal(recomputed[name], passed_through[name])


def test_the_record_carries_the_ledger_in_board_seat_order():
    """Unlike `offer_answered`, the ledger has no perspective-only filtering
    -- `known`/`unknown` are already the common-knowledge view, so the
    record carries every seat's entry in board-seat order (like `bank` or
    `hand_totals`) and `RecordEncoder` alone rotates and drops the
    perspective seat's own row."""
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
