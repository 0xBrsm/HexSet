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
from hexset.actions import legal_actions
from hexset.onnx_record import RECORD_FIELDS, record_batch, record_from_game, record_rows
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
    # The rows seat 0 knows: its own, where a steal showed it a card.
    rows = game.ledger.for_seat(0).seats
    rows[0] = SeatLedger(known=[1, 0, 0, 0, 0], unknown=2)
    rows[1] = SeatLedger(known=[0, 3, 0, 0, 1], unknown=0)
    space = space_for(game)

    row = record_from_game(game, 0, space)

    assert row["ledger_known"].shape == (game._state.num_players, 5)
    assert row["ledger_unknown"].shape == (game._state.num_players,)
    for seat in range(game._state.num_players):
        assert list(row["ledger_known"][seat]) == rows[seat].known
        assert int(row["ledger_unknown"][seat]) == rows[seat].unknown


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


def test_the_record_carries_roads_to_take_and_the_mover_in_board_seat_order():
    from hexset.game import to_move
    from hexset.roads import ROADS_TO_TAKE_CAP, roads_to_take_all

    game = a_game(seed=3, steps=300)
    space = space_for(game)
    for perspective in range(game._state.num_players):
        row = record_from_game(game, perspective, space)
        assert row["roads_to_take"].dtype == np.int64
        assert tuple(row["roads_to_take"]) == roads_to_take_all(game._state)
        assert all(0 <= n <= ROADS_TO_TAKE_CAP for n in row["roads_to_take"])
        assert row["to_move"].shape == () and int(row["to_move"]) == to_move(game)


_FLOAT_FIELDS = {"expected_hand", "dev_ages", "roll_odds"}


def test_every_field_has_its_declared_shape_and_dtype():
    from hexset.encoding import static_graph
    from hexset.onnx_record import record_shapes

    game = a_game(seed=4, steps=300)
    space = space_for(game)
    shapes = record_shapes(static_graph(game._state.board.topology), 4, space)
    row = record_from_game(game, None, space)
    assert set(shapes) == set(RECORD_FIELDS)
    for name in RECORD_FIELDS:
        assert row[name].shape == shapes[name], name
        want = np.bool_ if name == "action_mask" else np.float32 if name in _FLOAT_FIELDS else np.int64
        assert row[name].dtype == want, name


def test_the_contract_8_fields_read_the_rules_in_board_seat_order():
    from hexset.encoding import dev_age_rows, expected_hands, knights_needed_all
    from hexset.roads import reach_all
    from hexset.view import View

    game = a_game(seed=5, steps=300)
    space = space_for(game)
    state = game._state
    for perspective in range(state.num_players):
        row = record_from_game(game, perspective, space)
        view = View.from_game(game, perspective)
        unseen = view.unseen_dev_cards()
        assert row["reach"].tolist() == [list(r) for r in reach_all(state)]
        assert row["own_matured"].tolist() == list(state.dev_cards[perspective])
        assert int(row["dev_card_played"]) == int(game.dev_card_played)
        np.testing.assert_allclose(row["expected_hand"], expected_hands(view), rtol=1e-6)
        assert row["expected_hand"][perspective].tolist() == list(state.hands[perspective])
        np.testing.assert_array_equal(row["dev_ages"], dev_age_rows(state))
        assert tuple(row["knights_needed"]) == knights_needed_all(state, perspective, unseen[0])
        assert row["unseen_dev"].tolist() == list(unseen)


def test_the_contract_8_1_fields_state_the_tables_rules_and_the_next_rolls_odds():
    from hexset.game import roll_odds
    from hexset.rules import DUEL_VARIANT_GAME, STANDARD_GAME

    for game_type, players in ((STANDARD_GAME, 4), (DUEL_VARIANT_GAME, 2)):
        rng = random.Random(3)
        game = start(random_base_board(rng), players, rng, game_type=game_type)
        for _ in range(250):
            if is_over(game):
                break
            step_randomly(game, rng)
        rules = game_type.rules
        row = record_from_game(game, None, space_for(game))
        assert int(row["winning_points"]) == rules.winning_points
        assert int(row["discard_limit"]) == rules.discard_limit
        assert int(row["friendly_robber"]) == int(rules.friendly_robber)
        assert int(row["balanced_dice"]) == int(rules.balanced_dice)
        odds = dict(roll_odds(game))
        np.testing.assert_allclose(row["roll_odds"], [odds.get(t, 0.0) for t in range(2, 13)], rtol=1e-6)


def test_a_seats_observed_position_records_as_the_true_position_does():
    """A live client holds an observed state: every other seat's hand and
    cards hidden. The record from that seat must build there, and match the
    true position's record from the same seat."""
    from hexset.game import observe

    for seed in range(6):
        game = a_game(seed=seed, steps=150 + 11 * seed)
        if is_over(game):
            continue
        space = space_for(game)
        seat = seed % 4
        seen = record_from_game(observe(game, seat), seat, space, options=())
        true = record_from_game(game, seat, space, options=())
        for name in RECORD_FIELDS:
            np.testing.assert_array_equal(seen[name], true[name], err_msg=name)


def test_record_rows_builds_the_rows_record_from_game_builds_one_at_a_time():
    games = [a_game(seed=seed, steps=steps) for seed, steps in ((0, 40), (1, 90), (2, 150))]
    space = space_for(games[0])
    some = tuple(legal_actions(games[1])[:2])
    rows = [(games[0], None, None), (games[1], None, some), (games[2], 1, None), (games[0], 2, None)]
    for batch in (record_rows(rows, space),
                  record_rows(rows[:1] * 3, space),
                  record_rows(rows[1:2], space, [[space.index(a) for a in some]])):
        assert list(batch) == list(RECORD_FIELDS)
    batch = record_rows(rows, space)
    for i, (game, seat, options) in enumerate(rows):
        single = record_from_game(game, seat, space, options)
        for name in RECORD_FIELDS:
            assert batch[name][i].dtype == single[name].dtype, name
            np.testing.assert_array_equal(batch[name][i], single[name], err_msg=name)
    same = record_rows(rows[:1] * 3, space)
    for name in RECORD_FIELDS:
        np.testing.assert_array_equal(same[name], np.stack([record_from_game(games[0], None, space)[name]] * 3))
    indexed = record_rows(rows[1:2], space, [[space.index(a) for a in some]])
    np.testing.assert_array_equal(indexed["action_mask"][0], record_from_game(games[1], None, space, some)["action_mask"])
    with pytest.raises(ValueError, match="at least one"):
        record_rows([], space)


def test_a_records_board_fields_are_its_own():
    game = a_game(steps=10)
    space = space_for(game)
    first = record_from_game(game, None, space)
    first["terrain"][:] = -7
    np.testing.assert_array_equal(record_from_game(game, None, space)["terrain"],
                                  [int(t) for t in game._state.board.terrain])
