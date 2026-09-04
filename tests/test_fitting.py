# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random

import pytest

from hexset.dataset import Sample, base_rate, build, split_by_game
from hexset.bots.evaluate import TERM_NAMES
from hexset.fitting import (
    accuracy,
    fit,
    log_loss,
    scaling_of,
    to_weights,
)
from hexset.record import record_game
from hexset.board.board import random_base_board
from hexset.arena import PRESETS, spawn


def some_records(count: int = 6, bot: str = "greedy"):
    out = []
    for seed in range(count):
        board = random_base_board(random.Random(1000 + seed))
        bots = [
            spawn(PRESETS[bot], board, random.Random(seed * 16 + seat))
            for seat in range(4)
        ]
        out.append(record_game(bots, board, seed))
    return out


def a_sample(features, won, game=0):
    return Sample(features=features, won=won, game=game, seat=0, progress=0.5)


def test_samples_carry_one_row_per_seat_and_a_label():
    records = some_records(2)
    samples = build(records, stride=40)
    assert samples
    assert all(len(s.features) == len(TERM_NAMES) for s in samples)
    assert {s.won for s in samples} <= {0, 1}
    assert {s.seat for s in samples} == {0, 1, 2, 3}
    # Exactly one seat wins each game, so a quarter of rows are positive.
    assert base_rate(samples) == pytest.approx(0.25, abs=0.08)


def test_undecided_games_yield_nothing_because_they_have_no_label():
    board = random_base_board(random.Random(0))
    bots = [spawn(PRESETS["random"], board, random.Random(s)) for s in range(4)]
    unfinished = record_game(bots, board, 0, action_cap=12)
    assert unfinished.winner is None
    assert build([unfinished]) == []


def test_splitting_is_by_game_so_no_game_lands_on_both_sides():
    samples = build(some_records(6), stride=40)
    train, test = split_by_game(samples, holdout=0.34, seed=1)
    assert train and test
    assert not ({s.game for s in train} & {s.game for s in test})
    assert len(train) + len(test) == len(samples)


def test_scaling_survives_a_constant_column():
    scale = scaling_of([(1.0, 5.0), (3.0, 5.0)])
    assert scale.means == (2.0, 5.0)
    assert scale.sigmas[1] == 1.0


def test_the_fit_separates_a_linearly_separable_problem():
    rows = [a_sample((float(i), 0.0, 0, 0, 0, 0, 0, 0, 0), 1 if i > 5 else 0) for i in range(12)]
    result = fit(rows, epochs=300, rate=1.0)
    assert result.coefficients[0] > 0
    assert accuracy(
        [s.features for s in rows],
        [s.won for s in rows],
        result.coefficients,
        result.intercept,
    ) == 1.0


def test_the_fit_reduces_the_loss_it_started_from():
    samples = build(some_records(4), stride=20)
    rows = [s.features for s in samples]
    labels = [s.won for s in samples]
    zero = log_loss(rows, labels, [0.0] * len(TERM_NAMES), 0.0)
    result = fit(samples, epochs=200)
    assert result.train_loss < zero


def test_fitting_needs_samples():
    with pytest.raises(ValueError, match="cannot fit without samples"):
        fit([])


def test_coefficients_come_back_as_weights_anchored_at_one():
    coefficients = tuple(float(i + 2) for i in range(len(TERM_NAMES)))
    weights = to_weights(coefficients)
    assert weights.victory_point == 1.0
    assert weights.production == pytest.approx(3.0 / 2.0)


def test_a_wrong_width_is_refused():
    with pytest.raises(ValueError, match="expected"):
        to_weights((1.0, 2.0))


def test_a_negative_victory_point_coefficient_is_refused():
    """Rescaling by it would silently invert every other term."""
    with pytest.raises(ValueError, match="refusing to rescale"):
        to_weights((-1.0,) + (0.0,) * (len(TERM_NAMES) - 1))
