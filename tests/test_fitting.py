# SPDX-License-Identifier: GPL-3.0-only
"""The conditional-logit fit and the choice sets it reads."""

from __future__ import annotations

import random

import numpy as np
import pytest

from hexset.board.board import random_base_board
from hexset.bots import RandomBot
from hexset.bots.evaluate import TERM_NAMES, Weights
from hexset.dataset import build, split_by_game
from hexset.fitting import (
    SCARCE_PER_PRODUCTION,
    Design,
    fit,
    incumbent_beta,
    log_loss,
    uniform_loss,
)
from hexset.record import record_game


def _synthetic(n: int, beta: np.ndarray, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4, len(beta)))
    u = X @ beta
    p = np.exp(u - u.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(4, p=row) for row in p])
    games = np.arange(n) // 40  # forty positions a game
    return X, y, games


def test_the_fit_recovers_a_planted_coefficient_vector():
    truth = np.array([1.5, -0.7, 0.3])
    X, y, games = _synthetic(6000, truth)
    result = fit(X, y, games)
    assert result.converged
    beta = np.array(result.beta)
    assert beta == pytest.approx(truth, abs=0.12)
    # Within three cluster-robust standard errors, every coordinate.
    assert np.all(np.abs(beta - truth) < 3 * np.array(result.se))
    assert np.all(np.isfinite(result.se)) and np.all(np.array(result.se) > 0)
    assert result.train_loss < uniform_loss(4)


def test_the_design_folds_scarce_into_production_and_derives_it_back():
    design = Design.standard()
    assert design.names[0] == "victory_point"
    assert "scarce" not in design.names
    production = dict(design.columns)["production"]
    assert production == {"production": 1.0, "scarce": SCARCE_PER_PRODUCTION}

    beta = np.array([0.4] + [0.2] * (len(design.names) - 1))
    weights, temperature, extras = design.weights(beta)
    assert temperature == pytest.approx(2.5)
    assert weights.victory_point == 1.0
    assert weights.production == pytest.approx(0.5)
    assert weights.scarce == pytest.approx(SCARCE_PER_PRODUCTION * 0.5)
    assert extras == {}

    with pytest.raises(ValueError):
        design.weights(np.array([-0.1] + [0.2] * (len(design.names) - 1)))

    dropped = Design.standard(drop=("robber_risk",), extras=("leader_gap",))
    assert "robber_risk" not in dropped.names and dropped.names[-1] == "leader_gap"
    w, _, e = dropped.weights(np.ones(len(dropped.names)))
    assert w.robber_risk == 0.0 and e == {"leader_gap": 1.0}


def test_the_shipped_pair_reads_back_exactly_on_the_standard_design():
    """`Weights()`'s scarce is the fold's ratio times production, so the
    incumbent is expressible on the folded design without loss."""
    design = Design.standard()
    beta = incumbent_beta(design, Weights(), 2.0)
    weights, temperature, _ = design.weights(beta)
    assert temperature == pytest.approx(2.0)
    for name in TERM_NAMES:
        # `Weights().scarce` is the ratio times production rounded to four
        # figures; the fold re-derives it unrounded.
        assert getattr(weights, name) == pytest.approx(getattr(Weights(), name), abs=1e-4)


def _record(seed: int):
    board = random_base_board(random.Random(seed))
    bots = [RandomBot(random.Random(seed * 10 + s)) for s in range(4)]
    return record_game(bots, board, seed)


def test_choice_sets_carry_one_honest_row_per_seat_and_split_by_game():
    records = [_record(1), _record(2)]
    samples = build(records, stride=16)
    decided = [r for r in records if r.winner is not None]
    assert samples, "random games that finished must yield labelled positions"
    for s in samples:
        assert len(s.rows) == 4 and all(len(row) == len(TERM_NAMES) for row in s.rows)
        assert s.winner == records[s.game].winner
        assert 0 <= s.knower < 4 and 0 <= s.mover < 4
    knowers = {(s.game, s.turn, s.knower) for s in samples}
    assert len(knowers) == len(samples), "one choice set per knower per position"
    train, test = split_by_game(samples, holdout=0.5, seed=0)
    assert {s.game for s in train}.isdisjoint({s.game for s in test})
    assert len(train) + len(test) == len(samples)
    assert len({s.game for s in samples}) == len(decided)

    design = Design.standard()
    X = design.matrix(samples)
    assert X.shape == (len(samples), 4, len(design.names))
    beta = incumbent_beta(design, Weights(), 2.4766)
    assert np.isfinite(log_loss(X, np.array([s.winner for s in samples]), beta))
