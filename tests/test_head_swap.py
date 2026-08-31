# SPDX-License-Identifier: GPL-3.0-only
"""Gate A2's arithmetic against answers derived independently of the code.

The gate is a paired comparison between two heads on one dataset, so the two
things that could void it silently are the loss not fitting what it claims to
fit and the two arms not seeing the same data. Both are pinned here: the
quantile loss is checked to be minimised at the sample quantile the theory
names, and the two heads are checked to receive bit-identical minibatches.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch runs on the training box only")

from torch import nn  # noqa: E402

from benchmarks.head_swap import (  # noqa: E402
    Dataset,
    HeldOut,
    MeanHead,
    QuantileHead,
    fit,
    gate,
    labelled_rows,
    plateau,
    quantile_huber_loss,
    quantile_levels,
    score_held_out,
)
from benchmarks.head_shape import split as split_games  # noqa: E402
from hexset.rewards import reward  # noqa: E402

# Nine samples against four midpoint levels, chosen so that `tau * n` is never
# an integer: on an integer the pinball loss is flat between two order
# statistics and the minimiser is an interval rather than a point, which would
# make "minimised exactly at the quantile" untestable as an equality.
SAMPLE_SIZE = 9
LEVELS = 4


def _pinball_argmin(sample: torch.Tensor, tau: float) -> torch.Tensor:
    """Where `quantile_huber_loss` at level `tau` actually puts its minimum.

    Searched over a grid deliberately offset off every sample value, unioned
    with the order statistics themselves, so the winner can be compared to an
    order statistic by exact tensor equality rather than to grid resolution.
    """
    ordered = torch.sort(sample).values
    grid = torch.arange(-3.0, 3.0, 0.001, dtype=torch.float64) + 0.00037
    candidates = torch.cat([grid, ordered])
    level = torch.tensor([tau], dtype=torch.float64)
    losses = torch.tensor(
        [
            quantile_huber_loss(
                torch.full((sample.numel(), 1, 1), float(theta), dtype=torch.float64),
                sample.view(-1, 1),
                level,
                0.0,
            )
            for theta in candidates
        ]
    )
    return candidates[int(losses.argmin())]


def test_the_midpoint_levels_are_the_registered_ones_and_pair_about_a_half():
    """`(i + 0.5) / Q`, and symmetric -- the symmetry is what makes the mean of
    the quantiles the mean of a symmetric distribution rather than near it."""
    levels = quantile_levels(4)

    assert levels.tolist() == [0.125, 0.375, 0.625, 0.875]
    assert torch.equal(levels + levels.flip(0), torch.full((4,), 1.0))


def test_the_pinball_loss_is_minimised_exactly_at_the_sample_quantile():
    """For `L(theta) = sum_j rho_tau(y_j - theta)`, `dL/dtheta` is
    `#{y_j < theta} - n*tau`, which is negative below the `ceil(n*tau)`-th order
    statistic and positive above it. So the unique minimiser is `y_(ceil(n*tau))`
    -- derived from the definition here, not read off the implementation.
    """
    sample = torch.tensor(
        [-1.4, -0.3, 0.0, 0.2, 0.55, 0.9, 1.1, 1.6, 2.3], dtype=torch.float64
    )
    assert sample.numel() == SAMPLE_SIZE
    ordered = torch.sort(sample).values

    for tau in quantile_levels(LEVELS).tolist():
        rank = math.ceil(tau * SAMPLE_SIZE)
        assert rank != tau * SAMPLE_SIZE, "a tied level makes the minimiser an interval"

        assert _pinball_argmin(sample, tau) == ordered[rank - 1]


def test_a_symmetric_target_puts_the_mean_of_the_quantiles_on_the_mse_optimum():
    """The MSE-optimal prediction for a sample is its mean. For a symmetric
    sample of `n` the order statistics pair as `y_(k) + y_(n+1-k) = 2c`, and the
    midpoint levels pair as `tau` with `1 - tau`, whose minimising ranks are `k`
    and `n + 1 - k`. So the four quantile minimisers pair into two sums of `2c`
    and their mean is `c` exactly -- which is the claim that the quantile head's
    mean estimate is the number GAE would want, on a symmetric target.
    """
    centre = 0.4
    offsets = torch.tensor([-1.1, -0.6, -0.25, -0.05, 0.0], dtype=torch.float64)
    sample = torch.cat([centre + offsets, centre - offsets.flip(0)[1:]])
    assert sample.numel() == SAMPLE_SIZE
    assert float(sample.mean()) == pytest.approx(centre)

    minimisers = torch.stack(
        [_pinball_argmin(sample, tau) for tau in quantile_levels(LEVELS).tolist()]
    )

    # The head's mean estimate is the mean of its quantiles, so wire the
    # minimisers into a real `QuantileHead` and read the number the gate reads.
    head = QuantileHead(1, 1, 4, deep=False, quantiles=LEVELS, kappa=0.0)
    with torch.no_grad():
        head.module.weight.zero_()
        head.module.bias.copy_(minimisers.to(torch.float32))
        estimate = head.mean(torch.zeros(3, 1))

    assert float(minimisers.mean()) == pytest.approx(centre, abs=1e-12)
    assert estimate.shape == (3, 1)
    assert float(estimate[0, 0]) == pytest.approx(centre, abs=1e-6)


def test_the_huber_width_bends_the_loss_only_inside_one_lattice_step():
    """Both regimes against the closed form, which is what says how far the
    smoothing can displace the minimiser. Outside `kappa` the Huberised element
    is `|u| - kappa/2`: the pinball loss less a constant, so it has the same
    slope and moves no minimum. Inside it is `u^2 / (2*kappa)`, and only there
    can the estimator drift off the quantile -- by at most one `kappa`, which is
    one quantum of a label with no finer resolution than that.
    """
    kappa = 1.0 / 30.0
    tau = 0.375
    predicted = torch.zeros(1, 1, 1, dtype=torch.float64)
    levels = torch.tensor([tau], dtype=torch.float64)

    far = torch.tensor([[0.5]], dtype=torch.float64)
    pinball = float(quantile_huber_loss(predicted, far, levels, 0.0))
    assert pinball == pytest.approx(tau * 0.5)
    assert float(quantile_huber_loss(predicted, far, levels, kappa)) == pytest.approx(
        pinball - tau * kappa / 2, rel=1e-12
    )

    inside = kappa / 2
    near = torch.tensor([[inside]], dtype=torch.float64)
    assert float(quantile_huber_loss(predicted, near, levels, kappa)) == pytest.approx(
        tau * inside**2 / (2 * kappa), rel=1e-12
    )


def test_a_negative_huber_width_is_refused():
    with pytest.raises(ValueError):
        quantile_huber_loss(
            torch.zeros(1, 1, 1), torch.zeros(1, 1), quantile_levels(1), -0.1
        )


class _Outcome:
    """The two fields `hexset.rewards.reward` and `head_shape.rows` read."""

    points = (10, 6, 4, 2)
    actions = 40


class _Transition:
    def __init__(self, observation) -> None:
        self.observation = observation


class _Episode:
    def __init__(self, seed: int, index: int, lengths) -> None:
        self.seed = seed
        self.index = index
        self.outcome = _Outcome()
        self.trajectories = [
            [_Transition(f"{index}.{seat}.{step}") for step in range(length)]
            for seat, length in enumerate(lengths)
        ]


def _corpus(count: int) -> list[_Episode]:
    return [_Episode(7, index, (3, 2, 4, 1)) for index in range(count)]


def test_the_split_never_puts_one_game_on_both_sides_of_the_dataset():
    """Every row of a seat's trajectory carries the same terminal label, so a
    row-wise split would report a training loss as a held-out one. Checked on
    the labels the dataset actually carries, not on the episode lists."""
    corpus = _corpus(20)
    train, held = split_games(corpus, 0.25, random.Random(4))

    _, _, train_games = labelled_rows(train)
    _, _, held_games = labelled_rows(held)

    train_keys = {tuple(row) for row in train_games}
    held_keys = {tuple(row) for row in held_games}
    assert not train_keys & held_keys
    assert train_keys | held_keys == {(7, index) for index in range(20)}
    # Each game contributes all ten of its decisions to exactly one side.
    assert len(train_games) == 10 * len(train_keys)
    assert len(held_games) == 10 * len(held_keys)


def test_every_row_is_labelled_with_the_game_it_came_from():
    observations, targets, games = labelled_rows(_corpus(2))

    assert len(observations) == 20
    assert targets.shape == (20, 4)
    assert games.shape == (20, 2)
    assert [tuple(row) for row in games] == [(7, 0)] * 10 + [(7, 1)] * 10


class _Recording(nn.Module):
    """A head that keeps every minibatch of features it was asked to fit on."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.seen: list[torch.Tensor] = []

    def mean(self, features):
        return self.inner.mean(features)

    def loss(self, features, target):
        self.seen.append(features.clone())
        return self.inner.loss(features, target)


def test_both_heads_are_fit_on_bit_identical_features_in_the_same_order():
    """The experiment's whole claim is that the loss is the only difference
    between the arms, so the shuffle, the minibatch boundaries and the feature
    rows themselves have to come out identical from one seed."""
    generator = torch.Generator().manual_seed(11)
    data = Dataset(
        features=torch.randn(23, 6, generator=generator),
        targets=torch.randn(23, 4, generator=generator),
        games=np.zeros((23, 2), dtype=np.int64),
    )
    device = torch.device("cpu")
    torch.manual_seed(0)
    mse = _Recording(MeanHead(6, 4, 8, deep=False))
    torch.manual_seed(0)
    quantile = _Recording(QuantileHead(6, 4, 8, deep=False, quantiles=5, kappa=0.0))

    for head in (mse, quantile):
        fit(
            head,
            data,
            device=device,
            epochs=3,
            minibatch=7,
            learning_rate=1e-3,
            seed=5,
        )

    assert len(mse.seen) == len(quantile.seen) == 3 * 4
    for ours, theirs in zip(mse.seen, quantile.seen):
        assert torch.equal(ours, theirs)
    # Not vacuous: consecutive minibatches really are different rows, so the
    # equality above is about the shuffle agreeing rather than about there
    # being nothing to disagree on.
    assert not torch.equal(mse.seen[0], mse.seen[1])


def test_both_heads_emit_a_players_vector_mean_per_position():
    features = torch.randn(5, 6)
    mse = MeanHead(6, 4, 8, deep=False)
    quantile = QuantileHead(6, 4, 8, deep=False, quantiles=7, kappa=0.0)

    assert mse.mean(features).shape == (5, 4)
    assert quantile.mean(features).shape == (5, 4)
    # The mean estimate is the average of the quantiles and nothing else.
    assert torch.allclose(quantile.mean(features), quantile.spread(features).mean(-1))


def _held_out(returns, reference) -> HeldOut:
    return HeldOut(
        observations=tuple(range(len(returns))),
        progress=tuple(0.1 + 0.2 * i for i in range(len(returns))),
        seats=tuple(range(len(returns))),
        returns=tuple(np.asarray(row, dtype=np.float64) for row in returns),
        reference=tuple(reference),
    )


def test_the_held_out_score_is_the_floor_split_on_the_own_payoff_column():
    """`hexset.ppo.rotate` puts a seat's own payoff in component 0, and that is
    the component the dump's returns are measured in -- so bias^2 here has to be
    `(mean(rollout returns) - prediction[:, 0])^2`, position by position."""
    returns = [[0.4, -0.2, 0.1, 0.7, -0.5], [0.2, 0.2, 0.2, 0.2, 0.2]]
    predictions = [0.15, -0.3]
    held = _held_out(returns, predictions)

    scored = score_held_out(held, predictions, bins=2)

    wanted = [
        (np.mean(row) - prediction) ** 2
        for row, prediction in zip(returns, predictions)
    ]
    assert [row["bias_squared"] for row in scored["rows"]] == pytest.approx(wanted)
    # Each row names the seat its return and its prediction both belong to, so
    # the rotation can be checked against the dump rather than assumed.
    assert [row["seat"] for row in scored["rows"]] == [0, 1]
    assert scored["mean_bias_squared"] == pytest.approx(round(float(np.mean(wanted)), 5))
    # The identity `floor.split` exists for: mse is floor plus bias^2, exactly.
    for row, values, prediction in zip(scored["rows"], returns, predictions):
        assert row["mse"] == pytest.approx(
            float(((np.asarray(values) - prediction) ** 2).mean())
        )


def test_a_prediction_count_that_does_not_match_the_held_out_set_is_refused():
    held = _held_out([[0.1, 0.2]], [0.15])
    with pytest.raises(ValueError):
        score_held_out(held, [0.15, 0.2], bins=2)


def test_the_plateau_reads_the_fall_over_the_last_window_of_epochs():
    """A gate verdict is only about the heads if both arms have stopped moving,
    so the run reports how far each one still fell. Flat reads zero, and a curve
    shorter than the window reads nan rather than a number off two points."""
    assert plateau([1.0, 0.5, 0.4, 0.2, 0.1], 2) == pytest.approx(0.75)
    assert plateau([0.05, 0.05, 0.05], 2) == pytest.approx(0.0)
    assert math.isnan(plateau([0.4, 0.3], 5))
    # A loss that rose reads negative rather than folding into "flat".
    assert plateau([0.1, 0.2, 0.3], 2) == pytest.approx(-2.0)


def test_the_gate_reads_a_ratio_against_the_mse_arm():
    """The registered line: the quantile head cuts held-out bias^2 by >=20%."""
    # Dead on the line passes: the pass is stated as `quantile <= 0.8 * mse`,
    # not as `1 - quantile/mse >= 0.2`, whose float rounding reads 0.19999...
    assert gate(0.01, 0.008, 0.20)["pass"] is True
    assert gate(0.01, 0.008, 0.20)["bias_squared_reduction"] == pytest.approx(0.20)
    assert gate(0.01, 0.0081, 0.20)["pass"] is False
    # A quantile head that is worse reads as a negative reduction, not as zero.
    assert gate(0.01, 0.02, 0.20)["bias_squared_reduction"] == pytest.approx(-1.0)


def test_the_targets_are_the_seat_relative_terminal_return_unnormalised():
    """Both arms see the raw `rotate(reward(outcome), seat)` vector: the
    comparison is void if the two heads' labels are on different scales, so the
    label is the engine's own and neither arm touches it."""
    _, targets, _ = labelled_rows(_corpus(1))
    payoff = reward(_Outcome())

    assert float(targets[0][0]) == pytest.approx(payoff[0])
    assert float(targets.max()) == pytest.approx(max(payoff))
