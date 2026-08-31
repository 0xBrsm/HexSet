# SPDX-License-Identifier: GPL-3.0-only
"""The shape statistics against inputs whose right answer is known independently
of the implementation -- see `benchmarks.return_shape` for the mechanism and
the units each number is quoted in.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from statistics import NormalDist

from benchmarks.return_shape import (
    excess_kurtosis,
    wasserstein1,
    wasserstein1_vp,
)
from hexset.victory import WINNING_POINTS

_STANDARD_NORMAL = NormalDist()


def test_w1_of_a_gaussian_sample_against_its_own_fit_is_near_zero():
    """Drawing from exactly the family being matched to should leave only
    finite-sample noise, not a systematic mismatch -- a tight bound is the
    point, since a loose one would also pass a broken implementation."""
    rng = np.random.default_rng(20260827)
    sample = rng.normal(loc=1.3, scale=0.7, size=20_000)

    got = wasserstein1(sample)

    # A Gaussian's own W1-to-itself at n=20000 concentrates within a couple of
    # hundredths of the scale (0.7 here); 0.02 is comfortably inside that and
    # nowhere near what a real non-Gaussian mismatch (the 0.1 VP = 0.01 reward
    # unit gate line) would produce.
    assert got == pytest.approx(0.0, abs=0.02)


def test_w1_of_a_two_point_mixture_matches_the_closed_form_by_hand():
    """A +-a coin-flip mixture has mean 0 and variance a^2, so its moment-
    matched Gaussian is N(0, a^2). W1 to that Gaussian is derived here from
    the definition -- `integral |F_mix(x) - Phi(x/a)| dx` -- independently of
    `return_shape`'s quantile-domain implementation, using the antiderivative
    `integral Phi(x/a) dx = x*Phi(x/a) + a*phi(x/a)` (standard result, checked
    by differentiating it back).

    By symmetry the three pieces of the integral (below -a, between -a and a,
    above a) reduce to
        W1 = 2a * (2*Phi(1) + 2*phi(1) - 1.5 - phi(0))
    which is what this test computes and compares the implementation against,
    rather than reusing any of `return_shape`'s own arithmetic.
    """
    a = 2.5
    phi = _STANDARD_NORMAL.pdf
    Phi = _STANDARD_NORMAL.cdf
    analytic = 2 * a * (2 * Phi(1) + 2 * phi(1) - 1.5 - phi(0))

    n = 400_000
    half = n // 2
    sample = np.concatenate([np.full(half, a), np.full(n - half, -a)])

    got = wasserstein1(sample)

    assert got == pytest.approx(analytic, rel=1e-9)


def test_excess_kurtosis_of_a_gaussian_sample_is_close_to_zero():
    rng = np.random.default_rng(7)
    sample = rng.normal(size=50_000)

    got = excess_kurtosis(sample)

    # Sampling stderr of excess kurtosis is ~sqrt(24/n); at n=50000 that is
    # ~0.022, so 0.15 is a generous, non-tautological bound.
    assert got == pytest.approx(0.0, abs=0.15)


def test_excess_kurtosis_of_a_two_point_mixture_is_exactly_minus_two():
    """For X = +-a with equal probability: Var(X) = a^2, E[X^4] = a^4, so
    kurtosis = a^4/a^4 - 3 = -2, independent of a and exact -- a Bernoulli-like
    population has no fourth-moment slack a Gaussian's tails would otherwise
    supply."""
    a = 1.7
    sample = np.array([a, -a] * 5000)

    got = excess_kurtosis(sample)

    assert got == pytest.approx(-2.0, abs=1e-9)


def test_wasserstein1_vp_is_ten_times_the_reward_unit_figure():
    """`hexset.rewards.relative_points` divides by `WINNING_POINTS` (10) so a
    value head trains in units of a tenth of a VP; reporting the gate number
    without undoing that scaling would understate a real mismatch by 10x
    against the registered 0.10 VP pass line."""
    assert WINNING_POINTS == 10
    rng = np.random.default_rng(3)
    sample = rng.exponential(scale=0.3, size=500) - 0.3

    reward_units = wasserstein1(sample)
    vp = wasserstein1_vp(sample)

    assert vp == pytest.approx(10.0 * reward_units)
    assert not math.isclose(vp, reward_units)
