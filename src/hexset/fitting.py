# SPDX-License-Identifier: GPL-3.0-only
"""Fit evaluation weights to game outcomes by logistic regression.

The hill climb in `hexset.tuning` searches the same nine numbers, but blindly:
it can only ask "did this guess win more games", at roughly a 13% false-accept
rate per round. Once positions carry labels, the problem becomes ordinary
supervised learning — a convex loss with a gradient, fitted in one pass over the
data with no games played at all.

Deliberately no dependencies. Nine features and a convex loss do not need a
library, `hexset` stays importable anywhere, and the result drops straight back
into `Weights` so the search keeps using plain Python arithmetic in its hot loop.

The fitted coefficients are rescaled so `victory_point` is 1.0. Scaling every
weight alike cannot change which position the search prefers, so the unit is
free — and pinning it makes fitted weights comparable with hill-climbed ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Sequence

from .dataset import Sample
from .evaluate import TERM_NAMES, Weights

EPS = 1e-12


def _sigmoid(z: float) -> float:
    # Split by sign so neither branch can overflow exp().
    if z >= 0.0:
        return 1.0 / (1.0 + exp(-z))
    scaled = exp(z)
    return scaled / (1.0 + scaled)


@dataclass(frozen=True)
class Scaling:
    means: tuple[float, ...]
    sigmas: tuple[float, ...]


def scaling_of(rows: Sequence[tuple[float, ...]]) -> Scaling:
    """Per-feature mean and spread.

    Gradient descent on raw features crawls here: victory points run 0-10 while
    the production rate sits near 1. A sigma of zero means a constant column,
    held at 1.0 so the division is harmless and the coefficient stays at zero.
    """
    n = len(rows)
    width = len(rows[0])
    means = [0.0] * width
    for row in rows:
        for i, value in enumerate(row):
            means[i] += value
    means = [m / n for m in means]

    variances = [0.0] * width
    for row in rows:
        for i, value in enumerate(row):
            variances[i] += (value - means[i]) ** 2
    sigmas = [(v / n) ** 0.5 or 1.0 for v in variances]
    return Scaling(tuple(means), tuple(sigmas))


def log_loss(
    rows: Sequence[tuple[float, ...]],
    labels: Sequence[int],
    coefficients: Sequence[float],
    intercept: float,
) -> float:
    total = 0.0
    for row, label in zip(rows, labels):
        z = intercept + sum(c * v for c, v in zip(coefficients, row))
        p = min(max(_sigmoid(z), EPS), 1.0 - EPS)
        total += -(log(p) if label else log(1.0 - p))
    return total / len(rows)


def accuracy(
    rows: Sequence[tuple[float, ...]],
    labels: Sequence[int],
    coefficients: Sequence[float],
    intercept: float,
) -> float:
    right = 0
    for row, label in zip(rows, labels):
        z = intercept + sum(c * v for c, v in zip(coefficients, row))
        right += int((z > 0.0) == bool(label))
    return right / len(rows)


@dataclass(frozen=True)
class Fit:
    coefficients: tuple[float, ...]
    intercept: float
    epochs: int
    train_loss: float

    def weights(self) -> Weights:
        return to_weights(self.coefficients)


def fit(
    samples: Sequence[Sample],
    *,
    epochs: int = 400,
    rate: float = 0.5,
    l2: float = 1e-4,
) -> Fit:
    """Full-batch gradient descent on the log loss, in standardised space.

    Full batch rather than stochastic: the loss is convex, the data fits in
    memory, and a deterministic run is worth more here than a fast one.
    """
    if not samples:
        raise ValueError("cannot fit without samples")

    rows = [s.features for s in samples]
    labels = [s.won for s in samples]
    scale = scaling_of(rows)
    scaled = [
        tuple((v - m) / sd for v, m, sd in zip(row, scale.means, scale.sigmas))
        for row in rows
    ]

    width = len(scaled[0])
    coefficients = [0.0] * width
    intercept = 0.0
    n = len(scaled)

    for _ in range(epochs):
        gradient = [0.0] * width
        bias_gradient = 0.0
        for row, label in zip(scaled, labels):
            z = intercept + sum(c * v for c, v in zip(coefficients, row))
            error = _sigmoid(z) - label
            bias_gradient += error
            for i, value in enumerate(row):
                gradient[i] += error * value
        for i in range(width):
            coefficients[i] -= rate * (gradient[i] / n + l2 * coefficients[i])
        intercept -= rate * bias_gradient / n

    raw = tuple(c / sd for c, sd in zip(coefficients, scale.sigmas))
    raw_intercept = intercept - sum(
        c * m for c, m in zip(raw, scale.means)
    )
    return Fit(
        coefficients=raw,
        intercept=raw_intercept,
        epochs=epochs,
        train_loss=log_loss(rows, labels, raw, raw_intercept),
    )


def to_weights(coefficients: Sequence[float]) -> Weights:
    """Rescale so `victory_point` is 1.0 and hand back a `Weights`.

    A non-positive victory point coefficient would mean the fit decided that
    scoring points makes winning less likely, so the data or the labels are
    wrong and rescaling by it would silently invert every other term.
    """
    if len(coefficients) != len(TERM_NAMES):
        raise ValueError(
            f"expected {len(TERM_NAMES)} coefficients, got {len(coefficients)}"
        )
    anchor = coefficients[0]
    if anchor <= 0.0:
        raise ValueError(
            f"victory point coefficient is {anchor:.4g}; refusing to rescale by it"
        )
    return Weights(**{
        name: value / anchor for name, value in zip(TERM_NAMES, coefficients)
    })
