# SPDX-License-Identifier: GPL-3.0-only
"""Fit the evaluation weights and the win temperature by conditional logit.

The bot reads a position as `softmax(v / T)[seat]`, with `v_i = w . f_i` the
linear evaluation of seat `i` (`victory_point` pinned at 1.0). Label each
recorded position with the seat that went on to win and that reading is
exactly McFadden's conditional logit: four alternatives, one shared
coefficient vector `beta = w / T`. The log-likelihood is concave in `beta`,
so it is one Newton solve rather than a search, and `T = 1 / beta_vp`,
`w = beta / beta_vp` fall out of the same solve -- the pair is identified
jointly, which is why fitting them in turn oscillated.

What this can and cannot say. The winner is one label per game, so the
positions of a game add information only through how their features drift
toward the winner; intervals here are cluster-robust by game (the sandwich)
and optionally block-bootstrapped by game, never per position. And a
likelihood over positions ranks positions across games; the search only ever
compares siblings one move apart. Held-out log loss says the model predicts
winners better; only a paired duel says the bot plays better. Report both,
trust the duel.

`Design` decides which columns the model sees. `scarce` is not free -- it is
`0.91 * production / ROLLS` from the opening fit -- so it is folded into the
production column at that ratio and re-derived from the fitted production.
A term can be dropped (pinned at zero) and two cross-seat features that the
per-seat model cannot express can be added: the gap to the table's leader,
and production times the distance still to run.

numpy only. Nine coefficients over a few hundred thousand choice sets do not
need a library, and the result is a plain `Weights`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Sequence

import numpy as np

from .bots.evaluate import ROLLS, TERM_NAMES, Weights
from .dataset import ChoiceSet
from .victory import WINNING_POINTS

ANCHOR = "victory_point"
# `evaluate.FITTED_SCARCE`'s exchange rate: scarce is this many production
# weights per scarce resource reached. Folded, not fitted.
SCARCE_PER_PRODUCTION = 0.91 / ROLLS

# Cross-seat features a per-seat row cannot carry. `leader_gap` is the
# leader's points less this seat's (zero for the leader); `remaining_production`
# is production times the points still needed, so the same rate is worth more
# to a seat at 8 than at 2.
EXTRA_NAMES: tuple[str, ...] = ("leader_gap", "remaining_production")


@dataclass(frozen=True)
class Design:
    """Which columns the logit sees, as linear combinations of the base terms.

    Every column is `{base name: coefficient}` over `TERM_NAMES + EXTRA_NAMES`.
    `standard()` is the shipped model: every term, scarce folded into
    production. The anchor column must be present and first.
    """

    columns: tuple[tuple[str, dict[str, float]], ...]

    @staticmethod
    def standard(
        *, drop: Sequence[str] = (), extras: Sequence[str] = ()
    ) -> "Design":
        columns: list[tuple[str, dict[str, float]]] = []
        for name in TERM_NAMES:
            if name == "scarce" or name in drop:
                continue
            if name == "production":
                columns.append((name, {"production": 1.0, "scarce": SCARCE_PER_PRODUCTION}))
            else:
                columns.append((name, {name: 1.0}))
        for name in extras:
            if name not in EXTRA_NAMES:
                raise ValueError(f"unknown extra feature: {name}")
            columns.append((name, {name: 1.0}))
        return Design(tuple(columns))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.columns)

    def matrix(self, samples: Sequence[ChoiceSet]) -> np.ndarray:
        """`(choice sets, seats, columns)`."""
        return self.matrix_from(base_features(samples))

    def matrix_from(self, base: np.ndarray) -> np.ndarray:
        index = {name: i for i, name in enumerate(TERM_NAMES + EXTRA_NAMES)}
        out = np.zeros(base.shape[:2] + (len(self.columns),))
        for j, (_, combination) in enumerate(self.columns):
            for name, coefficient in combination.items():
                out[:, :, j] += coefficient * base[:, :, index[name]]
        return out

    def weights(self, beta: Sequence[float]) -> tuple[Weights, float, dict[str, float]]:
        """`(Weights, T, extras)` from a coefficient vector on these columns.

        `T = 1 / beta_vp`, every weight `beta_j / beta_vp`, scarce re-derived
        from the fitted production. Extras come back separately: they are not
        `Weights` fields until the evaluator grows them.
        """
        names = self.names
        if names[0] != ANCHOR:
            raise ValueError("the anchor must be the first column")
        anchor = beta[0]
        if anchor <= 0.0:
            raise ValueError(
                f"victory point coefficient is {anchor:.4g}; refusing to rescale by it"
            )
        values = {name: 0.0 for name in TERM_NAMES}
        extras: dict[str, float] = {}
        for name, b in zip(names, beta):
            if name in values:
                values[name] = b / anchor
            else:
                extras[name] = b / anchor
        values["scarce"] = SCARCE_PER_PRODUCTION * values["production"]
        return Weights(**values), 1.0 / anchor, extras


def base_features(samples: Sequence[ChoiceSet]) -> np.ndarray:
    """`(choice sets, seats, TERM_NAMES + EXTRA_NAMES)` -- the evaluator's rows
    with the two cross-seat features appended."""
    rows = np.array([s.rows for s in samples], dtype=float)
    points = rows[:, :, 0]
    production = rows[:, :, TERM_NAMES.index("production")]
    leader_gap = points.max(axis=1, keepdims=True) - points
    remaining = production * np.maximum(WINNING_POINTS - points, 0.0)
    return np.concatenate([rows, leader_gap[:, :, None], remaining[:, :, None]], axis=2)


def labels_of(samples: Sequence[ChoiceSet]) -> np.ndarray:
    return np.array([s.winner for s in samples], dtype=int)


def games_of(samples: Sequence[ChoiceSet]) -> np.ndarray:
    return np.array([s.game for s in samples], dtype=int)


def incumbent_beta(design: Design, weights: Weights, temperature: float) -> np.ndarray:
    """The shipped `(weights, T)` expressed on `design`'s columns, so its
    held-out loss is read on the same rows as the fit's. Only exact for
    designs whose columns are single terms (plus the production fold)."""
    beta = []
    for name, combination in design.columns:
        if name in EXTRA_NAMES:
            beta.append(0.0)
        else:
            beta.append(getattr(weights, name) / temperature)
    return np.array(beta)


# --- the likelihood -----------------------------------------------------------


def _probabilities(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    u = X @ beta
    u -= u.max(axis=1, keepdims=True)
    e = np.exp(u)
    return e / e.sum(axis=1, keepdims=True)


def log_loss(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
    """Mean cross-entropy of `softmax(X beta)` against the winner."""
    p = _probabilities(X, beta)
    picked = p[np.arange(len(y)), y]
    return float(-np.log(np.clip(picked, 1e-300, None)).mean())


def uniform_loss(seats: int) -> float:
    return log(seats)


def _score_rows(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Per-choice-set gradient of the negative log-likelihood, `(n, k)`."""
    p = _probabilities(X, beta)
    p[np.arange(len(y)), y] -= 1.0
    return np.einsum("nj,njk->nk", p, X)


def _hessian(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Total (not mean) Hessian of the negative log-likelihood, `(k, k)`."""
    p = _probabilities(X, beta)
    mean_x = np.einsum("nj,njk->nk", p, X)
    second = np.einsum("nj,njk,njl->kl", p, X, X)
    return second - mean_x.T @ mean_x


@dataclass(frozen=True)
class Fit:
    design: Design
    beta: tuple[float, ...]
    # Cluster-robust (by game) standard errors of `beta`, same order.
    se: tuple[float, ...]
    iterations: int
    train_loss: float
    samples: int
    games: int
    converged: bool
    bootstrap: tuple[tuple[float, ...], ...] = field(default=())

    def weights(self) -> tuple[Weights, float, dict[str, float]]:
        return self.design.weights(self.beta)

    def ratio_se(self) -> dict[str, float]:
        """Delta-method standard errors of `T` and of every `beta_j / beta_vp`.

        Uses the sandwich covariance carried in `se` only along the diagonal
        is wrong for a ratio, so this recomputes from the full covariance,
        which `fit()` stores on the instance as `_cov` for exactly this call.
        """
        cov = getattr(self, "_cov")
        beta = np.array(self.beta)
        anchor = beta[0]
        out: dict[str, float] = {}
        # T = 1 / anchor
        g = np.zeros(len(beta))
        g[0] = -1.0 / anchor**2
        out["temperature"] = float(np.sqrt(g @ cov @ g))
        for j, name in enumerate(self.design.names):
            if j == 0:
                continue
            g = np.zeros(len(beta))
            g[0] = -beta[j] / anchor**2
            g[j] = 1.0 / anchor
            out[name] = float(np.sqrt(max(g @ cov @ g, 0.0)))
        return out


def fit(
    X: np.ndarray,
    y: np.ndarray,
    games: np.ndarray,
    *,
    start: np.ndarray | None = None,
    l2: float = 1e-8,
    max_iterations: int = 50,
    tolerance: float = 1e-9,
    bootstrap: int = 0,
    seed: int = 0,
) -> Fit:
    """Newton's method on the conditional-logit negative log-likelihood.

    Columns are standardised for the solve and the coefficients mapped back,
    so the Newton step is well conditioned whatever units the terms come in.
    `l2` is a ridge on the standardised coefficients, small enough to be
    numerical rather than a prior. Steps that raise the loss are halved.

    Standard errors are the cluster-robust sandwich with games as clusters:
    every choice set of a game shares one label, so treating them as
    independent overstates the precision by roughly the positions per game.
    `bootstrap > 0` also resamples whole games that many times and refits,
    and the replicates' coefficients come back for the caller to summarise.
    """
    n, seats, k = X.shape
    scale = X.reshape(-1, k).std(axis=0)
    scale[scale == 0.0] = 1.0
    Z = X / scale
    b = np.zeros(k) if start is None else np.array(start) * scale

    loss = log_loss(Z, y, b) + 0.5 * l2 * float(b @ b)
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        gradient = _score_rows(Z, y, b).sum(axis=0) / n + l2 * b
        hessian = _hessian(Z, b) / n + l2 * np.eye(k)
        step = np.linalg.solve(hessian, gradient)
        # Backtracking: a Newton step from far away can overshoot.
        t = 1.0
        while True:
            candidate = b - t * step
            candidate_loss = log_loss(Z, y, candidate) + 0.5 * l2 * float(candidate @ candidate)
            if candidate_loss <= loss or t < 1e-6:
                break
            t *= 0.5
        improvement = loss - candidate_loss
        b, loss = candidate, candidate_loss
        if improvement < tolerance and float(np.abs(step).max()) < 1e-6:
            converged = True
            break

    beta = b / scale

    # Sandwich: H^-1 (sum_g s_g s_g^T) H^-1, on the raw scale.
    scores = _score_rows(X, y, beta)
    order = np.argsort(games, kind="stable")
    sorted_games = games[order]
    boundaries = np.flatnonzero(np.diff(sorted_games)) + 1
    by_game = np.add.reduceat(scores[order], np.concatenate([[0], boundaries]), axis=0)
    meat = by_game.T @ by_game
    bread = np.linalg.inv(_hessian(X, beta))
    cov = bread @ meat @ bread
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    replicates: list[tuple[float, ...]] = []
    if bootstrap:
        rng = np.random.default_rng(seed)
        unique = np.unique(games)
        index_by_game = {g: np.flatnonzero(games == g) for g in unique}
        for _ in range(bootstrap):
            drawn = rng.choice(unique, size=len(unique), replace=True)
            rows = np.concatenate([index_by_game[g] for g in drawn])
            replicate = fit(
                X[rows], y[rows], games[rows], start=beta, l2=l2,
                max_iterations=max_iterations, tolerance=tolerance,
            )
            replicates.append(replicate.beta)

    result = Fit(
        design=Design(()),  # replaced by the caller through `with_design`
        beta=tuple(float(v) for v in beta),
        se=tuple(float(v) for v in se),
        iterations=iterations,
        train_loss=log_loss(X, y, beta),
        samples=n,
        games=int(len(np.unique(games))),
        converged=converged,
        bootstrap=tuple(replicates),
    )
    object.__setattr__(result, "_cov", cov)
    return result


def fit_design(
    design: Design,
    train: Sequence[ChoiceSet],
    *,
    base: np.ndarray | None = None,
    **options,
) -> Fit:
    """`fit` on `design`'s columns of `train`, the `Fit` carrying the design."""
    X = design.matrix(train) if base is None else design.matrix_from(base)
    result = fit(X, labels_of(train), games_of(train), **options)
    out = Fit(
        design=design,
        beta=result.beta,
        se=result.se,
        iterations=result.iterations,
        train_loss=result.train_loss,
        samples=result.samples,
        games=result.games,
        converged=result.converged,
        bootstrap=result.bootstrap,
    )
    object.__setattr__(out, "_cov", getattr(result, "_cov"))
    return out


def to_weights(coefficients: Sequence[float]) -> Weights:
    """Rescale a full `TERM_NAMES`-ordered coefficient vector so
    `victory_point` is 1.0 and hand back a `Weights`. Kept for callers that
    fit every term directly; `Design.weights` is the folded path."""
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
