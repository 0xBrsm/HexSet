# SPDX-License-Identifier: GPL-3.0-only
"""Fit evaluation weights by playing candidates against the incumbent.

A hill climb, not a gradient method: the fitness is a win rate over sampled
games, so it is noisy and has no derivative worth trusting.

Two things keep the climb from drifting on noise. The scale is pinned, because
multiplying every weight by a constant leaves the argmax unchanged and would
otherwise let the search random-walk along a direction that cannot matter. And
a candidate is only accepted when the *lower* bound of its win-rate interval
clears half, so a duel that merely came out ahead is not enough.

How strict that second test is depends on `z`, and both extremes fail. At the
95% level a challenger needs about 65% of 40 games to be accepted, which real
improvements worth a few points will never manage. At zero it accepts anything
that came out ahead, which is a random walk. The default of one standard error
sits between: about 58% at 40 games, 53% at 200.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, fields, replace
from typing import Callable

from .arena import Z_95, Entrant, compete, wilson
from .evaluate import Weights
from .evaluate_tiered import Weights as TieredWeights

# Pinned so the search cannot wander along the scale direction. A position's
# score is only ever compared with another position's, so the unit is free.
# Both evaluations happen to anchor on the same term.
ANCHOR = "victory_point"

# One standard error. See the module docstring for why not 1.96.
ACCEPT_Z = 1.0


def _no_trade_weights() -> Weights:
    # Imported lazily: `hexset.heximax` reaches `hexset.mcts`, which wants
    # numpy, and not every caller of this module has it (`hexset.arena`
    # imports `heximax` the same way, for the same reason).
    from .heximax import NO_TRADE_WEIGHTS

    return NO_TRADE_WEIGHTS


# Which weights go with which evaluation. The two greedy/search evaluations do
# not share a term set — that is the point of keeping both — so a fit is
# always for one of them. `heximax-trading` and `heximax-notrade` share
# `evaluate.Weights` with "default" (heximax's `HonestEvaluator` wraps the
# same `Evaluator`), but they start the climb from heximax's own profile
# (`TRADING_WEIGHTS`, which is `Weights()`, and `NO_TRADE_WEIGHTS`) rather
# than from the bare default, so each gets its own registry entry.
WEIGHTS = {
    "default": Weights,
    "tiered": TieredWeights,
    "heximax-trading": Weights,
    "heximax-notrade": _no_trade_weights,
}

# `evaluator=` keys that build heximax entrants instead of greedy/search ones,
# and the heximax `mode` each one fits. `honest` plays at the shipped offer
# budget (3); `notrade` plays at zero, per `heximax.BY_MODE`.
HEXIMAX_MODES = {"heximax-trading": "honest", "heximax-notrade": "notrade"}


def tunable(weights: Weights | TieredWeights) -> tuple[str, ...]:
    """Every weight the climb may move — that is, all but the pinned anchor."""
    return tuple(f.name for f in fields(type(weights)) if f.name != ANCHOR)


def perturb(
    weights: Weights, rng: random.Random, *, sigma: float, count: int
) -> Weights:
    """Jitter `count` weights.

    The step is scaled by the weight's own size, with a floor so that a weight
    sitting at zero can still be revived, and so that a sign can flip. Scaling
    by the weight keeps the tiered evaluation's magnitude hierarchy intact,
    since a relative step cannot move a term out of its tier.
    """
    names = tunable(weights)
    changes = {}
    for name in rng.sample(names, min(count, len(names))):
        current = getattr(weights, name)
        step = rng.gauss(0.0, sigma * max(abs(current), 0.05))
        changes[name] = current + step
    return replace(weights, **changes)


def entrant_for(
    name: str,
    weights: Weights,
    depth: int,
    width: int | None,
    stance: str = "relative",
    evaluator: str = "default",
) -> Entrant:
    """Build the entrant a fit plays with `weights`.

    For a heximax `evaluator` (`heximax-trading`/`heximax-notrade`) this is a
    `kind="heximax"` bot in the matching mode, `weights` reaching the
    evaluator via `arena._spawn`'s heximax branch; the offer budget follows
    the mode the way `heximax.heximax`'s own `BY_MODE` does, since a fit has
    to compare bots that are heximax in every way but the vector under test.
    Otherwise it is the plain greedy/search entrant the harness always built.
    """
    if evaluator in HEXIMAX_MODES:
        mode = HEXIMAX_MODES[evaluator]
        return Entrant(
            name=name,
            kind="heximax",
            weights=weights,
            mode=mode,
            k=1,
            depth=depth,
            width=width,
            stance=stance,
            max_offers=0 if mode == "notrade" else 3,
        )
    kind = "greedy" if depth <= 1 else "search"
    return Entrant(
        name=name,
        kind=kind,
        weights=weights,
        depth=depth,
        width=width,
        stance=stance,
        evaluator=evaluator,
    )


def duel(
    challenger: Weights,
    incumbent: Weights,
    games: int,
    *,
    seed: int,
    depth: int,
    width: int | None,
    workers: int = 1,
    stance: str = "relative",
    evaluator: str = "default",
) -> tuple[int, int]:
    """Play two of each, seats rotated. Returns (challenger wins, decided games).

    Both sides read the vector the same way. A stance is what the weights are
    being fitted *for*, not one of the things under test.
    """
    a = entrant_for("challenger", challenger, depth, width, stance, evaluator)
    b = entrant_for("incumbent", incumbent, depth, width, stance, evaluator)
    result = compete([a, b, a, b], games, seed=seed, workers=workers)
    wins = sum(s.wins for s in result.standings if s.name == "challenger")
    return wins, result.games - result.unfinished


@dataclass(frozen=True)
class Step:
    round: int
    accepted: bool
    wins: int
    decided: int
    lower: float
    weights: Weights


def climb(
    start: Weights | None = None,
    *,
    rounds: int = 20,
    games: int = 40,
    sigma: float = 0.4,
    count: int = 2,
    seed: int = 0,
    depth: int = 1,
    width: int | None = None,
    z: float = ACCEPT_Z,
    workers: int = 1,
    stance: str = "relative",
    evaluator: str = "default",
    report: Callable[[Step], None] | None = None,
) -> tuple[Weights, list[Step]]:
    """Hill climb from `start`, returning the best weights and every step tried.

    `report` is called after each duel, because a real run takes long enough
    that waiting for the return value is not useful.
    """
    rng = random.Random(seed)
    incumbent = start or WEIGHTS[evaluator]()
    history: list[Step] = []

    for round_index in range(rounds):
        challenger = perturb(incumbent, rng, sigma=sigma, count=count)
        wins, decided = duel(
            challenger,
            incumbent,
            games,
            seed=seed + 1_000 * (round_index + 1),
            depth=depth,
            width=width,
            workers=workers,
            stance=stance,
            evaluator=evaluator,
        )
        lower = wilson(wins, decided, z)[0] if decided else 0.0
        accepted = lower > 0.5
        if accepted:
            incumbent = challenger
        step = Step(
            round=round_index,
            accepted=accepted,
            wins=wins,
            decided=decided,
            lower=lower,
            weights=challenger,
        )
        history.append(step)
        if report is not None:
            report(step)

    return incumbent, history


@dataclass(frozen=True)
class Confirmation:
    """Whether a finished climb actually beat where it started."""

    wins: int
    decided: int
    lower: float
    upper: float

    @property
    def real(self) -> bool:
        return self.lower > 0.5

    @property
    def win_rate(self) -> float:
        return self.wins / self.decided if self.decided else 0.0


def confirm(
    fitted: Weights,
    baseline: Weights | None = None,
    *,
    games: int = 400,
    seed: int = 999_000,
    depth: int = 1,
    width: int | None = None,
    workers: int = 1,
    stance: str = "relative",
    evaluator: str = "default",
    z: float = Z_95,
) -> Confirmation:
    """Play the fitted weights against the ones the climb started from.

    Without this a climb cannot be read at all. Acceptance is a per-round test
    at roughly a 13% false-positive rate, and that rate does not fall as the
    per-duel budget grows — the Wilson threshold moves with it. So a run that
    accepted a handful of candidates has very likely accepted noise, and the
    only way to tell is one honest high-budget duel at the end, judged at 95%.
    """
    wins, decided = duel(
        fitted,
        baseline or WEIGHTS[evaluator](),
        games,
        seed=seed,
        depth=depth,
        width=width,
        workers=workers,
        stance=stance,
        evaluator=evaluator,
    )
    low, high = wilson(wins, decided, z) if decided else (0.0, 1.0)
    return Confirmation(wins=wins, decided=decided, lower=low, upper=high)


def as_source(weights: Weights) -> str:
    """The weights as a `Weights(...)` literal, ready to paste back as defaults."""
    body = ",\n".join(
        f"    {f.name}={getattr(weights, f.name):.4g}" for f in fields(type(weights))
    )
    return f"Weights(\n{body},\n)"
