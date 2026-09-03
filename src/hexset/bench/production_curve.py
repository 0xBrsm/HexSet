# SPDX-License-Identifier: GPL-3.0-only
"""Measure whether the production weight is identifiable from self-play.

The evaluation can depend critically on a term at zero while being flat over
every plausible non-zero value.  An optimiser cannot recover a coefficient in
that landscape, however good its search rule is.  This benchmark therefore
maps candidate production values directly against the intact weights before
any more search is attempted.

The same games also test terminal victory points as a proxy for wins.  Every
duel retains both sides' terminal points, so point and win effects come from
the same seeded sample rather than from two experiments that can disagree by
chance.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from typing import Sequence

from hexset.bench.throughput import default_workers, environment
from hexset.arena import Z_95, compete, mean_interval, wilson
from hexset.bots.evaluate import Weights
from hexset.tuning import entrant_for

DEFAULT_LEVELS = (0.0, 1.0, 2.0, 3.0, 3.5, 5.0, Weights().production, 10.0, 14.0)


@dataclass(frozen=True)
class CurvePoint:
    production: float
    wins: int
    decided: int
    win_rate: float
    win_interval: tuple[float, float]
    candidate_points: float
    intact_points: float
    point_difference: float
    point_interval: tuple[float, float]
    outcome_point_correlation: float
    seconds: float

    @property
    def win_sign(self) -> int:
        return int(self.win_interval[0] > 0.5) - int(self.win_interval[1] < 0.5)

    @property
    def point_sign(self) -> int:
        return int(self.point_interval[0] > 0.0) - int(
            self.point_interval[1] < 0.0
        )


def correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Pearson correlation, returning zero when either input is constant."""
    if not left or len(left) != len(right):
        return 0.0
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_scale = sum((value - left_mean) ** 2 for value in left)
    right_scale = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator else 0.0


def ranks(values: Sequence[float]) -> list[float]:
    """Average ranks, so Spearman correlation remains defined across ties."""
    ordered = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + end - 1) / 2
        for index in ordered[start:end]:
            out[index] = rank
        start = end
    return out


def measure(
    production: float,
    games: int,
    *,
    seed: int,
    workers: int,
) -> CurvePoint:
    """Play one production value against intact and retain both outcome signals."""
    intact = Weights()
    candidate = replace(intact, production=production)
    # Two of each, so the pairing survives the seat rotation.
    entrants = tuple(
        entrant_for(f"{name}-{copy}", weights, 1, None)
        for copy in range(2)
        for name, weights in (("candidate", candidate), ("intact", intact))
    )
    result = compete(entrants, games, seed=seed, workers=workers)

    mine = [e for e, entrant in enumerate(entrants) if entrant.name.startswith("candidate")]
    theirs = [e for e, entrant in enumerate(entrants) if entrant.name.startswith("intact")]
    decided_rows = result.decided()
    outcomes = [float(winner in mine) for winner, _ in decided_rows]
    candidate_points = [statistics.mean(row[e] for e in mine) for _, row in decided_rows]
    intact_points = [statistics.mean(row[e] for e in theirs) for _, row in decided_rows]
    differences = [
        candidate_score - intact_score
        for candidate_score, intact_score in zip(candidate_points, intact_points)
    ]
    wins = sum(int(outcome) for outcome in outcomes)
    decided = len(outcomes)
    low, high = wilson(wins, decided, Z_95) if decided else (0.0, 1.0)
    point_estimate = mean_interval(differences, Z_95)
    return CurvePoint(
        production=production,
        wins=wins,
        decided=decided,
        win_rate=wins / decided if decided else 0.0,
        win_interval=(low, high),
        candidate_points=statistics.mean(candidate_points) if decided else 0.0,
        intact_points=statistics.mean(intact_points) if decided else 0.0,
        point_difference=point_estimate.mean,
        point_interval=(point_estimate.lower, point_estimate.upper),
        outcome_point_correlation=correlation(outcomes, differences)
        if decided
        else 0.0,
        seconds=result.seconds,
    )


def proxy_summary(points: Sequence[CurvePoint]) -> dict[str, float | int | None]:
    """Agreement between level strength as ranked by wins and by points."""
    win_rates = [point.win_rate for point in points]
    point_differences = [point.point_difference for point in points]
    fitted = Weights().production

    def distance(point: CurvePoint) -> float:
        return abs(point.production - fitted)

    nonzero = [point for point in points if point.production != 0.0]
    plateau = [point for point in points if point.production >= 3.0]
    comparisons = [point for point in points if point.production != fitted]
    closest = min(
        [point for point in comparisons if point.win_sign],
        key=distance,
        default=None,
    )
    # Looking at eight alternatives makes a lone pointwise 95% result fairly
    # likely under a flat null.  Keep that exploratory boundary, but also give
    # the Bonferroni boundary whose family-wise false-positive rate is 5%.
    family_z = statistics.NormalDist().inv_cdf(
        1.0 - 0.05 / (2 * max(1, len(comparisons)))
    )
    familywise_significant = [
        point
        for point in comparisons
        if (
            (interval := wilson(point.wins, point.decided, family_z))[0] > 0.5
            or interval[1] < 0.5
        )
    ]
    familywise_closest = min(familywise_significant, key=distance, default=None)
    return {
        "pearson": correlation(win_rates, point_differences),
        "spearman": correlation(ranks(win_rates), ranks(point_differences)),
        "pearson_without_ablation": correlation(
            [point.win_rate for point in nonzero],
            [point.point_difference for point in nonzero],
        ),
        "spearman_without_ablation": correlation(
            ranks([point.win_rate for point in nonzero]),
            ranks([point.point_difference for point in nonzero]),
        ),
        # The cliff can make two objectives look well aligned even when they
        # rank the plausible values an optimiser actually sees differently.
        "pearson_at_or_above_3": correlation(
            [point.win_rate for point in plateau],
            [point.point_difference for point in plateau],
        ),
        "spearman_at_or_above_3": correlation(
            ranks([point.win_rate for point in plateau]),
            ranks([point.point_difference for point in plateau]),
        ),
        "significant_sign_agreements": sum(
            point.win_sign == point.point_sign
            for point in comparisons
            if point.win_sign and point.point_sign
        ),
        "significant_sign_conflicts": sum(
            point.win_sign == -point.point_sign
            for point in comparisons
            if point.win_sign and point.point_sign
        ),
        "closest_distinguishable_production": None
        if closest is None
        else closest.production,
        "familywise_z": family_z,
        "closest_familywise_distinguishable_production": None
        if familywise_closest is None
        else familywise_closest.production,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="*", type=float, default=DEFAULT_LEVELS)
    parser.add_argument("--games", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=630_000)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    points = []
    for level in args.levels:
        point = measure(level, args.games, seed=args.seed, workers=args.workers)
        points.append(point)
        if not args.json:
            print(
                f"  production {level:>6g}  {point.wins:>4}/{point.decided}"
                f" = {point.win_rate:6.1%}  95% CI"
                f" [{point.win_interval[0]:.1%}, {point.win_interval[1]:.1%}]"
                f"  points {point.point_difference:+.3f}",
                flush=True,
            )

    payload = {
        "environment": environment(),
        "settings": vars(args),
        "seconds": round(time.perf_counter() - started, 1),
        "games": len(points) * args.games,
        "points": [asdict(point) for point in points],
        "proxy": proxy_summary(points),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload["proxy"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
