"""Is a Catan seat's conditional terminal return actually Gaussian?

Registered as Gate A1 for candidate 3 (`agents/reference/variance-screen.md`):
the case for a distributional value head rests entirely on the target being
non-Gaussian, and nothing on this project has ever measured that -- it was
inherited motivation, and week 1 withdrew the borrowed sentence it rested on.
`benchmarks.floor` already produces the thing this needs. It snapshots a
position, replays it many times with `catan.game.imagine`, and keeps only two
scalars from the resulting return sample: its variance (the floor) and the
squared gap between its mean and the head's prediction (bias^2). Everything
else about that sample -- its shape -- is thrown away. `--dump-returns` on
`benchmarks.floor` keeps it instead of throwing it away; this module reads the
dump and asks the shape question of it.

## Units

Three numbers come out per position, and they are not in the same units.
**Excess kurtosis** and the **bimodality coefficient** are both built from
standardized moments (each moment divided by a matching power of the standard
deviation), so they are pure numbers -- unaffected by whether the return is
read in reward units or in VP. The **Wasserstein-1 distance is not**: it is a
distance between two distributions of returns, so it carries the same units
the returns do. `catan.rewards.relative_points` scales terminal points by
`catan.victory.WINNING_POINTS` (=10) so a value head does not have to learn the
units -- which means a W1 computed directly on the dumped returns is in units
of a *tenth of a victory point*, and quoting it as-is would understate the
mismatch by 10x against the register's 0.10 VP pass line. `wasserstein1_vp`
below is the one to read; `wasserstein1` (reward units) exists so the
conversion itself is a testable fact rather than an assumption.

## Hartigan's dip statistic, substituted

The register asks for Hartigan's dip statistic. A correct implementation finds
the greatest convex minorant and least concave majorant of the empirical CDF
and iterates between them to locate the modal interval -- the algorithm behind
R's `diptest` runs to several hundred lines of C for a reason, and a from-
scratch port that fits this project's ~80-line budget would be an unverified
new algorithm, not a known one; a silently-wrong statistic in a registered gate
is worse than no statistic. Per the register's own escape hatch, this reports
**the bimodality coefficient** instead (Sarle's coefficient, popularized by
Pfister, Schwarz, Janczyk, Dale & Freudenthal, *Behavior Research Methods*
2013): with sample skewness g and sample excess kurtosis k over n draws,

    BC = (g^2 + 1) / (k + 3*(n-1)^2 / ((n-2)*(n-3)))

BC > 5/9 (~0.555, the value a uniform distribution gives) is the conventional
bimodality flag. It is a much weaker instrument than the dip statistic --
it is a moment-based heuristic, not a test with a null distribution, and it
can be fooled by heavy tails -- so it is reported alongside kurtosis and W1
rather than in place of either, and the decision-relevant number for the gate
stays W1, exactly as the register specifies.

## The Wasserstein-1 number, in closed form

W1 between two one-dimensional distributions equals the area between their
quantile functions: `W1(P, Q) = integral_0^1 |F_P^-1(u) - F_Q^-1(u)| du`. The
empirical quantile function of n sorted returns is the step function that
equals `x_(i)` on `((i-1)/n, i/n]`, so the integral splits into n pieces, one
per order statistic, each against the Gaussian's quantile function on that
same slice of probability mass. The Gaussian's inverse CDF has a closed-form
antiderivative (`integral Phi^-1(u) du = -phi(Phi^-1(u))`, phi the standard
normal density), so every piece is closed-form: evaluate the antiderivative at
the slice's two ends, split at the one point within the slice where the sign
of `x_(i) - F_Q^-1(u)` can flip (there is at most one, since `F_Q^-1` is
monotone), and sum. This is used instead of drawing samples from the matched
Gaussian and comparing empirical distributions because that would spend Monte
Carlo noise pricing a comparison that already has a closed form -- the dumped
sample is the only randomness this measurement should carry, not a second,
gratuitous draw for the reference side.

## Pooling

"Pooled" means the mean of each position's own statistic, exactly the
convention `benchmarks.floor.pool` already uses for floor and bias^2 -- never
the raw returns concatenated across positions. Concatenating would mix each
position's own conditional shape with the between-position spread of the mean
prediction itself, which is a different and much larger quantity that this
question was never about.

    python -m benchmarks.floor --checkpoint runs/lam095/latest.pt \\
        --positions 128 --rollouts 128 --dump-returns /tmp/returns.json
    python -m benchmarks.return_shape --dump /tmp/returns.json
"""

from __future__ import annotations

import argparse
import json
import sys
from statistics import NormalDist

import numpy as np

from catan.victory import WINNING_POINTS

_STANDARD_NORMAL = NormalDist()


def excess_kurtosis(returns: np.ndarray) -> float:
    """Fisher's g2 on the sample's own (population) moments.

    Population moments rather than a bias-corrected estimator, to match
    `benchmarks.floor.split`'s use of `returns.var()` (also population) on the
    same samples, and because it is the convention that makes the two-point
    mixture's kurtosis land on exactly -2 rather than a bias-adjusted
    near-neighbour of it.
    """
    arr = np.asarray(returns, dtype=np.float64)
    mean = arr.mean()
    m2 = np.mean((arr - mean) ** 2)
    if m2 == 0.0:
        return float("nan")
    m4 = np.mean((arr - mean) ** 4)
    return float(m4 / m2**2 - 3.0)


def bimodality_coefficient(returns: np.ndarray) -> float:
    """Sarle's coefficient -- see the module docstring for why this substitutes
    for Hartigan's dip statistic, and its formula."""
    arr = np.asarray(returns, dtype=np.float64)
    n = arr.size
    if n <= 3:
        return float("nan")
    mean = arr.mean()
    m2 = np.mean((arr - mean) ** 2)
    if m2 == 0.0:
        return float("nan")
    m3 = np.mean((arr - mean) ** 3)
    skew = m3 / m2**1.5
    kurtosis = excess_kurtosis(arr)
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((skew**2 + 1) / (kurtosis + correction))


def _antiderivative(u: float, target: float, mean: float, sigma: float) -> float:
    """`integral (target - mu - sigma*Phi^-1(v)) dv` from 0 to `u`.

    `Phi^-1(0)` and `Phi^-1(1)` are infinite, but the antiderivative's limit at
    both ends is finite (`phi` decays faster than `Phi^-1` grows), so 0 and 1
    are handled directly rather than by evaluating `inv_cdf` at an open-interval
    boundary it refuses.
    """
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return target - mean
    z = _STANDARD_NORMAL.inv_cdf(u)
    return (target - mean) * u + sigma * _STANDARD_NORMAL.pdf(z)


def wasserstein1(returns: np.ndarray) -> float:
    """W1 to the moment-matched Gaussian, in whatever units `returns` is in.

    See the module docstring's closed-form derivation. `sigma == 0` (every
    rollout returned the identical value) makes the matched "Gaussian" a point
    mass equal to the data, so the distance is exactly zero rather than
    undefined.
    """
    arr = np.sort(np.asarray(returns, dtype=np.float64))
    n = arr.size
    if n == 0:
        raise ValueError("wasserstein1 needs at least one rollout return")
    mean = float(arr.mean())
    sigma = float(arr.std())
    if sigma == 0.0:
        return 0.0

    total = 0.0
    for i in range(1, n + 1):
        target = float(arr[i - 1])
        low, high = (i - 1) / n, i / n
        z = (target - mean) / sigma
        crossing = _STANDARD_NORMAL.cdf(z)
        h_low = _antiderivative(low, target, mean, sigma)
        h_high = _antiderivative(high, target, mean, sigma)
        if low < crossing < high:
            h_cross = (target - mean) * crossing + sigma * _STANDARD_NORMAL.pdf(z)
            total += 2 * h_cross - h_low - h_high
        else:
            total += abs(h_high - h_low)
    return total


def wasserstein1_vp(returns: np.ndarray) -> float:
    """`wasserstein1`, converted from reward units to VP. See module docstring."""
    return WINNING_POINTS * wasserstein1(returns)


def shape(returns: np.ndarray) -> dict:
    """The three reported statistics for one position's rollout sample."""
    arr = np.asarray(returns, dtype=np.float64)
    return {
        "kurtosis": excess_kurtosis(arr),
        "bimodality_coefficient": bimodality_coefficient(arr),
        "wasserstein1_vp": wasserstein1_vp(arr),
    }


def load_dump(path: str) -> dict:
    """The file `benchmarks.floor --dump-returns` wrote."""
    with open(path) as handle:
        return json.load(handle)


def analyse(payload: dict, bins: int = 5) -> dict:
    """Per-position shape, pooled and by stage.

    `bins=5` is the "five game stages" the register names for Gate A1, matching
    `benchmarks.value_head`'s own default -- not `benchmarks.floor`'s CLI
    default of 4, which is that report's own choice and unrelated to this one.
    """
    rows = []
    for position in payload["positions"]:
        returns = np.asarray(position["returns"], dtype=np.float64)
        rows.append(
            {
                "progress": position["progress"],
                "seat": position["seat"],
                "prediction": position["prediction"],
                "rollouts": int(returns.size),
                **shape(returns),
            }
        )
    return {"positions": rows, "pooled": _pool(rows), "stages": _stages(rows, bins)}


def _pool(rows: list[dict]) -> dict:
    """Mean of each position's own statistic -- see module docstring."""
    return {
        "kurtosis": round(float(np.mean([r["kurtosis"] for r in rows])), 4),
        "bimodality_coefficient": round(
            float(np.mean([r["bimodality_coefficient"] for r in rows])), 4
        ),
        "wasserstein1_vp": round(float(np.mean([r["wasserstein1_vp"] for r in rows])), 4),
    }


def _stages(rows: list[dict], bins: int) -> list[dict]:
    """Same edge convention as `benchmarks.floor._stages`: right-open bins over
    `progress`, except the last, which includes 1.0."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    progress = np.asarray([r["progress"] for r in rows])
    out = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (progress >= low) & (progress < high if high < 1.0 else progress <= 1.0)
        if not inside.any():
            continue
        bucket = [r for r, keep in zip(rows, inside) if keep]
        out.append(
            {
                "from": round(float(low), 2),
                "to": round(float(high), 2),
                "positions": len(bucket),
                **_pool(bucket),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, help="path written by floor.py --dump-returns")
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = load_dump(args.dump)
    result = analyse(payload, bins=args.bins)

    output = {
        "dump": args.dump,
        "checkpoint": payload.get("checkpoint"),
        "iteration": payload.get("iteration"),
        "positions": len(result["positions"]),
        "pooled": result["pooled"],
        "stages": result["stages"],
        "rows": result["positions"],
    }

    if args.json:
        print(json.dumps(output, indent=2))
        return 0

    pooled = output["pooled"]
    print(
        f"{output['positions']} positions, checkpoint {output['checkpoint']} "
        f"(iteration {output['iteration']})"
    )
    print(f"  pooled excess kurtosis          {pooled['kurtosis']:.4f}")
    print(f"  pooled bimodality coefficient   {pooled['bimodality_coefficient']:.4f}")
    print(
        f"  pooled Wasserstein-1 mismatch   {pooled['wasserstein1_vp']:.4f} VP"
        "  (registered pass line: 0.10 VP)"
    )
    print("  by stage of the game:")
    for stage in output["stages"]:
        print(
            f"    {stage['from']:.2f}-{stage['to']:.2f}  {stage['positions']:>4} pos"
            f"  kurtosis {stage['kurtosis']:.4f}"
            f"  bimodality {stage['bimodality_coefficient']:.4f}"
            f"  W1 {stage['wasserstein1_vp']:.4f} VP"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
