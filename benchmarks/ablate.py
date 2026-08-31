# SPDX-License-Identifier: GPL-3.0-only
"""Ablate each evaluation term against the full fitted weights.

One term is zeroed at a time and the crippled evaluation plays the intact one.
The crippled side's win rate is the reading: 50% means the term earns nothing,
and the further below 50% the more the evaluation depends on it.

Reported with intervals, because a term worth two or three points of win rate
is indistinguishable from a term worth nothing at small sample sizes, and
"we ablated it and nothing happened" is a claim that needs the sample size
stated to mean anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields, replace

from benchmarks.throughput import default_workers, environment
from hexset.arena import Z_95, wilson
from hexset.tuning import WEIGHTS, duel


def ablate(
    term: str,
    games: int,
    *,
    seed: int,
    depth: int,
    width: int | None,
    workers: int,
    evaluator: str = "default",
) -> tuple[int, int]:
    full = WEIGHTS[evaluator]()
    return duel(
        replace(full, **{term: 0.0}),
        full,
        games,
        seed=seed,
        depth=depth,
        width=width,
        workers=workers,
        evaluator=evaluator,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--evaluator", choices=sorted(WEIGHTS), default="default")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    terms = [f.name for f in fields(WEIGHTS[args.evaluator])]
    started = time.perf_counter()
    rows = []
    for term in terms:
        wins, decided = ablate(
            term,
            args.games,
            seed=args.seed,
            depth=args.depth,
            width=args.width,
            workers=args.workers,
            evaluator=args.evaluator,
        )
        low, high = wilson(wins, decided, Z_95)
        rows.append(
            {
                "term": term,
                "wins": wins,
                "decided": decided,
                "win_rate": wins / decided if decided else 0.0,
                "interval_95": [low, high],
                # A term matters if removing it drops the side below half, so
                # the interval has to sit clear of 0.5 to say anything.
                "matters": high < 0.5,
            }
        )
        if not args.json:
            row = rows[-1]
            verdict = "matters" if row["matters"] else "not shown"
            print(
                f"  without {term:<14} {wins:>4}/{decided}  {row['win_rate']:6.1%}"
                f"  95% CI [{low:.1%}, {high:.1%}]  {verdict}",
                flush=True,
            )
    elapsed = time.perf_counter() - started

    if args.json:
        print(
            json.dumps(
                {
                    "environment": environment(),
                    "settings": vars(args),
                    "seconds": round(elapsed, 1),
                    "ablations": rows,
                },
                indent=2,
            )
        )
        return 0

    env = environment()
    print(f"commit {env['commit']}  {env['machine']}")
    print(
        f"{len(terms)} terms, {args.games} games each, depth {args.depth},"
        f" {elapsed:.0f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
