# SPDX-License-Identifier: GPL-3.0-only
"""Compare Heximax with each evaluation term removed against the full profile.

Each comparison uses two seats per side, paired boards, and board-level
normal intervals. All games remain in the win-rate denominator; unfinished
games prevent the automatic performance-drop flag. Results are exploratory:
intervals are not adjusted for testing multiple terms.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields, replace

from hexset.bench.metrics import side_metrics
from hexset.bench.throughput import default_workers, environment
from hexset.arena import Entrant, compete
from hexset.bots.heximax import TRADING_WEIGHTS, NO_TRADE_WEIGHTS

WEIGHTS = {"trading": TRADING_WEIGHTS, "notrade": NO_TRADE_WEIGHTS}


def _entrant_for(
    name: str, weights, depth: int, width: int | None, profile: str
) -> Entrant:
    return Entrant(
        name=name, kind="heximax", weights=weights, depth=depth, width=width,
        mode="notrade" if profile == "notrade" else "honest",
        max_trades=0 if profile == "notrade" else None,
    )


def _duel(
    challenger,
    incumbent,
    games: int,
    *,
    seed: int,
    depth: int,
    width: int | None,
    workers: int,
    profile: str,
) -> dict:
    """Compare two seats per side on paired boards; include unfinished games."""
    a = _entrant_for("challenger", challenger, depth, width, profile)
    b = _entrant_for("incumbent", incumbent, depth, width, profile)
    result = compete([a, a, b, b], games, seed=seed, workers=workers)
    return side_metrics(result, (0, 1))


def ablate(
    term: str,
    games: int,
    *,
    seed: int,
    depth: int,
    width: int | None,
    workers: int,
    profile: str = "trading",
) -> dict:
    full = WEIGHTS[profile]
    return _duel(
        replace(full, **{term: 0.0}),
        full,
        games,
        seed=seed,
        depth=depth,
        width=width,
        workers=workers,
        profile=profile,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--profile", choices=sorted(WEIGHTS), default="trading")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    terms = [f.name for f in fields(WEIGHTS[args.profile])]
    started = time.perf_counter()
    rows = []
    for term in terms:
        metrics = ablate(
            term,
            args.games,
            seed=args.seed,
            depth=args.depth,
            width=args.width,
            workers=args.workers,
            profile=args.profile,
        )
        low, high = metrics["interval_95"]
        rows.append(
            {
                "term": term,
                **metrics,
                # A term matters if removing it drops the side below half, so
                # the interval has to sit clear of 0.5 to say anything.
                "matters": not metrics["unfinished"] and high < 0.5,
            }
        )
        if not args.json:
            row = rows[-1]
            verdict = "matters" if row["matters"] else "not shown"
            print(
                f"  without {term:<14} {row['wins']:>4}/{row['games']}  {row['win_rate']:6.1%}"
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
