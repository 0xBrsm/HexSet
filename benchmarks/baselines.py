"""Run a lineup of baseline bots against each other and report win rates.

This is the measurement the network will eventually have to beat, so it records
the commit and environment alongside the result rather than leaving the number
floating free.
"""

from __future__ import annotations

import argparse
import json
import sys

from benchmarks.throughput import environment
from catan.arena import Z_95, compete, lineup_from_names

DEFAULT_LINEUP = ("greedy", "greedy", "random", "random")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lineup",
        nargs="+",
        default=list(DEFAULT_LINEUP),
        metavar="BOT",
        help="one bot per seat: random, greedy, search2, search3",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=40,
        help="must divide evenly over the seats, so the rotation completes",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args(argv)

    result = compete(lineup_from_names(args.lineup), args.games, seed=args.seed)
    payload = {
        "environment": environment(),
        "lineup": args.lineup,
        "seed": args.seed,
        "games": result.games,
        "unfinished": result.unfinished,
        "mean_turns": round(result.mean_turns, 1),
        "seconds": round(result.seconds, 1),
        "standings": [
            {
                "name": standing.name,
                "wins": standing.wins,
                "win_rate": round(standing.win_rate, 3),
                "interval_95": [round(bound, 3) for bound in standing.interval(Z_95)],
            }
            for standing in result.standings
        ],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    env = payload["environment"]
    print(f"commit {env['commit']}  python {env['python']}  {env['machine']}")
    print(f"{result.games} games, seed {args.seed}, {result.seconds:.1f}s")
    for standing in result.standings:
        low, high = standing.interval(Z_95)
        print(
            f"  {standing.name:<10} {standing.wins:>4}/{result.games}"
            f"  {standing.win_rate:6.1%}  95% CI [{low:.1%}, {high:.1%}]"
        )
    print(f"  {result.mean_turns:.1f} turns/game, {result.unfinished} unfinished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
