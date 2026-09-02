# SPDX-License-Identifier: GPL-3.0-only
"""Run a lineup of baseline bots against each other and report win rates.

This is the measurement the network will eventually have to beat, so it records
the commit and environment alongside the result rather than leaving the number
floating free.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from benchmarks.throughput import default_workers, environment
from hexset.arena import Z_95, compete, lineup_from_names, mean_interval, pooled

DEFAULT_LINEUP = ("greedy", "greedy", "random", "random")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lineup",
        nargs="+",
        default=list(DEFAULT_LINEUP),
        metavar="BOT",
        help=(
            "one bot per seat: a preset name (random, greedy, search2, ...) or "
            "network:<checkpoint> for a trained network"
        ),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=40,
        help="must divide evenly over the seats, so the rotation completes",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.add_argument(
        "--against",
        type=int,
        default=None,
        metavar="SEAT",
        help="pair every entrant's terminal points against this one, within games",
    )
    args = parser.parse_args(argv)

    result = compete(
        lineup_from_names(args.lineup),
        args.games,
        seed=args.seed,
        workers=args.workers,
    )
    payload = {
        "environment": environment(),
        "lineup": args.lineup,
        "seed": args.seed,
        "workers": args.workers,
        "games": result.games,
        "unfinished": result.unfinished,
        "mean_turns": round(result.mean_turns, 1),
        "seconds": round(result.seconds, 1),
        "seat_wins": list(result.seat_wins),
        "standings": [
            {
                "name": standing.name,
                "wins": standing.wins,
                "win_rate": round(standing.win_rate, 3),
                "interval_95": [round(bound, 3) for bound in standing.interval(Z_95)],
            }
            for standing in result.standings
        ],
        # Two seats a side is the usual duel, so the side's share of the games
        # is the number to quote and each seat's quarter is not.
        "pooled": [
            {
                "name": standing.name,
                "wins": standing.wins,
                "win_rate": round(standing.win_rate, 4),
                "interval_95": [round(bound, 4) for bound in standing.interval(Z_95)],
            }
            for standing in pooled(result.standings, result.games)
        ],
    }

    if args.against is not None:
        # Subtracting within a game cancels the board and the dice, so a
        # difference far smaller than the win-rate interval can still be seen.
        rows = [row for _, row in result.decided()]
        payload["paired_points"] = [
            {
                "name": entrant.name,
                **asdict(mean_interval([row[e] - row[args.against] for row in rows])),
            }
            for e, entrant in enumerate(result.standings)
        ]

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
    if len(payload["pooled"]) < len(payload["standings"]):
        print("  pooled by side:")
        for row in payload["pooled"]:
            low, high = row["interval_95"]
            print(
                f"    {row['name']:<10} {row['wins']:>4}/{result.games}"
                f"  {row['win_rate']:6.1%}  95% CI [{low:.1%}, {high:.1%}]"
            )
    for row in payload.get("paired_points", []):
        print(
            f"  {row['name']:<10} {row['mean']:+6.3f} points vs seat {args.against}"
            f"  95% CI [{row['lower']:+.3f}, {row['upper']:+.3f}]"
        )
    print(f"  {result.mean_turns:.1f} turns/game, {result.unfinished} unfinished")
    print("  by seat (even is what rotation and a correct snake draft should give):")
    for seat, wins, (low, high) in result.seat_balance():
        share = wins / max(1, sum(result.seat_wins))
        print(f"    seat {seat}  {wins:>4}  {share:6.1%}  95% CI [{low:.1%}, {high:.1%}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
