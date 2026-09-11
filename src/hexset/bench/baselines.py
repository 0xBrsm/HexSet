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

from hexset.bench.metrics import json_metrics, paired_mean, side_metrics
from hexset.experiment import provenance, result_document
from hexset.bench.throughput import default_workers, environment
from hexset.arena import base_name, compete, lineup_from_names, pooled

DEFAULT_LINEUP = ("heximax", "heximax", "random", "random")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lineup",
        nargs="+",
        default=list(DEFAULT_LINEUP),
        metavar="BOT",
        help=(
            "one bot per seat: a preset name (random, heximax, heximax:pin-weights=0) or "
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

    lineup = lineup_from_names(args.lineup)
    if args.against is not None and not 0 <= args.against < len(lineup):
        parser.error("--against must index an entrant in the lineup")
    run_provenance = provenance(lineup)
    result = compete(
        lineup,
        args.games,
        seed=args.seed,
        workers=args.workers,
    )
    payload = {
        "experiment": result_document(result, lineup, seed=args.seed, workers=args.workers,
                                      run_provenance=run_provenance),
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
            {"name": standing.name, **side_metrics(result, [e])}
            for e, standing in enumerate(result.standings)
        ],
        "pooled": [
            {"name": standing.name, **side_metrics(result, [
                e for e, entrant in enumerate(lineup)
                if base_name(entrant.name) == standing.name
            ])}
            for standing in pooled(result.standings, result.games)
        ],
    }

    if args.against is not None:
        # Subtracting within a game cancels the board and the dice, so a
        # difference far smaller than the win-rate interval can still be seen.
        rows = result.points
        payload["paired_points"] = [
            {
                "name": entrant.name,
                **asdict(paired_mean([row[e] - row[args.against] for row in rows])),
            }
            for e, entrant in enumerate(result.standings)
        ]

    if args.json:
        print(json.dumps(json_metrics(payload), indent=2, allow_nan=False))
        return 0

    env = payload["environment"]
    print(f"commit {env['commit']}  python {env['python']}  {env['machine']}")
    print(f"{result.games} games, seed {args.seed}, {result.seconds:.1f}s")
    for standing, row in zip(result.standings, payload["standings"]):
        low, high = row["interval_95"]
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
