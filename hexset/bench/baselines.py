# SPDX-License-Identifier: GPL-3.0-only
"""Run a lineup of baseline bots against each other and report win rates."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from dataclasses import asdict

from hexset.bench.duel import add_table_options
from hexset.bench.metrics import json_metrics, paired_mean, side_metrics
from hexset.experiment import provenance, result_document
from hexset.bench.throughput import default_workers, environment
from hexset.arena import RUNTIME_HELP, base_name, compete, lineup_from_names, load_runtime, pooled
from hexset.rules import game_type

__all__ = [
    "main",
]


def main(argv: list[str] | None = None) -> int:
    """`python -m hexset.bench.baselines`: play a lineup of bots against each
    other and report win rates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lineup",
        nargs="+",
        required=True,
        metavar="BOT",
        help=(
            "one bot per seat: a preset name, a spec a runtime registered, or "
            "network:<checkpoint> for a trained network"
        ),
    )
    parser.add_argument("--runtime", action="append", default=[], help=RUNTIME_HELP)
    parser.add_argument(
        "--games",
        type=int,
        default=40,
        help="must divide evenly over the seats, so the rotation completes",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    add_table_options(parser)
    parser.add_argument(
        "--against",
        type=int,
        default=None,
        metavar="SEAT",
        help="pair every entrant's terminal points against this one, within games",
    )
    parser.add_argument("--journal", default=None,
                        help="where each game is kept as it finishes (default under runs/baselines)")
    parser.add_argument("--resume", action="store_true",
                        help="continue the journal's run, playing only the games it lacks")
    args = parser.parse_args(argv)

    load_runtime(*args.runtime)
    lineup = lineup_from_names(args.lineup)
    if args.against is not None and not 0 <= args.against < len(lineup):
        parser.error("--against must index an entrant in the lineup")
    run_provenance = provenance(lineup)
    journal = args.journal or str(
        Path("runs/baselines") / f"{'-'.join(args.lineup)}.s{args.seed}.g{args.games}.games.jsonl")
    result = compete(
        lineup,
        args.games,
        seed=args.seed,
        workers=args.workers,
        worker_initializer=load_runtime if args.runtime else None,
        worker_initargs=tuple(args.runtime),
        game_type=game_type(args.game_type),
        turn_cap=args.turn_cap,
        trade_mode=args.trade_mode,
        board_mode=args.board,
        journal=journal,
        resume=args.resume,
    )
    payload = {
        "experiment": result_document(result, lineup, seed=args.seed, workers=args.workers,
                                      run_provenance=run_provenance),
        "environment": environment(),
        "lineup": args.lineup,
        "seed": args.seed,
        "workers": args.workers,
        "game_type": args.game_type,
        "board_mode": args.board,
        "turn_cap": args.turn_cap,
        "trade_mode": args.trade_mode,
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
        # Differencing within a game cancels the board and the dice.
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
