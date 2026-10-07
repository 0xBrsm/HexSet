# SPDX-License-Identifier: GPL-3.0-only
"""Generate a dataset of recorded games as appended JSON lines.

Records hold board and actions, not features, so a file can be re-encoded when
the encoder changes. `--games` must be a multiple of `--players`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from hexset.bench.throughput import default_workers, environment
from hexset.arena import RUNTIME_HELP, compete, entrant_from_name, load_runtime
from hexset.record import write
from hexset.bench.duel import add_table_options
from hexset.rules import game_type

__all__ = [
    "main",
]


def main(argv: list[str] | None = None) -> int:
    """`python -m hexset.bench.generate`: append a dataset of recorded games to
    a JSON lines file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="JSON lines file, appended to")
    parser.add_argument("--games", type=int, default=100)
    # A preset name, or any entrant spec the arena resolves, e.g. network:<path>.
    parser.add_argument("--bot", required=True)
    parser.add_argument("--runtime", action="append", default=[], help=RUNTIME_HELP)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    # The records carry the game type, so a fit reads each game at the
    # points it was actually played to.
    add_table_options(parser)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--journal", default=None,
                        help="where each game is kept as it finishes (default: beside --out)")
    parser.add_argument("--resume", action="store_true",
                        help="continue the journal's run, playing only the games it lacks")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.games % args.players:
        parser.error(
            f"--games must be a multiple of --players ({args.players}): the "
            "arena rotates the lineup through every seat and an incomplete "
            "rotation leaves the dataset seat-biased"
        )

    load_runtime(*args.runtime)
    entrant = entrant_from_name(args.bot)
    started = time.perf_counter()
    # Each game and its record are kept in the journal as they finish; the
    # flat file below is written from them once the run is whole.
    journal = args.journal or f"{args.out}.s{args.seed}.g{args.games}.journal.jsonl"
    tournament = compete(
        [entrant] * args.players,
        args.games,
        seed=args.seed,
        workers=args.workers,
        records=True,
        game_type=game_type(args.game_type),
        turn_cap=args.turn_cap,
        trade_mode=args.trade_mode,
        board_mode=args.board,
        worker_initializer=load_runtime if args.runtime else None,
        worker_initargs=tuple(args.runtime),
        journal=journal,
        resume=args.resume,
    )
    elapsed = time.perf_counter() - started
    records = tournament.records

    written = write(args.out, records)
    decided = sum(1 for r in records if r.decided)
    payload = {
        "environment": environment(),
        "out": args.out,
        "bot": args.bot,
        "game_type": args.game_type,
        "board_mode": args.board,
        "turn_cap": args.turn_cap,
        "trade_mode": args.trade_mode,
        "games": written,
        "workers": args.workers,
        "seed": args.seed,
        "decided": decided,
        "seconds": round(elapsed, 1),
        "games_per_second": round(args.games / elapsed, 2),
        "mean_turns": round(sum(r.turns for r in records) / len(records), 1),
        "mean_actions": round(sum(len(r.actions) for r in records) / len(records), 1),
        "bytes": os.path.getsize(args.out),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    env = payload["environment"]
    print(f"commit {env['commit']}  {env['machine']}")
    print(f"{written} {args.bot} {args.game_type} games -> {args.out} "
          f"({args.workers} worker(s))")
    print(f"  {payload['games_per_second']} games/sec, {payload['seconds']}s")
    print(f"  {decided}/{written} decided, {payload['mean_turns']} turns/game")
    print(f"  {payload['mean_actions']} actions/game, {payload['bytes']} bytes on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
