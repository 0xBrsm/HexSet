# SPDX-License-Identifier: GPL-3.0-only
"""Generate a dataset of recorded games.

Writes JSON lines, appending, so a run can be resumed or several runs pooled
into one file. Records hold board and actions rather than features, so the same
file can be re-encoded whenever the encoder changes.

The games come from `hexset.arena.compete` with one entrant in every seat and
`records=True`: a dataset is a tournament's own games, not a second play loop
that has to be kept in step with it. `--games` must therefore be a multiple of
`--players`, so the seat rotation completes and the dataset is not seat-biased.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from hexset.bench.throughput import default_workers, environment
from hexset.arena import compete, entrant_from_name
from hexset.record import write


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="JSON lines file, appended to")
    parser.add_argument("--games", type=int, default=100)
    # A preset name, or any checkpoint-prefixed entrant the arena resolves —
    # `network:<path>` records a trained policy's self-play, which is what
    # `hexset.bench.behaviour` needs to profile a checkpoint's style.
    parser.add_argument("--bot", default="greedy")
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.games % args.players:
        parser.error(
            f"--games must be a multiple of --players ({args.players}): the "
            "arena rotates the lineup through every seat and an incomplete "
            "rotation leaves the dataset seat-biased"
        )

    # One entrant in every seat, played through `hexset.arena.compete` -- the
    # one loop in this package that plays a game -- with `records=True`, so a
    # generated dataset is exactly what a tournament of the same bot played.
    entrant = entrant_from_name(args.bot)
    started = time.perf_counter()
    tournament = compete(
        [entrant] * args.players,
        args.games,
        seed=args.seed,
        workers=args.workers,
        records=True,
    )
    elapsed = time.perf_counter() - started
    records = tournament.records

    written = write(args.out, records)
    decided = sum(1 for r in records if r.decided)
    payload = {
        "environment": environment(),
        "out": args.out,
        "bot": args.bot,
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
    print(f"{written} {args.bot} games -> {args.out} ({args.workers} worker(s))")
    print(f"  {payload['games_per_second']} games/sec, {payload['seconds']}s")
    print(f"  {decided}/{written} decided, {payload['mean_turns']} turns/game")
    print(f"  {payload['mean_actions']} actions/game, {payload['bytes']} bytes on disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
