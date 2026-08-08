"""Generate a dataset of recorded games.

Writes JSON lines, appending, so a run can be resumed or several runs pooled
into one file. Records hold board and actions rather than features, so the same
file can be re-encoded whenever the encoder changes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

from benchmarks.throughput import environment
from catan.arena import FACTORIES
from catan.board.board import random_base_board
from catan.record import Record, record_game, write

BOARD_SEED_OFFSET = 1_000_000


def _record_one(job: tuple[int, str, int]) -> Record:
    seed, bot, players = job
    board = random_base_board(random.Random(BOARD_SEED_OFFSET + seed))
    factory = FACTORIES[bot]
    bots = [factory(board, random.Random(seed * 16 + seat)) for seat in range(players)]
    return record_game(bots, board, seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="JSON lines file, appended to")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--bot", default="greedy", choices=sorted(FACTORIES))
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    jobs = [(args.seed + i, args.bot, args.players) for i in range(args.games)]
    started = time.perf_counter()
    if args.workers > 1:
        with Pool(args.workers) as pool:
            records = pool.map(
                _record_one, jobs, chunksize=max(1, args.games // (args.workers * 4))
            )
    else:
        records = [_record_one(job) for job in jobs]
    elapsed = time.perf_counter() - started

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
