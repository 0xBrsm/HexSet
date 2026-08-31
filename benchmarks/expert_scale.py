# SPDX-License-Identifier: GPL-3.0-only
"""Scale expert collection across independent CPU search workers.

Each worker owns its network and collector, warms compilation independently,
then waits at one barrier. The reported wall rate therefore measures steady
collection under real CPU/cache contention without billing staggered compiler
startup to whichever worker happened to become ready first.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys

from benchmarks.expert_cost import measure
from benchmarks.throughput import environment


def _measure(queue, barrier, kwargs) -> None:
    try:
        queue.put((True, measure(**kwargs, barrier=barrier)))
    except BaseException as error:
        queue.put((False, repr(error)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--moves", type=int, default=200, help="ticks per worker")
    parser.add_argument("--simulations", type=int, default=256)
    parser.add_argument("--wave", type=int, default=16)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", dest="compile_mode", default="default")
    parser.add_argument("--actions-per-game", type=int, default=950)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    context = mp.get_context("spawn")
    points = []
    for workers in args.workers:
        barrier = context.Barrier(workers)
        queue = context.Queue()
        processes = []
        for worker in range(workers):
            kwargs = dict(
                checkpoint=args.checkpoint,
                simulations=args.simulations,
                wave=args.wave,
                lanes=1,
                moves=args.moves,
                players=args.players,
                seed=args.seed + worker,
                device="cpu",
                compile_mode=args.compile_mode,
                inference_batch=None,
                max_offers=None,
                actions_per_game=args.actions_per_game,
            )
            process = context.Process(target=_measure, args=(queue, barrier, kwargs))
            process.start()
            processes.append(process)

        results = [queue.get() for _ in processes]
        for process in processes:
            process.join()
        errors = [value for ok, value in results if not ok]
        if errors:
            raise RuntimeError("; ".join(errors))

        measured = [value for ok, value in results if ok]
        seconds = max(point.seconds for point in measured)
        decisions = sum(point.moves for point in measured)
        points.append(
            {
                "workers": workers,
                "seconds": seconds,
                "decisions": decisions,
                "decisions_per_second": round(decisions / seconds, 2),
                "games_per_hour": round(
                    decisions / seconds * 3600 / args.actions_per_game, 2
                ),
                "worker_ms_per_move": [point.ms_per_move for point in measured],
            }
        )

    payload = {
        "environment": environment(),
        "checkpoint": args.checkpoint,
        "compile_mode": args.compile_mode,
        "simulations": args.simulations,
        "wave": args.wave,
        "moves_per_worker": args.moves,
        "points": points,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for point in points:
            print(
                f"{point['workers']:>2} workers  "
                f"{point['decisions_per_second']:>6.2f} decisions/s  "
                f"{point['games_per_hour']:>7.2f} games/h"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
