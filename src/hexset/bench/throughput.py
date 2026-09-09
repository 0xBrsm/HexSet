# SPDX-License-Identifier: GPL-3.0-only
"""Measure how fast the engine can play random games.

Self-play cost is dominated by simulator throughput when search is in the
training loop, so this number is what decides whether a Python engine is
viable or whether it has to be rewritten in a compiled language.

The games are `hexset.arena.compete`'s, with the `random` entrant in every
seat: the figure quoted for the engine has to come from the loop everything
else in this package plays through, or it is a figure for a loop nobody uses.
`--games` must therefore be a multiple of `--players`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass

from hexset.arena import PRESETS, compete
from hexset.experiment import provenance


@dataclass
class Result:
    games: int
    players: int
    workers: int
    seconds: float
    games_per_second: float
    mean_turns: float
    finished_by_win: int


def run(games: int, players: int, seed: int, workers: int) -> Result:
    tournament = compete(
        [PRESETS["random"]] * players, games, seed=seed, workers=workers
    )
    return Result(
        games=games,
        players=players,
        workers=workers,
        seconds=round(tournament.seconds, 3),
        games_per_second=round(games / tournament.seconds, 1),
        mean_turns=round(tournament.mean_turns, 1),
        finished_by_win=games - tournament.unfinished,
    )


def default_workers() -> int:
    """Every core by default.

    Defaulting to one meant a forgotten flag silently used a single core, and
    that is exactly the kind of mistake that gets written into a results table
    as a throughput figure. Runners print the count they used.
    """
    return os.cpu_count() or 1


def environment() -> dict[str, str]:
    """Compact compatibility view of the shared experiment provenance."""
    source = provenance()
    return {
        "commit": source["commit"][:7] if source["commit"] else "unknown",
        "dirty": "unknown" if source["dirty"] is None else str(source["dirty"]).lower(),
        "python": source["python"],
        "platform": source["platform"],
        "machine": platform.machine(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args(argv)

    if args.games % args.players:
        parser.error(
            f"--games must be a multiple of --players ({args.players}): the "
            "arena rotates its lineup through every seat"
        )

    result = run(args.games, args.players, args.seed, args.workers)
    payload = {"environment": environment(), **asdict(result)}

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    env = payload["environment"]
    print(f"commit {env['commit']}  python {env['python']}  {env['machine']}")
    print(f"{result.games} games, {result.players} players, {result.workers} worker(s)")
    print(f"  {result.seconds}s total")
    print(f"  {result.games_per_second} games/sec")
    print(f"  {result.mean_turns} turns/game")
    print(f"  {result.finished_by_win}/{result.games} ended in a win")
    return 0


if __name__ == "__main__":
    sys.exit(main())
