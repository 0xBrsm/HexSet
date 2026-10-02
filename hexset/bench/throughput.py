# SPDX-License-Identifier: GPL-3.0-only
"""Measure how fast the engine plays random games: `hexset.arena.compete` with
the `random` entrant in every seat. `--games` must be a multiple of `--players`.
Random play takes several times the turns real play does, so the games run
under `hexset.game.UNSTRUCTURED_TURN_CAP` rather than `MAX_TURNS`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass

from hexset.arena import PRESETS, compete
from hexset.game import UNSTRUCTURED_TURN_CAP
from hexset.experiment import provenance

__all__ = [
    "Result",
    "run",
    "default_workers",
    "environment",
    "main",
]


@dataclass
class Result:
    """One timing run (`run`): games played, seats and workers, wall seconds,
    the games-per-second rate, mean turns and how many games ended in a win."""

    games: int
    players: int
    workers: int
    seconds: float
    games_per_second: float
    mean_turns: float
    finished_by_win: int


def run(games: int, players: int, seed: int, workers: int) -> Result:
    """Time `games` games with the `random` entrant in all `players` seats, as
    a `Result`."""
    tournament = compete(
        [PRESETS["random"]] * players, games, seed=seed, workers=workers,
        turn_cap=UNSTRUCTURED_TURN_CAP,
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
    """Every core by default; runners print the count they used."""
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
    """`python -m hexset.bench.throughput`: measure how fast the engine plays
    random games."""
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
