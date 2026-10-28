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
from pathlib import Path

from hexset.arena import PRESETS, compete
from hexset.game import UNSTRUCTURED_TURN_CAP
from hexset.experiment import provenance
from hexset.seeds import add_arguments as add_seed_arguments, resolve as resolve_seed

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


def run(
    games: int, players: int, seed: int | None, workers: int, journal: str | os.PathLike,
    *, seed_bypass: str | None = None, registration: str | os.PathLike | None = None,
) -> Result:
    """Time `games` games with the `random` entrant in all `players` seats,
    each kept in `journal` as `compete` keeps every game, as a `Result`.
    `seed`, `seed_bypass` and `registration` are `compete`'s."""
    tournament = compete(
        [PRESETS["random"]] * players, games, seed=seed, workers=workers,
        turn_cap=UNSTRUCTURED_TURN_CAP, journal=journal,
        seed_bypass=seed_bypass, registration=registration,
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
    """Available CPUs, respecting affinity and a container's CPU quota."""
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            available = min(available, max(1, int(quota) // int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    return max(1, available)


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
    add_seed_arguments(parser)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.add_argument("--journal", default=None,
                        help="where the games are kept (default: throughput.p<players>.s<seed>.g<games>"
                             ".games.jsonl here); a journal that already holds a run is refused")
    args = parser.parse_args(argv)

    if args.games % args.players:
        parser.error(
            f"--games must be a multiple of --players ({args.players}): the "
            "arena rotates its lineup through every seat"
        )

    resolve_seed(args, args.players)
    journal = args.journal or f"throughput.p{args.players}.s{args.seed}.g{args.games}.games.jsonl"
    result = run(args.games, args.players, args.seed, args.workers, journal,
                 seed_bypass=args.bypass_seed_policy, registration=args.registration)
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
