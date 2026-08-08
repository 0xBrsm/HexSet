"""Measure how fast the engine can play random games.

Self-play cost is dominated by simulator throughput when search is in the
training loop, so this number is what decides whether a Python engine is
viable or whether it has to be rewritten in a compiled language.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool

from catan.actions import legal_actions
from catan.board.board import random_base_board
from catan.game import is_over, start
from catan.play import step_randomly


@dataclass
class Result:
    games: int
    players: int
    workers: int
    seconds: float
    games_per_second: float
    actions_per_second: float
    mean_turns: float
    mean_actions: float
    finished_by_win: int


def _play_one(args: tuple[int, int]) -> tuple[int, int, bool]:
    seed, players = args
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    actions = 0
    while not is_over(game):
        step_randomly(game, rng)
        actions += 1
    return game.turns, actions, game.won_by is not None


def _count_actions_once(players: int, seed: int) -> int:
    """Sanity check that enumeration cost is included in the measurement."""
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    return len(legal_actions(game))


def run(games: int, players: int, seed: int, workers: int) -> Result:
    jobs = [(seed + i, players) for i in range(games)]

    start_time = time.perf_counter()
    if workers > 1:
        with Pool(workers) as pool:
            outcomes = pool.map(_play_one, jobs, chunksize=max(1, games // (workers * 4)))
    else:
        outcomes = [_play_one(job) for job in jobs]
    elapsed = time.perf_counter() - start_time

    turns = [t for t, _, _ in outcomes]
    actions = [a for _, a, _ in outcomes]
    return Result(
        games=games,
        players=players,
        workers=workers,
        seconds=round(elapsed, 3),
        games_per_second=round(games / elapsed, 1),
        actions_per_second=round(sum(actions) / elapsed, 1),
        mean_turns=round(statistics.mean(turns), 1),
        mean_actions=round(statistics.mean(actions), 1),
        finished_by_win=sum(1 for _, _, won in outcomes if won),
    )


def environment() -> dict[str, str]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
    return {
        "commit": sha,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args(argv)

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
    print(f"  {result.actions_per_second} actions/sec")
    print(f"  {result.mean_turns} turns/game, {result.mean_actions} actions/game")
    print(f"  {result.finished_by_win}/{result.games} ended in a win")
    return 0


if __name__ == "__main__":
    sys.exit(main())
