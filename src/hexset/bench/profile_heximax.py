# SPDX-License-Identifier: GPL-3.0-only
"""Profile a preset playing itself: where a decision's time goes.

Plays `--games` complete four-seat games, every seat the same preset, under
`cProfile`, single process (no worker pool -- this is a profiling tool, not a
throughput one). Reports wall time per game, decisions per game, ms per
decision (mean/p50/p95), and the top functions by cumulative and total time.
Saves the raw `.prof` to `--out` for `pstats`/`snakeviz` follow-up.

Mirrors `hexset.arena.play`/`_play_one`'s own game loop (board per game index,
one bot per seat, the publish-then-choose order) rather than importing `play`
directly, so a decision's wall time can be timed individually -- `arena.play`
only returns the finished `Game`.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import random
import statistics
import sys
import time

import hexset.bots  # noqa: F401 -- registers "heximax"/"heximax-notrade"/... presets

from hexset.actions import apply
from hexset.arena import MAX_ACTIONS, entrant_from_name, spawn
from hexset.board.board import random_base_board
from hexset.game import is_over, start, to_move
from hexset.trading import publish_valuation


def play_one_game(preset: str, seed: str, *, action_cap: int = MAX_ACTIONS):
    """One four-seat game, every seat `preset`. Returns (game, decision_times_s)."""
    entrant = entrant_from_name(preset)
    board = random_base_board(random.Random(f"{seed}:board"))
    bots = [spawn(entrant, board, random.Random(f"{seed}:{seat}")) for seat in range(4)]
    game = start(board, 4, random.Random(f"{seed}:game"))
    game.gates = tuple(bots)
    game.max_trades = None
    times: list[float] = []
    actions = 0
    while not is_over(game) and actions < action_cap:
        seat = to_move(game)
        bot = bots[seat]
        if game.publish_due(seat):
            publish_valuation(game, seat, bot)
        before = time.perf_counter()
        action = bot.choose(game)
        times.append(time.perf_counter() - before)
        apply(game, action)
        actions += 1
    return game, times


def run(preset: str, games: int, seed: int):
    """Play `games` games of `preset` under one profiler. Returns (profile, per-game seconds, decision times)."""
    profile = cProfile.Profile()
    per_game_seconds: list[float] = []
    decision_times: list[float] = []
    profile.enable()
    for i in range(games):
        start_t = time.perf_counter()
        _game, times = play_one_game(preset, f"{seed}:{i}")
        per_game_seconds.append(time.perf_counter() - start_t)
        decision_times.extend(times)
    profile.disable()
    return profile, per_game_seconds, decision_times


def _pctile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def report(preset: str, per_game_seconds: list[float], decision_times: list[float]) -> None:
    ms = [t * 1000.0 for t in decision_times]
    n_games = len(per_game_seconds)
    n_decisions = len(decision_times)
    print(f"\n=== {preset}: {n_games} games, {n_decisions} decisions ===")
    print(f"wall time per game:   mean {statistics.mean(per_game_seconds):.3f}s"
          f"  total {sum(per_game_seconds):.3f}s")
    print(f"decisions per game:   {n_decisions / n_games:.1f}")
    print(f"ms per decision:      mean {statistics.mean(ms):.3f}"
          f"  p50 {_pctile(ms, 0.50):.3f}  p95 {_pctile(ms, 0.95):.3f}")


def print_top(profile: cProfile.Profile, n: int = 25) -> None:
    stats = pstats.Stats(profile)
    print(f"\n--- top {n} by cumulative time ---")
    stats.sort_stats("cumulative").print_stats(n)
    print(f"\n--- top {n} by total (self) time ---")
    stats.sort_stats("tottime").print_stats(n)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", default="heximax")
    parser.add_argument("--out", default=None, help="path to write the .prof file")
    args = parser.parse_args(argv)

    profile, per_game_seconds, decision_times = run(args.preset, args.games, args.seed)
    report(args.preset, per_game_seconds, decision_times)
    print_top(profile)

    if args.out:
        profile.dump_stats(args.out)
        print(f"\nsaved profile: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
