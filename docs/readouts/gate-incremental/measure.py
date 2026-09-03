# SPDX-License-Identifier: GPL-3.0-only
"""Before/after readout for the incremental trade gate (commit 2 of
perf/gate-rows): 200 games of `heximax` vs itself, one process, same board
and bot seeds both runs -- run once against this tree, once with the
incremental `_delta`/`SearchBot.accepts` fast paths reverted (`git stash`),
diffing only the trade gate's own implementation, nothing else. Reports
trades/turn, mean seconds/game, and mean ms/decision. Not part of the
package; a one-off measurement script, kept beside its own README.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time

sys.path.insert(0, "src")

import hexset.bots  # noqa: F401 -- registers "heximax"

from hexset.actions import apply
from hexset.arena import PRESETS, spawn
from hexset.board.board import random_base_board
from hexset.game import is_over, start, to_move
from hexset.trading import publish_valuation

GAMES = 200
SEED = 424242
PLAYERS = 4
ACTION_CAP = 400


def play_one(index: int):
    board = random_base_board(random.Random(f"{SEED}:{index}:board"))
    bots = [
        spawn(PRESETS["heximax"], board, random.Random(f"{SEED}:{index}:{seat}"))
        for seat in range(PLAYERS)
    ]
    game = start(board, PLAYERS, random.Random(f"{SEED}:{index}:game"))
    game.gates = tuple(bots)
    game.max_trades = None
    times: list[float] = []
    actions = 0
    while not is_over(game) and actions < ACTION_CAP:
        seat = to_move(game)
        bot = bots[seat]
        if game.publish_due(seat):
            publish_valuation(game, seat, bot)
        before = time.perf_counter()
        action = bot.choose(game)
        times.append(time.perf_counter() - before)
        apply(game, action)
        actions += 1
    return game.turns, game.trades_made, times


def main() -> None:
    started = time.perf_counter()
    per_game_seconds: list[float] = []
    per_game_trades_per_turn: list[float] = []
    all_decision_ms: list[float] = []
    total_turns = 0
    total_trades = 0
    for i in range(GAMES):
        t0 = time.perf_counter()
        turns, trades, times = play_one(i)
        per_game_seconds.append(time.perf_counter() - t0)
        per_game_trades_per_turn.append(trades / turns if turns else 0.0)
        all_decision_ms.extend(t * 1000.0 for t in times)
        total_turns += turns
        total_trades += trades
    elapsed = time.perf_counter() - started

    result = {
        "games": GAMES,
        "seed": SEED,
        "wall_seconds": round(elapsed, 1),
        "mean_seconds_per_game": statistics.mean(per_game_seconds),
        "mean_trades_per_turn": total_trades / total_turns if total_turns else 0.0,
        "mean_trades_per_turn_per_game": statistics.mean(per_game_trades_per_turn),
        "total_turns": total_turns,
        "total_trades": total_trades,
        "mean_ms_per_decision": statistics.mean(all_decision_ms),
        "decisions": len(all_decision_ms),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
