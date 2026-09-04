"""Recompute MARGINAL_SCALE exactly as its comment defines it, one seed per worker."""
import random, sys
from multiprocessing import Pool

import hexset.bots  # noqa
from hexset.actions import apply
from hexset.arena import PRESETS, spawn
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES
from hexset.bots.heximax import TRADING_WEIGHTS, HonestEvaluator
from hexset.bots.heximax.search import Heximax
from hexset.game import is_over, start, to_move

SEEDS = (100, 101, 102, 103, 104)


def one(seed):
    rng = random.Random(seed)
    board = random_base_board(rng)
    game = start(board, 4, rng)
    bots = [
        spawn(PRESETS["heximax-notrade"], board, random.Random(f"{seed}:{seat}"))
        for seat in range(4)
    ]
    game.gates = tuple(bots)
    meter = Heximax(HonestEvaluator(board, TRADING_WEIGHTS))
    total = 0.0
    count = 0
    while not is_over(game):
        seat = to_move(game)
        view = game.state(seat)
        for r in range(NUM_RESOURCES):
            total += abs(meter._marginal_gain(view, r))
            count += 1
        meter.evaluator._walk_cache.clear()
        meter.evaluator._belief_cache.clear()
        meter.evaluator._evaluate_cache.clear()
        apply(game, bots[seat].choose(game))
    return total, count


if __name__ == "__main__":
    with Pool(5) as pool:
        parts = pool.map(one, SEEDS)
    total = sum(t for t, _ in parts)
    count = sum(c for _, c in parts)
    print(f"marginals={count}")
    print(repr(total / count))
