# SPDX-License-Identifier: GPL-3.0-only
"""Fit `search2.WIN_TEMPERATURE` by maximum likelihood against real outcomes.

The `win` stance reads a per-seat score vector as a win probability:
`softmax(vector / T)[seat]`. `T` is the exchange rate between the units the
evaluation is written in -- victory points -- and the log-odds of actually
winning from the position it describes. It is fitted, not chosen: sample one
score vector per turn at the mover's first decision, label each with the seat
that eventually won, and pick the `T` minimising mean cross-entropy.

    python -m hexset.bench.win_temperature --games 48 --workers 30 --json

Why this has to be refitted whenever the weights are. `T` is not independent
of the vector it divides: scaling every weight by k and dividing T by k give
the identical stance, so the pair (weights, T) is identified only jointly.
The shipped 2.476644394795811 was fitted against vectors produced by the
`progress`/`card`/`surplus_card` terms, which no longer exist. Carrying it
forward across a change to those terms silently rescales how sharply the bot
reads every position.

That coupling also makes this fit self-referential: the games sampled here
are played by a bot whose own stance reads `T`. `--temperature` overrides the
constant during sampling, so a caller can iterate to a fixed point -- fit,
re-sample at the fitted value, refit -- rather than assume one pass lands on
it. `--iterations` does that loop and reports every step, so the movement
between them is on the record instead of being asserted to be small.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from multiprocessing import Pool

from hexset.actions import apply, legal_actions
from hexset.arena import MAX_ACTIONS
from hexset.board.board import random_base_board
from hexset.bench.throughput import default_workers, environment
from hexset.bots import search2
from hexset.bots.heximax.search import heximax
from hexset.game import Phase, is_over, start, to_move

MAX_WORKERS = 30

# The bracket the search runs in. Wide enough to contain any sane exchange
# rate between victory points and win log-odds, and searched in log space
# because `T` is a scale: halving it and doubling it are the same size of
# step, which a linear bracket would not treat alike.
T_LOW = 0.05
T_HIGH = 200.0


def _play_one(job: tuple[int, int, float | None]) -> tuple[list[list[float]], int, int]:
    """Play one game, returning (vectors, winner seat, seats).

    One sample per turn, taken at the mover's first decision of that turn and
    read through that mover's own honest evaluator -- the same call the search
    makes at its root, so the vectors fitted here are the vectors the stance
    is actually applied to. A game the winner never reaches contributes
    nothing: there is no label to fit against.
    """
    index, seed, temperature = job
    if temperature is not None:
        search2.WIN_TEMPERATURE = temperature

    rng = random.Random(f"{seed}:{index}:game")
    board = random_base_board(random.Random(f"{seed}:{index}:board"))
    seats = 4
    game = start(board, seats, rng)
    bots = [
        heximax(board, random.Random(f"{seed}:{index}:{s}")) for s in range(seats)
    ]

    vectors: list[list[float]] = []
    seen: set[tuple[int, int]] = set()
    for _ in range(MAX_ACTIONS):
        if is_over(game) or not legal_actions(game):
            break
        seat = to_move(game)
        if game.phase is Phase.MAIN:
            key = (game.turns, seat)
            if key not in seen:
                seen.add(key)
                vectors.append(bots[seat].evaluator.evaluate_game(game, seat))
        apply(game, bots[seat].choose(game))

    return vectors, (-1 if game.won_by is None else game.won_by), seats


def mean_log_loss(samples: list[tuple[list[float], int]], temperature: float) -> float:
    """Mean cross-entropy of `softmax(vector / T)` against the winning seat."""
    total = 0.0
    for vector, winner in samples:
        scaled = [v / temperature for v in vector]
        m = max(scaled)
        exps = [math.exp(s - m) for s in scaled]
        total += (m + math.log(sum(exps))) - scaled[winner]
    return total / len(samples)


def fit_temperature(samples: list[tuple[list[float], int]]) -> tuple[float, float]:
    """The `T` minimising `mean_log_loss`, and the loss there.

    Golden-section over `log T`. The loss is smooth in `log T` and, on every
    sample set seen so far, single-troughed: at `T -> 0` the softmax hardens
    onto whichever seat leads and is confidently wrong whenever the leader
    loses; at `T -> infinity` it flattens to the uniform baseline. The
    minimum is between. A coarse scan over the bracket first, so a bracket
    that turns out to be multi-troughed is refined around the better trough
    rather than whichever end the section happens to start from.
    """
    lo, hi = math.log(T_LOW), math.log(T_HIGH)
    grid = [lo + (hi - lo) * i / 40 for i in range(41)]
    best = min(grid, key=lambda x: mean_log_loss(samples, math.exp(x)))
    lo, hi = max(lo, best - (hi - lo) / 40), min(hi, best + (hi - lo) / 40)

    phi = (math.sqrt(5) - 1) / 2
    c, d = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fc, fd = (
        mean_log_loss(samples, math.exp(c)),
        mean_log_loss(samples, math.exp(d)),
    )
    for _ in range(80):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - phi * (hi - lo)
            fc = mean_log_loss(samples, math.exp(c))
        else:
            lo, c, fc = c, d, fd
            d = lo + phi * (hi - lo)
            fd = mean_log_loss(samples, math.exp(d))
    temperature = math.exp((lo + hi) / 2)
    return temperature, mean_log_loss(samples, temperature)


def collect(games: int, seed: int, workers: int, temperature: float | None) -> tuple[
    list[tuple[list[float], int]], int
]:
    """Sample vectors from `games` games. Returns (samples, games that decided)."""
    jobs = [(i, seed, temperature) for i in range(games)]
    with Pool(workers) as pool:
        results = pool.map(_play_one, jobs)
    samples: list[tuple[list[float], int]] = []
    decided = 0
    for vectors, winner, _seats in results:
        if winner < 0:
            continue
        decided += 1
        samples.extend((vector, winner) for vector in vectors)
    return samples, decided


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=48)
    parser.add_argument("--seed", type=int, default=40000)
    parser.add_argument("--workers", type=int, default=min(MAX_WORKERS, default_workers()))
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="sample at this temperature instead of the shipped constant",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="re-sample at each fitted value and refit, this many times, so "
        "the fixed point is reached rather than assumed",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.workers > MAX_WORKERS:
        parser.error(f"--workers must be at most {MAX_WORKERS}")

    started = time.perf_counter()
    rounds: list[dict] = []
    temperature = args.temperature
    for step in range(args.iterations):
        samples, decided = collect(args.games, args.seed + step, args.workers, temperature)
        if not samples:
            parser.error("no decided games: nothing to fit against")
        fitted, loss = fit_temperature(samples)
        # The floor any temperature has to beat: a bot that reads every
        # position as a dead heat. `log(seats)` at four seats is 1.386.
        uniform = mean_log_loss(samples, float("inf"))
        rounds.append(
            {
                "sampled_at": temperature,
                "temperature": fitted,
                "log_loss": loss,
                "uniform_log_loss": uniform,
                "samples": len(samples),
                "decided_games": decided,
            }
        )
        print(
            f"  round {step}: sampled at "
            f"{'shipped' if temperature is None else f'{temperature:.6f}'} "
            f"-> T={fitted:.6f}  loss {loss:.4f} vs uniform {uniform:.4f}  "
            f"({len(samples)} samples, {decided}/{args.games} decided)",
            file=sys.stderr,
            flush=True,
        )
        temperature = fitted

    payload = {
        "environment": environment(),
        "settings": vars(args),
        "seconds": round(time.perf_counter() - started, 1),
        "rounds": rounds,
        "temperature": rounds[-1]["temperature"],
    }
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"WIN_TEMPERATURE = {payload['temperature']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
