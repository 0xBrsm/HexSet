"""How much of the value head's error no value head could remove.

`benchmarks.value_head` found the head accurate to about 2.1 victory points and
equally so on three different behaviour policies, which rules out the off-policy
explanation the search failure was filed under. What it cannot say is whether
2.1 points is a bad fit or a hard limit. A Monte Carlo target cannot beat the
conditional variance of its own outcome, and a Catan position twenty turns from
the end does not determine its terminal points — the dice do.

The stage split already hints at a limit: residual variance falls 0.071 → 0.015
from opening to endgame while target variance stays flat, on the same curve
under all three policies. That is what remaining dice look like. It is not what
a representational weakness looks like.

This measures it instead of inferring it. Snapshot a position, play it forward
many times under the same policy, and the spread of those terminal returns *is*
the irreducible variance at that position. The head's error there splits exactly
in two:

    E[(return - prediction)^2]  =  Var(return | position)  +  (E[return | position] - prediction)^2
             mean squared error  =         floor            +              bias^2

The identity is exact per position, so no modelling assumption enters. **The
fraction of the error that is floor is the number that decides what to do.**
Near 1 and the head is already close to optimal, more training is wasted, and
fixing the search means changing the target — bootstrapping off the search's own
backed-up value, or a shorter horizon. Well below 1 and the head is simply
underfitted and better training is the answer.

Rollouts branch with `catan.game.imagine`, which copies the position whole and
reshuffles only the deck. That is right rather than convenient: deck order is
not in the observation the head was given, so its uncertainty belongs in the
floor.

    python -m benchmarks.floor --checkpoint runs/ppo-overnight/latest.pt \\
        --positions 64 --rollouts 64
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, replace

import numpy as np

from benchmarks.throughput import environment
from catan.board.board import random_base_board
from catan.game import imagine
from catan.rewards import reward
from catan.selfplay import Collector, Episode

# Torch is imported inside `main`, so the sampling and branching machinery —
# everything with arithmetic worth getting wrong — stays testable without it.


@dataclass(frozen=True)
class Snapshot:
    """A position kept for replaying, with what the head said about it."""

    game: object
    seat: int
    prediction: float


class Sampling:
    """The policy, playing as usual, keeping a copy of the odd position.

    The copy is taken before the action is applied, and it is a copy: the lane
    game is handed out rather than cloned, so keeping the live object would tie
    the snapshot to a position that keeps moving.
    """

    def __init__(self, policy, *, rate, rng) -> None:
        self.policy = policy
        self.rate = rate
        self.rng = rng

    def act(self, requests):
        choices = self.policy.act(requests)
        for row, request in enumerate(requests):
            if self.rng.random() >= self.rate or not choices[row].value:
                continue
            choices[row] = replace(
                choices[row],
                aux=Snapshot(
                    game=imagine(request.game, self.rng),
                    seat=request.seat,
                    prediction=choices[row].value[0],
                ),
            )
        return choices


class Branching(Collector):
    """A collector whose every lane starts from one given position.

    Subclassed rather than parameterised because `catan.selfplay` is on the path
    of every result this project has recorded, and a benchmark should not be
    the reason it grows an argument.
    """

    def __init__(self, policy, position, *, rng, **kwargs) -> None:
        self._position = position
        self._rng = rng
        super().__init__(policy, **kwargs)

    def _fresh(self):
        lane = super()._fresh()
        if lane is None:
            return None
        lane.game = imagine(self._position, self._rng)
        return lane


def collect(episodes: list[Episode]) -> list[tuple[Snapshot, float]]:
    """Every kept position, with how far into its game it was."""
    out = []
    for episode in episodes:
        length = max(1, episode.outcome.actions - 1)
        for trajectory in episode.trajectories:
            for transition in trajectory:
                if isinstance(transition.aux, Snapshot):
                    out.append((transition.aux, transition.step / length))
    return out


def split(returns: np.ndarray, prediction: float) -> tuple[float, float]:
    """One position's error, as floor plus bias squared. Exact by construction."""
    floor = float(returns.var())
    bias = float(returns.mean() - prediction)
    return floor, bias * bias


def pool(rows: list[dict]) -> dict:
    """Aggregate floor/bias/mse over positions -- shared by `main` and a shard merge.

    Weights by row, not by shard: concatenating N shards' rows and pooling once
    is what makes the sharded total equal the single-process total it replaces.
    """
    floors = np.asarray([r["floor"] for r in rows])
    biases = np.asarray([r["bias_squared"] for r in rows])
    mses = floors + biases
    return {
        "mean_floor": round(float(floors.mean()), 5),
        "mean_bias_squared": round(float(biases.mean()), 5),
        "mean_squared_error": round(float(mses.mean()), 5),
        "irreducible_share": round(float(floors.mean() / max(mses.mean(), 1e-12)), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-games", type=int, default=24)
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--action-cap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    import torch

    from catan.netbot import load
    from catan.policy import NetworkPolicy

    board = random_base_board(random.Random(args.seed))
    loaded = load(args.checkpoint, board.topology)
    generator = torch.Generator().manual_seed(args.seed)
    policy = NetworkPolicy(
        loaded.policy.net,
        loaded.space,
        loaded.policy.layout,
        greedy=False,
        generator=generator,
    )

    started = time.perf_counter()
    rng = random.Random(args.seed + 2)
    seeding = Collector(
        Sampling(policy, rate=0.02, rng=rng),
        lanes=min(16, args.seed_games),
        players=args.players,
        seed=args.seed + 1,
        action_cap=args.action_cap,
        max_offers=loaded.max_offers,
        deal=args.seed_games,
        board=board,
    )
    kept = collect(seeding.drain())
    if len(kept) < args.positions:
        print(
            f"only {len(kept)} positions kept for {args.positions} asked; "
            "raise --seed-games",
            file=sys.stderr,
        )
    chosen = rng.sample(kept, min(args.positions, len(kept)))
    seeded = time.perf_counter() - started

    rows = []
    for snapshot, progress in chosen:
        branch = Branching(
            policy,
            snapshot.game,
            rng=rng,
            lanes=args.rollouts,
            players=args.players,
            seed=args.seed + 3,
            action_cap=args.action_cap,
            max_offers=loaded.max_offers,
            deal=args.rollouts,
            board=board,
        )
        returns = np.asarray(
            [reward(episode.outcome)[snapshot.seat] for episode in branch.drain()],
            dtype=np.float64,
        )
        floor, bias = split(returns, snapshot.prediction)
        rows.append(
            {
                "progress": round(progress, 3),
                "rollouts": int(returns.size),
                "floor": floor,
                "bias_squared": bias,
                "mse": floor + bias,
            }
        )
    elapsed = time.perf_counter() - started

    stages = _stages(rows, args.bins)

    payload = {
        "environment": environment(),
        "checkpoint": args.checkpoint,
        # So N shards, each its own `--seed`, can be pooled by concatenating
        # `rows` rather than re-deriving what one process was asked to do.
        "args": vars(args),
        "iteration": loaded.iteration,
        "positions": len(rows),
        "rollouts_each": args.rollouts,
        "seed_seconds": round(seeded, 1),
        "seconds": round(elapsed, 1),
        **pool(rows),
        "stages": stages,
        "rows": rows,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(
        f"{payload['positions']} positions x {args.rollouts} rollouts, "
        f"{payload['seconds']}s"
    )
    print(f"  mean squared error   {payload['mean_squared_error']:.5f}")
    print(f"  of which floor       {payload['mean_floor']:.5f}")
    print(f"  of which bias^2      {payload['mean_bias_squared']:.5f}")
    print(f"  irreducible share    {payload['irreducible_share']:.1%}")
    print("  by stage of the game:")
    for stage in stages:
        print(
            f"    {stage['from']:.2f}-{stage['to']:.2f}  {stage['positions']:>4} pos"
            f"  mse {stage['mean_squared_error']:.5f}"
            f"  floor {stage['mean_floor']:.5f}"
            f"  irreducible {stage['irreducible_share']:.1%}"
        )
    return 0


def _stages(rows: list[dict], bins: int) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    progress = np.asarray([r["progress"] for r in rows])
    floors = np.asarray([r["floor"] for r in rows])
    biases = np.asarray([r["bias_squared"] for r in rows])
    out = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (progress >= low) & (progress < high if high < 1.0 else progress <= 1.0)
        if not inside.any():
            continue
        mse = float((floors[inside] + biases[inside]).mean())
        out.append(
            {
                "from": round(float(low), 2),
                "to": round(float(high), 2),
                "positions": int(inside.sum()),
                "mean_floor": round(float(floors[inside].mean()), 5),
                "mean_bias_squared": round(float(biases[inside].mean()), 5),
                "mean_squared_error": round(mse, 5),
                "irreducible_share": round(
                    float(floors[inside].mean() / max(mse, 1e-12)), 4
                ),
            }
        )
    return out


if __name__ == "__main__":
    sys.exit(main())
