# SPDX-License-Identifier: GPL-3.0-only
"""How much of the value target's noise a shorter horizon actually removes.

`benchmarks.floor` established that ~80% of the head's squared error against
terminal returns is the conditional variance of those returns — dice not yet
rolled. `--value-horizon` in `hexset.distill` responds by relabelling: a decision
is trained toward the estimate recorded `horizon` of that seat's own decisions
later, instead of the game's final score. This measures whether that relabelling
does what it is supposed to, because so far it is an argument and not a number.

## What is being claimed

For a position `s` and the state `s'` that the seat reaches `horizon` decisions
later, the law of total variance splits the terminal label exactly:

    Var(terminal | s)  =  Var( V(s') | s )  +  E[ Var(terminal | s') | s ]
                          `-- where you end up   `-- dice rolled after s'

The terminal target makes the head fit a label carrying both terms. The
bootstrapped target hands it one carrying only the first. **The second term
stops entering the label at all**, and that is the whole mechanism.

The mean survives the change: `E[V(s') | s] = E[terminal | s]` when `V` is
unbiased, so the differences between sibling positions — the 0.0119 spread
`benchmarks.sibling` measured, the signal a search needs — are the same size
while the noise around them shrinks. Same signal, less variance.

That matters because the data required to resolve a gap scales as
`(sigma / gap)^2`. Halving the label spread quarters it.

## What this reports, and the number to read

`variance_ratio` — `Var(V(s') | s)` over `Var(terminal | s)`, averaged over
positions. It is the fraction of the label's variance that survives the
horizon, so `sample_factor` (its reciprocal) is how much the data requirement
falls. A ratio of 1 means the horizon bought nothing.

**A ratio near zero is a failure, not a triumph**, and this is the reason the
benchmark reports the ratio rather than the bootstrapped variance alone. If the
horizon is short enough that `s'` is barely distinguishable from `s`, the label
collapses toward the head's own opinion of `s`, which it can already predict
perfectly and which teaches it nothing. That degeneracy and a genuine variance
reduction look identical in a training curve — both show falling value loss and
rising explained variance — and they are told apart precisely here.

`mean_gap` is a weak second guard: the average of `E[V(s')|s] - E[terminal|s]`
across positions. The tower property says it should be near zero, so a large
value means the head's bias is being fed back into its own target rather than
cancelled by it. It averages *signed* gaps, so it detects overall bias and
nothing else — it was originally the only guard here and it was not enough.

## The statistic that actually decides it

`teacher_ratio`, and it is the number to read before `variance_ratio`.

Training toward a label only teaches the head something if the label is closer
to the truth than what the head already says. So compare, per position, two
errors against the same target `E[terminal | s]`:

    rms_head_error   how far the head's own prediction V(s) is from the truth
    rms_label_error  how far the bootstrapped label E[V(s')|s] is from the truth

`teacher_ratio` is the second over the first. **Below 1 the label is a better
teacher than the head's current opinion, and there is something to learn. At or
above 1 the horizon is decoration**: the head is being trained toward a
restatement of what it already believes, which is the degenerate case
`variance_ratio` cannot see. A horizon of 0 would score a ratio of exactly 1
with a variance ratio of 0 — maximally "clean" and maximally useless.

`signal_correlation` is the same idea in correlation form, kept because it is
scale-free and so survives a head whose values are systematically compressed.

**The two means are computed from disjoint halves of the rollouts.** The label
and the terminal return of the *same* rollout share that rollout's dice, so
averaging both over one set of games correlates their errors and flatters the
label. Even-indexed rollouts estimate the label, odd-indexed estimate the
truth, which costs a factor of sqrt(2) in the precision of each and buys an
honest comparison.

What remains, and is not corrected for: `E[terminal | s]` is itself estimated
from finitely many rollouts, so both errors are inflated by the same estimation
noise and the ratio is pulled toward 1. `teacher_ratio` is therefore a
conservative reading — a value below 1 is more meaningful than a value near it.

## Two substitutions, both deliberate

**The rollouts are played by the raw policy, and `V(s')` is read from the value
head rather than from a search.** The training run uses `SearchPolicy`, whose
`Transition.value` is the search's backed-up root mean. Running a 128-simulation
search at every ply of every rollout would cost hours and would measure two
things at once: the structural variance reduction, and the search estimator's
own error. Only the first is in the identity above, so only the first is
measured here. The search's error is a separate additive term and belongs in a
separate measurement.

**Both labels come from the same rollouts.** `floor` already replays each
snapshot to completion, and a rollout that runs to the end passes through `s'` on
the way, so the bootstrapped label and the terminal label are read from one set
of games. That makes them paired — the comparison is free of the difference
between two samples of positions — and it costs nothing beyond what `floor`
already pays.

One frame note, because the two policies disagree and it is easy to get
backwards. `NetworkPolicy` writes `Choice.value` in the **mover's** frame, so
`value[0]` is the mover's own estimate and no rotation is needed. `SearchPolicy`
writes it in **board order**, which is why `distill._value_targets` rotates and
this file does not.

    python -m benchmarks.horizon --checkpoint runs/ppo-overnight/latest.pt \\
        --positions 64 --rollouts 64 --horizon 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import numpy as np

from benchmarks.floor import Branching, Sampling, collect
from benchmarks.throughput import environment
from hexset.board.board import random_base_board
from hexset.rewards import reward
from hexset.selfplay import Collector

# Torch is imported inside `main`, so the arithmetic below stays testable
# without it — the same split `benchmarks.floor` makes.


def labels(episode, seat: int, horizon: int) -> tuple[float, float, bool]:
    """One rollout's two labels: what the horizon says, and what the game said.

    Returns the bootstrapped label, the terminal label, and whether the rollout
    was long enough to actually bootstrap. The training code falls back to the
    terminal return when the trajectory ends inside the horizon, so this does
    too — and counts it, because a horizon that mostly falls off the end of the
    game is measuring the terminal target under another name.
    """
    terminal = float(reward(episode.outcome)[seat])
    trajectory = episode.trajectories[seat]
    if horizon < len(trajectory):
        estimate = trajectory[horizon].value
        if estimate:
            return float(estimate[0]), terminal, True
    return terminal, terminal, False


def lambda_label(episode, seat: int, lam: float) -> float:
    """One rollout's TD(lambda) label: every horizon at once, geometrically weighted.

    With gamma 1 and a reward that is zero until the game ends, the n-step
    return *is* the head's own estimate n of the seat's decisions later, and the
    full-length return is the terminal outcome. So the lambda-mixture is

        G = (1 - lam) * sum_{n=1}^{T-1} lam^(n-1) V(s_n)  +  lam^(T-1) * terminal

    which is `V(s_1)` at lam=0 and the terminal return at lam=1 — the two rows
    the horizon sweep already measured, as the ends of one dial.

    This is the object the sweep pointed at. Every fixed horizon was
    monotonically a better teacher than the one below it and monotonically
    noisier, with no turning point, so no single `n` is right; a mixture takes
    the quiet short labels and the informative long ones together instead of
    choosing. It is what TD-Gammon used, in the same dice-driven setting, for
    the same reason.

    Estimates missing from a trajectory (a forced move records none) are dropped
    and the weights renormalised over what is left, rather than the position
    being skipped.
    """
    terminal = float(reward(episode.outcome)[seat])
    trajectory = episode.trajectories[seat]
    if lam >= 1.0 or len(trajectory) < 2:
        return terminal
    weights, values = [], []
    for n in range(1, len(trajectory)):
        estimate = trajectory[n].value
        if not estimate:
            continue
        weights.append((1.0 - lam) * lam ** (n - 1))
        values.append(float(estimate[0]))
    tail = lam ** (len(trajectory) - 1)
    weights.append(tail)
    values.append(terminal)
    total = sum(weights)
    if total <= 0:
        return terminal
    return float(np.dot(weights, values) / total)


def teacher(rows: list[dict]) -> dict:
    """Is the label closer to the truth than the head's own prediction?

    The condition under which bootstrapping teaches anything, reduced to one
    ratio. See the module docstring for why this and not `variance_ratio`.
    """
    head = np.asarray([r["head"] for r in rows])
    label = np.asarray([r["label_mean"] for r in rows])
    truth = np.asarray([r["truth_mean"] for r in rows])

    head_error = float(np.sqrt(np.mean((head - truth) ** 2)))
    label_error = float(np.sqrt(np.mean((label - truth) ** 2)))
    return {
        "rms_head_error": round(head_error, 5),
        "rms_label_error": round(label_error, 5),
        "teacher_ratio": round(label_error / max(head_error, 1e-12), 4),
        "signal_correlation": round(_correlate(label, truth), 4),
        "head_correlation": round(_correlate(head, truth), 4),
        "echo_correlation": round(_correlate(label, head), 4),
    }


def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson, returning 0 when either side is constant rather than nan."""
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--lambdas", default="",
                        help="comma-separated TD(lambda) mixtures to score off "
                             "the same rollouts, e.g. 0,0.5,0.9,0.95,0.99,1.0")
    parser.add_argument("--seed-games", type=int, default=24)
    parser.add_argument("--positions", type=int, default=64)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--action-cap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    import torch

    from hexset.netbot import load
    from hexset.policy import NetworkPolicy

    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]
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
        episodes = branch.drain()
        pairs = [labels(e, snapshot.seat, args.horizon) for e in episodes]
        mixtures = {
            lam: np.asarray(
                [lambda_label(e, snapshot.seat, lam) for e in episodes],
                dtype=np.float64,
            )
            for lam in lambdas
        }
        booted = np.asarray([p[0] for p in pairs], dtype=np.float64)
        finals = np.asarray([p[1] for p in pairs], dtype=np.float64)
        reached = float(np.mean([p[2] for p in pairs]))
        rows.append(
            {
                "progress": round(progress, 3),
                "rollouts": len(pairs),
                "reached_horizon": round(reached, 3),
                "terminal_variance": float(finals.var()),
                "bootstrap_variance": float(booted.var()),
                "gap": float(booted.mean() - finals.mean()),
                "head": float(snapshot.prediction),
                # Disjoint halves: the label and the terminal return of one
                # rollout share that rollout's dice, so averaging both over the
                # same games would correlate their errors and flatter the label.
                "label_mean": float(booted[0::2].mean()),
                "truth_mean": float(finals[1::2].mean()),
                # Same disjoint halves for every mixture: a label and a terminal
                # return off one rollout share that rollout's dice.
                "mixtures": {
                    str(lam): {
                        "label_mean": float(values[0::2].mean()),
                        "variance": float(values.var()),
                    }
                    for lam, values in mixtures.items()
                },
            }
        )
    elapsed = time.perf_counter() - started

    terminal = np.asarray([r["terminal_variance"] for r in rows])
    boot = np.asarray([r["bootstrap_variance"] for r in rows])
    gaps = np.asarray([r["gap"] for r in rows])
    reached = np.asarray([r["reached_horizon"] for r in rows])
    ratio = float(boot.mean() / max(terminal.mean(), 1e-12))

    payload = {
        "environment": environment(),
        "checkpoint": args.checkpoint,
        "iteration": loaded.iteration,
        "horizon": args.horizon,
        "positions": len(rows),
        "rollouts_each": args.rollouts,
        "seconds": round(elapsed, 1),
        "reached_horizon": round(float(reached.mean()), 4),
        "terminal_variance": round(float(terminal.mean()), 5),
        "bootstrap_variance": round(float(boot.mean()), 5),
        "variance_ratio": round(ratio, 4),
        "sigma_ratio": round(float(np.sqrt(ratio)), 4),
        "sample_factor": round(1.0 / max(ratio, 1e-12), 2),
        "mean_gap": round(float(gaps.mean()), 4),
        **teacher(rows),
        # Per-position, so a sharded run pools rows directly instead of
        # combining shards' aggregate RMS values, which is exact for `teacher`
        # but throws away the per-position mixture detail.
        "rows": rows,
    }
    if lambdas:
        payload["lambdas"] = [
            {
                "lam": lam,
                "variance": round(
                    float(np.mean([r["mixtures"][str(lam)]["variance"] for r in rows])), 5
                ),
                **teacher(
                    [
                        {
                            "head": r["head"],
                            "truth_mean": r["truth_mean"],
                            "label_mean": r["mixtures"][str(lam)]["label_mean"],
                        }
                        for r in rows
                    ]
                ),
            }
            for lam in lambdas
        ]

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(
        f"{payload['positions']} positions x {args.rollouts} rollouts, "
        f"horizon {args.horizon}, {payload['seconds']}s"
    )
    print(f"  reached the horizon   {payload['reached_horizon']:.1%}")
    print(f"  Var(terminal | s)     {payload['terminal_variance']:.5f}")
    print(f"  Var(V(s') | s)        {payload['bootstrap_variance']:.5f}")
    print(f"  variance ratio        {payload['variance_ratio']:.4f}")
    if lambdas:
        print("\n  TD(lambda) mixtures, off the same rollouts. `teacher` below 1")
        print("  means the label knows something the head does not; `echo` near 1")
        print("  means it is the head's own opinion handed back.")
        print(f"    {'lam':>5} {'sigma':>8} {'teacher':>9} {'echo':>7} {'signal':>8}")
        for block in payload["lambdas"]:
            print(f"    {block['lam']:>5.2f} {np.sqrt(block['variance']):>8.4f} "
                  f"{block['teacher_ratio']:>9.4f} {block['echo_correlation']:>7.3f} "
                  f"{block['signal_correlation']:>8.3f}")
    print(f"  sigma ratio           {payload['sigma_ratio']:.4f}")
    print(f"  data requirement      {payload['sample_factor']:.2f}x lower")
    print(f"  mean gap              {payload['mean_gap']:+.4f}")
    print(f"  rms error, head       {payload['rms_head_error']:.5f}")
    print(f"  rms error, label      {payload['rms_label_error']:.5f}")
    print(
        f"  teacher ratio         {payload['teacher_ratio']:.4f}"
        f"  ({'teaches' if payload['teacher_ratio'] < 1 else 'decoration'})"
    )
    print(f"  corr label/truth      {payload['signal_correlation']:+.4f}")
    print(f"  corr head/truth       {payload['head_correlation']:+.4f}")
    print(f"  corr label/head       {payload['echo_correlation']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
