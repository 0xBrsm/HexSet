# SPDX-License-Identifier: GPL-3.0-only
"""The table league: N learners share every game, each training on its own seats.

Stages 2-3 of the table-league design, shaped for hyperparameter
heats: the same architecture on every seat, warm-started from one checkpoint,
with per-learner `PPOConfig` overrides. Four seats a game means a four-arm
ablation costs ~1.1x one arm — collection is shared, and four quarter-sized
updates cost one full one. The arithmetic that licenses the seat split is the
noise-scale finding: the independent unit is the game, and a learner seated
once in every game keeps every game.

Standings come from the games themselves — every game scores every learner —
so the heat's gate reads off the log. The heat measures who learns best in
this shared ecology, not isolated self-play; external anchors calibrate.

    python -m hexset.league --base runs/scratch-mlp/iter-00450.pt \\
        --learner entropy=0.02 --learner entropy=0.03 \\
        --learner entropy=0.05 --learner entropy=0.08 \\
        --iterations 60 --checkpoint-dir runs/heat-entropy
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import torch

from .actions import build_space
from .board.board import random_base_board
from .collect import ParallelCollector, WorkerSpec
from .encoding import static_graph
from .model import VALUE_HEADS, HexNet, ModelConfig, packing, quantile_warm_start
from .policy import NetworkPolicy
from .ppo import PPOConfig, assemble, update
from .rewards import reward
from .selfplay import owned
from .train import preserve_blowout, prune_recent, save

# What a --learner override may set, mapped onto PPOConfig fields. Collection
# knobs are deliberately absent: they are table properties every seat shares,
# and a heat that varies one is a sequential arm, not a league.
OVERRIDES = {
    "lr": ("learning_rate", float),
    "entropy": ("entropy_coefficient", float),
    "clip": ("clip", float),
    "c_v": ("value_coefficient", float),
    "value_lam": ("value_lam", float),
    "epochs": ("epochs", int),
    "minibatch": ("minibatch", int),
    "kl_break": ("kl_break", float),
    "eps": ("adam_eps", float),
}

# The controller's original 10% nudge, kept as the default so every heat
# before Heat 5 reproduces unchanged when re-parsed.
DEFAULT_CONTROLLER_GAIN = 0.10


def parse_learner(spec: str, base: PPOConfig) -> tuple[PPOConfig, float | None, float]:
    """One seat's config: the base with `k=v` overrides applied.

    `target_entropy=<v>` and `gain=<v>` are the two keys that are not
    `PPOConfig` fields: they arm and tune the seat's entropy controller,
    returned separately. `gain` is a no-op without `target_entropy` alongside
    it.
    """
    fields = asdict(base)
    target: float | None = None
    gain = DEFAULT_CONTROLLER_GAIN
    if spec.strip():
        for part in spec.split(","):
            key, _, value = part.partition("=")
            key = key.strip()
            if key == "target_entropy":
                target = float(value)
                continue
            if key == "gain":
                gain = float(value)
                continue
            if key not in OVERRIDES:
                raise SystemExit(
                    f"unknown learner override {key!r}; one of "
                    f"{sorted(OVERRIDES) + ['target_entropy', 'gain']}"
                )
            name, cast = OVERRIDES[key]
            fields[name] = cast(value)
    return PPOConfig(**fields), target, gain


def nudged(coefficient: float, entropy: float, target: float, gain: float = DEFAULT_CONTROLLER_GAIN) -> float:
    """One controller step: hold the policy's entropy at the target.

    SAC's automatic temperature, discretised to the iteration: below target
    the coefficient rises by `gain`, above it falls by the same factor,
    clamped to [0.005, 0.10]. One-sided pressure through the entropy term
    only — the ppo7 rule is about a controller compounding through the step
    size, which this cannot do; the lr stays fixed and the clamp bounds the
    pressure. Heat 2 ran this at the default 10% and rang (coefficient
    oscillated 0.008-0.039 against the lagging plant); Heat 5 is the retest
    at a damped gain.
    """
    factor = (1 + gain) if entropy < target else 1 / (1 + gain)
    return min(0.10, max(0.005, coefficient * factor))


def standings(episodes, learners: int) -> tuple[list[int], list[float]]:
    """Wins and mean per-game VP by learner id, from the shared games."""
    wins = [0] * learners
    points = [0.0] * learners
    counted = 0
    for episode in episodes:
        cast = episode.cast or (0,) * episode.players
        if episode.outcome.winner is not None:
            wins[cast[episode.outcome.winner]] += 1
        payoffs = reward(episode.outcome)
        per = [[] for _ in range(learners)]
        for seat, owner in enumerate(cast):
            per[owner].append(payoffs[seat])
        for k in range(learners):
            if per[k]:
                points[k] += sum(per[k]) / len(per[k])
        counted += 1
    return wins, [p / counted if counted else 0.0 for p in points]


def build_parser() -> argparse.ArgumentParser:
    """The league's knobs, introspectable by `hexset.run` -- see `train.build_parser`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="", help="checkpoint every learner warm-starts from")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a heat from its own learner*/latest.pt checkpoints — "
        "the box restart that killed the entropy heat at 35/60 is why this exists",
    )
    parser.add_argument(
        "--learner",
        action="append",
        required=True,
        help="one seat's PPOConfig overrides, e.g. 'entropy=0.03,lr=3e-4'; "
        "repeat once per learner (2..players), '' for the unmodified base",
    )
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--games-per-iteration", type=int, default=128)
    parser.add_argument("--lanes", type=int, default=128)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--learner-order", default=None,
        help="comma-separated permutation of learner ids, applied by the caster "
             "before its rotation. Changes who sits next to whom while leaving "
             "every learner's share of every board seat balanced; rotation alone "
             "cannot, because the cyclic order round the table is always "
             "0,1,2,3. Use to test whether the two-tight-pairs structure both "
             "noise heats produced is turn-order adjacency: '0,2,1,3' reseats "
             "the cycle, so adjacency predicts the pairs become {0,2}/{1,3}.",
    )
    parser.add_argument(
        "--pair-boards",
        action="store_true",
        default=False,
        help="the variance screen's candidate-1 package, one flag because the "
        "registration treats it as one treatment: deal every board twice "
        "(games 2k and 2k+1 share 2k's board, each with its own dice), cast "
        "both halves of a pair identically, and set pair_baseline on every "
        "learner's PPOConfig so each seat's policy gradient is paid against "
        "its mate game's outcome. Default off: recorded runs replay "
        "bit-identically, and a frozen manifest that predates this flag "
        "refuses to load rather than silently meaning something new",
    )
    parser.add_argument(
        "--value-head",
        default="",
        choices=("",) + VALUE_HEADS,
        help="build every seat's value head with this shape instead of the "
        "base checkpoint's. Empty (the default) keeps the base's own shape, so "
        "every heat on record replays unchanged. 'quantile' is the variance "
        "screen's candidate 3: the head's mean is still V and the policy wire "
        "is untouched, but the value loss becomes the per-seat quantile Huber "
        "loss, which is the trunk-shaping mechanism Gate B tests. A scalar "
        "base is warm-started into it, so V at iteration 0 is the base's own",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=32,
        help="levels per seat for --value-head quantile; inert otherwise",
    )
    parser.add_argument("--action-cap", type=int, default=4000)
    parser.add_argument("--max-offers", type=int, default=3)
    parser.add_argument("--collect-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--keep-recent", type=int, default=5)
    parser.add_argument("--keep-every", type=int, default=25)
    parser.add_argument("--dump-blowout-batch", action="store_true", default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the directory named on the command line, and take nothing else.

    A run is `runs/<...>/` holding `run.json` and a frozen `config/`, created by
    `python -m hexset.run.init`. Flags are not accepted here: the manifest is the
    only input, so a recorded result is reproducible from the repository rather
    than from whichever script in gitignored `tmp/` happened to survive. See
    `hexset.run.manifest` for the three losses that motivated this.
    """
    from .run import load
    from .run.manifest import MODULES

    tokens = list(sys.argv[1:] if argv is None else argv)
    if len(tokens) != 1 or tokens[0].startswith("-"):
        raise SystemExit(
            "usage: python -m hexset.league <run-directory>\n"
            "  create one with: python -m hexset.run.init --mode league --name NAME -- <flags>\n"
            "  the flags this used to accept are frozen into the run's config/ instead."
        )
    manifest = load(tokens[0])
    if manifest.mode != "league":
        raise SystemExit(
            f"{tokens[0]} is a {manifest.mode} run; launch it with "
            f"python -m {MODULES[manifest.mode]}"
        )
    args = manifest.namespace()

    if not 2 <= len(args.learner) <= args.players:
        raise SystemExit(f"a league seats 2..{args.players} learners, got {len(args.learner)}")

    directory = Path(args.checkpoint_dir)
    if args.resume:
        seat_states = []
        for k in range(len(args.learner)):
            path = directory / f"learner{k}" / "latest.pt"
            if not path.exists():
                raise SystemExit(f"--resume needs {path}")
            seat_states.append(torch.load(path, map_location=args.device, weights_only=False))
        state = seat_states[0]
    elif args.base:
        seat_states = None
        state = torch.load(args.base, map_location=args.device, weights_only=False)
    else:
        raise SystemExit("pass --base for a fresh heat or --resume to continue one")
    stored = state.get("args", {})
    base_config = PPOConfig(
        **{
            key: value
            for key, value in state.get("config", {}).items()
            if key in asdict(PPOConfig())
        }
    )
    parsed = [parse_learner(spec, base_config) for spec in args.learner]
    configs = [config for config, _, _ in parsed]
    targets = [target for _, target, _ in parsed]
    gains = [gain for _, _, gain in parsed]
    learners = len(configs)

    if args.pair_boards:
        # One flag, three wires: paired dealing and casting ride the
        # WorkerSpec below; the baseline lands on every learner because the
        # treatment is a package — a seat trained on raw terminals inside a
        # paired cohort would be a different experiment than the one
        # registered.
        if args.games_per_iteration % 2:
            raise SystemExit(
                "--pair-boards deals boards in pairs; "
                "--games-per-iteration must be even"
            )
        if args.games_per_iteration % args.collect_workers:
            raise SystemExit(
                "--pair-boards needs every iteration to be complete pairs, "
                "which holds when worker quotas are equal: make "
                "--games-per-iteration divisible by --collect-workers "
                "(unequal quotas stagger the workers' index ranges, and a "
                "pair split across iterations cannot find its mate at "
                "assemble time)"
            )
        configs = [replace(config, pair_baseline=True) for config in configs]

    board = random_base_board(random.Random(args.seed))
    topology = board.topology
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, args.players
    )
    graph = static_graph(topology)
    base_head = str(stored.get("value_head", "linear"))
    model_config = ModelConfig(
        width=int(stored.get("width", 64)),
        rounds=int(stored.get("rounds", 2)),
        value_head=args.value_head or base_head,
        policy_head=str(stored.get("policy_head", "linear")),
        quantiles=int(args.quantiles),
    )
    # The shape the seats actually carry, not the base's, or the checkpoints
    # this heat writes would record a head they were not trained with and
    # `hexset.netbot.load` would rebuild the wrong module. `stored` is what
    # every payload below copies forward.
    stored = {
        **stored,
        "value_head": model_config.value_head,
        "quantiles": model_config.quantiles,
    }
    policies: list[NetworkPolicy] = []
    optimisers: list[torch.optim.Adam] = []
    for k, config in enumerate(configs):
        net = HexNet(space, graph, args.players, model_config).to(args.device)
        weights = (seat_states[k] if seat_states else state)["net"]
        if model_config.value_head == "quantile" and base_head != "quantile":
            # Every level starts at the scalar head's own prediction, so the
            # treatment arm opens on the same policy and the same V as its
            # control (to one float32 rounding). See `quantile_warm_start`.
            weights = quantile_warm_start(
                weights, args.players, model_config.quantiles
            )
        net.load_state_dict(weights)
        policies.append(
            NetworkPolicy(net, space, packing(graph, args.players), device=args.device)
        )
        optimiser = torch.optim.Adam(net.parameters(), lr=config.learning_rate, eps=config.adam_eps)
        if seat_states and "optimiser" in seat_states[k]:
            optimiser.load_state_dict(seat_states[k]["optimiser"])
        optimisers.append(optimiser)

    base_index = int(state.get("games_started", 0)) if args.resume else 0
    if args.pair_boards and base_index % 2:
        # `games_started` is the max over the workers' counters and lands odd
        # about half the time; dealing from an odd base would give the first
        # game a mate that finished last session. One skipped index is an
        # unused seed, not a lost game — the same rule a resume already
        # applies to the indices between the slowest and fastest worker.
        base_index += 1

    shard = max(1, -(-args.lanes // args.collect_workers))
    collector = ParallelCollector(
        [
            WorkerSpec(
                seed=args.seed,
                players=args.players,
                lanes=shard,
                action_cap=args.action_cap,
                max_offers=args.max_offers,
                stride=args.collect_workers,
                width=model_config.width,
                rounds=model_config.rounds,
                torch_seed=args.seed + 100_000 + worker,
                value_head=model_config.value_head,
                policy_head=model_config.policy_head,
                quantiles=model_config.quantiles,
                learners=learners,
                learner_order=(
                    tuple(int(x) for x in args.learner_order.split(","))
                    if args.learner_order else None
                ),
                first_game=base_index + worker,
                pair_boards=args.pair_boards,
            )
            for worker in range(args.collect_workers)
        ]
    )

    log = directory / "log.jsonl"
    directory.mkdir(parents=True, exist_ok=True)
    total_wins = [0] * learners
    start_iteration = 0
    if args.resume:
        start_iteration = int(state.get("iteration", 0))
        if log.exists():
            for line in log.open():
                total_wins = json.loads(line).get("standings", total_wins)
    began = time.perf_counter()

    for iteration in range(start_iteration, args.iterations):
        collector.sync_many([p.net for p in policies])
        started = time.perf_counter()
        episodes = collector.collect(args.games_per_iteration)
        collected = time.perf_counter() - started

        wins, vp = standings(episodes, learners)
        total_wins = [a + b for a, b in zip(total_wins, wins)]

        per_learner = []
        started = time.perf_counter()
        for k, (policy, optimiser, config) in enumerate(
            zip(policies, optimisers, configs)
        ):
            batch = assemble(owned(episodes, k), policy.layout, config)
            net_before = {
                key: value.detach().cpu().clone()
                for key, value in policy.net.state_dict().items()
            }
            stats = update(policy, optimiser, batch, config)
            if config.kl_break and stats.epochs_taken < config.epochs:
                preserve_blowout(
                    directory / f"learner{k}",
                    iteration + 1,
                    net_before,
                    optimiser.state_dict(),
                    batch,
                    args.dump_blowout_batch,
                )
            if targets[k] is not None:
                configs[k] = replace(
                    configs[k],
                    entropy_coefficient=nudged(
                        configs[k].entropy_coefficient, stats.entropy, targets[k], gains[k]
                    ),
                )
            per_learner.append(
                {
                    "learner": k,
                    "overrides": args.learner[k],
                    "entropy_coef": configs[k].entropy_coefficient,
                    "positions": stats.positions,
                    "entropy": stats.entropy,
                    "approx_kl_last_epoch": stats.approx_kl_last_epoch,
                    "epochs_taken": stats.epochs_taken,
                    "explained_variance": stats.explained_variance,
                    # Both, because a heat may run two value-head shapes:
                    # `value_loss` is the term that was differentiated and
                    # `value_mse` is the mean's squared error, the only column
                    # comparable across arms. Equal under every scalar head.
                    "value_loss": stats.value_loss,
                    "value_mse": stats.value_mse,
                    "wins": wins[k],
                    "vp": round(vp[k], 4),
                }
            )
        updated = time.perf_counter() - started

        record = {
            "iteration": iteration,
            "games": len(episodes),
            "collect_seconds": collected,
            "update_seconds": updated,
            "standings": total_wins,
            "learners": per_learner,
            "elapsed": time.perf_counter() - began,
        }
        line = json.dumps(record)
        print(line, flush=True)
        with log.open("a") as handle:
            handle.write(line + "\n")

        if (iteration + 1) % args.checkpoint_every == 0 or iteration + 1 == args.iterations:
            for k, (policy, optimiser, config) in enumerate(
                zip(policies, optimisers, configs)
            ):
                seat_dir = directory / f"learner{k}"
                payload = {
                    "iteration": iteration + 1,
                    "games_started": collector.games_started(),
                    "net": policy.net.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "args": {**stored, **{"league_overrides": args.learner[k]}},
                    "config": asdict(config),
                }
                save(seat_dir / "latest.pt", payload)
                if args.keep_recent:
                    save(seat_dir / f"recent-{iteration + 1:05d}.pt", payload)
                    prune_recent(seat_dir, args.keep_recent)
                if args.keep_every and (iteration + 1) % args.keep_every == 0:
                    save(seat_dir / f"iter-{iteration + 1:05d}.pt", payload)

    collector.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
