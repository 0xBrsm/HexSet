"""How large a batch this problem's gradient actually needs.

The campaign collects ~118k positions an iteration and steps on minibatches of
4096, and neither number was ever chosen by measurement. The gradient noise
scale answers whether they are buying anything: it is the batch size at which
the sampling noise in a gradient estimate stops dominating its signal. Below
it, doubling the batch roughly halves the steps you need; far above it, extra
samples buy almost nothing and the same wall clock is better spent on more
updates or more games.

Following McCandlish, Kaplan, Amodei & Brown, *An Empirical Model of Large-Batch
Training* (2018). Their estimator wants the true gradient's squared norm |G|^2
and the trace of the gradient covariance tr(S), and gets both from the same
quantity measured at two batch sizes, since the expected squared norm of a
`B`-sample estimate is

    E|G_B|^2 = |G|^2 + tr(S) / B

Two sizes therefore separate the two unknowns:

    |G|^2  ~  (B_big |G_big|^2 - B_small |G_small|^2) / (B_big - B_small)
    tr(S)  ~  (|G_small|^2 - |G_big|^2) / (1/B_small - 1/B_big)
    B_simple = tr(S) / |G|^2

Both are unbiased; their *ratio* is not, so the two are averaged over many
draws separately and divided only at the end — which is what the paper
prescribes and what makes a single-batch reading trustworthy.

The measurement is cheap: one collected batch, no training run. It reads the
gradient at the *start* of an update, where PPO's ratio is exactly 1 and the
policy term is the plain policy gradient, so what comes back is the gradient
the run is actually taking rather than one part-way through an epoch.

Two choices worth knowing. Advantages are normalised **once over the whole
batch** rather than per microbatch: normalising each draw by its own mean and
standard deviation would make small draws look artificially well-scaled and
manufacture exactly the batch-size dependence being measured. And the loss is
the full PPO objective, value and entropy terms included, because that is what
the optimiser sees.

    python -m benchmarks.noise_scale --checkpoint /w/runs/ppo4/latest.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from catan.actions import space_for
from catan.board.board import random_base_board
from catan.encoding import static_graph
from catan.game import start
from catan.model import CatanNet, ModelConfig, packing
from catan.policy import NetworkPolicy
from catan.ppo import PPOConfig, assemble, minibatch_terms
from catan.selfplay import Collector

from .throughput import environment


def _flat_gradient(net: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            (torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1)
            for p in net.parameters()
        ]
    )


TERMS = ("full", "policy", "value")


def _gradients(policy, batch, rows, advantage, config) -> dict[str, torch.Tensor]:
    """One forward, three partial backwards, each a flat gradient over `rows`.

    The decomposition is the point. The two halves of the objective are not
    alike: the policy term differentiates a clipped ratio against a normalised
    advantage, while the value term regresses on a Monte Carlo return whose
    own irreducible variance is 68-80% on this problem's measurement. If the
    batch a step sees is noise-dominated, which half is doing it decides
    whether the answer is a bigger minibatch, a smaller value coefficient or a
    different value target. `policy` carries the entropy bonus, since that is
    part of the same objective and lands on the same heads.
    """
    terms = minibatch_terms(
        policy,
        batch.buffer[rows],
        batch.mask[rows],
        batch.pair[rows],
        batch.chosen[rows],
        batch.offer[rows],
        batch.log_prob[rows],
        advantage[rows],
        batch.value_target[rows],
        config,
    )
    pieces = {
        "full": terms.loss,
        "policy": terms.policy_term + terms.entropy_term,
        "value": terms.value_term,
    }
    out = {}
    for name in TERMS:
        policy.net.zero_grad(set_to_none=True)
        pieces[name].backward(retain_graph=True)
        out[name] = _flat_gradient(policy.net)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/w/runs/ppo4/latest.pt")
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--max-offers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--entropy", type=float, default=0.02)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument(
        "--micro",
        type=int,
        default=256,
        help="B_small: the small gradient estimate. Every group of --group of "
        "these is averaged into one B_big estimate",
    )
    parser.add_argument("--group", type=int, default=16)
    parser.add_argument(
        "--repeats",
        type=int,
        default=8,
        help="reshuffles of the batch; each contributes one pair of estimates "
        "per group, and the two sides are averaged before dividing",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    stored = state.get("args", {})
    width = int(stored.get("width", 64))
    rounds = int(stored.get("rounds", 2))

    rng = random.Random(args.seed)
    board = random_base_board(rng)
    space = space_for(start(board, args.players, rng))
    graph = static_graph(board.topology)

    torch.manual_seed(args.seed)
    net = CatanNet(space, graph, args.players, ModelConfig(width=width, rounds=rounds))
    net.load_state_dict(state["net"])
    net = net.to(args.device)
    policy = NetworkPolicy(net, space, packing(graph, args.players), device=args.device)
    config = PPOConfig(
        entropy_coefficient=args.entropy, value_coefficient=args.value_coefficient
    )

    print(f"collecting {args.games} games...", flush=True)
    collector = Collector(
        policy,
        lanes=min(args.lanes, args.games),
        players=args.players,
        seed=args.seed,
        action_cap=4000,
        max_offers=args.max_offers,
        deal=args.games,
    )
    batch = assemble(collector.drain(), policy.layout, config).to(args.device)
    positions = len(batch)
    print(f"batch: {positions} positions", flush=True)

    # Once, over the whole batch — see the module docstring.
    advantage = batch.advantage
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    small, big = args.micro, args.micro * args.group
    if positions < big:
        raise SystemExit(
            f"{positions} positions is short of one B_big draw ({big}); "
            "collect more games or lower --group"
        )

    small_sq: dict[str, list[float]] = {name: [] for name in TERMS}
    big_sq: dict[str, list[float]] = {name: [] for name in TERMS}
    generator = torch.Generator().manual_seed(args.seed + 1)
    for repeat in range(args.repeats):
        order = torch.randperm(positions, generator=generator)
        for start_row in range(0, positions - big + 1, big):
            chunk = order[start_row : start_row + big]
            draws = [
                _gradients(policy, batch, chunk[i : i + small], advantage, config)
                for i in range(0, big, small)
            ]
            for name in TERMS:
                stacked = torch.stack([draw[name] for draw in draws])
                small_sq[name].append(float(stacked.pow(2).sum(dim=1).mean()))
                big_sq[name].append(float(stacked.mean(dim=0).pow(2).sum()))
        done = len(small_sq["full"])
        print(f"  repeat {repeat + 1}/{args.repeats}: {done} pairs", flush=True)

    measured = {}
    for name in TERMS:
        mean_small = sum(small_sq[name]) / len(small_sq[name])
        mean_big = sum(big_sq[name]) / len(big_sq[name])
        # Average the two sides over every draw and divide once: the paper's
        # estimator is a ratio of means, and a mean of ratios is not the same.
        signal = (big * mean_big - small * mean_small) / (big - small)
        noise = (mean_small - mean_big) / (1.0 / small - 1.0 / big)
        measured[name] = {
            "grad_sq_small": mean_small,
            "grad_sq_big": mean_big,
            "signal_g_squared": signal,
            "noise_trace_sigma": noise,
            "b_simple": noise / signal if signal > 0 else None,
            # What fraction of a B_big gradient's length is the true gradient.
            "signal_share_at_b_big": (signal / mean_big) ** 0.5 if mean_big else None,
        }

    result = {
        "environment": environment(),
        "checkpoint": args.checkpoint,
        "positions": positions,
        "games": args.games,
        "b_small": small,
        "b_big": big,
        "pairs": len(small_sq["full"]),
        "terms": measured,
    }
    print(json.dumps(result, indent=2))

    print(f"\n{'term':<8} {'B_simple':>12} {'signal share at ' + str(big):>22}")
    for name in TERMS:
        b = measured[name]["b_simple"]
        share = measured[name]["signal_share_at_b_big"]
        shown = f"{b:,.0f}" if b else "above the range probed"
        print(f"{name:<8} {shown:>12} {share:>21.1%}" if b else f"{name:<8} {shown:>12}")
    print(
        "\nB_simple is where sampling noise stops dominating. Read it against "
        "--minibatch, which is the batch each optimiser step actually sees."
    )
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
