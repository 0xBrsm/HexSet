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
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from catan.actions import space_for
from catan.board.board import random_base_board
from catan.encoding import static_graph
from catan.game import start
from catan.model import CatanNet, config_from_args, packing
from catan.policy import NetworkPolicy
from catan.ppo import PPOConfig, assemble, minibatch_terms
from catan.rewards import reward
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


def estimate(mean_small: float, mean_big: float, small: int, big: int) -> dict:
    """Separate signal from noise given `E|G_B|^2` measured at two batch sizes.

    Both `|G|^2` and `tr(S)` are unbiased here; their *ratio* is not, which is
    why the caller averages each side over many draws and calls this once
    rather than averaging the B_simple of each draw.
    """
    signal = (big * mean_big - small * mean_small) / (big - small)
    noise = (mean_small - mean_big) / (1.0 / small - 1.0 / big)
    return {
        "grad_sq_small": mean_small,
        "grad_sq_big": mean_big,
        "signal_g_squared": signal,
        "noise_trace_sigma": noise,
        # No number rather than a nonsense one. `signal <= 0` means the true
        # gradient is not resolved at either size; `noise < 0` means the larger
        # draw measured *more* variance than the smaller, which is sampling
        # error in the estimate itself.
        "b_simple": noise / signal if signal > 0 and noise >= 0 else None,
        # What fraction of a B_big gradient's length is the true gradient.
        "signal_share_at_b_big": (signal / mean_big) ** 0.5 if mean_big else None,
    }


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


def _measure(policy, batch, advantage, config, small: int, big: int, repeats: int, seed: int) -> dict:
    """The two-batch-size sweep for one advantage stream.

    Factored out of `main` so the paired probe can run it twice on the same
    positions and weights with the same shuffles — every draw then compares
    the two streams on identical rows, which is what makes the *difference*
    in B_simple a paired reading exempt from the estimator's own 2x
    cross-scale softness. `advantage` arrives already normalised over the
    whole batch, each stream by its own moments, matching what an update
    would actually feed the optimiser under either terminal.
    """
    positions = len(batch)
    small_sq: dict[str, list[float]] = {name: [] for name in TERMS}
    big_sq: dict[str, list[float]] = {name: [] for name in TERMS}
    generator = torch.Generator().manual_seed(seed)
    for repeat in range(repeats):
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
        print(f"  repeat {repeat + 1}/{repeats}: {done} pairs", flush=True)

    return {
        "pairs": len(small_sq["full"]),
        "terms": {
            # Average each side over every draw and estimate once: the paper's
            # estimator is a ratio of means, and a mean of ratios is not the same.
            name: estimate(
                sum(small_sq[name]) / len(small_sq[name]),
                sum(big_sq[name]) / len(big_sq[name]),
                small,
                big,
            )
            for name in TERMS
        },
    }


def _pair_correlations(episodes, players: int) -> dict:
    """The mechanism numbers behind the pair baseline, per seat then pooled.

    `rho = corr(r, r')` over the board pairs says how much of the terminal
    reward the shared (board, seat) geometry carries at all. `rho_v` runs the
    same correlation on `r - V(s_T)` — the head's own-payoff estimate at the
    seat's *last decision*, `value[0]` of the final transition, which is
    exactly the estimate `advantages()` bootstraps against at its final step
    (reused from the recorded trajectory rather than re-running the net). It
    is the share the head has NOT already priced, and the register's
    prediction is that this number, not rho, decides the gate: pairing with
    independent dice streams cannot cancel dice, only unpriced geometry.
    """
    by_index = {episode.index: episode for episode in episodes}
    r = [[] for _ in range(players)]
    r_mate = [[] for _ in range(players)]
    residual = [[] for _ in range(players)]
    residual_mate = [[] for _ in range(players)]
    for index in sorted(by_index):
        if index % 2:
            continue
        even, odd = by_index[index], by_index[index ^ 1]
        even_pay, odd_pay = reward(even.outcome), reward(odd.outcome)
        for seat in range(players):
            r[seat].append(even_pay[seat])
            r_mate[seat].append(odd_pay[seat])
            residual[seat].append(
                even_pay[seat] - even.trajectories[seat][-1].value[0]
            )
            residual_mate[seat].append(
                odd_pay[seat] - odd.trajectories[seat][-1].value[0]
            )

    def corr(a, b) -> float:
        return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])

    def rows(a, b) -> dict:
        return {
            "per_seat": [corr(a[seat], b[seat]) for seat in range(players)],
            "pooled": corr(np.concatenate(a), np.concatenate(b)),
        }

    return {"rho": rows(r, r_mate), "rho_v": rows(residual, residual_mate)}


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
    parser.add_argument(
        "--paired",
        action="store_true",
        default=False,
        help="Gate A of the variance screen's candidate 1: collect one "
        "board-paired cohort (--games must be even), run the estimator twice "
        "on the same batch — advantage stream from raw vs pair-adjusted "
        "terminals, same positions, same weights, same shuffles — and report "
        "the pair correlations rho and rho_v alongside",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    if args.paired and args.games % 2:
        raise SystemExit("--paired deals boards in pairs; --games must be even")

    torch.set_num_threads(args.threads)
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    stored = state.get("args", {})
    # Shape as well as size: see `catan.model.config_from_args`. This estimator
    # is the mechanism behind the minibatch block, so it has to be runnable on
    # whichever lineage the block calibrates against.
    model = config_from_args(stored)

    rng = random.Random(args.seed)
    board = random_base_board(rng)
    space = space_for(start(board, args.players, rng))
    graph = static_graph(board.topology)

    torch.manual_seed(args.seed)
    net = CatanNet(space, graph, args.players, model)
    net.load_state_dict(state["net"])
    net = net.to(args.device)
    net.detach_value = bool(stored.get("detach_value", False))
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
        pair_boards=args.paired,
    )
    episodes = collector.drain()
    batch = assemble(episodes, policy.layout, config).to(args.device)
    positions = len(batch)
    print(f"batch: {positions} positions", flush=True)

    # The comparison is the batch: same positions, same weights, same shuffle
    # seed, the streams differing only in the terminal fed to the advantage
    # recursion. `assemble` with `pair_baseline` is the exact production wire,
    # so the probe measures the gradient a paired run would take rather than a
    # re-derivation of it.
    streams = {"raw": batch.advantage}
    if args.paired:
        streams["pair_adjusted"] = (
            assemble(episodes, policy.layout, replace(config, pair_baseline=True))
            .advantage.to(args.device)
        )

    small, big = args.micro, args.micro * args.group
    if positions < big:
        raise SystemExit(
            f"{positions} positions is short of one B_big draw ({big}); "
            "collect more games or lower --group"
        )

    measured_streams = {}
    for stream_name, advantage in streams.items():
        if args.paired:
            print(f"stream: {stream_name}", flush=True)
        # Once, over the whole batch — see the module docstring. Each stream by
        # its own moments, which is what an update under that terminal would do.
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        measured_streams[stream_name] = _measure(
            policy, batch, advantage, config, small, big, args.repeats, args.seed + 1
        )
    measured = measured_streams["raw"]["terms"]

    result = {
        "environment": environment(),
        "checkpoint": args.checkpoint,
        "positions": positions,
        "games": args.games,
        "b_small": small,
        "b_big": big,
        "pairs": measured_streams["raw"]["pairs"],
        "terms": measured,
    }
    if args.paired:
        result["paired"] = {
            "terms": measured_streams["pair_adjusted"]["terms"],
            **_pair_correlations(episodes, args.players),
        }
    print(json.dumps(result, indent=2))

    for stream_name, sweep in measured_streams.items():
        if args.paired:
            print(f"\nstream: {stream_name}")
        print(f"\n{'term':<8} {'B_simple':>12} {'signal share at ' + str(big):>22}")
        for name in TERMS:
            b = sweep["terms"][name]["b_simple"]
            share = sweep["terms"][name]["signal_share_at_b_big"]
            shown = f"{b:,.0f}" if b else "above the range probed"
            print(f"{name:<8} {shown:>12} {share:>21.1%}" if b else f"{name:<8} {shown:>12}")
    if args.paired:
        raw_b = measured["policy"]["b_simple"]
        adjusted_b = result["paired"]["terms"]["policy"]["b_simple"]
        if raw_b and adjusted_b is not None:
            print(
                f"\npolicy-term B_simple: raw {raw_b:,.0f} -> pair-adjusted "
                f"{adjusted_b:,.0f} (fall {1 - adjusted_b / raw_b:+.1%}; the "
                "gate wants >= +15% at both probe scales)"
            )
        print(
            f"rho pooled {result['paired']['rho']['pooled']:+.3f}, "
            f"rho_v pooled {result['paired']['rho_v']['pooled']:+.3f} "
            "(rho_v near zero means the residual is unrolled dice, which "
            "pairing cannot cancel)"
        )
    print(
        "\nB_simple is where sampling noise stops dominating. Read it against "
        "--minibatch, which is the batch each optimiser step actually sees."
    )
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
