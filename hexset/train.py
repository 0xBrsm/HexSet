# SPDX-License-Identifier: GPL-3.0-only
"""The PPO training loop, runnable and resumable.

    python -m hexset.train --device cuda --lanes 256 --iterations 400

Everything this joins already exists: `hexset.selfplay` produces the episodes,
`hexset.policy` is the one forward per tick, `hexset.rewards` scalarises the
outcome and `hexset.ppo` does the arithmetic. This file is the run — the loop,
the log and the checkpoints.

## Checkpointing is for crashes, so it is written to survive one

An overnight run that cannot resume is an overnight run that has to be watched.
Two properties make that work and both are easy to leave out:

*The write is atomic.* A checkpoint is written to a temporary file and renamed
over the live one. A crash during `torch.save` otherwise leaves a truncated file
where the good one used to be, which is the failure mode where you lose the run
*and* the checkpoint at the same moment.

*The game counter is saved.* A game is a pure function of the seed and its
index, so a resumed run that restarts the counter replays exactly the games it
has already learned from. It would look like it was working.

What is deliberately not restored is the GPU's RNG state and the games that were
in flight at the crash. Neither is worth the complexity: the first only affects
which actions get sampled next, and the second costs at most `lanes` partial
games once.

## Throughput

Measured on this box: engine step 18.8 µs, `encode` 25.2 µs, `legal_actions`
1.8 µs — about 46 µs of Python per position against ~23 µs of GPU forward at
batch 512. The rollout is plumbing-bound by roughly 2:1, so the loop reports
collect and update time separately: if collect dominates, and it will, the
answer is more collector processes and not a faster forward. Nothing in this
file tries to make the GPU side quicker.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from .actions import ActionType, space_for
from .arena import wilson
from .board.board import random_base_board
from .collect import (
    ParallelCollector,
    WorkerSpec,
    alternating,
    check_mix,
    frozen,
    greedy_opponent,
    mix_caster,
    mix_names,
    mix_opponents,
    mixed_caster,
    named_opponent,
    parse_mix,
    table_pool,
)
from .encoding import static_graph
from .model import POLICY_HEADS, VALUE_HEADS, HexNet, ModelConfig, packing
from .policy import NetworkPolicy
from .ppo import ADAM_EPS, PPOConfig, assemble, update
from .rewards import reward
from .schedule import AdaptiveLR, current_lr, linear_anneal, set_lr
from .selfplay import BatchPolicy, Choice, Collector, Episode, RandomPolicy, Request
from .game import start



@dataclass
class Progress:
    iteration: int
    games: int
    positions: int
    collect_seconds: float
    assemble_seconds: float
    update_seconds: float
    positions_per_second: float
    truncated: float
    mean_actions: float
    mean_turns: float
    # The convention-collapse canary. The trained policy accepts 0.1-0.6 trades
    # a game where greedy accepts 21.6-27.1; if opponent mixing is doing its
    # job, this is the number that has to move, and it has to move within tens
    # of iterations or the accept branch is not being sampled at all.
    accepts_per_seat_game: float = 0.0
    proposes_per_seat_game: float = 0.0


class MixedPolicy:
    """One policy per seat, still exactly one forward per tick.

    Evaluation needs the network on some seats and a reference on the others,
    and the obvious implementation — call each policy per request — would pay
    the dispatch toll once per position instead of once per tick. This splits
    the batch by seat, hands each policy its whole share in one call, and
    reassembles in request order.
    """

    def __init__(self, network, other, network_seats: Sequence[int]) -> None:
        self.network = network
        self.other = other
        self.network_seats = set(network_seats)

    def act(self, requests: Sequence[Request]) -> list[Choice]:
        mine = [i for i, r in enumerate(requests) if r.seat in self.network_seats]
        theirs = [i for i, r in enumerate(requests) if r.seat not in self.network_seats]
        out: list[Choice | None] = [None] * len(requests)
        if mine:
            for slot, choice in zip(mine, self.network.act([requests[i] for i in mine])):
                out[slot] = choice
        if theirs:
            for slot, choice in zip(
                theirs, self.other.act([requests[i] for i in theirs])
            ):
                out[slot] = choice
        return out


def duel(
    policy: NetworkPolicy,
    *,
    games: int,
    lanes: int,
    players: int,
    seed: int,
    network_seats: Sequence[int],
    max_offers: int | None,
) -> dict:
    """Play the network against uniform-random opponents and report an interval.

    A win rate without an interval is not a result — at these game counts the
    interval is wide enough to swallow most of what a short run could achieve —
    so `hexset.arena.wilson` is applied here rather than left to the reader.

    The games counted are fixed in advance: indices `0..games-1`, each played to
    completion. Taking instead the first `games` episodes a collector happens to
    finish would select for short games, and game length is not independent of
    who is winning — a decisive network would finish faster and the estimate
    would flatter it. Waiting for the slowest of a fixed cohort costs some wall
    clock and removes that entirely.

    The cohort is a `deal` bound on the collector rather than a filter on its
    output. Filtering was the first version and it played every replacement game
    in full before discarding it, which is how a 400-game eval came to run for
    over ten minutes.

    Seats are fixed rather than rotated, which is a real limitation and is
    reported as one: the measured seat effect on this engine is mild and not
    significant, but it is not zero.

    `max_offers` is the training budget, passed in rather than defaulted, because
    a duel under a different budget measures a policy on a horizon it never saw.
    """
    reference = RandomPolicy(random.Random(seed))
    mixed = MixedPolicy(policy, reference, network_seats)
    collector = Collector(
        mixed,
        lanes=min(lanes, games),
        players=players,
        seed=seed,
        action_cap=4000,
        max_offers=max_offers,
        deal=games,
    )
    episodes: list[Episode] = collector.drain()

    wins = sum(1 for e in episodes if e.outcome.winner in network_seats)
    decided = sum(1 for e in episodes if e.outcome.winner is not None)
    points = [
        sum(reward(e.outcome)[s] for s in network_seats) / len(network_seats)
        for e in episodes
    ]
    low, high = wilson(wins, len(episodes)) if episodes else (0.0, 0.0)
    return {
        "games": len(episodes),
        "decided": decided,
        "wins": wins,
        "win_rate": wins / len(episodes) if episodes else 0.0,
        "wilson_low": low,
        "wilson_high": high,
        "expected_share": len(network_seats) / players,
        "mean_relative_points": sum(points) / len(points) if points else 0.0,
    }


def _paired(episode) -> tuple[float, bool]:
    """(learner minus reference mean VP, learner won) for one episode."""
    learner = [s for s, pid in enumerate(episode.cast) if pid == 0]
    others = [s for s, pid in enumerate(episode.cast) if pid != 0]
    points = episode.outcome.points
    margin = sum(points[s] for s in learner) / len(learner) - sum(
        points[s] for s in others
    ) / len(others)
    return margin, episode.outcome.winner in learner


def _antithetic(
    policy: BatchPolicy,
    reference: BatchPolicy,
    *,
    games: int,
    lanes: int,
    players: int,
    seed: int,
    max_offers: int | None,
) -> dict:
    """Every board played under both seat assignments, then averaged per board.

    Two cohorts over the *same* index range -- so the same boards and the same
    dice -- with complementary casters. Averaging a board's two readings kills
    the seat term algebraically instead of hoping it averages out, which it does
    not: `alternating` locks the cast to the index, so a single cohort never
    plays a board the other way round.

    `games` counts games, not boards, so it is halved between the two cohorts
    and an odd count loses one game rather than silently playing a board once.
    """
    half = games // 2
    if half == 0:
        raise ValueError("an antithetic duel needs at least two games")

    readings: dict[int, list[tuple[float, bool]]] = {}
    for flip in (False, True):
        collector = Collector(
            policy,
            lanes=min(lanes, half),
            players=players,
            seed=seed,
            action_cap=4000,
            max_offers=max_offers,
            deal=half,
            opponents=[reference],
            caster=alternating(players, flip=flip),
        )
        for episode in collector.drain():
            readings.setdefault(episode.index, []).append(_paired(episode))

    # Only boards seen both ways can have the seat term removed; a board seen
    # once would reintroduce exactly what this exists to delete.
    both = [v for v in readings.values() if len(v) == 2]
    paired = [(a[0] + b[0]) / 2 for a, b in both]
    wins = sum(a[1] + b[1] for a, b in both)
    boards = len(both)
    n = 2 * boards

    low, high = wilson(wins, n) if n else (0.0, 0.0)
    mean = sum(paired) / boards if boards else 0.0
    spread = (
        (sum((x - mean) ** 2 for x in paired) / (boards - 1)) ** 0.5
        if boards > 1
        else 0.0
    )
    half_width = 1.96 * spread / boards**0.5 if boards > 1 else 0.0
    return {
        "games": n,
        "boards": boards,
        "antithetic": True,
        "wins": wins,
        "win_rate": wins / n if n else 0.0,
        "wilson_low": low,
        "wilson_high": high,
        "paired_vp": mean,
        "paired_vp_low": mean - half_width,
        "paired_vp_high": mean + half_width,
    }


def versus(
    policy: BatchPolicy,
    reference: BatchPolicy,
    *,
    games: int,
    lanes: int,
    players: int,
    seed: int,
    max_offers: int | None,
    antithetic: bool = True,
) -> dict:
    """The learner against one reference, two seats each, seats rotating.

    Reports the win rate with its Wilson interval and — the finer instrument,
    three times on this project's record — paired terminal victory points:
    learner seats' mean minus reference seats' mean within each game, so the
    board and the dice cancel.

    The board and the dice cancel; **the seats do not**, unless `antithetic`.
    `alternating` keys the cast to the game index and the board is keyed to the
    index too, so a cohort samples each entrant on one seat-pair per board and
    never the other. The mean seat effect cancels, the parity-correlated
    residual does not, and it accounted for 55% of a single-order duel's
    variance here while being reported as ordinary noise. It is also why
    swapping the arguments did not negate the result.

    **On by default since 2026-08-24.** Numbers taken before that are still
    valid measurements; what was wrong was their intervals, which counted the
    seat residual as ordinary noise. Pass `antithetic=False` only to reproduce
    an old reading exactly.

    `antithetic=True` plays every board **both** ways -- half the cohort under
    each assignment, same boards, same dice -- and averages the two readings per
    board, so the seat term differences out exactly rather than statistically.
    A self-duel returns 0.0 under it, which is the test. Two orders at half the
    games each already beat one order at full length (SE 0.059 against 0.086);
    this gets the same cancellation inside one run.
    """
    if antithetic:
        return _antithetic(
            policy,
            reference,
            games=games,
            lanes=lanes,
            players=players,
            seed=seed,
            max_offers=max_offers,
        )
    collector = Collector(
        policy,
        lanes=min(lanes, games),
        players=players,
        seed=seed,
        action_cap=4000,
        max_offers=max_offers,
        deal=games,
        opponents=[reference],
        caster=alternating(players),
    )
    episodes = collector.drain()

    wins = 0
    paired: list[float] = []
    for episode in episodes:
        learner = [s for s, pid in enumerate(episode.cast) if pid == 0]
        others = [s for s, pid in enumerate(episode.cast) if pid != 0]
        if episode.outcome.winner in learner:
            wins += 1
        points = episode.outcome.points
        paired.append(
            sum(points[s] for s in learner) / len(learner)
            - sum(points[s] for s in others) / len(others)
        )
    n = len(episodes)
    low, high = wilson(wins, n) if n else (0.0, 0.0)
    mean = sum(paired) / n if n else 0.0
    spread = (sum((x - mean) ** 2 for x in paired) / (n - 1)) ** 0.5 if n > 1 else 0.0
    half = 1.96 * spread / n**0.5 if n > 1 else 0.0
    return {
        "games": n,
        "wins": wins,
        "win_rate": wins / n if n else 0.0,
        "wilson_low": low,
        "wilson_high": high,
        "paired_vp": mean,
        "paired_vp_low": mean - half,
        "paired_vp_high": mean + half,
    }


def rival_rung(directory: str, iteration: int, device: str, board, players: int):
    """The rival run's checkpoint at this exact iteration, or None.

    Matched iteration only — a nearest-checkpoint fallback would quietly turn
    the column into an unmatched comparison, which is the common-opponent trap
    with a second opponent. The caller records a miss rather than hiding it;
    align `--eval-every` with the rival's `--keep-every` (both 25) so evals
    land on checkpoints that exist.
    """
    path = Path(directory) / f"iter-{iteration:05d}.pt"
    if not path.exists():
        return None
    return frozen(str(path), device, board, players)


def ladder(
    policy: NetworkPolicy, rungs: dict[str, BatchPolicy], args
) -> dict[str, dict]:
    """The current weights, argmax'd, against every frozen rung.

    Argmax because sampling is the behaviour distribution PPO needs, not the
    policy worth scoring. One *fixed* eval seed rather than one per iteration:
    every eval replays the same boards, so differences between checkpoints are
    paired rather than riding board luck.
    """
    scorer = NetworkPolicy(
        policy.net, policy.space, policy.layout, device=policy.device, greedy=True
    )
    return {
        name: versus(
            scorer,
            reference,
            games=args.eval_games,
            lanes=args.lanes,
            players=args.players,
            seed=args.seed + 10_000,
            max_offers=args.max_offers,
        )
        for name, reference in rungs.items()
    }


def summarise(
    episodes: Sequence[Episode],
    iteration: int,
    positions: int,
    collect_seconds: float,
    update_seconds: float,
    games: int,
    # Last and defaulted, so `distill_train`'s older positional call still
    # binds: (episodes, iteration, positions, collect, update, games).
    assemble_seconds: float = 0.0,
) -> Progress:
    accepts = proposes = seat_games = 0
    for episode in episodes:
        cast = episode.cast or (0,) * episode.players
        seat_games += sum(1 for pid in cast if pid == 0)
        for trajectory in episode.trajectories:
            for transition in trajectory:
                kind = transition.action.type
                if kind is ActionType.ACCEPT_TRADE:
                    accepts += 1
                elif kind is ActionType.PROPOSE_TRADE:
                    proposes += 1
    return Progress(
        iteration=iteration,
        games=games,
        positions=positions,
        collect_seconds=collect_seconds,
        assemble_seconds=assemble_seconds,
        update_seconds=update_seconds,
        positions_per_second=positions / collect_seconds if collect_seconds else 0.0,
        truncated=sum(e.outcome.truncated for e in episodes) / len(episodes),
        mean_actions=sum(e.outcome.actions for e in episodes) / len(episodes),
        mean_turns=sum(e.outcome.turns for e in episodes) / len(episodes),
        accepts_per_seat_game=accepts / seat_games if seat_games else 0.0,
        proposes_per_seat_game=proposes / seat_games if seat_games else 0.0,
    )


def save(path: Path, payload: dict) -> None:
    """Write then rename, so a crash mid-save cannot destroy the last good one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def prune_recent(directory: Path, keep: int) -> None:
    """Trim the rolling recent-*.pt ring to the newest `keep`.

    The ring exists so a run can rewind to just before a bad update: `latest`
    is overwritten every save and `--keep-every` is too coarse, which is
    exactly how the third blowout's evidence was lost."""
    ring = sorted(directory.glob("recent-*.pt"))
    for stale in ring[: max(0, len(ring) - keep)]:
        stale.unlink()


def preserve_blowout(
    directory: Path,
    iteration: int,
    net_before: dict,
    optimiser_before: dict,
    batch,
    dump_batch: bool,
) -> None:
    """The KL brake fired: write the exact pre-update weights and, optionally,
    the batch that caused it. The next blowout arrives with its evidence
    attached instead of a guess — nothing upstream warns (measured three
    times), so preservation at the brake is the only place to catch it."""
    tag = f"blowout-{iteration:05d}"
    save(
        directory / f"{tag}-pre.pt",
        {"iteration": iteration - 1, "net": net_before, "optimiser": optimiser_before},
    )
    if dump_batch:
        save(directory / f"{tag}-batch.pt", {"batch": batch})


def add_head_flags(parser: argparse.ArgumentParser) -> None:
    """The readout shapes, on every trainer that writes a checkpoint.

    Shared rather than repeated because the shape has to reach the checkpoint's
    `args` under exactly these names: `hexset.netbot.load` rebuilds the config
    from that dict, and a run whose shape is not recorded there is a run whose
    checkpoint cannot be loaded back. Both default to the shape every run on
    record used, so adding these changes nothing until one is passed.
    """
    parser.add_argument(
        "--value-head",
        default="linear",
        choices=VALUE_HEADS,
        help="what the value head reads and how deeply. The head is globally "
        "calibrated (EV +0.474) and a poor sibling ranker (r=+0.574, 42.5%% "
        "top-1), which is what these shapes exist to ablate",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=32,
        help="how many quantiles per seat a --value-head quantile emits; "
        "inert under every other shape. Reaches the checkpoint's args under "
        "this name so `config_from_args` can rebuild the head",
    )
    parser.add_argument(
        "--policy-head",
        default="linear",
        choices=POLICY_HEADS,
        help="per-node-type policy readout depth; holds capacity comparable "
        "across a --value-head ablation",
    )
    parser.add_argument(
        "--detach-value",
        action="store_true",
        help="the value head trains on detached trunk features, so the value "
        "loss cannot reshape what the policy reads; see ModelConfig",
    )
    parser.add_argument(
        "--fused",
        action="store_true",
        help="the learner's forward runs the fused GEMM trunk — same math, "
        "fewer kernels; measured slower on CPU, so this is the GPU update's "
        "opt-in and the collect workers keep the reference path",
    )


def build(args) -> tuple[NetworkPolicy, torch.optim.Optimizer, object]:
    rng = random.Random(args.seed)
    board = random_base_board(rng)
    game = start(board, args.players, rng)
    space = space_for(game)
    graph = static_graph(board.topology)

    torch.manual_seed(args.seed)
    # `getattr` on the head shapes for the same reason `adam_eps` uses it below:
    # this helper is called with hand-built namespaces from the test fixtures,
    # and a namespace that predates a knob should mean "the default shape", not
    # an AttributeError.
    net = HexNet(
        space,
        graph,
        args.players,
        ModelConfig(
            width=args.width,
            rounds=args.rounds,
            value_head=getattr(args, "value_head", "linear"),
            policy_head=getattr(args, "policy_head", "linear"),
            quantiles=int(getattr(args, "quantiles", 32)),
        ),
    ).to(args.device)
    # The same instance attribute `hexset.distill_train` uses: gradient wiring,
    # not architecture, so it lives on the net rather than in ModelConfig and
    # changes nothing about what a checkpoint rebuilds for play.
    net.detach_value = getattr(args, "detach_value", False)
    # Learner-side only: collect workers build their own nets and keep the
    # reference path, which measured faster on CPU.
    net.fused = getattr(args, "fused", False)
    policy = NetworkPolicy(net, space, packing(graph, args.players), device=args.device)
    # eps 1e-5 rather than torch's 1e-8, which is the standard PPO value and
    # matters more here than usual. `masked_log_softmax` zeroes the gradient at
    # illegal positions, and a position offers ~6 legal actions out of 553, so a
    # given logit's row sees gradient from a small minority of a minibatch's
    # rows. At 1e-8 Adam normalises those few tiny, noisy gradients up to a
    # near-full +/-lr step, so rarely-legal logits random-walk at the full
    # learning rate; 1e-5 damps exactly that without touching the well-sampled
    # directions.
    # `getattr` rather than `args.adam_eps`: this helper is also called with
    # hand-built namespaces from the test fixtures and from `distill_train`,
    # whose parsers know nothing about a knob added for PPO. A missing attribute
    # should mean "the standard value", not an AttributeError three frames down.
    optimiser = torch.optim.Adam(
        net.parameters(),
        lr=args.learning_rate,
        eps=getattr(args, "adam_eps", ADAM_EPS),
    )
    return policy, optimiser, space


# Arbitrary but documented: the box every recorded number in this project came
# from has 32 cores, and a single in-process collector was measured pinned to
# 2-3 of them. Below this a lone collector is not obviously
# wrong -- a 4-core laptop has nowhere to shard to.
MANY_CORES = 8


def _crippled(device: str, collect_workers: int, cores: int) -> bool:
    """CPU training, or a lone collector idling most of a many-core box.

    Both are the *defaults* -- every real run on record overrides them -- so
    this only fires for a launch nobody pointed anywhere, which is exactly the
    failure ppo4 had with `--learning-rate`: a flag that quietly did nothing
    for 150 iterations because nothing checked it stuck.
    """
    return device == "cpu" or (collect_workers == 0 and cores > MANY_CORES)


def build_parser() -> argparse.ArgumentParser:
    """Every knob a run has, in one place `hexset.run` can introspect.

    Extracted from `main` so a run manifest can be checked against the real
    parameter set rather than a hand-kept copy of it: `hexset.run.manifest`
    walks this parser's actions to know what a frozen config must contain, so
    adding a flag here cannot silently produce manifests that omit it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    # At or below --games-per-iteration: a cohort deals that many games, so
    # lanes above it never fill. See --collect-mode.
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=32)
    parser.add_argument("--action-cap", type=int, default=4000)
    # Three, not the engine's eight. Greedy at three loses nothing measurable to
    # greedy at eight (52.0% over 2000 games, interval spanning even) and costs
    # about 0.1 victory points, while cutting a game from 2225 actions to 950.
    # The duel below inherits it, since a policy has to be measured on the
    # horizon it learned on.
    parser.add_argument("--max-offers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=2)
    add_head_flags(parser)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--adam-eps", type=float, default=ADAM_EPS)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="global grad-norm clip; measured median pre-clip norm is 0.256, so "
        "0.5 is not binding, and under Adam a constant rescale cannot change "
        "the step anyway. Exposed so that can be re-checked rather than assumed",
    )
    # The learning rate is a controller, not a constant — see `hexset.schedule`.
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "linear", "adaptive"),
        default="constant",
        help="constant keeps --learning-rate; linear anneals it to "
        "--lr-floor of itself by the final iteration; adaptive holds "
        "approx_kl in a band around --target-kl by scaling the rate",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help="the KL the adaptive schedule steers toward, read from the update's "
        "final epoch rather than its all-epoch mean",
    )
    parser.add_argument("--lr-band", type=float, default=2.0)
    parser.add_argument("--lr-factor", type=float, default=1.5)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=1e-2)
    parser.add_argument("--lr-floor", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=0.95)
    # The value head's own horizon, separate from the advantage estimator's.
    # 1.0 is the terminal outcome every run on record trained under; below 1 the
    # target becomes a lambda-return. `benchmarks.horizon --lambdas` measured the
    # label before this existed: at matched noise a mixture is a better teacher
    # than any fixed horizon, and 0.97-0.99 is where that is worth the variance.
    parser.add_argument("--value-lambda", type=float, default=1.0)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=4)
    # A ceiling on the finished epoch's mean approx_kl, 0 off. Pairs with a
    # raised --epochs as "reuse until the trust region objects"; see PPOConfig.
    parser.add_argument("--kl-break", type=float, default=0.0)
    # Which wires connect the value head to learning: both ("gae", every run on
    # record), neither ("none", REINFORCE on the zero-sum terminal return), or
    # only the trunk-shaping loss ("aux"). See PPOConfig for the full account.
    parser.add_argument("--critic", choices=("gae", "none", "aux"), default="gae")
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--checkpoint-dir", default="runs/ppo")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    # `latest.pt` is for resuming and is overwritten; these are for asking, after
    # the fact, when the policy stopped improving. The first run kept only
    # `latest.pt` and that question turned out to be unanswerable.
    parser.add_argument("--keep-every", type=int, default=25, help="0 disables")
    parser.add_argument(
        "--keep-recent",
        type=int,
        default=5,
        help="rolling ring of the last N periodic saves (recent-XXXXX.pt), so "
        "a run can rewind past a bad update without waiting for --keep-every; "
        "0 disables",
    )
    parser.add_argument(
        "--dump-blowout-batch",
        action="store_true",
        default=True,
        help="when the KL brake fires, also write the offending batch next to "
        "the pre-update weights (~0.7 GB per incident, incidents are rare)",
    )
    parser.add_argument("--eval-every", type=int, default=0, help="0 disables")
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument(
        "--rival",
        default="",
        help="another run's checkpoint dir; each eval also duels the rival's "
        "checkpoint at the *matched* iteration, so the ladder carries a "
        "recipe-vs-recipe column and not only the common-opponent one. Align "
        "--eval-every with the rival's --keep-every (both 25) or evals land "
        "between its checkpoints and the column records misses",
    )
    parser.add_argument(
        "--eval-at-start",
        action="store_true",
        help="run the ladder before the first update, so the trend has a baseline",
    )
    # The random duel saturated at 200/200 by iteration 49 of the first run and
    # measured nothing after. The ladder replaces it: fixed rungs whose strength
    # never moves, so a flat reading means the policy stopped improving rather
    # than the yardstick running out.
    parser.add_argument(
        "--parent",
        default="",
        help="checkpoint path: a ladder rung, and the 'parent' mix opponent",
    )
    parser.add_argument(
        "--search-rung",
        default="",
        help="an extra ladder rung by arena entrant name, e.g. 'search2-offers3' "
        "— measured at parity with catanatron AB:2, local, and unlike catanatron "
        "it plays the trading game. Costs search time per eval, so it is opt-in; "
        "adding it leaves the parent and greedy rungs untouched, so the existing "
        "trend stays comparable",
    )
    parser.add_argument(
        "--mix",
        default="",
        help="opponents in the training lanes, e.g. "
        "'greedy=0.15,search2-offers3=0.1' — the share of games each plays, on "
        "alternating seat pairs; their seats are never trained on. A name is "
        "'greedy', 'parent', or any arena entrant spec, so a held-out opponent "
        "can be collected against and not merely evaluated on",
    )
    # The single in-process collector measured ~43 s of a ~109 s iteration on
    # 2-3 of the box's 32 cores. Sharding it is the recorded largest untaken
    # speedup; see `hexset.collect` for the shape and what was rejected.
    parser.add_argument(
        "--collect-workers",
        type=int,
        default=0,
        help="shard collection across N processes with CPU inference; "
        "0 keeps the single in-process collector",
    )
    parser.add_argument(
        "--collect-mode",
        choices=("cohort", "stream"),
        default="cohort",
        help="cohort deals exactly --games-per-iteration games and plays every "
        "one to completion, so the batch is on-policy and unbiased in game "
        "length; stream is the old behaviour, which refills a lane the moment "
        "its game ends and carries the unfinished ones across the weight sync",
    )
    parser.add_argument(
        "--async-collect",
        action="store_true",
        help="prefetch iteration k+1 during the GPU update of k; requires "
        "--collect-workers > 1 and trains on a policy one iteration stale",
    )

    # The GPU update measured ~106 µs a position-pass with compile, precision,
    # minibatch size and CPU threads all ruled out (2026-08-17).
    # Data-parallel CPU processes are the one lever left; see `hexset.ddp`.
    parser.add_argument(
        "--update-workers",
        type=int,
        default=0,
        help="shard the PPO update across N CPU processes with a central Adam "
        "step; 0 keeps the single-device update",
    )
    parser.add_argument("--resume", action="store_true")
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
            "usage: python -m hexset.train <run-directory>\n"
            "  create one with: python -m hexset.run.init --mode train --name NAME -- <flags>\n"
            "  the flags this used to accept are frozen into the run's config/ instead."
        )
    manifest = load(tokens[0])
    if manifest.mode != "train":
        raise SystemExit(
            f"{tokens[0]} is a {manifest.mode} run; launch it with "
            f"python -m {MODULES[manifest.mode]}"
        )
    args = manifest.namespace()
    # Cross-parameter validation below still reports through the parser, so a
    # bad combination frozen into a manifest fails with the same sentence it
    # would have as a flag.
    parser = build_parser()
    if args.async_collect and args.collect_workers <= 1:
        parser.error("--async-collect requires --collect-workers > 1")
    if args.async_collect and args.update_workers > 1:
        parser.error("--async-collect is for the GPU update, not CPU update workers")
    if args.kl_break > 0 and args.update_workers > 1:
        parser.error("--kl-break is implemented in hexset.ppo.update only; the "
                     "sharded crew runs its own epoch loop and would ignore it")
    if args.kl_break > 0 and args.lr_schedule == "adaptive":
        parser.error("--kl-break truncates the epochs the adaptive controller "
                     "reads its gauge from; run one governor at a time")
    if args.collect_mode == "cohort":
        if args.lanes > args.games_per_iteration:
            parser.error(
                f"--lanes {args.lanes} exceeds --games-per-iteration "
                f"{args.games_per_iteration}: a cohort deals that many games and "
                "no more, so the surplus lanes would sit empty. Lanes are the "
                "concurrency, not the cohort — lower them, or raise the cohort"
            )
        if args.async_collect:
            parser.error(
                "--async-collect dispatches the next cohort before the update "
                "lands, which is exactly the staleness --collect-mode cohort "
                "exists to remove; pass --collect-mode stream to accept it"
            )

    # Printed unconditionally rather than only on the crippled path: a launch
    # that overrides every one of these still benefits from the confirmation,
    # and a launch that does not gets it without having to ask.
    cores = os.cpu_count() or 1
    print(
        f"device={args.device} collect-workers={args.collect_workers} "
        f"update-workers={args.update_workers} ({cores} cores on this box)",
        file=sys.stderr,
    )
    if _crippled(args.device, args.collect_workers, cores):
        print(
            "WARNING: this is the crippled default -- cpu training and/or a "
            "single in-process collector, which measured ~43s of a ~109s "
            "iteration on 2-3 of a 32-core box. Pass --device cuda and "
            "--collect-workers <N> unless this is deliberate (e.g. a smoke "
            "test).",
            file=sys.stderr,
        )

    # Deliberately absent: --gamma. See `hexset.rewards`.
    config = PPOConfig(
        lam=args.lam,
        value_lam=args.value_lambda,
        clip=args.clip,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy,
        max_grad_norm=args.max_grad_norm,
        epochs=args.epochs,
        minibatch=args.minibatch,
        learning_rate=args.learning_rate,
        critic=args.critic,
        kl_break=args.kl_break,
    )

    policy, optimiser, _ = build(args)
    directory = Path(args.checkpoint_dir)
    latest = directory / "latest.pt"
    log = directory / "log.jsonl"

    start_iteration = 0
    first_game = 0
    if args.resume and not latest.exists():
        # `--resume` with nothing to resume used to fall through and start from
        # iteration 0 on freshly initialised weights, silently. On a 150-iteration
        # GPU block that is hours of compute spent discarding the campaign, and
        # the log gives no sign of it — it simply begins at 0 and looks healthy.
        # A typo'd `--checkpoint-dir`, or a new directory nobody remembered to
        # seed, is all it takes. Same failure shape as the learning rate that
        # `--resume` used to discard: a flag that quietly does nothing.
        raise SystemExit(
            f"--resume was given but {latest} does not exist; "
            "seed the directory with the checkpoint to continue from, "
            "or drop --resume to start a fresh run"
        )
    if args.resume:
        state = torch.load(latest, map_location=args.device, weights_only=False)
        policy.net.load_state_dict(state["net"])
        optimiser.load_state_dict(state["optimiser"])
        # `Optimizer.load_state_dict` rebuilds `param_groups` from the *saved*
        # groups, keeping only `params` from the live ones — so every
        # hyperparameter, `lr` and `eps` included, comes back from the
        # checkpoint and the command line is silently discarded. That cost this
        # project a whole 150-iteration block: ppo4 launched with
        # `--learning-rate 6e-4`, recorded 6e-4 in both its `args` and `config`
        # blobs, and stepped Adam at the 3e-4 baked into the checkpoint it
        # resumed from. Every gauge matched the previous run exactly, because
        # the configuration *was* the previous run's. Re-assert the CLI values,
        # and log the live rate every iteration so a knob that does not move is
        # visible in the run rather than in a post-mortem.
        set_lr(optimiser, args.learning_rate)
        for group in optimiser.param_groups:
            group["eps"] = args.adam_eps
        start_iteration = state["iteration"]
        first_game = state["games_started"]
        # `map_location` moved everything in the checkpoint to the training
        # device, but the RNG state must be a *CPU* ByteTensor — on a cuda
        # resume this line is the difference between restoring and crashing.
        torch.set_rng_state(state["torch_rng"].cpu())
        print(
            f"resumed at iteration {start_iteration}, game {first_game}",
            file=sys.stderr,
        )

    # The same board `build` derived from the seed, for loading the parent:
    # base boards all share one topology, so any of them names the right shapes.
    board = random_base_board(random.Random(args.seed))
    parent = (
        frozen(args.parent, args.device, board, args.players) if args.parent else None
    )

    mix = parse_mix(args.mix)
    check_mix(mix, have_parent=parent is not None)

    rungs: dict[str, BatchPolicy] = {}
    if parent is not None:
        rungs["parent"] = parent
    rungs["greedy"] = greedy_opponent(args.seed + 78, args.max_offers, args.lanes)
    if args.search_rung:
        rungs[args.search_rung] = named_opponent(
            args.search_rung, args.seed + 79, args.lanes
        )

    if args.collect_workers > 1:
        shard = max(1, -(-args.lanes // args.collect_workers))
        collector = ParallelCollector(
            [
                WorkerSpec(
                    seed=args.seed,
                    players=args.players,
                    lanes=shard,
                    action_cap=args.action_cap,
                    max_offers=args.max_offers,
                    first_game=first_game + worker,
                    stride=args.collect_workers,
                    width=args.width,
                    rounds=args.rounds,
                    torch_seed=args.seed + 100_000 + worker,
                    value_head=args.value_head,
                    policy_head=args.policy_head,
                    quantiles=args.quantiles,
                    mix=tuple(mix),
                    parent=args.parent,
                    cohort=args.collect_mode == "cohort",
                )
                for worker in range(args.collect_workers)
            ]
        )
    else:
        # Note what the comprehension this replaced did with a name that was
        # neither: it fell through to `parent`, so a third opponent would have
        # silently been the parent checkpoint. Only the validation above kept
        # that unreachable, and the validation now admits every entrant spec.
        opponents: list[BatchPolicy] = mix_opponents(
            mix,
            seed=args.seed,
            max_offers=args.max_offers,
            lanes=args.lanes,
            parent=(lambda: parent) if parent is not None else None,
        )
        collector = Collector(
            policy,
            lanes=args.lanes,
            fill=args.collect_mode != "cohort",
            players=args.players,
            seed=args.seed,
            action_cap=args.action_cap,
            first_game=first_game,
            max_offers=args.max_offers,
            opponents=opponents,
            caster=mix_caster(mix, args.players, args.seed) if mix else None,
        )

    crew = None
    if args.update_workers > 1:
        from .ddp import UpdateCrew, UpdateSpec

        crew = UpdateCrew(
            [
                UpdateSpec(
                    seed=args.seed,
                    players=args.players,
                    width=args.width,
                    rounds=args.rounds,
                    value_head=args.value_head,
                    policy_head=args.policy_head,
                    quantiles=args.quantiles,
                )
                for _ in range(args.update_workers)
            ]
        )

    controller = AdaptiveLR(
        target_kl=args.target_kl,
        band=args.lr_band,
        factor=args.lr_factor,
        min_lr=args.lr_min,
        max_lr=args.lr_max,
    )

    directory.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()

    if args.eval_at_start:
        # A resumed run's first ladder reading doubles as a regression control:
        # against its own parent the starting weights must duel to a dead heat,
        # so a first reading away from 50% flags the harness, not the policy.
        baseline = ladder(policy, rungs, args)
        line = json.dumps({"iteration": start_iteration - 1, "ladder": baseline})
        print(line, flush=True)
        with log.open("a") as handle:
            handle.write(line + "\n")

    # `ParallelCollector.collect` already forwards to each worker's `cohort`
    # when the specs ask for it, so only the in-process collector needs steering.
    if args.collect_mode == "cohort" and isinstance(collector, Collector):
        draw = collector.cohort
    else:
        draw = collector.collect

    prefetched: list[Episode] | None = None
    prefetched_seconds = 0.0
    if args.async_collect and start_iteration < args.iterations:
        # Prime the pipeline. Later cohorts are dispatched immediately before
        # the preceding update and therefore act with the pre-update policy.
        # That one-iteration staleness is why this path is explicit opt-in.
        started = time.perf_counter()
        collector.sync(policy.net)
        prefetched = draw(args.games_per_iteration)
        prefetched_seconds = time.perf_counter() - started

    for iteration in range(start_iteration, args.iterations):
        if args.async_collect:
            assert prefetched is not None
            episodes = prefetched
            collected = prefetched_seconds
        else:
            started = time.perf_counter()
            if isinstance(collector, ParallelCollector):
                # The workers act with last iteration's weights until this lands,
                # so the sync is inside the loop and before the collect, always.
                collector.sync(policy.net)
            episodes = draw(args.games_per_iteration)
            collected = time.perf_counter() - started
        games = collector.games

        started = time.perf_counter()
        batch = assemble(episodes, policy.layout, config)
        assembled = time.perf_counter() - started
        next_collect_started: float | None = None
        if args.async_collect and iteration + 1 < args.iterations:
            # Ship the current policy before `update` mutates it, then let the
            # CPU workers play while the GPU owns the learner's critical path.
            next_collect_started = time.perf_counter()
            collector.sync(policy.net)
            collector.start_collect(args.games_per_iteration)

        # ~3 MB a clone: cheap insurance. The brake's third firing lost its
        # evidence because nothing held the pre-update state.
        net_before = {k: v.detach().cpu().clone() for k, v in policy.net.state_dict().items()}
        optimiser_before = {
            "state": {
                k: {kk: (vv.detach().cpu().clone() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
                for k, v in optimiser.state_dict()["state"].items()
            },
            "param_groups": optimiser.state_dict()["param_groups"],
        }

        started = time.perf_counter()
        if crew is not None:
            stats = crew.update(policy, optimiser, batch, config)
        else:
            stats = update(policy, optimiser, batch, config)
        updated = time.perf_counter() - started

        if config.kl_break and stats.epochs_taken < config.epochs:
            preserve_blowout(
                directory,
                iteration + 1,
                net_before,
                optimiser_before,
                batch,
                args.dump_blowout_batch,
            )

        if next_collect_started is not None:
            prefetched = collector.finish_collect()
            prefetched_seconds = time.perf_counter() - next_collect_started
        else:
            prefetched = None

        progress = summarise(
            episodes,
            iteration,
            len(batch),
            collected,
            updated,
            games,
            assemble_seconds=assembled,
        )
        record = {
            **asdict(progress),
            **asdict(stats),
            # A column, not just a config field. `PPOConfig.learning_rate` read
            # back a correct-looking rate that nothing applied for 150
            # iterations; the rule that came out of it is that the quantity a
            # run is steered by has to appear in its own log.
            "collect_mode": args.collect_mode,
            # Same rule, applied ahead of the blocks that sweep them rather than
            # after: `lam` is the only horizon control there is under gamma 1,
            # and `minibatch` sets how many steps an iteration takes.
            "lam": config.lam,
            "value_lam": config.value_lam,
            "minibatch": config.minibatch,
            # Same rule again: an arm's treatment appears in its own log, so a
            # row can never be attributed to the wrong critic wiring.
            "critic": config.critic,
            "elapsed": time.perf_counter() - began,
        }

        # The rate for the *next* update, chosen from the gauge this one just
        # measured. Read off the final epoch rather than `approx_kl`: the
        # all-epoch mean includes epoch 1, where nothing has stepped yet, and
        # understates the finished update's divergence by ~1.7x on measurement —
        # steering a controller by it would target the wrong number by that
        # factor. Applied after the record is built, so a row always reports the
        # rate its own update used (`stats.lr`), never the successor's.
        if args.lr_schedule == "adaptive":
            gauge = stats.approx_kl_last_epoch
            moved = controller.next_lr(current_lr(optimiser), gauge)
            if controller.deaf(moved, gauge):
                # Saturated at a clamp and still out of band. Distinct from a
                # converged controller, which also holds the rate still.
                record["lr_deaf"] = True
            set_lr(optimiser, moved)
        elif args.lr_schedule == "linear":
            set_lr(
                optimiser,
                linear_anneal(
                    args.learning_rate, iteration + 1, args.iterations, args.lr_floor
                ),
            )

        if args.eval_every and (iteration + 1) % args.eval_every == 0:
            eval_rungs = dict(rungs)
            if args.rival:
                opponent = rival_rung(
                    args.rival, iteration + 1, args.device, board, args.players
                )
                if opponent is None:
                    # A miss is data, not silence: the alignment note on the
                    # flag is only checkable if misses appear in the log.
                    record["rival_checkpoint_missing"] = iteration + 1
                else:
                    eval_rungs["rival"] = opponent
            record["ladder"] = ladder(policy, eval_rungs, args)

        line = json.dumps(record)
        print(line, flush=True)
        with log.open("a") as handle:
            handle.write(line + "\n")

        if (iteration + 1) % args.checkpoint_every == 0 or iteration + 1 == args.iterations:
            state = {
                "iteration": iteration + 1,
                "games_started": collector.games_started(),
                "net": policy.net.state_dict(),
                "optimiser": optimiser.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "args": vars(args),
                "config": asdict(config),
            }
            save(latest, state)
            if args.keep_recent:
                save(directory / f"recent-{iteration + 1:05d}.pt", state)
                prune_recent(directory, args.keep_recent)
            if args.keep_every and (iteration + 1) % args.keep_every == 0:
                save(directory / f"iter-{iteration + 1:05d}.pt", state)

    if isinstance(collector, ParallelCollector):
        collector.close()
    if crew is not None:
        crew.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
