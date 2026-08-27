"""Gate A2 of the variance screen: does a quantile head give a *better mean*?

Registered in `agents/reference/variance-screen.md` under candidate 3. Gate A1
closed PASS -- the seat-conditional terminal return is measurably non-Gaussian
(+0.2215 VP of Wasserstein-1 over the instrument's own matched null) -- so a
per-seat quantile head has a premise. This asks the only question that decides
whether the *critic wire* improves: GAE reads `V`, one scalar per seat, and
nothing downstream reads a spread, so a richer head helps the run through that
wire only if its mean estimate is better than the MSE head's.

It is a pure offline supervised experiment. No self-play training run, no
league, no heat. The trunk is frozen and both heads read the identical cached
feature vector, so the only difference between the two arms is the loss.

## What is scored, and why it is not MSE

`benchmarks.floor` splits a head's squared error exactly, per position:

    E[(return - prediction)^2] = Var(return | position) + (E[return|position] - prediction)^2
             mean squared error =        floor          +            bias^2

At the frontier checkpoint the floor is 83% of the error (`a1-floor-lam095`:
0.04185 of 0.05015). No head can remove it, so an MSE comparison between two
heads would be five parts dice to one part head. **The pass line is on
held-out bias^2 -- the quantile head must cut it by at least 20%** -- and
plain MSE and the floor share are reported beside it as context only.

A per-position bias^2 needs `E[return | position]`, which needs many rollouts
from the one position. That is exactly what `benchmarks.floor --dump-returns`
already produces and `benchmarks.return_shape` already reads, so the held-out
set here is a floor dump: this module reuses `floor.split` and `floor.pool` on
it, unchanged, so the bias^2 quoted here and the bias^2 in the A1 floor report
are the same arithmetic on the same samples.

## Recovering the dump's positions

A floor dump keeps each position's rollout returns, its seat, its progress and
the checkpoint head's prediction -- but not the position, so the features a new
head would need are not in the file. They are recoverable: `benchmarks.floor`'s
seeding phase is a pure function of the checkpoint and `(seed, seed_games,
positions, players, action_cap)`, and it costs ~8 seconds against the ~45
minutes the rollouts cost. `recover` replays it and then proves the replay
landed on the same 128 positions two ways -- every position's `(seat, progress,
prediction)` triple must match the dump *bit for bit*, and the checkpoint's own
value on the re-encoded observation must match the recorded prediction to
`--prediction-tolerance`. The second check is a tolerance rather than an
equality because the recorded number came out of a batch of 16 collector lanes
and this one comes out of a batch of 128, and float32 matmul reassociation
moves the last bit or two; the first check is the exact one and it is the proof
that the positions are the same.

## The two traps this construction has been bitten by

**Rotation.** `catan.ppo.rotate` puts a seat's own payoff in component 0, which
is the frame the encoder fed the trunk and the frame the head emits in. The
dump's `prediction` is `Choice.value[0]` and its `returns` are
`reward(outcome)[seat]` -- both the acting seat's own payoff -- so both heads
are scored on column 0 against them. Targets come from `head_shape.rows`, which
makes the same two calls PPO's `assemble` makes, rather than a fourth copy of
the rotation.

**Label scale.** Neither head normalises its target. Both see the raw
`rotate(reward(outcome), seat)` vector in reward units, and the comparison
would be void if they did not.

## Both arms have to be sitting at their own plateau

The register says identical optimiser, learning rate, steps and seed, and this
does that -- but "identical steps" is not "equally converged", and the
difference decides the gate. The pinball loss has a gradient of size `tau` or
`1 - tau` however small the residual is, while the squared error's shrinks with
it, so at this project's residual scale (~0.2 reward units) the quantile arm
descends far faster from the same initialisation at the same learning rate.

Measured while building this, on a 128-game shakedown cohort at a quarter of
the default size -- **a wiring check, not the gate, and not the registered
configuration**: at 8 epochs, `head_shape`'s default, the quantile arm read a
43.7% bias^2 cut over the MSE arm and the pass line was cleared; at 60 epochs,
where both training curves are flat to the fifth decimal, the two arms landed
on 0.006184 and 0.006191 and it was not. The 8-epoch reading was a measurement
of convergence speed wearing the gate's clothes.

So `--epochs` defaults high enough for both curves to flatten, and every run
reports each arm's `plateau` -- the relative fall in training loss over its last
`--plateau-window` epochs. **A gate verdict from a run whose arms have not
plateaued is a verdict about the learning rate.** The number is in the JSON and
on the stdout line so that cannot be assumed rather than read.

## Held out by game, never by position

Every row of a seat's trajectory carries that game's terminal outcome, so two
rows of one game share a label and a row-wise split reports a training loss as
a held-out one. The cohort split reuses `head_shape.split`, which is over
episodes. The dump held-out set is disjoint from the cohort at a stronger
level: its games were dealt by `benchmarks.floor`'s own collector at seed
`dump_seed + 1`, and the cohort is dealt at `--seed + COHORT_SEED_OFFSET`,
which `main` refuses to let collide.

## Why the cohort plays the dump's board by default

The dump's positions all sit on one board -- `benchmarks.floor` fixes it so its
branching rollouts replay one position. Training the two heads on random boards
would fold a board-generalisation gap into both arms and could swamp the 20%
the gate is looking for; it is a confound the register never asked about.
`--random-boards` plays the collector's production default instead. Sharing the
board is not a leak: the games differ, so a dump position's label -- a mean over
128 fresh rollouts -- is not in the training set under any seat or any game.

    python -m benchmarks.head_swap --checkpoint runs/lam095/latest.pt \\
        --held-out-dump runs/eval/a1-returns-lam095.json --json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

# Unlike `benchmarks.floor` and `benchmarks.head_shape`, torch is imported at
# module scope: the arithmetic worth getting wrong here *is* the differentiable
# loss, so there is nothing left to keep importable without it.
from benchmarks.floor import Sampling, collect, pool, split as split_error, _stages
from benchmarks.head_shape import rows, split as split_games
from benchmarks.throughput import environment
from benchmarks.value_head import explained
from catan.board.board import random_base_board
from catan.encoding import encode
# `_head`, `_output` and `_DEEP` are `catan.model`'s own, and are imported
# rather than restated so that a change to the production head's shape or its
# initialisation cannot silently leave this comparison measuring a head the run
# does not have. `_head` is what builds `self.value`; `_DEEP` is which shapes
# get a hidden layer; `_output` is the layer whose gain sets the output scale.
from catan.model import _DEEP, _head, _output, pack, unpack
from catan.netbot import load
from catan.policy import NetworkPolicy
from catan.selfplay import Collector

# `benchmarks.floor` seeds the collector that produces the dump's positions at
# `dump_seed + 1`. The training cohort is offset far away from that so the two
# cannot be dealt the same games when they share a board, which is the one leak
# this construction has to rule out; `main` checks it rather than trusting it.
COHORT_SEED_OFFSET = 8191

# The sampling rate and the lane count `benchmarks.floor` seeds with. Replaying
# its position sampling means replaying these too, so they are named here and
# not retyped at the call site.
FLOOR_SAMPLE_RATE = 0.02
FLOOR_SEED_LANES = 16


def quantile_levels(count: int) -> Tensor:
    """The midpoint levels `(i + 0.5) / Q`, as the register specifies.

    Midpoints rather than `i / (Q - 1)` because the pinball loss at level 0 or 1
    is one-sided: it is minimised by the sample minimum or maximum, which no
    finite sample estimates stably, and those two atoms would then drag the
    mean-of-quantiles the gate reads. Midpoints also make the levels symmetric
    about 0.5, which is what makes the mean of the quantiles equal the mean of a
    symmetric distribution exactly rather than approximately.
    """
    if count < 1:
        raise ValueError(f"a quantile head needs at least one level, got {count}")
    return (torch.arange(count, dtype=torch.float32) + 0.5) / count


def quantile_huber_loss(
    predicted: Tensor, target: Tensor, levels: Tensor, kappa: float
) -> Tensor:
    """Pinball loss at `levels`, Huberised below `kappa`, meaned over everything.

    `predicted` is `(B, players, Q)`, `target` is `(B, players)`: one sampled
    return per seat per position, which is the same one-sample-per-transition
    setting QR-DQN trains in. The quantile levels are not constrained to come
    out sorted and are not sorted here -- the gate reads their mean, which no
    reordering changes.

    **`kappa` is small on purpose and this is the one number that had to be
    chosen rather than inherited.** Huberising the pinball loss replaces it,
    within `kappa` of the target, with a quadratic -- and a quadratic weighted
    by `|tau - 1{u<0}|` is minimised at the *expectile*, not the quantile. QR-
    DQN's kappa=1 is small against Atari returns in the hundreds; against this
    project's returns, which `catan.rewards.relative_points` divides by ten into
    a range of about +-1 with a standard deviation near 0.2, kappa=1 would make
    the loss quadratic everywhere and quietly turn the head into an expectile
    head. The default is one lattice step of the return (1/30 of a reward unit,
    a third of a victory point, the resolution the label actually has), so the
    minimiser is displaced from the true quantile by at most one quantum of a
    label that has no finer resolution than that. `kappa=0` is the exact
    pinball loss.

    Meaned over Q rather than summed, so the loss sits at the same magnitude as
    the MSE arm's and one learning rate serves both. Summing would multiply the
    quantile arm's gradient by Q and make the two arms differ in effective step
    size as well as in loss, which is precisely the confound this experiment is
    built to avoid.
    """
    if kappa < 0.0:
        raise ValueError(f"a Huber width cannot be negative, got {kappa}")
    difference = target.unsqueeze(-1) - predicted
    magnitude = difference.abs()
    if kappa > 0.0:
        element = torch.where(
            magnitude <= kappa,
            0.5 * difference * difference,
            kappa * (magnitude - 0.5 * kappa),
        ) / kappa
    else:
        element = magnitude
    # The indicator is piecewise constant in `predicted`, so it carries no
    # gradient; taken off the detached difference to say so rather than to rely
    # on a bool cast happening not to build one.
    below = difference.detach().lt(0.0).to(element.dtype)
    return ((levels - below).abs() * element).mean()


def initialise_head(head: nn.Module) -> None:
    """`CatanNet._initialise`'s convention for a value head, on a bare head.

    Orthogonal everywhere with zeroed biases, gain sqrt(2) on a hidden layer
    because it feeds a SiLU and is trunk by every property that matters, gain
    1.0 on the layer that emits the number because it predicts at its target's
    scale. Restating the rule here rather than reusing `_initialise` is
    unavoidable -- that method walks a whole `CatanNet` -- so both heads are put
    through this one function rather than through two lookalikes.
    """
    for module in head.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=2**0.5)
            nn.init.zeros_(module.bias)
    nn.init.orthogonal_(_output(head).weight, gain=1.0)


class MeanHead(nn.Module):
    """The production value head: `players` outputs, squared error.

    Built by `catan.model._head` at the checkpoint's own width and depth, so
    this arm is the head the run already has, refit -- not a lookalike.
    """

    def __init__(self, value_in: int, players: int, width: int, *, deep: bool) -> None:
        super().__init__()
        self.module = _head(value_in, players, width, deep=deep)
        initialise_head(self.module)

    def mean(self, features: Tensor) -> Tensor:
        return self.module(features)

    def loss(self, features: Tensor, target: Tensor) -> Tensor:
        return (self.module(features) - target).pow(2).mean()


class QuantileHead(nn.Module):
    """`players x Q` outputs, quantile Huber loss, mean estimate = mean of quantiles.

    The same `_head` builder at the same width and depth as `MeanHead`, widened
    only in its output: everything the two arms could differ in other than the
    loss is held fixed by construction rather than by care.
    """

    def __init__(
        self,
        value_in: int,
        players: int,
        width: int,
        *,
        deep: bool,
        quantiles: int,
        kappa: float,
    ) -> None:
        super().__init__()
        self.players = players
        self.quantiles = quantiles
        self.kappa = kappa
        self.module = _head(value_in, players * quantiles, width, deep=deep)
        initialise_head(self.module)
        # A buffer, so `.to(device)` moves the levels with the parameters.
        self.register_buffer("levels", quantile_levels(quantiles))

    def spread(self, features: Tensor) -> Tensor:
        return self.module(features).view(-1, self.players, self.quantiles)

    def mean(self, features: Tensor) -> Tensor:
        """The number GAE would read. Nothing downstream reads the spread."""
        return self.spread(features).mean(-1)

    def loss(self, features: Tensor, target: Tensor) -> Tensor:
        return quantile_huber_loss(self.spread(features), target, self.levels, self.kappa)


@dataclass
class Dataset:
    """Cached trunk features, their targets, and which game each row came from.

    `games` is `(N, 2)` of `(seed, index)` -- the identity `catan.ppo.assemble`
    already keys a game by -- so a split can be audited for the leak it exists
    to prevent instead of assumed clean.
    """

    features: Tensor
    targets: Tensor
    games: np.ndarray

    def __len__(self) -> int:
        return int(self.features.shape[0])


def labelled_rows(episodes: Sequence) -> tuple[list, np.ndarray, np.ndarray]:
    """`head_shape.rows`, plus the `(seed, index)` of the game behind each row."""
    observations, targets = rows(episodes)
    games = np.concatenate(
        [
            np.repeat(
                np.asarray([[episode.seed, episode.index]], dtype=np.int64),
                sum(len(trajectory) for trajectory in episode.trajectories),
                axis=0,
            )
            for episode in episodes
        ]
    )
    return observations, targets, games


def trunk_features(net, layout, observations: Sequence, *, device, chunk: int) -> Tensor:
    """Exactly the tensor `CatanNet.value` is applied to, for every observation.

    Captured with a forward pre-hook on `net.value` rather than reassembled
    here. `CatanNet._read_value` concatenates a different set of trunk tensors
    per `value_head` shape -- the global token alone for `linear`, plus three
    max-pools for `pooled`, plus an attention read for `attn` -- and a second
    copy of that dispatch in this file would be a way for the two heads to end
    up reading something the production head does not. The hook takes whatever
    the checkpoint's own shape assembled, so both arms read the checkpoint's own
    features by construction.

    `no_grad` and a frozen trunk: the features are a fixed design matrix, so
    they are computed once and the fit never touches the trunk again.
    """
    captured: list[Tensor] = []
    handle = net.value.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().cpu())
    )
    try:
        with torch.no_grad():
            for start in range(0, len(observations), chunk):
                packed = pack(layout, list(observations[start : start + chunk]))
                net(*unpack(layout, packed.to(device)))
    finally:
        handle.remove()
    return torch.cat(captured)


def dataset_from(
    net, layout, episodes: Sequence, *, device, chunk: int, block: int
) -> Dataset:
    """Trunk features and targets for every decision in these games.

    Episodes are encoded a block at a time because an `Observation` is about
    5 KB and a 512-game cohort is roughly half a million of them -- packing the
    whole cohort before running the trunk would need a couple of gigabytes to
    produce 120 MB of features.
    """
    features, targets, games = [], [], []
    for start in range(0, len(episodes), block):
        group = list(episodes[start : start + block])
        observations, target, game = labelled_rows(group)
        features.append(
            trunk_features(net, layout, observations, device=device, chunk=chunk)
        )
        targets.append(torch.from_numpy(target))
        games.append(game)
    return Dataset(torch.cat(features), torch.cat(targets), np.concatenate(games))


def fit(
    head: nn.Module,
    data: Dataset,
    *,
    device,
    epochs: int,
    minibatch: int,
    learning_rate: float,
    seed: int,
) -> list[float]:
    """Adam over the head alone; returns each epoch's mean training loss.

    The shuffle runs off a `torch.Generator` seeded from `seed` alone, so two
    heads fit with the same seed walk the same rows in the same order in the
    same minibatches. That is checked by a test rather than left to inspection,
    because the whole experiment is the claim that the loss is the only
    difference between the arms.
    """
    optimiser = torch.optim.Adam(head.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    size = len(data)
    history = []
    for _ in range(epochs):
        order = torch.randperm(size, generator=generator)
        total = 0.0
        for start in range(0, size, minibatch):
            index = order[start : start + minibatch]
            features = data.features.index_select(0, index).to(device)
            target = data.targets.index_select(0, index).to(device)

            loss = head.loss(features, target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            # Weighted by rows, so a short final minibatch does not count as
            # much as a full one in the epoch's mean.
            total += float(loss.detach()) * index.numel()
        history.append(round(total / size, 5))
    return history


def plateau(history: Sequence[float], window: int) -> float:
    """How much further an arm's training loss fell over its last `window` epochs.

    Reported per arm because the gate is only a statement about the two heads
    if both are at their own optimum; see the module docstring for the run where
    reading it at 8 epochs instead of 60 flipped the verdict. A value near zero
    is a flat curve. Negative means the loss rose, which at a fixed learning
    rate is the other way an arm can fail to be converged.
    """
    if len(history) <= window:
        return float("nan")
    earlier = history[-window - 1]
    if earlier == 0.0:
        return 0.0
    return (earlier - history[-1]) / abs(earlier)


def predict_mean(head: nn.Module, features: Tensor, *, device, chunk: int) -> np.ndarray:
    """A head's `(N, players)` mean estimate over a whole split, in chunks."""
    out = []
    with torch.no_grad():
        for start in range(0, features.shape[0], chunk):
            block = features[start : start + chunk].to(device)
            out.append(head.mean(block).cpu().numpy())
    return np.concatenate(out)


@dataclass(frozen=True)
class HeldOut:
    """A floor dump's positions, recovered far enough to score a new head on.

    `returns` is each position's rollout sample -- the estimate of
    `E[return | position]` that makes a per-position bias^2 possible at all.
    `reference` is the checkpoint's own prediction, carried through so the two
    refit heads are reported beside the head the run actually has.
    """

    observations: tuple
    progress: tuple[float, ...]
    seats: tuple[int, ...]
    returns: tuple[np.ndarray, ...]
    reference: tuple[float, ...]


def recover(
    dump: dict,
    checkpoint: str,
    *,
    seed_games: int,
    players: int,
    action_cap: int,
    tolerance: float,
) -> tuple[HeldOut, object]:
    """Replay `benchmarks.floor`'s seeding phase to get the dumped positions back.

    See the module docstring for why this is a replay and how it is proved to
    have landed on the same positions. Returns the held-out set and the
    `catan.netbot.Loaded` checkpoint, so the caller does not load it twice.
    """
    seed = int(dump["seed"])
    board = random_base_board(random.Random(seed))
    loaded = load(checkpoint, board.topology)
    if loaded.iteration != dump.get("iteration"):
        raise SystemExit(
            f"{checkpoint} is iteration {loaded.iteration} but the dump was "
            f"written from iteration {dump.get('iteration')}; the replay below "
            "reproduces the dump's positions only under the dump's own net"
        )

    generator = torch.Generator().manual_seed(seed)
    policy = NetworkPolicy(
        loaded.policy.net,
        loaded.space,
        loaded.policy.layout,
        greedy=False,
        generator=generator,
    )
    rng = random.Random(seed + 2)
    seeding = Collector(
        Sampling(policy, rate=FLOOR_SAMPLE_RATE, rng=rng),
        lanes=min(FLOOR_SEED_LANES, seed_games),
        players=players,
        seed=seed + 1,
        action_cap=action_cap,
        max_offers=loaded.max_offers,
        deal=seed_games,
        board=board,
    )
    kept = collect(seeding.drain())
    wanted = dump["positions"]
    if len(kept) < len(wanted):
        raise SystemExit(
            f"the replay kept {len(kept)} positions but the dump holds "
            f"{len(wanted)}; --dump-seed-games does not match the run that "
            "wrote it"
        )
    chosen = rng.sample(kept, len(wanted))

    for row, ((snapshot, progress), recorded) in enumerate(zip(chosen, wanted)):
        if (
            snapshot.seat != recorded["seat"]
            or round(progress, 3) != recorded["progress"]
            or snapshot.prediction != recorded["prediction"]
        ):
            raise SystemExit(
                f"the replay diverged from the dump at position {row}: replayed "
                f"seat {snapshot.seat} progress {round(progress, 3)} prediction "
                f"{snapshot.prediction!r}, dump has seat {recorded['seat']} "
                f"progress {recorded['progress']} prediction "
                f"{recorded['prediction']!r}"
            )

    observations = tuple(encode(snapshot.game, snapshot.seat) for snapshot, _ in chosen)
    reference = np.asarray([row["prediction"] for row in wanted], dtype=np.float64)
    # The one thing the triple match cannot prove: that re-encoding the snapshot
    # reproduces the observation the head was given. `catan.game.imagine`
    # reshuffles the deck, which the encoder does not see, so this should hold
    # -- and if the encoding ever grows a field that it does see, this is where
    # it is caught rather than in a silently worse bias^2.
    again = policy.values(list(observations))[:, 0]
    drift = float(np.abs(again - reference).max())
    if drift > tolerance:
        raise SystemExit(
            f"the checkpoint's value on the re-encoded positions differs from "
            f"the dump's recorded prediction by up to {drift:.3e}, over the "
            f"{tolerance:.0e} tolerance; the observation is not the one the "
            "dump was scored on"
        )

    return (
        HeldOut(
            observations=observations,
            progress=tuple(round(progress, 3) for _, progress in chosen),
            seats=tuple(snapshot.seat for snapshot, _ in chosen),
            returns=tuple(
                np.asarray(row["returns"], dtype=np.float64) for row in wanted
            ),
            reference=tuple(float(value) for value in reference),
        ),
        loaded,
    )


def score_held_out(held: HeldOut, own: Sequence[float], bins: int) -> dict:
    """One head's floor decomposition over the dump, pooled and by stage.

    `own` is the head's own-payoff prediction -- column 0, the component
    `catan.ppo.rotate` puts the acting seat in and the component the dump's
    returns are measured in. `floor.split` and `floor.pool` do the arithmetic
    unchanged, so this number and the A1 floor report's are the same quantity.

    Rows carry `seat` for the same reason the dump does: the rotation is the
    documented trap here, and a row that names the seat its return and its
    prediction both belong to can be audited against the dump position by
    position rather than taken on the docstring's word.
    """
    if len(own) != len(held.returns):
        raise ValueError(
            f"{len(own)} predictions for {len(held.returns)} held-out positions"
        )
    scored = []
    for progress, seat, returns, prediction in zip(
        held.progress, held.seats, held.returns, own
    ):
        floor, bias = split_error(returns, float(prediction))
        scored.append(
            {
                "progress": progress,
                "seat": seat,
                "rollouts": int(returns.size),
                "floor": floor,
                "bias_squared": bias,
                "mse": floor + bias,
            }
        )
    return {**pool(scored), "stages": _stages(scored, bins), "rows": scored}


def gate(mse_bias: float, quantile_bias: float, threshold: float) -> dict:
    """The registered pass line: the quantile head cuts held-out bias^2 by >=20%.

    A ratio, not a difference, and against the MSE arm rather than against the
    checkpoint's own head: the register's line is a paired comparison between
    two heads fit on one dataset, which is what makes it immune to the
    absolute-threshold problem that Gate A1's W1 line ran into.
    """
    reduction = 1.0 - quantile_bias / mse_bias if mse_bias > 0 else float("nan")
    return {
        "mse_bias_squared": round(mse_bias, 6),
        "quantile_bias_squared": round(quantile_bias, 6),
        "bias_squared_reduction": round(reduction, 4),
        "threshold": threshold,
        # Stated as the register states it -- the quantile arm's bias^2 is at
        # most `1 - threshold` of the MSE arm's -- rather than as
        # `reduction >= threshold`, whose float rounding decides a dead-on 20%
        # cut the wrong way.
        "pass": bool(quantile_bias <= (1.0 - threshold) * mse_bias),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/lam095/latest.pt")
    parser.add_argument(
        "--held-out-dump",
        required=True,
        help=(
            "a file written by `benchmarks.floor --dump-returns`. Its positions "
            "are the held-out set and its rollouts are what make a per-position "
            "bias^2 -- the gate's pass line -- computable at all."
        ),
    )
    # The seeding parameters of the floor run that wrote the dump. Only `seed`
    # is recorded in the dump itself, and the other three select which positions
    # its sampling kept, so a dump from a non-default floor run needs them here.
    parser.add_argument("--dump-seed-games", type=int, default=24)
    parser.add_argument("--dump-action-cap", type=int, default=4000)
    parser.add_argument("--prediction-tolerance", type=float, default=1e-5)

    parser.add_argument("--games", type=int, default=512, help="cohort size")
    parser.add_argument("--lanes", type=int, default=64)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--action-cap", type=int, default=4000)
    parser.add_argument(
        "--random-boards",
        action="store_true",
        help=(
            "deal the cohort a board per game instead of the dump's board. See "
            "the module docstring: the default matches the held-out board so a "
            "board-generalisation gap cannot be read as a head difference."
        ),
    )
    parser.add_argument("--holdout", type=float, default=0.2, help="fraction of games")
    # Not head_shape's 8. See the module docstring: at 8 the MSE arm is still
    # descending, the quantile arm is not, and the gate reads the difference.
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--plateau-window",
        type=int,
        default=5,
        help="epochs of training loss the reported plateau statistic looks back over",
    )
    parser.add_argument("--minibatch", type=int, default=1024)
    # head_shape's rate, and for its reason: PPO's 3e-4 was chosen for a whole
    # trunk under a clipped surrogate, which is not the relevant precedent for a
    # few thousand head parameters on a fixed target.
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--quantiles", type=int, default=32)
    parser.add_argument(
        "--huber-kappa",
        type=float,
        default=1.0 / 30.0,
        help=(
            "Huber width of the quantile loss, in reward units. Default is one "
            "lattice step of the return (a third of a VP); see "
            "`quantile_huber_loss` for why a large kappa would fit expectiles."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--block", type=int, default=32, help="games encoded at once")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    started = time.perf_counter()

    with open(args.held_out_dump) as handle:
        dump = json.load(handle)
    held, loaded = recover(
        dump,
        args.checkpoint,
        seed_games=args.dump_seed_games,
        players=args.players,
        action_cap=args.dump_action_cap,
        tolerance=args.prediction_tolerance,
    )
    recovered = time.perf_counter() - started

    cohort_seed = args.seed + COHORT_SEED_OFFSET
    board = random_base_board(random.Random(int(dump["seed"])))
    if not args.random_boards and cohort_seed == int(dump["seed"]) + 1:
        raise SystemExit(
            f"the cohort would be dealt at seed {cohort_seed} on the dump's own "
            "board, which is the seed the dump's positions were played at; pass "
            "--random-boards or a different --seed"
        )

    net = loaded.policy.net
    layout = loaded.policy.layout
    players = loaded.players
    width = net.config.width
    deep = net.config.value_head in _DEEP

    generator = torch.Generator().manual_seed(cohort_seed)
    cohort_policy = NetworkPolicy(
        net, loaded.space, layout, greedy=False, generator=generator
    )
    collector = Collector(
        cohort_policy,
        lanes=args.lanes,
        players=args.players,
        seed=cohort_seed,
        action_cap=args.action_cap,
        max_offers=loaded.max_offers,
        deal=args.games,
        board=None if args.random_boards else board,
    )
    episodes = collector.drain()
    collected = time.perf_counter() - started

    train_episodes, cohort_held_episodes = split_games(
        episodes, args.holdout, random.Random(args.seed + 1)
    )
    train = dataset_from(
        net, layout, train_episodes, device=device, chunk=args.chunk, block=args.block
    )
    cohort_held = dataset_from(
        net,
        layout,
        cohort_held_episodes,
        device=device,
        chunk=args.chunk,
        block=args.block,
    )
    held_features = trunk_features(
        net, layout, held.observations, device=device, chunk=args.chunk
    )
    encoded = time.perf_counter() - started

    # The leak the split exists to prevent, checked on the rows rather than
    # assumed from the episode lists: every row of a seat's trajectory carries
    # its game's terminal outcome, so one game on both sides would report a
    # training loss as a held-out one and neither arm would look wrong.
    shared = {tuple(row) for row in np.unique(train.games, axis=0)} & {
        tuple(row) for row in np.unique(cohort_held.games, axis=0)
    }
    if shared:
        raise SystemExit(
            f"{len(shared)} game(s) reached both sides of the cohort split, "
            f"first {sorted(shared)[0]}; the held-out loss would be a training loss"
        )

    value_in = int(train.features.shape[1])
    heads: dict[str, nn.Module] = {}
    # Seeded immediately before each build so the two heads' initialisations are
    # drawn from the same stream. They cannot be identical -- the shapes differ
    # by a factor of Q in the output -- so this is the closest thing available.
    torch.manual_seed(args.seed)
    heads["mse"] = MeanHead(value_in, players, width, deep=deep).to(device)
    torch.manual_seed(args.seed)
    heads["quantile"] = QuantileHead(
        value_in,
        players,
        width,
        deep=deep,
        quantiles=args.quantiles,
        kappa=args.huber_kappa,
    ).to(device)

    arms = {}
    for name, head in heads.items():
        history = fit(
            head,
            train,
            device=device,
            epochs=args.epochs,
            minibatch=args.minibatch,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        cohort_prediction = predict_mean(
            head, cohort_held.features, device=device, chunk=args.chunk
        )
        actual = cohort_held.targets.numpy()
        own = predict_mean(head, held_features, device=device, chunk=args.chunk)[:, 0]
        arms[name] = {
            "training_loss": history,
            "plateau": round(plateau(history, args.plateau_window), 5),
            "cohort_held_out": {
                "positions": int(cohort_prediction.shape[0]),
                "mean_squared_error": round(
                    float(((cohort_prediction - actual) ** 2).mean()), 5
                ),
                "rms_error": round(
                    float(
                        np.sqrt(((cohort_prediction[:, 0] - actual[:, 0]) ** 2).mean())
                    ),
                    4,
                ),
                "explained_variance": round(
                    explained(cohort_prediction[:, 0], actual[:, 0]), 4
                ),
            },
            "dump_held_out": score_held_out(held, own, args.bins),
        }
    # The checkpoint's own head on the same positions, from the prediction the
    # dump already recorded: the two refits are only interpretable beside the
    # head the run actually has.
    arms["reference"] = {
        "training_loss": [],
        "plateau": None,
        "cohort_held_out": None,
        "dump_held_out": score_held_out(held, held.reference, args.bins),
    }
    elapsed = time.perf_counter() - started

    # From the unrounded rows, not from `pool`'s five-decimal summary: a bias^2
    # near 0.008 is three significant figures once rounded, and the gate is a
    # ratio of two of them.
    exact_bias = {
        name: float(
            np.mean([row["bias_squared"] for row in arm["dump_held_out"]["rows"]])
        )
        for name, arm in arms.items()
    }

    payload = {
        "environment": environment(),
        "checkpoint": args.checkpoint,
        "iteration": loaded.iteration,
        "args": vars(args),
        "dump": {
            "path": args.held_out_dump,
            "seed": dump["seed"],
            "positions": len(held.returns),
            "rollouts_each": int(held.returns[0].size),
        },
        "cohort": {
            "games": len(episodes),
            "board": "random" if args.random_boards else "dump",
            "seed": cohort_seed,
            "train_games": len(train_episodes),
            "train_positions": len(train),
            "held_out_games": len(cohort_held_episodes),
            "held_out_positions": len(cohort_held),
            "value_in": value_in,
        },
        "seconds": {
            "recover": round(recovered, 1),
            "collect": round(collected - recovered, 1),
            "encode": round(encoded - collected, 1),
            "fit_and_score": round(elapsed - encoded, 1),
            "total": round(elapsed, 1),
        },
        "gate": gate(exact_bias["mse"], exact_bias["quantile"], args.threshold),
        "arms": arms,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    _report(payload)
    return 0


def _report(payload: dict) -> None:
    cohort = payload["cohort"]
    dump = payload["dump"]
    print(
        f"checkpoint {payload['checkpoint']} (iteration {payload['iteration']}), "
        f"{payload['seconds']['total']}s"
    )
    print(
        f"  cohort   {cohort['games']} games on the {cohort['board']} board -> "
        f"{cohort['train_positions']} train / {cohort['held_out_positions']} "
        f"held-out positions, split by game"
    )
    print(
        f"  held out {dump['positions']} rollout positions x "
        f"{dump['rollouts_each']} rollouts from {dump['path']}"
    )
    print("  on the held-out rollout positions, per head:")
    print(
        f"    {'head':<10}{'bias^2':>10}{'floor':>10}{'mse':>10}"
        f"{'floor share':>13}{'cohort mse':>12}"
    )
    for name in ("reference", "mse", "quantile"):
        arm = payload["arms"][name]
        scored = arm["dump_held_out"]
        cohort_mse = arm["cohort_held_out"]
        cell = "-" if cohort_mse is None else f"{cohort_mse['mean_squared_error']:.5f}"
        print(
            f"    {name:<10}{scored['mean_bias_squared']:>10.5f}"
            f"{scored['mean_floor']:>10.5f}{scored['mean_squared_error']:>10.5f}"
            f"{scored['irreducible_share']:>12.1%}{cell:>12}"
        )
    verdict = payload["gate"]
    window = payload["args"]["plateau_window"]
    print(
        "  training loss still falling over the last "
        f"{window} epochs:  "
        + ",  ".join(
            f"{name} {payload['arms'][name]['plateau']:+.2%}"
            for name in ("mse", "quantile")
        )
        + "   <- both must be flat for the verdict to be about the heads"
    )
    print(
        f"  bias^2 reduction, quantile vs mse: "
        f"{verdict['bias_squared_reduction']:+.1%}  "
        f"(pass line {verdict['threshold']:.0%})  ->  "
        f"{'PASS' if verdict['pass'] else 'FAIL'}"
    )
    print("  by stage of the game (bias^2):")
    stages = {
        name: {
            (stage["from"], stage["to"]): stage
            for stage in payload["arms"][name]["dump_held_out"]["stages"]
        }
        for name in ("reference", "mse", "quantile")
    }
    for key, stage in stages["mse"].items():
        print(
            f"    {key[0]:.2f}-{key[1]:.2f}  {stage['positions']:>4} pos"
            f"  reference {stages['reference'][key]['mean_bias_squared']:.5f}"
            f"  mse {stage['mean_bias_squared']:.5f}"
            f"  quantile {stages['quantile'][key]['mean_bias_squared']:.5f}"
        )


if __name__ == "__main__":
    sys.exit(main())
