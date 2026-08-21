"""Whether the value head's blindness to siblings is a readout shape.

`benchmarks.value_head` says the head is globally calibrated — explained
variance +0.474 on its own positions, +0.453 off-policy, flat across stages.
`benchmarks.sibling` says it cannot tell two positions one action apart from
each other: the spread of its value across a position's legal children sits
about 17x *below* its own RMS error, so PUCT ranks children on noise and does
it confidently.

The architectural suspicion is the readout, and it is specific. `self.value` is
`nn.Linear(width, players)` reading the global token alone, and the global
token's only view of the board is `h.mean(1), v.mean(1), e.mean(1)`. One
settlement moved between two siblings is a 1/54th change in one of those means,
put through one linear layer. A head that cannot resolve that is not obviously
a head that was trained badly.

## Refit, don't retrain

Answering this properly is a fresh PPO run per shape, which is days. This is
the cheap form of the same question: freeze the trunk the run already produced,
throw away the head, fit a new one of a different shape on the same value
targets, and hand the result to `benchmarks.sibling`. What that can show is
whether the *information* is present in the trunk and merely unreadable by a
linear map on `g` — because a frozen trunk is a fixed feature extractor, and if
a richer readout of those same features resolves siblings then the trunk was
never the problem. What it cannot show is what a shape would do if the trunk
were trained alongside it. A negative here is therefore weaker than a positive:
it rules the shape out as a cheap fix, not as a fix.

## The trunk is frozen, and that is checked rather than intended

A silently-unfrozen trunk invalidates the whole experiment and produces no
error — it produces a better number. So there are two independent checks, both
fatal: every parameter outside the head has `requires_grad` false before the
optimiser is built, and every trunk tensor in `state_dict` — buffers included —
is compared bit-for-bit against a clone taken before the fit.

Freezing also makes the fit cheap for free. With no trunk parameter requiring
grad and no grad on the inputs, autograd never builds a graph through the
trunk, so backward runs over the head alone. Nothing here asks for that; it
falls out of `requires_grad`.

## The head is always fit from a fresh initialisation

Including `--shape linear`, which is therefore the control and the run to do
first. It is the same shape the checkpoint already has, so whatever it gains
over the `reference` figure below is what *refitting on this corpus* is worth,
and only the excess over `linear` belongs to the shape. Warm-starting the
matching shape would have made that unreadable.

`reference` is the checkpoint's own head, unmodified, scored on the same
held-out rows in the same run — so no number here rests on a cross-run
assumption, the property `benchmarks.sibling` was also built for.

## Held out by game, never by position

Every row of a seat's trajectory carries the same target: the terminal outcome
rotated into that seat's frame. A row-wise split therefore puts the answer to
the held-out rows in the training set — same game, same seat, same number —
and reports a held-out loss that is a training loss. The split is over
episodes.

The targets themselves are `catan.rewards.reward` rotated by `catan.ppo.rotate`,
the same two calls PPO's `assemble` makes, reused rather than rewritten because
the frame is a documented past bug: `rotate` puts the seat itself in slot 0 to
match what the encoder fed the network, and getting it backwards trains
without complaint and plays nonsense.

## This does not measure the thing we care about

It measures value loss, which is the quantity already known to be a poor guide
here — a head can fit the outcome well and still be flat across siblings, which
is exactly the situation being investigated. The measurement is:

    python -m benchmarks.head_shape --checkpoint runs/ppo4/latest.pt \\
        --corpus tmp/valued.corpus --shape mlp --out runs/heads/mlp.pt --json
    python -m benchmarks.sibling --checkpoint runs/heads/mlp.pt --games 64

The checkpoint written carries `width`, `rounds`, `players` and `value_head` in
its `args`, which is what `catan.netbot.load` rebuilds the config from, and the
run ends by loading its own output through that path so a shape mismatch is
caught here rather than in the benchmark that was supposed to answer the
question.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from benchmarks.throughput import environment

# Torch is imported inside `main`, as in `benchmarks.sibling`: `split` and
# `rows` are the parts with arithmetic worth getting wrong, and they stay
# importable and testable on a machine with no torch.

# The attribute `CatanNet` keeps its value readout on. A tuple rather than a
# constant because the head is another agent's module and only the name it
# emits through — `Prediction.value` — is contractual; a head that moved to
# `value_head` would otherwise be silently treated as trunk and frozen, which
# is a fit that trains nothing and reports a loss that never moves.
HEAD_ATTRIBUTES = ("value", "value_head")


def split(
    episodes: Sequence, holdout: float, rng: random.Random
) -> tuple[list, list]:
    """Episodes divided into fit and held-out, by game and never by position.

    See the module docstring: the target is terminal, so every row of a seat's
    trajectory is the same number and a row-wise split leaks it.
    """
    if not 0.0 < holdout < 1.0:
        raise ValueError(f"--holdout must be between 0 and 1, got {holdout}")
    if len(episodes) < 2:
        raise ValueError(
            f"need at least two games to hold one out, corpus has {len(episodes)}"
        )
    order = list(range(len(episodes)))
    rng.shuffle(order)
    # At least one game on each side, so a small corpus fails on its size rather
    # than by reporting a held-out loss over nothing.
    cut = min(max(1, round(len(order) * holdout)), len(order) - 1)
    held = set(order[:cut])
    return (
        [episode for index, episode in enumerate(episodes) if index not in held],
        [episode for index, episode in enumerate(episodes) if index in held],
    )


def rows(episodes: Sequence) -> tuple[list, np.ndarray]:
    """Every decision in these games, beside what the game turned out to be worth.

    One target per seat per game, repeated across that seat's whole trajectory,
    and all `players` components of it — the head predicts the table and the
    search's max^n backup reads every column, so training column 0 alone would
    leave three of the four outputs unsupervised.
    """
    from catan.ppo import rotate
    from catan.rewards import reward

    observations: list = []
    values: list[np.ndarray] = []
    for episode in episodes:
        payoff = reward(episode.outcome)
        for seat, trajectory in enumerate(episode.trajectories):
            target = np.asarray(rotate(payoff, seat), dtype=np.float32)
            for transition in trajectory:
                observations.append(transition.observation)
                values.append(target)
    if not observations:
        raise ValueError("no transitions in these episodes")
    return observations, np.stack(values)


def load_corpus(path: str) -> list:
    """The pickled `list[Episode]` at `path`, checked for being one."""
    with open(path, "rb") as handle:
        episodes = pickle.load(handle)
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{path} is not a non-empty list of episodes")
    first = episodes[0]
    if not hasattr(first, "trajectories") or not hasattr(first, "outcome"):
        raise ValueError(
            f"{path} holds {type(first).__name__}, not catan.selfplay.Episode"
        )
    return episodes


def configure(width: int, rounds: int, shape: str):
    """`ModelConfig` for one head shape, or a legible failure if it predates it."""
    from dataclasses import fields

    from catan.model import ModelConfig

    if "value_head" not in {field.name for field in fields(ModelConfig)}:
        raise SystemExit(
            "catan.model.ModelConfig has no `value_head` field, so there is no "
            "head shape to select; this benchmark needs that interface."
        )
    return ModelConfig(width=width, rounds=rounds, value_head=shape)


def head_of(net) -> tuple[str, object]:
    """The value readout's attribute name and module, so the rest can key on it."""
    import torch

    for name in HEAD_ATTRIBUTES:
        module = getattr(net, name, None)
        if isinstance(module, torch.nn.Module):
            if next(module.parameters(), None) is None:
                raise AssertionError(f"the value head at `{name}` has no parameters")
            return name, module
        if isinstance(module, torch.nn.Parameter):
            raise AssertionError(f"`{name}` is a bare Parameter, not a module")
    raise AssertionError(
        f"no value head found on CatanNet under any of {HEAD_ATTRIBUTES}"
    )


def freeze(net, prefix: str) -> list:
    """Turn off grad everywhere but the head, and refuse to continue if it stuck.

    The assertion is the point of the function. Freezing that half-worked is
    the failure this whole module is exposed to and it reports as a better
    number rather than as an error.
    """
    head = []
    for name, parameter in net.named_parameters():
        trainable = name.startswith(prefix + ".")
        parameter.requires_grad_(trainable)
        if trainable:
            head.append(parameter)
    if not head:
        raise AssertionError(f"the head at `{prefix}` has no parameters to fit")

    grad = sorted(name for name, p in net.named_parameters() if p.requires_grad)
    wanted = sorted(
        name for name, _ in net.named_parameters() if name.startswith(prefix + ".")
    )
    if grad != wanted:
        raise AssertionError(
            f"trunk parameters still require grad: {sorted(set(grad) - set(wanted))}"
        )
    return head


def fingerprint(net, prefix: str) -> dict:
    """A clone of every trunk tensor, buffers included, to compare against later."""
    return {
        name: tensor.detach().clone()
        for name, tensor in net.state_dict().items()
        if not name.startswith(prefix + ".")
    }


def assert_unchanged(net, prefix: str, before: dict) -> None:
    import torch

    after = net.state_dict()
    moved = [
        name
        for name, tensor in before.items()
        if not torch.equal(tensor, after[name].detach())
    ]
    if moved:
        raise AssertionError(
            "the trunk moved during the fit, which invalidates the run: "
            f"{moved[:8]}{' ...' if len(moved) > 8 else ''}"
        )


def predict(net, layout, buffer, device, chunk: int) -> np.ndarray:
    """The head's `(N, players)` output over a whole split, in chunks."""
    import torch

    from catan.model import unpack

    out = []
    with torch.no_grad():
        for start in range(0, buffer.shape[0], chunk):
            block = buffer[start : start + chunk].to(device)
            out.append(net(*unpack(layout, block)).value.detach().cpu().numpy())
    return np.concatenate(out)


def report(predicted: np.ndarray, actual: np.ndarray) -> dict:
    """Loss over every seat, explained variance over the acting seat's column.

    Two different reductions on purpose. The loss is the quantity being
    minimised and is a mean over all `players` outputs, exactly as
    `catan.ppo.update` computes it, so the numbers here are comparable with a
    training log's `value_loss`. Explained variance is reported on column 0
    alone because that is what `benchmarks.value_head` and the run log both
    mean by it, and a figure pooled over the other three seats would not be
    comparable with either.
    """
    from benchmarks.value_head import explained

    return {
        "loss": round(float(((predicted - actual) ** 2).mean()), 5),
        "rms_error": round(float(np.sqrt(((predicted[:, 0] - actual[:, 0]) ** 2).mean())), 4),
        "explained_variance": round(explained(predicted[:, 0], actual[:, 0]), 4),
    }


def fit(
    net,
    layout,
    parameters,
    buffer,
    target,
    *,
    device,
    epochs: int,
    minibatch: int,
    learning_rate: float,
    generator,
) -> list[float]:
    """Fit the head alone; returns the mean training loss of each epoch.

    Adam over the head's parameters only — not over `net.parameters()` with the
    trunk frozen, which would work but would leave the freeze as the single
    thing standing between this and a trunk update.
    """
    import torch

    from catan.model import unpack

    optimiser = torch.optim.Adam(parameters, lr=learning_rate)
    size = buffer.shape[0]
    history = []
    for _ in range(epochs):
        order = torch.randperm(size, generator=generator)
        total = 0.0
        for start in range(0, size, minibatch):
            index = order[start : start + minibatch]
            rows_ = buffer.index_select(0, index).to(device)
            wanted = target.index_select(0, index).to(device)

            prediction = net(*unpack(layout, rows_))
            loss = (prediction.value - wanted).pow(2).mean()

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            # Weighted by rows, so a short final minibatch does not count as
            # much as a full one in the epoch's mean.
            total += float(loss) * index.numel()
        history.append(round(total / size, 5))
    return history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="the trunk to freeze")
    parser.add_argument(
        "--corpus",
        required=True,
        help="pickled list[Episode] to take value targets from",
    )
    parser.add_argument(
        "--shape",
        default="linear",
        help=(
            "the ModelConfig.value_head shape to fit — linear, mlp, pooled, "
            "mlp_pooled, attn. `linear` is the control: same shape as the "
            "checkpoint, so it prices the refit rather than the shape."
        ),
    )
    parser.add_argument("--out", required=True, help="where to write the refit checkpoint")
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument(
        "--games",
        type=int,
        default=0,
        help="cap on games read from the corpus; 0 is all of it. The packed "
        "buffer is the memory here, at roughly 5 KB a decision.",
    )
    parser.add_argument("--holdout", type=float, default=0.2, help="fraction of games")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--minibatch", type=int, default=1024)
    # Higher than PPO's 3e-4, which was chosen for a whole trunk under a clipped
    # surrogate and is not the relevant precedent for a few thousand parameters
    # fit on a fixed target.
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    import torch

    from catan.actions import build_space
    from catan.board.board import random_base_board
    from catan.encoding import static_graph
    from catan.model import CatanNet, pack, packing
    from catan.train import save

    device = torch.device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trained = state["net"]
    source = dict(state.get("args", {}))
    width = int(source.get("width", 64))
    rounds = int(source.get("rounds", 2))
    players = int(source.get("players", 4))

    episodes = load_corpus(args.corpus)
    if args.games:
        episodes = episodes[: args.games]
    train_episodes, held_episodes = split(
        episodes, args.holdout, random.Random(args.seed + 1)
    )

    board = random_base_board(random.Random(args.seed))
    topology = board.topology
    graph = static_graph(topology)
    space = build_space(
        topology.num_vertices, topology.num_edges, topology.num_hexes, players
    )
    layout = packing(graph, players)

    torch.manual_seed(args.seed)
    net = CatanNet(space, graph, players, configure(width, rounds, args.shape))
    prefix, _ = head_of(net)

    # The head's trained weights are deliberately dropped rather than loaded:
    # every shape starts from a fresh initialisation, including the one that
    # matches, so `--shape linear` prices the refit and the others price the
    # shape against it.
    trunk_state = {
        name: tensor
        for name, tensor in trained.items()
        if not name.startswith(prefix + ".")
    }
    missing, unexpected = net.load_state_dict(trunk_state, strict=False)
    stray = [name for name in missing if not name.startswith(prefix + ".")]
    if stray or unexpected:
        print(
            "the checkpoint's trunk does not match this model: "
            f"missing {stray}, unexpected {list(unexpected)}",
            file=sys.stderr,
        )
        return 1

    # `eval`, not `train`: there is no dropout or batch norm in this model, so
    # this changes no arithmetic — it pins the trunk to the same mode
    # `catan.netbot.load` will score the result under.
    net.to(device).eval()
    parameters = freeze(net, prefix)
    before = fingerprint(net, prefix)

    train_observations, train_target = rows(train_episodes)
    held_observations, held_target = rows(held_episodes)
    train_buffer = pack(layout, train_observations)
    held_buffer = pack(layout, held_observations)
    # The observations are the larger half of the memory and are now packed.
    del train_observations, held_observations

    started = time.perf_counter()
    history = fit(
        net,
        layout,
        parameters,
        train_buffer,
        torch.from_numpy(train_target),
        device=device,
        epochs=args.epochs,
        minibatch=args.minibatch,
        learning_rate=args.learning_rate,
        generator=torch.Generator().manual_seed(args.seed + 2),
    )
    elapsed = time.perf_counter() - started
    assert_unchanged(net, prefix, before)

    fitted = {
        "train": report(
            predict(net, layout, train_buffer, device, args.minibatch), train_target
        ),
        "holdout": report(
            predict(net, layout, held_buffer, device, args.minibatch), held_target
        ),
    }
    # The checkpoint's own head, unrefit, on the same held-out rows — the
    # control the two fitted figures are only interpretable against. It needs a
    # second network because `net` dropped those weights on purpose; the
    # checkpoint still holds them, so nothing was lost by fitting first.
    #
    # Not fatal if it cannot be built: the fit is the expensive part and it is
    # already done, so a checkpoint whose own head shape will not rebuild loses
    # its control figure rather than the run.
    try:
        untouched = CatanNet(
            space,
            graph,
            players,
            configure(width, rounds, str(source.get("value_head", "linear"))),
        )
        untouched.load_state_dict(trained)
        untouched.to(device).eval()
        fitted["reference"] = report(
            predict(untouched, layout, held_buffer, device, args.minibatch),
            held_target,
        )
    except Exception as error:  # noqa: BLE001 - reported, not raised
        fitted["reference"] = {"error": str(error)}

    head_parameters = sum(p.numel() for p in parameters)
    trunk_parameters = sum(
        p.numel()
        for name, p in net.named_parameters()
        if not name.startswith(prefix + ".")
    )

    out_args = dict(source)
    out_args.update(
        {
            "width": width,
            "rounds": rounds,
            "players": players,
            "value_head": args.shape,
        }
    )
    payload = {
        "environment": environment(),
        "checkpoint": args.checkpoint,
        "corpus": args.corpus,
        "out": args.out,
        "shape": args.shape,
        "iteration": int(state.get("iteration", 0)),
        "device": str(device),
        "games": {"fit": len(train_episodes), "holdout": len(held_episodes)},
        "positions": {
            "fit": int(train_buffer.shape[0]),
            "holdout": int(held_buffer.shape[0]),
        },
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seconds": round(elapsed, 1),
        "head_parameters": head_parameters,
        "trunk_parameters": trunk_parameters,
        "epoch_loss": history,
        **fitted,
    }

    save(
        Path(args.out),
        {
            "iteration": int(state.get("iteration", 0)),
            "net": {
                name: tensor.detach().cpu() for name, tensor in net.state_dict().items()
            },
            "args": out_args,
            "source_checkpoint": args.checkpoint,
            "head_shape": args.shape,
            "fit": payload,
        },
    )

    # The output exists to be handed to `benchmarks.sibling`, which reaches it
    # through `catan.netbot.load`. Loading it here turns a config that cannot be
    # rebuilt from `args` into a failure of the run that wrote it.
    from catan.netbot import load

    try:
        load(str(Path(args.out)), topology, str(device))
    except Exception as error:  # noqa: BLE001 - the message is the whole point
        print(
            f"wrote {args.out} but catan.netbot.load cannot read it back: {error}",
            file=sys.stderr,
        )
        return 1
    payload["loadable"] = True

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(
        f"{args.shape} head on the {args.checkpoint} trunk  |  "
        f"{payload['games']['fit']}+{payload['games']['holdout']} games, "
        f"{payload['positions']['fit']} fit / {payload['positions']['holdout']} "
        f"held-out positions, {payload['seconds']}s"
    )
    print(f"  head parameters {head_parameters} against a trunk of {trunk_parameters}")
    print(f"  epoch loss {history}")
    for name in ("reference", "train", "holdout"):
        line = fitted[name]
        if "error" in line:
            print(f"  {name:<10} unavailable: {line['error']}")
            continue
        print(
            f"  {name:<10} loss {line['loss']:.5f}"
            f"  RMS {line['rms_error']:.4f}"
            f"  EV {line['explained_variance']:+.4f}"
        )
    print(f"  wrote {args.out}; measure it with benchmarks.sibling --checkpoint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
