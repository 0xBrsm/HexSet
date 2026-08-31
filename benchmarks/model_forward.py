# SPDX-License-Identifier: GPL-3.0-only
"""What one self-play move actually costs, now that there is a network.

`benchmarks.throughput` measures the engine alone, which says little about a
training loop: under expert iteration a move is dominated by network forwards,
not by the simulator. This splits a move into its three parts — engine step,
`encode`, and a batched forward — so the next optimisation goes where the time
is rather than where it was guessed to be.

The batch figure is the one that matters. A search evaluates many leaves at
once, so per-position cost at batch 1 is a worst case nothing in the loop will
actually pay.

The crossings are timed separately from the forward. A forward on tensors that
are already resident says nothing about what a search pays, because a search
has to build the batch, push it over, and pull the logits back for every
expansion. Those three costs are `collate`, `host to device` and `device to
host`, and they are per batch, not per position.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass

import torch

from benchmarks.throughput import environment
from hexset.actions import legal_actions, space_for
from hexset.board.board import random_base_board
from hexset.encoding import encode, static_graph
from hexset.game import imagine, is_over, start
from hexset.model import HexNet, ModelConfig, collate, pack, packing
from hexset.play import step_randomly


@dataclass
class Result:
    device: str
    torch_version: str
    players: int
    width: int
    rounds: int
    parameters: int
    batch: int
    compiled: bool
    engine_step_us: float
    encode_us: float
    legal_actions_us: float
    collate_us: float
    host_to_device_us: float
    device_to_host_us: float
    pack_us: float
    host_to_device_packed_us: float
    forward_batch1_us: float
    forward_batched_us_per_position: float
    positions_per_second: float


def _timed(fn, repeats: int, device: str) -> float:
    """Median microseconds per call, after a warmup."""
    for _ in range(max(3, repeats // 10)):
        fn()
    if device != "cpu":
        torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        if device != "cpu":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e6)
    return round(statistics.median(samples), 1)


def _positions(count: int, players: int, seed: int):
    """Distinct positions from one playthrough, all sharing a board.

    A search evaluates many leaves of the same game, so a batch that shares a
    board is what the training loop will actually encode. Giving each position
    its own random board instead measures cache misses no real batch pays —
    doing that tripled the reported `encode` cost, which was an artifact of the
    benchmark rather than anything about the encoder.
    """
    rng = random.Random(seed)
    game = start(random_base_board(rng), players, rng)
    for _ in range(60):
        step_randomly(game, rng)

    out = []
    while len(out) < count:
        out.append(imagine(game, random.Random(seed + len(out))))
        for _ in range(3):
            if is_over(game):
                return out + [out[-1]] * (count - len(out))
            step_randomly(game, rng)
    return out


def _over(items):
    """Cycle through positions, so a per-call cost is not one lucky phase.

    `legal_actions` in particular ranges from a two-element list in
    `TRADE_RESPOND` to a full enumeration in the main phase, so timing it
    against a single fixed state measures whichever phase that state happened
    to be in rather than anything about the engine.
    """
    index = 0

    def take():
        nonlocal index
        item = items[index % len(items)]
        index += 1
        return item

    return take


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def run(
    batch: int,
    players: int,
    seed: int,
    device: str,
    config: ModelConfig,
    repeats: int,
    compile_mode: str = "none",
) -> Result:
    games = _positions(batch, players, seed)
    space = space_for(games[0])
    graph = static_graph(games[0].state.board.topology)

    torch.manual_seed(seed)
    net = HexNet(space, graph, players, config).to(device).eval()
    if compile_mode != "none":
        net = torch.compile(net, mode=compile_mode)

    observations = [encode(game) for game in games]
    host = collate(observations)
    batched = [t.to(device) for t in host]
    single = [t[:1] for t in batched]

    layout = packing(graph, players)
    packed = pack(layout, observations)

    rng = random.Random(seed)
    stepping = start(random_base_board(rng), players, rng)

    def one_step() -> None:
        nonlocal stepping
        if is_over(stepping):
            stepping = start(random_base_board(rng), players, rng)
        step_randomly(stepping, rng)

    next_game = _over(games)

    with torch.no_grad():
        forward_one = _timed(lambda: net(*single), repeats, device)
        forward_many = _timed(lambda: net(*batched), repeats, device)
        prediction = net(*batched)

    def read_back() -> None:
        prediction.logits.to("cpu")
        prediction.value.to("cpu")

    return Result(
        device=device,
        torch_version=torch.__version__,
        players=players,
        width=config.width,
        rounds=config.rounds,
        parameters=sum(p.numel() for p in net.parameters()),
        batch=batch,
        compiled=compile_mode != "none",
        engine_step_us=_timed(one_step, repeats, "cpu"),
        encode_us=_timed(lambda: encode(next_game()), repeats, "cpu"),
        legal_actions_us=_timed(lambda: legal_actions(next_game()), repeats, "cpu"),
        collate_us=_timed(lambda: collate(observations), repeats, "cpu"),
        host_to_device_us=_timed(lambda: [t.to(device) for t in host], repeats, device),
        device_to_host_us=_timed(read_back, repeats, device),
        pack_us=_timed(lambda: pack(layout, observations), repeats, "cpu"),
        host_to_device_packed_us=_timed(lambda: packed.to(device), repeats, device),
        forward_batch1_us=forward_one,
        forward_batched_us_per_position=round(forward_many / batch, 2),
        positions_per_second=round(batch / (forward_many / 1e6)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--compile",
        dest="compile_mode",
        default="none",
        choices=["none", "default", "reduce-overhead", "max-autotune"],
        help="reduce-overhead uses CUDA graphs, which is what a launch-bound model wants",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = run(
        batch=args.batch,
        players=args.players,
        seed=args.seed,
        device=resolve_device(args.device),
        config=ModelConfig(width=args.width, rounds=args.rounds),
        repeats=args.repeats,
        compile_mode=args.compile_mode,
    )
    payload = {"environment": environment(), **asdict(result)}

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    env = payload["environment"]
    print(f"commit {env['commit']}  dirty {env['dirty']}  {env['machine']}")
    compiled = f", compiled {args.compile_mode}" if result.compiled else ""
    print(f"torch {result.torch_version} on {result.device}{compiled}")
    print(
        f"width {result.width}, {result.rounds} rounds, "
        f"{result.parameters:,} parameters, {result.players} players"
    )
    print(f"  engine step        {result.engine_step_us:>9} us")
    print(f"  encode             {result.encode_us:>9} us")
    print(f"  legal_actions      {result.legal_actions_us:>9} us")
    print(f"  collate            {result.collate_us:>9} us  per batch of {result.batch}")
    print(f"  host to device     {result.host_to_device_us:>9} us  per batch")
    print(f"  device to host     {result.device_to_host_us:>9} us  per batch")
    print(f"  pack               {result.pack_us:>9} us  per batch")
    print(f"  host to device x1  {result.host_to_device_packed_us:>9} us  per batch, packed")
    print(f"  forward, batch 1   {result.forward_batch1_us:>9} us")
    print(
        f"  forward, batch {result.batch:<3} {result.forward_batched_us_per_position:>9} us"
        f" per position  ({result.positions_per_second:,}/sec)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
