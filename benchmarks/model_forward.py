"""What one self-play move actually costs, now that there is a network.

`benchmarks.throughput` measures the engine alone, which says little about a
training loop: under expert iteration a move is dominated by network forwards,
not by the simulator. This splits a move into its three parts — engine step,
`encode`, and a batched forward — so the next optimisation goes where the time
is rather than where it was guessed to be.

The batch figure is the one that matters. A search evaluates many leaves at
once, so per-position cost at batch 1 is a worst case nothing in the loop will
actually pay.
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
from catan.actions import legal_actions, space_for
from catan.board.board import random_base_board
from catan.encoding import encode, static_graph
from catan.game import start
from catan.model import CatanNet, ModelConfig, collate
from catan.play import step_randomly


@dataclass
class Result:
    device: str
    torch_version: str
    players: int
    width: int
    rounds: int
    parameters: int
    batch: int
    engine_step_us: float
    encode_us: float
    legal_actions_us: float
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
    """Distinct mid-game positions, so the batch is not one state repeated."""
    out = []
    for i in range(count):
        rng = random.Random(seed + i)
        game = start(random_base_board(rng), players, rng)
        for _ in range(60 + i % 40):
            step_randomly(game, rng)
        out.append(game)
    return out


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def run(
    batch: int, players: int, seed: int, device: str, config: ModelConfig, repeats: int
) -> Result:
    games = _positions(batch, players, seed)
    space = space_for(games[0])
    graph = static_graph(games[0].state.board.topology)

    torch.manual_seed(seed)
    net = CatanNet(space, graph, players, config).to(device).eval()

    observations = [encode(game) for game in games]
    batched = [t.to(device) for t in collate(observations)]
    single = [t[:1] for t in batched]

    rng = random.Random(seed)
    stepping = start(random_base_board(rng), players, rng)
    for _ in range(60):
        step_randomly(stepping, rng)

    with torch.no_grad():
        forward_one = _timed(lambda: net(*single), repeats, device)
        forward_many = _timed(lambda: net(*batched), repeats, device)

    return Result(
        device=device,
        torch_version=torch.__version__,
        players=players,
        width=config.width,
        rounds=config.rounds,
        parameters=sum(p.numel() for p in net.parameters()),
        batch=batch,
        engine_step_us=_timed(lambda: step_randomly(stepping, rng), repeats, "cpu"),
        encode_us=_timed(lambda: encode(games[0]), repeats, "cpu"),
        legal_actions_us=_timed(lambda: legal_actions(games[0]), repeats, "cpu"),
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
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = run(
        batch=args.batch,
        players=args.players,
        seed=args.seed,
        device=resolve_device(args.device),
        config=ModelConfig(width=args.width, rounds=args.rounds),
        repeats=args.repeats,
    )
    payload = {"environment": environment(), **asdict(result)}

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    env = payload["environment"]
    print(f"commit {env['commit']}  dirty {env['dirty']}  {env['machine']}")
    print(f"torch {result.torch_version} on {result.device}")
    print(
        f"width {result.width}, {result.rounds} rounds, "
        f"{result.parameters:,} parameters, {result.players} players"
    )
    print(f"  engine step        {result.engine_step_us:>9} us")
    print(f"  encode             {result.encode_us:>9} us")
    print(f"  legal_actions      {result.legal_actions_us:>9} us")
    print(f"  forward, batch 1   {result.forward_batch1_us:>9} us")
    print(
        f"  forward, batch {result.batch:<3} {result.forward_batched_us_per_position:>9} us"
        f" per position  ({result.positions_per_second:,}/sec)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
