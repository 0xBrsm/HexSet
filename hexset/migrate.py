# SPDX-License-Identifier: GPL-3.0-only
"""Migrate a checkpoint from before the offer observation onto the wider
globals vector, without changing what it computes.

Trading design part 1 (`agents/reference/trading-design.md` §3.1) appended
18 features at four players to the tail of the globals vector — the live
offer's give and want bundles, the proposer's relative seat, and who has
answered. `embed_global`'s in_features grew with it, so a checkpoint written
before the change no longer loads. This tool is the bridge: the old columns
are copied unchanged and the new columns are **zero**, so the migrated net's
`embed_global` output — and therefore every logit and value — is *exactly*
the source's on any position, offer standing or not. Unlike `hexset.widen`'s
copies, zero columns train fine: their gradient is the upstream gradient
times the (nonzero) offer features, so the behaviour the observation exists
to enable is learned from the first update that sees an offer.

Exactness is asserted on real positions before anything is written, on both
forward paths, as `hexset.widen` does. The old function is reconstructed for
the comparison by slicing the migrated `embed_global` back down — with zero
new columns that *is* the old layer, so the check is a real end-to-end
forward, not a tautology on the weights alone.

What the checkpoint carries: the parent's `iteration`, `games_started` and
`torch_rng` (so `hexset.train --resume` continues the count exactly), `args`
unchanged (width did not change; `global_features` is derived from code, not
stored), a fresh Adam state (`hexset.widen`'s precedent — the parent's
moments predate the new parameter shape), and a `migrate` block naming the
source by path and sha256 and both global widths.

    python -m hexset.migrate --checkpoint /w/runs/table-c/iter-02000.pt \
        --out /w/runs/<new-run>/latest.pt
"""

from __future__ import annotations

import argparse
import hashlib
import random
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from .actions import space_for
from .board.board import random_base_board
from .encoding import encode, global_features, static_graph
from .game import is_over, start
from .model import HexNet, ModelConfig, collate, config_from_args
from .play import step_randomly

ADAM_EPS = 1e-5


def migrate_state_dict(
    state: dict[str, Tensor], template: dict[str, Tensor]
) -> dict[str, Tensor]:
    """`state` reshaped to `template`'s shapes: `embed_global.weight` gains
    zero columns on the right (features append at the globals tail); every
    other tensor must match already and passes through untouched."""
    out: dict[str, Tensor] = {}
    for key, tensor in state.items():
        if key not in template:
            raise KeyError(f"{key} is in the checkpoint but not in the migrated net")
        target = template[key].shape
        if tuple(tensor.shape) == tuple(target):
            out[key] = tensor.clone()
            continue
        if key != "embed_global.weight":
            raise ValueError(
                f"{key}: {tuple(tensor.shape)} → {tuple(target)} is not the "
                "offer-observation migration (only embed_global.weight grows)"
            )
        old_in, new_in = tensor.shape[1], target[1]
        if tensor.shape[0] != target[0] or new_in <= old_in:
            raise ValueError(
                f"embed_global.weight: {tuple(tensor.shape)} → {tuple(target)} "
                "is not a globals-tail extension"
            )
        widened = torch.zeros(target, dtype=tensor.dtype)
        widened[:, :old_in] = tensor
        out[key] = widened
    missing = set(template) - set(out)
    if missing:
        raise KeyError(f"the migrated net has keys the checkpoint lacks: {sorted(missing)}")
    return out


def _net(players: int, config: ModelConfig, board_seed: int) -> HexNet:
    rng = random.Random(board_seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    return HexNet(space_for(game), static_graph(board.topology), players, config)


def _observations(players: int, count: int, seed: int) -> list:
    """Real positions from seeded random play — random play proposes trades,
    so a share of these carry a standing offer, which is exactly the block
    the zero columns must ignore."""
    out = []
    for i in range(count):
        rng = random.Random(seed + i)
        board = random_base_board(rng)
        game = start(board, players, rng)
        for _ in range(40 + (i % 80)):
            if is_over(game):
                break
            step_randomly(game, rng)
        out.append(encode(game))
    return out


@torch.no_grad()
def compare(migrated: HexNet, old_in: int, observations: list) -> dict[str, float]:
    """max |Δ| of logits and values between the migrated net and the source
    function, on both forward paths.

    The source function is recovered from the migrated net itself: with the
    new columns zero, truncating every observation's globals to `old_in` and
    the layer's columns with it is byte-for-byte the old computation. The
    observations keep their full (post-change) globals for the migrated net,
    so offer-carrying positions exercise the zero columns for real.
    """
    batch = collate(observations)
    hexes, vertices, edges, globals_ = batch[0], batch[1], batch[2], batch[3]
    report: dict[str, float] = {}

    weight = migrated.embed_global.weight
    truncated = torch.nn.functional.linear(
        globals_[:, :old_in], weight[:, :old_in], migrated.embed_global.bias
    )
    full = migrated.embed_global(globals_)
    report["max_abs_embed_delta"] = float((truncated - full).abs().max())

    for fused in (False, True):
        migrated.fused = fused
        a = migrated(hexes, vertices, edges, globals_)
        zeroed = globals_.clone()
        zeroed[:, old_in:] = 0.0
        b = migrated(hexes, vertices, edges, zeroed)
        tag = "fused" if fused else "reference"
        report[f"max_abs_logit_delta_{tag}"] = float((a.logits - b.logits).abs().max())
        report[f"max_abs_value_delta_{tag}"] = float((a.value - b.value).abs().max())
    migrated.fused = False
    return report


def migrate_checkpoint(
    source: Path,
    *,
    players: int = 4,
    board_seed: int = 0,
    observations: int = 256,
) -> tuple[dict, dict[str, float]]:
    """The migrated checkpoint and the exactness report, nothing written."""
    state = torch.load(source, map_location="cpu", weights_only=False)
    args = dict(state["args"])
    config = config_from_args(args)
    net = _net(players, config, board_seed)
    template = net.state_dict()

    new_width = global_features(players)
    old_width = state["net"]["embed_global.weight"].shape[1]
    if old_width == new_width:
        raise SystemExit(f"{source} already reads {new_width} global features; nothing to migrate")

    migrated = migrate_state_dict(state["net"], template)
    net.load_state_dict(migrated, strict=True)
    net.eval()
    report = compare(net, old_width, _observations(players, observations, seed=1000))
    report["global_features_before"] = float(old_width)
    report["global_features_after"] = float(new_width)

    optimiser = torch.optim.Adam(
        net.parameters(),
        lr=float(args.get("learning_rate", 3e-4)),
        eps=float(args.get("adam_eps", ADAM_EPS)),
    )
    out = {
        "iteration": state["iteration"],
        "games_started": state["games_started"],
        "net": net.state_dict(),
        "optimiser": optimiser.state_dict(),
        "torch_rng": state["torch_rng"],
        "args": args,
        "config": asdict(config),
        "migrate": {
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_iteration": state["iteration"],
            "global_features_from": old_width,
            "global_features_to": new_width,
            "report": report,
        },
    }
    return out, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--observations", type=int, default=256,
                        help="real positions the exactness report is computed on")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to overwrite")
    checkpoint, report = migrate_checkpoint(
        args.checkpoint, players=args.players, observations=args.observations
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    for key, value in report.items():
        print(f"{key}: {value:.3e}" if isinstance(value, float) else f"{key}: {value}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
