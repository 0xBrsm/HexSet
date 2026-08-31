# SPDX-License-Identifier: GPL-3.0-only
"""Widen a trained `HexNet` checkpoint without changing what it computes.

Net2WiderNet (Chen, Goodfellow & Shlens, 2016), applied to this trunk. Every
hidden unit at the old width `d` is copied to fill the new width `D`: unit `j`
of the wide net is a copy of unit `j % d`. A copy keeps its source's *incoming*
weights, so it produces the same pre-activation and — through any elementwise
activation, `SiLU` here — the same activation. Every layer that *reads* the
hidden vector divides its weight on each copy by the number of copies of that
source, so the sum over the wide vector equals the sum over the narrow one.
Residual adds, neighbour means (`_pass`, the `A_*` GEMMs), the global token's
`h.mean(1)` and the max-pool of the pooled heads all act per dimension, so the
widened net emits the same logits and values to float precision. `widen_state`
checks that on real observations before anything is written, on both the
reference and the fused forward.

Why the copies get noise. Two exactly identical units receive exactly
identical gradients and stay identical for the rest of training: a wide net
built at `--noise 0` is a narrow net with a bigger bill, and a capacity probe
run on it would measure nothing. `--noise σ` adds Gaussian noise of `σ` times
each copied row's RMS to the copies' incoming weights — the sources are left
untouched — which separates the copies while moving the function by an amount
the tool reports (max |Δ| of the outputs and the mean policy KL against the
source) so the registration can bound it.

What the checkpoint carries. The parent's `iteration`, `games_started` and
`torch_rng`, so `hexset.train --resume` continues the count and the game index
exactly as a same-width continuation would; `args` with the new `width`, which
is what `hexset.netbot.load` and every benchmark rebuild the shape from; a
*fresh* Adam state over the new parameters (the parent's moments have the old
shapes and no meaning for the copies); and a `widen` block naming the source
by path and sha256, both widths, the noise and the seed.

    python -m hexset.widen --checkpoint /w/runs/lam095/latest.pt --width 128 \
        --noise 0.01 --seed 0 --out /w/runs/cap-w128/latest.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from .actions import space_for
from .board.board import random_base_board
from .encoding import encode, static_graph
from .game import is_over, start
from .model import HexNet, ModelConfig, collate, config_from_args
from .play import step_randomly

ADAM_EPS = 1e-5


def _plan(old: int, new: int) -> tuple[Tensor, Tensor]:
    """`source[j]` is the old unit copied into new unit `j`; `copies[i]` how
    many new units copy old unit `i`. Unit `j < old` copies itself, so the
    first `old` units of the wide net *are* the narrow net."""
    if new < old:
        raise ValueError(f"cannot widen {old} down to {new}")
    source = torch.arange(new) % old
    copies = torch.bincount(source, minlength=old).to(torch.float32)
    return source, copies


def _widen_columns(weight: Tensor, old: int, new: int) -> Tensor:
    """Every input block of `old` columns becomes `new` columns that read the
    copies, each scaled by 1/copies so the block's contribution is unchanged."""
    source, copies = _plan(old, new)
    if weight.shape[1] % old:
        raise ValueError(f"in_features {weight.shape[1]} is not a multiple of width {old}")
    blocks = [
        weight[:, i : i + old].index_select(1, source) / copies.index_select(0, source)
        for i in range(0, weight.shape[1], old)
    ]
    return torch.cat(blocks, dim=1)


def _widen_rows(weight: Tensor, old: int, new: int, noise: float, generator: torch.Generator) -> Tensor:
    """Output units: copies of their source rows, with `noise` × the row's RMS
    of Gaussian noise on the copies (rows `>= old`) and none on the sources."""
    source, _ = _plan(old, new)
    out = weight.index_select(0, source).clone()
    if noise > 0.0 and new > old:
        copies = out[old:]
        rms = copies.pow(2).mean(dim=tuple(range(1, copies.dim())), keepdim=True).sqrt()
        out[old:] = copies + noise * rms * torch.randn(copies.shape, generator=generator)
    return out


def _widen_vector(bias: Tensor, old: int, new: int) -> Tensor:
    source, _ = _plan(old, new)
    return bias.index_select(0, source).clone()


def widen_state_dict(
    state: dict[str, Tensor],
    template: dict[str, Tensor],
    old: int,
    new: int,
    *,
    noise: float = 0.0,
    seed: int = 0,
) -> dict[str, Tensor]:
    """The narrow `state` as the wide shapes `template` has, function preserved.

    The template — a fresh `HexNet` at the new width with the same head
    shapes — is the oracle for *which* axes widen: an axis widens exactly
    when the template's shape differs from the narrow shape along it. That
    keeps this free of key patterns, so a head shape added later widens
    correctly as long as it is built from `nn.Linear` over trunk features.
    Buffers and every unchanged tensor pass through untouched.
    """
    generator = torch.Generator().manual_seed(seed)
    out: dict[str, Tensor] = {}
    for key, tensor in state.items():
        if key not in template:
            raise KeyError(f"{key} is in the checkpoint but not in a width-{new} net")
        target = template[key].shape
        if tuple(tensor.shape) == tuple(target):
            out[key] = tensor.clone()
            continue
        if tensor.dim() == 2:
            widened = tensor
            if target[1] != tensor.shape[1]:
                if target[1] * old != tensor.shape[1] * new:
                    raise ValueError(f"{key}: in_features {tensor.shape[1]} → {target[1]} is not a width change")
                widened = _widen_columns(widened, old, new)
            if target[0] != tensor.shape[0]:
                if (tensor.shape[0], target[0]) != (old, new):
                    raise ValueError(f"{key}: out_features {tensor.shape[0]} → {target[0]} is not the trunk width")
                widened = _widen_rows(widened, old, new, noise, generator)
            out[key] = widened
        elif tensor.dim() == 1:
            if (tensor.shape[0], target[0]) != (old, new):
                raise ValueError(f"{key}: length {tensor.shape[0]} → {target[0]} is not the trunk width")
            out[key] = _widen_vector(tensor, old, new)
        else:
            raise ValueError(f"{key}: cannot widen a rank-{tensor.dim()} tensor")
        if key.startswith("value_query."):
            # The attention head's score is `(v · q) / sqrt(width)`. Copies
            # multiply the dot product by their count and the divisor grows to
            # sqrt(new), so each query unit is rescaled by sqrt(new/old) over
            # its copy count to leave every score — and the softmax — as it was.
            source, copies = _plan(old, new)
            scale = (new / old) ** 0.5 / copies.index_select(0, source)
            out[key] = out[key] * scale.view(-1, *([1] * (out[key].dim() - 1)))
        if tuple(out[key].shape) != tuple(target):
            raise AssertionError(f"{key}: widened to {tuple(out[key].shape)}, template has {tuple(target)}")
    missing = set(template) - set(out)
    if missing:
        raise KeyError(f"the width-{new} net has keys the checkpoint lacks: {sorted(missing)}")
    return out


def _observations(players: int, count: int, seed: int) -> list:
    """Real positions from seeded random play, the same recipe the model
    tests use, so the exactness check sees legal masks and full hands."""
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
def compare(narrow: HexNet, wide: HexNet, observations: list) -> dict[str, float]:
    """max |Δ| of logits and values, and the mean masked policy KL, narrow ‖ wide.

    Run on both forward paths of the wide net, because the fused round reads
    the first trunk linear in width-sized column blocks and has to agree with
    the reference path on a widened weight as it does on a trained one.
    """
    batch = collate(observations)
    a = narrow(*batch)
    report: dict[str, float] = {}
    for fused in (False, True):
        wide.fused = fused
        b = wide(*batch)
        tag = "fused" if fused else "reference"
        report[f"max_abs_logit_delta_{tag}"] = float((a.logits - b.logits).abs().max())
        report[f"max_abs_value_delta_{tag}"] = float((a.value - b.value).abs().max())
        # Relative to the largest output, because float32 reassociation across
        # twice as many summands is ~1e-6 of the magnitude — the exactness gate
        # has to be read against that, not against an absolute number.
        report[f"max_rel_logit_delta_{tag}"] = report[f"max_abs_logit_delta_{tag}"] / max(
            float(a.logits.abs().max()), 1e-12
        )
        report[f"max_rel_value_delta_{tag}"] = report[f"max_abs_value_delta_{tag}"] / max(
            float(a.value.abs().max()), 1e-12
        )
        pa = torch.log_softmax(a.logits, -1)
        pb = torch.log_softmax(b.logits, -1)
        report[f"mean_policy_kl_{tag}"] = float((pa.exp() * (pa - pb)).sum(-1).mean())
    wide.fused = False
    return report


def _net(players: int, config: ModelConfig, board_seed: int) -> HexNet:
    rng = random.Random(board_seed)
    board = random_base_board(rng)
    game = start(board, players, rng)
    return HexNet(space_for(game), static_graph(board.topology), players, config)


def widen_checkpoint(
    source: Path,
    width: int,
    *,
    noise: float,
    seed: int,
    players: int = 4,
    board_seed: int = 0,
    observations: int = 256,
) -> tuple[dict, dict[str, float]]:
    """The widened checkpoint and the exactness report, nothing written."""
    state = torch.load(source, map_location="cpu", weights_only=False)
    args = dict(state["args"])
    old_config = config_from_args(args)
    if width < old_config.width:
        raise SystemExit(f"{source} is width {old_config.width}; cannot widen down to {width}")
    new_config = ModelConfig(
        width=width,
        rounds=old_config.rounds,
        value_head=old_config.value_head,
        policy_head=old_config.policy_head,
        quantiles=old_config.quantiles,
    )
    narrow = _net(players, old_config, board_seed)
    narrow.load_state_dict(state["net"])
    wide = _net(players, new_config, board_seed)
    widened = widen_state_dict(
        state["net"], wide.state_dict(), old_config.width, width, noise=noise, seed=seed
    )
    wide.load_state_dict(widened, strict=True)
    narrow.eval()
    wide.eval()
    report = compare(narrow, wide, _observations(players, observations, seed=1000))
    report["parameters_before"] = float(sum(p.numel() for p in narrow.parameters()))
    report["parameters_after"] = float(sum(p.numel() for p in wide.parameters()))

    optimiser = torch.optim.Adam(
        wide.parameters(),
        lr=float(args.get("learning_rate", 3e-4)),
        eps=float(args.get("adam_eps", ADAM_EPS)),
    )
    args["width"] = width
    out = {
        "iteration": state["iteration"],
        "games_started": state["games_started"],
        "net": wide.state_dict(),
        "optimiser": optimiser.state_dict(),
        "torch_rng": state["torch_rng"],
        "args": args,
        "config": asdict(new_config),
        "widen": {
            "source": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_iteration": state["iteration"],
            "width_from": old_config.width,
            "width_to": width,
            "noise": noise,
            "seed": seed,
            "report": report,
        },
    }
    return out, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--noise", type=float, default=0.01,
                        help="Gaussian noise on the copies' incoming weights, as a "
                             "fraction of each row's RMS; 0 is exact and useless to train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--observations", type=int, default=256,
                        help="real positions the exactness report is computed on")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to overwrite")
    checkpoint, report = widen_checkpoint(
        args.checkpoint, args.width, noise=args.noise, seed=args.seed,
        players=args.players, observations=args.observations,
    )
    # The exact path has to be exact, whatever noise the caller asked for:
    # re-run at noise 0 so a wiring defect cannot hide behind the noise.
    if args.noise > 0.0:
        _, exact = widen_checkpoint(
            args.checkpoint, args.width, noise=0.0, seed=args.seed,
            players=args.players, observations=args.observations,
        )
        report = {**report, **{f"exact_{k}": v for k, v in exact.items() if "delta" in k or "kl" in k}}
        checkpoint["widen"]["report"] = report
    worst = max(
        v for k, v in report.items()
        if k.startswith("exact_max_rel") or (args.noise == 0.0 and k.startswith("max_rel"))
    )
    if worst > 1e-4:
        raise SystemExit(f"widening is not function-preserving at noise 0: max relative |Δ| = {worst:.3e}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    print(json.dumps({"out": str(args.out), **report}, indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
