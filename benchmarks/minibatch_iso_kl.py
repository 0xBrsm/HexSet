"""What learning rate makes a bigger minibatch travel the same distance?

The campaign's minibatch blocks (ppo7, ppo8) both moved `--minibatch` at a
fixed rate, and both were confounded by the same thing: at a fixed rate a
bigger minibatch travels *less far per iteration*, because the update takes
proportionally fewer steps. Measured in production, 4096 -> 16384 at 3e-4
halved it:

    ppo6 @4096 : 95 steps/iteration, approx_kl_last_epoch 0.00408, grad_norm 0.244
    ppo8 @16384: 23 steps/iteration, approx_kl_last_epoch 0.00203, grad_norm 0.142

So "minibatch 16384 read flat" cannot be told apart from "half a step read
flat". The block those runs were trying to run needs the rate re-calibrated at
every minibatch so that only the *direction* of the step changes, not its
length — and calibrating it with a training run is what the last five blocks
did wrong.

This is the cheap version: one collected batch, no training run, every arm
starting from identical weights and identical *warm* Adam state, the same
seeded minibatch order, so `minibatch` and `lr` are the only differences.

Per arm it reports the honest end-to-end distance — the k3 KL of the finished
policy against the batch's own `old_log_prob`, not the per-minibatch average
that understates a finished update by ~1.7x — plus the entropy the update left
behind and the median pre-clip gradient norm.

Then it solves, per minibatch, for the rate that matches the reference arm's
end-to-end KL. Those rates are the arms of the block; nothing here is a guess.

**Read the entropy column as carefully as the KL column.** ppo7 collapsed
because the entropy bonus is the most *coherent* term in a policy objective
whose gradient is 86% noise, so raising the rate amplifies it preferentially.
If an iso-KL arm also lands on the reference arm's entropy, the entropy
coefficient needs no adjustment and the block moves one thing. If it does not,
that is a second calibration — and a finding.

    python -m benchmarks.minibatch_iso_kl --checkpoint /w/runs/ppo6/latest.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch

from catan.actions import space_for
from catan.board.board import random_base_board
from catan.collect import ParallelCollector, WorkerSpec, parse_mix
from catan.encoding import static_graph
from catan.game import start
from catan.model import CatanNet, ModelConfig, config_from_args, packing
from catan.policy import NetworkPolicy
from catan.ppo import PPOConfig, _minibatches, assemble, minibatch_terms


# `torch.OutOfMemoryError` is only an alias from 2.5; fall back to the message.
_OOM_TYPES = tuple(
    t for t in (getattr(torch, "OutOfMemoryError", None),
                getattr(torch.cuda, "OutOfMemoryError", None)) if t is not None
)
_OOM = _OOM_TYPES + (RuntimeError,) if _OOM_TYPES else (RuntimeError,)


def _grid(spec: str) -> list[tuple[int, list[float]]]:
    """`4096:3e-4,6e-4|16384:6e-4,1.2e-3` -> [(4096, [...]), (16384, [...])].

    `full` as a minibatch means the whole batch in one step.
    """
    out = []
    for chunk in spec.split("|"):
        head, _, tail = chunk.partition(":")
        size = -1 if head.strip() == "full" else int(head)
        out.append((size, [float(x) for x in tail.split(",")]))
    return out


def iso_kl_rate(target: float, arms: list[dict]) -> dict | None:
    """The rate whose end-to-end KL matches `target`, bracketed from `arms`.

    Interpolated in log-log, which is exact for a power-law response and very
    nearly exact for a linear one. The measured KL response is a power law
    whose exponent is ~1 below 1.2e-3 and ~1.9 above it, so a linear read of a
    bracket straddling that knee mis-states the rate — by 4% on the recorded
    response, low rather than high. Returns `None` when the target is outside
    the swept range, which is a real answer about the grid rather than
    something to extrapolate through.
    """
    usable = sorted((a for a in arms if not a.get("oom")), key=lambda a: a["lr"])
    below = [a for a in usable if a["end_to_end_kl"] <= target]
    above = [a for a in usable if a["end_to_end_kl"] > target]
    if not below or not above:
        return None
    lo, hi = below[-1], above[0]
    span = math.log(hi["end_to_end_kl"]) - math.log(lo["end_to_end_kl"])
    frac = 0.0 if span == 0 else (math.log(target) - math.log(lo["end_to_end_kl"])) / span
    return {
        "lr": math.exp(math.log(lo["lr"]) + frac * (math.log(hi["lr"]) - math.log(lo["lr"]))),
        "entropy": lo["entropy"] + frac * (hi["entropy"] - lo["entropy"]),
        "bracket": [lo["lr"], hi["lr"]],
        "steps": lo["steps"],
    }


def worker_specs(
    args: argparse.Namespace,
    model: ModelConfig,
    mix: list[tuple[str, float]] | tuple[tuple[str, float], ...],
    parent: str,
) -> list[WorkerSpec]:
    """One spec per collect worker, carrying the checkpoint's *shape*.

    A worker builds its own net and then has the learner's parameters pushed
    into it, so head shapes left at the default here fail at the first sync
    rather than at construction -- a failure two processes away from its cause.
    """
    shard = max(1, -(-args.lanes // args.collect_workers))
    return [
        WorkerSpec(
            seed=args.seed, players=args.players, lanes=shard,
            action_cap=args.action_cap, max_offers=args.max_offers,
            first_game=worker, stride=args.collect_workers,
            width=model.width, rounds=model.rounds,
            value_head=model.value_head, policy_head=model.policy_head,
            torch_seed=args.seed + 100_000 + worker,
            mix=tuple(mix), parent=parent, cohort=True,
        )
        for worker in range(args.collect_workers)
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="/w/runs/ppo6/latest.pt")
    p.add_argument("--parent", default="", help="mix parent; default: the checkpoint's own")
    p.add_argument("--mix", default="greedy=0.15,parent=0.15")
    p.add_argument("--games", type=int, default=128)
    p.add_argument("--lanes", type=int, default=128)
    p.add_argument("--collect-workers", type=int, default=24)
    p.add_argument("--players", type=int, default=4)
    p.add_argument("--max-offers", type=int, default=3)
    p.add_argument("--action-cap", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument(
        "--grid",
        default="4096:3e-4,4.5e-4,6e-4"
        "|16384:3e-4,6e-4,9e-4,1.2e-3"
        "|32768:6e-4,1.2e-3,1.8e-3,2.4e-3"
        "|49152:9e-4,1.8e-3,2.7e-3,3.6e-3"
        "|full:1.2e-3,2.4e-3,3.6e-3,4.8e-3",
    )
    p.add_argument("--reference", default="4096:3e-4", help="the arm every other matches")
    p.add_argument("--json", default="/w/tmp/minibatch_iso_kl.json")
    args = p.parse_args(argv)

    torch.set_num_threads(args.threads)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stored = state.get("args", {})
    # Shape as well as size. Reading only width and rounds here is what stopped
    # this probe loading the first `--policy-head mlp` lineage at all, and the
    # whole point of the probe is to calibrate the lineage that will run.
    model = config_from_args(stored)
    parent = args.parent or stored.get("parent", "")

    # The launch configuration of the run being calibrated against, so the block
    # can be written against what the parent actually ran rather than prose.
    print("checkpoint args: " + json.dumps(
        {k: stored.get(k) for k in (
            "learning_rate", "minibatch", "epochs", "entropy", "lam",
            "games_per_iteration", "lanes", "collect_mode", "mix", "parent",
            "value_coefficient", "clip", "adam_eps", "max_grad_norm",
        )}, indent=1), flush=True)

    rng = random.Random(args.seed)
    board = random_base_board(rng)
    space = space_for(start(board, args.players, rng))
    graph = static_graph(board.topology)

    torch.manual_seed(args.seed)
    net = CatanNet(space, graph, args.players, model)
    net.load_state_dict(state["net"])
    net = net.to(args.device)
    # Gradient wiring, mirroring `catan.train`: the probe reproduces the run's
    # own update, and a detached value loss is a different update.
    net.detach_value = bool(stored.get("detach_value", False))
    policy = NetworkPolicy(net, space, packing(graph, args.players), device=args.device)

    mix = parse_mix(args.mix)
    collector = ParallelCollector(worker_specs(args, model, mix, parent))

    print(f"collecting a {args.games}-game cohort on {args.collect_workers} "
          f"workers (mix {args.mix or 'none'})...", flush=True)
    began = time.perf_counter()
    collector.sync(policy.net)
    episodes = collector.collect(args.games)
    collect_seconds = time.perf_counter() - began
    collector.close()

    base = PPOConfig(entropy_coefficient=args.entropy, epochs=args.epochs)
    batch = assemble(episodes, policy.layout, base).to(args.device)
    positions = len(batch)
    print(f"batch: {positions} positions in {collect_seconds:.1f}s\n", flush=True)

    start_net = copy.deepcopy(net.state_dict())
    start_opt = copy.deepcopy(state["optimiser"])

    def over_batch(chunk: int = 16384) -> tuple[float, float]:
        """End-to-end k3 KL against the batch's own old log-probs, and entropy.

        `torch.expm1` rather than `exp(x) - 1`: on a near-on-policy batch the
        log-ratios reach ~1e-9, where float32 `exp` rounds to exactly 1.0 and
        the estimator returns the negated drift instead of a KL. Nothing above
        ~1e-3 changes, but the guard costs nothing.
        """
        kls, ents, rows_seen = 0.0, 0.0, 0
        with torch.no_grad():
            for rows in _minibatches(positions, chunk, torch.Generator().manual_seed(7)):
                ev = policy.evaluate(batch.buffer[rows], batch.mask[rows],
                                     batch.pair[rows], batch.chosen[rows],
                                     batch.offer[rows])
                ratio = ev.log_prob - batch.log_prob[rows]
                kls += float((torch.expm1(ratio) - ratio).sum())
                ents += float(ev.entropy.sum())
                rows_seen += len(rows)
        return kls / rows_seen, ents / rows_seen

    net.load_state_dict(start_net)
    _, entropy_at_start = over_batch()
    print(f"entropy at start: {entropy_at_start:.4f}\n", flush=True)

    results = []
    for size, rates in _grid(args.grid):
        minibatch = positions if size < 0 else size
        label = "full" if size < 0 else str(size)
        for lr in rates:
            net.load_state_dict(start_net)
            optimiser = torch.optim.Adam(net.parameters(), lr=lr,
                                         eps=float(stored.get("adam_eps", 1e-5)))
            optimiser.load_state_dict(copy.deepcopy(start_opt))
            for group in optimiser.param_groups:
                group["lr"] = lr

            config = PPOConfig(entropy_coefficient=args.entropy,
                               epochs=args.epochs, minibatch=minibatch)
            norms, steps = [], 0
            began = time.perf_counter()
            try:
                generator = torch.Generator().manual_seed(args.seed + 1)
                for _ in range(config.epochs):
                    for rows in _minibatches(positions, config.minibatch, generator):
                        advantage = batch.advantage[rows]
                        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                        terms = minibatch_terms(
                            policy, batch.buffer[rows], batch.mask[rows],
                            batch.pair[rows], batch.chosen[rows], batch.offer[rows],
                            batch.log_prob[rows], advantage,
                            batch.value_target[rows], config,
                        )
                        optimiser.zero_grad(set_to_none=True)
                        terms.loss.backward()
                        norms.append(float(torch.nn.utils.clip_grad_norm_(
                            net.parameters(), config.max_grad_norm)))
                        optimiser.step()
                        steps += 1
            except _OOM as exc:
                # A rung that does not fit is a real answer about the ladder's
                # top, not a crash: record it and carry on.
                if not isinstance(exc, _OOM_TYPES) and "out of memory" not in str(exc):
                    raise
                print(f"  minibatch {label:>6s} lr {lr:<8.1e} OUT OF MEMORY", flush=True)
                results.append({"minibatch": label, "size": minibatch, "lr": lr,
                                "oom": True})
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                continue

            seconds = time.perf_counter() - began
            kl, entropy = over_batch()
            norms.sort()
            row = {
                "minibatch": label, "size": minibatch, "lr": lr, "steps": steps,
                "end_to_end_kl": kl,
                "entropy": entropy,
                "entropy_delta": entropy - entropy_at_start,
                "grad_norm_median": norms[len(norms) // 2],
                "update_seconds": seconds,
            }
            results.append(row)
            print(f"  minibatch {label:>6s} lr {lr:<8.1e} steps {steps:>4d} "
                  f"| END-TO-END kl {kl:.5f} | entropy {entropy:.4f} "
                  f"({row['entropy_delta']:+.4f}) | grad-norm med "
                  f"{row['grad_norm_median']:.3f} | {seconds:5.1f}s", flush=True)

    # Solve for the rate that matches the reference arm, per minibatch. KL is
    # close to linear in lr below ~1.2e-3 and superlinear above it, so
    # interpolate in log-log and say so rather than pretending it is linear.
    ref_size, _, ref_lr = args.reference.partition(":")
    ref = next((r for r in results if r["minibatch"] == ref_size
                and abs(r["lr"] - float(ref_lr)) < 1e-12 and not r.get("oom")), None)
    iso = []
    if ref is not None:
        target = ref["end_to_end_kl"]
        print(f"\nreference arm {args.reference}: end-to-end KL {target:.5f}, "
              f"entropy {ref['entropy']:.4f}, {ref['steps']} steps", flush=True)
        for size, _ in _grid(args.grid):
            label = "full" if size < 0 else str(size)
            solved = iso_kl_rate(target, [r for r in results if r["minibatch"] == label])
            if solved is None:
                iso.append({"minibatch": label, "iso_kl_lr": None,
                            "note": "target outside the swept range"})
                print(f"  {label:>6s}: target outside the swept range", flush=True)
                continue
            gap = solved["entropy"] - ref["entropy"]
            iso.append({"minibatch": label, "iso_kl_lr": solved["lr"],
                        "entropy_at_iso_kl": solved["entropy"],
                        "entropy_gap_vs_reference": gap,
                        "bracket": solved["bracket"], "steps": solved["steps"]})
            print(f"  {label:>6s}: lr {solved['lr']:.3e} "
                  f"(bracket {solved['bracket'][0]:.1e}-{solved['bracket'][1]:.1e}) "
                  f"| entropy {solved['entropy']:.4f} ({gap:+.4f} vs reference)", flush=True)

    out = {
        "checkpoint": args.checkpoint, "checkpoint_args": stored,
        "positions": positions, "collect_seconds": collect_seconds,
        "epochs": args.epochs, "entropy_coefficient": args.entropy,
        "mix": args.mix, "games": args.games,
        "entropy_at_start": entropy_at_start,
        "reference": args.reference, "arms": results, "iso_kl": iso,
    }
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
