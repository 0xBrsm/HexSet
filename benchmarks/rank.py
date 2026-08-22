"""Does the value head order siblings the way the truth does?

`benchmarks.sibling` measured the head's error against the spread it has to
resolve and found the error eleven times larger. That is an *upper bound on the
damage, not a measurement of it*, and its own write-up says why: sibling
positions are one action apart and mostly identical, so their biases are
probably correlated, and a bias common to both children cancels in the
comparison. `sibling` cannot see that cancellation, because it never learns any
child's true value.

This does. Every legal child of a probed position is rolled out many times, so
each one gets a Monte Carlo estimate of what it is actually worth, and the
head's ordering is compared against that. Rank correlation is the number the
open question wants.

**It answers the GAE question too, and that is not a coincidence.** With gamma
1 the low-lambda credit term is `V(s_next) - V(s)`, and the parent's value is a
constant across the row, so how well `V` orders the children *is* how much
information the one-step residual carries about the action taken. A head that
ranks siblings well makes low lambda viable and cuts the policy gradient's
variance at its source; a head that does not means the residual is its own
noise, and credit has to come from the outcome instead.

## Scoring the row with a tree, not just the head

`--simulations N` adds a second column. Every child is scored again by the
*backed-up value* of a PUCT search rooted at it, against the same truth on the
same positions, which asks the label question directly: the value head is
trained on terminal returns whose variance is mostly dice twenty turns out, and
the standing proposal is to bootstrap off the search's own estimate instead.
That is only worth doing if the tree's estimate is the better one, and two
columns over one row of truth is the paired way to find out.

The search draws from its own generator, so the truth column of a
`--simulations N` run is what a `--simulations 0` run at the same seed produces.
The comparison is paired by construction rather than by re-running.

## Common random numbers, because the differences are what is small

The true gap between the best two children averages 0.017 while a single
rollout's spread is about 0.19, so an unpaired estimate would need thousands of
rollouts a child to resolve an ordering. Every child of a position is therefore
rolled out from the same seed: lane `k` of one child gets the same deck shuffle
and the same action-sampling stream as lane `k` of its siblings, so whatever
luck they share cancels in the comparison. This is the trick the 400-game paired
duel already uses, applied one level down.

Pairing is claimed rather than assumed, so it is also measured: `paired_se` is
the standard error of the lane-matched difference between the top two children,
`unpaired_se` the same difference estimated as if the rollouts were independent.
Their ratio is what the pairing bought. **Read `resolved` before any conclusion**
— it is the share of positions where the measurement could tell the top two
children apart at all, and a low value means the answer is "more rollouts", not
"the head is fine".

    python -m benchmarks.rank --checkpoint runs/ppo4/iter-00585.pt \\
        --positions 24 --rollouts 96
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, replace

import numpy as np

from benchmarks.throughput import environment
from catan.actions import ActionType, apply, legal_actions, within_offer_budget
from catan.board.board import random_base_board
from catan.game import imagine, is_over, to_move
from catan.rewards import relative_points, reward
from catan.selfplay import Collector, Episode
from catan.victory import victory_points

# Torch is imported inside `main`, so everything below with arithmetic worth
# getting wrong stays importable and testable on a machine without it.


@dataclass(frozen=True)
class Fork:
    """A position kept for branching, and the seat whose value is at stake."""

    game: object
    seat: int


class Forking:
    """The policy, playing as usual, keeping a copy of the odd position.

    A copy, not the live lane object, which keeps moving. Positions are kept on
    the policy's own distribution — the one most favourable to the head — so a
    ranking that fails here fails everywhere.
    """

    def __init__(self, policy, *, max_offers, rate, rng) -> None:
        self.policy = policy
        self.max_offers = max_offers
        self.rate = rate
        self.rng = rng

    def act(self, requests):
        choices = self.policy.act(requests)
        for row, request in enumerate(requests):
            if self.rng.random() >= self.rate:
                continue
            if not probeable(request.game, self.max_offers):
                continue
            choices[row] = replace(
                choices[row],
                aux=Fork(
                    game=imagine(request.game, self.rng),
                    seat=request.seat,
                ),
            )
        return choices


def options(game, max_offers) -> tuple:
    """Everything legal here, within the offer budget the run trains under."""
    if is_over(game):
        return ()
    return tuple(within_offer_budget(game, legal_actions(game), max_offers))


def probeable(game, max_offers) -> tuple:
    """The same, but only where ranking children is a question worth asking.

    Roll positions are skipped: the spread across dice outcomes is chance, not
    something the head is being asked to rank. This filter belongs to the
    *parent* being forked from — a child whose own next action is a roll is a
    perfectly good leaf, and filtering it would hand `LeafEvaluator` an empty
    option list.
    """
    legal = options(game, max_offers)
    if len(legal) < 2 or any(a.type is ActionType.ROLL for a in legal):
        return ()
    return legal


def kept(episodes: list[Episode]) -> list[Fork]:
    return [
        transition.aux
        for episode in episodes
        for trajectory in episode.trajectories
        for transition in trajectory
        if isinstance(transition.aux, Fork)
    ]


def ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not manufacture an ordering."""
    order = values.argsort()
    out = np.empty(len(values), dtype=np.float64)
    out[order] = np.arange(len(values), dtype=np.float64)
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            out[tied] = out[tied].mean()
    return out


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson, returning 0.0 for a constant row rather than nan."""
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def backed_up(root, seat: int) -> float | None:
    """The root's visit-weighted mean over its edges, in board-order seats.

    `Search._backup` adds the whole per-seat vector into every edge on a path,
    so a column's sum over the visits is what the search believes about the
    position once its budget is spent — the tree's answer to the question one
    forward pass of the value head answers alone.

    `None` where there was nothing to search: a finished child, or one whose
    single legal action `run_many` short-circuits without backing anything up,
    or a root that never ran. The caller falls back to the head there. `None`
    rather than 0.0 on purpose — zero is an ordinary value on this scale and
    would pass silently into the correlation.
    """
    if root.terminal or len(root.options) < 2:
        return None
    visits = float(np.asarray(root.visits).sum())
    if visits <= 0:
        return None
    return float(np.asarray(root.totals)[:, seat].sum() / visits)


def assess(head: np.ndarray, true: np.ndarray) -> dict:
    """One position's row, scored the four ways that matter.

    `regret` is the decision-relevant one: how much true value is given up by
    taking the head's favourite instead of the best child, in reward units.
    Chance top-1 is `1/n`, so the hit rate is only meaningful against it.
    """
    best = int(true.argmax())
    picked = int(head.argmax())
    return {
        "children": int(len(head)),
        "spearman": correlation(ranks(head), ranks(true)),
        "pearson": correlation(head, true),
        "top1": bool(picked == best),
        "chance_top1": 1.0 / len(head),
        "regret": float(true[best] - true[picked]),
        "true_gap": float(np.sort(true)[::-1][0] - np.sort(true)[::-1][1]),
        "head_spread": float(head.std()),
        "true_spread": float(true.std()),
    }


def resolution(returns: list[np.ndarray], true: np.ndarray) -> dict:
    """How well this position's top two children were told apart.

    `paired` differences the lane-matched rollouts; `unpaired` adds the two
    standard errors in quadrature as if they were independent runs. The ratio is
    what common random numbers bought, measured rather than asserted.
    """
    ordered = np.argsort(true)[::-1]
    first, second = int(ordered[0]), int(ordered[1])
    a, b = returns[first], returns[second]
    width = min(len(a), len(b))
    difference = a[:width] - b[:width]
    paired = float(difference.std(ddof=1) / np.sqrt(width)) if width > 1 else float("nan")
    unpaired = float(
        np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    ) if min(len(a), len(b)) > 1 else float("nan")
    gap = float(true[first] - true[second])
    return {
        "paired_se": paired,
        "unpaired_se": unpaired,
        "pairing_gain": unpaired / paired if paired and paired == paired else float("nan"),
        "resolved": bool(paired == paired and gap > 1.96 * paired),
    }


def pooled(head_rows, true_rows, se_rows) -> dict:
    """Correlation over every child of every position, corrected for the noise
    in the truth itself.

    Common random numbers were tried first and bought a measured 1.1x: with a
    stochastic policy and hundreds of actions still to play, two rollouts from
    the same position decorrelate within a few plies whatever luck they were
    handed. So each child's true value carries real sampling error, and error in
    the *y* variable attenuates a correlation toward zero — it does not bias its
    sign. That is correctable when the error is known, and it is measured here.

        reliability = (Var(y_observed) - mean(se^2)) / Var(y_observed)
        r_true      = r_observed / sqrt(reliability)

    Each position is centred before pooling, because positions sit at different
    value levels and the question is only about ordering *within* a row. The
    corrected figure is the estimate; the raw one is what was seen. Report both,
    and report `reliability` beside them — below ~0.2 the correction is dividing
    by a small number and the interval is wide enough to be worthless.
    """
    x, y, variances = [], [], []
    for head, true, ses in zip(head_rows, true_rows, se_rows):
        n = len(head)
        if n < 2:
            continue
        x.append(np.asarray(head) - np.mean(head))
        y.append(np.asarray(true) - np.mean(true))
        # Centring a row of `n` shrinks independent errors by (n-1)/n, and the
        # observed variance below is measured *after* centring. Comparing it
        # against the raw se^2 drives reliability to zero and the correction to
        # nan — which is what the test caught.
        variances.append(np.asarray(ses) ** 2 * (n - 1) / n)
    if not x:
        return {"children": 0}
    x = np.concatenate(x)
    y = np.concatenate(y)
    noise = float(np.concatenate(variances).mean())
    observed = correlation(x, y)
    spread = float(y.var())
    reliability = max(0.0, (spread - noise) / spread) if spread > 0 else 0.0
    corrected = observed / np.sqrt(reliability) if reliability > 1e-6 else float("nan")
    return {
        "children": int(len(x)),
        "pearson_observed": observed,
        "true_variance_observed": spread,
        "noise_variance": noise,
        "reliability": reliability,
        "pearson_corrected": float(min(1.0, corrected)) if corrected == corrected else float("nan"),
    }


def standardised(head_rows, true_rows, se_rows) -> dict:
    """`pooled`, with every row scaled to unit variance before it is pooled.

    `pooled` centres a row but does not scale it, and Pearson weights by
    variance — so a position whose children differ ten times more than typical
    carries a hundred times the leverage. Measured over 240 positions: five of
    them held 47% of the total influence, and dropping a single position moved
    the pooled head-versus-tree difference by 0.083. That is why its interval
    stops tightening as positions are added.

    Report both, because they answer different questions. Spread-weighted is the
    right weighting for a *regression target*, whose loss is in absolute units
    and where a position with more at stake genuinely matters more.
    Row-standardised is the right one for a *decision rule*, which chooses once
    per position however much is at stake. A conclusion that holds under one and
    not the other is a statement about which positions carry it, and should be
    written up that way rather than as a single number.
    """
    xs, ys, reliabilities = [], [], []
    for head, true, ses in zip(head_rows, true_rows, se_rows):
        n = len(head)
        if n < 2:
            continue
        x = np.asarray(head, dtype=np.float64)
        y = np.asarray(true, dtype=np.float64)
        spread = float(y.var())
        noise = float((np.asarray(ses, dtype=np.float64) ** 2 * (n - 1) / n).mean())
        if spread <= 0 or x.std() < 1e-12:
            continue
        xs.append((x - x.mean()) / x.std())
        ys.append((y - y.mean()) / y.std())
        reliabilities.append(max(0.0, (spread - noise) / spread))
    if not xs:
        return {"positions": 0}
    observed = correlation(np.concatenate(xs), np.concatenate(ys))
    reliability = float(np.mean(reliabilities))
    corrected = observed / np.sqrt(reliability) if reliability > 1e-6 else float("nan")
    return {
        "positions": len(xs),
        "pearson_observed": observed,
        "reliability": reliability,
        "pearson_corrected": float(min(1.0, corrected))
        if corrected == corrected
        else float("nan"),
    }


def summarise(rows: list[dict]) -> dict:
    """The aggregate, with top-1 read against its own chance rate."""
    spearman = np.asarray([r["spearman"] for r in rows])
    regret = np.asarray([r["regret"] for r in rows])
    hits = np.asarray([1.0 if r["top1"] else 0.0 for r in rows])
    chance = np.asarray([r["chance_top1"] for r in rows])
    paired = np.asarray([r["paired_se"] for r in rows])
    unpaired = np.asarray([r["unpaired_se"] for r in rows])
    return {
        "positions": len(rows),
        "mean_children": float(np.mean([r["children"] for r in rows])),
        "spearman_mean": float(spearman.mean()),
        "spearman_sem": float(spearman.std(ddof=1) / np.sqrt(len(rows)))
        if len(rows) > 1
        else float("nan"),
        "pearson_mean": float(np.mean([r["pearson"] for r in rows])),
        "top1_rate": float(hits.mean()),
        "top1_chance": float(chance.mean()),
        "regret_mean": float(regret.mean()),
        "regret_mean_victory_points": float(regret.mean() * 10.0),
        "head_spread_mean": float(np.mean([r["head_spread"] for r in rows])),
        "true_spread_mean": float(np.mean([r["true_spread"] for r in rows])),
        "true_gap_mean": float(np.mean([r["true_gap"] for r in rows])),
        "paired_se_mean": float(np.nanmean(paired)),
        "unpaired_se_mean": float(np.nanmean(unpaired)),
        "pairing_gain_mean": float(np.nanmean(unpaired / paired)),
        "resolved": float(np.mean([1.0 if r["resolved"] else 0.0 for r in rows])),
    }


class PairedBranching(Collector):
    """Every lane replays the position on its *own* seeded random stream.

    `floor.Branching` hands one `rng` to every lane, so the lanes interleave
    their draws off a single stream. That is exactly right for `floor`, which
    only ever wanted one position's spread — and useless here, because it means
    lane `k` of child A and lane `k` of child B share no luck at all. Measured:
    a pairing gain of 1.1-1.2x, i.e. nothing.

    Seeding per lane index instead makes lane `k` start from the same deck and
    the same dice everywhere in the row, so the shared luck cancels in the
    sibling difference. Alignment decays as the paths diverge — the siblings are
    one action apart, so it decays slowly — and `pairing_gain` reports what
    actually survived rather than what was hoped for.
    """

    def __init__(self, policy, position, *, stream_seed: int, **kwargs) -> None:
        self._position = position
        self._stream_seed = stream_seed
        super().__init__(policy, **kwargs)

    def _fresh(self):
        lane = super()._fresh()
        if lane is None:
            return None
        lane.game = imagine(self._position, random.Random(self._stream_seed + lane.index))
        return lane


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-games", type=int, default=24)
    parser.add_argument("--positions", type=int, default=24)
    parser.add_argument("--rollouts", type=int, default=96)
    parser.add_argument("--max-children", type=int, default=8,
                        help="cap the row; wide positions cost linearly and the "
                             "ordering question is the same on a sample of them")
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--action-cap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fork-rate", type=float, default=0.02)
    parser.add_argument("--simulations", type=int, default=0,
                        help="also score every child by the backed-up value of a "
                             "PUCT search rooted at it; 0 scores the head alone")
    parser.add_argument("--wave", type=int, default=16)
    parser.add_argument("--exploration", type=float, default=1.25)
    # `netbot.load` pins torch to one thread, which is right for the 30-process
    # CPU sharding it was built for and crippling for this single process. The
    # rollouts batch every lane per tick, which is what the iGPU is for.
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    import torch

    from catan.mcts import Leaf, Search
    from catan.netbot import LeafEvaluator, load
    from catan.policy import NetworkPolicy

    board = random_base_board(random.Random(args.seed))
    loaded = load(args.checkpoint, board.topology, device=args.device)
    torch.set_num_threads(args.threads)  # after `load`, which sets it to 1
    # `torch.multinomial` wants a generator on the sampled tensor's device,
    # which only bites once the policy is not on the CPU.
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    policy = NetworkPolicy(
        loaded.policy.net,
        loaded.space,
        loaded.policy.layout,
        greedy=False,
        generator=generator,
        device=args.device,
    )
    evaluator = LeafEvaluator(policy=policy, space=loaded.space)
    # Its own generator, so the rollouts below draw what they draw at
    # `--simulations 0` and both columns are scored against one truth.
    search = (
        Search(
            evaluator,
            simulations=args.simulations,
            wave=args.wave,
            exploration=args.exploration,
            max_offers=loaded.max_offers,
            rng=random.Random(args.seed + 9001),
        )
        if args.simulations
        else None
    )

    started = time.perf_counter()
    rng = random.Random(args.seed + 2)
    seeding = Collector(
        Forking(policy, max_offers=loaded.max_offers, rate=args.fork_rate, rng=rng),
        lanes=min(16, args.seed_games),
        players=args.players,
        seed=args.seed + 1,
        action_cap=args.action_cap,
        max_offers=loaded.max_offers,
        deal=args.seed_games,
        board=board,
    )
    forks = kept(seeding.drain())
    if len(forks) < args.positions:
        print(
            f"only {len(forks)} positions kept for {args.positions} asked; "
            "raise --seed-games or --fork-rate",
            file=sys.stderr,
        )
    chosen = rng.sample(forks, min(args.positions, len(forks)))
    seeded = time.perf_counter() - started

    rows = []
    for index, fork in enumerate(chosen):
        children = probeable(fork.game, loaded.max_offers)
        if len(children) < 2:
            continue
        if len(children) > args.max_children:
            children = tuple(
                random.Random(args.seed + 11 + index).sample(
                    list(children), args.max_children
                )
            )

        # The head's opinion of the row, every child encoded from the choosing
        # seat's frame so the only thing varying is the position.
        head, leaves, slots = [], [], []
        games = []
        for action in children:
            child = imagine(fork.game, rng)
            apply(child, action)
            games.append(child)
            if is_over(child):
                points = tuple(
                    victory_points(child.state, s)
                    for s in range(child.state.num_players)
                )
                head.append(relative_points(points)[fork.seat])
            else:
                slots.append(len(head))
                head.append(0.0)
                leaves.append(Leaf(child, fork.seat, options(child, loaded.max_offers)))
        for slot, (_, value) in zip(slots, evaluator.evaluate(leaves)):
            head[slot] = value[fork.seat]

        # The same row, searched. `run_many` batches every child's leaves into
        # shared forwards, so a whole row costs one search rather than n.
        tree = list(head)
        if search is not None:
            for slot, (root, _, _) in enumerate(search.run_many(games)):
                value = backed_up(root, fork.seat)
                if value is not None:
                    tree[slot] = value

        # The truth, by rollout. Every child gets its own collector seeded
        # identically, so lane k across the row shares deck and sampling stream.
        returns = []
        for child in games:
            if is_over(child):
                points = tuple(
                    victory_points(child.state, s)
                    for s in range(child.state.num_players)
                )
                returns.append(
                    np.full(args.rollouts, relative_points(points)[fork.seat])
                )
                continue
            generator.manual_seed(args.seed + 7)
            branch = PairedBranching(
                policy,
                child,
                stream_seed=args.seed + 5000,
                lanes=args.rollouts,
                players=args.players,
                seed=args.seed + 3,
                action_cap=args.action_cap,
                max_offers=loaded.max_offers,
                deal=args.rollouts,
                board=board,
            )
            returns.append(
                np.asarray(
                    [reward(e.outcome)[fork.seat] for e in branch.drain()],
                    dtype=np.float64,
                )
            )

        true = np.asarray([r.mean() for r in returns], dtype=np.float64)
        ses = np.asarray(
            [
                r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
                for r in returns
            ],
            dtype=np.float64,
        )
        row = assess(np.asarray(head, dtype=np.float64), true)
        row.update(resolution(returns, true))
        row["head_values"] = [float(v) for v in head]
        row["true_values"] = [float(v) for v in true]
        row["standard_errors"] = [float(v) for v in ses]
        if search is not None:
            row["tree_values"] = [float(v) for v in tree]
            row["tree"] = assess(np.asarray(tree, dtype=np.float64), true)
        rows.append(row)
        print(
            f"[{len(rows)}/{len(chosen)}] children {row['children']:>2d} "
            f"spearman {row['spearman']:+.3f} top1 {'Y' if row['top1'] else 'n'} "
            f"regret {row['regret']:+.4f} gap {row['true_gap']:.4f} "
            f"paired-se {row['paired_se']:.4f} "
            f"({row['pairing_gain']:.1f}x) "
            + (f"tree {row['tree']['spearman']:+.3f} " if "tree" in row else "")
            + f"{'RESOLVED' if row['resolved'] else 'unresolved'}",
            flush=True,
        )

    if not rows:
        print("no position produced a row; raise --seed-games", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started
    payload = {
        "environment": environment(),
        "checkpoint": args.checkpoint,
        "args": vars(args),
        "iteration": loaded.iteration,
        "rollouts_each": args.rollouts,
        "seed_seconds": round(seeded, 1),
        "seconds": round(elapsed, 1),
        "summary": summarise(rows),
        "pooled": pooled(
            [r["head_values"] for r in rows],
            [r["true_values"] for r in rows],
            [r["standard_errors"] for r in rows],
        ),
        "standardised": standardised(
            [r["head_values"] for r in rows],
            [r["true_values"] for r in rows],
            [r["standard_errors"] for r in rows],
        ),
        "rows": rows,
    }
    if search is not None:
        payload["simulations"] = args.simulations
        payload["exploration"] = args.exploration
        payload["pooled_tree"] = pooled(
            [r["tree_values"] for r in rows],
            [r["true_values"] for r in rows],
            [r["standard_errors"] for r in rows],
        )
        payload["standardised_tree"] = standardised(
            [r["tree_values"] for r in rows],
            [r["true_values"] for r in rows],
            [r["standard_errors"] for r in rows],
        )
    if args.json:
        from pathlib import Path

        Path(args.json).write_text(json.dumps(payload, indent=1) + "\n")

    s = payload["summary"]
    print(f"\n{s['positions']} positions, {s['mean_children']:.1f} children each, "
          f"{args.rollouts} rollouts a child, {elapsed / 60:.1f} min")
    print(f"  resolved (top two told apart)   {s['resolved']:.0%}")
    print(f"  pairing gain over independent   {s['pairing_gain_mean']:.1f}x "
          f"(paired se {s['paired_se_mean']:.4f} vs {s['unpaired_se_mean']:.4f})")
    print(f"  spearman, head vs truth         {s['spearman_mean']:+.3f} "
          f"+/- {1.96 * s['spearman_sem']:.3f}")
    print(f"  top-1 hit rate                  {s['top1_rate']:.0%} "
          f"against {s['top1_chance']:.0%} chance")
    print(f"  regret of trusting the head     {s['regret_mean']:.4f} "
          f"({s['regret_mean_victory_points']:.2f} victory points)")
    print(f"  head spread {s['head_spread_mean']:.4f} against true "
          f"{s['true_spread_mean']:.4f}")
    q = payload["pooled"]
    print(f"\npooled over {q['children']} children, each position centred:")
    print(f"  reliability of the truth         {q['reliability']:.3f} "
          f"(signal {q['true_variance_observed']:.5f} vs noise {q['noise_variance']:.5f})")
    print(f"  pearson, as observed             {q['pearson_observed']:+.3f}")
    print(f"  pearson, corrected for that      {q['pearson_corrected']:+.3f}"
          f"{'   <- the number' if q['reliability'] > 0.2 else '   (reliability too low to trust)'}")
    z = payload["standardised"]
    print(f"\nthe same rows, each scaled to unit variance so one position counts "
          f"once ({z['positions']}):")
    print(f"  pearson, corrected               {z['pearson_corrected']:+.3f}"
          "   <- a typical position, not the widest few")
    if "pooled_tree" in payload:
        t = payload["pooled_tree"]
        hits = float(np.mean([1.0 if r["tree"]["top1"] else 0.0 for r in rows]))
        regret = float(np.mean([r["tree"]["regret"] for r in rows]))
        print(f"\nthe same children scored by a {args.simulations}-simulation "
              f"tree over the same head:")
        print(f"  pearson, as observed             {t['pearson_observed']:+.3f}")
        print(f"  pearson, corrected for that      {t['pearson_corrected']:+.3f}")
        print(f"  tree minus head, corrected       "
              f"{t['pearson_corrected'] - q['pearson_corrected']:+.3f}"
              "   <- positive means the search's value is the better label")
        print(f"  top-1 hit rate                   {hits:.0%} against "
              f"{payload['summary']['top1_rate']:.0%} for the head")
        print(f"  regret of trusting the tree      {regret:.4f} "
              f"({regret * 10.0:.2f} victory points) against "
              f"{payload['summary']['regret_mean_victory_points']:.2f} for the head")
        zt = payload["standardised_tree"]
        print(f"  row-standardised, tree           {zt['pearson_corrected']:+.3f} "
              f"against {z['pearson_corrected']:+.3f} for the head "
              f"({zt['pearson_corrected'] - z['pearson_corrected']:+.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
