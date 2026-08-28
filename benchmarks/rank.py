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

## The teacher: what expert iteration would actually distil

`--simulations N` runs **one search rooted at the parent** and reads its visit
counts over the same children the truth is known for. That is the object
distillation trains the policy toward, and it has never been measured against
anything. The question is one line: **do visits pick the true best child more
often than the policy's own prior does?**

If they do, the teacher is real at the level that matters and the recorded
distillation failure is in the training step. If they do not, the visit
distribution is the prior with noise on it, expert iteration cannot work at these
search settings, and the lever is the search's configuration rather than the
network. The prior is read from the same forward the search's root expansion
uses, over the same option tuple, so the two columns differ in nothing but the
search.

`--contested-only` is how to ask that question affordably, and the first run of
it did not. Truth costs 384 games a child; a search costs ~60 ms and the argmax
comparison is free once it exists. Buying truth for every position and spending
it only on the disagreements cost **~56,000 rollouts per informative row**. With
this flag a position is searched first, skipped outright when the prior and the
visits agree, and rolled out only when they differ — and then only for the **two
contested children**, since no other child can enter the comparison. About 70x
the informative rows per unit of compute.

One semantic change comes with it, and it is the more relevant question anyway.
Without the flag, `improved` means the visits found the globally best child.
With it, the row has only the two contested children in it, so `improved` means
**the search moved to the better of the two** — which is exactly what
distillation would be learning from.

`--child-trees` is the separate, more expensive pass: score every child by a
search rooted at *it*, which asks whether the backed-up value is a better label
than the head's read. That measurement is recorded and closed; the flag stays
because the harness is the same.

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

## Averaging a chance child, and what changed on 2026-08-28

Four transitions in this engine hide a draw. `ROLL` is excluded at the parent by
`probeable`, but the other three are not: `BUY_DEV_CARD`, `PLAY_KNIGHT` and a
`Phase.ROBBER` row's `MOVE_ROBBER` all resolve hidden information *inside*
`apply`. Building a child with `imagine` then `apply` therefore froze **one
sampled outcome per child** into the row, so head-versus-truth across siblings
was partly a comparison of dice and decks rather than of decisions. Measured on
this engine, that reaches **7-9% of probeable rows and 3.5-5.0% of children**;
every `Phase.ROBBER` row is affected.

**The fix averages, because averaging is what the search experiences.** After
`3e9d03a` the tree resamples a chance edge on every visit and keys the outcome,
so the edge's `Q` converges on the mean over outcomes and PUCT orders *actions*,
not realised children. That is precisely the quantity this metric exists to
predict, so `--chance-draws N` scores such a child as the mean over N
independent draws. The tree's *keying* is not borrowed — see
`catan.mcts.sampled_children` for why a harness that rolls its children out for
hundreds of plies cannot discard a repeated outcome's fresh copy the way a tree
can. Keying is the tree's implementation of the average; only the meaning
transfers.

**Both columns are averaged, or the comparison would break rather than mend.**
Under one draw, head and truth at a chance child were at least *consistent* —
both conditioned on the same realised outcome, which quietly inflated their
agreement there by a shock they shared. Averaging only the head would leave the
truth conditioned on one draw and measure a mismatch instead of an ordering. So
the `--rollouts` budget is **partitioned across the N draws rather than
multiplied**: total games rolled out is unchanged, and the lane-to-stream map is
unchanged too, so draw `d` lane `k` of one child still shares its deck and its
sampling stream with draw `d` lane `k` of every sibling. `--chance-draws 1`
restores the old path exactly, and every number this module produced before
2026-08-28 was taken under it.

**Cost.** Only chance children multiply, and they are 5% of children, so the head
column costs 1.35x at N=8 and the rollout column — 99% of the wall clock — costs
nothing. `--simulations` at the parent is untouched: the tree has handled its own
chance since `3e9d03a`.

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
from catan.actions import ActionType, legal_actions, within_offer_budget
from catan.board.board import random_base_board
from catan.game import imagine, is_over, to_move
from catan.mcts import Leaf, draws_hidden, sampled_children
from catan.rewards import relative_points, reward
from catan.selfplay import Collector, Episode
from catan.victory import victory_points

DRAWS = 8
"""Draws per chance child. See "Averaging a chance child" in the docstring."""

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


def share(total: int, parts: int) -> list[int]:
    """`total` split over `parts` as evenly as it goes, summing to `total`.

    How a chance child's rollout budget is spread across its draws. It has to
    sum exactly: the budget is what makes every child's lane count equal, and
    `resolution` pairs the top two children lane by lane. Front-loading the
    remainder rather than dropping it keeps every child's total at `total`
    whatever its draw count, so a row that mixes chance children with
    deterministic ones stays paired across both.

    A draw with nothing left gets zero and is skipped, which is what happens
    when a probe is run with fewer rollouts than draws — a smoke setting, not a
    measurement one.
    """
    if parts < 1:
        raise ValueError("a budget needs at least one part")
    base, extra = divmod(max(total, 0), parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def lane_plan(rollouts: int, draws: int) -> list[tuple[int, int]]:
    """`(lanes, stream offset)` per draw, for one child's rollout budget.

    The offsets run consecutively, so a child's lanes cover stream seeds
    `stream_seed + 0 .. + rollouts - 1` in order however many draws it has.
    Every sibling therefore uses the same stream set in the same order, which is
    what `resolution` needs to pair the top two children lane by lane, and what
    makes a one-draw child's plan `[(rollouts, 0)]` — exactly the single
    collector the harness built before the chance fix.

    Draws with no budget left are dropped rather than run empty.
    """
    plan, offset = [], 0
    for lanes in share(rollouts, draws):
        if lanes:
            plan.append((lanes, offset))
        offset += lanes
    return plan


@dataclass(frozen=True)
class Row:
    """One position's children as the head reads them, before any rollout.

    `games[i]` is child `i`'s draw *list* — length one wherever the action drew
    nothing — and `drawn[i]` the head's value for each of those draws.
    `head[i]` is their mean, which is the expectimax value a chance edge's `Q`
    converges on in the tree and therefore the number the search's ordering is
    actually made of.
    """

    head: list[float]
    drawn: list[list[float]]
    games: list[list[object]]
    hot: list[bool]


def head_row(
    game, children, seat: int, *, evaluator, max_offers, draws: int, rng, extra
) -> Row:
    """Score every child of one position, averaging over each one's draws.

    Extracted from `main` because this is where the chance defect lived: the row
    is the object the whole metric is computed from, and it was built inline in a
    two-hundred-line function where nothing could reach it. Every child is
    encoded from the choosing seat's frame, so the only thing varying across the
    row is the position.

    `extra` is a *factory*, called once per child, so each child's draws two and
    up come off a stream in the same state — common random numbers on the chance
    dimension, matching what this module already does for the rollouts. Draw one
    comes off the shared `rng`, which is what keeps a chance-free row
    bit-identical to the single-draw path.
    """
    hot = [draws_hidden(game, action) for action in children]
    drawn: list[list[float]] = []
    games: list[list[object]] = []
    leaves, slots = [], []
    for action in children:
        outcomes = sampled_children(
            game, action, draws=draws, rng=rng, extra=extra()
        )
        games.append(outcomes)
        row: list[float] = []
        for child in outcomes:
            if is_over(child):
                # A finished child has a known value on the same scale, which is
                # what the tree would back up.
                points = tuple(
                    victory_points(child.state, s)
                    for s in range(child.state.num_players)
                )
                row.append(relative_points(points)[seat])
            else:
                slots.append((len(drawn), len(row)))
                row.append(0.0)
                leaves.append(Leaf(child, seat, options(child, max_offers)))
        drawn.append(row)
    for (child_at, draw_at), (_, value) in zip(slots, evaluator.evaluate(leaves)):
        drawn[child_at][draw_at] = value[seat]
    return Row(
        head=[float(np.mean(v)) for v in drawn],
        drawn=drawn,
        games=games,
        hot=hot,
    )


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


def teacher_row(prior: np.ndarray, visits: np.ndarray, true: np.ndarray) -> dict:
    """Does searching move the prior toward the truth, or just around?

    `improved` is the decision-relevant cell: the search changed its mind about
    the best child *and* was right to. `damaged` is the same move in the wrong
    direction. Expert iteration needs the first to outnumber the second; a
    teacher that moves the argmax at random has nothing to teach however often
    it moves it.
    """
    best = int(true.argmax())
    prior_pick = int(prior.argmax())
    visit_pick = int(visits.argmax())
    return {
        "prior_top1": prior_pick == best,
        "visits_top1": visit_pick == best,
        "moved": prior_pick != visit_pick,
        "improved": prior_pick != best and visit_pick == best,
        "damaged": prior_pick == best and visit_pick != best,
        "prior_spearman": correlation(ranks(prior), ranks(true)),
        "visits_spearman": correlation(ranks(visits), ranks(true)),
        "prior_regret": float(true[best] - true[prior_pick]),
        "visits_regret": float(true[best] - true[visit_pick]),
    }


def chance(rows: list[dict], draws: int) -> dict:
    """How much of this run was contaminated, and how much noise is left.

    `spread` is the mean standard deviation of the head's read of one chance
    child across that child's draws — the noise a single-draw probe left in the
    row, measured on the run rather than assumed by the power calculation.
    `residual` is what averaging leaves of it, `spread / sqrt(draws)`, which is
    the figure to read against `true_gap_mean`: the argmax has to resolve that
    gap, so the residual has to sit well inside it.

    `None` for both where nothing was drawn — either a run with no chance row in
    it, or `--chance-draws 1`, which measures no spread because it takes no
    second draw. That is the honest reading and not a zero.
    """
    within = [s for row in rows for s in row.get("chance_spread", ())]
    hot = [1.0 if row.get("chance_children", 0) else 0.0 for row in rows]
    spread = float(np.mean(within)) if within else None
    return {
        "draws": draws,
        "rows": int(sum(hot)),
        "row_share": float(np.mean(hot)) if hot else 0.0,
        "children": int(sum(row.get("chance_children", 0) for row in rows)),
        "spread": spread,
        "residual": spread / np.sqrt(draws) if spread is not None else None,
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
    parser.add_argument(
        "--chance-draws",
        type=int,
        default=DRAWS,
        help="draws to average a chance child over, head and truth alike; the "
             "rollout budget is partitioned across them, not multiplied. 1 "
             "restores the single-draw path every number before 2026-08-28 was "
             "taken under",
    )
    parser.add_argument("--simulations", type=int, default=0,
                        help="also score every child by the backed-up value of a "
                             "PUCT search rooted at it; 0 scores the head alone")
    parser.add_argument("--contested-only", action="store_true",
                        help="search first and roll out only positions where the "
                             "visits and the prior disagree, and only their two "
                             "contested children; the cheap way to buy McNemar "
                             "rows, see the module docstring")
    parser.add_argument("--child-trees", action="store_true",
                        help="also score each child by a search rooted at it; "
                             "costs a search per child rather than one per "
                             "position, and answers the label question rather "
                             "than the teacher one")
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

    from catan.mcts import Search
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
    agreed = 0
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

        # The head's opinion of the row. A chance child is averaged over
        # `--chance-draws` draws; every other child is drawn once off the shared
        # stream exactly as before.
        scored = head_row(
            fork.game,
            children,
            fork.seat,
            evaluator=evaluator,
            max_offers=loaded.max_offers,
            draws=args.chance_draws,
            rng=rng,
            # A fresh generator on the same seed per child, so draw `d` of one
            # child comes off the same stream state as draw `d` of its siblings.
            extra=lambda: random.Random(args.seed + 13 + index),
        )
        head, drawn, games, hot = scored.head, scored.drawn, scored.games, scored.hot

        # The same row, searched. `run_many` batches every child's leaves into
        # shared forwards, so a whole row costs one search rather than n — and
        # one search rather than n*draws, since every draw of every child goes
        # into the same flat wave. A chance child's tree value is the mean over
        # whichever of its draws produced a searchable root, which is the same
        # average the head column takes; a draw that produced none falls back to
        # the head for that draw, as it always did.
        tree = list(head)
        if search is not None and args.child_trees:
            flat = [child for outcomes in games for child in outcomes]
            searched = [
                backed_up(root, fork.seat) for root, _, _ in search.run_many(flat)
            ]
            at = 0
            for slot, outcomes in enumerate(games):
                values = [
                    searched[at + d] if searched[at + d] is not None else drawn[slot][d]
                    for d in range(len(outcomes))
                ]
                at += len(outcomes)
                tree[slot] = float(np.mean(values))

        # One search at the *parent*: the teacher, and the prior it perturbs.
        # `Search._options` is the same filter `options` applies here and
        # `Action` is a NamedTuple, so the two tuples align by equality.
        prior = np.zeros(len(children), dtype=np.float64)
        visits = np.zeros(len(children), dtype=np.float64)
        if search is not None:
            _, root_options, root_visits = search.run(fork.game)
            where = {action: i for i, action in enumerate(root_options)}
            (prior_row, _), = evaluator.evaluate(
                [Leaf(fork.game, fork.seat, root_options)]
            )
            for slot, action in enumerate(children):
                at = where.get(action)
                if at is None:
                    continue
                visits[slot] = float(root_visits[at])
                prior[slot] = float(prior_row[at])

        if args.contested_only:
            prior_pick, visit_pick = int(prior.argmax()), int(visits.argmax())
            if prior_pick == visit_pick:
                agreed += 1
                continue
            # Only the two children the search actually chose between: no other
            # child can appear in the comparison, so no other child's truth is
            # worth 384 games.
            keep = [prior_pick, visit_pick]
            children = tuple(children[i] for i in keep)
            games = [games[i] for i in keep]
            head = [head[i] for i in keep]
            tree = [tree[i] for i in keep]
            drawn = [drawn[i] for i in keep]
            hot = [hot[i] for i in keep]
            prior, visits = prior[keep], visits[keep]

        # The truth, by rollout. Every child gets its own collector seeded
        # identically, so lane k across the row shares deck and sampling stream.
        #
        # A chance child spends the same `--rollouts` budget spread over its
        # draws rather than N times the budget on one of them, so the whole
        # measurement costs what it always did. `lane_plan` gives each draw its
        # lane count and its `stream_seed` offset, consecutively, so a child's
        # lanes still cover streams `stream_seed + 0 .. + rollouts - 1` in the
        # same order as every sibling's however many draws it has. That is what
        # keeps the pairing in `resolution` intact and what makes a single-draw
        # row bit-identical.
        returns = []
        for outcomes in games:
            plan = lane_plan(args.rollouts, len(outcomes))
            got = []
            for child, (lanes, offset) in zip(outcomes, plan):
                if is_over(child):
                    points = tuple(
                        victory_points(child.state, s)
                        for s in range(child.state.num_players)
                    )
                    got.append(np.full(lanes, relative_points(points)[fork.seat]))
                    continue
                generator.manual_seed(args.seed + 7)
                branch = PairedBranching(
                    policy,
                    child,
                    stream_seed=args.seed + 5000 + offset,
                    lanes=lanes,
                    players=args.players,
                    seed=args.seed + 3,
                    action_cap=args.action_cap,
                    max_offers=loaded.max_offers,
                    deal=lanes,
                    board=board,
                )
                got.append(
                    np.asarray(
                        [reward(e.outcome)[fork.seat] for e in branch.drain()],
                        dtype=np.float64,
                    )
                )
            returns.append(np.concatenate(got))

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
        # The chance columns: how many of this row's children hid a draw, and
        # how far the head's own read of such a child moved across its draws.
        # `chance_spread` is the noise a single-draw probe was leaving in the
        # row, and the number the recommended `DRAWS` has to be read against.
        row["chance_children"] = int(sum(hot))
        row["chance_spread"] = [float(np.std(v)) for v in drawn if len(v) > 1]
        if search is not None:
            if args.child_trees:
                row["tree_values"] = [float(v) for v in tree]
                row["tree"] = assess(np.asarray(tree, dtype=np.float64), true)
            row["prior_values"] = [float(v) for v in prior]
            row["visit_values"] = [float(v) for v in visits]
            row["teacher"] = teacher_row(prior, visits, true)
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
        if args.contested_only and agreed:
            # Not a failure: every screened position agreed, which is the
            # measurement rather than the absence of one.
            print(f"screened {agreed} positions, none contested", file=sys.stderr)
            return 0
        print("no position produced a row; raise --seed-games", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started
    payload = {
        "environment": environment(),
        "checkpoint": args.checkpoint,
        "args": vars(args),
        "iteration": loaded.iteration,
        "rollouts_each": args.rollouts,
        "chance": chance(rows, args.chance_draws),
        "seed_seconds": round(seeded, 1),
        "agreed_skipped": agreed,
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
    # Only `--child-trees` produces a tree column to pool; `--simulations`
    # alone measures the teacher at the parent and leaves this untouched.
    if search is not None and args.child_trees:
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
    c = payload["chance"]
    print(f"  chance rows {c['rows']} ({c['row_share']:.0%}), {c['children']} "
          f"chance children, {c['draws']} draws each")
    if c["spread"] is not None:
        print(f"  spread within a chance child    {c['spread']:.4f}, so "
              f"{c['residual']:.4f} left after averaging, against a top-two gap "
              f"of {s['true_gap_mean']:.4f}"
              + ("   <- residual inside the gap"
                 if c["residual"] < s["true_gap_mean"] / 2
                 else "   <- RAISE --chance-draws"))
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
    if rows and "teacher" in rows[0]:
        cells = [r["teacher"] for r in rows]
        if args.contested_only:
            print(f"\nscreened {agreed + len(rows)} positions; {agreed} agreed and "
                  f"were skipped before any rollout, {len(rows)} were contested")
        share = lambda key: float(np.mean([1.0 if c[key] else 0.0 for c in cells]))
        print(f"\nthe teacher: one search at the parent, {args.simulations} "
              f"simulations, against what the prior already said")
        print(f"  top-1 against the truth          prior {share('prior_top1'):.0%}"
              f"   visits {share('visits_top1'):.0%}"
              f"   (chance {payload['summary']['top1_chance']:.0%})")
        print(f"  spearman against the truth       prior "
              f"{np.mean([c['prior_spearman'] for c in cells]):+.3f}"
              f"   visits {np.mean([c['visits_spearman'] for c in cells]):+.3f}")
        print(f"  regret, victory points           prior "
              f"{np.mean([c['prior_regret'] for c in cells]) * 10:.3f}"
              f"   visits {np.mean([c['visits_regret'] for c in cells]) * 10:.3f}")
        print(f"  search moved the argmax          {share('moved'):.0%} of rows: "
              f"**{share('improved'):.0%} improved, {share('damaged'):.0%} damaged**")
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
