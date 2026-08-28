"""Our policy against recorded human decisions, one decision at a time.

This project has no human strength referent, and a win rate against humans is
not going to supply one. Resolving a 5 pp difference needs on the order of 1,500
games; at ~20 minutes a game that is ~500 human-hours, and the arena that could
produce them is days old. So the quantity everyone reaches for first is the one
quantity that stays out of reach.

**A single 4-player game is hundreds of labelled human decisions**, though, and
those are measurable now. Fifty games is on the order of 10^4 decisions, which
is a large sample by any standard, and a per-decision claim is stronger than a
per-game one rather than weaker: it says where the policy and a human diverge,
not merely that one of them wins more often.

Two numbers per decision, each against its own matched null:

    top-1 agreement — does the policy's argmax over the legal option set equal
    the action the human took.

    log-loss — the policy's own distribution evaluated at the human's action,
    in nats.

**The null is uniform over the legal option set at that same position, and is
computed there rather than assumed.** A statistic without its null is not a
measurement, and here the null is not a constant: the option count runs from 2
to sixty-odd, so the uniform baseline for top-1 is the mean of `1/n` over the
positions actually scored and for log-loss the mean of `log n`. Quoting a single
global figure -- 7.2 options at a non-trivial decision, hence 13.9% and 1.97
nats -- would be a different number, and it is `mean(1/n)` against `1/mean(n)`
that separates them. `catan.placement` is the precedent for the comparison
itself: held-out log-loss 1.3746 against 1.3863 for chance.

Both differences are reported **paired**, per position, because the difference
is what the null is for and pairing is free once the null is per-position.

**The aggregate alone would be a statement about trading.** `PROPOSE_TRADE` is
the most common sibling kind on record and `_offer_actions` is ~26% of
`legal_actions`, so an unstratified mean is dominated by the one decision type
whose option set is largest and whose stakes are smallest. Everything is
therefore broken down per `ActionType`, per `Phase` and by game progress, and
each breakdown partitions the decisions exactly, so it sums back to the
aggregate.

**Trivial decisions are excluded and counted.** Where one action is legal,
agreement is 1.0 by construction and carries no information; a policy could be
scored at 60% on a corpus that is mostly forced rolls. The excluded count is
reported, and `nontrivial_per_game` is the number that says how many recorded
games the measurement actually needs.

**The distribution scored is the one the search sees**, via
`catan.netbot.LeafEvaluator.evaluate` on a `catan.mcts.Leaf` -- not a
separately-built softmax. That matters most at the trade slot, which is one slot
in the flat categorical standing for "propose something" whose mass the
evaluator splits across the legal offers by the pair distribution. A policy that
wants to trade 40% of the time and has twenty legal offers puts ~2% on each, so
a *single* recorded proposal is a hard target by construction. Scoring a
hand-rolled distribution instead would quietly measure a different agent, and
scoring the flat slot alone would credit the policy for naming an offer it did
not name.

Records come from anywhere that produces a `catan.record.Record`. The motivating
source is a first-party human arena that journals every action with its hidden
information and hands back this project's own action space, but nothing here
depends on it: it takes `Record`s.

    python -m benchmarks.human_agreement --records /w/tmp/human.jsonl \\
        --checkpoint /w/runs/lam095/latest.pt --label lam095-human \\
        --verdicts /w/runs/eval

    python -m benchmarks.human_agreement --records /w/tmp/selfplay.jsonl \\
        --checkpoint /w/runs/lam095/latest.pt --label lam095-control --control \\
        --verdicts /w/runs/eval

`--verdicts` is spelled absolutely there on purpose. It defaults to `runs/eval`
relative to the working directory, which is `benchmarks.duel`'s default and is
right only when the runner's cwd is the repo root; every module here is normally
run from `src/`, where the relative path would quietly make `src/runs/eval`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Collection, Iterable, Iterator, NamedTuple, Sequence

import numpy as np

from benchmarks.throughput import environment
from catan.actions import (
    Action,
    ActionSpace,
    ActionType,
    apply,
    legal_actions,
    within_offer_budget,
)
from catan.arena import Z_95, wilson
from catan.game import imagine, is_over, start, to_move
from catan.mcts import Leaf
from catan.record import Record, actions_of, board_of, read as read_records

PROGRESS_BUCKETS = 5

# The floor under a probability before it is logged. `LeafEvaluator._prior`
# exponentiates a masked log-softmax, so an action the policy has all but ruled
# out can arrive as an exact 0.0 and `log 0` would make one decision swallow the
# whole mean. Positions that hit the floor are counted, so a log-loss it
# distorted cannot be read as one it did not.
FLOOR = 1e-12


def option_key(action: Action) -> tuple:
    """What the policy actually chose, with the parts it does not choose dropped.

    `Action` is a `NamedTuple`, so `==` includes `ask` -- the proposer's
    preferred responder order. `actions._offer_actions` enumerates every offer
    with `ask=()` while a recorded game may carry a real order there, so
    comparing whole actions would fail to match *every* proposal and silently
    file the most common decision type in the corpus as unrepresentable. The
    network has no `ask` head; it is not part of the decision being scored.
    """
    return (
        int(action.type),
        action.a,
        action.b,
        tuple(action.give),
        tuple(action.want),
    )


@dataclass
class Tally:
    """What the replay saw and what it did not score, so the total reconciles."""

    games: int = 0
    actions: int = 0
    considered: int = 0
    trivial: int = 0
    unrepresented: int = 0
    off_seat: int = 0
    unsampled: int = 0
    floored: int = 0
    unrepresented_by_kind: dict[str, int] = field(default_factory=dict)

    def payload(self) -> dict:
        out = asdict(self)
        out["nontrivial"] = self.considered - self.trivial
        out["nontrivial_per_game"] = (
            round((self.considered - self.trivial) / self.games, 1)
            if self.games
            else 0.0
        )
        return out


@dataclass(frozen=True)
class Decision:
    """One scored human decision, and the null it is scored against.

    `options` is the whole null: the uniform policy's chance of agreeing here is
    `1 / options` and its loss is `log options`. Keeping the count on the row
    rather than a baseline on the row is what makes every aggregate -- overall
    or per bucket -- carry its own matched null with no global constant anywhere.
    """

    game: int
    step: int
    seat: int
    progress: float
    phase: str
    kind: int
    options: int
    agreed: bool
    log_prob: float

    @property
    def loss(self) -> float:
        return -self.log_prob

    @property
    def null_agreement(self) -> float:
        return 1.0 / self.options

    @property
    def null_loss(self) -> float:
        return math.log(self.options)


class _Pending(NamedTuple):
    """A position waiting for a forward pass, and where its answer belongs."""

    leaf: Leaf
    index: int
    game: int
    step: int
    seat: int
    progress: float
    phase: str
    kind: int


def positions(
    record: Record,
    game: int,
    tally: Tally,
    *,
    max_offers: int | None = None,
    seats: Collection[int] | None = None,
    sample: float = 1.0,
    rng: random.Random | None = None,
) -> Iterator[_Pending]:
    """Replay one record, yielding every non-trivial decision worth scoring.

    The loop is `catan.dataset.samples_from`'s and `catan.behaviour.walk`'s:
    start from the board and seed the record carries, step `actions_of` in order,
    and read the live game before each action rather than after. Legality is not
    rechecked -- `catan.record.replay` is what verifies a record -- but a taken
    action that is not in the enumerated option set is counted, because for
    proposals the enumeration is a *sample* and not the whole legal set.

    A snapshot rides on each `Leaf` so a batch can be scored after the replay
    has moved on. `randomize_deck=False`: the copy is scored, never stepped, and
    reshuffling would put a different deck under an encoder that reads it.
    """
    live = start(board_of(record), record.num_players, random.Random(record.seed))
    scratch = random.Random(0)
    total = max(1, len(record.actions))
    tally.games += 1

    for step, action in enumerate(actions_of(record)):
        if is_over(live):
            break
        tally.actions += 1
        seat = to_move(live)

        if seats is not None and seat not in seats:
            tally.off_seat += 1
            apply(live, action)
            continue
        if sample < 1.0 and (rng or random).random() >= sample:
            tally.unsampled += 1
            apply(live, action)
            continue

        options = tuple(within_offer_budget(live, legal_actions(live), max_offers))
        tally.considered += 1
        if len(options) < 2:
            tally.trivial += 1
        else:
            keys = [option_key(option) for option in options]
            wanted = option_key(action)
            if wanted in keys:
                yield _Pending(
                    leaf=Leaf(
                        imagine(live, scratch, randomize_deck=False), seat, options
                    ),
                    index=keys.index(wanted),
                    game=game,
                    step=step,
                    seat=seat,
                    progress=step / total,
                    phase=live.phase.name,
                    kind=int(action.type),
                )
            else:
                tally.unrepresented += 1
                name = ActionType(action.type).name
                tally.unrepresented_by_kind[name] = (
                    tally.unrepresented_by_kind.get(name, 0) + 1
                )
        apply(live, action)


def _check_space(record: Record, space: ActionSpace | None) -> None:
    """Refuse a record the network cannot be asked about.

    A 3-player record scored by a 4-player checkpoint, or a Seafarers layout
    scored by a base-board one, does not fail loudly on its own: the encoder
    would build a differently shaped observation and the mismatch would surface
    somewhere unrelated. `catan.netbot._check_players` makes the same argument
    for the arena path.
    """
    if space is None:
        return
    topology = board_of(record).topology
    shape = (topology.num_vertices, topology.num_edges, topology.num_hexes)
    expected = (space.num_vertices, space.num_edges, space.num_hexes)
    if shape != expected:
        raise ValueError(f"record layout {shape} is not the network's {expected}")
    if record.num_players != space.num_players:
        raise ValueError(
            f"record has {record.num_players} players, "
            f"network was trained for {space.num_players}"
        )


def score(
    records: Iterable[Record],
    evaluator,
    *,
    max_offers: int | None = None,
    seats: Collection[int] | None = None,
    space: ActionSpace | None = None,
    batch: int = 64,
    sample: float = 1.0,
    seed: int = 0,
) -> tuple[list[Decision], Tally]:
    """Every non-trivial decision in `records`, scored by `evaluator`.

    `evaluator` is a `catan.mcts.Evaluator` -- in production
    `catan.netbot.LeafEvaluator`, which is the same object a batched search
    scores its leaves with, so the distribution measured here and the
    distribution the search acts on cannot drift apart.

    `sample` below 1 subsamples decisions off `seed`'s stream. It is a cost
    control on a large corpus, not a default: the draw happens after the seat
    filter and before the option enumeration, so the reported per-position
    counts stay unbiased but the excluded-trivial tally becomes an estimate.
    """
    tally = Tally()
    out: list[Decision] = []
    rng = random.Random(seed)
    pending: list[_Pending] = []

    for game, record in enumerate(records):
        _check_space(record, space)
        for item in positions(
            record,
            game,
            tally,
            max_offers=max_offers,
            seats=seats,
            sample=sample,
            rng=rng,
        ):
            pending.append(item)
            if len(pending) >= batch:
                _flush(pending, evaluator, out, tally)
    _flush(pending, evaluator, out, tally)
    return out, tally


def _flush(
    pending: list[_Pending], evaluator, out: list[Decision], tally: Tally
) -> None:
    if not pending:
        return
    scored = evaluator.evaluate([item.leaf for item in pending])
    for item, (prior, _value) in zip(pending, scored):
        probability = float(np.asarray(prior, dtype=np.float64)[item.index])
        best = int(np.argmax(np.asarray(prior, dtype=np.float64)))
        if probability < FLOOR:
            tally.floored += 1
            probability = FLOOR
        out.append(
            Decision(
                game=item.game,
                step=item.step,
                seat=item.seat,
                progress=item.progress,
                phase=item.phase,
                kind=item.kind,
                options=len(item.leaf.options),
                agreed=best == item.index,
                log_prob=math.log(probability),
            )
        )
    pending.clear()


def _clustered(values: Sequence[float], games: Sequence[int]) -> tuple[float, float | None]:
    """Pooled mean, and a 95% half-width whose sample size is games not positions.

    Consecutive decisions in one game differ by a single build and are heavily
    correlated -- `catan.dataset.split_by_game` exists for the same reason -- so
    an interval taken over positions would report a precision the data does not
    have. The cluster statistic is the per-game mean and the spread across games
    is the standard error. `None` where one game cannot supply a spread.
    """
    if not values:
        return 0.0, None
    grouped: dict[int, list[float]] = defaultdict(list)
    for value, game in zip(values, games):
        grouped[game].append(value)
    pooled = sum(values) / len(values)
    means = [sum(rows) / len(rows) for rows in grouped.values()]
    if len(means) < 2:
        return pooled, None
    centre = sum(means) / len(means)
    variance = sum((m - centre) ** 2 for m in means) / (len(means) - 1)
    return pooled, Z_95 * math.sqrt(variance / len(means))


def _band(mean: float, half: float | None) -> list[float] | None:
    return None if half is None else [mean - half, mean + half]


def summarise(scored: Sequence[Decision]) -> dict:
    """Both metrics, both matched nulls, and the paired difference between them.

    The nulls are means over the rows present -- `mean(1/n)` and `mean(log n)` --
    so this block is self-contained whether it covers the whole corpus or one
    `ActionType`. Nothing here reads a global option count, which is the whole
    point: the aggregate's null and a bucket's null are different numbers and
    both are right.
    """
    total = len(scored)
    if not total:
        return {"decisions": 0, "games": 0}

    games = [d.game for d in scored]
    agreed = [1.0 if d.agreed else 0.0 for d in scored]
    losses = [d.loss for d in scored]
    null_agreed = [d.null_agreement for d in scored]
    null_losses = [d.null_loss for d in scored]
    lift = [a - b for a, b in zip(agreed, null_agreed)]
    gain = [a - b for a, b in zip(null_losses, losses)]

    wins = int(sum(agreed))
    low, high = wilson(wins, total)
    top1, top1_half = _clustered(agreed, games)
    loss, loss_half = _clustered(losses, games)
    lift_mean, lift_half = _clustered(lift, games)
    gain_mean, gain_half = _clustered(gain, games)

    # Nothing is rounded. A bucket's figures have to pool back to the
    # aggregate's exactly, and rounding here put a floor under that check for
    # no gain: the printing formats its own decimals, and `benchmarks.duel`
    # writes its verdict unrounded for the same reason.
    return {
        "decisions": total,
        "games": len(set(games)),
        "mean_options": sum(d.options for d in scored) / total,
        "top1": top1,
        "top1_null": sum(null_agreed) / total,
        "top1_lift": lift_mean,
        "top1_lift_ci": _band(lift_mean, lift_half),
        # Over positions rather than games, and so an understatement of the
        # width. Kept because it is the interval every other benchmark here
        # quotes, and labelled so the two are not confused.
        "top1_wilson_positions": [low, high],
        "top1_ci": _band(top1, top1_half),
        "log_loss": loss,
        "log_loss_null": sum(null_losses) / total,
        "log_loss_gain": gain_mean,
        "log_loss_gain_ci": _band(gain_mean, gain_half),
        "log_loss_ci": _band(loss, loss_half),
    }


def grouped(
    scored: Sequence[Decision], key, *, by_size: bool = True
) -> dict[str, dict]:
    """`summarise` per bucket. Buckets partition, so they sum to the aggregate.

    Ordered by size, because which decision type dominates the aggregate is the
    first thing to read off the table. `by_size=False` keeps the bucket's own
    order, which is what an ordered axis like game progress wants.
    """
    buckets: dict[str, list[Decision]] = defaultdict(list)
    for decision in scored:
        buckets[key(decision)].append(decision)
    order = (
        (lambda pair: (-len(pair[1]), pair[0])) if by_size else (lambda pair: pair[0])
    )
    return {name: summarise(rows) for name, rows in sorted(buckets.items(), key=order)}


def progress_bucket(decision: Decision, buckets: int = PROGRESS_BUCKETS) -> str:
    index = min(int(decision.progress * buckets), buckets - 1)
    return f"{index * 100 // buckets:02d}-{(index + 1) * 100 // buckets:02d}%"


def report(scored: Sequence[Decision], tally: Tally) -> dict:
    return {
        "overall": summarise(scored),
        "by_action": grouped(scored, lambda d: ActionType(d.kind).name),
        "by_phase": grouped(scored, lambda d: d.phase),
        "by_progress": grouped(scored, progress_bucket, by_size=False),
        "tally": tally.payload(),
    }


# ---------------------------------------------------------------------------
# CLI


def _seats(text: str | None) -> Collection[int] | None:
    """Which seats are the humans of interest. `None` means every seat."""
    if not text or text == "all":
        return None
    return frozenset(int(part) for part in text.split(",") if part.strip())


def _verdict_path(directory: str, label: str) -> Path:
    return Path(directory) / f"humanagree-{label.replace('/', '-')}.json"


def _line(name: str, block: dict) -> str:
    top1 = block["top1"] * 100
    null = block["top1_null"] * 100
    return (
        f"  {name:<22} {block['decisions']:>7}  "
        f"{top1:>5.1f}% vs {null:>5.1f}%  "
        f"{block['log_loss']:>6.3f} vs {block['log_loss_null']:>6.3f}  "
        f"{block['mean_options']:>5.1f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        action="append",
        required=True,
        help="JSON lines of `catan.record.Record`; repeatable",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", default=None, help="names the verdict file")
    parser.add_argument(
        "--seats",
        default="all",
        help="comma-separated seats to score, or `all`. A human arena scores "
        "the human's seat; a self-play control scores every seat",
    )
    parser.add_argument("--games", type=int, default=0, help="0 reads them all")
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="fraction of decisions to score, off --seed's stream",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-offers",
        type=int,
        default=None,
        help="offer budget the option set is enumerated under. Default: the "
        "budget the checkpoint records training under, which is the horizon it "
        "learned on. A recorded game played to a wider budget will have "
        "proposals this rules out, and they are counted as unrepresented",
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="stamp the verdict as a harness control, not a result. Use it for "
        "self-play records: a policy scored against its own games is an "
        "upper-bound check on the harness and says nothing about humans",
    )
    parser.add_argument("--verdicts", default="runs/eval")
    parser.add_argument("--json", default=None, help="write here instead")
    parser.add_argument("--no-json", action="store_true")
    args = parser.parse_args(argv)

    from catan.netbot import LeafEvaluator, load

    records: list[Record] = []
    for path in args.records:
        for record in read_records(path):
            records.append(record)
            if args.games and len(records) >= args.games:
                break
        if args.games and len(records) >= args.games:
            break
    if not records:
        print("no records read", file=sys.stderr)
        return 1

    loaded = load(args.checkpoint, board_of(records[0]).topology, args.device)
    budget = loaded.max_offers if args.max_offers is None else args.max_offers
    evaluator = LeafEvaluator(policy=loaded.policy, space=loaded.space)

    started = time.perf_counter()
    scored, tally = score(
        records,
        evaluator,
        max_offers=budget,
        seats=_seats(args.seats),
        space=loaded.space,
        batch=args.batch,
        sample=args.sample,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - started

    if not scored:
        print("no non-trivial decision was scored", file=sys.stderr)
        return 1

    label = args.label or Path(args.checkpoint).parent.name
    payload = {
        "environment": environment(),
        "label": label,
        "control": bool(args.control),
        "checkpoint": args.checkpoint,
        "iteration": loaded.iteration,
        "records": args.records,
        "seats": args.seats,
        "max_offers": budget,
        "sample": args.sample,
        "seed": args.seed,
        "seconds": round(elapsed, 1),
        **report(scored, tally),
    }

    destination = None
    if not args.no_json:
        destination = (
            Path(args.json) if args.json else _verdict_path(args.verdicts, label)
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")

    overall = payload["overall"]
    counts = payload["tally"]
    if args.control:
        print("CONTROL: a policy against its own games. Not a human result.")
    print(f"{label} @ iteration {loaded.iteration}, offer budget {budget}")
    print(
        f"{counts['games']} games, {counts['actions']} actions, "
        f"{overall['decisions']} non-trivial decisions scored "
        f"({counts['nontrivial_per_game']} per game), {payload['seconds']}s"
    )
    print(
        f"  excluded: {counts['trivial']} trivial, "
        f"{counts['unrepresented']} not in the enumerated option set, "
        f"{counts['off_seat']} off-seat, {counts['floored']} floored"
    )
    print(
        f"  top-1 agreement   {overall['top1']*100:.1f}% "
        f"against a matched null of {overall['top1_null']*100:.1f}%  "
        f"(lift {overall['top1_lift']*100:+.1f} pp"
        + (
            f" [{overall['top1_lift_ci'][0]*100:+.1f}, "
            f"{overall['top1_lift_ci'][1]*100:+.1f}])"
            if overall["top1_lift_ci"]
            else ")"
        )
    )
    print(
        f"  log-loss          {overall['log_loss']:.4f} nats "
        f"against a matched null of {overall['log_loss_null']:.4f}  "
        f"(gain {overall['log_loss_gain']:+.4f}"
        + (
            f" [{overall['log_loss_gain_ci'][0]:+.4f}, "
            f"{overall['log_loss_gain_ci'][1]:+.4f}])"
            if overall["log_loss_gain_ci"]
            else ")"
        )
    )
    print(f"  mean legal options at a scored decision {overall['mean_options']:.2f}")

    for title, key in (
        ("by decision type", "by_action"),
        ("by phase", "by_phase"),
        ("by game progress", "by_progress"),
    ):
        print(f"\n{title}    decisions   top-1 vs null    log-loss vs null  options")
        for name, block in payload[key].items():
            print(_line(name, block))

    if destination is not None:
        print(f"\nappended to {destination}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
