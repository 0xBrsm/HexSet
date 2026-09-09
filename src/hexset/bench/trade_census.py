# SPDX-License-Identifier: GPL-3.0-only
"""Measure executed trades by bot label using the arena's recorded census.

Each exchange retains both resource bundles and each participant's private gain.
Summaries report card counts and trade participation, not shared economic utility.
Use --records to retain the games underlying the measurements.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import hexset.bots  # noqa: F401 -- registers heximax presets with hexset.arena
from hexset.arena import MAX_ACTIONS, Entrant, base_name, compete
from hexset.record import Record

# A hand of eight or more resource cards is subject to discarding on a seven.
LARGE_HAND_THRESHOLD = 8


@dataclass(frozen=True)
class TradeRecord:
    """An exchange with unsigned resource bundles in engine resource order.

    Hands are card counts at the start of the action step, as recorded by
    the arena. Gains retain each bot's own value units; comparisons between
    different evaluators do not imply a common measure of utility.
    """

    game: int
    turn: int
    phase: str
    seat_a: int
    seat_b: int
    name_a: str
    name_b: str
    given_a: tuple[int, ...]
    given_b: tuple[int, ...]
    hand_before_a: int
    hand_before_b: int
    gain_a: float
    gain_b: float


def _resource_split(received: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """`received` (signed towards `a`) as `(given_a, given_b)`, both unsigned."""
    given_a = tuple(max(0, -x) for x in received)
    given_b = tuple(max(0, x) for x in received)
    return given_a, given_b


@dataclass
class CensusResult:
    games: int
    entrant_names: tuple[str, ...] = ()
    trades: list[TradeRecord] = field(default_factory=list)
    winners: list[int | None] = field(default_factory=list)
    turns: list[int] = field(default_factory=list)
    # Terminal victory points, in entrant order, per game -- `hexset.arena.
    # Tournament.points`'s own convention, so a lineup's win rate and its VP
    # margin can be read off the same record.
    points: list[tuple[int, ...]] = field(default_factory=list)
    # One `Record` per game, in the same order as `winners`/`turns`/`points`
    # -- only when `run_census(records=True)` asked for them.
    records: list[Record] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "games": self.games,
            "entrant_names": self.entrant_names,
            "unfinished": sum(w is None for w in self.winners),
            "winners": self.winners,
            "turns": self.turns,
            "points": self.points,
            "trades": [asdict(t) for t in self.trades],
        }


def run_census(
    entrants: Sequence[Entrant],
    games: int,
    *,
    seed: int = 0,
    workers: int = 1,
    action_cap: int = MAX_ACTIONS,
    records: bool = False,
) -> CensusResult:
    """Play `games` games (a multiple of the lineup) and return every trade
    that cleared.

    The games are `hexset.arena.compete`'s, played with `records=True` so the
    tournament carries its own `ClearedTrade` census: turn, phase, both
    seats, the signed 5-vector, each side's hand at the top of the step and
    each side's private gain. This function is the rollup from that into
    `TradeRecord` rows -- which side gave what, under which bot's name --
    not a second way of playing the game.

    `records=True` additionally keeps the `hexset.record.Record` of every
    game in `CensusResult.records`; the games a `--records` file holds are
    exactly the games this census counted, since both come from the one job.
    """
    tournament = compete(
        list(entrants),
        games,
        seed=seed,
        action_cap=action_cap,
        workers=workers,
        records=True,
    )

    result = CensusResult(games=games, entrant_names=tuple(base_name(e.name) for e in entrants))
    result.winners = list(tournament.winners)
    result.turns = list(tournament.turns)
    result.points = list(tournament.points)
    if records:
        result.records = list(tournament.records)

    for index, (cleared, seating) in enumerate(
        zip(tournament.cleared, tournament.seating)
    ):
        # `seating[e]` is the seat entrant `e` took this game; the engine
        # reports a trade by seat, so invert it once per game.
        names: dict[int, str] = {
            seat: base_name(entrants[e].name) for e, seat in enumerate(seating)
        }
        for trade in cleared:
            given_a, given_b = _resource_split(trade.received)
            result.trades.append(
                TradeRecord(
                    game=index,
                    turn=trade.turn,
                    phase=trade.phase,
                    seat_a=trade.a,
                    seat_b=trade.b,
                    name_a=names[trade.a],
                    name_b=names[trade.b],
                    given_a=given_a,
                    given_b=given_b,
                    hand_before_a=trade.hand_a,
                    hand_before_b=trade.hand_b,
                    gain_a=trade.gain_a,
                    gain_b=trade.gain_b,
                )
            )
    return result


# ---------------------------------------------------------------------------
# Summaries


def _bundle_category(x: int, y: int) -> str:
    hi, lo = (x, y) if x >= y else (y, x)
    if (hi, lo) == (1, 1):
        return "1:1"
    if (hi, lo) == (2, 1):
        return "2:1"
    if (hi, lo) == (2, 2):
        return "2:2"
    if lo == 1 and hi >= 3:
        return "3+:1"
    return "other"


@dataclass
class BotSummary:
    name: str
    trade_sides: int  # number of (trade, side) rows contributing
    games_present: int
    seat_games: int
    seat_game_turns: int
    trade_sides_per_seat_game_turn: float
    bundle_distribution: dict
    bulk_share: float
    mean_given: float
    mean_received: float
    mean_imbalance: float
    large_hand_share: float
    mean_net_cards: float


def summarize(result: CensusResult) -> dict[str, BotSummary]:
    """Aggregate by base entrant label, retaining repeated seats in exposure.

    A trade contributes one participation per side, including two when the
    labels match. The rate divides participations by sum(game turns) times
    that label's seat count. It measures participation per occupied seat per
    elapsed game turn, not trades on that seat's own turns. Unfinished games
    contribute their observed turns and exchanges. Zero-trade labels remain.
    """
    if not result.entrant_names:
        raise ValueError("census needs the complete lineup, including repeated labels")
    if len(result.turns) != result.games:
        raise ValueError("census needs one turn count per game")
    counts = Counter(base_name(name) for name in result.entrant_names)
    bundle_counts: dict[str, Counter] = defaultdict(Counter)
    bulk_hits: dict[str, int] = defaultdict(int)
    given_totals: dict[str, list[int]] = defaultdict(list)
    received_totals: dict[str, list[int]] = defaultdict(list)
    imbalances: dict[str, list[float]] = defaultdict(list)
    dumps: dict[str, int] = defaultdict(int)
    swings: dict[str, list[float]] = defaultdict(list)
    rows: dict[str, int] = defaultdict(int)

    for t in result.trades:
        x = sum(t.given_a)
        y = sum(t.given_b)
        category = _bundle_category(x, y)
        is_bulk = max(x, y) >= 3
        imbalance = abs(x - y) / (x + y) if (x + y) else 0.0

        for name, given, received, is_dump_side in (
            (t.name_a, x, y, t.hand_before_a >= LARGE_HAND_THRESHOLD),
            (t.name_b, y, x, t.hand_before_b >= LARGE_HAND_THRESHOLD),
        ):
            rows[name] += 1
            bundle_counts[name][category] += 1
            if is_bulk:
                bulk_hits[name] += 1
            given_totals[name].append(given)
            received_totals[name].append(received)
            imbalances[name].append(imbalance)
            if is_dump_side:
                dumps[name] += 1
            swings[name].append(received - given)

    out: dict[str, BotSummary] = {}
    for name, seats in counts.items():
        n = rows[name]
        exposure = sum(result.turns) * seats
        out[name] = BotSummary(
            name=name,
            trade_sides=n,
            games_present=result.games,
            seat_games=result.games * seats,
            seat_game_turns=exposure,
            trade_sides_per_seat_game_turn=n / exposure if exposure else 0.0,
            bundle_distribution=dict(bundle_counts[name]),
            bulk_share=(bulk_hits[name] / n) if n else 0.0,
            mean_given=(statistics.mean(given_totals[name]) if n else 0.0),
            mean_received=(statistics.mean(received_totals[name]) if n else 0.0),
            mean_imbalance=(statistics.mean(imbalances[name]) if n else 0.0),
            large_hand_share=(dumps[name] / n) if n else 0.0,
            mean_net_cards=(statistics.mean(swings[name]) if n else 0.0),
        )
    return out


def table(summaries: dict[str, BotSummary]) -> str:
    header = (
        f"{'bot':<18}{'sides/seat/t':>12}{'mean give':>11}{'mean recv':>11}"
        f"{'imbalance':>11}{'bulk%':>8}{'hand8+%':>8}{'net cards':>11}"
    )
    lines = [header, "-" * len(header)]
    for name, s in summaries.items():
        lines.append(
            f"{name:<18}{s.trade_sides_per_seat_game_turn:>12.3f}{s.mean_given:>11.2f}"
            f"{s.mean_received:>11.2f}{s.mean_imbalance:>11.2f}{s.bulk_share*100:>7.1f}%"
            f"{s.large_hand_share*100:>7.1f}%{s.mean_net_cards:>+11.3f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bots", nargs="*", help="entrant names, e.g. heximax heximax heximax-notrade heximax-notrade")
    parser.add_argument("--games", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--records",
        default=None,
        help="append every game played as a v2 record (hexset.record.Record) "
        "here.",
    )
    args = parser.parse_args(argv)

    if not args.bots:
        parser.error("provide one entrant name per seat")
    if args.workers < 1:
        parser.error("--workers must be positive")
    from hexset.arena import lineup_from_names

    entrants = lineup_from_names(args.bots)
    result = run_census(
        entrants, args.games, seed=args.seed, workers=args.workers,
        records=bool(args.records),
    )
    summaries = summarize(result)

    print(table(summaries))
    if args.records:
        from hexset.record import write

        Path(args.records).parent.mkdir(parents=True, exist_ok=True)
        written = write(args.records, result.records)
        print(f"appended {written} records to {args.records}")
    if args.out:
        payload = {
            "census": result.to_json(),
            "summaries": {k: asdict(v) for k, v in summaries.items()},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
