# SPDX-License-Identifier: GPL-3.0-only
"""Measure executed trades by bot label using the arena's recorded census.

Summaries report card counts and trade participation, not shared utility.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from hexset.arena import MAX_ACTIONS, RUNTIME_HELP, Entrant, base_name, compete, load_runtime
from hexset.bench.duel import add_table_options
from hexset.game import MAX_TURNS
from hexset.record import Record
from hexset.rules import STANDARD_GAME, GameType, game_type as game_type_named

__all__ = [
    "LARGE_HAND_THRESHOLD",
    "TradeRecord",
    "CensusResult",
    "run_census",
    "BotSummary",
    "summarize",
    "table",
    "main",
]


# A hand of eight or more resource cards is subject to discarding on a seven.
LARGE_HAND_THRESHOLD = 8


@dataclass(frozen=True)
class TradeRecord:
    """An exchange with unsigned resource bundles in engine resource order.

    Hands are each side's resource-card count just before the exchange: at
    the start of its action step, moved by any earlier exchange in the same
    step (`hexset.arena.ClearedTrade`). Gains are in each bot's own value
    units and are not comparable across evaluators.
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
    """What `run_census` collected: every executed trade (`TradeRecord`),
    and per game the winner, turns, terminal points and, with `records=True`,
    the `Record`."""

    games: int
    entrant_names: tuple[str, ...] = ()
    trades: list[TradeRecord] = field(default_factory=list)
    winners: list[int | None] = field(default_factory=list)
    turns: list[int] = field(default_factory=list)
    # Terminal victory points per game, in entrant order, as `Tournament.points`.
    points: list[tuple[int, ...]] = field(default_factory=list)
    # One `Record` per game, ordered with `winners`; only under `records=True`.
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
    journal: str | Path | None = None,
    resume: bool = False,
    runtimes: Sequence[str] = (),
    game_type: GameType = STANDARD_GAME,
    turn_cap: int = MAX_TURNS,
    trade_mode: str = "round",
) -> CensusResult:
    """Play `games` games (a multiple of the lineup) and roll `compete`'s
    `ClearedTrade` census up into `TradeRecord` rows; `records=True` also keeps
    each game's `Record` in `CensusResult.records`. `runtimes` are loaded in
    every worker (`hexset.arena.load_runtime`); `game_type`, `turn_cap` and
    `trade_mode` are `compete`'s.
    """
    tournament = compete(
        list(entrants),
        games,
        seed=seed,
        action_cap=action_cap,
        workers=workers,
        records=True,
        worker_initializer=load_runtime if runtimes else None,
        worker_initargs=tuple(runtimes),
        game_type=game_type,
        turn_cap=turn_cap,
        trade_mode=trade_mode,
        journal=journal,
        resume=resume,
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
        # `seating[e]` is the seat entrant `e` took; trades are reported by seat.
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
    """One label's exchanges, aggregated over the seats it occupied.

    `seat_game_turns`, occupied seats times summed elapsed game turns, is the
    denominator of `trade_sides_per_seat_game_turn`. `mean_net_cards` is
    received minus given cards per participation -- a count, not a value.
    `large_hand_share` counts participations holding `LARGE_HAND_THRESHOLD`
    cards just before the exchange, as `TradeRecord` counts hands: a
    snapshot that may precede the step's production, though not an earlier
    exchange in it.
    """

    name: str
    trade_sides: int  # (trade, side) rows contributing
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

    A trade contributes one participation per side, two when labels match;
    unfinished games contribute their observed turns.
    """
    if not result.entrant_names:
        raise ValueError("census needs the complete lineup, including repeated labels")
    if len(result.turns) != result.games:
        raise ValueError("census needs one turn count per game")
    counts = Counter(base_name(name) for name in result.entrant_names)
    bundle_counts: dict[str, Counter] = defaultdict(Counter)
    totals: dict[str, Counter] = defaultdict(Counter)

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
            bundle_counts[name][category] += 1
            totals[name].update(sides=1, given=given, received=received,
                                imbalance=imbalance, bulk=is_bulk, large_hand=is_dump_side)

    out: dict[str, BotSummary] = {}
    for name, seats in counts.items():
        total = totals[name]
        n = total["sides"]
        divisor = n or 1
        exposure = sum(result.turns) * seats
        out[name] = BotSummary(
            name=name,
            trade_sides=n,
            games_present=result.games,
            seat_games=result.games * seats,
            seat_game_turns=exposure,
            trade_sides_per_seat_game_turn=n / exposure if exposure else 0.0,
            bundle_distribution=dict(bundle_counts[name]),
            bulk_share=total["bulk"] / divisor,
            mean_given=total["given"] / divisor,
            mean_received=total["received"] / divisor,
            mean_imbalance=total["imbalance"] / divisor,
            large_hand_share=total["large_hand"] / divisor,
            mean_net_cards=(total["received"] - total["given"]) / divisor,
        )
    return out


def table(summaries: dict[str, BotSummary]) -> str:
    """`summaries` as a fixed-width text table, one row per bot."""
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
    """`python -m hexset.bench.trade_census`: measure executed trades by bot
    label and print the summary table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bots", nargs="*", help="one entrant name per seat")
    parser.add_argument("--runtime", action="append", default=[], help=RUNTIME_HELP)
    parser.add_argument("--games", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    add_table_options(parser)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--journal", type=Path, default=None,
                        help="where each game is kept as it finishes (default beside --out, "
                             "else under runs/census)")
    parser.add_argument("--resume", action="store_true",
                        help="continue the journal's run, playing only the games it lacks")
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

    load_runtime(*args.runtime)
    entrants = lineup_from_names(args.bots)
    journal = args.journal or (
        args.out.with_name(f"{args.out.stem}.s{args.seed}.g{args.games}.games.jsonl")
        if args.out else Path("runs/census") / f"{'-'.join(args.bots)}.s{args.seed}.g{args.games}.games.jsonl")
    result = run_census(
        entrants, args.games, seed=args.seed, workers=args.workers,
        records=bool(args.records), journal=journal, resume=args.resume,
        runtimes=args.runtime, game_type=game_type_named(args.game_type),
        turn_cap=args.turn_cap, trade_mode=args.trade_mode,
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
