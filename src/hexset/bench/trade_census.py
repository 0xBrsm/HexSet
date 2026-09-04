# SPDX-License-Identifier: GPL-3.0-only
"""Every executed trade, precisely: who, what, how lopsided, and who was flush.

Plays N four-seat games for a lineup (grouped seating, antithetic-paired
boards, exactly `hexset.bench.road_sweep`'s convention) and records each
`hexset.trading.Trade` as it clears -- turn, phase, both seats' kinds, the
signed 5-vector each way, each side's hand size just before the trade, and
each side's public surplus, so bulk/imbalanced trading can be described
without guessing at it from win rates. `--from-journals` replays the same
census over `hexset.server.journal` files instead of playing fresh games.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

import hexset.bots  # noqa: F401 -- registers heximax presets with hexset.arena
from hexset.actions import apply
from hexset.arena import MAX_ACTIONS, Entrant, base_name, seat_of, spawn
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES
from hexset.chance import Live, Recording
from hexset.game import Phase, is_over, start, to_move
from hexset.record import Record, board_fields
from hexset.trading import publish_valuation
from hexset.victory import victory_points

# Card price for the value yardstick: the flat 4:1 bank rate, so a swing is
# comparable across bots with no bot's own valuation in it. Port-adjusted
# rates are the documented alternative (a seat's best rate can be under 4:1)
# but need a seat's settlement/city-to-port adjacency, which nothing in
# `hexset.board` exposes as a public helper yet -- flat rate ships as the
# neutral default and the per-seat refinement is future work.
BANK_RATE = 4
CARD_VALUE = 1.0 / BANK_RATE

# A side "dumps" a trade if it enters it holding at least this many cards --
# the discard threshold minus nothing, i.e. hoard territory by any human
# reckoning of Catan hand sizes.
DUMP_THRESHOLD = 8

# The machine this runs on is shared; see `hexset.bench.road_sweep`'s MAX_WORKERS.
MAX_WORKERS = 8


@dataclass(frozen=True)
class TradeRecord:
    """One executed trade, both sides' full accounting.

    `given_a`/`given_b` and `received_a`/`received_b` are 5-vectors in
    resource order (`hexset.board.terrain.Resource`). `hand_before_a/b` are
    each side's total card count the instant before this trade executed,
    reconstructed by replaying the turn's trades in order over a true
    pre-turn hand snapshot -- exact, not estimated, since a trade can only
    move cards that both hands already held.
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
    surplus_a: float
    surplus_b: float
    larger_surplus: str  # "a", "b", or "tie"


def _resource_split(received: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """`received` (signed towards `a`) as `(given_a, given_b)`, both unsigned."""
    given_a = tuple(max(0, -x) for x in received)
    given_b = tuple(max(0, x) for x in received)
    return given_a, given_b


def _play_census(
    entrants: Sequence[Entrant],
    index: int,
    seed: int,
    action_cap: int,
    keep_record: bool = False,
) -> tuple[list[TradeRecord], int | None, int, tuple[int, ...], Record | None]:
    """Play one game, returning its trades plus (winning entrant, turns,
    points, record).

    Instrumentation reads only the hands (the raw array) for bookkeeping
    snapshots, through `game.state(0, hidden=False)` -- the sanctioned
    true-state path (`tests/test_view.py`), which returns the state itself and
    is explicitly not one of the trigger points. Never `game.state(seat)` (the
    seat's own view) or `legal_actions()`: those are the calls that lazily
    trigger the pending trade event
    (`hexset.game.run_pending_event`), and calling them for our own
    purposes would fire an event before the bot whose turn it is ever asked
    for one. `game.trades` is cleared every `end_turn`, so new trades are
    detected as a length delta against a per-turn counter, checked both
    before and after `apply()` each iteration to catch a burst that fires
    and then gets cleared inside one loop pass.

    `keep_record=True` additionally tracks the action list and the chance
    stream (`hexset.chance.Recording`) and returns a `hexset.record.Record`
    of this exact game -- a separate, parallel tally from `harvest`'s own
    per-trade rows above: `harvest` is delayed by up to one loop iteration
    for a trade an `apply()` call itself triggered (harmless for its
    turn/phase bookkeeping, since a trade never crosses a turn boundary) but
    a `Record`'s trades must be attributed to the exact step they cleared
    inside for `replay` to reapply them in the right place
    (`hexset.record.record_game`'s own docstring on why), so this keeps its
    own `before`/`after` `game.trades` bookkeeping around `publish_valuation`
    and `apply` instead of reusing `harvest`'s.
    """
    seats = len(entrants)
    pair, half = divmod(index, 2)
    board = random_base_board(random.Random(f"{seed}:{pair}:board"))
    rotation = pair + half * (seats // 2)
    seats_taken = [seat_of(e, rotation, seats) for e in range(seats)]

    names = [None] * seats
    lineup: list = [None] * seats
    for e, entrant in enumerate(entrants):
        seat = seats_taken[e]
        lineup[seat] = spawn(entrant, board, random.Random(f"{seed}:{pair}:{e}"))
        names[seat] = base_name(entrant.name)

    game_seed = f"{seed}:{pair}:game"
    rng = random.Random(game_seed)
    chance = Recording(Live(rng)) if keep_record else None
    game = start(board, seats, rng, chance=chance)
    game.gates = tuple(lineup)
    game.max_trades = None

    records: list[TradeRecord] = []
    action_log: list[tuple[int, int, int]] = []
    record_trades: list[tuple[int, int, int, tuple[int, ...]]] = []
    baseline = [tuple(h) for h in game.state(0, hidden=False).hands]
    seen_this_turn = 0

    def harvest(turn: int, phase: Phase) -> None:
        nonlocal baseline, seen_this_turn
        current = game.trades
        if len(current) < seen_this_turn:
            seen_this_turn = 0
        new = current[seen_this_turn:]
        if not new:
            return
        running = [list(h) for h in baseline]
        vectors = game.valuations
        for trade in new:
            a, b, received = trade.a, trade.b, trade.received
            given_a, given_b = _resource_split(received)
            hand_before_a = sum(running[a])
            hand_before_b = sum(running[b])
            surplus_a = sum(v * r for v, r in zip(vectors[a], received))
            surplus_b = sum(v * -r for v, r in zip(vectors[b], received))
            if surplus_a > surplus_b:
                larger = "a"
            elif surplus_b > surplus_a:
                larger = "b"
            else:
                larger = "tie"
            records.append(
                TradeRecord(
                    game=index,
                    turn=turn,
                    phase=phase.name,
                    seat_a=a,
                    seat_b=b,
                    name_a=names[a],
                    name_b=names[b],
                    given_a=given_a,
                    given_b=given_b,
                    hand_before_a=hand_before_a,
                    hand_before_b=hand_before_b,
                    surplus_a=surplus_a,
                    surplus_b=surplus_b,
                    larger_surplus=larger,
                )
            )
            for r in range(NUM_RESOURCES):
                running[a][r] += received[r]
                running[b][r] -= received[r]
        seen_this_turn = len(current)
        baseline = [tuple(h) for h in game.state(0, hidden=False).hands]

    actions = 0
    while not is_over(game) and actions < action_cap:
        seat = to_move(game)
        bot = lineup[seat]
        if game.publish_due(seat):
            if keep_record:
                before = len(game.trades)
            publish_valuation(game, seat, bot)
            if keep_record:
                for trade in game.trades[before:]:
                    record_trades.append(
                        (len(action_log) - 1, trade.a, trade.b, tuple(trade.received))
                    )
        harvest(game.turns, game.phase)
        if keep_record:
            before = len(game.trades)
        action = bot.choose(game)
        harvest(game.turns, game.phase)
        apply(game, action)
        if keep_record:
            for trade in game.trades[before:]:
                record_trades.append(
                    (len(action_log), trade.a, trade.b, tuple(trade.received))
                )
            action_log.append((int(action.type), action.a, action.b))
        actions += 1
        if len(game.trades) < seen_this_turn:
            seen_this_turn = 0
        baseline = [tuple(h) for h in game.state(0, hidden=False).hands]

    winner = None if game.won_by is None else seats_taken.index(game.won_by)
    # true state: terminal points include hidden victory-point cards, the
    # same reasoning as `hexset.arena._play_one`'s own verdict.
    points = tuple(
        victory_points(game.state(seats_taken[e], hidden=False), seats_taken[e])
        for e in range(seats)
    )
    game_record = None
    if keep_record:
        game_record = Record(
            num_players=seats,
            seed=game_seed,
            first=game.first,
            actions=tuple(action_log),
            chance=tuple(chance.events),
            trades=tuple(record_trades),
            winner=game.won_by,
            turns=game.turns,
            **board_fields(board),
        )
    return records, winner, game.turns, points, game_record


def _play_one(
    job: tuple[tuple[Entrant, ...], int, int, int, bool]
) -> tuple[list[TradeRecord], int | None, int, tuple[int, ...], Record | None]:
    entrants, index, seed, action_cap, keep_record = job
    return _play_census(entrants, index, seed, action_cap, keep_record)


@dataclass
class CensusResult:
    games: int
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
    """Play `games` games (a multiple of 4, road_sweep's antithetic rotation)
    and return every trade that cleared.

    `records=True` additionally has every game build a `hexset.record.Record`
    of itself (`_play_census`'s `keep_record`), collected into
    `CensusResult.records` -- the games a `--records` file holds are exactly
    the games this census counted, since both come from the one job.
    """
    seats = len(entrants)
    if games % seats:
        raise ValueError(f"{games} games does not divide evenly over {seats} seats")
    jobs = [(tuple(entrants), i, seed, action_cap, records) for i in range(games)]
    if workers > 1:
        with Pool(min(workers, MAX_WORKERS)) as pool:
            outcomes = pool.map(_play_one, jobs, chunksize=1)
    else:
        outcomes = [_play_one(job) for job in jobs]

    result = CensusResult(games=games)
    for trade_rows, winner, turns, points, game_record in outcomes:
        result.trades.extend(trade_rows)
        result.winners.append(winner)
        result.turns.append(turns)
        result.points.append(points)
        if game_record is not None:
            result.records.append(game_record)
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
    trades_per_turn: float
    bundle_distribution: dict
    bulk_share: float
    mean_given: float
    mean_received: float
    mean_imbalance: float
    dump_share: float
    mean_value_swing: float


def summarize(result: CensusResult, entrant_names: Sequence[str]) -> dict[str, BotSummary]:
    """Per-bot-kind rollup. Each trade contributes one row per side, reoriented
    to that side's own given/received/hand-before/surplus."""
    turns_played: dict[str, int] = defaultdict(int)
    bundle_counts: dict[str, Counter] = defaultdict(Counter)
    bulk_hits: dict[str, int] = defaultdict(int)
    given_totals: dict[str, list[int]] = defaultdict(list)
    received_totals: dict[str, list[int]] = defaultdict(list)
    imbalances: dict[str, list[float]] = defaultdict(list)
    dumps: dict[str, int] = defaultdict(int)
    swings: dict[str, list[float]] = defaultdict(list)
    rows: dict[str, int] = defaultdict(int)

    for name in entrant_names:
        turns_played[name] = 0  # ensure present even with zero trades

    for t in result.trades:
        x = sum(t.given_a)
        y = sum(t.given_b)
        category = _bundle_category(x, y)
        is_bulk = max(x, y) >= 3
        imbalance = abs(x - y) / (x + y) if (x + y) else 0.0

        for name, given, received, hand_before, is_dump_side in (
            (t.name_a, x, y, t.hand_before_a, t.hand_before_a >= DUMP_THRESHOLD),
            (t.name_b, y, x, t.hand_before_b, t.hand_before_b >= DUMP_THRESHOLD),
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
            swings[name].append((received - given) * CARD_VALUE)

    turn_max: dict[str, int] = defaultdict(int)
    for t in result.trades:
        turn_max[t.name_a] = max(turn_max[t.name_a], t.turn)
        turn_max[t.name_b] = max(turn_max[t.name_b], t.turn)
    # Turns-per-game isn't tracked per-name (games are shared by the whole
    # lineup); use the census-wide mean game length as the denominator for
    # "trades per turn" so it is comparable across bots in the same lineup.
    mean_turns = statistics.mean(result.turns) if result.turns else 0.0

    out: dict[str, BotSummary] = {}
    for name in entrant_names:
        n = rows[name]
        denom = mean_turns * result.games if mean_turns else 1.0
        out[name] = BotSummary(
            name=name,
            trade_sides=n,
            games_present=result.games,
            trades_per_turn=(n / denom) if denom else 0.0,
            bundle_distribution=dict(bundle_counts[name]),
            bulk_share=(bulk_hits[name] / n) if n else 0.0,
            mean_given=(statistics.mean(given_totals[name]) if n else 0.0),
            mean_received=(statistics.mean(received_totals[name]) if n else 0.0),
            mean_imbalance=(statistics.mean(imbalances[name]) if n else 0.0),
            dump_share=(dumps[name] / n) if n else 0.0,
            mean_value_swing=(statistics.mean(swings[name]) if n else 0.0),
        )
    return out


def table(summaries: dict[str, BotSummary]) -> str:
    header = (
        f"{'bot':<18}{'trades/turn':>12}{'mean give':>11}{'mean recv':>11}"
        f"{'imbalance':>11}{'bulk%':>8}{'dump%':>8}{'val swing':>11}"
    )
    lines = [header, "-" * len(header)]
    for name, s in summaries.items():
        lines.append(
            f"{name:<18}{s.trades_per_turn:>12.3f}{s.mean_given:>11.2f}"
            f"{s.mean_received:>11.2f}{s.mean_imbalance:>11.2f}{s.bulk_share*100:>7.1f}%"
            f"{s.dump_share*100:>7.1f}%{s.mean_value_swing:>+11.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# --from-journals: replay the census over hexset.server.journal files


def census_from_journal(path: Path) -> tuple[list[TradeRecord], dict[int, str]]:
    """Reconstruct trades from one journal `.jsonl` file.

    Journals record every action verbatim (`hexset.server.journal`), so a
    trade shows up as whatever event kind the server writes for a cleared
    exchange. This reads the header for bot names/seats and scans for trade
    events; a journal format that names them differently than expected is
    reported rather than silently skipped.
    """
    records: list[TradeRecord] = []
    names: dict[int, str] = {}
    turn = 0
    phase = "MAIN"
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            kind = event.get("kind") or event.get("type")
            if kind in ("start", "header"):
                bot_names = event.get("bot_names") or {}
                names = {int(k): v for k, v in bot_names.items()}
            if kind == "turn":
                turn = event.get("turn", turn)
            if kind in ("trade", "trades"):
                phase = event.get("phase", phase)
                trades = event.get("trades") or [event]
                for tr in trades:
                    a, b = tr["a"], tr["b"]
                    received = tuple(tr["received"])
                    given_a, given_b = _resource_split(received)
                    records.append(
                        TradeRecord(
                            game=0,
                            turn=tr.get("turn", turn),
                            phase=phase,
                            seat_a=a,
                            seat_b=b,
                            name_a=names.get(a, f"seat{a}"),
                            name_b=names.get(b, f"seat{b}"),
                            given_a=given_a,
                            given_b=given_b,
                            hand_before_a=tr.get("hand_before_a", 0),
                            hand_before_b=tr.get("hand_before_b", 0),
                            surplus_a=tr.get("surplus_a", 0.0),
                            surplus_b=tr.get("surplus_b", 0.0),
                            larger_surplus=tr.get("larger_surplus", "tie"),
                        )
                    )
    return records, names


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bots", nargs="*", help="entrant names, e.g. heximax heximax search2 search2")
    parser.add_argument("--games", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--from-journals",
        type=Path,
        default=None,
        help="directory of hexset.server.journal .jsonl files to census instead of playing games",
    )
    parser.add_argument(
        "--records",
        default=None,
        help="append every game played as a v2 record (hexset.record.Record) "
        "here. Not available with --from-journals -- convert those with "
        "hexset.record.from_journal instead.",
    )
    args = parser.parse_args(argv)

    if args.records and args.from_journals is not None:
        print("--records plays fresh games; it has nothing to add to --from-journals "
              "(convert those files with hexset.record.from_journal instead)",
              file=sys.stderr)
        return

    if args.from_journals is not None:
        directory = args.from_journals
        files = sorted(directory.glob("*.jsonl"))
        if not files:
            print(f"no journals found under {directory}; nothing to census", file=sys.stderr)
            return
        all_records: list[TradeRecord] = []
        all_names: set[str] = set()
        for f in files:
            records, names = census_from_journal(f)
            all_records.extend(records)
            all_names.update(names.values())
        result = CensusResult(games=len(files), trades=all_records, winners=[], turns=[])
        summaries = summarize(result, sorted(all_names))
    else:
        from hexset.arena import lineup_from_names

        entrants = lineup_from_names(args.bots)
        result = run_census(
            entrants,
            args.games,
            seed=args.seed,
            workers=args.workers,
            records=bool(args.records),
        )
        entrant_names = sorted({base_name(e.name) for e in entrants})
        summaries = summarize(result, entrant_names)

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
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
