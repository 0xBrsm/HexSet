# SPDX-License-Identifier: GPL-3.0-only
"""The trade event lifted out of the game: an ablation over the engine's
three selection rules (`actor`, `egalitarian`, `nash`), and the paired-
chance judge that measures the shipped gate's resolution.

Registration: `agents/reference/trade-lab.md` (dev-hexset), `agents/
reference/trading-final.md` items 3-4. Ported onto contract 6
(`agents/reference/trading-final.md`): there is no public valuation layer
any more (`Game.valuations`, `Bot.valuation`, `publish_valuation` are gone),
`Bot.gains_many` is every seat's whole trading surface, and the engine
itself now implements `actor`/`egalitarian`/`nash` selection
(`hexset.trading._best_clearing`, `Game.trade_rule`) -- this module no
longer has to reimplement the shipped rule's own tie-break, only walk the
same primitives (`trading._candidates`, a seat's `gains_many`/`_delta`) to
compare rules against each other and to build the phase-3 judge.

**Phase 1-2 recap** (`trade-lab.md`'s own post-data notes): the public
vector vetoed four mutually-acceptable trades in five; honesty is a fixed
point under `egalitarian` and not strictly under `nash` (an exaggeration
incentive of about 3% of the per-event gain, economically nil); the
rollout judge (300 trades/rule, independent playouts before/after) ran 8h
against a ~40-min projection with no output and was killed -- no progress
output, no partial writes, a 20,000-action cap, six playouts a side too few
to resolve a 3e-4 effect.

**Phase 3** replaces the rollout judge with a paired-chance design on the
new engine: for a sampled pre-event position, one throwaway continuation
records a chance-event script (`hexset.chance.Recording(Live(...))`), and
both the *untraded* continuation (the event suppressed for one turn only)
and the *traded* continuation (the historical clearing applied outright)
replay against that same script -- identical dice, steals and draws -- for
up to `--cap` actions each. One stream is one such paired playout; several
streams (`--streams`) per position substitute for the old design's many
independent playouts a side, at a fraction of the cost, because the two
forks of one stream differ only in the cards the historical trade moved,
not in the dice either one draws.

The script is replayed by *kind*, independently per kind
(`_PairedChance`), not as one strictly-ordered sequence
(`hexset.chance.Scripted`): the two forks' hands already differ (by the
historical trade), so they diverge in *which* action comes next almost
immediately -- whether a knight is played before or after rolling decides
whether a steal or a roll comes first -- and a single ordered script
would read that reordering as a divergence and end the pairing within the
first turn or two, on nearly every stream, as this run's own smoke test
found. Reading each kind from its own queue means the fork that has
rolled five times so far gets the fifth recorded roll regardless of
what fell between the fourth and fifth roll on the *other* fork's path.
A fork that needs more of some kind than the throwaway script recorded
(`ChanceExhausted`) falls back to a freshly seeded `Live` for the
remainder and is counted in the row, per the registration.

**Why a bank of records, not live games** (unchanged from phase 1, restated
briefly): no bulk game records existed before this lab; `hexset.record`
replays a `Record` forever, so the bank is built once and replayed as many
times as a later phase needs. See phase 1's own note in `trade-lab.md` for
the full accounting.

**Where a position is captured** (unchanged mechanism, phase 1's own
module docstring): `hexset.game.run_trade_event` is monkeypatched (a name
looked up fresh, as a module global, by every action function that calls
it unqualified) to snapshot the game immediately before delegating to the
original. Positions now also record whether the captured event was this
turn's *first* (`main_entry`) and what the original, historical call
actually cleared (`historical_trades`) -- the phase-3 judged set is exactly
the sub-sequence where both hold: a MAIN-entry event under the engine's own
default rule (`egalitarian`) that cleared at least one trade.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

import hexset.bots  # noqa: F401 -- registers "heximax" with hexset.arena
import hexset.game as _game_mod
from hexset import trading
from hexset.actions import apply
from hexset.arena import mean_interval
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES
from hexset.bots.heximax import heximax
from hexset.bots.heximax.search import Heximax, _thin_copy
from hexset.chance import Chance, ChanceExhausted, Live, Recording
from hexset.game import Game, Phase, imagine, is_over, start, to_move
from hexset.record import Record, ReplayError, actions_of, board_of, from_json, read, record_game, write
from hexset.trading import Bundle, Trade
from hexset.victory import victory_points

# The machine this runs on is shared; see `hexset.bench.trade_census`'s own
# `MAX_WORKERS` for the convention this mirrors.
MAX_WORKERS = 8

# The engine's own selectable rules (`hexset.trading.TRADE_RULES`), in the
# order this module has always reported them. `maximin-public` is gone with
# the public layer it filtered on (`trading-final.md`); every bank position
# is now recorded and replayed under the shipped default, `egalitarian`.
RULES: tuple[str, ...] = ("actor", "egalitarian", "nash")

# The human-corpus bundle-shape figures the registration measures against
# (`trade-lab.md`, "bundle shape table ... beside the human corpus"), in the
# same five bins `_bundle_shape` below assigns a trade to.
HUMAN_CORPUS_SHAPE: dict[str, float] = {
    "1:1": 0.634,
    "1:2": 0.286,
    "2:2": 0.040,
    "≥3 a side": 0.040,
    ">3 one way": 0.009,
}


# ---------------------------------------------------------------------------
# The interception: snapshot every position a trade event is about to clear,
# tagging whether it was this turn's first event and what it historically
# cleared.
#
# `_capture_sink` is `None` outside of `positions()` (the default, and the
# state every worker process starts in): the patched wrapper then does
# nothing but call the original function straight through, so `bank()`'s own
# games -- and every hypothetical child a bot's tree search spawns via
# `imagine()`, which never carries real `gates` -- pay only the `is not
# None`/`is not None` checks below, not a snapshot.

_capture_sink: list[tuple[Game, bool, tuple[Trade, ...]]] | None = None
_capture_last_turn: int | None = None


def _snapshot_position(game: Game) -> Game:
    """An inert copy of `game`, gates and all, safe to read but never to
    advance: `imagine` is the engine's sanctioned copy (a fresh `GameState`,
    a fresh ledger, a fresh `Live` chance -- none of which this snapshot
    ever draws from), so `gates` is set here to the *same* bot objects
    `game` is seated with -- freshly spawned once per replayed game
    (`positions()`, below), not once per position, and safe to share across
    every position from that one replay since `Heximax.gains_many`/`_delta`
    are pure reads of a view, never of `self.rng`.
    """
    copy = imagine(game, random.Random(0))
    copy.gates = game.gates
    return copy


_original_run_trade_event = _game_mod.run_trade_event


def _capturing_run_trade_event(game: Game) -> None:
    global _capture_last_turn
    if _capture_sink is not None and game.phase is Phase.MAIN and game.gates is not None:
        main_entry = game.turns != _capture_last_turn
        _capture_last_turn = game.turns
        snapshot = _snapshot_position(game)
        before = len(game.trades)
        _original_run_trade_event(game)
        cleared = tuple(game.trades[before:])
        _capture_sink.append((snapshot, main_entry, cleared))
        return
    return _original_run_trade_event(game)


_game_mod.run_trade_event = _capturing_run_trade_event


# ---------------------------------------------------------------------------
# Bank: play and record heximax x4 games.


def _spawn_bots(seed: int, num_players: int, board) -> list[Heximax]:
    """The per-seat construction `bank()` and `positions()` must agree on
    byte-for-byte -- same board, same seed string per seat -- for a replay's
    respawned bots to make the exact decisions the recorded ones made."""
    return [heximax(board, random.Random(f"{seed}:{seat}")) for seat in range(num_players)]


def _bank_job(job: tuple[int, int]) -> Record:
    index, seed = job
    del index  # the seed alone determines the board and every bot's stream
    board = random_base_board(random.Random(f"{seed}:board"))
    bots = _spawn_bots(seed, 4, board)
    return record_game(bots, board, seed)


def run_bank(games: int, seed0: int, out_path: Path, workers: int) -> int:
    """Play `games` heximax x4 games, seeds `seed0`..`seed0 + games - 1`,
    each its own board, and write them to `out_path` as `Record`s (one JSON
    line a game, truncating whatever was there before)."""
    jobs = [(i, seed0 + i) for i in range(games)]
    if workers > 1:
        with Pool(min(workers, MAX_WORKERS)) as pool:
            records = pool.map(_bank_job, jobs, chunksize=1)
    else:
        records = [_bank_job(job) for job in jobs]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    return write(str(out_path), records)


# ---------------------------------------------------------------------------
# Positions: replay a record to every point a trade event fires.


@dataclass
class Position:
    game_index: int
    position_index: int
    turn: int
    actor: int
    game: Game
    # Whether this was this turn's *first* trade event (fired from
    # `hexset.game.enter_main`, before any MAIN action) rather than one of
    # the events that follow every later MAIN action the same turn.
    main_entry: bool
    # What the historical event -- seated with the game's own bots, the
    # engine's shipped `egalitarian` rule -- actually cleared here, in
    # order. Empty when nothing cleared.
    historical_trades: tuple[Trade, ...]


def positions(record: Record, *, game_index: int = 0):
    """Every position `record` passed through right before a trade event
    fired, replayed by respawning the same bots and driving the recorded
    actions -- see the module docstring for why this reproduces the
    original trades without any tree search. Raises `hexset.record.
    ReplayError` if it does not: a wrong winner/turn count, or a trade
    sequence that does not match what the bank recorded, means the bots
    were not respawned identically (or something about them changed since
    the bank was written) and the position bank is not trustworthy.
    """
    global _capture_sink, _capture_last_turn
    board = board_of(record)
    bots = _spawn_bots(record.seed, record.num_players, board)
    game = start(board, record.num_players, random.Random(record.seed))
    game.gates = tuple(bots)
    game.max_trades = None

    sink: list[tuple[Game, bool, tuple[Trade, ...]]] = []
    _capture_sink = sink
    _capture_last_turn = None
    # `game.trades` is reset every turn (`end_turn`) -- a per-turn scratch
    # list, not a whole-game log -- so the fidelity check below diffs it
    # around every action rather than reading it once at the end. There is
    # no separate publish step any more (`Bot.gains_many` is asked from
    # inside `apply` itself, via each action function's own trailing
    # `run_trade_event`), so driving the recorded actions is the whole of
    # replay.
    replayed: list[tuple[int, int, tuple[int, ...]]] = []
    try:
        for action in actions_of(record):
            before = len(game.trades)
            apply(game, action)
            for t in game.trades[before:]:
                replayed.append((t.a, t.b, tuple(t.received)))
    finally:
        _capture_sink = None
        _capture_last_turn = None

    if (game.won_by, game.turns) != (record.winner, record.turns):
        raise ReplayError(
            f"trade_lab replay of game {game_index} ended {game.won_by} after "
            f"{game.turns} turns, the bank says {record.winner} after {record.turns}"
        )
    recorded = [(a, b, tuple(r)) for _step, a, b, r in record.trades]
    if replayed != recorded:
        raise ReplayError(
            f"trade_lab replay of game {game_index} re-cleared "
            f"{len(replayed)} trades; the bank recorded {len(recorded)}"
        )

    for i, (snapshot, main_entry, historical_trades) in enumerate(sink):
        yield Position(
            game_index=game_index,
            position_index=i,
            turn=snapshot.turns,
            actor=snapshot.current_player,
            game=snapshot,
            main_entry=main_entry,
            historical_trades=historical_trades,
        )


def judged_positions(record: Record, *, game_index: int = 0):
    """The phase-3 judged set at one game: positions before a MAIN-entry
    event where the engine's own shipped rule (`egalitarian`, the bank's
    recording default) cleared at least one trade."""
    for pos in positions(record, game_index=game_index):
        if pos.main_entry and pos.historical_trades:
            yield pos


# ---------------------------------------------------------------------------
# Candidate rows: every coverable candidate, both private gains, unfiltered.


@dataclass(frozen=True)
class CandidateRow:
    them: int
    bundle: Bundle
    gain_actor: float
    gain_counterparty: float


def candidate_rows(game: Game, me: int) -> list[CandidateRow]:
    """Every candidate `trading._candidates` enumerates for `me`, with both
    private gains (`Heximax._delta`, the gate's own reading, never a
    boolean accept/reject) unfiltered by either side: `_best_clearing`
    already discards a non-positive candidate before a rule ever gets to
    see it, so this walks `_candidates`'s raw output directly instead."""
    state = game.state(0, hidden=False)
    raw = list(trading._candidates(state, me, game.locked))
    if not raw:
        return []

    views: dict[int, object] = {}

    def view(seat: int):
        got = views.get(seat)
        if got is None:
            got = game.state(seat)
            views[seat] = got
        return got

    rows = []
    for them, bundle in raw:
        bot_me = game.gates[me]
        bot_them = game.gates[them]
        gain_me = bot_me._delta(view(me), me, me, bundle, them, bot_me._rank)
        mirror = tuple(-n for n in bundle)
        gain_them = bot_them._delta(view(them), them, them, mirror, me, bot_them._rank)
        rows.append(CandidateRow(them, bundle, gain_me, gain_them))
    return rows


def _canonical(row: CandidateRow) -> tuple[int, ...]:
    """Negated bundle: maximizing this picks the smallest bundle on a tie,
    the same trick `hexset.trading._best_clearing` uses."""
    return tuple(-n for n in row.bundle)


def clearing_set(rule: str, rows: Sequence[CandidateRow]) -> list[CandidateRow]:
    """Candidates both private gates strictly accept. The same set for
    every rule now that there is no public pre-filter left (`trading-
    final.md`): the mechanic's one approximation is the gate itself, and a
    rule only chooses among what already clears both gates."""
    if rule not in RULES:
        raise ValueError(f"unknown rule: {rule}")
    return [r for r in rows if r.gain_actor > 0 and r.gain_counterparty > 0]


def select(rule: str, clearing: Sequence[CandidateRow]) -> CandidateRow | None:
    """The clearing candidate `rule` picks -- `clearing` is already the
    output of `clearing_set(rule, ...)`. Mirrors the engine's own tie-break
    (`hexset.trading._best_clearing`): ties break on the actor's own gain,
    then canonical bundle order, then the lower counterparty seat, purely
    for determinism."""
    if not clearing:
        return None
    if rule == "actor":
        key = lambda r: (r.gain_actor, r.gain_actor, _canonical(r), -r.them)
    elif rule == "egalitarian":
        key = lambda r: (min(r.gain_actor, r.gain_counterparty), r.gain_actor, _canonical(r), -r.them)
    elif rule == "nash":
        key = lambda r: (r.gain_actor * r.gain_counterparty, r.gain_actor, _canonical(r), -r.them)
    else:
        raise ValueError(f"unknown rule: {rule}")
    return max(clearing, key=key)


def _bystander_delta(bot, view, bystander: int, mover: int, counterparty: int, received: Bundle) -> float:
    """A non-party's own reading of a trade it took no part in: the change
    in `bystander`'s own row, read entirely through `bystander`'s own
    belief, before and after `mover` receives `received` from
    `counterparty`.

    `Heximax._delta`'s fast path only ever prices `target == knower`
    (`gains_many`/`accepts`'s one shape); here the knower is the bystander
    but the hand that moves is the mover's, exactly the `target != knower`
    shape `_delta` itself routes to `_delta_reference` -- so this mirrors
    `_delta_reference`'s clone-based computation instead of the belief-
    shift fast path, reading `bystander`'s row rather than `mover`'s.
    """
    state = view.state
    rank = bot._rank
    before = bot._read_row(state, view.ledger, bystander, bystander, rank)
    after = _thin_copy(state)
    gains = [max(0, n) for n in received]
    losses = [max(0, -n) for n in received]
    exact = bot.omniscient
    bot._move_hand(after, bystander, mover, gains=gains, losses=losses, exact=exact)
    bot._move_hand(after, bystander, counterparty, gains=losses, losses=gains, exact=exact)
    ledger = view.ledger.copy()
    for r in range(NUM_RESOURCES):
        if losses[r]:
            ledger.spend(mover, r, losses[r])
            ledger.receive(counterparty, r, losses[r])
        if gains[r]:
            ledger.spend(counterparty, r, gains[r])
            ledger.receive(mover, r, gains[r])
    return bot._read_row(after, ledger, bystander, bystander, rank) - before


# ---------------------------------------------------------------------------
# Census: clear each rule to exhaustion at each position.


def _run_rules_on_position(pos: Position) -> dict:
    game0 = pos.game
    me = game0.current_player
    n = game0.num_players
    state0 = game0.state(0, hidden=False)
    cards_cap = sum(sum(hand) for hand in state0.hands)
    vp = [victory_points(game0.state(s, hidden=False), s) for s in range(n)]

    out: dict[str, dict] = {}
    for rule in RULES:
        work = imagine(game0, random.Random(0))
        work.gates = game0.gates

        trades: list[dict] = []
        first_trade: tuple[int, tuple[int, ...]] | None = None
        executed = 0

        while executed < cards_cap:
            rows = candidate_rows(work, me)
            if not rows:
                break
            clearing = clearing_set(rule, rows)
            if not clearing:
                break
            best = select(rule, clearing)
            them, bundle = best.them, best.bundle

            byst_sum = sum(
                _bystander_delta(work.gates[s], work.state(s), s, me, them, bundle)
                for s in range(n) if s not in (me, them)
            )

            state = work.state(0, hidden=False)
            before_hands = [hand[:] for hand in state.hands]
            trading.exchange(state, me, them, bundle)
            work.ledger.apply_hand_diff(before_hands, state.hands)
            work.trades.append(trading.Trade(me, them, bundle, best.gain_actor, best.gain_counterparty))
            work.trades_made += 1
            executed += 1

            given = sum(max(0, -x) for x in bundle)
            received = sum(max(0, x) for x in bundle)
            leader_among_opponents = max(vp[s] for s in range(n) if s != me)
            trades.append({
                "rule": rule,
                "game": pos.game_index,
                "position": pos.position_index,
                "turn": pos.turn,
                "actor": me,
                "counterparty": them,
                "bundle": list(bundle),
                "given": given,
                "received": received,
                "max_side": max(given, received),
                "actor_gain": best.gain_actor,
                "counterparty_gain": best.gain_counterparty,
                "bystander_delta_sum": byst_sum,
                "counterparty_was_leader": bool(vp[them] >= leader_among_opponents),
            })
            if first_trade is None:
                first_trade = (them, tuple(bundle))

        out[rule] = {"first_trade": first_trade, "trades": trades}
    return out


def _census_worker(job: tuple[list[str], int]) -> list[dict]:
    lines, start_index = job
    out = []
    for offset, line in enumerate(lines):
        record = from_json(line)
        game_index = start_index + offset
        for pos in positions(record, game_index=game_index):
            out.append({
                "game": pos.game_index,
                "position": pos.position_index,
                "turn": pos.turn,
                "actor": pos.actor,
                "rules": _run_rules_on_position(pos),
            })
    return out


def _chunk_lines(lines: list[str], workers: int) -> list[tuple[list[str], int]]:
    total = len(lines)
    if total == 0:
        return []
    workers = max(1, min(workers, total))
    size = -(-total // workers)  # ceil division
    chunks = []
    start = 0
    while start < total:
        chunks.append((lines[start:start + size], start))
        start += size
    return chunks


def _read_lines(bank_path: Path) -> list[str]:
    return [line for line in bank_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_census(bank_path: Path, workers: int) -> list[dict]:
    lines = _read_lines(bank_path)
    jobs = _chunk_lines(lines, workers)
    if not jobs:
        return []
    if workers > 1 and len(jobs) > 1:
        with Pool(min(workers, MAX_WORKERS)) as pool:
            chunks = pool.map(_census_worker, jobs, chunksize=1)
    else:
        chunks = [_census_worker(job) for job in jobs]
    return [row for chunk in chunks for row in chunk]


# ---------------------------------------------------------------------------
# Readouts (census)


QUANTILE_LEVELS: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)


def _quantiles(values: Sequence[float], levels: Sequence[float] = QUANTILE_LEVELS) -> dict[str, float]:
    """Linear-interpolated percentiles (numpy's default method), keyed
    `"p10"`, `"p25"`, ... An empty `values` reports every level as `0.0`
    rather than raising -- a rule with zero executed trades still owes a
    row in the table."""
    if not values:
        return {f"p{int(round(level * 100))}": 0.0 for level in levels}
    ordered = sorted(values)
    n = len(ordered)
    out: dict[str, float] = {}
    for level in levels:
        if n == 1:
            out[f"p{int(round(level * 100))}"] = ordered[0]
            continue
        pos = level * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        out[f"p{int(round(level * 100))}"] = ordered[lo] + (ordered[hi] - ordered[lo]) * frac
    return out


def _bundle_shape(given: int, received: int) -> str:
    hi, lo = (given, received) if given >= received else (received, given)
    if hi == 1 and lo == 1:
        return "1:1"
    if hi == 2 and lo == 1:
        return "1:2"
    if hi == 2 and lo == 2:
        return "2:2"
    if hi >= 3 and lo >= 3:
        return "≥3 a side"
    return ">3 one way"


def compute_readouts(all_positions: list[dict]) -> dict:
    n_positions = len(all_positions)
    out: dict = {"n_positions": n_positions, "rules": {}}

    for rule in RULES:
        trades = [t for p in all_positions for t in p["rules"][rule]["trades"]]
        n_trades = len(trades)
        trades_per_event = (n_trades / n_positions) if n_positions else 0.0

        shape_counts = Counter(_bundle_shape(t["given"], t["received"]) for t in trades)
        shape_share = {
            label: (shape_counts.get(label, 0) / n_trades if n_trades else 0.0)
            for label in HUMAN_CORPUS_SHAPE
        }

        actor_gains = [t["actor_gain"] for t in trades]
        cp_gains = [t["counterparty_gain"] for t in trades]
        mean_actor = statistics.mean(actor_gains) if trades else 0.0
        mean_cp = statistics.mean(cp_gains) if trades else 0.0
        ratio = (mean_actor / mean_cp) if mean_cp else float("inf")
        share_lopsided = (
            sum(1 for t in trades if t["counterparty_gain"] < 0.1 * t["actor_gain"]) / n_trades
            if n_trades else 0.0
        )

        mean_bystander = statistics.mean(t["bystander_delta_sum"] for t in trades) if trades else 0.0
        share_lifts_leader = (
            sum(1 for t in trades if t["counterparty_was_leader"]) / n_trades if n_trades else 0.0
        )

        min_gains = [min(t["actor_gain"], t["counterparty_gain"]) for t in trades]
        gain_distribution = {
            "actor_gain": _quantiles(actor_gains),
            "counterparty_gain": _quantiles(cp_gains),
            "min_gain": _quantiles(min_gains),
            "share_min_gain_under_1e-4": (
                sum(1 for g in min_gains if g < 1e-4) / n_trades if n_trades else 0.0
            ),
            "share_min_gain_under_1e-3": (
                sum(1 for g in min_gains if g < 1e-3) / n_trades if n_trades else 0.0
            ),
        }

        out["rules"][rule] = {
            "n_trades": n_trades,
            "trades_per_event": trades_per_event,
            "bundle_shape": shape_share,
            "surplus_split": {
                "mean_actor_gain": mean_actor,
                "mean_counterparty_gain": mean_cp,
                "ratio_actor_over_counterparty": ratio,
                "share_counterparty_under_10pct_of_actor": share_lopsided,
            },
            "bystander_damage": {
                "mean_sum_bystander_delta": mean_bystander,
                "share_lifts_leader": share_lifts_leader,
            },
            "gain_distribution": gain_distribution,
        }

    disagreement: dict[str, float] = {}
    for a, b in itertools.combinations(RULES, 2):
        differ = sum(
            1 for p in all_positions if p["rules"][a]["first_trade"] != p["rules"][b]["first_trade"]
        )
        disagreement[f"{a} vs {b}"] = (differ / n_positions) if n_positions else 0.0
    out["rule_disagreement"] = disagreement
    return out


def table_md(readouts: dict) -> str:
    lines: list[str] = []

    lines.append(f"Positions: {readouts['n_positions']}\n")

    lines.append("## 1. Trades per event and bundle shape\n")
    header = "| rule | trades | trades/event | 1:1 | 1:2 | 2:2 | ≥3 a side | >3 one way |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|")
    for rule in RULES:
        r = readouts["rules"][rule]
        shape = r["bundle_shape"]
        lines.append(
            f"| {rule} | {r['n_trades']} | {r['trades_per_event']:.4f} | "
            f"{shape['1:1']*100:.1f}% | {shape['1:2']*100:.1f}% | {shape['2:2']*100:.1f}% | "
            f"{shape['≥3 a side']*100:.1f}% | {shape['>3 one way']*100:.1f}% |"
        )
    lines.append(
        f"| human corpus | -- | -- | {HUMAN_CORPUS_SHAPE['1:1']*100:.1f}% | "
        f"{HUMAN_CORPUS_SHAPE['1:2']*100:.1f}% | {HUMAN_CORPUS_SHAPE['2:2']*100:.1f}% | "
        f"{HUMAN_CORPUS_SHAPE['≥3 a side']*100:.1f}% | {HUMAN_CORPUS_SHAPE['>3 one way']*100:.1f}% |"
    )

    lines.append("\n## 2. Surplus split\n")
    lines.append("| rule | mean actor gain | mean counterparty gain | ratio | share cp < 10% of actor |")
    lines.append("|---|---|---|---|---|")
    for rule in RULES:
        s = readouts["rules"][rule]["surplus_split"]
        lines.append(
            f"| {rule} | {s['mean_actor_gain']:.5f} | {s['mean_counterparty_gain']:.5f} | "
            f"{s['ratio_actor_over_counterparty']:.2f} | {s['share_counterparty_under_10pct_of_actor']*100:.1f}% |"
        )

    lines.append("\n## 3. Bystander damage\n")
    lines.append("| rule | mean sum bystander ΔV | share lifts leader |")
    lines.append("|---|---|---|")
    for rule in RULES:
        b = readouts["rules"][rule]["bystander_damage"]
        lines.append(
            f"| {rule} | {b['mean_sum_bystander_delta']:+.5f} | {b['share_lifts_leader']*100:.1f}% |"
        )

    lines.append("\n## 4. Rule disagreement (share of positions, first trade differs)\n")
    lines.append("| pair | disagreement |")
    lines.append("|---|---|")
    for pair, share in readouts["rule_disagreement"].items():
        lines.append(f"| {pair} | {share*100:.1f}% |")

    lines.append("\n## 5. Gain distribution (executed trades)\n")
    lines.append(
        "| rule | side | p10 | p25 | p50 | p75 | p90 | p99 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for rule in RULES:
        gd = readouts["rules"][rule]["gain_distribution"]
        for side in ("actor_gain", "counterparty_gain", "min_gain"):
            q = gd[side]
            lines.append(
                f"| {rule} | {side} | {q['p10']:.5f} | {q['p25']:.5f} | {q['p50']:.5f} | "
                f"{q['p75']:.5f} | {q['p90']:.5f} | {q['p99']:.5f} |"
            )
    lines.append("\n| rule | share min(gain) < 1e-4 | share min(gain) < 1e-3 |")
    lines.append("|---|---|---|")
    for rule in RULES:
        gd = readouts["rules"][rule]["gain_distribution"]
        lines.append(
            f"| {rule} | {gd['share_min_gain_under_1e-4']*100:.1f}% | "
            f"{gd['share_min_gain_under_1e-3']*100:.1f}% |"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Strategic: is the honest gate a fixed point? One seat's gate is shaded and
# the rest stay honest; measured against its own realised gain at tau=0, k=1.

STRATEGIC_TAUS: tuple[float, ...] = (0.0, 1e-4, 5e-4, 1e-3, 5e-3)
STRATEGIC_KS: tuple[float, ...] = (1.5, 2.0)
EXAGGERATION_RULES: tuple[str, ...] = ("egalitarian", "nash")


def _tau_label(tau: float) -> str:
    return f"tau={tau:g}"


def _k_label(k: float) -> str:
    return f"k={k:g}"


HONEST_ARM = _tau_label(0.0)


def _shaded_pick(
    rule: str, rows: Sequence[CandidateRow], me: int, shaded_seat: int, tau: float, k: float,
) -> CandidateRow | None:
    """The clearing candidate `rule` picks when `shaded_seat` plays a shaded
    gate and every other seat is honest -- `None` if nothing clears.

    Admission always reads the shaded seat's *true* gain against `tau`
    (never `k`); `k` -- 1.0 outside the exaggeration arms -- only scales the
    value fed to the selection key, mirroring `select`'s own tie-break
    exactly but keyed on the reported figure instead of the true one.
    """
    admitted: list[tuple[CandidateRow, float, float]] = []
    for r in rows:
        if shaded_seat == me:
            admit = r.gain_actor > tau and r.gain_counterparty > 0
            rep_actor, rep_cp = r.gain_actor * k, r.gain_counterparty
        elif r.them == shaded_seat:
            admit = r.gain_actor > 0 and r.gain_counterparty > tau
            rep_actor, rep_cp = r.gain_actor, r.gain_counterparty * k
        else:
            admit = r.gain_actor > 0 and r.gain_counterparty > 0
            rep_actor, rep_cp = r.gain_actor, r.gain_counterparty
        if admit:
            admitted.append((r, rep_actor, rep_cp))
    if not admitted:
        return None
    if rule == "actor":
        key = lambda t: (t[1], t[1], _canonical(t[0]), -t[0].them)
    elif rule == "egalitarian":
        key = lambda t: (min(t[1], t[2]), t[1], _canonical(t[0]), -t[0].them)
    elif rule == "nash":
        key = lambda t: (t[1] * t[2], t[1], _canonical(t[0]), -t[0].them)
    else:
        raise ValueError(f"unknown rule: {rule}")
    return max(admitted, key=key)[0]


def _run_arm_on_position(
    game0: Game, me: int, cards_cap: int, rule: str, shaded_seat: int, tau: float, k: float,
) -> tuple[float, int]:
    """Clear `rule` to exhaustion at `game0` with `shaded_seat` shaded by
    `(tau, k)`. Returns (realised gain, trade count) for the shaded seat --
    the honest evaluator's own reading of its true gain summed over every
    executed trade the shaded seat was a party to, 0 if none."""
    work = imagine(game0, random.Random(0))
    work.gates = game0.gates

    realised = 0.0
    shaded_trades = 0
    executed = 0
    while executed < cards_cap:
        rows = candidate_rows(work, me)
        if not rows:
            break
        best = _shaded_pick(rule, rows, me, shaded_seat, tau, k)
        if best is None:
            break
        them, bundle = best.them, best.bundle

        state = work.state(0, hidden=False)
        before_hands = [hand[:] for hand in state.hands]
        trading.exchange(state, me, them, bundle)
        work.ledger.apply_hand_diff(before_hands, state.hands)
        work.trades.append(trading.Trade(me, them, bundle, best.gain_actor, best.gain_counterparty))
        work.trades_made += 1
        executed += 1

        if shaded_seat == me:
            realised += best.gain_actor
            shaded_trades += 1
        elif them == shaded_seat:
            realised += best.gain_counterparty
            shaded_trades += 1
    return realised, shaded_trades


def _arms_for_rule(rule: str) -> list[tuple[str, float, float]]:
    """(label, tau, k) for every arm this rule is measured under."""
    arms = [(_tau_label(tau), tau, 1.0) for tau in STRATEGIC_TAUS]
    if rule in EXAGGERATION_RULES:
        arms += [(_k_label(k), 0.0, k) for k in STRATEGIC_KS]
    return arms


def _run_strategic_on_position(pos: Position) -> dict | None:
    game0 = pos.game
    me = game0.current_player
    shaded_seat = pos.position_index % 4
    state0 = game0.state(0, hidden=False)
    cards_cap = sum(sum(hand) for hand in state0.hands)

    rows0 = candidate_rows(game0, me)
    out: dict[str, dict] = {}
    any_qualifies = False
    for rule in RULES:
        clearing0 = clearing_set(rule, rows0)
        if me == shaded_seat:
            qualifies = bool(clearing0)
        else:
            qualifies = any(r.them == shaded_seat for r in clearing0)
        if not qualifies:
            out[rule] = {"qualifies": False, "shaded_seat": shaded_seat, "arms": {}}
            continue
        any_qualifies = True
        arms: dict[str, tuple[float, int]] = {}
        for label, tau, k in _arms_for_rule(rule):
            arms[label] = _run_arm_on_position(game0, me, cards_cap, rule, shaded_seat, tau, k)
        out[rule] = {"qualifies": True, "shaded_seat": shaded_seat, "arms": arms}
    if not any_qualifies:
        return None
    return out


def _strategic_worker(job: tuple[list[str], int]) -> list[dict]:
    lines, start_index = job
    out = []
    for offset, line in enumerate(lines):
        record = from_json(line)
        game_index = start_index + offset
        for pos in positions(record, game_index=game_index):
            row = _run_strategic_on_position(pos)
            if row is not None:
                out.append({"game": game_index, "position": pos.position_index, "rules": row})
    return out


def run_strategic(bank_path: Path, workers: int) -> list[dict]:
    lines = _read_lines(bank_path)
    jobs = _chunk_lines(lines, workers)
    if not jobs:
        return []
    if workers > 1 and len(jobs) > 1:
        with Pool(min(workers, MAX_WORKERS)) as pool:
            chunks = pool.map(_strategic_worker, jobs, chunksize=1)
    else:
        chunks = [_strategic_worker(job) for job in jobs]
    return [row for chunk in chunks for row in chunk]


def _bootstrap_ci(
    diffs: Sequence[float], resamples: int = 1000, seed: int = 1,
) -> tuple[float, float, float]:
    """(mean, 2.5th pct, 97.5th pct) of the mean of `diffs` under resampling
    with replacement. `(0.0, 0.0, 0.0)` for an empty input -- no positions
    qualified, nothing to report."""
    if not diffs:
        return 0.0, 0.0, 0.0
    mean = statistics.mean(diffs)
    if len(diffs) < 2:
        return mean, mean, mean
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(0.025 * resamples)]
    hi = means[min(int(0.975 * resamples), resamples - 1)]
    return mean, lo, hi


def compute_strategic_readouts(records: list[dict]) -> dict:
    out: dict = {"n_positions_considered": len(records), "rules": {}}
    for rule in RULES:
        arm_labels = [label for label, _, _ in _arms_for_rule(rule)]
        qualifying = [r["rules"][rule] for r in records if r["rules"][rule]["qualifies"]]
        n_qualifying = len(qualifying)
        arm_out: dict[str, dict] = {}
        honest_realised = [q["arms"][HONEST_ARM][0] for q in qualifying]
        for label in arm_labels:
            realised = [q["arms"][label][0] for q in qualifying]
            trades = [q["arms"][label][1] for q in qualifying]
            mean_realised = statistics.mean(realised) if realised else 0.0
            trades_per_event = statistics.mean(trades) if trades else 0.0
            if label == HONEST_ARM:
                arm_out[label] = {
                    "mean_realised_gain": mean_realised,
                    "trades_per_event": trades_per_event,
                    "paired_diff_vs_honest": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0},
                    "deviation_pays": False,
                }
                continue
            diffs = [r - h for r, h in zip(realised, honest_realised)]
            d_mean, d_lo, d_hi = _bootstrap_ci(diffs)
            arm_out[label] = {
                "mean_realised_gain": mean_realised,
                "trades_per_event": trades_per_event,
                "paired_diff_vs_honest": {"mean": d_mean, "ci_low": d_lo, "ci_high": d_hi},
                "deviation_pays": d_lo > 0.0,
            }
        out["rules"][rule] = {"n_qualifying": n_qualifying, "arms": arm_out}
    return out


def strategic_table_md(readouts: dict) -> str:
    lines: list[str] = [f"Positions considered: {readouts['n_positions_considered']}\n"]
    lines.append(
        "| rule | arm | n qualifying | mean realised gain | trades/event | "
        "paired diff vs honest | 95% CI | deviation pays |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for rule in RULES:
        r = readouts["rules"][rule]
        for label, arm in r["arms"].items():
            diff = arm["paired_diff_vs_honest"]
            lines.append(
                f"| {rule} | {label} | {r['n_qualifying']} | {arm['mean_realised_gain']:.6f} | "
                f"{arm['trades_per_event']:.3f} | {diff['mean']:+.6f} | "
                f"[{diff['ci_low']:+.6f}, {diff['ci_high']:+.6f}] | "
                f"{'yes' if arm['deviation_pays'] else 'no'} |"
            )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Phase 3: the judged position set.
#
# A judged position is sampled with a stable, small integer id (`pid`) --
# assigned once, at sampling time, in `run_judged_positions` -- and every
# downstream file (`phase3-positions.jsonl`, `phase3.jsonl`) is keyed by it
# rather than by the `(game, position)` pair alone, because `pid` is what
# the paired chance seed is built from (below).

SAMPLE_PER_GAME_CAP = 6
SAMPLE_TOTAL = 300
SAMPLE_SEED = 7


def _judged_row(pos: "Position") -> dict:
    first = pos.historical_trades[0]
    counterparty = first.b if first.a == pos.actor else first.a
    return {
        "game": pos.game_index,
        "position": pos.position_index,
        "turn": pos.turn,
        "actor": pos.actor,
        "counterparty": counterparty,
        "n_historical_trades": len(pos.historical_trades),
        "bundle": list(first.received) if first.a == pos.actor else [-n for n in first.received],
        "actor_gain": first.gain_a if first.a == pos.actor else first.gain_b,
        "counterparty_gain": first.gain_b if first.a == pos.actor else first.gain_a,
    }


def _judged_worker(job: tuple[list[str], int]) -> list[dict]:
    lines, start_index = job
    out = []
    for offset, line in enumerate(lines):
        record = from_json(line)
        game_index = start_index + offset
        for pos in judged_positions(record, game_index=game_index):
            out.append(_judged_row(pos))
    return out


def run_judged_positions(bank_path: Path, workers: int) -> list[dict]:
    """Every judged position in `bank_path`, unsampled -- one row per
    position before a MAIN-entry event where the bank's own recording rule
    (`egalitarian`) cleared at least one trade."""
    lines = _read_lines(bank_path)
    jobs = _chunk_lines(lines, workers)
    if not jobs:
        return []
    if workers > 1 and len(jobs) > 1:
        with Pool(min(workers, MAX_WORKERS)) as pool:
            chunks = pool.map(_judged_worker, jobs, chunksize=1)
    else:
        chunks = [_judged_worker(job) for job in jobs]
    return [row for chunk in chunks for row in chunk]


def sample_judged_positions(
    all_judged: list[dict], *, per_game_cap: int = SAMPLE_PER_GAME_CAP,
    total: int = SAMPLE_TOTAL, seed: int = SAMPLE_SEED,
) -> list[dict]:
    """Sample the phase-3 judged set: at most `per_game_cap` per game, then
    at most `total` overall, both deterministic on `seed`. Assigns each
    sampled row its stable `pid` (0-based, in the order sampled) -- this is
    what `judge`'s chance seed (`90000 + 100*pid + k`) is keyed on, not the
    `(game, position)` pair, so the seed stays a small, collision-free
    integer whatever the bank's own indices happen to be.

    Fewer than `total` rows in `all_judged` (after the per-game cap) is not
    an error -- every one available is taken, same convention phase 2's own
    stratified sampler used.
    """
    by_game: dict[int, list[dict]] = defaultdict(list)
    for row in all_judged:
        by_game[row["game"]].append(row)

    rng = random.Random(seed)
    pool: list[dict] = []
    for game in sorted(by_game):
        rows = by_game[game]
        k = min(per_game_cap, len(rows))
        pool.extend(rng.sample(rows, k))

    k = min(total, len(pool))
    sampled = rng.sample(pool, k)
    sampled.sort(key=lambda r: (r["game"], r["position"]))
    for pid, row in enumerate(sampled):
        row["pid"] = pid
    return sampled


# ---------------------------------------------------------------------------
# Phase 3: the paired-chance judge.


class _PairedChance(Chance):
    """Replays a recorded chance script by *kind*, independently per kind,
    rather than as one strictly-ordered sequence (`hexset.chance.Scripted`).

    This is what actually delivers "identical dice, steals and draws for
    the traded and untraded games": the two forks of one stream start from
    hands that already differ (by the one historical trade), so they all
    but immediately diverge in *which* action they take next -- whether a
    knight is played before or after rolling, in particular, decides
    whether `chance.steal` or `chance.roll` comes first. A single ordered
    script (`Scripted`) treats that reordering as a `ChanceMismatch` and
    ends the pairing within the first turn or two, almost every stream, as
    measured on this run's own smoke test. Reading each kind from its own
    queue sidesteps that entirely: the fork that has rolled five times so
    far gets the fifth recorded roll whether or not a steal happened to
    fall between the fourth and fifth roll on the *other* fork's path.
    Only true exhaustion -- a fork needs a sixth roll and the throwaway
    script never recorded one -- raises, `ChanceExhausted`; there is no
    `ChanceMismatch` left to raise, since a kind is never checked against
    another kind's queue at all. `deck_order` is not implemented: nothing
    mid-game reads it (`hexset.chance.deck_order` is only ever consulted
    once, at `hexset.game.start`), and neither fork here is a fresh game.

    **A recorded steal is only replayed when the current hand can actually
    produce it.** `hexset.robber.steal` does not itself check that the
    resource it is handed is held (`hand[resource] -= 1` with no floor),
    by design -- a real `Chance.steal(hand)` always draws proportionally
    to `hand`, so this can never come up on any other caller. But a
    fork's victim hand is exactly the thing that has diverged from the
    throwaway's own path by the time a steal fires (found on this run's
    own smoke test, ~100 actions and ~46 turns in, on a position that ran
    clean for 400+ actions under a live source): blindly replaying "took a
    WOOD" against a hand holding no wood drives that count negative,
    silently corrupting every accounting that reads `state.hands`
    afterwards -- including the very cycle check whose assertion this
    surfaced as. Treating a would-be-invalid steal as exhaustion (of this
    kind, from here) reuses the fallback this class already has for the
    same underlying reason a length exhaustion gets one: the script no
    longer describes a state this fork can be in.
    """

    def __init__(self, events_by_kind: dict[str, list[int]]) -> None:
        self._events = {kind: list(values) for kind, values in events_by_kind.items()}
        self._index = {kind: 0 for kind in self._events}

    def _next(self, kind: str) -> int:
        values = self._events.get(kind, ())
        i = self._index.get(kind, 0)
        if i >= len(values):
            raise ChanceExhausted(i, kind)
        self._index[kind] = i + 1
        return values[i]

    def roll(self) -> int:
        return self._next("roll")

    def steal(self, hand: Sequence[int]) -> int | None:
        if sum(hand) == 0:
            return None
        values = self._events.get("steal", ())
        i = self._index.get("steal", 0)
        if i >= len(values) or hand[values[i]] <= 0:
            raise ChanceExhausted(i, "steal")
        self._index["steal"] = i + 1
        return values[i]

    def discard(self, hand: Sequence[int], n: int) -> list[int]:
        """As `steal` above: a recorded discard pick is only replayed while
        the hand can still afford it, in case this kind is ever exercised
        (heximax and search2 both choose their own discards explicitly,
        never through `Chance.discard`, so this path is not reached by
        this module's own bots today)."""
        remaining = list(hand)
        picks: list[int] = []
        values = self._events.get("discard", ())
        i = self._index.get("discard", 0)
        for _ in range(n):
            if i >= len(values) or remaining[values[i]] <= 0:
                self._index["discard"] = i
                raise ChanceExhausted(i, "discard")
            remaining[values[i]] -= 1
            picks.append(values[i])
            i += 1
        self._index["discard"] = i
        return picks


def _record_chance_script(game0: Game, board, seed_value: int, action_cap: int) -> dict[str, list[int]]:
    """A throwaway continuation from `game0`, seated with fresh heximax
    bots trading under the engine's own shipped rule, run for up to
    `action_cap` actions purely to log a chance-event script
    (`hexset.chance.Recording(Live(...))`), grouped by kind (`_PairedChance`'s
    own shape) -- long enough, kind for kind, for both the untraded and
    traded forks of this stream to draw from. The playout itself -- who
    wins, what it does -- is discarded; only the grouped events survive.

    `randomize_deck=False`: the throwaway (and both real forks, in
    `_play_fork`) must start from the *same* remaining deck order as the
    captured position itself, not a freshly shuffled one -- `imagine`'s
    default reshuffle exists to keep a bot's own tree search from peeking
    at what it is about to draw, which is not a concern here and would
    otherwise desync the two paired forks' future draws for no reason
    (`hexset.chance.deck_order` is only ever consulted once, at
    `hexset.game.start`; nothing mid-game reads it again).
    """
    chance_rng = random.Random(seed_value)
    recorder = Recording(Live(chance_rng))
    work = imagine(game0, random.Random(f"{seed_value}:throwaway"), randomize_deck=False)
    work.chance = recorder
    n = game0.num_players
    bots = [heximax(board, random.Random(f"{seed_value}:throwaway:{s}")) for s in range(n)]
    work.gates = tuple(bots)
    work.max_trades = None
    actions = 0
    while not is_over(work) and actions < action_cap:
        seat = to_move(work)
        apply(work, bots[seat].choose(work))
        actions += 1
    by_kind: dict[str, list[int]] = defaultdict(list)
    for kind, value in recorder.events:
        by_kind[kind].append(value)
    return dict(by_kind)


@dataclass
class _ForkResult:
    winner: int | None
    turns: int
    actions: int
    chance_exhausted: bool
    chance_exhausted_at: int | None
    # Set when this fork raised something other than the chance divergence
    # `_play_fork` already handles -- a long continuation (up to `--cap`
    # actions, hundreds of positions, several streams each) can hit a rare
    # engine-level invariant (`hexset.trading.trade_event`'s own
    # position-revisit assertion is exactly the kind of thing its own
    # docstring calls "a bug to surface", and at this volume "rare" is not
    # "never"); recorded here rather than crashing the whole judge run,
    # since one job failing is not a reason to lose every job that already
    # finished. `winner`/`turns`/`actions` are `None`/`-1`/`-1` when this is
    # set -- there is no result to read.
    error: str | None = None


def _play_fork(
    game0: Game, board, *, seed: str, chance_by_kind: dict[str, list[int]],
    action_cap: int, suppress_current_turn: bool,
) -> _ForkResult:
    """Continue `game0` to a winner (or `None`, capped at `action_cap`),
    seating a fresh `heximax` at every seat and driving chance from
    `chance_by_kind` (`_PairedChance`) rather than a live source -- the
    paired half of one stream. `suppress_current_turn` is the untraded
    fork's own suppression: `max_trades=0` for the turn `game0` is
    mid-way through only, restored the moment `Game.turns` first advances
    past it (`hexset.game.end_turn`), so the rest of the game trades
    exactly as the engine would.

    `ChanceExhausted` (this fork needs more of some kind than the
    throwaway script recorded) falls back to a freshly seeded `Live` for
    the remainder and is counted, per the registration.
    """
    rng = random.Random(seed)
    work = imagine(game0, rng, randomize_deck=False)
    work.chance = _PairedChance(chance_by_kind)
    n = game0.num_players
    bots = [heximax(board, random.Random(f"{seed}:{s}")) for s in range(n)]
    work.gates = tuple(bots)
    turn0 = work.turns
    if suppress_current_turn:
        work.max_trades = 0
    else:
        work.max_trades = None

    chance_exhausted = False
    chance_exhausted_at = None
    actions = 0
    while not is_over(work) and actions < action_cap:
        if suppress_current_turn and work.max_trades == 0 and work.turns != turn0:
            work.max_trades = None
        seat = to_move(work)
        action = bots[seat].choose(work)
        if chance_exhausted:
            # Already permanently on `Live` (which never raises
            # `ChanceExhausted`) -- nothing left to guard against.
            apply(work, action)
        else:
            # A checkpoint taken *before* applying: some action functions
            # mutate state ahead of their own chance draw (`move_robber_to`
            # moves the robber, then steals), so a `ChanceExhausted`
            # partway through leaves `work` non-idempotent to retry in
            # place -- retrying the same `move_robber` a second time
            # raises "the robber must move to a different hex". Falling
            # back to this untouched clone instead of the mutated `work`
            # sidesteps that for every action shape, not only the one this
            # was found on.
            checkpoint = imagine(work, random.Random(0), randomize_deck=False)
            checkpoint.gates = work.gates
            checkpoint.chance = work.chance
            try:
                apply(work, action)
            except ChanceExhausted:
                chance_exhausted = True
                chance_exhausted_at = actions
                work = checkpoint
                work.chance = Live(random.Random(f"{seed}:exhausted"))
                apply(work, action)
        actions += 1
    return _ForkResult(work.won_by, work.turns, actions, chance_exhausted, chance_exhausted_at)


def _play_fork_safely(*args, **kwargs) -> _ForkResult:
    """`_play_fork`, guarded: see `_ForkResult.error`'s own docstring for
    why one job's rare engine-level failure must not crash a run judging
    thousands of them."""
    try:
        return _play_fork(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see above
        return _ForkResult(None, -1, -1, False, None, error=f"{type(exc).__name__}: {exc}")


def _locate_judged_position(row: dict, record: Record) -> tuple[Position, Game]:
    """Replay `record` up to the judged position `row` names (cheap: no
    bot search runs here, only the recorded actions), check it against what
    `phase3-positions.jsonl` recorded, and build the traded fork's shared
    starting game (the historical clearing applied outright). Raises
    `ReplayError` on any mismatch -- the bank or the positions file is
    stale relative to the other."""
    pos = None
    for candidate in judged_positions(record, game_index=row["game"]):
        if candidate.position_index == row["position"]:
            pos = candidate
            break
    if pos is None:
        raise ReplayError(
            f"trade_lab judge: game {row['game']} has no judged position {row['position']} "
            "on replay -- the bank or the positions file is stale"
        )
    if pos.actor != row["actor"] or len(pos.historical_trades) != row["n_historical_trades"]:
        raise ReplayError(
            f"trade_lab judge: game {row['game']} position {row['position']} replayed "
            f"actor {pos.actor}/{len(pos.historical_trades)} historical trades, "
            f"the positions file recorded actor {row['actor']}/{row['n_historical_trades']}"
        )

    # `randomize_deck=False`: the traded fork's deck must stay exactly the
    # captured position's own remaining order -- see `_record_chance_
    # script`'s docstring for why a reshuffle here would desync it from
    # `pos.game` (the untraded fork's own starting point, never reshuffled).
    traded0 = imagine(pos.game, random.Random(0), randomize_deck=False)
    state = traded0.state(0, hidden=False)
    for t in pos.historical_trades:
        before_hands = [hand[:] for hand in state.hands]
        trading.exchange(state, t.a, t.b, t.received)
        traded0.ledger.apply_hand_diff(before_hands, state.hands)
        traded0.trades.append(t)
        traded0.trades_made += 1
    return pos, traded0


def _errored_row(row: dict, stream: int, error: str) -> dict:
    """The row shape `_judge_stream` returns when nothing playable could
    even be built for this (position, stream) -- the throwaway chance
    script itself hit the same class of rare engine-level failure
    `_ForkResult.error` guards a fork against (`_record_chance_script` has
    no fork of its own to catch it in, since it runs before either fork
    exists). Both sides are marked errored, uniformly with a fork-level
    error's own shape, so a reader never has to special-case where in the
    pipeline a row failed."""
    empty = {"winner": None, "turns": -1, "actions": -1, "chance_exhausted": False, "chance_exhausted_at": None, "error": error}
    return {
        "pid": row["pid"], "game": row["game"], "position": row["position"], "turn": row["turn"],
        "actor": row["actor"], "counterparty": row["counterparty"],
        "n_historical_trades": row["n_historical_trades"], "bundle": row["bundle"],
        "actor_gain": row["actor_gain"], "counterparty_gain": row["counterparty_gain"],
        "stream": stream,
        "untraded": dict(empty), "traded": dict(empty),
        "win_actor_untraded": None, "win_actor_traded": None, "delta_actor": None,
        "win_counterparty_untraded": None, "win_counterparty_traded": None, "delta_counterparty": None,
        "win_bystanders_untraded": None, "win_bystanders_traded": None, "delta_bystanders": None,
    }


def _judge_stream(row: dict, pos: Position, traded0: Game, board, *, stream: int, cap: int) -> dict:
    """One (position, stream)'s paired result: the throwaway chance script,
    the untraded fork (this turn's event suppressed) and the traded fork
    (the historical clearing already applied to `traded0`), both driven by
    that same script."""
    pid = row["pid"]
    actor, counterparty = row["actor"], row["counterparty"]
    n = pos.game.num_players
    bystanders = [s for s in range(n) if s not in (actor, counterparty)]

    try:
        events = _record_chance_script(pos.game, board, 90000 + 100 * pid + stream, cap)
    except Exception as exc:  # noqa: BLE001 -- see `_errored_row`'s own docstring
        return _errored_row(row, stream, f"{type(exc).__name__}: {exc}")

    untraded = _play_fork_safely(
        pos.game, board,
        seed=f"phase3:{pid}:{stream}:untraded", chance_by_kind=events,
        action_cap=cap, suppress_current_turn=True,
    )
    traded = _play_fork_safely(
        traded0, board,
        seed=f"phase3:{pid}:{stream}:traded", chance_by_kind=events,
        action_cap=cap, suppress_current_turn=False,
    )

    win_actor_u = 1.0 if untraded.winner == actor else 0.0
    win_actor_t = 1.0 if traded.winner == actor else 0.0
    win_cp_u = 1.0 if untraded.winner == counterparty else 0.0
    win_cp_t = 1.0 if traded.winner == counterparty else 0.0
    win_byst_u = [1.0 if untraded.winner == b else 0.0 for b in bystanders]
    win_byst_t = [1.0 if traded.winner == b else 0.0 for b in bystanders]

    return {
        "pid": pid, "game": row["game"], "position": row["position"], "turn": row["turn"],
        "actor": actor, "counterparty": counterparty,
        "n_historical_trades": row["n_historical_trades"], "bundle": row["bundle"],
        "actor_gain": row["actor_gain"], "counterparty_gain": row["counterparty_gain"],
        "stream": stream,
        "untraded": {
            "winner": untraded.winner, "turns": untraded.turns, "actions": untraded.actions,
            "chance_exhausted": untraded.chance_exhausted, "chance_exhausted_at": untraded.chance_exhausted_at,
            "error": untraded.error,
        },
        "traded": {
            "winner": traded.winner, "turns": traded.turns, "actions": traded.actions,
            "chance_exhausted": traded.chance_exhausted, "chance_exhausted_at": traded.chance_exhausted_at,
            "error": traded.error,
        },
        # `None` (not 0.0/1.0) whenever either fork errored -- an errored
        # fork has no winner to compare, and silently reading its `None`
        # winner as "nobody among actor/counterparty/bystanders won" would
        # score it as a real, informative outcome instead of a missing one.
        "win_actor_untraded": None if untraded.error else win_actor_u,
        "win_actor_traded": None if traded.error else win_actor_t,
        "delta_actor": None if (untraded.error or traded.error) else win_actor_t - win_actor_u,
        "win_counterparty_untraded": None if untraded.error else win_cp_u,
        "win_counterparty_traded": None if traded.error else win_cp_t,
        "delta_counterparty": None if (untraded.error or traded.error) else win_cp_t - win_cp_u,
        "win_bystanders_untraded": None if untraded.error else win_byst_u,
        "win_bystanders_traded": None if traded.error else win_byst_t,
        "delta_bystanders": (
            None if (untraded.error or traded.error) else [t - u for t, u in zip(win_byst_t, win_byst_u)]
        ),
    }


def judge_position(row: dict, record: Record, board, *, streams: int, cap: int) -> list[dict]:
    """Every stream's paired result for one judged position (`row`, from
    `phase3-positions.jsonl`), 0..`streams`-1 -- the convenience form the
    smoke test and a single-process run use; `run_judge`'s own worker calls
    `_judge_stream` directly, one (position, stream) job at a time, so
    resuming never recomputes a stream that already finished."""
    pos, traded0 = _locate_judged_position(row, record)
    return [_judge_stream(row, pos, traded0, board, stream=k, cap=cap) for k in range(streams)]


def _judge_job(job: tuple[dict, Record, object, int, int]) -> dict:
    row, record, board, stream, cap = job
    pos, traded0 = _locate_judged_position(row, record)
    return _judge_stream(row, pos, traded0, board, stream=stream, cap=cap)


def _existing_pairs(out_path: Path) -> set[tuple[int, int]]:
    """`(pid, stream)` pairs already written to `out_path` -- what makes
    `judge` resumable: a job whose pair is already here is skipped."""
    if not out_path.exists():
        return set()
    done: set[tuple[int, int]] = set()
    with out_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((row["pid"], row["stream"]))
    return done


def _judge_progress_line(result: dict) -> str:
    """One line per finished (position, stream). `delta_actor`/
    `delta_counterparty` are `None` whenever either fork errored
    (`_ForkResult.error`) -- printed as the fork's own error instead of a
    number, so a crashed fork is visible in the progress stream, not just
    in the output file."""
    if result["delta_actor"] is None or result["delta_counterparty"] is None:
        err = result["untraded"]["error"] or result["traded"]["error"]
        return f"pid={result['pid']} stream={result['stream']} ERROR: {err}"
    return (
        f"pid={result['pid']} stream={result['stream']} "
        f"delta_actor={result['delta_actor']:+.1f} "
        f"delta_counterparty={result['delta_counterparty']:+.1f}"
    )


def run_judge(
    positions_path: Path, bank_path: Path, out_path: Path, *, streams: int, cap: int, workers: int,
) -> None:
    """Judge every position in `positions_path`, `streams` paired chance
    streams each, appending one JSON line per finished (pid, stream) to
    `out_path` and printing one line as each finishes. Resumable: a
    (pid, stream) already in `out_path` is skipped, so a killed and
    restarted run picks up where it left off rather than repeating work.
    Each (position, stream) is its own job -- `_locate_judged_position`'s
    replay (cheap) runs once per job, never once per position for its
    whole stream range, so resuming a partly-done position recomputes
    only the streams still missing.
    """
    rows = [json.loads(line) for line in positions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = list(read(str(bank_path)))
    boards: dict[int, object] = {}
    done = _existing_pairs(out_path)

    jobs = []
    for row in rows:
        game_index = row["game"]
        if game_index not in boards:
            boards[game_index] = board_of(records[game_index])
        board = boards[game_index]
        pid = row["pid"]
        for k in range(streams):
            if (pid, k) in done:
                continue
            jobs.append((row, records[game_index], board, k, cap))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        if workers > 1 and len(jobs) > 1:
            with Pool(min(workers, MAX_WORKERS)) as pool:
                results = pool.imap_unordered(_judge_job, jobs, chunksize=1)
                for result in results:
                    handle.write(json.dumps(result) + "\n")
                    handle.flush()
                    print(_judge_progress_line(result), file=sys.stderr)
        else:
            for job in jobs:
                result = _judge_job(job)
                handle.write(json.dumps(result) + "\n")
                handle.flush()
                print(_judge_progress_line(result), file=sys.stderr)


# ---------------------------------------------------------------------------
# Phase 3: readouts.
#
# The claim under test (`trading-final.md` item 4): "No threshold: tau = 0
# until the paired-chance judge measures a gate's resolution, and then tau
# is that measurement, never a chosen number." The bins below are phase 2's
# own `STRATEGIC_TAUS` grid (the finest resolution phase 2 found reason to
# use, straddling the observed gain distribution of ~1e-4 to ~6e-3); the
# decision rule below is this module's own construction for turning that
# grid into one number, since no exact rule text survived into this
# module's registration section -- see the run's own report for the
# disclosure.

GAIN_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 1e-4), (1e-4, 5e-4), (5e-4, 1e-3), (1e-3, 5e-3), (5e-3, float("inf")),
)


def _bin_label(lo: float, hi: float) -> str:
    if hi == float("inf"):
        return f"≥{lo:g}"
    return f"[{lo:g}, {hi:g})"


def _cluster_by_position(rows: list[dict]) -> dict[int, list[dict]]:
    by_pid: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_pid[r["pid"]].append(r)
    return by_pid


def _position_means(rows: list[dict], field: str) -> dict[int, tuple[float, float]]:
    """`{pid: (mean claimed gain, mean of field over streams)}` -- one row
    per position, the cluster unit every bootstrap below resamples. A
    stream whose `field` is `None` (one of its forks errored, `_ForkResult.
    error`) is dropped from the mean rather than treated as a real 0 --
    when every stream at a position errored, the position itself is
    dropped, since there is nothing to cluster."""
    by_pid = _cluster_by_position(rows)
    out = {}
    for pid, group in by_pid.items():
        valid = [r[field] for r in group if r[field] is not None]
        if not valid:
            continue
        gain = group[0]["actor_gain"]
        out[pid] = (gain, statistics.mean(valid))
    return out


def _cluster_bootstrap(values: Sequence[float], resamples: int = 2000, seed: int = 3) -> tuple[float, float, float]:
    """(mean, 2.5th pct, 97.5th pct) resampling positions (clusters) with
    replacement -- the paired bootstrap the registration calls for, 2,000
    resamples, so within-position stream correlation never leaks into the
    interval as if streams were independent samples."""
    if not values:
        return 0.0, 0.0, 0.0
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean, mean
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(0.025 * resamples)]
    hi = means[min(int(0.975 * resamples), resamples - 1)]
    return mean, lo, hi


def _binned_readout(rows: list[dict], field: str) -> dict:
    """Readout 1/2's shared shape: bin positions by their claimed
    `actor_gain`, and within each bin, the paired-bootstrap mean (over
    positions, streams pre-averaged) of `field` (`delta_actor`,
    `delta_counterparty`, or a bystander delta)."""
    means = _position_means(rows, field)
    out: dict[str, dict] = {}
    for lo, hi in GAIN_BINS:
        label = _bin_label(lo, hi)
        values = [v for gain, v in means.values() if lo <= gain < hi]
        mean, ci_lo, ci_hi = _cluster_bootstrap(values)
        out[label] = {
            "n_positions": len(values), "mean": mean, "ci_low": ci_lo, "ci_high": ci_hi,
            "resolved_positive": ci_lo > 0.0,
        }
    return out


def _bystander_field_rows(rows: list[dict]) -> list[dict]:
    """One row per (position, stream, bystander seat), `delta` the that
    seat's own win delta -- bystander deltas are per-seat lists in
    `phase3.jsonl`, flattened here so `_binned_readout` can treat "mean
    bystander delta" the same way it treats actor/counterparty. A stream
    with no bystander deltas at all (either fork errored) contributes no
    rows."""
    out = []
    for r in rows:
        if r["delta_bystanders"] is None:
            continue
        for d in r["delta_bystanders"]:
            out.append({"pid": r["pid"], "actor_gain": r["actor_gain"], "delta": d})
    return out


def _calibration_slope(rows: list[dict]) -> dict:
    """Readout 3: an OLS slope of realised Δ-win-actor on the gate's own
    claimed `actor_gain`, one point per position (streams pre-averaged),
    clustered bootstrap on the slope. A slope near the claimed gain's own
    scale says the gate's magnitude, not only its sign, tracks a real
    change in win probability; a slope near zero says it carries no such
    information."""
    means = _position_means(rows, "delta_actor")
    pairs = list(means.values())
    if len(pairs) < 2:
        return {"n_positions": len(pairs), "slope": 0.0, "intercept": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    def slope_of(sample: Sequence[tuple[float, float]]) -> tuple[float, float]:
        xs = [x for x, _ in sample]
        ys = [y for _, y in sample]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in sample)
        den = sum((x - mx) ** 2 for x in xs)
        if den == 0.0:
            return 0.0, my
        b = num / den
        a = my - b * mx
        return b, a

    slope, intercept = slope_of(pairs)
    rng = random.Random(5)
    n = len(pairs)
    slopes = []
    for _ in range(2000):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        b, _ = slope_of(sample)
        slopes.append(b)
    slopes.sort()
    lo = slopes[int(0.025 * 2000)]
    hi = slopes[min(int(0.975 * 2000), 1999)]
    return {"n_positions": len(pairs), "slope": slope, "intercept": intercept, "ci_low": lo, "ci_high": hi}


def _decide_tau(actor_readout: dict) -> tuple[str, str]:
    """The pre-stated decision rule (see this section's own note on where
    it comes from): scanning `GAIN_BINS` low to high, tau is the lower edge
    of the first bin whose paired-bootstrap CI on mean realised Δ-win-actor
    excludes zero and is positive (branch "resolved-at-bin", or
    "resolved-at-zero" when that is already the lowest bin) -- the smallest
    claimed gain above which the gate's own positive claim reliably
    corresponds to a measured win-probability gain. If no bin resolves,
    branch "unresolved": the judge does not support any tau, and none is
    adopted.
    """
    labels = [_bin_label(lo, hi) for lo, hi in GAIN_BINS]
    for i, (lo, hi) in enumerate(GAIN_BINS):
        label = labels[i]
        if actor_readout[label]["resolved_positive"]:
            branch = "resolved-at-zero" if i == 0 else "resolved-at-bin"
            return f"{lo:g}", branch
    return "unresolved", "unresolved"


def compute_phase3_readouts(rows: list[dict]) -> dict:
    n_positions = len(_cluster_by_position(rows))
    readout1 = _binned_readout(rows, "delta_actor")
    readout2 = {
        "counterparty": _binned_readout(rows, "delta_counterparty"),
        "bystanders": _binned_readout(_bystander_field_rows(rows), "delta"),
    }
    readout3 = _calibration_slope(rows)
    tau, branch = _decide_tau(readout1)

    errored = sum(1 for r in rows if r["untraded"]["error"] or r["traded"]["error"])
    chance_exhausted = sum(
        1 for r in rows
        if not (r["untraded"]["error"] or r["traded"]["error"])
        and (r["untraded"]["chance_exhausted"] or r["traded"]["chance_exhausted"])
    )
    capped = sum(
        1 for r in rows
        if not (r["untraded"]["error"] or r["traded"]["error"])
        and (r["untraded"]["winner"] is None or r["traded"]["winner"] is None)
    )
    return {
        "n_positions": n_positions,
        "n_rows": len(rows),
        "share_errored": errored / len(rows) if rows else 0.0,
        "share_chance_exhausted": chance_exhausted / len(rows) if rows else 0.0,
        "share_capped": capped / len(rows) if rows else 0.0,
        "readout_1_actor": readout1,
        "readout_2_counterparty": readout2["counterparty"],
        "readout_2_bystanders": readout2["bystanders"],
        "readout_3_calibration": readout3,
        "tau": tau,
        "tau_branch": branch,
    }


def phase3_table_md(readouts: dict) -> str:
    lines = [
        f"Positions: {readouts['n_positions']}, rows: {readouts['n_rows']}, "
        f"errored: {readouts['share_errored']*100:.1f}%, "
        f"chance-exhausted: {readouts['share_chance_exhausted']*100:.1f}%, "
        f"capped: {readouts['share_capped']*100:.1f}%\n",
    ]

    def bin_table(title: str, binned: dict) -> list[str]:
        out = [f"## {title}\n", "| gain bin | n | mean Δwin | 95% CI | resolved |", "|---|---|---|---|---|"]
        for label, b in binned.items():
            out.append(
                f"| {label} | {b['n_positions']} | {b['mean']:+.4f} | "
                f"[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] | {'yes' if b['resolved_positive'] else 'no'} |"
            )
        out.append("")
        return out

    lines += bin_table("Readout 1: actor", readouts["readout_1_actor"])
    lines += bin_table("Readout 2a: counterparty", readouts["readout_2_counterparty"])
    lines += bin_table("Readout 2b: bystanders", readouts["readout_2_bystanders"])

    c = readouts["readout_3_calibration"]
    lines.append("## Readout 3: calibration slope\n")
    lines.append(
        f"n={c['n_positions']}, slope={c['slope']:+.4f} "
        f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}], intercept={c['intercept']:+.4f}\n"
    )

    lines.append(f"**tau = {readouts['tau']}** (branch: {readouts['tau_branch']})\n")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    bank_p = sub.add_parser("bank", help="play and record heximax x4 games")
    bank_p.add_argument("--games", type=int, default=60)
    bank_p.add_argument("--seed", type=int, default=70000)
    bank_p.add_argument("--out", type=Path, default=Path("runs/eval/trade-lab/bank-c6.jsonl"))
    bank_p.add_argument("--workers", type=int, default=MAX_WORKERS)

    census_p = sub.add_parser("census", help="clear three rules over every position in a bank")
    census_p.add_argument("--bank", type=Path, default=Path("runs/eval/trade-lab/bank-c6.jsonl"))
    census_p.add_argument("--out", type=Path, default=Path("runs/eval/trade-lab/census.json"))
    census_p.add_argument("--workers", type=int, default=MAX_WORKERS)

    strategic_p = sub.add_parser(
        "strategic", help="one seat's gate shaded (tau-gate + exaggeration arms); is honesty a fixed point?"
    )
    strategic_p.add_argument("--bank", type=Path, default=Path("runs/eval/trade-lab/bank-c6.jsonl"))
    strategic_p.add_argument("--out", type=Path, default=Path("runs/eval/trade-lab/strategic.json"))
    strategic_p.add_argument("--workers", type=int, default=MAX_WORKERS)

    positions_p = sub.add_parser(
        "positions", help="the phase-3 judged position set: sample and write phase3-positions.jsonl"
    )
    positions_p.add_argument("--bank", type=Path, default=Path("runs/eval/trade-lab/bank-c6.jsonl"))
    positions_p.add_argument("--out", type=Path, default=Path("runs/eval/trade-lab/phase3-positions.jsonl"))
    positions_p.add_argument("--per-game-cap", type=int, default=SAMPLE_PER_GAME_CAP)
    positions_p.add_argument("--total", type=int, default=SAMPLE_TOTAL)
    positions_p.add_argument("--sample-seed", type=int, default=SAMPLE_SEED)
    positions_p.add_argument("--workers", type=int, default=MAX_WORKERS)

    judge_p = sub.add_parser("judge", help="the paired-chance judge (phase 3)")
    judge_p.add_argument("--positions", type=Path, default=Path("runs/eval/trade-lab/phase3-positions.jsonl"))
    judge_p.add_argument("--bank", type=Path, default=Path("runs/eval/trade-lab/bank-c6.jsonl"))
    judge_p.add_argument("--out", type=Path, default=Path("runs/eval/trade-lab/phase3.jsonl"))
    judge_p.add_argument("--streams", type=int, default=8)
    judge_p.add_argument("--cap", type=int, default=600)
    judge_p.add_argument("--workers", type=int, default=MAX_WORKERS)

    readouts_p = sub.add_parser("phase3-readouts", help="readouts 1-3 over a judge's output, and tau")
    readouts_p.add_argument("--rows", type=Path, default=Path("runs/eval/trade-lab/phase3.jsonl"))
    readouts_p.add_argument("--out", type=Path, default=Path("runs/eval/trade-lab/phase3-readouts.json"))

    args = parser.parse_args(argv)

    if args.cmd == "bank":
        t0 = time.time()
        n = run_bank(args.games, args.seed, args.out, args.workers)
        print(f"wrote {n} games to {args.out} ({time.time() - t0:.1f}s)", file=sys.stderr)
        return

    if args.cmd == "census":
        t0 = time.time()
        all_positions = run_census(args.bank, args.workers)
        wall = time.time() - t0
        readouts = compute_readouts(all_positions)
        print(f"{len(all_positions)} positions, wall time {wall:.1f}s", file=sys.stderr)
        print(table_md(readouts))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "readouts": readouts,
            "wall_time_seconds": wall,
            "positions": all_positions,
        }, indent=2))
        print(f"wrote {args.out}", file=sys.stderr)
        return

    if args.cmd == "strategic":
        t0 = time.time()
        records = run_strategic(args.bank, args.workers)
        wall = time.time() - t0
        readouts = compute_strategic_readouts(records)
        print(f"{readouts['n_positions_considered']} positions considered, wall time {wall:.1f}s", file=sys.stderr)
        print(strategic_table_md(readouts))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "readouts": readouts,
            "wall_time_seconds": wall,
            "records": records,
        }, indent=2))
        print(f"wrote {args.out}", file=sys.stderr)
        return

    if args.cmd == "positions":
        t0 = time.time()
        all_judged = run_judged_positions(args.bank, args.workers)
        sampled = sample_judged_positions(
            all_judged, per_game_cap=args.per_game_cap, total=args.total, seed=args.sample_seed,
        )
        wall = time.time() - t0
        n_games = len({r["game"] for r in all_judged}) if all_judged else 0
        print(
            f"{len(all_judged)} judged positions over {n_games} games, "
            f"sampled {len(sampled)}, wall time {wall:.1f}s", file=sys.stderr,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for row in sampled:
                handle.write(json.dumps(row) + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
        return

    if args.cmd == "judge":
        t0 = time.time()
        run_judge(
            args.positions, args.bank, args.out,
            streams=args.streams, cap=args.cap, workers=args.workers,
        )
        wall = time.time() - t0
        print(f"wall time {wall:.1f}s", file=sys.stderr)
        return

    if args.cmd == "phase3-readouts":
        rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
        readouts = compute_phase3_readouts(rows)
        print(phase3_table_md(readouts))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(readouts, indent=2))
        print(f"wrote {args.out}", file=sys.stderr)
        return


if __name__ == "__main__":
    main()
