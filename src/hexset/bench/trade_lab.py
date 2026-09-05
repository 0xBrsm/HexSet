# SPDX-License-Identifier: GPL-3.0-only
"""The trade event lifted out of the game: a static ablation over four
selection rules among candidates both private gates accept.

Registration: `agents/reference/trade-lab.md` (dev-hexset). Reads engine
primitives (`hexset.trading._candidates`, `hexset.trading.exchange`,
`Heximax._delta`/`Heximax.accepts`) and re-implements selection rules on its
own; `hexset.trading` is never modified. This module covers the *static*
half of the registration -- the position bank and the fixed-point census
over it; the strategic (shaded-gate) and rollout (box-judged) halves bolt on
later, over the same bank and the same `candidate_rows`/`select` primitives.

**Why a bank of records, not live games.** No bulk game records existed
before this (`trade-lab.md`'s "why the game logs cannot seed this"), and a
`hexset.record.Record` -- board, seed, actions -- replays forever. But a
replayed record's own `Game` seats nobody (`hexset.record.replay`'s
`game.gates` stays `None`), so its trade event never fires and its
`game.valuations` stay all-zero. The bots that played the recorded game are
deterministic given their construction, so `positions()` respawns the exact
same bots (same seed convention `bank()` used) and drives the game through
the *recorded actions* with `game.gates` seated -- the engine's own trade
event re-fires, live, exactly as it did when the game was first played (no
tree search is spent: only `valuation`/`accepts`/`accepts_many` run, never
`choose()`, since the actions are already decided), reproducing the
original trades bit for bit. `positions()` checks this: the replayed
trades and the recorded ones must match, or it raises.

**Where a position is captured.** The mechanic fires a trade event at MAIN
entry (lazily, on the current player's first publish/observe of the turn)
and again after every MAIN action the current player takes
(`hexset.trading`'s module docstring; `hexset.game.run_trade_event`,
`run_pending_event`). Both of those call sites bundle the state mutation and
the clearing into one call (`build_road` pays and places, *then* calls
`run_trade_event` in the same function body), so "the position right before
the event" cannot be read by calling public functions in a different order
-- there is no gap between the two to observe from outside. This module
monkeypatches `hexset.game.run_trade_event` itself (a name looked up fresh,
as a module global, every time `build_road`/`build_settlement`/... calls it
unqualified -- exactly the mechanism `tmp/win-stance/win_stance.py` uses to
register a stance and `rank_probe.py` uses to intercept `_best_clearing`,
just aimed at a different name) to snapshot the game (`hexset.game.imagine`,
the engine's sanctioned copy) immediately before delegating to the real
function. No file under `hexset/` is edited; the patch is inert (falls
straight through to the original) whenever nothing is capturing, so
`bank()`'s own games -- and every `imagine()`d child a bot's internal search
spawns, which never carries real `gates` -- pay only a `None`-check.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

import hexset.bots  # noqa: F401 -- registers "heximax" with hexset.arena
import hexset.game as _game_mod
from hexset import trading
from hexset.actions import apply
from hexset.arena import MAX_ACTIONS, mean_interval
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES
from hexset.bots.heximax import heximax
from hexset.bots.heximax.search import Heximax, _thin_copy
from hexset.game import Game, Phase, imagine, is_over, start, to_move
from hexset.record import Record, ReplayError, actions_of, board_of, from_json, read, record_game, write
from hexset.trading import Bundle, publish_valuation
from hexset.victory import victory_points

# The machine this runs on is shared; see `hexset.bench.trade_census`'s own
# `MAX_WORKERS` for the convention this mirrors.
MAX_WORKERS = 8

RULES: tuple[str, ...] = ("maximin-public", "actor", "egalitarian", "nash")

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
# The interception: snapshot every position a trade event is about to clear.
#
# `_capture_sink` is `None` outside of `positions()` (the default, and the
# state every worker process starts in): the patched wrapper then does
# nothing but call the original function straight through, so `bank()`'s own
# games -- and every hypothetical child a bot's tree search spawns via
# `imagine()`, which never carries real `gates` -- pay only the `is not
# None`/`is not None` checks below, not a snapshot.

_capture_sink: list[Game] | None = None


def _snapshot_position(game: Game) -> Game:
    """An inert copy of `game`, gates and all, safe to read but never to
    advance: `imagine` is the engine's sanctioned copy (a fresh `GameState`,
    a fresh ledger), `gates` is not one of the fields it carries by
    principle (a hypothetical must not reach real opponents), so it is set
    here to the *same* bot objects `game` is seated with -- "freshly
    spawned" once per replayed game (`positions()`, below), not once per
    position, and safe to share across every position from that one replay
    since `Heximax.valuation`/`accepts`/`_delta` are pure reads of a view,
    never of `self.rng`. `event_pending` is force-cleared so a later
    `copy.state(seat)` (every rule's own clearing loop calls it, to build
    each seat's view) never re-fires a second, spurious event on this
    already-frozen copy using its own borrowed gates.
    """
    copy = imagine(game, random.Random(0))
    copy.gates = game.gates
    copy.event_pending = False
    return copy


_original_run_trade_event = _game_mod.run_trade_event


def _capturing_run_trade_event(game: Game) -> None:
    if _capture_sink is not None and game.phase is Phase.MAIN and game.gates is not None:
        _capture_sink.append(_snapshot_position(game))
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
    global _capture_sink
    board = board_of(record)
    bots = _spawn_bots(record.seed, record.num_players, board)
    game = start(board, record.num_players, random.Random(record.seed))
    game.gates = tuple(bots)
    game.max_trades = None

    sink: list[Game] = []
    _capture_sink = sink
    # `game.trades` is reset every turn (`end_turn`) -- a per-turn scratch
    # list, not a whole-game log (`hexset.record.record_game`'s own reason
    # for snapshotting `game.trades[before:]` around both calls below rather
    # than reading `game.trades` once at the end). Mirrored here so the
    # fidelity check compares the *whole* replayed game, not its last turn.
    replayed: list[tuple[int, int, tuple[int, ...]]] = []
    try:
        for action in actions_of(record):
            seat = to_move(game)
            if game.publish_due(seat):
                before = len(game.trades)
                publish_valuation(game, seat, bots[seat])
                for t in game.trades[before:]:
                    replayed.append((t.a, t.b, tuple(t.received)))
            before = len(game.trades)
            apply(game, action)
            for t in game.trades[before:]:
                replayed.append((t.a, t.b, tuple(t.received)))
    finally:
        _capture_sink = None

    if (game.won_by, game.turns) != (record.winner, record.turns):
        raise ReplayError(
            f"trade_lab replay of game {game_index} ended {game.won_by} after "
            f"{game.turns} turns, the bank says {record.winner} after {record.turns}"
        )
    recorded = [(a, b, tuple(r)) for _step, a, b, r in record.trades]
    if replayed != recorded:
        raise ReplayError(
            f"trade_lab replay of game {game_index} re-published valuations "
            f"that cleared {len(replayed)} trades; the bank recorded {len(recorded)}"
        )

    for i, snapshot in enumerate(sink):
        yield Position(
            game_index=game_index,
            position_index=i,
            turn=snapshot.turns,
            actor=snapshot.current_player,
            game=snapshot,
        )


# ---------------------------------------------------------------------------
# Candidate rows: both public surpluses and both private gains, unfiltered.


@dataclass(frozen=True)
class CandidateRow:
    them: int
    bundle: Bundle
    pub_actor: float
    pub_counterparty: float
    gain_actor: float
    gain_counterparty: float


def candidate_rows(game: Game, me: int) -> list[CandidateRow]:
    """Every candidate `trading._candidates` enumerates for `me`, with both
    public surpluses (as the engine computes them) and both private gains
    (`Heximax._delta`, the gate's own reading, never a boolean accept/
    reject) -- unfiltered by either: `_rank_candidates_loop`/`_vectorized`
    already discard a public-surplus-negative candidate before a private
    rule ever gets to see it, so this walks `_candidates`'s raw output
    directly instead of reusing them."""
    state = game.state(0, hidden=False)
    vectors = game.valuations
    raw = list(trading._candidates(state, me, game.locked, vectors))
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
        pub_me = sum(vectors[me][r] * n for r, n in enumerate(bundle))
        pub_them = sum(-vectors[them][r] * n for r, n in enumerate(bundle))
        bot_me = game.gates[me]
        bot_them = game.gates[them]
        gain_me = bot_me._delta(view(me), me, me, bundle, them, bot_me._rank)
        mirror = tuple(-n for n in bundle)
        gain_them = bot_them._delta(view(them), them, them, mirror, me, bot_them._rank)
        rows.append(CandidateRow(them, bundle, pub_me, pub_them, gain_me, gain_them))
    return rows


def _canonical(row: CandidateRow) -> tuple[int, ...]:
    """Negated bundle: maximizing this picks the smallest bundle on a tie,
    the same `_rank_candidates_loop` trick (`hexset.trading`)."""
    return tuple(-n for n in row.bundle)


def clearing_set(rule: str, rows: Sequence[CandidateRow]) -> list[CandidateRow]:
    """Candidates both private gates strictly accept -- and, for
    `maximin-public` only, both public surpluses too (the shipped rule's own
    pre-filter; the other three rules skip it, per the registration)."""
    if rule == "maximin-public":
        return [
            r for r in rows
            if r.pub_actor > 0 and r.pub_counterparty > 0 and r.gain_actor > 0 and r.gain_counterparty > 0
        ]
    if rule in ("actor", "egalitarian", "nash"):
        return [r for r in rows if r.gain_actor > 0 and r.gain_counterparty > 0]
    raise ValueError(f"unknown rule: {rule}")


def select(rule: str, clearing: Sequence[CandidateRow]) -> CandidateRow | None:
    """The clearing candidate `rule` picks -- `clearing` is already the
    output of `clearing_set(rule, ...)`. `maximin-public` reproduces the
    shipped engine's own four-key tie-break (`hexset.trading.
    _rank_candidates_loop`) over the *public* surpluses; the other three
    break ties by the actor's own private gain, then canonical bundle
    order, then the lower counterparty seat -- purely for determinism,
    exactly as the shipped rule's own last two keys are."""
    if not clearing:
        return None
    if rule == "maximin-public":
        key = lambda r: (
            min(r.pub_actor, r.pub_counterparty), r.pub_actor,
            r.pub_actor + r.pub_counterparty, _canonical(r), -r.them,
        )
    elif rule == "actor":
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
    (`accepts`/`accepts_many`'s one shape); here the knower is the
    bystander but the hand that moves is the mover's, exactly the `target
    != knower` shape `_delta` itself routes to `_delta_reference` -- so
    this mirrors `_delta_reference`'s clone-based computation instead of
    the belief-shift fast path, built from the same exposed primitives
    (`_thin_copy`, `Heximax._move_hand`, `Heximax._read_row`) `_delta_
    reference` uses, reading `bystander`'s row rather than `mover`'s (the
    one reading neither method returns on its own) without editing
    `hexset/bots/heximax/search.py`.
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
        work.event_pending = False

        trades: list[dict] = []
        extra_admitted = 0
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

            if rule != "maximin-public":
                extra_admitted += sum(
                    1 for r in clearing if not (r.pub_actor > 0 and r.pub_counterparty > 0)
                )

            byst_sum = sum(
                _bystander_delta(work.gates[s], work.state(s), s, me, them, bundle)
                for s in range(n) if s not in (me, them)
            )

            state = work.state(0, hidden=False)
            before_hands = [hand[:] for hand in state.hands]
            trading.exchange(state, me, them, bundle)
            work.ledger.apply_hand_diff(before_hands, state.hands)
            work.trades.append(trading.Trade(me, them, bundle))
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
                "pub_actor": best.pub_actor,
                "pub_counterparty": best.pub_counterparty,
            })
            if first_trade is None:
                first_trade = (them, tuple(bundle))

        out[rule] = {"first_trade": first_trade, "extra_admitted": extra_admitted, "trades": trades}
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


def run_census(bank_path: Path, workers: int) -> list[dict]:
    lines = [line for line in bank_path.read_text(encoding="utf-8").splitlines() if line.strip()]
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
# Readouts


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

        mean_extra_admitted = (
            statistics.mean(p["rules"][rule]["extra_admitted"] for p in all_positions)
            if n_positions else 0.0
        )

        # Phase 2A -- gain distribution. `min_gains` is the weaker side of
        # each executed trade, the figure the verdict rule's "positive for
        # both parties" test ultimately turns on.
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
            "mean_extra_candidates_from_dropping_public_filter": mean_extra_admitted,
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
    lines.append("| rule | mean sum bystander ΔV | share lifts leader | mean extra candidates (no public filter) |")
    lines.append("|---|---|---|---|")
    for rule in RULES:
        b = readouts["rules"][rule]["bystander_damage"]
        extra = readouts["rules"][rule]["mean_extra_candidates_from_dropping_public_filter"]
        lines.append(
            f"| {rule} | {b['mean_sum_bystander_delta']:+.5f} | {b['share_lifts_leader']*100:.1f}% | {extra:.2f} |"
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

# Acceptance thresholds (`accept iff own gain > tau`), win probability. The
# registration (`trade-lab.md`) names {0, 0.005, 0.01, 0.02, 0.05}; phase 2's
# own brief supersedes that with these five, an order of magnitude finer,
# because phase 1 found mean private gains of ~3e-4 -- the registration's
# smallest nonzero tau (0.005) sits above nearly every gain in the bank and
# would make every tau > 0 indistinguishable from "never accept". These five
# straddle the observed distribution instead.
STRATEGIC_TAUS: tuple[float, ...] = (0.0, 1e-4, 5e-4, 1e-3, 5e-3)
# Exaggeration multipliers on the shaded seat's *reported* gain, selection-key
# only -- its own acceptance still reads the true gain against tau=0. Tested
# only under the two rules whose key reads a magnitude the exaggeration can
# move (`egalitarian`'s min, `nash`'s product); `actor`'s key is already the
# shaded seat's own claimed gain when it is the actor (exaggerating it always
# looks better, a trivial result) and `maximin-public`'s key never reads a
# private gain at all.
STRATEGIC_KS: tuple[float, ...] = (1.5, 2.0)
EXAGGERATION_RULES: tuple[str, ...] = ("egalitarian", "nash")

# Arm labels this module writes and reads back; `"tau=0"` (tau=0, k=1) is the
# honest baseline every other arm is a paired difference against.
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

    `shaded_seat` is `me` (the actor) for every row in one position (the
    actor is fixed for the whole event), or `them` for the subset of rows
    that name it as counterparty; a row naming neither is fully honest.
    Admission always reads the shaded seat's *true* gain against `tau`
    (never `k`); `k` -- 1.0 outside the exaggeration arms -- only scales the
    value fed to the selection key, mirroring `select`'s own tie-break
    exactly but keyed on the reported figure instead of the true one.
    """
    admitted: list[tuple[CandidateRow, float, float]] = []
    for r in rows:
        if rule == "maximin-public" and not (r.pub_actor > 0 and r.pub_counterparty > 0):
            continue
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
    if rule == "maximin-public":
        key = lambda t: (
            min(t[0].pub_actor, t[0].pub_counterparty), t[0].pub_actor,
            t[0].pub_actor + t[0].pub_counterparty, _canonical(t[0]), -t[0].them,
        )
    elif rule == "actor":
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
    the honest evaluator's own reading of its true gain (`CandidateRow.
    gain_actor`/`gain_counterparty`, never the exaggerated figure) summed
    over every executed trade the shaded seat was a party to, 0 if none."""
    work = imagine(game0, random.Random(0))
    work.gates = game0.gates
    work.event_pending = False

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
        work.trades.append(trading.Trade(me, them, bundle))
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
    """`{rule: {"qualifies": bool, "shaded_seat": int, "arms": {label: (realised, trades)}}}`
    for one position, or `None` if `pos`'s designated seat is not a party to
    any rule's honest clearing set (skipped -- shading a seat that was never
    going to trade here measures nothing)."""
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
        # `me == shaded_seat` alone is enough (the actor is a party to every
        # row); otherwise the seat must appear as `them` in at least one row
        # this rule's own honest clearing set admits.
        if me == shaded_seat:
            # `me` participates in every row `candidate_rows(game0, me)`
            # enumerates (it is always the actor's own candidate set), so
            # any honestly-clearing row at all makes the actor a party.
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
    lines = [line for line in bank_path.read_text(encoding="utf-8").splitlines() if line.strip()]
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
# Rollout judge: does the gate's claimed private gain track a realised change
# in win probability, played out with the game's own bots from both sides of
# a sampled trade?

ROLLOUT_STRATA: tuple[tuple[str, object], ...] = (
    ("1", lambda side: side == 1),
    ("2", lambda side: side == 2),
    ("≥3", lambda side: side >= 3),
)


def _stratum_of(max_side: int) -> str:
    for label, pred in ROLLOUT_STRATA:
        if pred(max_side):
            return label
    raise AssertionError(f"unreachable: max_side={max_side}")


def _sample_rule_trades(all_positions: list[dict], rule: str, per_stratum: int, rng: random.Random) -> list[dict]:
    """Up to `per_stratum` executed `rule` trades from each of the three
    `ROLLOUT_STRATA`, each identified by (game, position, trade_index) --
    `trade_index` is this trade's own position in that position's `rule`
    trade list, since census.json does not number them itself. Fewer than
    `per_stratum` in a stratum is not an error: every one available is
    taken (the registration's own "if available")."""
    pools: dict[str, list[dict]] = {label: [] for label, _ in ROLLOUT_STRATA}
    for p in all_positions:
        trades = p["rules"][rule]["trades"]
        for trade_index, t in enumerate(trades):
            pools[_stratum_of(t["max_side"])].append({**t, "trade_index": trade_index})
    sampled: list[dict] = []
    for label, _ in ROLLOUT_STRATA:
        pool = pools[label]
        k = min(per_stratum, len(pool))
        sampled.extend(rng.sample(pool, k))
    return sampled


def _replay_rule_to_index(
    game0: Game, me: int, rule: str, stop_after: int, cards_cap: int,
) -> tuple[Game, list[CandidateRow]]:
    """Clear `rule` honestly on a fresh working copy of `game0`, stopping
    right after the `stop_after`-th executed trade (0-based, inclusive) --
    "the trades the rule executed up to and including the sampled trade"
    (`trade-lab.md`'s rollout judge). Deterministic and bit-identical to
    `run_census`'s own per-position, per-rule loop (same `imagine(game0,
    random.Random(0))`), which is what lets the caller check the replayed
    trade against what census.json already recorded rather than trusting it
    silently."""
    work = imagine(game0, random.Random(0))
    work.gates = game0.gates
    work.event_pending = False
    executed_rows: list[CandidateRow] = []
    executed = 0
    while executed <= stop_after and executed < cards_cap:
        rows = candidate_rows(work, me)
        if not rows:
            break
        clearing = clearing_set(rule, rows)
        if not clearing:
            break
        best = select(rule, clearing)
        them, bundle = best.them, best.bundle
        state = work.state(0, hidden=False)
        before_hands = [hand[:] for hand in state.hands]
        trading.exchange(state, me, them, bundle)
        work.ledger.apply_hand_diff(before_hands, state.hands)
        work.trades.append(trading.Trade(me, them, bundle))
        work.trades_made += 1
        executed_rows.append(best)
        executed += 1
    return work, executed_rows


def _continue_and_finish(game0: Game, board, seed: str, action_cap: int) -> int | None:
    """Play `game0` to a winner (or `None`, capped at `action_cap`), seating
    a fresh `heximax` at every seat -- the position's own current player
    continues its own turn from wherever `game0` left off, and every trade
    event after this one runs through the engine's own shipped mechanic
    (`hexset.game.run_trade_event`, never a lab rule) since `work.gates` is
    the freshly seated bots, not this module's interception. Mirrors
    `hexset.arena.play`'s own loop exactly, starting from `game0` instead of
    `start()`."""
    rng = random.Random(seed)
    work = imagine(game0, rng)
    n = len(game0.valuations)
    bots = [heximax(board, random.Random(f"{seed}:{seat}")) for seat in range(n)]
    work.gates = tuple(bots)
    work.max_trades = None
    actions = 0
    while not is_over(work) and actions < action_cap:
        seat = to_move(work)
        bot = bots[seat]
        if work.publish_due(seat):
            publish_valuation(work, seat, bot)
        apply(work, bot.choose(work))
        actions += 1
    return work.won_by


def _rollout_job(job: tuple) -> dict:
    board, before_game, after_game, meta, playouts, action_cap = job

    def winshares(game0: Game, tag: str) -> list[float]:
        wins = [0, 0, 0, 0]
        for i in range(playouts):
            seed = f"{meta['game']}:{meta['position']}:{meta['rule']}:{meta['trade_index']}:{tag}:{i}"
            winner = _continue_and_finish(game0, board, seed, action_cap)
            if winner is not None:
                wins[winner] += 1
        return [w / playouts for w in wins]

    before_ws = winshares(before_game, "before")
    after_ws = winshares(after_game, "after")
    delta = [after_ws[s] - before_ws[s] for s in range(4)]
    actor, cp = meta["actor"], meta["counterparty"]
    bystanders = [s for s in range(4) if s not in (actor, cp)]

    out = dict(meta)
    out["before_winshare"] = before_ws
    out["after_winshare"] = after_ws
    out["delta_actor"] = delta[actor]
    out["delta_counterparty"] = delta[cp]
    out["delta_bystanders"] = [delta[b] for b in bystanders]
    return out


def build_rollout_jobs(
    bank_path: Path, census_path: Path, *, per_stratum: int = 100, playouts: int = 64,
    action_cap: int = MAX_ACTIONS, sample_seed: int = 1,
) -> tuple[list[tuple], dict[str, int]]:
    """Reconstruct every sampled trade's two starting games (replay is cheap
    -- no bot search runs here, only `candidate_rows`/`clearing_set`/`select`)
    and package one playout job per (rule, sampled trade). Raises
    `hexset.record.ReplayError` if a rule's own replayed trade sequence does
    not reproduce what census.json recorded at that index -- the same
    fidelity guarantee `positions()` already gives the bank itself."""
    census = json.loads(census_path.read_text(encoding="utf-8"))
    all_positions = census["positions"]
    records = list(read(str(bank_path)))

    rng = random.Random(sample_seed)
    counts: dict[str, int] = {}
    needed: dict[int, dict[int, list[dict]]] = {}
    for rule in RULES:
        sampled = _sample_rule_trades(all_positions, rule, per_stratum, rng)
        counts[rule] = len(sampled)
        for s in sampled:
            needed.setdefault(s["game"], {}).setdefault(s["position"], []).append({**s, "rule": rule})

    jobs: list[tuple] = []
    for game_index, positions_needed in needed.items():
        record = records[game_index]
        board = board_of(record)
        for pos in positions(record, game_index=game_index):
            entries = positions_needed.get(pos.position_index)
            if not entries:
                continue
            me = pos.game.current_player
            state0 = pos.game.state(0, hidden=False)
            cards_cap = sum(sum(hand) for hand in state0.hands)
            before_job = imagine(pos.game, random.Random(0))
            for entry in entries:
                rule = entry["rule"]
                trade_index = entry["trade_index"]
                after_game, executed_rows = _replay_rule_to_index(pos.game, me, rule, trade_index, cards_cap)
                if trade_index >= len(executed_rows):
                    raise ReplayError(
                        f"trade_lab rollouts: {rule} at game {game_index} position "
                        f"{pos.position_index} executed only {len(executed_rows)} trades "
                        f"replaying it, census.json recorded a trade at index {trade_index}"
                    )
                chosen = executed_rows[trade_index]
                if list(chosen.bundle) != list(entry["bundle"]):
                    raise ReplayError(
                        f"trade_lab rollouts: {rule} at game {game_index} position "
                        f"{pos.position_index} trade {trade_index} replayed bundle "
                        f"{list(chosen.bundle)}, census.json recorded {entry['bundle']}"
                    )
                after_game.gates = None
                meta = {
                    "rule": rule, "game": game_index, "position": pos.position_index,
                    "trade_index": trade_index, "turn": entry["turn"], "max_side": entry["max_side"],
                    "actor": entry["actor"], "counterparty": entry["counterparty"],
                    "actor_gain": entry["actor_gain"], "counterparty_gain": entry["counterparty_gain"],
                }
                jobs.append((board, before_job, after_game, meta, playouts, action_cap))
    return jobs, counts


def run_rollouts(
    bank_path: Path, census_path: Path, workers: int, *, per_stratum: int = 100,
    playouts: int = 64, action_cap: int = MAX_ACTIONS, sample_seed: int = 1,
) -> dict:
    jobs, counts = build_rollout_jobs(
        bank_path, census_path, per_stratum=per_stratum, playouts=playouts,
        action_cap=action_cap, sample_seed=sample_seed,
    )
    if workers > 1 and len(jobs) > 1:
        with Pool(workers) as pool:
            results = pool.map(_rollout_job, jobs, chunksize=1)
    else:
        results = [_rollout_job(job) for job in jobs]
    return {"samples_per_rule": counts, "n_jobs": len(jobs), "playouts": playouts, "results": results}


def compute_rollout_readouts(results: list[dict]) -> dict:
    out: dict = {"rules": {}}
    for rule in RULES:
        rows = [r for r in results if r["rule"] == rule]
        n = len(rows)
        actor_deltas = [r["delta_actor"] for r in rows]
        cp_deltas = [r["delta_counterparty"] for r in rows]
        byst_deltas = [d for r in rows for d in r["delta_bystanders"]]
        actor_est = mean_interval(actor_deltas)
        cp_est = mean_interval(cp_deltas)
        byst_est = mean_interval(byst_deltas)
        sign_actor = (
            sum(1 for r in rows if (r["actor_gain"] > 0) == (r["delta_actor"] > 0)) / n if n else 0.0
        )
        sign_cp = (
            sum(1 for r in rows if (r["counterparty_gain"] > 0) == (r["delta_counterparty"] > 0)) / n
            if n else 0.0
        )
        by_stratum: dict[str, dict] = {}
        for label, pred in ROLLOUT_STRATA:
            sub = [r for r in rows if pred(r["max_side"])]
            by_stratum[label] = {
                "n": len(sub),
                "mean_delta_actor": statistics.mean(r["delta_actor"] for r in sub) if sub else 0.0,
                "mean_delta_counterparty": statistics.mean(r["delta_counterparty"] for r in sub) if sub else 0.0,
            }
        out["rules"][rule] = {
            "n_sampled": n,
            "mean_delta_actor": actor_est.mean, "actor_ci": (actor_est.lower, actor_est.upper),
            "mean_delta_counterparty": cp_est.mean, "counterparty_ci": (cp_est.lower, cp_est.upper),
            "sign_agreement_actor": sign_actor, "sign_agreement_counterparty": sign_cp,
            "mean_delta_bystander": byst_est.mean, "bystander_ci": (byst_est.lower, byst_est.upper),
            "by_max_side_stratum": by_stratum,
        }
    return out


def rollout_table_md(readouts: dict) -> str:
    lines: list[str] = ["## Rollout judge\n"]
    lines.append(
        "| rule | n | mean Δ actor | 95% CI | mean Δ counterparty | 95% CI | "
        "sign agree actor | sign agree cp | mean Δ bystander |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for rule in RULES:
        r = readouts["rules"][rule]
        lines.append(
            f"| {rule} | {r['n_sampled']} | {r['mean_delta_actor']:+.4f} | "
            f"[{r['actor_ci'][0]:+.4f}, {r['actor_ci'][1]:+.4f}] | "
            f"{r['mean_delta_counterparty']:+.4f} | "
            f"[{r['counterparty_ci'][0]:+.4f}, {r['counterparty_ci'][1]:+.4f}] | "
            f"{r['sign_agreement_actor']*100:.1f}% | {r['sign_agreement_counterparty']*100:.1f}% | "
            f"{r['mean_delta_bystander']:+.4f} |"
        )
    lines.append("\n| rule | stratum | n | mean Δ actor | mean Δ counterparty |")
    lines.append("|---|---|---|---|---|")
    for rule in RULES:
        for label, _ in ROLLOUT_STRATA:
            s = readouts["rules"][rule]["by_max_side_stratum"][label]
            lines.append(
                f"| {rule} | {label} | {s['n']} | {s['mean_delta_actor']:+.4f} | "
                f"{s['mean_delta_counterparty']:+.4f} |"
            )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    bank_p = sub.add_parser("bank", help="play and record heximax x4 games")
    bank_p.add_argument("--games", type=int, default=60)
    bank_p.add_argument("--seed", type=int, default=70000)
    bank_p.add_argument("--out", type=Path, default=Path("runs/eval/trade-lab/bank.jsonl"))
    bank_p.add_argument("--workers", type=int, default=MAX_WORKERS)

    census_p = sub.add_parser("census", help="clear four rules over every position in a bank")
    census_p.add_argument("--bank", type=Path, default=Path("runs/eval/trade-lab/bank.jsonl"))
    census_p.add_argument("--out", type=Path, default=Path("runs/eval/trade-lab/census.json"))
    census_p.add_argument("--workers", type=int, default=MAX_WORKERS)

    strategic_p = sub.add_parser(
        "strategic", help="one seat's gate shaded (tau-gate + exaggeration arms); is honesty a fixed point?"
    )
    strategic_p.add_argument("--bank", type=Path, default=Path("runs/eval/trade-lab/bank.jsonl"))
    strategic_p.add_argument("--out", type=Path, default=Path("runs/eval/trade-lab/strategic.json"))
    strategic_p.add_argument("--workers", type=int, default=MAX_WORKERS)

    rollouts_p = sub.add_parser(
        "rollouts", help="300 executed trades per rule, judged by playouts with the game's own bots"
    )
    rollouts_p.add_argument("--bank", type=Path, default=Path("runs/eval/trade-lab/bank.jsonl"))
    rollouts_p.add_argument("--census", type=Path, default=Path("runs/eval/trade-lab/census.json"))
    rollouts_p.add_argument("--out", type=Path, default=Path("runs/eval/trade-lab/rollouts.json"))
    rollouts_p.add_argument("--per-stratum", type=int, default=100, help="sampled trades per max-side stratum, per rule")
    rollouts_p.add_argument("--playouts", type=int, default=64, help="playouts per starting game (before/after)")
    rollouts_p.add_argument("--action-cap", type=int, default=MAX_ACTIONS)
    rollouts_p.add_argument("--sample-seed", type=int, default=1)
    rollouts_p.add_argument("--workers", type=int, default=MAX_WORKERS)

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

    if args.cmd == "rollouts":
        t0 = time.time()
        payload = run_rollouts(
            args.bank, args.census, args.workers, per_stratum=args.per_stratum,
            playouts=args.playouts, action_cap=args.action_cap, sample_seed=args.sample_seed,
        )
        wall = time.time() - t0
        readouts = compute_rollout_readouts(payload["results"])
        print(
            f"{payload['n_jobs']} sampled trades, {args.playouts} playouts each side, "
            f"wall time {wall:.1f}s", file=sys.stderr,
        )
        print(rollout_table_md(readouts))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "readouts": readouts,
            "wall_time_seconds": wall,
            "samples_per_rule": payload["samples_per_rule"],
            "playouts": args.playouts,
            "results": payload["results"],
        }, indent=2))
        print(f"wrote {args.out}", file=sys.stderr)
        return


if __name__ == "__main__":
    main()
