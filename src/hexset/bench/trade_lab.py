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
from hexset.board.board import random_base_board
from hexset.board.terrain import NUM_RESOURCES
from hexset.bots.heximax import heximax
from hexset.bots.heximax.search import Heximax, _thin_copy
from hexset.game import Game, Phase, imagine, start, to_move
from hexset.record import Record, ReplayError, actions_of, board_of, from_json, record_game, write
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


if __name__ == "__main__":
    main()
