# SPDX-License-Identifier: GPL-3.0-only
"""A duel between batched policies, over lanes rather than a process pool.

`hexset.arena.compete` is the tournament: it owns the pairing law, the
rotation, the Wilson interval and the per-game record, and it plays every
game through a `hexset.bots.Bot` that answers one position at a time. That
is the right shape for a scripted bot and the wrong one for a network,
where the measured dispatch toll is ~1.5 ms per forward against ~25 us per
position -- one position per call wastes the hardware by a factor of fifty,
which is the whole reason `hexset.gym.lanes` exists.

So this is `compete`'s verdict over `LaneEnv`'s driver: the same boards
under the same casts, `arena.wilson` on the same counts, `arena.Standing`
for the standings, and one batched call per tick across every lane. A
training loop evaluating a checkpoint needs exactly this and had been
writing its own; two evaluations of the same pair that derive the boards
alike rather than calling the one law are two evaluations that can disagree
about what they measured.

**The pairing law is `compete`'s.** Under `antithetic`, the two halves of a
pair are the same board played under complementary casts -- ids exchanged,
so the seat term cancels per board rather than only in the mean -- and the
board index is the pair, not the game. Written as casters
(`hexset.casting.rotating`, `hexset.casting.swapped`) instead of as
`compete`'s inline `divmod`, because a lane environment is cast by a pure
function of the game index and a tournament is not.

**What it reports.** The win rate with its Wilson interval, and the finer
instrument: paired terminal victory points, the learner's seats' mean minus
the reference's within a board, with a normal interval on the per-board
mean (`arena.mean_interval`). Boards seen only one way are dropped, since a
board played once reintroduces exactly the seat term the pairing exists to
delete.
"""

from __future__ import annotations

import inspect
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ..arena import MAX_ACTIONS, Standing, mean_interval, wilson
from ..casting import Caster, rotating, swapped
from ..gym.lanes import BoardBots, Episode, LaneEnv, Request

if TYPE_CHECKING:  # annotation-only: no import cost, and no cycle through bots
    from ..actions import Action
    from ..board.board import Board
    from ..bots import Bot
    from ..clients.policy import Checkpoint, Policy
    from ..game import Game


class BatchPolicy(Protocol):
    """The batched analogue of `hexset.bots.Bot`: a whole tick per call.

    One call per tick is the entire point -- a torch or ONNX implementation
    collates, moves and runs the network once for every lane it holds
    rather than once per position. Must return one answer per `Request`, in
    order.

    An answer is a `hexset.actions.Action`. A richer per-decision record is
    accepted too and its `.action` taken, so a training loop whose policy
    already returns a log-probability and a value estimate alongside the
    move needs no adapter to be evaluated here.

    A policy may also carry `gate(game, seat)`, returning the object that
    answers that seat's private trade verdict for that game
    (`hexset.trading`); one that does not is seated as a seat that never
    trades. `BotPolicy` supplies its own bot, which is already a gate.

    A gate written `gate(game, seat, max_trades)` is handed the run's own
    trade budget instead (`compete_batched`'s `max_trades`, whatever the
    caller passed). A checkpoint-backed gate needs it: `bot_for`'s budget
    defaults to the switch the *checkpoint* trained under, so a duel run at
    `max_trades=0` would otherwise seat a trading gate into a no-trade
    evaluation and measure a different game than it asked for. Which form a
    gate takes is read off its signature, so a two-argument gate is
    unaffected.
    """

    def act(self, requests: Sequence[Request]) -> Sequence[object]: ...


class BotPolicy:
    """A `hexset.bots.Bot` behind `BatchPolicy`, one bot per board.

    A scripted bot pays no dispatch toll and gains nothing from batching,
    but a duel needs both sides behind one protocol, and the handcrafted
    evaluators cache per-vertex pips for the board they were built on while
    every lane plays its own board. `hexset.gym.lanes.BoardBots` already
    keeps a bot per board under a bounded cache, so this is that bench plus
    the two methods the protocol asks for.
    """

    def __init__(self, spawn: Callable[[Board], Bot], capacity: int = 256) -> None:
        self.bench = BoardBots(spawn, capacity=capacity)

    def _bot(self, game: Game) -> Bot:
        # public field: a bot reads the board off the true state, as every
        # `hexset.arena` caller does.
        return self.bench.bot(game.state(0, hidden=False).board)

    def act(self, requests: Sequence[Request]) -> list[Action]:
        return [self._bot(r.game).choose(r.game) for r in requests]

    def gate(self, game: Game, seat: int) -> Bot:
        """The seated bot itself: `hexset.bots.Bot` is already the gate."""
        return self._bot(game)


class PolicyPolicy:
    """A `hexset.clients.policy.Policy` behind `BatchPolicy`.

    A `Policy` is already batched -- `act_rows` takes a whole list of
    positions and answers one action each -- so this is the adapter and
    nothing more: the tick's requests as `(game, seat, options)` rows, in
    order. It exists so that evaluating a checkpoint means loading a runtime
    and seating it, rather than every caller writing its own eight-line
    shim and each one differing about what a row is.

    A bare policy is seated as a seat that never trades, which is right for
    a value-only yardstick and wrong for measuring a checkpoint on the game
    it learned. Pass the `checkpoint` as well and the seat is gated by
    `hexset.clients.netbot.bot_for`, the one trade gate a checkpoint has --
    priced on the run's own trade budget, since `gate` takes it.
    """

    def __init__(self, policy: Policy, checkpoint: Checkpoint | None = None) -> None:
        self.policy = policy
        self.checkpoint = checkpoint

    def act(self, requests: Sequence[Request]) -> list[Action]:
        return self.policy.act_rows([(r.game, r.seat, r.options) for r in requests])

    def gate(self, game: Game, seat: int, max_trades: int | None = None) -> object:
        """This seat's trade verdict: the checkpoint's own gate, or none."""
        if self.checkpoint is None:
            return None
        from ..clients.netbot import bot_for

        return bot_for(self.checkpoint, max_trades=max_trades)


def _budgeted(
    gate: Callable[..., object], max_trades: int | None
) -> Callable[[Game, int], object]:
    """`gate` as the two-argument seater `hexset.gym.lanes` calls.

    A gate that spells its signature `(game, seat, max_trades)` is asking
    for the run's trade budget and is given it; every other gate is passed
    through untouched. Read off the signature rather than switched on by a
    flag, because the caller seating a policy is not the one that wrote its
    gate.
    """
    try:
        inspect.signature(gate).bind(None, None, None)
    except (TypeError, ValueError):
        return gate
    return lambda game, seat: gate(game, seat, max_trades)


@dataclass(frozen=True)
class Verdict:
    """What a duel measured, in `hexn.train._antithetic`'s own vocabulary.

    `paired_vp` is the headline: the learner's seats' mean terminal victory
    points minus the reference's, averaged over the boards, with the seat
    term differenced out. `wins`/`win_rate` are the coarser reading kept
    beside it because a margin is not a result anyone bets on.

    `boards` is how many boards were counted, `games` the readings that made
    them -- twice `boards` under `antithetic`. `exhausted` is a game that
    ran out of turns without a winner, which `truncated` (the action cap
    stopped it) is deliberately not folded into: they are different faults.
    """

    games: int
    boards: int
    antithetic: bool
    wins: int
    win_rate: float
    wilson_low: float
    wilson_high: float
    paired_vp: float
    paired_vp_low: float
    paired_vp_high: float
    turns_mean: float
    turns_median: float
    turns_max: int
    exhausted: int
    truncated: int
    seconds: float
    standings: tuple[Standing, ...] = ()
    # Every episode counted, when `episodes=True` asked for them: what a
    # `Tournament.records` is to `compete`, and what a caller checking the
    # verdict against the games themselves needs. Empty otherwise, never
    # partially filled.
    episodes: tuple[Episode, ...] = ()

    def metrics(self) -> dict:
        """The verdict as the flat dict a training run logs.

        Key for key what `hexn.train._antithetic` returned, so a run can
        call this instead of its own harness without a line of its logging
        or its plotting changing.
        """
        return {
            "games": self.games,
            "boards": self.boards,
            "antithetic": self.antithetic,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "wilson_low": self.wilson_low,
            "wilson_high": self.wilson_high,
            "paired_vp": self.paired_vp,
            "paired_vp_low": self.paired_vp_low,
            "paired_vp_high": self.paired_vp_high,
            "turns_mean": self.turns_mean,
            "turns_median": self.turns_median,
            "turns_max": self.turns_max,
            "exhausted": self.exhausted,
            "truncated": self.truncated,
            # The wall clock the duel took. A logged evaluation that cannot
            # say how long it cost is one nobody can budget the next one
            # from, and `seconds` was on the `Verdict` all along.
            "seconds": self.seconds,
        }


@dataclass(frozen=True)
class _Reading:
    """One episode reduced to what the verdict counts."""

    margin: float
    won: bool
    winner: int | None
    turns: int
    truncated: bool
    exhausted: bool


def _read(episode: Episode, learner: int) -> _Reading:
    """Learner minus reference mean victory points, and how the game ended.

    "Exhausted" is a game that reached `hexset.game.MAX_TURNS` with nobody
    over the line; `Outcome.truncated` already separates that from a game
    the action cap cut off, so no engine constant is needed here.
    """
    mine = [s for s, pid in enumerate(episode.cast) if pid == learner]
    theirs = [s for s, pid in enumerate(episode.cast) if pid != learner]
    if not mine or not theirs:
        raise ValueError(
            f"cast {episode.cast} seats no duel for policy {learner}"
        )
    points = episode.outcome.points
    winner = episode.outcome.winner
    return _Reading(
        margin=sum(points[s] for s in mine) / len(mine)
        - sum(points[s] for s in theirs) / len(theirs),
        won=winner in mine,
        winner=None if winner is None else episode.cast[winner],
        turns=episode.outcome.turns,
        truncated=episode.outcome.truncated,
        exhausted=winner is None and not episode.outcome.truncated,
    )


def _answer(
    policies: Mapping[int, BatchPolicy],
) -> Callable[[Sequence[Request]], list[Action | None]]:
    """Route a tick to its policies, one call each, and reassemble in order.

    The obvious implementation -- ask the seat's policy about the seat --
    pays the dispatch toll once per position instead of once per tick,
    which is the cost this whole module exists to avoid. So the batch is
    split by policy id, each policy is handed its whole share in one call,
    and the answers go back into request order. An id with no policy is
    left unanswered, which `LaneEnv.step` reads as "played by its own bot".
    """

    def answer(requests: Sequence[Request]) -> list[Action | None]:
        out: list[Action | None] = [None] * len(requests)
        shares: dict[int, list[int]] = {}
        for slot, request in enumerate(requests):
            shares.setdefault(request.policy, []).append(slot)
        for pid, slots in shares.items():
            policy = policies.get(pid)
            if policy is None:
                continue
            answers = policy.act([requests[slot] for slot in slots])
            if len(answers) != len(slots):
                raise ValueError(
                    f"policy {pid} answered {len(answers)} of {len(slots)} requests"
                )
            for slot, choice in zip(slots, answers):
                out[slot] = getattr(choice, "action", choice)
        return out

    return answer


def compete_batched(
    policies: Mapping[int, BatchPolicy],
    games: int,
    *,
    caster: Caster | None = None,
    players: int = 4,
    seed: int = 0,
    lanes: int = 8,
    action_cap: int = MAX_ACTIONS,
    max_trades: int | None = None,
    antithetic: bool = True,
    learner: int = 0,
    names: Mapping[int, str] | None = None,
    gates: Mapping[int, Callable[[Game, int], object]] | None = None,
    episodes: bool = False,
    records: bool = False,
) -> Verdict:
    """Play `games` games between batched policies and report the verdict.

    `policies` maps policy id to the thing that answers for it; `learner`
    names the id the margin is reported *for*, every other id being the
    reference. With no `caster`, the ids are seated as `compete`'s own
    lineup -- an equal, adjacent share of the seats each, rotating by game
    index (`hexset.casting.rotating`) -- so a duel run here and the same
    duel run through `compete` play the same boards under the same casts.

    `antithetic` plays every board **both** ways: half the games under
    `caster`, half under `swapped(caster)`, same indices and so the same
    boards and the same dice, and averages a board's two readings. The seat
    term then differences out exactly instead of being counted as ordinary
    noise -- it was 55% of a single-order duel's variance on this engine --
    and a self-duel reads 0.0, which is the test. It needs exactly two
    policies, since exchanging more than two ids is not one complement but
    a choice of several.

    `games` counts games, not boards, so an antithetic run halves it
    between the two orders and an odd count loses a game rather than
    silently playing a board once.

    `lanes` changes only the wall clock: a game is a function of `(seed,
    index)`, so the verdict at one lane and at sixty-four is the same
    verdict.

    `max_trades` is passed through rather than defaulted, because an
    evaluation that trades where the run did not is measuring a different
    game. It reaches a policy's own gate too, where that gate asks for it
    (`_budgeted`).

    `records=True` has every episode carry a `hexset.record.Record` of its
    game (`hexset.gym.lanes.LaneEnv(records=True)`) -- what
    `arena.Tournament.records` is to `compete`. A record only reaches the
    caller on an `Episode`, so it is passed through only under
    `episodes=True` and costs nothing without it.
    """
    if len(policies) < 2:
        raise ValueError("a duel needs at least two policies")
    if learner not in policies:
        raise ValueError(f"policy {learner} is not one of the duellists")
    ids = sorted(policies)

    if caster is None:
        if players % len(ids):
            raise ValueError(
                f"{len(ids)} policies do not divide evenly over {players} seats; "
                "pass a caster"
            )
        share = players // len(ids)
        caster = rotating(tuple(pid for pid in ids for _ in range(share)))

    # A seat that answers its own trade gate answers it here too, so a game
    # played against a scripted reference trades exactly as it would under
    # `arena.play` -- where seating a bot *is* seating its gate.
    seated: dict[int, Callable[[Game, int], object]] = {}
    for pid, policy in policies.items():
        gate = getattr(policy, "gate", None)
        if gate is not None:
            seated[pid] = gate
    seated.update(gates or {})
    seated = {pid: _budgeted(gate, max_trades) for pid, gate in seated.items()}

    if antithetic:
        if len(ids) != 2:
            raise ValueError(
                "an antithetic pair exchanges two policies' seats; "
                f"got {len(ids)}"
            )
        half = games // 2
        if half == 0:
            raise ValueError("an antithetic duel needs at least two games")
        cohorts = ((caster, half), (swapped(caster, *ids), half))
    else:
        if games < 1:
            raise ValueError("a duel needs at least one game")
        cohorts = ((caster, games),)

    answer = _answer(policies)
    started = time.perf_counter()
    played: list[Episode] = []
    readings: dict[int, list[_Reading]] = {}
    for cohort_caster, count in cohorts:
        env = LaneEnv(
            players,
            seed,
            min(lanes, count),
            deal=count,
            action_cap=action_cap,
            caster=cohort_caster,
            gates=seated,
            max_trades=max_trades,
            records=records and episodes,
        )
        for episode in env.drain(answer):
            readings.setdefault(episode.index, []).append(_read(episode, learner))
            if episodes:
                played.append(episode)
    elapsed = time.perf_counter() - started

    if antithetic:
        # Only boards seen both ways can have the seat term removed; a board
        # seen once reintroduces exactly what the pairing exists to delete.
        both = [pair for pair in readings.values() if len(pair) == 2]
        paired = [(one.margin + other.margin) / 2 for one, other in both]
        rows = [row for pair in both for row in pair]
        boards = len(both)
    else:
        rows = [row for pair in readings.values() for row in pair]
        paired = [row.margin for row in rows]
        boards = len(rows)

    n = len(rows)
    wins = sum(1 for row in rows if row.won)
    turns = [row.turns for row in rows]
    low, high = wilson(wins, n) if n else (0.0, 0.0)
    margin = mean_interval(paired)
    return Verdict(
        games=n,
        boards=boards,
        antithetic=antithetic,
        wins=wins,
        win_rate=wins / n if n else 0.0,
        wilson_low=low,
        wilson_high=high,
        paired_vp=margin.mean,
        paired_vp_low=margin.lower,
        paired_vp_high=margin.upper,
        turns_mean=statistics.mean(turns) if turns else 0.0,
        turns_median=statistics.median(turns) if turns else 0.0,
        turns_max=max(turns) if turns else 0,
        exhausted=sum(1 for row in rows if row.exhausted),
        truncated=sum(1 for row in rows if row.truncated),
        seconds=elapsed,
        standings=tuple(
            Standing(
                name=(names or {}).get(pid, f"policy {pid}"),
                wins=sum(1 for row in rows if row.winner == pid),
                games=n,
            )
            for pid in ids
        ),
        episodes=tuple(played) if episodes else (),
    )


__all__ = [
    "BatchPolicy",
    "BotPolicy",
    "PolicyPolicy",
    "Verdict",
    "compete_batched",
]
