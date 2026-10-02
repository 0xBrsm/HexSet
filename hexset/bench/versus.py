# SPDX-License-Identifier: GPL-3.0-only
"""Compare batched policies through LaneEnv on reproducible paired boards.

Margins and win-share intervals use board means as samples; win rates include
games the action cap cut short and games that ran out of turns. Wilson bounds
are game-level and assume an independence paired games do not have.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .metrics import json_metrics
from ..arena import MAX_ACTIONS, Standing, mean_interval, wilson
from ..game import MAX_TURNS
from ..casting import Caster, rotating, swapped
from ..gym.lanes import BoardBots, Episode, LaneEnv, Request

if TYPE_CHECKING:  # annotation-only
    from ..actions import Action
    from ..board.board import Board
    from ..bots import Bot
    from ..clients.policy import Checkpoint, Policy
    from ..game import Game


class BatchPolicy(Protocol):
    """The batched analogue of `hexset.bots.Bot`: a whole tick per call.

    Returns one answer per `Request`, in order: an `Action`, or a record whose
    `.action` is taken. An optional `gate(game, seat)` returns that seat's
    trade gate, which bargains on its own `TradeParams`; without one the seat
    never trades.
    """

    def act(self, requests: Sequence[Request]) -> Sequence[object]: ...


class BotPolicy:
    """A `hexset.bots.Bot` behind `BatchPolicy`, one bot per board."""

    def __init__(self, spawn: Callable[[Board], Bot], capacity: int = 256) -> None:
        self.bench = BoardBots(spawn, capacity=capacity)

    def _bot(self, game: Game) -> Bot:
        return self.bench.bot(game.state(0, hidden=False).board)

    def act(self, requests: Sequence[Request]) -> list[Action]:
        return [self._bot(r.game).choose(r.game) for r in requests]

    def gate(self, game: Game, seat: int) -> Bot:
        """The seated bot itself: `hexset.bots.Bot` is already the gate."""
        return self._bot(game)


class PolicyPolicy:
    """A `hexset.clients.policy.Policy` behind `BatchPolicy`.

    A bare policy never trades; pass the `checkpoint` too and the seat is gated
    by `hexset.clients.netbot.bot_for`, bargaining as the checkpoint declares.
    Custom policy action sampling must be seeded by its caller.
    """

    def __init__(self, policy: Policy, checkpoint: Checkpoint | None = None) -> None:
        self.policy = policy
        self.checkpoint = checkpoint

    def act(self, requests: Sequence[Request]) -> list[Action]:
        return self.policy.act_rows([(r.game, r.seat, r.options) for r in requests])

    def gate(self, game: Game, seat: int) -> object:
        """This seat's trade verdict: the checkpoint's own gate, or none."""
        if self.checkpoint is None:
            return None
        from ..clients.netbot import bot_for

        # An unseated NetworkBot prices every candidate at -1.0 and never trades.
        bot = bot_for(self.checkpoint)
        bot.seat = seat
        bot.seat_at(game)
        return bot


@dataclass(frozen=True)
class Verdict:
    """Batched evaluation outcomes and board-level uncertainty.

    `paired_vp`: learner seats' mean terminal victory points minus the
    reference's, averaged over boards. `games` is the readings, twice `boards`
    under `antithetic`. `exhausted` ran out of turns with no winner;
    `truncated` hit the action cap. Both are readings: neither side won.
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
    board_win_rate_low: float = 0.0
    board_win_rate_high: float = 1.0
    standings: tuple[Standing, ...] = ()
    # Every episode counted, under `episodes=True`; empty otherwise.
    episodes: tuple[Episode, ...] = ()

    def metrics(self) -> dict:
        """Flat metrics for training logs, including interval assumptions."""
        return json_metrics({
            "games": self.games,
            "boards": self.boards,
            "antithetic": self.antithetic,
            "wins": self.wins,
            "win_rate": self.win_rate,
            "wilson_low": self.wilson_low,
            "wilson_high": self.wilson_high,
            "board_win_rate_low": self.board_win_rate_low,
            "board_win_rate_high": self.board_win_rate_high,
            "interval_unit": "board",
            "win_rate_denominator": "all games, including unfinished",
            "wilson_assumption": "independent games; paired games are correlated",
            "paired_vp": self.paired_vp,
            "paired_vp_low": self.paired_vp_low,
            "paired_vp_high": self.paired_vp_high,
            "turns_mean": self.turns_mean,
            "turns_median": self.turns_median,
            "turns_max": self.turns_max,
            "exhausted": self.exhausted,
            "truncated": self.truncated,
            "seconds": self.seconds,
        })


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
    """Learner minus reference mean victory points, and how the game ended."""
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

    An id with no policy is unanswered: `LaneEnv.step` plays it by its own bot.
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
    turn_cap: int = MAX_TURNS,
    antithetic: bool = True,
    learner: int = 0,
    names: Mapping[int, str] | None = None,
    gates: Mapping[int, Callable[[Game, int], object]] | None = None,
    episodes: bool = False,
    records: bool = False,
) -> Verdict:
    """Play `games` games between batched policies and report the verdict.

    `policies` maps policy id to what answers for it; `learner` names the id the
    margin is reported *for*, the rest being the reference. With no `caster`,
    ids are seated as `compete`'s own lineup, rotating by game index.
    `antithetic` plays every board both ways, averaging its two readings; it
    needs exactly two policies and an even `games`, which counts games, not
    boards. `lanes` changes only the wall clock. Each seat bargains as its
    own gate declares (`gates`, or a policy's own `gate`). `records=True`
    attaches a `Record` to each episode, reaching the caller only under
    `episodes=True`.

    A game that reaches `turn_cap` with no winner ends there: a reading in
    which neither side won, scored on the points it reached and counted in
    `Verdict.exhausted`. `arena.compete` raises `Exhausted` instead.
    """
    if antithetic and games % 2:
        raise ValueError("an antithetic duel requires an even number of games")
    if lanes < 1 or action_cap < 1:
        raise ValueError("lanes and action_cap must be positive")
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

    # A seat with its own trade gate is gated here too, as under `arena.play`.
    seated: dict[int, Callable[[Game, int], object]] = {}
    for pid, policy in policies.items():
        gate = getattr(policy, "gate", None)
        if gate is not None:
            seated[pid] = gate
    seated.update(gates or {})

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
            turn_cap=turn_cap,
            caster=cohort_caster,
            gates=seated,
            records=records and episodes,
        )
        for episode in env.drain(answer):
            readings.setdefault(episode.index, []).append(_read(episode, learner))
            if episodes:
                played.append(episode)
    elapsed = time.perf_counter() - started

    if antithetic:
        # Only boards seen both ways can have the seat term removed.
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
    win_share = mean_interval(
        [sum(row.won for row in pair) / 2 for pair in both]
        if antithetic else [float(row.won) for row in rows]
    )
    return Verdict(
        games=n,
        boards=boards,
        antithetic=antithetic,
        wins=wins,
        win_rate=wins / n if n else 0.0,
        wilson_low=low,
        wilson_high=high,
        board_win_rate_low=max(0.0, win_share.lower),
        board_win_rate_high=min(1.0, win_share.upper),
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
