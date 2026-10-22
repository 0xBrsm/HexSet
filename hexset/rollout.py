# SPDX-License-Identifier: GPL-3.0-only
"""Paired rollouts: a decision searched by playing its candidate moves out on
shared chance.

At a position the policy is unsure of, its `top` moves are each played out in
`worlds` sampled worlds. A world fixes everything chance would otherwise vary
between the candidates: the hidden cards, drawn from the mover's own belief as
`hexset.mcts.Search.worlds` draws them; the development deck's order; and the
dice and steal streams, split (`hexset.chance.Live`) so the n-th roll and the
n-th steal are the same in every candidate's copy whatever was played before
them. After the candidate, the policy plays every seat, one forward per
decision, to the game's end or `horizon` turns on.

Candidates are compared world by world. Where a move does not change how a
world plays out, the shared chance cancels, and what is left is the
difference the move makes: a far smaller variance than independent playouts
give, so a few dozen worlds separate moves whose difference independent
samples would bury in the dice.

Only the root branches. Below it the policy plays, and the policy never sees
a world's coming rolls, so no choice in a playout plans around dice it could
not know.

Every decision is kept in `PairedRollouts.decision_log` with what it was
made from: the outcome of each candidate in each world, and the value head's
read of each playout at the mover's turn starts `marks` turns on. Fewer
worlds, fewer candidates or a shorter horizon can then be scored on the same
playouts afterwards, without playing again.
"""

from __future__ import annotations

import multiprocessing
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .actions import Action, apply, legal_actions
from .chance import Balanced, Live
from .game import Game, Phase, imagine, is_over, to_move
from .victory import public_victory_points
from .view import HoldReading

__all__ = [
    "MARKS", "SETUP_PHASES", "STEP_LIMIT", "Decision", "Lanes", "PairedRollouts", "RolloutPool", "RolloutBatch",
    "paired_copy", "play_lanes",
]

#: The most decisions a playout makes before it is scored where it
#: stands, by the value head: a bound on a game the policy never finishes.
STEP_LIMIT = 3000

#: Turns after the root (every seat's, as `Game.turns` counts them) at which
#: each playout's value is read for the decision log, at the mover's first
#: turn start at least that far on.
MARKS = (2, 4, 8, 16)

#: The opening's phases, which `PairedRollouts(full_setup=True)` always searches.
SETUP_PHASES = (Phase.SETUP_SETTLEMENT, Phase.SETUP_ROAD)


@dataclass(frozen=True)
class Decision:
    """One decision as `PairedRollouts.decide` made it: the root's `options`,
    the policy's `prior` over them and its `value` (board-seat order), the
    option indices played out (`candidates`, highest prior first), each one's
    mean outcome for the mover over the worlds (`scores`, empty where nothing
    was played out), and the option `chosen`. `worlds` is how many worlds the
    candidates were played out in: 0 for a root the policy decides alone.

    What it was made from, for the log: the root's `turn` and `phase`, the
    deciding `seat`, every seat's public victory `points`; `outcomes`, one
    row per world of each candidate's score there (1 or 0 for a playout that
    ended, the value head's read for one stopped short); `marks` and
    `mark_values`, per mark, the same table read by the value head at the
    mover's first turn start that many turns on (a playout over by then
    keeps its score); `after_values`, the same table read right after each
    candidate is applied, before any playout: the one-forward comparison,
    kept where marks are read; `steps`, the lockstep decisions the playouts took;
    `row_steps`, the playout moves made over every copy, and
    `mark_row_steps`, those it took to reach each mark (a copy that ended
    first counts its own), so a horizon's cost reads off the same playouts;
    `truncated`, the playouts stopped short; and `cpu_seconds`, the process
    time the decision took."""

    options: tuple[Action, ...]
    prior: np.ndarray
    value: tuple[float, ...]
    candidates: tuple[int, ...]
    scores: tuple[float, ...]
    chosen: int
    worlds: int
    turn: int = 0
    phase: str = ""
    seat: int = 0
    points: tuple[int, ...] = ()
    outcomes: tuple[tuple[float, ...], ...] = ()
    marks: tuple[int, ...] = ()
    mark_values: tuple[tuple[tuple[float, ...], ...], ...] = ()
    after_values: tuple[tuple[float, ...], ...] = ()
    steps: int = 0
    row_steps: int = 0
    mark_row_steps: tuple[int, ...] = ()
    truncated: int = 0
    cpu_seconds: float = 0.0

    def to_json(self) -> dict:
        """The decision as one JSON object. A decision the policy made alone
        keeps its top prior and the option count, not the full prior."""
        out = {
            "turn": self.turn, "phase": self.phase, "seat": self.seat,
            "points": list(self.points), "options": len(self.options),
            "top_prior": round(float(self.prior[self.candidates[0]]), 4),
            "chosen": self.chosen, "worlds": self.worlds,
            "cpu_seconds": round(self.cpu_seconds, 4),
        }
        if not self.worlds:
            return out
        r = lambda x: round(float(x), 4)  # noqa: E731
        out.update(
            candidates=list(self.candidates),
            actions=[[self.options[i].type.name, self.options[i].a, self.options[i].b]
                     for i in self.candidates],
            prior=[r(self.prior[i]) for i in self.candidates],
            value=[r(v) for v in self.value],
            scores=[r(s) for s in self.scores],
            outcomes=[[_outcome(x) for x in row] for row in self.outcomes],
            marks=list(self.marks),
            mark_values=[[[_outcome(x) for x in row] for row in table]
                         for table in self.mark_values],
            after_values=[[_outcome(x) for x in row] for row in self.after_values],
            steps=self.steps, truncated=self.truncated,
            row_steps=self.row_steps, mark_row_steps=list(self.mark_row_steps),
        )
        return out


def _outcome(x: float) -> float | int:
    """A playout's score as the log writes it: a finished game's 1 or 0
    exactly, a value head's read to three places."""
    return int(x) if x in (0.0, 1.0) else round(float(x), 3)


def paired_copy(world: Game, seed: int, seat: int) -> Game:
    """A copy of `world` that draws its chance off `seed`, split: two copies
    on one seed roll the same n-th roll and take the same n-th steal, and
    deal from the same deck order, whatever was played in either. `seat`
    names whose world it is (`hexset.game.imagine`)."""
    rng = random.Random(seed)
    copy = imagine(world, rng, randomize_deck=False, seat=seat)
    dice = copy._state.dice
    copy.chance = Live(rng, split=True) if dice is None else Balanced.from_deck(rng, dice, split=True)
    return copy


class PairedRollouts:
    """Decides a position by paired rollouts of the policy's own top moves.

    `policy` is a batched `hexset.clients.policy.Policy`: `score_rows` for the
    root's prior and value, `act_rows` for every move of a playout and
    `value_rows` for a playout stopped short of the game's end.

    `worlds` is the number of shared-chance worlds each candidate plays out
    in; `top`, how many of the policy's moves, by prior, are candidates. A
    root with one legal move, or, under `decisive`, whose prior on one move
    is at least `decisive`, plays that move with no playout. Under
    `full_setup`, `decisive` skips no setup root (`SETUP_PHASES`): every
    opening settlement and road with a choice is played out.

    A playout runs to the game's end, scored 1 where the mover won and 0
    otherwise, or under `horizon` stops at the first turn start of the mover
    at least `horizon` turns (every seat's, counted as `Game.turns` counts
    them) after the root's. A playout stopped short of the end -- at the
    horizon, at `STEP_LIMIT`, or where the seat to move has no legal move --
    is scored by the value head's chance for the mover, read from the
    mover's own frame whoever is to move there, so the mover's own cards in
    that world stay known to the read. The candidate with the highest mean
    score over the worlds is played, the higher prior on a tie.
    `truncated_rollouts` counts the playouts scored that way over this
    searcher's life, `last_truncated_rollouts` those of its last decision.

    `decision_log` keeps every `Decision` in order. `marks` are the turns on
    at which each playout's value is also read for it (`MARKS`; `()` reads
    none). Those reads draw nothing from the rollouts' random stream and
    change no game, so the moves played are the same with or without them.

    `hold` is a reading of other seats' held development cards for the world
    samples (`View.sample`); without one every unseen card is dealt alike.

    `pool` (a `RolloutPool`) plays the worlds out in worker processes beside
    this one. The worlds and their seeds are still drawn here, in the same
    order, so a decision is the one made without the pool wherever the
    policy's move for a row does not depend on the other rows batched with
    it. A decision's `cpu_seconds` is then this process's share alone."""

    def __init__(
        self,
        policy,
        *,
        worlds: int = 32,
        top: int = 3,
        decisive: float | None = None,
        full_setup: bool = False,
        horizon: int | None = None,
        marks: Sequence[int] = MARKS,
        hold: HoldReading | None = None,
        rng: random.Random | None = None,
        pool: "RolloutPool | None" = None,
    ) -> None:
        if worlds < 1:
            raise ValueError("paired rollouts need at least one world")
        if top < 2:
            raise ValueError("paired rollouts compare at least two candidates")
        if decisive is not None and not 0.0 < decisive <= 1.0:
            raise ValueError(f"decisive is a prior in (0, 1], not {decisive!r}")
        if horizon is not None and horizon < 1:
            raise ValueError(f"horizon is at least one turn, not {horizon!r}")
        marks = tuple(int(m) for m in marks)
        if any(m < 1 for m in marks) or list(marks) != sorted(set(marks)):
            raise ValueError(f"marks are distinct increasing turns, 1 or more, not {marks!r}")
        self.policy = policy
        self.worlds = worlds
        self.top = top
        self.decisive = None if decisive is None else float(decisive)
        self.full_setup = bool(full_setup)
        self.horizon = horizon
        self.marks = marks
        self.hold = hold
        self.rng = rng or random.Random()
        self.pool = pool
        self.truncated_rollouts = 0
        self.last_truncated_rollouts = 0
        self.decision_log: list[Decision] = []
        self._steps = 0
        self._row_steps = 0
        self._mark_row_steps: tuple[int, ...] = ()
        self._mark_reads: np.ndarray | None = None

    def world(self, game: Game, seat: int) -> Game:
        """One world for `seat`'s decision: the hidden cards sampled from its
        own belief and the development deck in an order of the world's own."""
        state = game.state(seat).sample(self.rng, self.hold)
        world = imagine(game, self.rng, randomize_deck=False, seat=seat)
        world.set_state(state)
        self.rng.shuffle(world._state.deck)
        return world

    def decide(self, game: Game) -> Decision:
        if is_over(game):
            raise ValueError("no decision in a finished game")
        began = time.process_time()
        self.last_truncated_rollouts = 0
        seat = to_move(game)
        options = tuple(legal_actions(game))
        if not options:
            raise ValueError("no legal action to choose from")
        seen = game.state(seat).state
        context = dict(
            turn=game.turns, phase=game.phase.name, seat=seat,
            points=tuple(public_victory_points(seen, s) for s in range(seen.num_players)),
        )
        ((prior, value),) = self.policy.score_rows([(game, seat, options)])
        prior = np.asarray(prior, dtype=np.float64)
        value = tuple(float(v) for v in value)
        order = sorted(range(len(options)), key=lambda i: -prior[i])
        if len(options) == 1 or (
            self.decisive is not None and prior[order[0]] >= self.decisive
            and not (self.full_setup and game.phase in SETUP_PHASES)
        ):
            decision = Decision(
                options, prior, value, (order[0],), (), order[0], 0, **context,
                cpu_seconds=time.process_time() - began,
            )
            self.decision_log.append(decision)
            return decision
        candidates = tuple(order[: self.top])

        # Worlds and seeds are drawn here, in order, whoever plays them out:
        # a pool's decision is the one this process would have made alone.
        tasks = [(self.world(game, seat), self.rng.getrandbits(64)) for _ in range(self.worlds)]
        lanes = self._lanes(tasks, tuple(options[i] for i in candidates), seat, game.turns)
        self._keep(lanes)
        after = lanes.after
        outcomes = lanes.outcomes
        table = outcomes.reshape(self.worlds, len(candidates))
        scores = table.mean(axis=0)
        best = max(range(len(candidates)), key=lambda j: (scores[j], -j))
        reads = self._mark_reads
        decision = Decision(
            options, prior, value, candidates,
            tuple(float(s) for s in scores), candidates[best], self.worlds, **context,
            outcomes=tuple(tuple(float(x) for x in row) for row in table),
            marks=self.marks,
            mark_values=tuple(
                tuple(tuple(float(x) for x in row)
                      for row in reads[m].reshape(self.worlds, len(candidates)))
                for m in range(len(self.marks))
            ),
            after_values=() if after is None else tuple(
                tuple(float(x) for x in row) for row in after.reshape(self.worlds, len(candidates))
            ),
            steps=self._steps, truncated=self.last_truncated_rollouts,
            row_steps=self._row_steps, mark_row_steps=self._mark_row_steps,
            cpu_seconds=time.process_time() - began,
        )
        self.decision_log.append(decision)
        return decision

    def choose(self, game: Game) -> Action:
        decision = self.decide(game)
        return decision.options[decision.chosen]

    def _lanes(self, tasks, moves, seat: int, start: int) -> "Lanes":
        """Every world of `tasks` played out after each of `moves`: here, or
        split across the pool's workers, with this process playing a share."""
        if self.pool is None:
            return play_lanes(self.policy, tasks, moves, seat, start, self.horizon, self.marks)
        return self.pool.play(self.policy, tasks, moves, seat, start, self.horizon, self.marks)

    def _keep(self, lanes: "Lanes") -> None:
        self.last_truncated_rollouts = lanes.truncated
        self.truncated_rollouts += lanes.truncated
        self._mark_reads = lanes.reads
        self._steps = lanes.steps
        self._row_steps = lanes.row_steps
        self._mark_row_steps = lanes.mark_row_steps

    def play_out(self, games: Sequence[Game], seat: int, start: int) -> np.ndarray:
        """Each game played on by the policy for every seat, in lockstep, one
        batched `act_rows` per step, and scored for `seat` (see the class)."""
        lanes = _play(self.policy, games, seat, start, self.horizon, self.marks, after=False)
        self._keep(lanes)
        return lanes.outcomes


@dataclass(frozen=True)
class Lanes:
    """What playing a decision's copies out gives back, copy by copy in
    world-major, candidate-minor order: `outcomes`, each copy's score;
    `after`, the value head's read right after the candidate (`None` with
    no `marks`); `reads`, the read at each mark, `(len(marks), copies)`; and
    the counts `Decision` keeps (`steps`, `row_steps`, `mark_row_steps`,
    `truncated`)."""

    outcomes: np.ndarray
    after: np.ndarray | None
    reads: np.ndarray
    steps: int
    row_steps: int
    mark_row_steps: tuple[int, ...]
    truncated: int

    @staticmethod
    def joined(parts: Sequence["Lanes"]) -> "Lanes":
        """`parts`, each a run of whole worlds in order, as one: the copies
        side by side; `steps`, the longest part's lockstep count."""
        return Lanes(
            outcomes=np.concatenate([p.outcomes for p in parts]),
            after=None if parts[0].after is None else np.concatenate([p.after for p in parts]),
            reads=np.concatenate([p.reads for p in parts], axis=1),
            steps=max(p.steps for p in parts),
            row_steps=sum(p.row_steps for p in parts),
            mark_row_steps=tuple(int(sum(c)) for c in zip(*(p.mark_row_steps for p in parts))),
            truncated=sum(p.truncated for p in parts),
        )


def play_lanes(
    policy, tasks: Sequence[tuple[Game, int]], moves: Sequence[Action], seat: int, start: int,
    horizon: int | None, marks: tuple[int, ...],
) -> Lanes:
    """Each `(world, seed)` of `tasks` copied once per move of `moves`
    (`paired_copy`), the move applied, and every copy played out by `policy`
    (`PairedRollouts.play_out`). A function of its arguments alone, so a
    worker process plays a share of the worlds as this one would."""
    return RolloutBatch.from_tasks(policy, tasks, moves, seat, start, marks).advance(horizon)


def _stops(game: Game, seat: int, start: int, horizon: int | None) -> bool:
    if is_over(game):
        return True
    return (
        horizon is not None
        and game.turns - start >= horizon
        and to_move(game) == seat
        and game.phase is Phase.ROLL
    )


def _read_after(policy, games: Sequence[Game], seat: int) -> np.ndarray:
    """Each copy's score for `seat`: terminal 1/0 or its own-frame value."""
    out = np.zeros(len(games))
    read = []
    for i, game in enumerate(games):
        if is_over(game):
            out[i] = 1.0 if game.won_by == seat else 0.0
        else:
            read.append(i)
    if read:
        for i, vector in zip(read, policy.value_rows([(games[i], seat) for i in read])):
            out[i] = float(vector[seat])
    return out


class RolloutBatch:
    """Persistent playout copies, advanced selectively to later turn cutoffs.

    ``games`` are distinct, already constructed copies, with their candidate
    moves applied. The batch owns their continuation: it retains chance
    streams, reached mark values and cumulative per-copy decision counts.
    ``from_tasks`` constructs them using the same paired chance as
    ``play_lanes``. ``advance(horizon, indices)`` returns a ``Lanes`` snapshot
    of the selected copies, in that order. Later work never rewrites it.

    Horizons count from ``start`` and can only increase for a copy; ``None``
    continues to the end or its cumulative ``STEP_LIMIT``. Copies stop at the
    root seat's turn boundary, as in one-shot play. A paused copy's game and
    chance source are retained, so extending it does not replay its prefix.

    Identity across different schedules requires a policy whose row actions
    and values are independent of batch composition and unrelated cache
    eviction. Keep the policy fixed while retaining copies for a one-shot
    comparison. The policy remains shared; its per-game state must stay
    valid while other copies run.
    """

    def __init__(self, policy, games: Sequence[Game], seat: int, start: int,
                 marks: Sequence[int] = (), *, after: bool = False) -> None:
        self.policy, self.games = policy, tuple(games)
        if len({id(game) for game in games}) != len(games):
            raise ValueError("each rollout copy must be a distinct game")
        self.seat, self.start = seat, start
        self.marks = tuple(int(mark) for mark in marks)
        if any(mark < 1 for mark in self.marks) or list(self.marks) != sorted(set(self.marks)):
            raise ValueError("marks must be distinct increasing positive turns")
        size = len(games)
        self.counts = np.zeros(size, dtype=np.int64)
        self.blocked = np.zeros(size, dtype=bool)
        self.horizons = np.zeros(size, dtype=np.float64)
        self.reads = np.full((len(self.marks), size), np.nan)
        self.at = np.full((len(self.marks), size), -1, dtype=np.int64)
        self.reached = np.zeros(size, dtype=np.int64)
        self.after = _read_after(policy, self.games, seat) if after else None
        self.scores = np.full(size, np.nan) if self.after is None else self.after.copy()

    @classmethod
    def from_tasks(cls, policy, tasks: Sequence[tuple[Game, int]], moves: Sequence[Action],
                   seat: int, start: int, marks: Sequence[int] = ()) -> "RolloutBatch":
        """A paired copy per world and move, in world-major order."""
        batch = cls(policy, (), seat, start, marks, after=bool(marks))
        batch.add_tasks(tasks, moves)
        return batch

    def add_tasks(self, tasks: Sequence[tuple[Game, int]], moves: Sequence[Action]) -> tuple[int, ...]:
        """Append paired copies for new worlds or moves; return their indices."""
        copies = []
        for world, seed in tasks:
            for move in moves:
                copy = paired_copy(world, seed, self.seat)
                apply(copy, move)
                copies.append(copy)
        return self.add(copies)

    def add(self, games: Sequence[Game]) -> tuple[int, ...]:
        """Append distinct, already constructed copies with their move applied.

        New copies start with no playout work. Existing copies, cutoffs and
        mark reads stay in place; the caller can advance the returned indices.
        """
        games = tuple(games)
        identities = [id(game) for game in (*self.games, *games)]
        if len(set(identities)) != len(identities):
            raise ValueError("each rollout copy must be a distinct game")
        begin, size = len(self.games), len(games)
        initial = _read_after(self.policy, games, self.seat) if self.after is not None else None
        self.games += games
        self.counts = np.concatenate([self.counts, np.zeros(size, dtype=np.int64)])
        self.blocked = np.concatenate([self.blocked, np.zeros(size, dtype=bool)])
        self.horizons = np.concatenate([self.horizons, np.zeros(size)])
        self.reads = np.concatenate([self.reads, np.full((len(self.marks), size), np.nan)], axis=1)
        self.at = np.concatenate([self.at, np.full((len(self.marks), size), -1, dtype=np.int64)], axis=1)
        self.reached = np.concatenate([self.reached, np.zeros(size, dtype=np.int64)])
        if self.after is not None:
            self.after = np.concatenate([self.after, initial])
        self.scores = np.concatenate([self.scores, np.full(size, np.nan) if initial is None else initial])
        return tuple(range(begin, begin + size))

    def _values(self, indices: Sequence[int]) -> np.ndarray:
        unread = [i for i in indices if np.isnan(self.scores[i])]
        if unread:
            self.scores[unread] = _read_after(self.policy, [self.games[i] for i in unread], self.seat)
        return self.scores[list(indices)].copy()

    def _marks(self, indices: Sequence[int], scores: np.ndarray | None = None) -> None:
        due = [(j, i) for j, i in enumerate(indices)
               if self.reached[i] < len(self.marks)
               and not is_over(self.games[i])
               and self.games[i].turns - self.start >= self.marks[self.reached[i]]
               and to_move(self.games[i]) == self.seat
               and self.games[i].phase is Phase.ROLL]
        if not due:
            return
        values = (self._values([i for _, i in due])
                  if scores is None else [scores[j] for j, _ in due])
        for (_, i), value in zip(due, values):
            while self.reached[i] < len(self.marks) and (
                self.games[i].turns - self.start >= self.marks[self.reached[i]]
            ):
                m = self.reached[i]
                self.reads[m, i], self.at[m, i] = value, self.counts[i]
                self.reached[i] += 1

    def advance(self, horizon: int | None, indices: Sequence[int] | None = None) -> Lanes:
        """Continue selected copies; return their cumulative work and values."""
        if horizon is not None and horizon < 1:
            raise ValueError("horizon must be a positive turn count or None")
        selected = list(range(len(self.games))) if indices is None else list(indices)
        if len(set(selected)) != len(selected) or any(i < 0 or i >= len(self.games) for i in selected):
            raise ValueError("indices must name distinct rollout copies")
        requested = float('inf') if horizon is None else horizon
        if any(requested < self.horizons[i] for i in selected):
            raise ValueError("a rollout horizon cannot move backwards")
        self.horizons[selected] = requested
        live = [i for i in selected if not self.blocked[i] and self.counts[i] < STEP_LIMIT
                and not _stops(self.games[i], self.seat, self.start, horizon)]
        while live:
            rows = [(self.games[i], to_move(self.games[i]), tuple(legal_actions(self.games[i]))) for i in live]
            moving = [(i, row) for i, row in zip(live, rows) if row[2]]
            self.counts[live] += 1
            for i, row in zip(live, rows):
                if not row[2]:
                    self.blocked[i] = True
            chosen = self.policy.act_rows([row for _, row in moving])
            for (i, _), action in zip(moving, chosen):
                apply(self.games[i], action)
                self.scores[i] = np.nan
            live = [i for i, _ in moving if self.counts[i] < STEP_LIMIT
                    and not _stops(self.games[i], self.seat, self.start, horizon)]
            if self.marks:
                self._marks(live)
        scores = self._values(selected)
        if self.marks:
            self._marks(selected, scores)
        reads = self.reads[:, selected].copy()
        reads[np.isnan(reads)] = np.broadcast_to(scores, reads.shape)[np.isnan(reads)]
        counts = self.counts[selected]
        at = self.at[:, selected]
        return Lanes(outcomes=scores, after=None if self.after is None else self.after[selected].copy(),
                     reads=reads, steps=int(counts.max(initial=0)), row_steps=int(counts.sum()),
                     mark_row_steps=tuple(int(np.where(row >= 0, np.minimum(row, counts), counts).sum()) for row in at),
                     truncated=sum(not is_over(self.games[i]) for i in selected))


def _play(
    policy, games: Sequence[Game], seat: int, start: int, horizon: int | None,
    marks: tuple[int, ...], *, after: bool,
) -> Lanes:
    """`games` played on in lockstep, one batched `act_rows` per step, and
    scored for `seat` (`PairedRollouts`); with `after`, each read first right
    after its candidate (`_read_after`)."""
    return RolloutBatch(policy, games, seat, start, marks, after=after).advance(horizon)


def _shares(tasks: Sequence, parts: int) -> list[list]:
    """`tasks` cut into at most `parts` runs, in order, none empty, their
    sizes as even as they can be."""
    parts = max(1, min(parts, len(tasks)))
    size, extra = divmod(len(tasks), parts)
    out, at = [], 0
    for k in range(parts):
        n = size + (k < extra)
        out.append(list(tasks[at: at + n]))
        at += n
    return out


# The policy each worker process loads once (`RolloutPool`).
_worker_policy = None


def _start_worker(load_policy: Callable[[], object]) -> None:
    global _worker_policy
    _worker_policy = load_policy()


def _ready(seconds: float) -> int:
    time.sleep(seconds)
    return os.getpid()


def _play_share(tasks, moves, seat, start, horizon, marks) -> Lanes:
    return play_lanes(_worker_policy, tasks, moves, seat, start, horizon, marks)


class RolloutPool:
    """Worker processes that play a decision's worlds out beside this one.

    A playout is mostly Python (legal moves, encoding, applying moves), so
    threads would share one core; processes do not. `load_policy` is a
    picklable callable that each worker runs once for its own copy of the
    searcher's policy: it must give the same policy, the same file loaded
    the same way. All `workers` start here and load it before this returns,
    so a failing load fails now and the first decision pays no start-up.

    `play` cuts a decision's worlds into `workers + 1` runs of whole worlds,
    plays the first here while the workers play the rest, and joins them in
    world order: what `play_lanes` gives on all of them at once, wherever
    the policy's move for a row does not depend on the rows batched with it.
    A worker reads `STEP_LIMIT` as its module defines it.

    A worker that dies breaks the pool. That decision, and every one after,
    plays the worlds the workers had here instead, and `broken` says so: a
    slower decision rather than a lost one. `close` stops the workers."""

    def __init__(self, load_policy: Callable[[], object], workers: int) -> None:
        if workers < 1:
            raise ValueError(f"a pool has at least one worker, not {workers!r}")
        methods = multiprocessing.get_all_start_methods()
        # Not fork: the searching process may hold threads (a client's socket)
        # that a forked child would inherit mid-state.
        method = "forkserver" if "forkserver" in methods else "spawn"
        self.workers = workers
        self.broken = False
        self._executor = ProcessPoolExecutor(
            workers, mp_context=multiprocessing.get_context(method),
            initializer=_start_worker, initargs=(load_policy,),
        )
        # Busy tasks, one a worker: the executor starts a process for each
        # task no idle worker can take, so all of them start and load now.
        started = [self._executor.submit(_ready, 0.2) for _ in range(workers)]
        try:
            for future in started:
                future.result()
        except BrokenProcessPool as error:
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise RuntimeError("a rollout worker could not load its policy") from error

    def play(
        self, policy, tasks: Sequence[tuple[Game, int]], moves: Sequence[Action], seat: int,
        start: int, horizon: int | None, marks: tuple[int, ...],
    ) -> Lanes:
        """`play_lanes(policy, tasks, ...)`, with the workers playing all
        but the first run of worlds; `policy` is this process's own."""
        if self.broken:
            return play_lanes(policy, tasks, moves, seat, start, horizon, marks)
        runs = _shares(tasks, self.workers + 1)
        futures = []
        try:
            for run in runs[1:]:
                futures.append(self._executor.submit(_play_share, run, moves, seat, start, horizon, marks))
        except BrokenProcessPool:
            self.broken = True
        here = play_lanes(policy, runs[0], moves, seat, start, horizon, marks)
        parts = [here]
        if not self.broken:
            try:
                parts += [future.result() for future in futures]
            except BrokenProcessPool:
                self.broken = True
        if self.broken:
            rest = [task for run in runs[1:] for task in run]
            parts = [here] + ([play_lanes(policy, rest, moves, seat, start, horizon, marks)] if rest else [])
        return Lanes.joined(parts)

    def close(self) -> None:
        """Stops the workers; a closed pool plays every decision here."""
        self.broken = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "RolloutPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
