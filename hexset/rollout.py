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

import random
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .actions import Action, apply, legal_actions
from .chance import Balanced, Live
from .game import Game, Phase, imagine, is_over, to_move
from .victory import public_victory_points
from .view import HoldReading

__all__ = ["MARKS", "SETUP_PHASES", "STEP_LIMIT", "Decision", "PairedRollouts", "paired_copy"]

#: The most lockstep decisions a playout makes before it is scored where it
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
    samples (`View.sample`); without one every unseen card is dealt alike."""

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

        copies: list[Game] = []
        for _ in range(self.worlds):
            world = self.world(game, seat)
            seed = self.rng.getrandbits(64)
            for index in candidates:
                copy = paired_copy(world, seed, seat)
                apply(copy, options[index])
                copies.append(copy)
        after = self._read_after(copies, seat) if self.marks else None
        outcomes = self.play_out(copies, seat, game.turns)
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

    def _stops(self, game: Game, seat: int, start: int) -> bool:
        if is_over(game):
            return True
        return (
            self.horizon is not None
            and game.turns - start >= self.horizon
            and to_move(game) == seat
            and game.phase is Phase.ROLL
        )

    def _read_after(self, games: Sequence[Game], seat: int) -> np.ndarray:
        """Each copy's value for `seat` right after its candidate, before any
        playout: 1 or 0 where the candidate ended the game (`decide`'s log
        only)."""
        out = np.zeros(len(games))
        read = []
        for i, game in enumerate(games):
            if is_over(game):
                out[i] = 1.0 if game.won_by == seat else 0.0
            else:
                read.append(i)
        if read:
            for i, vector in zip(read, self.policy.value_rows([(games[i], seat) for i in read])):
                out[i] = float(vector[seat])
        return out

    def _read_marks(
        self, games: Sequence[Game], live: Sequence[int], seat: int, start: int,
        reads: np.ndarray, reached: list[int], at: np.ndarray, steps: int,
    ) -> None:
        """The value head's read of each live game at the mover's turn start,
        for every mark it has newly reached (`decide`'s log only)."""
        marks = self.marks
        due = [
            i for i in live
            if reached[i] < len(marks)
            and games[i].turns - start >= marks[reached[i]]
            and to_move(games[i]) == seat
            and games[i].phase is Phase.ROLL
        ]
        if not due:
            return
        values = self.policy.value_rows([(games[i], seat) for i in due])
        for i, vector in zip(due, values):
            while reached[i] < len(marks) and games[i].turns - start >= marks[reached[i]]:
                reads[reached[i], i] = float(vector[seat])
                at[reached[i], i] = steps
                reached[i] += 1

    def play_out(self, games: Sequence[Game], seat: int, start: int) -> np.ndarray:
        """Each game played on by the policy for every seat, in lockstep, one
        batched `act_rows` per step, and scored for `seat` (see the class)."""
        reads = np.full((len(self.marks), len(games)), np.nan)
        reached = [0] * len(games)
        at = np.full((len(self.marks), len(games)), -1)
        ended = np.zeros(len(games), dtype=np.int64)
        steps = 0
        live = [i for i, game in enumerate(games) if not self._stops(game, seat, start)]
        for _ in range(STEP_LIMIT):
            if not live:
                break
            rows = []
            for i in live:
                game = games[i]
                rows.append((game, to_move(game), tuple(legal_actions(game))))
            moving = [(i, row) for i, row in zip(live, rows) if row[2]]
            chosen = self.policy.act_rows([row for _, row in moving])
            for (i, _), action in zip(moving, chosen):
                apply(games[i], action)
            steps += 1
            # A seat with no legal move is scored where it stands.
            before = live
            live = [i for i, _ in moving if not self._stops(games[i], seat, start)]
            if len(live) < len(before):
                kept = set(live)
                for i in before:
                    if i not in kept:
                        ended[i] = steps
            if self.marks:
                self._read_marks(games, live, seat, start, reads, reached, at, steps)
        scores = np.zeros(len(games))
        read = []
        for i, game in enumerate(games):
            if is_over(game):
                scores[i] = 1.0 if game.won_by == seat else 0.0
            else:
                read.append(i)
        if read:
            values = self.policy.value_rows([(games[i], seat) for i in read])
            for i, vector in zip(read, values):
                scores[i] = float(vector[seat])
        self.last_truncated_rollouts = len(read)
        self.truncated_rollouts += len(read)
        # A mark a playout never reached -- it ended or stopped first -- keeps
        # the playout's own score.
        unread = np.isnan(reads)
        reads[unread] = np.broadcast_to(scores, reads.shape)[unread]
        self._mark_reads = reads
        self._steps = steps
        # A copy still live at the step limit has moved on every step.
        ended[[i for i in live]] = steps
        self._row_steps = int(ended.sum())
        self._mark_row_steps = tuple(
            int(np.where(at[m] >= 0, np.minimum(at[m], ended), ended).sum())
            for m in range(len(self.marks))
        )
        return scores
