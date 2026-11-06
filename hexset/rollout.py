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

Where the native engine is built and the policy can read it
(`native_contract`, a contract-9 policy), each world's copy is held natively
(`hexset.native.playout.NativeCopy`) from the candidate on: the same moves,
the same chance streams and the same rows for the policy, at a fraction of
the Python engine's cost per move.

Every decision is kept in `PairedRollouts.decision_log` with what it was
made from: the outcome of each candidate in each world, and the value head's
read of each playout at the mover's turn starts `marks` turns on. Fewer
worlds, fewer candidates or a shorter horizon can then be scored on the same
playouts afterwards, without playing again.

**The policy's own move.** A root the search does not play out is played as
the policy plays it alone (`act_rows`), not as its prior's favourite: a
policy that chooses whole decisions (`hexset.decisions`) may open its best
whole decision with a move its prior, marginalised onto first moves, does
not rank first. At a root that is played out, that move is the first
candidate and wins every tie. Where the search plays anything else, the
policy's plan for the rest of its own decision is dropped
(`hexset.decisions.Plans.drop`).

**Whole decisions.** A policy that lists whole options (`option_rows`, see
`hexset.clients.policy.Policy`) is searched over them at a root that opens a
compound decision (a setup turn, the double turn, a Road Building card, a
discard): each candidate is the whole decision, applied in every world
before the playout, and the one chosen is played a move at a time, its
later parts answering the asks inside it with no search of their own.

**A mixed strategy.** Under `mix`, a searched root plays a candidate drawn
from the policy's prior regularised towards the search's values
(`mixed_strategy`) instead of the best-scoring one: candidates the search
scores about alike are mixed by the prior, and one that clearly plays out
better is played nearly always. The draws come from a stream of the
searcher's own, seeded off its `rng`, so the worlds, the scores and every
other draw are the greedy search's.

**A deterministic search.** Under `det`, a searched root draws its worlds,
their playout seeds and its mixed strategy's draw from a stream seeded by
what the deciding seat can see (`hexset.game.observed_digest`) and a fixed
`salt`, not from the searcher's running `rng`: the same visible position
always searches the same worlds and plays the same move, in any process,
on any fresh searcher, and a position rebuilt from a record searches as it
was searched.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .actions import Action, ActionType, apply, legal_actions
from .chance import Balanced, Live
from .decisions import Compound, Option, Plans, first_move, inside, opens
from .game import Game, Phase, imagine, is_over, observed_digest, to_move
from .native.playout import NativeCopy, copyable, held
from .victory import public_victory_points
from .view import HoldReading

__all__ = [
    "MARKS", "SETUP_PHASES", "STEP_LIMIT", "Decision", "Lanes", "PairedRollouts", "RolloutPool", "RolloutBatch",
    "SharedPools",
    "mixed_strategy", "paired_copy", "play_lanes",
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
    option indices played out (`candidates`: the policy's own move first,
    then the rest by prior), each one's mean outcome for the mover over the
    worlds (`scores`, empty where nothing was played out), and the option
    `chosen`; `move` is the atomic move it opens with. `worlds` is how many
    worlds the candidates were played out in: 0 for a root the policy decides
    alone, where `candidates` is the policy's own move alone.

    Under `whole` the options are the root's whole decisions
    (`hexset.decisions.Option`: an action, or a `Compound` of several) and
    `prior` the policy's probability of each (`option_rows`); otherwise they
    are the root's legal atomic moves and `prior` the policy's `score_rows`.

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
    time the decision took.

    Under a deterministic search (`PairedRollouts.det`), `det_key` is the
    hex digest the decision's stream was seeded by and `salt` the salt in it;
    `""` and `None` otherwise.

    Under a mixed strategy (`PairedRollouts.mix`), `mix` is each candidate's
    probability (`mixed_strategy`, in `candidates` order), `chosen` the
    candidate drawn from it and `greedy` the one the best mean score would
    have played; both are empty (`()`, `None`) for a greedy decision."""

    options: tuple[Option, ...]
    prior: np.ndarray
    value: tuple[float, ...]
    candidates: tuple[int, ...]
    scores: tuple[float, ...]
    chosen: int
    worlds: int
    whole: bool = False
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
    mix: tuple[float, ...] = ()
    greedy: int | None = None
    det_key: str = ""
    salt: int | None = None

    @property
    def move(self) -> Action:
        """The atomic move played: the chosen option, or its first part."""
        return first_move(self.options[self.chosen])

    def to_json(self) -> dict:
        """The decision as one JSON object. A decision the policy made alone
        keeps its top prior and the option count, not the full prior. A
        candidate is written `[type, a, b]`, or for a whole decision of
        several moves `[kind, [[type, a, b], ...]]`."""
        out = {
            "turn": self.turn, "phase": self.phase, "seat": self.seat,
            "points": list(self.points), "options": len(self.options),
            "top_prior": round(float(np.max(self.prior)), 4),
            "chosen": self.chosen, "worlds": self.worlds,
            "cpu_seconds": round(self.cpu_seconds, 4),
        }
        if self.whole:
            out["whole"] = True
        if not self.worlds:
            return out
        r = lambda x: round(float(x), 4)  # noqa: E731
        out.update(
            candidates=list(self.candidates),
            actions=[_option_json(self.options[i]) for i in self.candidates],
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
        if self.mix:
            out.update(mix=[round(float(x), 6) for x in self.mix], greedy=self.greedy)
        if self.det_key:
            out.update(det_key=self.det_key, salt=self.salt)
        return out


def _option_json(option: Option) -> list:
    if isinstance(option, Compound):
        return [option.kind.name, [[p.type.name, p.a, p.b] for p in option.parts]]
    return [option.type.name, option.a, option.b]


def _outcome(x: float) -> float | int:
    """A playout's score as the log writes it: a finished game's 1 or 0
    exactly, a value head's read to three places."""
    return int(x) if x in (0.0, 1.0) else round(float(x), 3)


def mixed_strategy(scores: Sequence[float], prior: Sequence[float], mix: float,
                   anchor: float = 1.0) -> np.ndarray:
    """The candidates' probabilities under a mixed strategy: the policy's
    `prior` over the candidates, raised to `anchor` and regularised towards
    `scores` (each one's mean outcome for the mover, a win chance) at
    temperature `mix`, in the scores' units::

        pi(a)  proportional to  prior(a) ** anchor * exp(scores(a) / mix)

    the policy that maximises the expected score less `mix` times its KL
    divergence from `q`, the prior raised to `anchor` and renormalised over
    the candidates. Its expected shortfall from the best score is at most
    `mix * ln(1 / q(best))`, by the scores themselves: a candidate trailing
    the best by many `mix` is all but never played, and candidates within
    about one `mix` of it are mixed in proportion to `q`. As `mix` falls to 0
    it plays the best score (`q` breaking a tie); as it grows it plays `q`.
    `anchor` 1 anchors to the prior itself, 0 to every candidate alike (a
    softmax over the scores), between them to a flattened prior. A
    candidate the prior gives nothing is never played at an `anchor` above
    0; a prior that gives every candidate nothing counts them alike."""
    if not mix > 0:
        raise ValueError(f"mix is a temperature above 0, not {mix!r}")
    if not (math.isfinite(anchor) and anchor >= 0.0):
        raise ValueError(f"anchor is an exponent of 0 or more, not {anchor!r}")
    scores = np.asarray(scores, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    if prior.shape != scores.shape or not scores.size:
        raise ValueError("a score and a prior for each candidate")
    if not (prior > 0).any() or anchor == 0.0:
        prior = np.ones_like(prior)
    with np.errstate(divide="ignore"):
        logits = anchor * np.log(np.clip(prior, 0.0, None)) + (scores - scores.max()) / mix
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def paired_copy(world: Game, seed: int, seat: int) -> Game:
    """A copy of `world` that draws its chance off `seed`, split: two copies
    on one seed roll the same n-th roll and take the same n-th steal, and
    deal from the same deck order, whatever was played in either. `seat`
    names whose world it is (`hexset.game.imagine`)."""
    rng = random.Random(seed)
    copy = imagine(world, rng, randomize_deck=False, seat=seat)
    dice = copy._state.dice
    copy.chance = Live(rng, split=True) if dice is None else Balanced.from_deck(rng, dice, split=True)
    # A policy that samples its moves keys its draws by this, so the copies
    # of one world draw alike (`hexset.clients.conditioned.ConditionedPolicy`).
    copy.paired_seed = seed
    return copy


class PairedRollouts:
    """Decides a position by paired rollouts of the policy's own top moves.

    `policy` is a batched `hexset.clients.policy.Policy`: `score_rows` for the
    root's prior and value, `act_rows` for every move of a playout and
    `value_rows` for a playout stopped short of the game's end.

    `worlds` is the number of shared-chance worlds each candidate plays out
    in; `top`, how many candidates: the policy's own move (`act_rows`), then
    its other moves by prior. A root with one legal move, or, under
    `decisive`, whose prior on one move is at least `decisive`, plays the
    policy's own move with no playout. Under `full_setup`, `decisive` skips
    no setup root (`SETUP_PHASES`): every opening settlement and road with a
    choice is played out. Where the search plays another move than the
    policy's own, the policy's plan for the game (`plans`, where it keeps
    one) is dropped.

    A policy that lists whole options (`option_rows`, where `has_options`
    is not false) is searched over whole decisions at a root that opens a
    compound one (the module docstring): the candidates are its own whole
    decision, then its others by probability; `decisive` and `full_setup`
    read the root that opens the decision, and the asks inside it are
    answered from the chosen decision's parts without a search. Inside a
    decision begun a move at a time without such a plan, the root is
    searched a move at a time.

    A playout runs to the game's end, scored 1 where the mover won and 0
    otherwise, or under `horizon` stops at the first turn start of the mover
    at least `horizon` turns (every seat's, counted as `Game.turns` counts
    them) after the root's. A playout stopped short of the end -- at the
    horizon, at `STEP_LIMIT`, or where the seat to move has no legal move --
    is scored by the value head's chance for the mover, read from the
    mover's own frame whoever is to move there, so the mover's own cards in
    that world stay known to the read. The candidate with the highest mean
    score over the worlds is played, the earlier candidate on a tie (so the
    policy's own move first). Under `mix`, a temperature above 0 in the
    scores' units (win chance), the candidate is drawn instead from
    `mixed_strategy` over the candidates' scores and their prior raised to
    `anchor` (1, the prior itself, by default), from a
    stream seeded off `rng`'s state that draws nothing else: the worlds, and
    so the scores, are the greedy search's, and a game stays a function of
    the seeds it was dealt. A root decided without playouts draws nothing.

    Under `det`, a searched root takes its worlds, their seeds and its mixed
    draw from a stream of its own, seeded by `decision_key`: what the mover
    can see of the position (`hexset.game.observed_digest`) and `salt`. It
    draws nothing from `rng`, so the same visible position searches the same
    worlds and plays the same move on any searcher, in any process, and a
    run stays a function of the seeds that dealt its games.

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
        mix: float = 0.0,
        anchor: float = 1.0,
        det: bool = False,
        salt: int = 0,
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
        mix = float(mix)
        if not (math.isfinite(mix) and mix >= 0.0):
            raise ValueError(f"mix is a temperature of 0 or more, not {mix!r}")
        anchor = float(anchor)
        if not (math.isfinite(anchor) and anchor >= 0.0):
            raise ValueError(f"anchor is an exponent of 0 or more, not {anchor!r}")
        if isinstance(salt, bool) or not isinstance(salt, int) or salt < 0:
            raise ValueError(f"salt is a whole number of 0 or more, not {salt!r}")
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
        self.mix = mix
        self.anchor = anchor
        self.det = bool(det)
        self.salt = salt
        # The mixed strategy's stream: off `rng`'s state, salted, so a draw never
        # moves the worlds and the worlds never move a draw.
        self._draws = random.Random(
            hashlib.sha256((repr(self.rng.getstate()) + ":mix").encode()).digest()
        ) if mix else None
        self.truncated_rollouts = 0
        self.last_truncated_rollouts = 0
        self.decision_log: list[Decision] = []
        # The rest of each whole decision this searcher chose, played a move at a time.
        self._plans = Plans()
        self._steps = 0
        self._row_steps = 0
        self._mark_row_steps: tuple[int, ...] = ()
        self._mark_reads: np.ndarray | None = None

    def world(self, game: Game, seat: int) -> Game:
        """One world for `seat`'s decision: the hidden cards sampled from its
        own belief and the development deck in an order of the world's own,
        drawn from `rng` (under `det`, the decision's own stream)."""
        state = game.state(seat).sample(self.rng, self.hold)
        world = imagine(game, self.rng, randomize_deck=False, seat=seat)
        world.set_state(state)
        self.rng.shuffle(world._state.deck)
        return world

    def decision_key(self, game: Game, seat: int) -> bytes:
        """The seed of `seat`'s decision at `game` under `det`: a sha256 of
        `salt` and `hexset.game.observed_digest(game, seat)`, so of what the
        seat can see and nothing it cannot."""
        return hashlib.sha256(
            b"hexset.rollout.det\x1f" + str(self.salt).encode() + b"\x1f" + observed_digest(game, seat)
        ).digest()

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
        if self._plans.planned(game, seat, options):
            # Inside a whole decision this searcher chose: its next part, unsearched.
            move = self._plans.act(game, seat, options, None, None)
            return self._alone(options, prior, value, options.index(move), context, began)
        self._plans.drop(game)      # a plan the game has moved past is never played
        own = options.index(self._own(game, seat, options))
        whole = self._opens_whole(game, options)
        order = sorted(range(len(options)), key=lambda i: -prior[i])
        if len(options) == 1 or (
            self.decisive is not None and prior[order[0]] >= self.decisive
            and not (self.full_setup and game.phase in SETUP_PHASES)
        ):
            if whole:
                self._hold_own(game, seat, options[own])
            return self._alone(options, prior, value, own, context, began)
        if whole:
            return self._decide_whole(game, seat, options[own], value, context, began)
        candidates = (own,) + tuple(i for i in order if i != own)[: self.top - 1]
        decision = self._search(game, seat, options, prior, value, candidates, context, began, whole=False)
        if decision.chosen != own:
            self._drop(game)
        return decision

    def _search(self, game: Game, seat: int, options: tuple, prior: np.ndarray, value: tuple,
                candidates: tuple[int, ...], context: dict, began: float, *, whole: bool) -> Decision:
        """`candidates` of `options` played out in shared worlds, the best
        mean score chosen (the earlier candidate on a tie) or, under `mix`,
        one drawn from `mixed_strategy`, and logged."""
        # Worlds and seeds are drawn here, in order, whoever plays them out:
        # a pool's decision is the one this process would have made alone.
        draws, key = self._draws, ""
        if not self.det:
            tasks = [(self.world(game, seat), self.rng.getrandbits(64)) for _ in range(self.worlds)]
        else:
            seed = self.decision_key(game, seat)
            key = seed.hex()
            draws = random.Random(hashlib.sha256(seed + b"\x1fmix").digest())
            # `world` draws from `rng`: the decision's stream stands in for it,
            # and the searcher's own is put back untouched.
            running, self.rng = self.rng, random.Random(seed)
            try:
                tasks = [(self.world(game, seat), self.rng.getrandbits(64)) for _ in range(self.worlds)]
            finally:
                self.rng = running
        lanes = self._lanes(tasks, tuple(options[i] for i in candidates), seat, game.turns)
        self._keep(lanes)
        after = lanes.after
        outcomes = lanes.outcomes
        table = outcomes.reshape(self.worlds, len(candidates))
        scores = table.mean(axis=0)
        best = max(range(len(candidates)), key=lambda j: (scores[j], -j))
        mixed: tuple[float, ...] = ()
        picked = best
        if self.mix:
            probabilities = mixed_strategy(scores, prior[list(candidates)], self.mix, self.anchor)
            mixed = tuple(float(x) for x in probabilities)
            picked = self._draw(probabilities, draws)
        reads = self._mark_reads
        decision = Decision(
            options, prior, value, candidates,
            tuple(float(s) for s in scores), candidates[picked], self.worlds, whole, **context,
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
            mix=mixed, greedy=candidates[best] if mixed else None,
            det_key=key, salt=self.salt if key else None,
        )
        self.decision_log.append(decision)
        return decision

    def _draw(self, probabilities: np.ndarray, draws: random.Random) -> int:
        """An index drawn from `probabilities` off `draws` (this searcher's own
        stream, or under `det` the decision's): one uniform a draw, never
        landing on a candidate of probability 0."""
        point = draws.random()
        running = 0.0
        for index, p in enumerate(probabilities):
            running += float(p)
            if point < running:
                return index
        return max(i for i, p in enumerate(probabilities) if p > 0)

    def _alone(self, options: tuple, prior: np.ndarray, value: tuple, index: int, context: dict,
               began: float, *, whole: bool = False) -> Decision:
        """The root decided with no playout: `options[index]`, logged."""
        decision = Decision(options, prior, value, (index,), (), index, 0, whole, **context,
                            cpu_seconds=time.process_time() - began)
        self.decision_log.append(decision)
        return decision

    def _own(self, game: Game, seat: int, options: tuple[Action, ...]) -> Action:
        """The policy's own move at the root, as it plays alone (`act_rows`):
        continuing a plan it holds, or opening the whole decision it chooses.
        A lone legal move is asked too where the policy keeps plans, so its
        plan moves on with the game."""
        if len(options) == 1 and self._policy_plans() is None:
            return options[0]
        (move,) = self.policy.act_rows([(game, seat, options)])
        if move not in options:
            raise RuntimeError(f"the policy chose {move}, not among {options}")
        return move

    def _policy_plans(self) -> Plans | None:
        return getattr(self.policy, "plans", None)

    def _drop(self, game: Game) -> None:
        """Forget the policy's plan for `game`: the search played something
        else than the move it opened with or continued."""
        plans = self._policy_plans()
        if plans is not None:
            plans.drop(game)

    def _opens_whole(self, game: Game, options: tuple[Action, ...]) -> bool:
        """Whether the root is searched over whole decisions: the policy lists
        them, and the root opens a compound one (not inside one begun)."""
        return (callable(getattr(self.policy, "option_rows", None))
                and getattr(self.policy, "has_options", True)
                and not inside(game)
                and any(opens(game, action) is not None for action in options))

    def _whole_own(self, game: Game, seat: int, first: Action, listed: tuple[Option, ...],
                   probs: np.ndarray | None) -> int | None:
        """Which of `listed` is the policy's own whole decision, opened by
        `first`: the one whose parts are `first` and the plan the policy
        keeps for the rest; without a plan, its most probable option opened
        by `first`."""
        plans = self._policy_plans()
        if plans is not None:
            parts = (first, *plans.peek(game, seat))
            for i, option in enumerate(listed):
                if (option.parts if isinstance(option, Compound) else (option,)) == parts:
                    return i
        opened = [i for i, option in enumerate(listed) if first_move(option) == first]
        if not opened or probs is None:
            return None
        return max(opened, key=lambda i: (probs[i], -i))

    def _hold_own(self, game: Game, seat: int, first: Action) -> None:
        """The policy's own whole decision, opened by `first` and played with
        no search: the rest of it held here, so the asks inside it are
        answered from it."""
        plans = self._policy_plans()
        if plans is not None:
            rest = plans.peek(game, seat)
            plans.drop(game)
        else:
            ((listed, probs),) = self.policy.option_rows([(game, seat)])
            own = self._whole_own(game, seat, first, tuple(listed), np.asarray(probs, dtype=np.float64))
            option = None if own is None else listed[own]
            rest = option.parts[1:] if isinstance(option, Compound) else ()
        self._plans.keep(game, seat, rest)

    def _decide_whole(self, game: Game, seat: int, first: Action, value: tuple, context: dict,
                      began: float) -> Decision:
        """A root that opens a compound decision, searched over the policy's
        whole options (`option_rows`): its own first, then the rest by
        probability. The chosen decision's first part is played and the rest
        held for the asks inside it."""
        ((listed, probs),) = self.policy.option_rows([(game, seat)])
        listed = tuple(listed)
        probs = np.asarray(probs, dtype=np.float64)
        own = self._whole_own(game, seat, first, listed, probs)
        self._drop(game)
        order = np.argsort(-probs, kind="stable")
        if own is None:
            own = int(order[0])
        if len(listed) == 1:
            decision = self._alone(listed, probs, value, own, context, began, whole=True)
        else:
            rest = [int(i) for i in order[: self.top] if i != own][: self.top - 1]
            candidates = (own, *rest)
            decision = self._search(game, seat, listed, probs, value, candidates, context, began, whole=True)
        option = listed[decision.chosen]
        self._plans.keep(game, seat, option.parts[1:] if isinstance(option, Compound) else ())
        return decision

    def choose(self, game: Game) -> Action:
        return self.decide(game).move

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
    policy, tasks: Sequence[tuple[Game, int]], moves: Sequence[Option], seat: int, start: int,
    horizon: int | None, marks: tuple[int, ...],
) -> Lanes:
    """Each `(world, seed)` of `tasks` copied once per move of `moves`
    (`paired_copy`), the move applied, and every copy played out by `policy`
    (`PairedRollouts.play_out`). A function of its arguments alone, so a
    worker process plays a share of the worlds as this one would."""
    return RolloutBatch.from_tasks(policy, tasks, moves, seat, start, marks).advance(horizon)


def _to_move(game) -> int:
    return game.to_move() if type(game) is NativeCopy else to_move(game)


def _legal(game) -> tuple[Action, ...]:
    return game.legal() if type(game) is NativeCopy else tuple(legal_actions(game))


def _apply(game, action: Action) -> None:
    if type(game) is NativeCopy:
        game.apply(action)
    else:
        apply(game, action)


def _apply_option(game, option: Option, seat: int) -> None:
    """A candidate applied to a copy: an action, or a whole decision's parts
    in order (`hexset.decisions.apply_option`), `seat` the discarding seat."""
    for part in (option.parts if isinstance(option, Compound) else (option,)):
        if is_over(game):
            return
        if type(game) is NativeCopy:
            game.apply(part, seat)
        else:
            apply(game, part, seat if part.type is ActionType.DISCARD else None)


def _stops(game: Game, seat: int, start: int, horizon: int | None) -> bool:
    if is_over(game):
        return True
    return (
        horizon is not None
        and game.turns - start >= horizon
        and _to_move(game) == seat
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
    def from_tasks(cls, policy, tasks: Sequence[tuple[Game, int]], moves: Sequence[Option],
                   seat: int, start: int, marks: Sequence[int] = ()) -> "RolloutBatch":
        """A paired copy per world and move, in world-major order."""
        batch = cls(policy, (), seat, start, marks, after=bool(marks))
        batch.add_tasks(tasks, moves)
        return batch

    def add_tasks(self, tasks: Sequence[tuple[Game, int]], moves: Sequence[Option]) -> tuple[int, ...]:
        """Append paired copies for new worlds or moves; return their indices.
        A world the native engine can hold for this policy (`copyable`) is
        copied once and held natively, each move's copy a fork of it."""
        copies = []
        contract = getattr(self.policy, "native_contract", None)
        for world, seed in tasks:
            base = paired_copy(world, seed, self.seat)
            native = held(base) if contract is not None and copyable(base, contract, self.policy.space) else None
            for k, move in enumerate(moves):
                if native is not None:
                    copy = native.fork()
                else:
                    copy = base if k == 0 else paired_copy(world, seed, self.seat)
                _apply_option(copy, move, self.seat)
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
               and _to_move(self.games[i]) == self.seat
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
            rows = [(self.games[i], _to_move(self.games[i]), _legal(self.games[i])) for i in live]
            moving = [(i, row) for i, row in zip(live, rows) if row[2]]
            self.counts[live] += 1
            for i, row in zip(live, rows):
                if not row[2]:
                    self.blocked[i] = True
            chosen = self.policy.act_rows([row for _, row in moving])
            for (i, _), action in zip(moving, chosen):
                _apply(self.games[i], action)
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


def _play_share(tasks, moves, seat, start, horizon, marks, context=None) -> Lanes:
    if context is not None:
        _worker_policy.set_context(context)
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
        self, policy, tasks: Sequence[tuple[Game, int]], moves: Sequence[Option], seat: int,
        start: int, horizon: int | None, marks: tuple[int, ...],
    ) -> Lanes:
        """`play_lanes(policy, tasks, ...)`, with the workers playing all
        but the first run of worlds; `policy` is this process's own."""
        if self.broken:
            return play_lanes(policy, tasks, moves, seat, start, horizon, marks)
        runs = _shares(tasks, self.workers + 1)
        # A policy that plays by state its searcher holds (`context`, as
        # `hexset.clients.conditioned.SeatRouted` holds each seat's condition)
        # sends it with every share; each worker's copy takes it first.
        context = policy.context() if hasattr(policy, "context") else None
        futures = []
        try:
            for run in runs[1:]:
                futures.append(self._executor.submit(_play_share, run, moves, seat, start, horizon, marks, context))
        except (BrokenProcessPool, RuntimeError):   # RuntimeError: retired since the check above
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

    def retire(self) -> None:
        """Takes no more work, without waiting: a decision whose worlds are
        out with the workers gets them back, and every later one plays here.
        The workers stop once their last share is done."""
        self.broken = True
        self._executor.shutdown(wait=False)

    def __enter__(self) -> "RolloutPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class SharedPools:
    """Rollout worker pools kept across bots, one per model file, so every
    seat a server spawns from one file plays its worlds out in the same
    `workers` processes instead of starting its own (`get`). A file loaded
    another way, or replaced on disk, gets a new pool, and the old one is
    retired (`RolloutPool.retire`): a seat still holding it plays its
    decisions here. A broken pool is replaced the same way. `close` stops
    every pool."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held: dict[str, tuple[tuple, RolloutPool]] = {}

    def get(self, path: str, key: tuple, load_policy: Callable[[], object], workers: int) -> RolloutPool:
        """The pool for the file at `path` loaded as `key` says (`key` names
        the file's state on disk and how its policy is loaded), started at
        the first call."""
        with self._lock:
            held = self._held.get(path)
            if held is not None and held[0] == key and not held[1].broken:
                return held[1]
            if held is not None:
                held[1].retire()
            pool = RolloutPool(load_policy, workers)
            self._held[path] = (key, pool)
            return pool

    def close(self) -> None:
        with self._lock:
            pools = [pool for _, pool in self._held.values()]
            self._held.clear()
        for pool in pools:
            pool.close()
