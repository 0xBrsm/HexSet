# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ..actions import Action, ActionType, apply, legal_actions
from .evaluate import Evaluator, hand_shifted
from ..game import ROLL_ODDS, Game, imagine, is_over, roll_dice, to_move
from ..play import Stuck
from ..trading import Bundle, exchange
from ..view import View


class Bot(Protocol):
    """Anything that can pick an action, and what it brings to a trade.

    `choose` is the whole of the old protocol and still the only required
    method. The other three are the trade mechanic's seam
    (`hexset.trading`), and all have a default that means "this seat never
    trades" or "answer one candidate at a time", so an existing bot keeps
    working untouched:

    * `accepts(view, received, counterparty)` -- this seat's private
      judgement of one concrete exchange, `received` signed and positive
      towards this seat. Default: False.
    * `accepts_many(view, received, counterparties)` -- this seat's private
      judgement of every candidate in `received` at once, in the same
      order. Default: loop over `accepts`, so a bot that only ever answers
      one candidate at a time is unaffected; a bot that can answer a whole
      batch in one forward (`hexset.clients.onnxbot.NetworkBot`) overrides
      it.
    * `gains_many(view, received, counterparties)` -- this seat's private
      *gain* from every candidate in `received` at once: a signed float, in
      whatever unit this seat's value is, positive meaning it wants the
      trade. This is the mechanic's actual gate -- `hexset.trading.
      trade_event` clears the candidate both sides clear `TRADE_FLOOR` on
      and `Game.trade_rule` ranks highest. Default: `+1.0`/`-1.0` from
      `accepts_many`, for a bot that only ever has a boolean gate.

    All three are handed the engine's information-set `View` for that seat
    and nothing else, so none can be a function of anything the seat may
    not know. The defaults are applied by `hexset.trading.valued`/
    `valued_many` rather than by inheritance, so they hold for a bot that
    satisfies this protocol structurally.
    """

    def choose(self, game: Game) -> Action: ...

    def accepts(self, view: View, received: Bundle, counterparty: int) -> bool:
        return False

    def accepts_many(
        self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[bool]:
        return [self.accepts(view, r, c) for r, c in zip(received, counterparties)]

    def gains_many(
        self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[float]:
        verdicts = self.accepts_many(view, received, counterparties)
        return [1.0 if ok else -1.0 for ok in verdicts]


def own(vector: Sequence[float], seat: int) -> float:
    """Plain max^n: each seat wants its own score high and ignores the rest."""
    return vector[seat]


def relative(vector: Sequence[float], seat: int) -> float:
    """Own score less the average of everyone else's.

    A constant-sum reading of the vector. Catan has exactly one winner, so a
    position is only worth what it is worth *compared to* the table, and an
    action that lifts everyone equally has achieved nothing.
    """
    others = [v for p, v in enumerate(vector) if p != seat]
    return vector[seat] - sum(others) / len(others)


def paranoid(vector: Sequence[float], seat: int) -> float:
    """Own score less the best opponent's. The leader is the only rival."""
    others = [v for p, v in enumerate(vector) if p != seat]
    return vector[seat] - max(others)


# Fitted by maximum likelihood over 4,463 per-turn score-vector samples from
# 48 `heximax`x4 games (stance `relative`, the shipped mechanic at fit time),
# one sample per turn at the mover's first decision, labelling the eventual
# winner, seed 40000: mean log loss 1.107 against the eventual winner, vs.
# 1.386 for the uniform baseline. `agents/reference/heximax.md`, "Registration
# 2026-09-04: the objective — a win-probability stance against the
# relative-VP stance".
WIN_TEMPERATURE = 2.476644394795811


def win(vector: Sequence[float], seat: int) -> float:
    """Softmax(vector / WIN_TEMPERATURE)[seat]: the seat's win probability.

    This reads the vector as the seat's win probability, which is what the
    game actually pays, rather than a margin over the table — at a
    temperature fitted by maximum likelihood against real game outcomes (see
    `WIN_TEMPERATURE`). Numerically stable: the max is subtracted before
    exponentiating. `relative` remains search2's frozen stance.
    """
    scaled = [v / WIN_TEMPERATURE for v in vector]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    return exps[seat] / sum(exps)


# How a seat turns the per-seat vector into the one number it maximises. The
# evaluation is unchanged; only the reading of it differs.
STANCES = {"own": own, "relative": relative, "paranoid": paranoid, "win": win}


def options_for(game: Game) -> list[Action]:
    options = legal_actions(game)
    if not options:
        raise Stuck(f"no legal action in {game.phase.name} for player {to_move(game)}")
    return options


@dataclass
class RandomBot:
    """Uniform over the legal actions. Never advertises, never trades: the
    trade mechanic has no random baseline to offer, and a seat that
    publishes nothing simply does not participate (`Bot`'s defaults)."""

    rng: random.Random = field(default_factory=random.Random)

    def choose(self, game: Game) -> Action:
        return self.rng.choice(options_for(game))


@dataclass
class SearchBot:
    """Max^n search over the handcrafted evaluation.

    `depth` counts decisions, not turns. A Catan turn contains many actions, so
    depth two plans a pair of the mover's own actions rather than reaching an
    opponent; passing the turn is itself one of the actions searched.

    Each seat maximises its own component of the evaluation vector, which is
    max^n rather than minimax — with more than two players there is no single
    opponent to minimise, and assuming everyone ganged up on the mover would
    model the table badly.

    A roll is a chance node expanded over all eleven outcomes and weighted by
    probability rather than sampled, so the value is not noisy. `width` beams
    the branching, since the main phase can offer sixty-odd actions.

    `stance` is how a seat reads the per-seat vector — see `STANCES`. Every
    seat in the tree reads it the same way, so a relative stance models a table
    that all thinks relatively, not one bot that has noticed something. It
    defaults to `relative`, which beat plain max^n 53.6% over 2000 games: the
    baseline exists to be beaten, so it should be the best one available.
    """

    evaluator: Evaluator
    depth: int = 2
    width: int | None = 6
    rng: random.Random = field(default_factory=random.Random)
    stance: str = "relative"
    # The trade off switch. `0` publishes nothing and refuses everything, so
    # this seat never trades; anything else (including the `None` default)
    # trades. Not a budget -- the engine has no cap (`hexset.trading`) -- and
    # kept on the bot rather than on the engine because a knob every seat
    # receives cannot be duelled against itself: only a bot that declines
    # what its opponent still has can say what it was worth.
    max_trades: int | None = None

    def __post_init__(self) -> None:
        if self.stance not in STANCES:
            raise ValueError(f"unknown stance: {self.stance}")
        self._rank = STANCES[self.stance]
        # A leaf evaluation that wants the whole `Game` rather than the state
        # says so by offering `evaluate_game`. The learned one does, because the
        # encoder reads the phase, the turn count and the free-road counter, and
        # none of those are on `GameState`. The handcrafted evaluations need
        # only the state and keep the cheaper call — this is resolved once here
        # rather than tested at every leaf.
        self._leaf = getattr(self.evaluator, "evaluate_game", None) or self._from_state

    def _from_state(self, game: Game, seat: int) -> list[float]:
        # true state: search2 is a sanctioned true-state reader by design
        # (the project's one held-out perfect-information referent).
        return self.evaluator.evaluate(game.state(seat, hidden=False), seat)

    def choose(self, game: Game) -> Action:
        options = options_for(game)
        if len(options) == 1:
            return options[0]
        # Only the seat to move may count its own hidden cards, and it stays the
        # perspective for the whole search: deeper nodes are still this seat's
        # reasoning about the game, not somebody else's.
        seat = to_move(game)
        candidates = self._beam(game, options, seat, seat)
        return max(
            candidates,
            key=lambda a: self._rank(self._after(game, a, self.depth, seat), seat),
        )

    # -- trading (`hexset.trading`) -----------------------------------------

    def _gain(self, view: View, received: Bundle, counterparty: int) -> float:
        """This seat's own evaluator delta from one candidate exchange:
        `Eval(after) - Eval(before)`, scored under this bot's stance -- which
        under `relative` already prices who got stronger. This is the private
        gate the mechanic clears on; `accepts`/`gains_many` are both read off
        it.
        """
        seat = view.perspective
        state = view.state
        before = self._rank(self.evaluator.evaluate(state, seat), seat)
        mirror = tuple(-n for n in received)
        after = hand_shifted(state, {seat: received, counterparty: mirror})
        return self._rank(self.evaluator.evaluate(after, seat), seat) - before

    def gains_many(
        self, view: View, received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[float]:
        if self.max_trades == 0:
            return [-1.0] * len(received)
        return [self._gain(view, r, c) for r, c in zip(received, counterparties)]

    def accepts(self, view: View, received: Bundle, counterparty: int) -> bool:
        """Take the exchange iff the imagined post-trade position scores
        strictly better under this bot's stance -- `gains_many(...)[0] > 0`,
        the engine's termination argument rests on strictness
        (`hexset.trading.trade_event`).
        """
        return self.gains_many(view, [received], [counterparty])[0] > 0.0

    def _beam(
        self, game: Game, options: list[Action], mover: int, knower: int
    ) -> list[Action]:
        if self.width is None or len(options) <= self.width:
            return options
        ranked = sorted(
            options,
            key=lambda a: -self._rank(self._after(game, a, 1, knower), mover),
        )
        return ranked[: self.width]

    def _after(self, game: Game, action: Action, depth: int, knower: int) -> list[float]:
        """Value of the position `action` leads to, with `depth - 1` plies left."""
        if action.type is ActionType.ROLL:
            return self._over_dice(game, depth, knower)
        child = imagine(game, self.rng)
        apply(child, action)
        return self._value(child, depth - 1, knower)

    def _over_dice(self, game: Game, depth: int, knower: int) -> list[float]:
        total = [0.0] * game.num_players
        for roll, weight in ROLL_ODDS:
            child = imagine(game, self.rng)
            roll_dice(child, roll)
            for p, value in enumerate(self._value(child, depth - 1, knower)):
                total[p] += weight * value
        return total

    def _value(self, game: Game, depth: int, knower: int) -> list[float]:
        if depth <= 0 or is_over(game):
            return self._leaf(game, knower)
        options = legal_actions(game)
        if not options:
            return self._leaf(game, knower)

        mover = to_move(game)
        best: list[float] | None = None
        best_rank = 0.0
        for action in self._beam(game, options, mover, knower):
            vector = self._after(game, action, depth, knower)
            rank = self._rank(vector, mover)
            if best is None or rank > best_rank:
                best, best_rank = vector, rank
        assert best is not None
        return best


def greedy(
    evaluator: Evaluator,
    rng: random.Random | None = None,
    stance: str = "relative",
    max_trades: int | None = None,
) -> SearchBot:
    """One ply: take the action with the best position after it.

    Cheap enough to run tens of thousands of games, and the reference the deeper
    search has to beat before depth is worth paying for.
    """
    return SearchBot(
        evaluator,
        depth=1,
        width=None,
        rng=rng or random.Random(),
        stance=stance,
        max_trades=max_trades,
    )
