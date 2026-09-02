# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ..actions import Action, ActionType, apply, legal_actions, within_offer_budget
from .evaluate import Evaluator
from ..game import ROLL_ODDS, Game, imagine, is_over, roll_dice, to_move
from ..play import Stuck
from ..trading import Offer, execute as execute_trade, responders


class Bot(Protocol):
    """Anything that can pick an action. The network will implement this too."""

    def choose(self, game: Game) -> Action: ...


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


# How a seat turns the per-seat vector into the one number it maximises. The
# evaluation is unchanged; only the reading of it differs.
STANCES = {"own": own, "relative": relative, "paranoid": paranoid}


def options_for(game: Game) -> list[Action]:
    options = legal_actions(game)
    if not options:
        raise Stuck(f"no legal action in {game.phase.name} for player {to_move(game)}")
    return options


@dataclass
class RandomBot:
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
    partner_choice: bool = False
    # How many offers this bot will propose in a turn, below whatever the engine
    # allows. `None` spends the engine's whole budget. Kept here rather than in
    # the engine because a cap every seat receives cannot be duelled against
    # itself: only a bot that declines an action its opponent still has can say
    # what the action was worth.
    max_offers: int | None = None

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
        return self.evaluator.evaluate(game.state, seat)

    def choose(self, game: Game) -> Action:
        options = within_offer_budget(game, options_for(game), self.max_offers)
        if len(options) == 1:
            return self._addressed(game, options[0], to_move(game))
        # Only the seat to move may count its own hidden cards, and it stays the
        # perspective for the whole search: deeper nodes are still this seat's
        # reasoning about the game, not somebody else's.
        seat = to_move(game)
        candidates = self._beam(game, options, seat, seat)
        best = max(
            candidates,
            key=lambda a: self._rank(self._after(game, a, self.depth, seat), seat),
        )
        return self._addressed(game, best, seat)

    def _addressed(self, game: Game, action: Action, seat: int) -> Action:
        """Name who the proposer would rather have take the offer, best first.

        Only worth computing when more than one player could cover it. The
        search valued the offer under the engine's neutral order, so ordering it
        afterwards can only improve on what was searched, never contradict it.
        """
        if not self.partner_choice or action.type is not ActionType.PROPOSE_TRADE:
            return action
        offer = Offer(proposer=seat, give=action.give, want=action.want)
        willing = responders(game.state, offer)
        if len(willing) < 2:
            return action

        def value(responder: int) -> float:
            child = imagine(game, self.rng)
            execute_trade(child.state, offer, responder)
            return self._rank(self._leaf(child, seat), seat)

        return action._replace(ask=tuple(sorted(willing, key=value, reverse=True)))

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
        total = [0.0] * game.state.num_players
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
    partner_choice: bool = False,
    max_offers: int | None = None,
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
        partner_choice=partner_choice,
        max_offers=max_offers,
    )
