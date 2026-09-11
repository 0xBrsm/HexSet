# SPDX-License-Identifier: GPL-3.0-only
"""Heximax's evaluator driving `hexset.mcts` instead of its expectimax/max-n.

The experiment this module exists for: does PUCT search add anything on top
of a strong handcrafted evaluator, or is Heximax's fitted vector already
doing the work? The neural path (`hexset.clients.netbot.LeafEvaluator`)
plugs a policy/value head into the same `hexset.mcts.Evaluator` protocol;
this plugs `ViewEvaluator` in instead, so the comparison is search-vs-
search with the knowledge held fixed.

Values are per-seat win probabilities: each leaf's VP-denominated vector
goes through the `win` stance's fitted softmax (`WIN_TEMPERATURE`, maximum
likelihood against real game outcomes), and terminals are the one-hot
winner. That is the same scale the neural `LeafEvaluator` backs up, so the
tree maximises the mover's win probability -- `relative` on win
probabilities is exactly that in a duel -- with the knowledge held fixed
to Heximax's fitted vector. The tree's backup only implements linear
stances, so the (nonlinear) win conversion happens per leaf in the
evaluator rather than in the tree.

Each leaf is scored from its mover's information set (`Leaf.seat` is the
mover), the multiplayer-correct choice and what the neural evaluator does.

Priors are the evaluator's own one-ply read: each option is applied to one
determinization of the leaf and the mover's resulting score, softmaxed,
becomes the prior. That is the policy-head half of what MCTS wants,
computed from the same fitted vector as the value half rather than
learned separately. Chance actions (roll, hidden draws) get one sample,
the way the tree itself samples them per simulation -- exact expectation
here would cost what the search is for.

Setup and discard phases are not searched: they delegate to a plain
`Heximax`, whose `choose` resolves both directly (placement prior /
smallest marginal loss) before any search begins.

Importing this module registers the `heximax-mcts` arena preset
(`heximax(board)` defaults: trading mode, trading on).
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from hexset.actions import ActionType, apply
from hexset.arena import Entrant, register_entrant_kind, register_preset
from hexset.board.board import Board
from hexset.game import Game, Phase, imagine, is_over, roll_dice

from .heximax.evaluate import TRADING_WEIGHTS, ViewEvaluator, Weights
from .heximax.search import heximax
from hexset.mcts import Evaluator, Search


class HeximaxMCTSEvaluator(Evaluator):
    """`hexset.mcts.Evaluator` over a `ViewEvaluator`.

    Values are the `win`-stance reading of the evaluator's vector, so the
    tree maximises the mover's win probability. `stance` ranks the prior's
    one-ply scores, which stay in the evaluator's native VP units --
    `prior_temperature` is in VP units, so 1.0 means a one-ply edge of one
    VP buys e× the prior.
    """

    def __init__(
        self,
        board: Board,
        weights: Weights | None = None,
        rng: random.Random | None = None,
        *,
        stance: str = "relative",
        prior_temperature: float = 1.0,
    ) -> None:
        # Lazy, like `hexset.mcts.Search`: `hexset.bots` imports this module,
        # so a module-level import would meet the package half-built.
        from hexset.bots import STANCES
        from hexset.bots.stances import WIN_TEMPERATURE, win_at

        self._heximax = ViewEvaluator(board, weights)
        self._rng = rng or random.Random()
        self._rank = STANCES[stance]
        self._win_temperature = WIN_TEMPERATURE
        self._win_at = win_at
        self._prior_temperature = prior_temperature

    def new_search(self) -> None:
        """Drop the evaluator's per-position caches; called per decision."""
        self._heximax._walk_cache.clear()
        self._heximax._belief_cache.clear()
        self._heximax._evaluate_cache.clear()

    def _one_ply_scores(self, leaf) -> list[float]:
        """The mover's ranked score after each option, one determinization.

        Mirrors the beam ranking in `Heximax._value` -- the mover's `_rank`
        of the position the action leads to -- with one sample standing in
        for exact chance expectation. Ranked for the leaf's mover, not the
        child position's: the prior answers "which of my options looked
        best", the same question the beam asks.
        """
        mover = leaf.seat
        scores = []
        for action in leaf.options:
            child = imagine(leaf.game, self._rng)
            if action.type is ActionType.ROLL:
                roll_dice(child)
            else:
                apply(child, action)
            scores.append(self._rank(self._heximax.evaluate_game(child, mover), mover))
        return scores

    def evaluate(
        self, leaves: Sequence
    ) -> Sequence[tuple[Sequence[float], Sequence[float]]]:
        out = []
        for leaf in leaves:
            scores = self._heximax.evaluate_game(leaf.game, leaf.seat)
            seats = len(scores)
            # The `win`-stance reading of the vector, per seat: the tree backs
            # up win probabilities and maximises the mover's, which is what
            # shipped Heximax maximises at every decision node.
            values = tuple(
                self._win_at(scores, seat, self._win_temperature)
                for seat in range(seats)
            )
            priors = (
                _softmax(self._one_ply_scores(leaf), self._prior_temperature)
                if leaf.options
                else ()
            )
            out.append((priors, values))
        return out

    def terminal(self, game: Game) -> Sequence[float]:
        # One-hot winner, like the neural evaluator: a finished game is a
        # known win, not the fitted model's probabilistic reading of it.
        if not is_over(game):
            raise ValueError("terminal() called on a game that has not finished")
        players = game.state(0, hidden=False).num_players
        winner = game.won_by
        return tuple(1.0 if seat == winner else 0.0 for seat in range(players))


def _softmax(scores: Sequence[float], temperature: float) -> tuple[float, ...]:
    peak = max(scores)
    exps = [math.exp((s - peak) / temperature) for s in scores]
    total = sum(exps)
    return tuple(e / total for e in exps)


class HeximaxMCTS:
    """`Heximax`'s evaluator searched with PUCT instead of expectimax."""

    def __init__(
        self,
        board: Board,
        rng: random.Random | None = None,
        *,
        weights: Weights | None = None,
        simulations: int = 256,
        wave: int = 32,
    ) -> None:
        if weights is None:
            weights = TRADING_WEIGHTS
        stance = "relative"
        self._setup_bot = heximax(board, rng, mode="trading", weights=weights)
        self._evaluator = HeximaxMCTSEvaluator(board, weights, rng, stance=stance)
        self._search = Search(
            self._evaluator,
            simulations=simulations,
            wave=wave,
            stance=stance,
            rng=rng,
        )

    def choose(self, game: Game):
        if game.phase is Phase.SETUP_SETTLEMENT or game.phase is Phase.DISCARD:
            return self._setup_bot.choose(game)
        self._evaluator.new_search()
        return self._search.choose(game)

    # Trading: the search does not gate exchanges itself, so the bot trades
    # through the same gate (and floor) as the `Heximax` it borrows its
    # evaluator from. Without this the engine reads "defines no gate" as
    # "never trades", which would make the test about trading, not search.
    @property
    def trade_floor(self) -> float:
        return self._setup_bot.trade_floor

    def gains_many(self, view, received, counterparties) -> list[float]:
        return self._setup_bot.gains_many(view, received, counterparties)

    def accepts(self, view, received, counterparty: int) -> bool:
        return self._setup_bot.accepts(view, received, counterparty)


def _spawn(entrant: Entrant, board: Board, rng: random.Random) -> HeximaxMCTS:
    return HeximaxMCTS(
        board,
        rng,
        weights=entrant.weights,
        simulations=entrant.simulations,
        wave=entrant.wave,
    )


register_entrant_kind("heximax-mcts", _spawn)
register_preset(
    "heximax-mcts",
    Entrant("heximax-mcts", kind="heximax-mcts", simulations=256, wave=32),
)
