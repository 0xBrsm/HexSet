# SPDX-License-Identifier: GPL-3.0-only
"""`evaluate.Evaluator`'s term set, read through a `View`.

`HonestEvaluator` scores every seat from one knower's information. Board
terms are the existing evaluator's own `survey`, reused rather than copied
since it reads only public state. The three hand terms (`evaluate.hand_terms`
-- purchase progress, spare cards, robber exposure) are read on the true hand
for the knower (or for everyone, when `omniscient`) and on
`View.expected_hand` for everyone else; victory point cards count only for
the knower. `TRADING_WEIGHTS` and
`NO_TRADE_WEIGHTS` are the two shipped profiles `heximax()` picks between by
mode -- see their own comments for provenance; `weights=` overrides either
with a candidate vector, which is the hook `hexset.tuning` fits through.
"""

from __future__ import annotations

import random
from typing import Sequence

from hexset.board.board import Board
from ..evaluate import (
    ROLLS,
    WIN_SCORE,
    Evaluator,
    Survey,
    Weights,
    hand_terms,
)
from hexset.game import Game
from hexset.ledger import PublicLedger
from hexset.state import GameState
from hexset.victory import WINNING_POINTS, award_points, card_points

from hexset.view import View


# Today's fit, made under trading (`evaluate.Weights`' own docstring).
TRADING_WEIGHTS = Weights()

# The fit that preceded the trading refit, recovered from git: `87d9095`
# (parent of `1dd9045`, "refit the weights for trading"), `src/catan/evaluate.py`.
# That fit predates the scarcity term; it is set here at the corpus exchange
# rate `evaluate.FITTED_SCARCE` uses -- 0.91 pips at that fit's own
# `production / ROLLS` -- rather than at zero, so the no-trade profile carries
# the one term that was adopted untuned and won anyway; still to be refit.
NO_TRADE_WEIGHTS = Weights(
    victory_point=1.0,
    production=2.785,
    diversity=0.358,
    scarce=0.91 * 2.785 / ROLLS,
    # The three hand terms are the redesign's, at the trading table's fitted
    # values: they were refit under trading and this profile has not been
    # refit since the pre-trading fit, so carrying its own stale numbers
    # forward would mean the no-trade bot alone kept the cliff.
    buy_progress=Weights.buy_progress,
    road=0.1237,
    knight=0.1026,
    spare_card=Weights.spare_card,
    robber_risk=Weights.robber_risk,
    port=0.03063,
)


class HonestEvaluator:
    """`evaluate.Evaluator`'s model, read through a `View`.

    Board terms are the existing evaluator's own `survey`, reused rather
    than copied since it reads only public state. The three hand terms
    (`evaluate.hand_terms`) are read on the true hand for the knower and on
    `View.expected_hand` for everyone else (or everyone, when `omniscient`);
    victory-point cards count only for the knower. `buy_progress` on an
    expected hand is an approximation -- a maximum of minimums, so the value
    on the mean differs from the mean of the values -- which
    `exact_progress_samples > 0` replaces with an average over that many
    sampled hands. Supply- and board-aware: `hand_terms` prices only the
    purchases `survey` says this seat can still make.
    """

    def __init__(
        self, board: Board, weights: Weights | None = None, *, omniscient: bool = False,
        exact_progress_samples: int = 0,
    ) -> None:
        self.inner = Evaluator(board, weights)
        self.weights = self.inner.weights
        self.vector = self.inner.vector
        self.omniscient = omniscient
        self.exact_progress_samples = exact_progress_samples
        self._walk_cache: dict[tuple, Survey] = {}
        self._belief_cache: dict[tuple, View] = {}
        self._evaluate_cache: dict[tuple, list[float]] = {}

    def belief_for(
        self, state: GameState, ledger: PublicLedger, perspective: int, *,
        certify: Sequence[tuple[int, Sequence[int]]] = (),
    ) -> View:
        """`View(state, ledger, perspective, ...)`, memoized for the life of
        one `Heximax.choose()`.

        Exact by construction: the key is every field `View.__init__` reads
        to build `known`/`unknown`/`pool` -- each seat's hand *size*, the
        ledger's known/unknown, the bank, `num_players`, `perspective`,
        `certify` -- **plus the board occupancy and the robber**, which
        `View.__init__` does not read but `View.state` carries. Without
        those last two a hit could hand back a `View` whose `.state` is a
        different game's (right hands, stale board), which was a live trap
        the moment anything read `.state` off a cached view: the trade gate
        does exactly that (`Heximax._delta` reads `view.state`), so the key
        covers it rather than the caller having to remember not to.
        (`omniscient` is fixed for this evaluator's life.)

        `sample` and `deck_odds` read `self.state` further still -- deck,
        dev cards, knights played -- which this key does not capture, so
        `worlds`/`draw_children`, their only callers, build a fresh
        `View.from_game` instead.

        Built to be cheap on a hit, not just correct: this runs on every
        call, hit or miss, so its own cost is pure overhead on a cache that
        exists to avoid work. `map(tuple, ...)` over the two nested fields
        (hands, each seat's `known`) skips a generator's per-item frame
        switch that a comprehension pays for the same values in the same
        order; `certify` is `()` at both of this method's call sites today,
        so the common case skips building its sub-tuple at all rather than
        running an empty generator to discover it is empty. Every field is
        still the same one `View.__init__` reads plus board/robber, just
        assembled more directly -- the key's *value* is unchanged.
        """
        key = (
            tuple(map(tuple, state.hands)),
            tuple([tuple(seat_ledger.known) for seat_ledger in ledger.seats]),
            tuple([seat_ledger.unknown for seat_ledger in ledger.seats]),
            tuple(state.bank),
            tuple(state.vertex_owner),
            tuple(state.vertex_building),
            state.robber,
            state.num_players,
            perspective,
            () if not certify else tuple((who, tuple(bundle)) for who, bundle in certify),
        )
        cached = self._belief_cache.get(key)
        if cached is None:
            cached = View(
                state, ledger, perspective, omniscient=self.omniscient, certify=certify
            )
            self._belief_cache[key] = cached
        return cached

    def belief_from_game(self, game: Game, perspective: int) -> View:
        """`belief_for` for a live game -- see `belief_for`'s docstring for
        the exactness argument and which callers may use this.

        `game.state(perspective, hidden=False)` (true state: `belief_for`'s
        own content-keyed cache, not `Game.state`'s per-call construction,
        is what controls how often a `View` actually gets built --
        `View.from_game` builds unconditionally, so routing through it here
        would construct one throwaway `View` per call on top of the cached
        one. `hidden=False` costs nothing: the same object the engine's
        private state always was, never a copy. `View.__init__` -- engine
        code -- is what enforces honesty on it via `known`/`unknown`, not
        restricted access to the object.)
        """
        return self.belief_for(game.state(perspective, hidden=False), game.ledger, perspective)

    def _walk(self, state: GameState, seat: int) -> Survey:
        """`Evaluator.survey(state, seat)`, memoized for the life of one
        `Heximax.choose()`.

        A pure function of the board occupancy, the robber and the edges, so
        the key decides the value outright: caching changes nothing about
        what `terms` reads, only how often it is recomputed. The k sampled
        worlds share the root's occupancy, and most tree nodes move neither
        a vertex nor the robber. `Heximax.choose` clears the cache every
        decision.
        """
        key = (
            tuple(state.vertex_owner),
            tuple(state.vertex_building),
            tuple(state.edge_owner),
            state.robber,
            seat,
        )
        cached = self._walk_cache.get(key)
        if cached is None:
            cached = self.inner.survey(state, seat)
            self._walk_cache[key] = cached
        return cached

    def survey(self, state: GameState, seat: int) -> Survey:
        """`_walk`, under the name the rest of the codebase knows it by."""
        return self._walk(state, seat)

    def hand_terms(
        self, state: GameState, seat: int, hand: Sequence[float],
        walk: Survey | None = None,
    ) -> tuple[float, float, float]:
        """`evaluate.hand_terms` for `hand`, on this seat's memoized board facts."""
        return hand_terms(
            hand,
            self._walk(state, seat) if walk is None else walk,
            num_players=state.num_players,
            deck_left=len(state.deck),
        )

    def _hand_terms_of(
        self, state: GameState, seat: int, hand: Sequence[float], belief: View | None,
        walk: Survey | None = None,
    ) -> tuple[float, float, float]:
        """`hand_terms` on an estimated hand, optionally averaged over samples.

        `buy_progress` is a maximum of minimums, so its value on the mean hand
        is not the mean of its values; `exact_progress_samples > 0` draws that
        many hands from the belief instead.
        """
        if (
            belief is None
            or belief.exact(seat)
            or not self.exact_progress_samples
            or not belief.unknown[seat]
        ):
            return self.hand_terms(state, seat, hand, walk)
        rng = random.Random(seat)
        cards = belief._pool_cards()
        totals = [0.0, 0.0, 0.0]
        for _ in range(self.exact_progress_samples):
            counts = belief.known[seat][:]
            for r in rng.sample(cards, belief.unknown[seat]):
                counts[r] += 1
            for i, value in enumerate(self.hand_terms(state, seat, counts, walk)):
                totals[i] += value
        return (
            totals[0] / self.exact_progress_samples,
            totals[1] / self.exact_progress_samples,
            totals[2] / self.exact_progress_samples,
        )

    def terms(
        self, state: GameState, seat: int, hand: Sequence[float], *, knower: int | None = None,
        belief: View | None = None,
    ) -> tuple[float, ...]:
        """The raw term values `score` weights, in `evaluate.TERM_NAMES` order.

        `hand` is `state.hands[seat]` for the knower (or when `omniscient`)
        and `View.expected_hand(seat)` otherwise -- `evaluate` decides
        which and passes it in, so this method itself never has to ask.
        """
        walk = self._walk(state, seat)
        points = walk.buildings + award_points(state, seat)
        if seat == knower:
            points += card_points(state, seat)
        progress, spare, risk = self._hand_terms_of(state, seat, hand, belief, walk)
        return (
            points,
            walk.rate,
            walk.kinds,
            walk.scarce,
            progress,
            walk.roads,
            state.knights_played[seat],
            spare,
            risk,
            walk.port_gain,
        )

    def score(
        self, state: GameState, seat: int, hand: Sequence[float], *, knower: int | None = None,
        belief: View | None = None,
    ) -> float:
        """`terms` dotted with the weight vector, plus the win bonus at 10 VP."""
        values = self.terms(state, seat, hand, knower=knower, belief=belief)
        total = 0.0
        for weight, value in zip(self.vector, values):
            total += weight * value
        if values[0] >= WINNING_POINTS:
            total += WIN_SCORE
        return total

    def evaluate(
        self, state: GameState, knower: int | None = None, belief: View | None = None,
    ) -> list[float]:
        """Score every seat from `knower`'s information.

        Without a `belief` there is no ledger to read, so every opponent hand
        is taken as wholly untyped: `known` empty, `unknown` the public size.
        `evaluate_game` builds the real belief from the game's ledger.

        Memoized for the life of one `Heximax.choose()`, exactly: the key
        names every input `terms`/`score` read besides `hand`/`belief` --
        board occupancy and the robber (`survey`), road and knight counts,
        the longest-road/largest-army holders, the knower's own development
        cards (the only seat `card_points` scores) -- plus every seat's hand
        and the belief's `signature()`, which is `expected_hand`'s only
        input. A hit is byte-identical to recomputing, whether the belief
        came from `belief_for`, a fresh `View.from_game`, or the untyped
        fallback above.
        """
        if belief is None and knower is not None and not self.omniscient:
            belief = View(
                state, PublicLedger.new(state.num_players), knower, omniscient=False
            )
        key = (
            tuple(state.vertex_owner),
            tuple(state.vertex_building),
            state.robber,
            tuple(state.edge_owner),
            tuple(state.knights_played),
            state.longest_road_holder,
            state.largest_army_holder,
            knower,
            tuple(state.dev_cards[knower]) if knower is not None else None,
            tuple(state.new_dev_cards[knower]) if knower is not None else None,
            tuple(tuple(hand) for hand in state.hands),
            None if belief is None else belief.signature(),
        )
        cached = self._evaluate_cache.get(key)
        if cached is not None:
            return list(cached)
        out = []
        for seat in range(state.num_players):
            if self.omniscient or seat == knower or belief is None:
                hand: Sequence[float] = state.hands[seat]
            else:
                hand = belief.expected_hand(seat)
            out.append(self.score(state, seat, hand, knower=knower, belief=belief))
        self._evaluate_cache[key] = out
        return list(out)

    def evaluate_game(self, game: Game, seat: int) -> list[float]:
        """`evaluate`, building the belief from `game`'s own ledger. The leaf call.

        `belief_from_game` rather than `View.from_game` directly: this is
        the dominant caller of both (39.7 leaves/decision on the profile's
        own sample), and `evaluate`'s own memo only ever reads the belief's
        `known`/`unknown`/`pool`, never `self.state` -- exactly the subset
        `belief_from_game`'s cache is safe for (see its docstring).
        """
        belief = self.belief_from_game(game, seat)
        # true state: `evaluate`'s own memo keys on board occupancy and the
        # rest of the position, so it needs this game's own state object.
        # `belief.state` is now the same object -- `belief_for`'s key covers
        # occupancy and the robber too -- but reading it from the game is
        # what says so.
        return self.evaluate(game.state(seat, hidden=False), seat, belief)
