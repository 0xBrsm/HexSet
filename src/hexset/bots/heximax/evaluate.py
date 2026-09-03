# SPDX-License-Identifier: GPL-3.0-only
"""`evaluate.Evaluator`'s term set, read through a `View`.

`HonestEvaluator` scores every seat from one knower's information. Board
terms are the existing evaluator's own `survey`, reused rather than copied
since it reads only public state. The three hand terms (`progress`, `held`,
`surplus_card`) are read on the true hand for the knower (or for everyone,
when `omniscient`) and on `View.expected_hand` for everyone else; victory
point cards count only for the knower. `TRADING_WEIGHTS` and
`NO_TRADE_WEIGHTS` are the two shipped profiles `heximax()` picks between by
mode -- see their own comments for provenance; `weights=` overrides either
with a candidate vector, which is the hook `hexset.tuning` fits through.
"""

from __future__ import annotations

import random
from typing import Sequence

from hexset.board.board import Board
from hexset.economy import COSTS, Purchase
from ..evaluate import (
    PROGRESS_PURCHASES,
    ROLLS,
    WIN_SCORE,
    Evaluator,
    Survey,
    Weights,
)
from hexset.game import Game
from hexset.ledger import PublicLedger
from hexset.robber import DISCARD_THRESHOLD
from hexset.state import MAX_CITIES, MAX_SETTLEMENTS, Building, GameState
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
    progress=0.1371,
    road=0.1237,
    knight=0.1026,
    card=0.005406,
    surplus_card=-0.3891,
    port=0.03063,
)


# `COSTS[purchase]` with the zero entries dropped, and its own divisor:
# `progress_toward`'s inner loop, run three times per seat per leaf.
_PROGRESS_COST: dict[Purchase, tuple[tuple[tuple[int, int], ...], int]] = {
    purchase: (
        tuple((r, n) for r, n in enumerate(COSTS[purchase]) if n),
        sum(COSTS[purchase]),
    )
    for purchase in PROGRESS_PURCHASES
}


class HonestEvaluator:
    """`evaluate.Evaluator`'s model, read through a `View`.

    Board terms are the existing evaluator's own `survey`, reused rather
    than copied since it reads only public state. The three hand terms
    (`progress`, `held`, `surplus_card`) are read on the true hand for the
    knower and on `View.expected_hand` for everyone else (or everyone,
    when `omniscient`); victory-point cards count only for the knower.
    `progress` on an expected hand is an approximation -- a maximum of
    minimums, so the value on the mean differs from the mean of the values --
    which `exact_progress_samples > 0` replaces with an average over that
    many sampled hands. Supply-aware: progress toward a purchase whose piece
    supply is exhausted is zero, so the maximum falls back to what can still
    be built.
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
        self._walk_cache: dict[tuple, tuple[Survey, tuple[int, int]]] = {}
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
        """
        key = (
            tuple(tuple(hand) for hand in state.hands),
            tuple(tuple(seat_ledger.known) for seat_ledger in ledger.seats),
            tuple(seat_ledger.unknown for seat_ledger in ledger.seats),
            tuple(state.bank),
            tuple(state.vertex_owner),
            tuple(state.vertex_building),
            state.robber,
            state.num_players,
            perspective,
            tuple((who, tuple(bundle)) for who, bundle in certify),
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

    def _walk(self, state: GameState, seat: int) -> tuple[Survey, tuple[int, int]]:
        """`(Evaluator.survey(state, seat), _pieces(state, seat))`, memoized
        for the life of one `Heximax.choose()`.

        Both are pure functions of the board occupancy (`survey` also of the
        robber), so the key decides the value outright: caching changes
        nothing about what `terms` reads, only how often it is recomputed.
        92.4% of the calls in one decision repeat a key already seen -- the
        k sampled worlds share the root's occupancy, and most tree nodes
        move neither a vertex nor the robber. They share one key because
        `terms` needs both and the key is the expensive part.
        `Heximax.choose` clears the cache every decision.
        """
        key = (tuple(state.vertex_owner), tuple(state.vertex_building), state.robber, seat)
        cached = self._walk_cache.get(key)
        if cached is None:
            cached = (self.inner.survey(state, seat), _pieces(state, seat))
            self._walk_cache[key] = cached
        return cached

    def survey(self, state: GameState, seat: int) -> Survey:
        """The board half of `_walk`."""
        return self._walk(state, seat)[0]

    def progress_toward(
        self, state: GameState, seat: int, hand: Sequence[float], purchase: Purchase,
        pieces: tuple[int, int] | None = None,
    ) -> float:
        if purchase is Purchase.SETTLEMENT or purchase is Purchase.CITY:
            settlements, cities = pieces if pieces is not None else self._walk(state, seat)[1]
            if purchase is Purchase.SETTLEMENT and settlements >= MAX_SETTLEMENTS:
                return 0.0
            if purchase is Purchase.CITY and cities >= MAX_CITIES:
                return 0.0
        # `_PROGRESS_COST` is `COSTS[purchase]` with the zero entries dropped
        # and the divisor kept alongside; this runs three times per seat per
        # leaf. The `min`/`sum` pair stays: since 3.12 `sum` compensates its
        # float error (Neumaier), so an accumulator loop here is a different
        # number in the last bit -- which is enough to flip a near-tie. A
        # list comprehension over the same operands in the same order feeds
        # `sum` the identical values, so the result is bit-for-bit the same
        # as the generator it replaces -- only the generator's per-item
        # frame-switch overhead (needless for `needed`'s two or three pairs)
        # is gone.
        needed, total = _PROGRESS_COST[purchase]
        return sum([min(hand[r], n) for r, n in needed]) / total

    def progress(
        self, state: GameState, seat: int, hand: Sequence[float],
        pieces: tuple[int, int] | None = None,
    ) -> float:
        # The vertex walk SETTLEMENT and CITY both need is done once -- here,
        # or by `terms`, which has it from the same memoized `_walk`.
        if pieces is None:
            pieces = self._walk(state, seat)[1]
        best = 0.0
        for purchase in PROGRESS_PURCHASES:
            toward = self.progress_toward(state, seat, hand, purchase, pieces)
            if toward > best:
                best = toward
        return best

    def _progress_of(
        self, state: GameState, seat: int, hand: Sequence[float], belief: View | None,
        pieces: tuple[int, int] | None = None,
    ) -> float:
        if (
            belief is None
            or belief.exact(seat)
            or not self.exact_progress_samples
            or not belief.unknown[seat]
        ):
            return self.progress(state, seat, hand, pieces)
        rng = random.Random(seat)
        cards = belief._pool_cards()
        total = 0.0
        for _ in range(self.exact_progress_samples):
            counts = belief.known[seat][:]
            for r in rng.sample(cards, belief.unknown[seat]):
                counts[r] += 1
            total += self.progress(state, seat, counts, pieces)
        return total / self.exact_progress_samples

    def terms(
        self, state: GameState, seat: int, hand: Sequence[float], *, knower: int | None = None,
        belief: View | None = None,
    ) -> tuple[float, ...]:
        """The raw term values `score` weights, in `evaluate.TERM_NAMES` order.

        `hand` is `state.hands[seat]` for the knower (or when `omniscient`)
        and `View.expected_hand(seat)` otherwise -- `evaluate` decides
        which and passes it in, so this method itself never has to ask.
        """
        walk, pieces = self._walk(state, seat)
        held = sum(hand)
        points = walk.buildings + award_points(state, seat)
        if seat == knower:
            points += card_points(state, seat)
        return (
            points,
            walk.rate,
            walk.kinds,
            walk.scarce,
            self._progress_of(state, seat, hand, belief, pieces),
            state.edge_owner.count(seat),  # `list.count`: the same walk, in C
            state.knights_played[seat],
            held,
            max(0, held - DISCARD_THRESHOLD),
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


def _pieces(state: GameState, seat: int) -> tuple[int, int]:
    """(settlements, cities) `seat` has on the board, in one walk."""
    settlements = cities = 0
    for vertex, owner in enumerate(state.vertex_owner):
        if owner != seat:
            continue
        if state.vertex_building[vertex] == Building.SETTLEMENT:
            settlements += 1
        elif state.vertex_building[vertex] == Building.CITY:
            cities += 1
    return settlements, cities
