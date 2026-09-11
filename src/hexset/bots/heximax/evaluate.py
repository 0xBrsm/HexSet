# SPDX-License-Identifier: GPL-3.0-only
"""`evaluate.Evaluator`'s term set, read through a `View`.

`HonestEvaluator` scores every seat from one knower's information. Board
terms are the existing evaluator's own `survey`, reused rather than copied
since it reads only public state. The three hand terms (`evaluate.hand_terms`
-- purchase progress, spare cards, robber exposure) are read on the true hand
for the knower and on `View.expected_hand` for everyone else; victory point cards are exact for
the knower and an expectation over the unseen pool for everyone else
(`expected_card_points`). `TRADING_WEIGHTS` and `NO_TRADE_WEIGHTS` are the
two shipped profiles `heximax()` picks between by mode -- see their own
comments for provenance; `weights=` overrides either with a candidate
vector, which is how `hexset.fitting`'s result is played before adoption.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from hexset.board.board import Board
from ..evaluate import (
    PURCHASE_COST,
    PURCHASE_VALUE,
    ROLLS,
    WIN_SCORE,
    Evaluator,
    Survey,
    Weights,
    affordable,
    hand_terms,
    seven_before_next_turn,
)
from hexset.board.terrain import NUM_RESOURCES
from hexset.cards import DECK_COMPOSITION, DevCard
from hexset.devcards import holdings
from hexset.economy import COSTS
from hexset.game import Game
from hexset.ledger import PublicLedger
from hexset.state import GameState
from hexset.victory import award_points, card_points

from hexset.view import View

# Victory-point cards in the deck. None is ever revealed before its holder
# wins, so every one not in the knower's own hand is in the unseen pool.
VP_CARDS = DECK_COMPOSITION[DevCard.VICTORY_POINT]


def expected_card_points(state: GameState, seat: int, knower: int | None) -> float:
    """Victory-point cards `seat` is expected to hold, from `knower`'s information.

    The knower's own VP cards are exact (`card_points`); an opponent's are
    hidden until it wins, so its row gets the number of development cards it
    holds -- public -- times the share of the *unseen* pool that is a VP
    card. The unseen pool is the deck plus every development hand the knower
    cannot see, and every VP card outside the knower's own hand is in it.

    Without this the anchor term (`points`) means one thing in the knower's
    row and another in every other row: a seat holding three development
    cards is on average most of a point ahead of one holding none, and an
    evaluator that scores that seat at zero for them ranks the table wrong
    in exactly the term whose weight is pinned to one.
    """
    held = sum(holdings(state, seat))
    if not held:
        return 0.0
    unseen_vp = VP_CARDS - (card_points(state, knower) if knower is not None else 0)
    unseen = len(state.deck) + sum(
        sum(holdings(state, s)) for s in range(state.num_players) if s != knower
    )
    return held * unseen_vp / unseen if unseen else 0.0


# `hand_terms`' purchases in the order it considers them, so `score_many`'s
# argmax breaks a tie the same way that loop's strictly-greater test does.
_PURCHASES: tuple = tuple(PURCHASE_VALUE)

# Their costs as rows, plus a trailing all-zeros row `score_many` indexes for
# "no purchase scored above zero", which makes the whole hand spare.
_PURCHASE_COSTS = np.array(
    [list(COSTS[purchase]) for purchase in _PURCHASES] + [[0] * NUM_RESOURCES],
    dtype=float,
)


# `evaluate.Weights` (the one-ply fit, `evaluate.Weights`' own docstring)
# with the one term the play sweep moved. `hexset.bench.weight_sweep`,
# 2026-09-07 (trading play sweep): every free
# term rung at x0, x0.5, x2 and then x0.71, x1.41 of its value against the
# incumbent on 1,024 paired boards; only `robber_risk` cleared the bar
# (-0.15 -> -0.30: 54.2% [51.1, 57.2]; zero reads 36.8%), and the swept
# vector confirmed 52.5% [50.7, 54.3] over 3,072 fresh boards against the
# start. The other seven terms sit within a point and a half of their best
# ring value, which is the resolution 1,024 paired games buy.
TRADING_WEIGHTS = Weights(robber_risk=-0.30)

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
    # The three hand terms are the redesign's, then swept by play on the
    # no-trade table (`hexset.bench.weight_sweep --profile notrade`,
    # 2026-09-07, no-trade play sweep): the same
    # rings as the trading profile, 1,024 paired games a cell against the
    # incumbent. Two moves cleared the bar -- `robber_risk` -0.15 -> -0.30
    # (53.2% [50.2, 56.3]; zero reads 37.8%) and `spare_card` 0.15 -> 0.1065
    # (53.1% [50.1, 56.2]; x1.41 reads 39.6%) -- and the swept vector
    # confirmed 52.9% [51.1, 54.7] over 3,072 fresh boards against the start.
    # The six other terms held at every ring value.
    buy_progress=Weights.buy_progress,
    road=0.1237,
    knight=0.1026,
    spare_card=0.1065,
    robber_risk=-0.30,
    port=0.03063,
)


class HonestEvaluator:
    """`evaluate.Evaluator`'s model, read through a `View`.

    Board terms are the existing evaluator's own `survey`, reused rather
    than copied since it reads only public state. The three hand terms
    (`evaluate.hand_terms`) are read on the true hand for the knower and on
    `View.expected_hand` for everyone else; victory-point cards are exact
    for the knower and `expected_card_points` otherwise. `buy_progress` on an
    expected hand is an approximation -- a maximum of minimums, so the value
    on the mean differs from the mean of the values -- which
    `exact_progress_samples > 0` replaces with an average over that many
    sampled hands. Supply- and board-aware: `hand_terms` prices only the
    purchases `survey` says this seat can still make.
    """

    def __init__(
        self, board: Board, weights: Weights | None = None, *,
        exact_progress_samples: int = 0, development_value: float = 0.0,
    ) -> None:
        self.inner = Evaluator(board, weights)
        self.weights = self.inner.weights
        self.vector = self.inner.vector
        self.exact_progress_samples = exact_progress_samples
        self.development_value = development_value
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
            cached = View(state, ledger, perspective, certify=certify)
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
            state.rules,
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
            discard_limit=state.rules.discard_limit,
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

        `hand` is `state.hands[seat]` for the knower and
        `View.expected_hand(seat)` otherwise -- `evaluate` decides
        which and passes it in, so this method itself never has to ask.
        """
        walk = self._walk(state, seat)
        points = walk.buildings + award_points(state, seat)
        if seat == knower:
            points += card_points(state, seat)
        else:
            points += expected_card_points(state, seat, knower)
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

    def development_bonus(self, state: GameState, seat: int, knower: int | None) -> float:
        """Optional value for unused non-VP cards, honest about hidden types.

        The holder knows its own VP count. For opponents, subtract only the
        expected VP count from their public total. Played cards are excluded
        by holdings(). A zero coefficient preserves the shipped evaluator.
        """
        if not self.development_value:
            return 0.0
        vp = card_points(state, seat) if seat == knower else expected_card_points(state, seat, knower)
        return self.development_value * (sum(holdings(state, seat)) - vp)

    def score(
        self, state: GameState, seat: int, hand: Sequence[float], *, knower: int | None = None,
        belief: View | None = None,
    ) -> float:
        """`terms` dotted with the weight vector, plus the win bonus at the
        game type's winning points."""
        values = self.terms(state, seat, hand, knower=knower, belief=belief)
        total = 0.0
        for weight, value in zip(self.vector, values):
            total += weight * value
        if values[0] >= state.rules.winning_points:
            total += WIN_SCORE
        return total + self.development_bonus(state, seat, knower)

    def score_many(self, state: GameState, knower: int, hands: np.ndarray) -> np.ndarray:
        """`score`, over every row of `hands` at once: `(candidates, seat,
        resource)` in, `(candidates, seat)` out.

        `hands` is built by the caller (`Heximax._post_trade_hands`) from
        whatever mixture of exact and `View.expected_hand` values each
        seat's row calls for; this method never reads a belief itself, only
        the array, so it has no opinion on where a row came from.

        `terms` reads exactly three of its ten values off `hand` -- the
        `evaluate.hand_terms` trio -- everything else (`points`,
        `walk.rate/kinds/scarce/port_gain`, road count, knights played) is a
        pure function of `state` and `seat`, so it is computed once per
        seat here (the same calls `terms` makes, `_walk` already memoized)
        and broadcast over every candidate rather than repeated per row.

        The three hand terms are `hand_terms` transposed onto the candidate
        axis, term for term. Which purchases a seat can still make is a
        board fact (`evaluate.affordable`), so the gate is computed once per
        seat and an unaffordable purchase scores -1 rather than 0 -- below
        the `best = 0.0` the scalar loop starts from, so it can never win an
        argmax the scalar `>` would have refused. `argmax` takes the first
        maximum, which is the same tie-break as that strictly-greater loop
        over `PURCHASE_VALUE` in order. `best_cost` is then gathered by that
        index, and the "nothing is worth saving for" case is the all-zeros
        cost row, which makes `spare` the whole hand at bank rate without a
        second branch.

        The accumulation below follows `score`'s own term-by-term loop in
        the same order, so a batch of one row agrees with `score` to
        floating-point noise, not by construction -- `gains_many`'s own
        caller is what checks that (`atol=1e-12` over real trade events),
        not this method.

        Silently wrong for a `HonestEvaluator` built with
        `exact_progress_samples > 0`: that reads a belief's sampled hands
        per non-exact seat, which this array has no belief to resample
        from. Nothing in this repository sets it above zero; the one caller
        that might (`Heximax._delta_scalar_honest`) is routed around this
        method entirely rather than asked to fake a belief for it.
        """
        n = hands.shape[0]
        num_players = state.num_players
        vector = self.vector

        points = np.empty(num_players)
        rate = np.empty(num_players)
        kinds = np.empty(num_players)
        scarce = np.empty(num_players)
        edge = np.empty(num_players)
        knights = np.empty(num_players)
        port = np.empty(num_players)
        # Per seat: the cheapest rate it can trade each resource at, and
        # which of the four purchases the board still leaves open to it.
        rates = np.empty((num_players, NUM_RESOURCES))
        open_to = np.empty((num_players, len(_PURCHASES)))

        deck_left = len(state.deck)
        for seat in range(num_players):
            walk = self._walk(state, seat)
            pts = walk.buildings + award_points(state, seat)
            if seat == knower:
                pts += card_points(state, seat)
            else:
                pts += expected_card_points(state, seat, knower)
            points[seat] = pts
            rate[seat] = walk.rate
            kinds[seat] = walk.kinds
            scarce[seat] = walk.scarce
            edge[seat] = walk.roads
            knights[seat] = state.knights_played[seat]
            port[seat] = walk.port_gain
            rates[seat] = walk.ratios
            for i, purchase in enumerate(_PURCHASES):
                open_to[seat, i] = affordable(purchase, walk, deck_left)

        held = hands.sum(axis=2)
        # What the whole hand fetches at this seat's own bank or port rate.
        bank_value = (hands / rates).sum(axis=2)

        # (candidates, seat, purchase): how far each hand has got towards
        # each purchase, priced by `PURCHASE_VALUE`, with anything the board
        # has closed sent below zero so it cannot win the argmax.
        scored = np.empty((n, num_players, len(_PURCHASES)))
        for i, purchase in enumerate(_PURCHASES):
            needed, total = PURCHASE_COST[purchase]
            toward = np.zeros((n, num_players))
            for r, need in needed:
                toward += np.minimum(hands[:, :, r], need)
            scored[:, :, i] = PURCHASE_VALUE[purchase] * toward / total
        scored = np.where(open_to > 0.0, scored, -1.0)

        chosen = scored.argmax(axis=2)
        progress = np.maximum(np.take_along_axis(
            scored, chosen[:, :, None], axis=2
        )[:, :, 0], 0.0)
        # Index `len(_PURCHASES)` is the all-zeros row: no purchase scored
        # above zero, so nothing in the hand is committed to one.
        best_cost = _PURCHASE_COSTS[np.where(progress > 0.0, chosen, len(_PURCHASES))]
        spare = (np.maximum(0.0, hands - best_cost) / rates).sum(axis=2)

        over = held - state.rules.discard_limit
        ramp = np.clip(over, 0.0, 1.0)
        risk = seven_before_next_turn(num_players) * 0.5 * ramp * bank_value

        values = (points, rate, kinds, scarce, progress, edge, knights, spare, risk, port)
        total = np.zeros((n, num_players))
        for weight, value in zip(vector, values):
            total = total + weight * value
        total = total + WIN_SCORE * (points >= state.rules.winning_points)
        if self.development_value:
            total = total + np.array([self.development_bonus(state, seat, knower)
                                     for seat in range(num_players)])
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
        the longest-road/largest-army holders, every seat's development-card
        holdings and the deck size (`card_points`/`expected_card_points`) --
        plus every seat's hand and the belief's `signature()`, which is
        `expected_hand`'s only input. A hit is byte-identical to
        recomputing, whether the belief came from `belief_for`, a fresh
        `View.from_game`, or the untyped fallback above.
        """
        if belief is None and knower is not None:
            belief = View(state, PublicLedger.new(state.num_players), knower)
        key = (
            state.rules,
            tuple(state.vertex_owner),
            tuple(state.vertex_building),
            state.robber,
            tuple(state.edge_owner),
            tuple(state.knights_played),
            state.longest_road_holder,
            state.largest_army_holder,
            knower,
            tuple(
                tuple(map(sum, zip(held, fresh)))
                for held, fresh in zip(state.dev_cards, state.new_dev_cards)
            ),
            len(state.deck),
            tuple(tuple(hand) for hand in state.hands),
            None if belief is None else belief.signature(),
        )
        cached = self._evaluate_cache.get(key)
        if cached is not None:
            return list(cached)
        out = []
        for seat in range(state.num_players):
            if seat == knower or belief is None:
                hand: Sequence[float] = state.hands[seat]
            else:
                hand = belief.expected_hand(seat)
            out.append(self.score(state, seat, hand, knower=knower, belief=belief))
        self._evaluate_cache[key] = out
        return list(out)

    def rows(
        self, state: GameState, knower: int, belief: View | None = None,
    ) -> list[tuple[float, ...]]:
        """`terms` for every seat from `knower`'s information -- the per-seat
        feature rows `evaluate` dots with the weights, before the dot.

        What `hexset.fitting` regresses on: the same model the search scores,
        term for term and read through the same belief, so a fitted vector
        drops straight into `Weights` with nothing lost in translation. Not
        memoized -- the fit replays recorded games once, it never revisits a
        position the way a search does.
        """
        if belief is None:
            belief = View(state, PublicLedger.new(state.num_players), knower)
        out = []
        for seat in range(state.num_players):
            if seat == knower or belief is None:
                hand: Sequence[float] = state.hands[seat]
            else:
                hand = belief.expected_hand(seat)
            out.append(self.terms(state, seat, hand, knower=knower, belief=belief))
        return out

    def rows_game(self, game: Game, knower: int) -> list[tuple[float, ...]]:
        """`rows`, with the belief built from `game`'s own ledger."""
        belief = self.belief_from_game(game, knower)
        # true state: the same object `evaluate_game` reads for the same
        # reason -- `View.__init__` enforces honesty through
        # `known`/`unknown`, and `rows` reads an opponent's hand only through
        # `belief.expected_hand`.
        return self.rows(game.state(knower, hidden=False), knower, belief)

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
