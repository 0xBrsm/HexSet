# SPDX-License-Identifier: GPL-3.0-only
"""heximax: the handcrafted baseline that does not read the opponents' hands.

`bots.SearchBot` over `evaluate.Evaluator` -- `search2` -- is the project's
one clean held-out referent, and it cheats: its evaluation reads every seat's
true hand, its tree expands opponents from their true hands and development
cards, and a steal or a dev-card buy is valued on one frozen draw. heximax
is the next generation of that bot, built to the design in
`agents/reference/heximax.md`. It is **information-set honest by default**:
every quantity about an opponent is read through the public ledger
(`game.ledger`, `known[5]` + `unknown`) and the public counts, never through
`state.hands[opponent]` or `state.dev_cards[opponent]`. Its own hand is
exact. An `omniscient` mode keeps the old reading, so the price of honesty
can be measured rather than assumed.

One file, four sections, so that a downstream copy takes one file:

* ``belief``   -- the information set as an object: what is certified, what
  is hidden, and the residual pool the hidden cards are drawn from.
* ``evaluate`` -- `evaluate.Evaluator`'s term set read through the belief,
  with progress zeroed where the piece supply is exhausted, and two weight
  profiles (a trading table and a no-trade table).
* ``search``   -- max^n with a node budget and iterative deepening, opponents
  expanded from determinized samples of the belief (PIMC over `k` worlds),
  and every hidden draw averaged over its distribution.
* ``trade``    -- a valuation, protocol-free (marginal values, `deficit` and
  `surplus`, `candidate_bundles`, `score_proposal`, `accept_rule`,
  `counter_of`, `rank_partners`), then a thin protocol adapter over it
  (`Heximax.propose_actions`, and `_options_in`'s `accept_rule` gate) --
  mechanical, untuned, and expected to be rewritten whenever the trading
  protocol changes; only the valuation is fitted or tested for strength.

The adapter replaces the engine's one-for-one `PROPOSE_TRADE` sample with
heximax's own top-`propose_top_n` bundle proposals while `max_offers` still
has room, and gates the tree's own `ACCEPT_TRADE` with `accept_rule`
wherever a `TRADE_RESPOND` node is reached, root or not -- the tree's own
responses are still searched from the responder's own seat. `max_offers=0`
never proposes and always declines.

Cost: leaf evaluations per move are capped by `max_nodes`
(`DEFAULT_MAX_NODES`, 600). Measured 2026-09-02, before the structural
performance pass (three four-seat games a side, board seeds 0/1/2,
`search2-offers3`'s `max_offers=3` matching heximax's own budget): `heximax`
5.14 ms/move vs `search2-offers3` 1.79 ms/move, **2.87x** -- over the
design's 2x ceiling. The pass's two exact, census-preserving steps
(`agents/reference/heximax.md`'s structural registration, step (a): one
`Belief` per node, memoized per decision; step (b): `HonestEvaluator.evaluate`
memoized per decision) followed the same day, on the same protocol, on a
box under concurrent load (a registered bridge run): pooled over four
independent three-games-a-side round trips (12 games a side, 4336 heximax
moves, 9672 `search2-offers3` moves, so the absolute ms are inflated by
contention but the paired ratio is not), `heximax` 14.87 -> 14.07 ms/move,
`search2-offers3` (unchanged, the control) 5.19 -> 5.22 ms/move (flat, the
run-to-run noise floor), ratio **2.86x -> 2.69x** -- still over the 2x
ceiling. `Belief.from_game` calls drop from 37.2 to 3.9/decision (the
belief cache's own hit rate, 64.1%, on the profile's 200-position sample;
`worlds`/`draw_children` stay uncached, see `HonestEvaluator.belief_for`'s
docstring), and the belief category's tottime share drops from 16.0% to
5.0%; the evaluator memo (step (b)) hits 41.1% of its own lookups. Neither
step is a compute problem to begin with: `score_proposal`'s crisp `willing`
gate, read under `relative`, proposes far more selectively than the
engine's naive one-for-one sample, so a real game trades roughly a third as
often (`test_multi_card_and_one_for_one_proposals_both_occur_over_twenty_games`),
and those fewer, cheap negotiation actions average against the
leaf-budgeted build decisions that dominate the rest; the ROBBER phase's
leaf count (mean 247.2 vs `search2`'s 141.4 on the same corpus) is the
larger remaining lever, and it is a *decision-count* problem steps (a)/(b)
do not touch by design -- exactness meant caching identical recomputation,
not spending the node budget differently. Whether the trade gate is too
strict, `relative` is the wrong stance for `willing`, the ceiling needs a
protocol allowance, or a behaviour-changing step ((c)-(e) in the
registration) is warranted is a design question, not one this pass answers
by loosening a gate to hit a number. `bot.choose()`'s own choices, and the
number of leaves it spends getting to them, are checked on every position
by `test_choices_are_byte_identical_to_the_recorded_census`, byte-identical
through both steps above. History (the optimization pass and its
per-change breakdown) is in `agents/reference/heximax.md`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from .actions import (
    Action,
    ActionType,
    apply,
    legal_actions,
    victim_of,
    within_offer_budget,
)
from .board.board import Board
from .board.terrain import NUM_RESOURCES
from .bots import STANCES, options_for
from .cards import DECK_COMPOSITION, NUM_DEV_CARDS, DevCard
from .economy import BANK_TRADE_RATIO, COSTS, Purchase, trade_ratios
from .evaluate import (
    PROGRESS_PURCHASES,
    ROLLS,
    WIN_SCORE,
    Evaluator,
    Survey,
    Weights,
)
from .game import ROLL_ODDS, Game, Phase, imagine, is_over, roll_dice, to_move
from .ledger import PublicLedger
from .mcts import draws_hidden
from .placement import best as best_opening
from .robber import DISCARD_THRESHOLD
from .trading import Bundle, Offer, can_accept, can_propose
from .state import (
    BANK_PER_RESOURCE,
    MAX_CITIES,
    MAX_SETTLEMENTS,
    Building,
    GameState,
    copy_state,
)
from .victory import WINNING_POINTS, award_points, card_points

MODES = ("honest", "omniscient", "notrade")

# Sentinel for `heximax(max_offers=...)`: "whatever the mode's own budget is".
BY_MODE: int = object()  # type: ignore[assignment]

# --- public API ---------------------------------------------------------------
#
# heximax(board, ...) -> Heximax   the three shipped presets, by `mode`
# Heximax                          the bot: a `Bot.choose(game) -> Action`
# Belief                           the information set (`Belief.from_game`)
# HonestEvaluator                  the honest evaluation
#
# Sections below, in dependency order: belief, evaluate, search, trade.
# Private helpers are grouped under the section that owns them.


# --- belief -------------------------------------------------------------------


class Belief:
    """What one seat can know about every hand, and how to draw from it.

    Per seat: `known[s]` is the certified lower bound on each resource and
    `unknown[s]` the number of cards the record cannot type. The perspective's
    own seat is exact (`known` is the hand, `unknown` is zero), and so is every
    seat when `omniscient`. Everything hidden is drawn from one shared
    **residual pool**: per resource, the cards that are neither in the bank
    nor certified in any seat's `known`, sized from the bank's initial count
    rather than the true hands, which the belief may not read. An open offer
    certifies one thing the ledger does not: the proposer holds what it
    offers (see `from_game`). Robustness over purity: a test fixture that
    writes `state.hands` behind the ledger's back can leave `known` summing
    past the public hand size, or the pool short; the belief clamps `known`
    to size and pads the pool proportionally rather than raise, because a
    baseline that cannot cope with a position is not a baseline.
    """

    def __init__(
        self,
        state: GameState,
        ledger: PublicLedger,
        perspective: int,
        *,
        omniscient: bool = False,
        certify: Sequence[tuple[int, Sequence[int]]] = (),
    ) -> None:
        self.state = state
        self.perspective = perspective
        self.omniscient = omniscient
        n = state.num_players
        self.num_players = n
        self.sizes = [sum(hand) for hand in state.hands]
        self.known: list[list[int]] = []
        self.unknown: list[int] = []
        for seat in range(n):
            if omniscient or seat == perspective:
                self.known.append(state.hands[seat][:])
                self.unknown.append(0)
                continue
            known = ledger.seats[seat].known[:]
            for who, bundle in certify:
                if who == seat:
                    for r, need in enumerate(bundle):
                        if need > known[r]:
                            known[r] = need
            excess = sum(known) - self.sizes[seat]
            while excess > 0:
                # The record overclaims (a desynced fixture): shed certainty,
                # largest entry first, until it fits the public size.
                r = max(range(NUM_RESOURCES), key=lambda i: known[i])
                taken = min(excess, known[r])
                known[r] -= taken
                excess -= taken
            self.known.append(known)
            self.unknown.append(self.sizes[seat] - sum(known))

        pool = [
            BANK_PER_RESOURCE - state.bank[r] - sum(self.known[s][r] for s in range(n))
            for r in range(NUM_RESOURCES)
        ]
        pool = [max(0, p) for p in pool]
        deficit = sum(self.unknown) - sum(pool)
        if deficit > 0:
            pool = _padded(pool, deficit)
        self.pool = pool
        self.pool_size = sum(pool)

    @classmethod
    def from_game(cls, game: Game, perspective: int, *, omniscient: bool = False) -> Belief:
        # Only the proposer's side of a standing offer is certified: it is
        # announced and `can_propose` requires holding it.
        # `game.pending_responders` is the engine's true eligibility list, but
        # whether OTHER pending seats can cover the offer is deliberately not
        # read -- a decline reveals nothing -- so a sampled world may hand a
        # later responder a hand that cannot cover `want`; the search guards
        # `ACCEPT_TRADE` with `can_accept` there instead.
        return cls(
            game.state,
            game.ledger,
            perspective,
            omniscient=omniscient,
            certify=_offer_certify(game),
        )

    def exact(self, seat: int) -> bool:
        """Whether `seat`'s hand is read verbatim rather than estimated."""
        return self.omniscient or seat == self.perspective

    def expected_hand(self, seat: int) -> list[float]:
        """`known` plus the hidden cards spread in the pool's proportions.

        Exact in expectation for every linear term; the nonlinear `progress`
        term read on it is an approximation, see `HonestEvaluator`.
        """
        known = self.known[seat]
        hidden = self.unknown[seat]
        if not hidden or not self.pool_size:
            return [float(k) for k in known]
        share = hidden / self.pool_size
        return [k + share * p for k, p in zip(known, self.pool)]

    def table_holding(self, resource: int) -> float:
        """How much of `resource` every other seat is expected to hold, together."""
        return sum(
            self.expected_hand(seat)[resource]
            for seat in range(self.num_players)
            if seat != self.perspective
        )

    def steal_odds(self, victim: int) -> list[float]:
        """Probability that a card taken at random from `victim` is each resource."""
        size = self.sizes[victim]
        if not size:
            return [0.0] * NUM_RESOURCES
        return [e / size for e in self.expected_hand(victim)]

    def unseen_dev_cards(self) -> list[int]:
        """The development cards this seat has not seen, by type.

        The deck's initial composition less the perspective's own holdings and
        every knight anyone has played. Road building, year of plenty and
        monopoly plays are not counted by the state once resolved, so those
        cards stay in the estimate after they have gone: an approximation
        that overstates the unseen count slightly and is documented here.
        When `omniscient`, it is the true remaining deck.
        """
        state = self.state
        if self.omniscient:
            counts = [0] * NUM_DEV_CARDS
            for card in state.deck:
                counts[card] += 1
            return counts
        counts = [DECK_COMPOSITION[card] for card in DevCard]
        me = self.perspective
        for card in range(NUM_DEV_CARDS):
            counts[card] -= state.dev_cards[me][card] + state.new_dev_cards[me][card]
        counts[DevCard.KNIGHT] -= sum(state.knights_played)
        return [max(0, c) for c in counts]

    def deck_odds(self) -> list[float]:
        """Probability that the top card of the deck is each type."""
        unseen = self.unseen_dev_cards()
        total = sum(unseen)
        if not total:
            return [0.0] * NUM_DEV_CARDS
        return [c / total for c in unseen]

    def p_holds(
        self,
        seat: int,
        bundle: Sequence[int],
        *,
        draws: int = 64,
        rng: random.Random | None = None,
    ) -> float:
        """Probability that `seat` can cover `bundle`.

        Exact where the answer is decided by `known` alone, or where exactly
        one more card of one type is needed (a hypergeometric tail over the
        pool). Otherwise a Monte-Carlo estimate over `draws` hands sampled the
        way `sample` samples them; `rng` defaults to a fixed stream so the
        estimate is repeatable.
        """
        known = self.known[seat]
        need = [max(0, b - k) for b, k in zip(bundle, known)]
        short = sum(need)
        if short == 0:
            return 1.0
        hidden = self.unknown[seat]
        if short > hidden or self.exact(seat):
            return 0.0
        if any(n > p for n, p in zip(need, self.pool)):
            return 0.0
        if short == 1:
            r = need.index(1)
            return 1.0 - _hypergeometric_miss(self.pool_size, self.pool[r], hidden)
        rng = rng or random.Random(0)
        cards = self._pool_cards()
        hits = 0
        for _ in range(draws):
            drawn = rng.sample(cards, hidden)
            counts = known[:]
            for r in drawn:
                counts[r] += 1
            if all(c >= b for c, b in zip(counts, bundle)):
                hits += 1
        return hits / draws

    def _pool_cards(self) -> list[int]:
        return [r for r in range(NUM_RESOURCES) for _ in range(self.pool[r])]

    def sample(self, rng: random.Random) -> GameState:
        """One determinized world consistent with everything public.

        Every hidden hand is `known` plus `unknown` cards drawn without
        replacement from the shared pool, seat after seat, so no card is dealt
        twice. The perspective's hand is untouched. Opponents keep their public
        development-card *count* while the types are redrawn from the unseen
        composition and placed in `dev_cards` -- all matured, a conservative
        reading that lets a sampled opponent play any card it could hold. The
        deck is rebuilt from what remains unseen, at its public length, in
        random order. Invariant: every seat's hand size and development-card
        count match the real position.
        """
        state = copy_state(self.state)
        if self.omniscient:
            rng.shuffle(state.deck)
            return state

        cards = self._pool_cards()
        rng.shuffle(cards)
        cursor = 0
        for seat in range(self.num_players):
            if seat == self.perspective:
                continue
            hand = self.known[seat][:]
            hidden = self.unknown[seat]
            for r in cards[cursor : cursor + hidden]:
                hand[r] += 1
            cursor += hidden
            state.hands[seat] = hand

        composition = self.unseen_dev_cards()
        unseen = [c for c in range(NUM_DEV_CARDS) for _ in range(composition[c])]
        rng.shuffle(unseen)
        cursor = 0
        for seat in range(self.num_players):
            if seat == self.perspective:
                continue
            count = sum(state.dev_cards[seat]) + sum(state.new_dev_cards[seat])
            held = [0] * NUM_DEV_CARDS
            for card in unseen[cursor : cursor + count]:
                held[card] += 1
            dealt = min(count, max(0, len(unseen) - cursor))
            held[DevCard.KNIGHT] += count - dealt
            cursor += count
            state.dev_cards[seat] = held
            state.new_dev_cards[seat] = [0] * NUM_DEV_CARDS
        deck = unseen[cursor : cursor + len(state.deck)]
        deck.extend([int(DevCard.KNIGHT)] * (len(state.deck) - len(deck)))
        state.deck = deck
        return state


def _offer_certify(game: Game) -> list[tuple[int, Sequence[int]]]:
    """`Belief.from_game`'s certify list, factored out so `HonestEvaluator`'s
    memoized `belief_for` (below) can build the same list a cache key needs
    without duplicating the offer-reading logic."""
    if game.offer is not None:
        return [(game.offer.proposer, game.offer.give)]
    return []


def _padded(pool: list[int], deficit: int) -> list[int]:
    """`pool` grown by `deficit` cards in its own proportions (uniform if empty)."""
    total = sum(pool)
    weights = [p / total for p in pool] if total else [1 / len(pool)] * len(pool)
    shares = [w * deficit for w in weights]
    grown = [p + int(s) for p, s in zip(pool, shares)]
    left = deficit - sum(int(s) for s in shares)
    order = sorted(range(len(pool)), key=lambda r: -(shares[r] - int(shares[r])))
    for r in order[:left]:
        grown[r] += 1
    return grown


def _hypergeometric_miss(population: int, successes: int, draws: int) -> float:
    """P(no success in `draws` without replacement) = C(N-K, n) / C(N, n)."""
    if successes <= 0:
        return 1.0
    if draws > population - successes:
        return 0.0
    return math.comb(population - successes, draws) / math.comb(population, draws)


# --- evaluate -----------------------------------------------------------------

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
    """`evaluate.Evaluator`'s model, read through a `Belief`.

    Board terms are the existing evaluator's own `survey`, reused rather
    than copied since it reads only public state. The three hand terms
    (`progress`, `held`, `surplus_card`) are read on the true hand for the
    knower and on `Belief.expected_hand` for everyone else (or everyone,
    when `omniscient`); victory-point cards count only for the knower.
    `progress` on an expected hand is an approximation -- a maximum of
    minimums, so the value on the mean differs from the mean of the values --
    that `exact_progress_samples > 0` replaces with an average over that many
    hands sampled from the belief. Supply-aware: progress toward a purchase
    whose piece supply is exhausted is zero, so the maximum falls back to
    what can still be built.
    """

    def __init__(
        self,
        board: Board,
        weights: Weights | None = None,
        *,
        omniscient: bool = False,
        exact_progress_samples: int = 0,
    ) -> None:
        self.inner = Evaluator(board, weights)
        self.weights = self.inner.weights
        self.vector = self.inner.vector
        self.omniscient = omniscient
        self.exact_progress_samples = exact_progress_samples
        self._walk_cache: dict[tuple, tuple[Survey, tuple[int, int]]] = {}
        self._belief_cache: dict[tuple, Belief] = {}
        self._evaluate_cache: dict[tuple, list[float]] = {}

    def belief_for(
        self,
        state: GameState,
        ledger: PublicLedger,
        perspective: int,
        *,
        certify: Sequence[tuple[int, Sequence[int]]] = (),
    ) -> Belief:
        """`Belief(state, ledger, perspective, ...)`, memoized for the life of
        one `Heximax.choose()` -- the structural pass's step (a).

        Exact by construction: the key is every field `Belief.__init__` reads
        to build `known`/`unknown`/`pool` (every seat's hand *size*, the
        ledger's known/unknown per seat, the bank, `num_players`,
        `perspective` and `certify`) -- `omniscient` is fixed for this
        evaluator's whole life, so it need not be in the key. Two calls
        sharing a key are the same `Belief` byte-for-byte because `Belief` is
        a pure function of exactly these inputs; nothing about *which* node
        produced them can leak in, since none of them lives on `self.state`.

        Restricted to callers that only ever read the memoized fields off the
        result (`expected_hand`/`table_holding`/`steal_odds`/`p_holds`/
        `exact`) -- `Belief.sample` and `Belief.deck_odds`
        (`unseen_dev_cards`) read `self.state` in full (board occupancy, the
        deck, dev cards, knights played), which this key does not capture, so
        a `Belief` cached under one node's key could hand back a different
        node's board there. `Heximax.worlds`/`draw_children`, the only two
        callers that use `sample`/`deck_odds`, keep building a fresh
        `Belief.from_game` directly, never through this cache.
        """
        key = (
            tuple(tuple(hand) for hand in state.hands),
            tuple(tuple(seat_ledger.known) for seat_ledger in ledger.seats),
            tuple(seat_ledger.unknown for seat_ledger in ledger.seats),
            tuple(state.bank),
            state.num_players,
            perspective,
            tuple((who, tuple(bundle)) for who, bundle in certify),
        )
        cached = self._belief_cache.get(key)
        if cached is None:
            cached = Belief(
                state, ledger, perspective, omniscient=self.omniscient, certify=certify
            )
            self._belief_cache[key] = cached
        return cached

    def belief_from_game(self, game: Game, perspective: int) -> Belief:
        """`belief_for`, reading `certify` off `game.offer` the way
        `Belief.from_game` does -- see `belief_for`'s docstring for the
        exactness argument and which callers may use this."""
        return self.belief_for(
            game.state, game.ledger, perspective, certify=_offer_certify(game)
        )

    def _walk(self, state: GameState, seat: int) -> tuple[Survey, tuple[int, int]]:
        """`(Evaluator.survey(state, seat), _pieces(state, seat))`, memoized
        for the life of one `Heximax.choose()`.

        Both are pure functions of the board occupancy (`survey` also of the
        robber -- see its own docstring), so the same key always yields the
        same value: caching changes nothing about what `terms` reads, only
        how often it is recomputed. Within one decision, 92.4% of these calls
        are exact repeats of an already-seen key (the k sampled worlds share
        the root's board occupancy, and many tree nodes never move a vertex
        or the robber), so this turns most of that into a dict lookup. The
        two are cached together under one key because `terms` needs both on
        every call and the key is the expensive part to build.
        `Heximax.choose` clears the cache at the top of every call, so it
        never grows across decisions.
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
        self,
        state: GameState,
        seat: int,
        hand: Sequence[float],
        purchase: Purchase,
        pieces: tuple[int, int] | None = None,
    ) -> float:
        if purchase is Purchase.SETTLEMENT or purchase is Purchase.CITY:
            settlements, cities = pieces if pieces is not None else self._walk(state, seat)[1]
            if purchase is Purchase.SETTLEMENT and settlements >= MAX_SETTLEMENTS:
                return 0.0
            if purchase is Purchase.CITY and cities >= MAX_CITIES:
                return 0.0
        # `_PROGRESS_COST` is `COSTS[purchase]` with the zero entries dropped
        # and the divisor kept alongside: this runs three times per seat per
        # leaf, which is where an `enumerate`/`sum` per call showed up.
        needed, total = _PROGRESS_COST[purchase]
        held = 0.0
        for r, n in needed:
            have = hand[r]
            held += have if have < n else n
        return held / total

    def progress(self, state: GameState, seat: int, hand: Sequence[float]) -> float:
        # The vertex walk SETTLEMENT and CITY both need is done once here
        # (memoized with the survey) and passed down.
        pieces = self._walk(state, seat)[1]
        best = 0.0
        for purchase in PROGRESS_PURCHASES:
            toward = self.progress_toward(state, seat, hand, purchase, pieces)
            if toward > best:
                best = toward
        return best

    def _progress_of(
        self, state: GameState, seat: int, hand: Sequence[float], belief: Belief | None
    ) -> float:
        if (
            belief is None
            or belief.exact(seat)
            or not self.exact_progress_samples
            or not belief.unknown[seat]
        ):
            return self.progress(state, seat, hand)
        rng = random.Random(seat)
        cards = belief._pool_cards()
        total = 0.0
        for _ in range(self.exact_progress_samples):
            counts = belief.known[seat][:]
            for r in rng.sample(cards, belief.unknown[seat]):
                counts[r] += 1
            total += self.progress(state, seat, counts)
        return total / self.exact_progress_samples

    def terms(
        self,
        state: GameState,
        seat: int,
        hand: Sequence[float],
        *,
        knower: int | None = None,
        belief: Belief | None = None,
    ) -> tuple[float, ...]:
        """The raw term values `score` weights, in `evaluate.TERM_NAMES` order.

        `hand` is `state.hands[seat]` for the knower (or when `omniscient`)
        and `Belief.expected_hand(seat)` otherwise -- `evaluate` decides
        which and passes it in, so this method itself never has to ask.
        """
        walk = self.survey(state, seat)
        held = sum(hand)
        points = walk.buildings + award_points(state, seat)
        if seat == knower:
            points += card_points(state, seat)
        return (
            points,
            walk.rate,
            walk.kinds,
            walk.scarce,
            self._progress_of(state, seat, hand, belief),
            state.edge_owner.count(seat),  # `list.count`: the same walk, in C
            state.knights_played[seat],
            held,
            max(0, held - DISCARD_THRESHOLD),
            walk.port_gain,
        )

    def score(
        self,
        state: GameState,
        seat: int,
        hand: Sequence[float],
        *,
        knower: int | None = None,
        belief: Belief | None = None,
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
        self, state: GameState, knower: int | None = None, belief: Belief | None = None
    ) -> list[float]:
        """Score every seat from `knower`'s information.

        Without a `belief` there is no ledger to read, so every opponent hand
        is taken as wholly untyped: `known` empty, `unknown` the public size.
        `evaluate_game` builds the real belief from the game's ledger.

        Memoized for the life of one `Heximax.choose()` -- the structural
        pass's step (b). Exact by construction: the key names every input
        `terms`/`score` (and what they call -- `survey`, `progress`,
        `victory.award_points`/`card_points`) read besides `hand`/`belief`
        themselves -- board occupancy and the robber (`survey`), road and
        knight counts, the longest-road/largest-army holders, the knower's
        own development cards (the only seat `card_points` scores) -- plus
        every seat's hand and `belief`'s own `known`/`unknown`/`pool`
        (`expected_hand`'s only inputs), so a hit is byte-identical to
        recomputing regardless of whether `belief` came from `belief_for`,
        a fresh `Belief.from_game`, or this method's own untyped fallback.
        """
        if belief is None and knower is not None and not self.omniscient:
            belief = Belief(
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
            None
            if belief is None
            else (
                tuple(tuple(k) for k in belief.known),
                tuple(belief.unknown),
                tuple(belief.pool),
            ),
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

        `belief_from_game` rather than `Belief.from_game` directly: this is
        the dominant caller of both (39.7 leaves/decision on the profile's
        own sample), and `evaluate`'s own memo only ever reads the belief's
        `known`/`unknown`/`pool`, never `self.state` -- exactly the subset
        `belief_from_game`'s cache is safe for (see its docstring).
        """
        belief = self.belief_from_game(game, seat)
        return self.evaluate(game.state, seat, belief)


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


# --- search -------------------------------------------------------------------

# Leaf evaluations a move may spend. Chosen so the default configuration costs
# no more than twice `search2` per move (the design's ceiling): at 600 the
# mean is 1.5x and the per-move tail, which the unbounded search takes to
# ~1500 leaves, is cut at the budget. Figures in the module docstring.
DEFAULT_MAX_NODES = 600

# A roll taken this many plies or fewer below the root is expanded over all
# eleven outcomes; deeper rolls are sampled once. At the default depth of two
# every roll in the tree is exact.
EXACT_ROLL_PLIES = 2

# How many of `candidate_bundles`' candidates `propose_actions` runs the
# partner-aware `score_proposal` on, after the cheap `deficit`/`surplus`
# pre-filter. `score_proposal` is the adapter's dominant cost (a handful of
# `bundle_delta` calls per candidate, each two evaluations); this bound is
# what keeps that cost off the leaf budget's books. See `propose_actions`.
PROPOSE_SHORTLIST = 5


class _Exhausted(Exception):
    """The leaf budget ran out mid-search; the caller falls back."""


class _Forced:
    """A stand-in for the game's RNG that makes `robber.steal` take one card.

    `steal` draws `randrange(total)` and walks the hand in resource order, so
    returning the index of the first card of the wanted resource makes the
    draw deterministic. Nothing else on the steal path consults the RNG.
    """

    __slots__ = ("index",)

    def __init__(self, index: int) -> None:
        self.index = index

    def randrange(self, _stop: int) -> int:
        return self.index


@dataclass
class Heximax:
    """Max^n over the honest evaluation, within a leaf budget.

    `depth` counts decisions, `width` beams the branching, and `max_nodes`
    caps the leaf evaluations one `choose` may spend: the search deepens
    iteratively, one ply at a time, while the next ply's estimated cost fits
    what is left, and a ply that overruns the budget is abandoned for the
    last completed one -- whatever the branching, no move costs more than
    `max_nodes` leaves. Opponents are expanded from `k` determinized worlds
    drawn from the belief at the root (`Belief.sample`) and the root values
    averaged across them -- perfect-information Monte Carlo; in `omniscient`
    mode `k` is ignored and the true state is searched. Hidden draws are
    expectations, not one sample: a steal is averaged over the victim's
    expected composition, a dev-card buy over the unseen deck composition,
    each weighted by its probability. Rolls are exact eleven-way within
    `EXACT_ROLL_PLIES` of the root and sampled beyond.

    Opening settlements come from `placement.best` when `placement` is set;
    opening roads are searched. A discard gives up the card with the smallest
    marginal loss; a monopoly names the resource the table is expected to
    hold most of. `max_offers` is the bot's own budget below the engine's; at
    zero it never proposes and always declines. The trade adapter
    (`# --- trade` section): while `max_offers` has room, `PROPOSE_TRADE`
    root options are the top `propose_top_n` candidates from
    `candidate_bundles`, ranked by `score_proposal` and cut off at
    `propose_margin` (`propose_actions`); a `TRADE_RESPOND` node may only
    offer `ACCEPT_TRADE` to the search when `accept_rule` clears
    `accept_margin` there too (`_options_in`). Both margins are unfitted;
    `0.0` accepts or proposes whenever the valuation itself is positive.

    Every random draw comes from `rng`; the real game's stream is never read.
    """

    evaluator: HonestEvaluator
    depth: int = 2
    width: int | None = 6
    max_nodes: int = DEFAULT_MAX_NODES
    k: int = 1
    rng: random.Random = field(default_factory=random.Random)
    stance: str = "relative"
    max_offers: int | None = 3
    placement: bool = True
    mode: str = "honest"
    exact_roll_plies: int = EXACT_ROLL_PLIES
    # The trade adapter's own knobs (unfitted): how many of
    # `candidate_bundles`' scored proposals become root options, and the
    # margins below which a proposal is not offered or an offer not
    # accepted. See the class docstring's trade-adapter paragraph.
    propose_top_n: int = 3
    propose_margin: float = 0.0
    accept_margin: float = 0.0

    def __post_init__(self) -> None:
        if self.stance not in STANCES:
            raise ValueError(f"unknown stance: {self.stance}")
        if self.k < 1:
            raise ValueError("k must be at least one world")
        self._rank = STANCES[self.stance]
        self._spent = 0
        self._budget = self.max_nodes
        self.depth_reached = 0

    @property
    def omniscient(self) -> bool:
        """Whether this bot reads every seat's true hand (mode="omniscient")."""
        return self.evaluator.omniscient

    @property
    def nodes(self) -> int:
        """Leaf evaluations the last `choose` spent."""
        return self._spent

    # -- the decision --------------------------------------------------------

    def choose(self, game: Game) -> Action:
        """The bot's one public entry point: `Bot.choose(game) -> Action`.

        Setup, a no-trade bot's forced decline, and discard are resolved
        directly (placement's prior, `marginal_loss`, or the only option);
        everything else determinizes the belief into `k` worlds, builds the
        root's own options (`_root_options` -- engine legality plus, in
        MAIN, the trade adapter's `propose_actions`), and either returns the
        one option available or hands the rest to `_search`.
        """
        seat = to_move(game)
        self._spent = 0
        self._budget = self.max_nodes
        self.depth_reached = 0
        self.evaluator._walk_cache.clear()
        self.evaluator._belief_cache.clear()
        self.evaluator._evaluate_cache.clear()

        if game.phase is Phase.SETUP_SETTLEMENT and self.placement:
            options = options_for(game)
            chosen = best_opening(game.state, seat, [a.a for a in options])
            return Action(ActionType.SETUP_SETTLEMENT, chosen)
        if game.phase is Phase.TRADE_RESPOND and self.max_offers == 0:
            return Action(ActionType.DECLINE_TRADE)
        if game.phase is Phase.DISCARD:
            options = options_for(game)
            if len(options) == 1:
                return options[0]
            return min(options, key=lambda a: self.marginal_loss(game, seat, a.a))

        worlds = self.worlds(game, seat)
        options = self._root_options(game, worlds, seat)
        if len(options) == 1:
            return options[0]
        return self._search(worlds, options, seat)

    def worlds(self, game: Game, seat: int) -> list[Game]:
        """The determinizations this decision is searched in.

        Each is an `imagine` copy whose hidden hands and cards are one draw
        from the belief; in omniscient mode, one copy of the truth.
        """
        if self.omniscient:
            return [imagine(game, self.rng)]
        belief = Belief.from_game(game, seat)
        out = []
        for _ in range(self.k):
            world = imagine(game, self.rng, randomize_deck=False)
            world.state = belief.sample(self.rng)
            out.append(world)
        return out

    def root_options(self, game: Game) -> list[Action]:
        """What the search chooses among at `game`, after the bot's own rules."""
        seat = to_move(game)
        return self._root_options(game, self.worlds(game, seat), seat)

    def _root_options(self, game: Game, worlds: list[Game], seat: int) -> list[Action]:
        # The engine's offer sample is built from the opponents' true hands
        # (who could cover what), so the root's options are read off the
        # worlds instead, in first-seen order. Every action legal in a world
        # is legal in the truth: builds, cards and bank trades depend only on
        # the mover's hand, and an offer needs only what the proposer holds.
        seen: dict[Action, None] = {}
        for world in worlds:
            for action in self._options_in(world, seat):
                seen.setdefault(action, None)
        options = list(seen)
        if game.phase is Phase.MAIN:
            # The engine's one-for-one sample (`actions._offer_actions`,
            # already folded into `seen` above) is replaced wholesale by
            # `propose_actions`. Guarded by the same budget test
            # `within_offer_budget` applies below, so a bot with no offers
            # left never pays for the candidate search.
            options = [a for a in options if a.type is not ActionType.PROPOSE_TRADE]
            if self.max_offers is None or game.offers_made < self.max_offers:
                options.extend(self.propose_actions(game, seat))
        options = within_offer_budget(game, options, self.max_offers)
        if not options:
            options = options_for(game)

        monopolies = [a for a in options if a.type is ActionType.PLAY_MONOPOLY]
        if len(monopolies) > 1:
            belief = self.evaluator.belief_from_game(game, seat)
            target = max(
                range(NUM_RESOURCES), key=lambda r: (belief.table_holding(r), -r)
            )
            keep = Action(ActionType.PLAY_MONOPOLY, target)
            options = [a for a in options if a.type is not ActionType.PLAY_MONOPOLY or a == keep]
        return options

    def _search(self, worlds: list[Game], options: list[Action], seat: int) -> Action:
        best = options[0]
        ranked: list[tuple[float, Action]] | None = None
        costs: list[int] = []
        for depth in range(1, self.depth + 1):
            if ranked is None:
                candidates = options
            elif self.width is None:
                candidates = [a for _, a in ranked]
            else:
                candidates = [a for _, a in ranked[: self.width]]
            if depth > 1:
                estimate = self._estimate(candidates, len(options), costs)
                if self._spent + estimate > self._budget:
                    break
            before = self._spent
            partial: list[tuple[float, Action]] = []
            try:
                totals = self._root_values(worlds, candidates, depth, seat, partial)
            except _Exhausted:
                if ranked is None and partial:
                    best = max(partial, key=lambda pair: pair[0])[1]
                break
            ranked = sorted(
                ((self._rank(v, seat), a) for a, v in zip(candidates, totals)),
                key=lambda pair: -pair[0],
            )
            best = ranked[0][1]
            self.depth_reached = depth
            costs.append(self._spent - before)
        return best

    def _estimate(self, candidates: list[Action], branching: int, costs: list[int]) -> int:
        """Leaves the next ply is expected to cost, from what the last ones did."""
        if len(costs) >= 2 and costs[-2] > 0:
            return int(costs[-1] * costs[-1] / costs[-2]) + 1
        total = 0
        for action in candidates:
            if action.type is ActionType.ROLL:
                total += len(ROLL_ODDS) * branching
            elif action.type is ActionType.END_TURN:
                total += len(ROLL_ODDS)
            else:
                total += branching
        return total

    def _root_values(
        self,
        worlds: list[Game],
        candidates: list[Action],
        depth: int,
        seat: int,
        partial: list[tuple[float, Action]],
    ) -> list[list[float]]:
        share = 1.0 / len(worlds)
        totals = []
        for action in candidates:
            total = [0.0] * worlds[0].state.num_players
            for world in worlds:
                vector = self._after(world, action, depth, seat)
                for p, value in enumerate(vector):
                    total[p] += share * value
            totals.append(total)
            partial.append((self._rank(total, seat), action))
        return totals

    # -- the tree ------------------------------------------------------------

    def _leaf(self, game: Game, knower: int) -> list[float]:
        if self._spent >= self._budget:
            raise _Exhausted
        self._spent += 1
        return self.evaluator.evaluate_game(game, knower)

    def _after(
        self, game: Game, action: Action, depth: int, knower: int, ply: int = 0
    ) -> list[float]:
        """Value of the position `action` leads to, with `depth - 1` plies left."""
        if action.type is ActionType.ROLL:
            return self._over_dice(game, depth, knower, ply)
        if draws_hidden(game, action):
            total = [0.0] * game.state.num_players
            for weight, child in self.draw_children(game, action, knower):
                for p, value in enumerate(self._value(child, depth - 1, knower, ply + 1)):
                    total[p] += weight * value
            return total
        return self._value(self._plain_child(game, action), depth - 1, knower, ply + 1)

    def _over_dice(self, game: Game, depth: int, knower: int, ply: int) -> list[float]:
        # Unreached at every shipped preset (depth=2 <= exact_roll_plies=2, so
        # `ply` never gets this high) -- kept because `depth` is a public
        # field a deeper search may raise, and this is what bounds its cost.
        if ply >= self.exact_roll_plies:
            child = imagine(game, self.rng)
            roll_dice(child)
            return self._value(child, depth - 1, knower, ply + 1)
        total = [0.0] * game.state.num_players
        for roll, weight in ROLL_ODDS:
            child = imagine(game, self.rng)
            roll_dice(child, roll)
            for p, value in enumerate(self._value(child, depth - 1, knower, ply + 1)):
                total[p] += weight * value
        return total

    def draw_children(
        self, game: Game, action: Action, knower: int
    ) -> list[tuple[float, Game]]:
        """Every outcome of a hidden draw, with its probability under the belief.

        A steal is resolved to each resource the victim might hold, weighted by
        the knower's belief about that hand; a purchase to each card type,
        weighted by the unseen deck composition. The child for an outcome is
        built in the world the tree is playing in: if that world happens not to
        hold the outcome (a sampled hand without the resource, a shuffled deck
        without the card), one untyped card is swapped so that it does --
        conditioning the determinization on the outcome rather than discarding
        it. Only cards the record has not certified are ever swapped.
        """
        belief = Belief.from_game(game, knower, omniscient=self.omniscient)
        if action.type is ActionType.BUY_DEV_CARD:
            odds = belief.deck_odds()
            children = []
            for card, weight in enumerate(odds):
                if weight <= 0:
                    continue
                child = imagine(game, self.rng)
                _put_on_top(child.state.deck, card)
                apply(child, action)
                children.append((weight, child))
            return children or [(1.0, self._plain_child(game, action))]

        victim = victim_of(game, action.b)
        assert victim is not None
        odds = belief.steal_odds(victim)
        children = []
        for resource, weight in enumerate(odds):
            if weight <= 0:
                continue
            child = imagine(game, self.rng)
            hand = child.state.hands[victim]
            if hand[resource] == 0:
                donor = _donor(hand, belief.known[victim])
                if donor is None:
                    continue
                hand[donor] -= 1
                hand[resource] += 1
            child.rng = _Forced(sum(hand[:resource]))  # type: ignore[assignment]
            apply(child, action)
            child.rng = self.rng
            children.append((weight, child))
        if not children:
            return [(1.0, self._plain_child(game, action))]
        scale = 1.0 / sum(weight for weight, _ in children)
        return [(weight * scale, child) for weight, child in children]

    def _plain_child(self, game: Game, action: Action) -> Game:
        child = imagine(game, self.rng)
        apply(child, action)
        return child

    def _options_in(self, world: Game, knower: int) -> list[Action]:
        """`legal_actions` in a determinized world, minus an ACCEPT it cannot
        honour or one `accept_rule` would refuse.

        The engine offers ACCEPT to whoever it is asking because it built the
        list of responders from the true hands; a world sampled without that
        knowledge may have dealt the seat being asked a hand that cannot
        cover the offer, so that hard constraint drops ACCEPT outright. The
        soft constraint runs at every `TRADE_RESPOND` node the tree reaches,
        not only the root: the search may only offer ACCEPT_TRADE where
        `accept_rule` clears `accept_margin`, and `knower` there is always
        the search's root seat, never the responder itself when the two
        differ -- so a simulated opponent's row is read off `knower`'s own
        belief (`_partner_delta`), never that world's sampled truth, and its
        accept cannot depend on which of the `k` worlds it was sampled into.
        """
        options = legal_actions(world)
        if world.phase is Phase.TRADE_RESPOND and world.offer is not None:
            responder = to_move(world)
            if not can_accept(
                world.state, world.offer, responder
            ) or not self.accept_rule(
                world, responder, world.offer, self.accept_margin, knower=knower
            ):
                options = [a for a in options if a.type is not ActionType.ACCEPT_TRADE]
        return options

    def _value(self, game: Game, depth: int, knower: int, ply: int) -> list[float]:
        if depth <= 0 or is_over(game):
            return self._leaf(game, knower)
        options = self._options_in(game, knower)
        if not options:
            return self._leaf(game, knower)
        mover = to_move(game)

        if depth == 1 or self.width is None or len(options) <= self.width:
            return self._best_of(game, options, depth, mover, knower, ply)
        ranked = sorted(
            ((self._rank(self._after(game, a, 1, knower, ply), mover), a) for a in options),
            key=lambda pair: -pair[0],
        )
        beam = [a for _, a in ranked[: self.width]]
        return self._best_of(game, beam, depth, mover, knower, ply)

    def _best_of(
        self,
        game: Game,
        options: list[Action],
        depth: int,
        mover: int,
        knower: int,
        ply: int,
    ) -> list[float]:
        best: list[float] | None = None
        best_rank = 0.0
        for action in options:
            vector = self._after(game, action, depth, knower, ply)
            rank = self._rank(vector, mover)
            if best is None or rank > best_rank:
                best, best_rank = vector, rank
        assert best is not None
        return best

    # --- trade --------------------------------------------------------------

    def _vector(self, state: GameState, ledger: PublicLedger, knower: int) -> list[float]:
        """The per-seat vector for `state`, read through `knower`'s own belief
        (its hand exact, everyone else's `expected_hand`)."""
        belief = self.evaluator.belief_for(state, ledger, knower)
        return self.evaluator.evaluate(state, knower, belief)

    def _read_row(
        self,
        state: GameState,
        ledger: PublicLedger,
        knower: int,
        target: int,
        rank,
        *,
        vector: list[float] | None = None,
    ) -> float:
        """`target`'s row of `knower`'s vector, under `rank` -- how the
        valuation ever estimates "what would someone else do" without
        reading that someone's true hand, see `_partner_delta`. `vector`,
        when given, skips rebuilding it for a caller that already has it."""
        if vector is None:
            vector = self._vector(state, ledger, knower)
        return rank(vector, target)

    def _read(
        self,
        state: GameState,
        ledger: PublicLedger,
        seat: int,
        *,
        vector: list[float] | None = None,
    ) -> float:
        """`seat`'s own row of the vector, under the bot's configured stance
        (`_read_row` with `knower == target == seat`) -- the common case
        every valuation method that isn't comparing seats uses; the others
        call `_partner_delta` directly with an explicit `rank`."""
        return self._read_row(state, ledger, seat, seat, self._rank, vector=vector)

    def _before_vector(self, game: Game, seat: int) -> list[float]:
        """`game`'s own, unmutated vector under `seat`'s belief -- the
        `before_vector` every call in one `propose_actions` shares, see
        `_partner_delta`."""
        return self._vector(game.state, game.ledger, seat)

    def marginal_gain(
        self, game: Game, seat: int, resource: int, *, before_vector: list[float] | None = None
    ) -> float:
        """Eval(hand + one `resource`) - Eval(hand), from `seat`'s own reading."""
        before = self._read(game.state, game.ledger, seat, vector=before_vector)
        state = copy_state(game.state)
        state.hands[seat][resource] += 1
        if state.bank[resource] > 0:
            state.bank[resource] -= 1
        return self._read(state, game.ledger, seat) - before

    def marginal_loss(
        self, game: Game, seat: int, resource: int, *, before_vector: list[float] | None = None
    ) -> float:
        """Eval(hand) - Eval(hand less one `resource`); zero when none is held."""
        if game.state.hands[seat][resource] < 1:
            return 0.0
        before = self._read(game.state, game.ledger, seat, vector=before_vector)
        state = copy_state(game.state)
        state.hands[seat][resource] -= 1
        state.bank[resource] += 1
        ledger = game.ledger.copy()
        ledger.spend(seat, resource, 1)
        return before - self._read(state, ledger, seat)

    def deficit(
        self, game: Game, seat: int, *, before_vector: list[float] | None = None
    ) -> dict[int, float]:
        """Marginal gain of receiving one more of each resource, `seat`'s own
        reading. Every resource is included, even ones already held: a
        second wheat can still be worth more than nothing."""
        return {
            r: self.marginal_gain(game, seat, r, before_vector=before_vector)
            for r in range(NUM_RESOURCES)
        }

    def surplus(
        self, game: Game, seat: int, *, before_vector: list[float] | None = None
    ) -> dict[int, float]:
        """Marginal loss of giving up one of each resource `seat` actually
        holds. Resources not held are left out rather than scored zero:
        `marginal_loss` reads 0.0 for an empty resource because there is
        nothing to give, not because giving it away would cost nothing, and
        keeping it in would make an absent resource look like the cheapest
        one to part with."""
        return {
            r: self.marginal_loss(game, seat, r, before_vector=before_vector)
            for r in range(NUM_RESOURCES)
            if game.state.hands[seat][r] > 0
        }

    def _partner_delta(
        self,
        game: Game,
        knower: int,
        target: int,
        give: Sequence[int],
        want: Sequence[int],
        counterparty: int,
        rank,
        *,
        before_vector: list[float] | None = None,
    ) -> float:
        """`target`'s row of the vector if `target` gave `give` and got `want`
        from `counterparty`, read entirely through `knower`'s own belief.

        `bundle_delta(game, seat, give, want, counterparty)` is the case
        `target == knower == seat`, where the trader's own hand is
        legitimately exact; this generalisation estimates what a DIFFERENT
        seat would make of a trade without ever reading that seat's true
        hand, only `knower`'s belief about it (`score_proposal`'s `willing`,
        `rank_partners`, and the search's gate on a simulated opponent's
        `ACCEPT_TRADE` all read a row other than their own this way). The
        ledger is updated from `give`/`want` directly (`spend`/`receive`),
        never by diffing `state.hands` before and after, so a third party's
        hidden composition cannot leak through a clamp-at-zero. `state.hands`
        is mutated exactly for `knower` and folded into one total for anyone
        else -- only `knower`'s own row is ever read verbatim. `before_vector`,
        when given, is `_before_vector(game, knower)`: the "before" side
        never depends on `give`/`want`/`counterparty`, so a caller making
        several of these calls against the same unmutated game computes it
        once and passes it through.
        """
        before = self._read_row(
            game.state, game.ledger, knower, target, rank, vector=before_vector
        )
        state = copy_state(game.state)
        exact = self.omniscient
        self._move_hand(state, knower, target, gains=want, losses=give, exact=exact)
        self._move_hand(state, knower, counterparty, gains=give, losses=want, exact=exact)
        ledger = game.ledger.copy()
        for r in range(NUM_RESOURCES):
            if give[r]:
                ledger.spend(target, r, give[r])
                ledger.receive(counterparty, r, give[r])
            if want[r]:
                ledger.spend(counterparty, r, want[r])
                ledger.receive(target, r, want[r])
        return self._read_row(state, ledger, knower, target, rank) - before

    @staticmethod
    def _move_hand(
        state: GameState,
        knower: int,
        seat: int,
        *,
        gains: Sequence[int],
        losses: Sequence[int],
        exact: bool = False,
    ) -> None:
        """`seat`'s hand in `state` after gaining `gains` and losing `losses`.

        Exact, per resource, when `seat == knower` -- its own hand is read
        verbatim, so it must reflect the trade precisely. Otherwise only the
        total moves (folded into one resource slot): see `_partner_delta`.

        `exact` forces the per-resource move for every seat, and
        `_partner_delta` passes `self.omniscient`. The fold is not an
        approximation the honest bot tolerates but an exact identity for it:
        a non-knower's hand reaches the honest evaluation only through
        `Belief.expected_hand`, which reads `known`/`unknown`/`pool` off the
        ledger and the bank, and the one thing it takes from `state.hands` is
        the *size* (`Belief.sizes`) -- which the fold preserves while a
        per-resource move, clamped at zero when the seat cannot cover
        `losses`, would not. Under omniscience that reasoning is void: `known`
        *is* `state.hands`, every row is scored on the real cards, and folding
        would price an all-one-resource fiction whose `progress`, `diversity`
        and `scarce` terms are nothing like the position's. Measured on the
        seat's own row, the fold overstated how much a one-for-one trade cost
        the partner by ~10x (`test_an_omniscient_partner_read_...`), so
        `score_proposal`'s `willing` gate almost never fired and
        `accept_rule`, reading a counterparty it had just impoverished under
        `relative`, cleared far too easily.
        """
        hand = state.hands[seat]
        if seat == knower or exact:
            for r in range(len(hand)):
                hand[r] += gains[r] - losses[r]
                if hand[r] < 0:
                    hand[r] = 0
        else:
            net = sum(gains) - sum(losses)
            state.hands[seat] = [max(0, sum(hand) + net)] + [0] * (len(hand) - 1)

    def bundle_delta(
        self,
        game: Game,
        seat: int,
        give: Sequence[int],
        want: Sequence[int],
        counterparty: int,
        *,
        before_vector: list[float] | None = None,
    ) -> float:
        """How much `seat` gains by giving `give` for `want` with `counterparty`.

        Read from `seat`'s own information under the stance, so under
        `relative` it depends on who the counterparty is: the ledger
        certifies `counterparty` as having received `give` and spent `want`
        regardless of what it truly held, and what it can do with the
        result is part of the reading. `seat` is assumed to hold `give`
        (true for every candidate `candidate_bundles` emits, `can_propose`
        checked); its own hand is clamped at zero rather than driven
        negative if not. `before_vector`: see `_partner_delta`.
        """
        return self._partner_delta(
            game, seat, seat, give, want, counterparty, self._rank, before_vector=before_vector
        )

    def candidate_bundles(
        self, game: Game, seat: int, *, max_side: int = 2
    ) -> list[tuple[Bundle, Bundle]]:
        """Bundles built from `deficit` x `surplus`'s resources, 1-2 cards a side.

        The give side is drawn from resources `seat` holds (a proposer can
        only offer what it has); the want side from every resource, since
        wanting one needs no holding of `seat`'s own, only a table that
        might supply it (`score_proposal`'s `p_holds` prices that later). A
        2-for-1 of a single surplus resource is added whenever a port makes
        it cheaper than the bank, independent of `max_side`. Every returned
        pair is `well_formed` and `can_propose` against the current state,
        so every candidate is legal to emit as-is. The resource sets are
        built directly rather than through `deficit`/`surplus` (which would
        each spend an `evaluate()`-backed marginal read this method never
        uses) -- `propose_actions` calls those itself where the values are
        actually read.
        """
        state = game.state
        hand = state.hands[seat]
        deficits = list(range(NUM_RESOURCES))
        surpluses = [r for r in range(NUM_RESOURCES) if hand[r] > 0]

        give_options = _bundle_options(surpluses, max_side, cap=lambda r: hand[r])
        ratios = trade_ratios(state, seat)
        for r in surpluses:
            if ratios[r] < BANK_TRADE_RATIO and hand[r] >= 2:
                ported = _one_hot(r, 2)
                if ported not in give_options:
                    give_options.append(ported)

        want_options = _bundle_options(deficits, max_side)

        # `_bundle_options` never repeats a bundle, so the pairs are distinct
        # by construction and no seen-set is needed; `can_propose` is
        # `well_formed` and `holds`, so calling it alone is the whole test.
        return [
            (give, want)
            for give in give_options
            for want in want_options
            if can_propose(state, Offer(proposer=seat, give=give, want=want))
        ]

    def score_proposal(
        self,
        game: Game,
        seat: int,
        give: Sequence[int],
        want: Sequence[int],
        *,
        before_vector: list[float] | None = None,
    ) -> float:
        """`dEval_me(after, best counterparty) x sum_opp p_holds(opp, want) * willing(opp)`.

        `dEval_me` is `bundle_delta` maximised over which opponent ends up on
        the other side of the trade, so this asks who the trade would be
        best made *with*, not merely whether to make it at all. `willing(opp)`
        reads `opp`'s row of the vector on `opp`'s *expected* hand under
        `seat`'s own belief (`_partner_delta`, never `opp`'s true hand), the
        same evaluator applied from that seat's point of view -- hence
        `self._rank` rather than a fixed stance. It is a crisp 0/1 test
        rather than a smoothed one: there is no fitted slope to smooth it
        with yet. `before_vector`: see `_partner_delta`.
        """
        opponents = [p for p in range(game.state.num_players) if p != seat]
        if not opponents:
            return 0.0
        if before_vector is None:
            before_vector = self._before_vector(game, seat)
        delta_me = max(
            self.bundle_delta(game, seat, give, want, opp, before_vector=before_vector)
            for opp in opponents
        )
        belief = self.evaluator.belief_from_game(game, seat)
        weight = 0.0
        for opp in opponents:
            chance = belief.p_holds(opp, want)
            if chance <= 0:
                continue
            theirs = self._partner_delta(
                game, seat, opp, want, give, seat, self._rank, before_vector=before_vector
            )
            if theirs > 0:
                weight += chance
        return delta_me * weight

    def accept_rule(
        self, game: Game, seat: int, offer: Offer, margin: float, *, knower: int | None = None
    ) -> bool:
        """Accept iff taking `offer` clears `margin`.

        Taking it means `seat` gives `offer.want` and receives `offer.give`
        with `offer.proposer` as the counterparty. `margin` is a parameter
        rather than always `self.accept_margin` so the rule can be probed at
        any threshold. `knower` is whose belief the read is anchored to; it
        defaults to `seat` itself (the bot's own real decision, hand exact),
        but `_options_in` passes the search's root seat instead when this
        gates a simulated opponent deeper in the tree, so that opponent's
        hand stays read as `knower`'s `expected_hand`, never the sampled
        world's truth.
        """
        return self._partner_delta(
            game,
            seat if knower is None else knower,
            seat,
            offer.want,
            offer.give,
            offer.proposer,
            self._rank,
        ) > margin

    def counter_of(self, game: Game, seat: int, offer: Offer) -> tuple[Bundle, Bundle] | None:
        """The candidate nearest `offer.want`, among my own surplus-built bundles.

        Distance is taxicab on the `want` side: the candidate I would ask
        for that most resembles what the table has already shown interest
        in. `None` when I have no candidate at all. Today's engine has no
        counter action, so nothing calls this yet; it is unit-tested only.
        """
        candidates = self.candidate_bundles(game, seat)
        if not candidates:
            return None

        def distance(pair: tuple[Bundle, Bundle]) -> int:
            _, want = pair
            return sum(abs(a - b) for a, b in zip(want, offer.want))

        return min(candidates, key=distance)

    def rank_partners(
        self,
        game: Game,
        seat: int,
        give: Sequence[int],
        want: Sequence[int],
        *,
        before_vector: list[float] | None = None,
    ) -> tuple[int, ...]:
        """Opponents ranked for `Action.ask`, first to whoever it helps least.

        `p_holds(opp, want) * (-bundle_delta_them)` under `paranoid`
        regardless of the bot's own configured stance -- `paranoid` is the
        only stance that can tell opponents apart, which is what ordering an
        ask needs. `p_holds` uses the belief, never `state.hands[opp]`;
        "how much would it help them" is `opp`'s row read through `seat`'s
        own belief (`_partner_delta`, `opp`'s *expected* hand, never its true
        one). `before_vector`: see `_partner_delta` -- still the vector read
        under `seat`'s own belief; only the read-out differs by stance.
        """
        belief = self.evaluator.belief_from_game(game, seat)
        paranoid = STANCES["paranoid"]
        opponents = [p for p in range(game.state.num_players) if p != seat]
        if before_vector is None:
            before_vector = self._before_vector(game, seat)

        def score(opp: int) -> float:
            chance = belief.p_holds(opp, want)
            theirs = self._partner_delta(
                game, seat, opp, want, give, seat, paranoid, before_vector=before_vector
            )
            return chance * -theirs

        return tuple(sorted(opponents, key=score, reverse=True))

    # -- the adapter: everything above is protocol-free valuation; this is
    # where it meets today's protocol, at two touch points called from
    # `search` above: `_root_options` calls `propose_actions` (entry);
    # `_options_in` calls `accept_rule` (exit, a plain predicate reused as a
    # gate). A later protocol changes this method and that call site only.

    def propose_actions(self, game: Game, seat: int) -> list[Action]:
        """The adapter: the top `propose_top_n` scored proposals.

        Replaces the engine's one-for-one `PROPOSE_TRADE` sample
        (`actions._offer_actions`) among the root options -- mechanical and
        untuned, rewritten whenever the protocol changes (module docstring's
        `trade` section). Candidates and their scores are read off the true
        game directly, not per determinized world: an offer's legality
        depends only on what the proposer holds, which is always exact, and
        the valuation already reads opponents through the honest belief
        built from the real ledger. The `k` worlds still decide what a
        proposal is *worth* once it is a root option, searched like any
        other action. Candidates scoring at or below `propose_margin` are
        dropped, and `score_proposal` -- a `bundle_delta`/`_partner_delta`
        call per opponent, twice over -- is the module's single largest cost,
        so `deficit`/`surplus` (cheap: no opponent belief) shortlist the
        field to `PROPOSE_SHORTLIST` first and only that shortlist pays for
        the partner-aware score.
        """
        # Every "before" read below shares this one (game.state, game.ledger,
        # seat) triple, computed once rather than per candidate.
        before_vector = self._before_vector(game, seat)
        candidates = self.candidate_bundles(game, seat)
        if not candidates:
            return []
        deficit = self.deficit(game, seat, before_vector=before_vector)
        surplus = self.surplus(game, seat, before_vector=before_vector)

        def cheap_score(pair: tuple[Bundle, Bundle]) -> float:
            give, want = pair
            gain = sum(want[r] * deficit.get(r, 0.0) for r in range(NUM_RESOURCES))
            loss = sum(give[r] * surplus.get(r, 0.0) for r in range(NUM_RESOURCES))
            return gain - loss

        shortlist = sorted(candidates, key=cheap_score, reverse=True)[:PROPOSE_SHORTLIST]
        scored = [
            (self.score_proposal(game, seat, give, want, before_vector=before_vector), give, want)
            for give, want in shortlist
        ]
        scored = [s for s in scored if s[0] > self.propose_margin]
        scored.sort(key=lambda s: -s[0])
        return [
            Action(
                ActionType.PROPOSE_TRADE,
                give=give,
                want=want,
                ask=self.rank_partners(game, seat, give, want, before_vector=before_vector),
            )
            for _, give, want in scored[: self.propose_top_n]
        ]


def _put_on_top(deck: list[int], card: int) -> None:
    """Make `card` the next draw (`devcards.buy` pops the end of the deck)."""
    if not deck:
        return
    for index in range(len(deck) - 1, -1, -1):
        if deck[index] == card:
            deck[index], deck[-1] = deck[-1], deck[index]
            return
    deck[-1] = card


def _donor(hand: list[int], known: list[int]) -> int | None:
    """A resource the hand holds beyond what the record certifies, if any."""
    for r in range(NUM_RESOURCES):
        if hand[r] > known[r]:
            return r
    for r in range(NUM_RESOURCES):
        if hand[r] > 0:
            return r
    return None


def _one_hot(resource: int, n: int) -> Bundle:
    """`n` cards of one `resource`, as a `Bundle`."""
    out = [0] * NUM_RESOURCES
    out[resource] = n
    return tuple(out)


def _two_hot(r1: int, r2: int) -> Bundle:
    """One card each of two distinct resources, as a `Bundle`."""
    out = [0] * NUM_RESOURCES
    out[r1] += 1
    out[r2] += 1
    return tuple(out)


def _bundle_options(resources: list[int], max_side: int, *, cap=None) -> list[Bundle]:
    """One-hot bundles of each resource up to `max_side` (or `cap(r)` if
    given), plus every two-hot pair once `max_side >= 2`."""
    size = (lambda r: min(max_side, cap(r))) if cap else (lambda r: max_side)
    options = [_one_hot(r, n) for r in resources for n in range(1, size(r) + 1)]
    if max_side >= 2:
        options.extend(
            _two_hot(r1, r2) for i, r1 in enumerate(resources) for r2 in resources[i + 1 :]
        )
    return options


def heximax(
    board: Board,
    rng: random.Random | None = None,
    *,
    mode: str = "honest",
    depth: int = 2,
    width: int | None = 6,
    max_offers: int | None = BY_MODE,  # type: ignore[assignment]
    max_nodes: int = DEFAULT_MAX_NODES,
    k: int = 1,
    stance: str = "relative",
    placement: bool = True,
    exact_progress_samples: int = 0,
    weights: Weights | None = None,
    propose_top_n: int = 3,
    propose_margin: float = 0.0,
    accept_margin: float = 0.0,
) -> Heximax:
    """The three shipped configurations, by `mode`.

    `honest` reads the ledger and the trading-table weights; `omniscient`
    reads every true hand with the same weights; `notrade` is honest with the
    no-trade weights. Left at `BY_MODE`, the offer budget is three for the
    first two and zero for `notrade`; any explicit value, `None` included,
    is taken as given. `propose_top_n`, `propose_margin` and `accept_margin`
    are the trade adapter's own knobs, unfitted -- see `Heximax`'s docstring.

    `weights` overrides the mode's own profile (`TRADING_WEIGHTS` or
    `NO_TRADE_WEIGHTS`) with the given vector, leaving everything else about
    the mode -- the offer budget, `omniscient` -- unchanged. This is the hook
    `hexset.tuning` fits through: a candidate and the incumbent are otherwise
    identical heximax bots, differing only in this vector.
    """
    if mode not in MODES:
        raise ValueError(f"unknown heximax mode: {mode}")
    if max_offers is BY_MODE:
        max_offers = 0 if mode == "notrade" else 3
    if weights is None:
        weights = NO_TRADE_WEIGHTS if mode == "notrade" else TRADING_WEIGHTS
    evaluator = HonestEvaluator(
        board,
        weights,
        omniscient=(mode == "omniscient"),
        exact_progress_samples=exact_progress_samples,
    )
    return Heximax(
        evaluator,
        depth=depth,
        width=width,
        max_nodes=max_nodes,
        k=k,
        rng=rng or random.Random(),
        stance=stance,
        max_offers=max_offers,
        placement=placement,
        mode=mode,
        propose_top_n=propose_top_n,
        propose_margin=propose_margin,
        accept_margin=accept_margin,
    )


__all__ = [
    "BY_MODE",
    "Belief",
    "DEFAULT_MAX_NODES",
    "Heximax",
    "HonestEvaluator",
    "MODES",
    "NO_TRADE_WEIGHTS",
    "TRADING_WEIGHTS",
    "heximax",
]
