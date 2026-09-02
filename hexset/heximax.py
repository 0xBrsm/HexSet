# SPDX-License-Identifier: GPL-3.0-only
"""heximax: the handcrafted baseline that does not read the opponents' hands.

`bots.SearchBot` over `evaluate.Evaluator` -- `search2` -- is the project's
one clean held-out referent, and it cheats: its evaluation reads every seat's
true hand, its tree expands opponents from their true hands and development
cards, and a steal or a dev-card buy is valued on one frozen draw. heximax
is the next generation of that bot, built to the design in
`agents/reference/heximax.md` (P1 of §8). It is **information-set honest by
default**: every quantity about an opponent is read through the public
ledger (`game.ledger`, `known[5]` + `unknown`) and the public counts, never
through `state.hands[opponent]` or `state.dev_cards[opponent]`. Its own hand
is exact. An `omniscient` mode keeps the old reading, so the price of honesty
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
  `counter_of`, `rank_partners`), then P2's thin protocol adapter over it
  (`Heximax.propose_actions`, and `_options_in`'s `accept_rule` gate) --
  mechanical, untuned, and expected to be rewritten whenever the trading
  protocol changes; only the valuation is fitted or tested for strength.

P1 played the engine's one-for-one `PROPOSE_TRADE` sample from
`legal_actions`, valued by the search like any other action. P2 replaces
that sample with heximax's own top-`propose_top_n` bundle proposals while
`max_offers` still has room, and gates the tree's own `ACCEPT_TRADE` with
`accept_rule` wherever a `TRADE_RESPOND` node is reached, root or not; the
tree's own responses are still searched from the responder's seat exactly as
in P1. `max_offers=0` never proposes and always declines, unchanged.

Cost: leaf evaluations per move are capped by `max_nodes`
(`DEFAULT_MAX_NODES`, 600). Measured 2026-09-02, after an optimization pass
that memoized `Evaluator.survey` per decision, walked `_pieces` once per
`progress()`, computed `propose_actions`' "before" reading once and reused it,
and stopped `candidate_bundles` paying for `deficit`/`surplus` values it
never read (see `agents/reference/heximax.md`, its own CHANGELOG entry, and
this range's commits for the per-change breakdown) -- same methodology as
before the pass (three four-seat games a side, same boards, board seeds
0/1/2, `search2-offers3`'s `max_offers=3` matching heximax's own budget):
`heximax` 5.14 ms/move vs `search2-offers3` 1.79 ms/move, **2.87x** -- down
from 7.10 ms/move and 3.91x measured the same way before the pass, but still
over the design's 2x ceiling. `bot.choose()`'s own choices, and the number of
leaves it spends getting to them, are unchanged by the pass on every position
checked (`test_choices_are_byte_identical_to_the_recorded_census`); every bit
of the saving is the same leaves and the same decisions, computed with less
redundant work. The remainder is not a compute problem: `score_proposal`'s
crisp `willing` gate, read under `relative`, proposes far more selectively
than the engine's naive one-for-one sample, so a real game trades roughly a
third as often (the 20-game census,
`test_multi_card_and_one_for_one_proposals_both_occur_over_twenty_games`) --
fewer cheap negotiation actions average against the leaf-budgeted build
decisions that dominate the rest, confirmed by disabling `propose_actions`'s
scoring entirely and re-measuring (still ~1.45x over the P1-only figure from
the trade-volume drop alone). Whether the gate is too strict, `relative` is
the wrong stance for `willing`, or the ceiling needs a protocol-P0 allowance
is a P3 question, not one this adapter should answer by loosening the gate to
hit a number.
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
from .trading import Bundle, Offer, can_accept, can_propose, well_formed
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


# --- belief -------------------------------------------------------------------


class Belief:
    """What one seat can know about every hand, and how to draw from it.

    Per seat: `known[s]` is the certified lower bound on each resource and
    `unknown[s]` the number of cards the record cannot type. The perspective's
    own seat is exact (`known` is the hand, `unknown` is zero), and so is every
    seat when `omniscient`. Everything hidden is drawn from one shared
    **residual pool**: per resource, the cards that are neither in the bank,
    nor certified in any seat's `known`. The pool is derived from the bank's
    initial size (`state.BANK_PER_RESOURCE`, nineteen a resource in the base
    game) rather than from the true hands, which the belief may not read.

    An open offer certifies one thing the ledger does not record: the
    proposer holds what it offers. Who else can cover it is deliberately not
    read, see `from_game`.

    Robustness over purity: a test fixture that writes `state.hands` behind
    the ledger's back can leave `known` summing past the public hand size, or
    the pool short of the hidden cards. The belief clamps `known` to the size
    and pads the pool proportionally rather than raise, because a baseline
    that cannot cope with a position is not a baseline.
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
        # Only the proposer's side of a standing offer is certified: the offer
        # is announced and `can_propose` requires holding it. Who else could
        # cover it is NOT read -- `game.pending_responders` is the engine's
        # true eligibility list, and under the rules a decline reveals
        # nothing, so from the responder's seat the other pending seats'
        # coverage is hidden information. A sampled world may therefore hand
        # a later responder a hand that cannot cover `want`; the search
        # guards `ACCEPT_TRADE` with `can_accept` in that world instead.
        certify: list[tuple[int, Sequence[int]]] = []
        if game.offer is not None:
            certify.append((game.offer.proposer, game.offer.give))
        return cls(
            game.state, game.ledger, perspective, omniscient=omniscient, certify=certify
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
# the one term that was adopted untuned and won anyway. Refit at P3 (§7).
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


class HonestEvaluator:
    """`evaluate.Evaluator`'s model, read through a `Belief`.

    Board terms are the existing evaluator's own `survey` -- it reads only
    public state, so it is reused rather than copied. The three hand terms
    (`progress`, `held`, `surplus_card`) are read on the true hand for the
    knower and on `Belief.expected_hand` for everyone else; when `omniscient`,
    on the true hand for everyone. Victory-point cards count only for the
    knower, as before.

    `progress` on an expected hand is an approximation: it is a maximum of
    minimums, so the value on the mean differs from the mean of the values.
    `exact_progress_samples > 0` replaces it with an average over that many
    hands sampled from the belief, for the readout.

    Supply-aware progress: a hand cannot be on its way to a sixth settlement
    or a fifth city. Progress toward a purchase whose piece supply is
    exhausted is zero, so the maximum falls back to what can still be built.
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
        self._survey_cache: dict[tuple, Survey] = {}

    def survey(self, state: GameState, seat: int) -> Survey:
        """`self.inner.survey`, memoized for the life of one `Heximax.choose()`.

        `Evaluator.survey` reads only `vertex_owner`, `vertex_building` and
        `robber` (see its own docstring), so the same key always yields the
        same value -- caching changes nothing about what `terms` reads, only
        how often it is recomputed. Within one decision, 92.4% of `survey`
        calls are exact repeats of an already-seen key (the k sampled worlds
        share the root's board occupancy, and many tree nodes never move a
        vertex or the robber), so this turns most of that into a dict lookup.
        `Heximax.choose` clears the cache at the top of every call, so it
        never grows across decisions.
        """
        key = (tuple(state.vertex_owner), tuple(state.vertex_building), state.robber, seat)
        cached = self._survey_cache.get(key)
        if cached is None:
            cached = self.inner.survey(state, seat)
            self._survey_cache[key] = cached
        return cached

    def progress_toward(
        self,
        state: GameState,
        seat: int,
        hand: Sequence[float],
        purchase: Purchase,
        pieces: tuple[int, int] | None = None,
    ) -> float:
        if purchase is Purchase.SETTLEMENT or purchase is Purchase.CITY:
            settlements, cities = pieces if pieces is not None else _pieces(state, seat)
            if purchase is Purchase.SETTLEMENT and settlements >= MAX_SETTLEMENTS:
                return 0.0
            if purchase is Purchase.CITY and cities >= MAX_CITIES:
                return 0.0
        cost = COSTS[purchase]
        return sum(min(hand[r], n) for r, n in enumerate(cost) if n) / sum(cost)

    def progress(self, state: GameState, seat: int, hand: Sequence[float]) -> float:
        # `_pieces` walks every vertex; SETTLEMENT and CITY in
        # `PROGRESS_PURCHASES` both need it, so it is walked once here and
        # passed down rather than twice inside `progress_toward`.
        pieces = _pieces(state, seat)
        return max(
            self.progress_toward(state, seat, hand, purchase, pieces)
            for purchase in PROGRESS_PURCHASES
        )

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
            sum(1 for owner in state.edge_owner if owner == seat),
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
        """
        if belief is None and knower is not None and not self.omniscient:
            belief = Belief(
                state, PublicLedger.new(state.num_players), knower, omniscient=False
            )
        out = []
        for seat in range(state.num_players):
            if self.omniscient or seat == knower or belief is None:
                hand: Sequence[float] = state.hands[seat]
            else:
                hand = belief.expected_hand(seat)
            out.append(self.score(state, seat, hand, knower=knower, belief=belief))
        return out

    def evaluate_game(self, game: Game, seat: int) -> list[float]:
        """`evaluate`, building the belief from `game`'s own ledger. The leaf call."""
        belief = Belief.from_game(game, seat, omniscient=self.omniscient)
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
    what is left, and a ply that overruns the budget is abandoned for the last
    completed one. Whatever the branching, no move costs more than
    `max_nodes` leaves.

    Opponents are expanded from `k` determinized worlds drawn from the
    belief at the root (`Belief.sample`) and the root values averaged across
    them -- perfect-information Monte Carlo. The world is what the tree plays
    in; the leaf is read from the knower's own information at that node,
    through the node's own ledger, which the hypothetical play has kept
    up to date. In `omniscient` mode `k` is ignored and the true state is
    searched.

    Hidden draws are expectations, not one sample: a steal is averaged over
    the victim's expected composition, a dev-card buy over the unseen deck
    composition, each outcome weighted by its probability. Rolls are exact
    eleven-way within `EXACT_ROLL_PLIES` of the root and sampled beyond.

    Opening settlements come from `placement.best` when `placement` is set;
    opening roads are searched. A discard gives up the card with the smallest
    marginal loss; a monopoly names the resource the table is expected to hold
    most of. `max_offers` is the bot's own budget below the engine's, exactly
    as in `SearchBot`; at zero it never proposes and always declines.

    P2 (the protocol-P0 adapter, `# --- trade` section): while `max_offers`
    still has room this turn, `PROPOSE_TRADE` root options are the top
    `propose_top_n` candidates from `candidate_bundles`, ranked by
    `score_proposal` and cut off at `propose_margin` -- see
    `propose_actions`. A `TRADE_RESPOND` node may only offer `ACCEPT_TRADE`
    to the search when `accept_rule` clears `accept_margin` there too --
    see `_options_in`. Both margins are unfitted (P3 refits them alongside
    the weight profiles); `0.0` accepts or proposes whenever the valuation
    itself is positive.

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
    # The protocol-P0 adapter's own knobs (unfitted, P3): how many of
    # `candidate_bundles`' scored proposals become root options, and the
    # margins below which a proposal is not offered or an offer not
    # accepted. See the class docstring's P2 paragraph.
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
        MAIN, the P2 adapter's `propose_actions`), and either returns the
        one option available or hands the rest to `_search`.
        """
        seat = to_move(game)
        self._spent = 0
        self._budget = self.max_nodes
        self.depth_reached = 0
        self.evaluator._survey_cache.clear()

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
            # `propose_actions` -- see `Heximax`'s P2 docstring paragraph.
            # Guarded by the same budget test `within_offer_budget` applies
            # below, so a bot with no offers left never pays for the
            # candidate search.
            options = [a for a in options if a.type is not ActionType.PROPOSE_TRADE]
            if self.max_offers is None or game.offers_made < self.max_offers:
                options.extend(self.propose_actions(game, seat))
        options = within_offer_budget(game, options, self.max_offers)
        if not options:
            options = options_for(game)

        monopolies = [a for a in options if a.type is ActionType.PLAY_MONOPOLY]
        if len(monopolies) > 1:
            belief = Belief.from_game(game, seat, omniscient=self.omniscient)
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
        # Unreached at every shipped preset (depth=2 <= exact_roll_plies=2,
        # so `ply` never gets this high) -- kept because `depth` is a public
        # field the design anticipates raising ("rolls stay exact eleven-way
        # at depth <= 2 and sampled beyond", heximax.md §3(b)); this is what
        # bounds a deeper search's cost when it is.
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
        list of responders from the true hands. A world sampled without that
        knowledge may have dealt the seat being asked a hand that does not
        cover the offer; there the only response is to decline, and
        `execute_trade` is never reached with a hand that cannot pay. Beyond
        that hard constraint, P2 gates the soft one: the search may only
        offer ACCEPT_TRADE where `accept_rule` clears `accept_margin`, so a
        table of heximax-like responders is searched playing the rule rather
        than whatever a bare max^n over the evaluation would pick at the
        margin. This runs at every `TRADE_RESPOND` node the tree reaches, not
        only the root, so a simulated opponent's accept is gated the same
        way a real one would be -- `knower` is always the SEARCH's root
        seat (threaded down from `choose`), never the responder itself when
        the two differ, so `accept_rule` reads that responder's row off
        `knower`'s own belief (`_partner_delta`) rather than the world's
        sampled truth. Without that, a simulated opponent's accept would
        depend on which of the `k` worlds it happened to be sampled into,
        exactly the leak `test_heximax_cannot_tell_ledger_consistent_worlds_apart`
        guards against.
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
        belief = Belief(state, ledger, knower, omniscient=self.omniscient)
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

        `bundle_delta(game, seat, give, want, counterparty)` is exactly
        `_partner_delta(game, seat, seat, give, want, counterparty, self._rank)`
        -- the case where the reader and the trader are the same seat, so
        the trader's own hand is legitimately exact. This generalisation is
        for the opposite case: estimating what a DIFFERENT seat (`target`,
        possibly also different from `counterparty`) would make of a trade,
        without ever looking at `target`'s true hand -- only `knower`'s
        belief about it. `score_proposal`'s `willing`, `rank_partners`, and
        the search's gate on a simulated opponent's `ACCEPT_TRADE`
        (`_options_in`) all read a row other than their own this way; it is
        the literal "same evaluator ... on its expected hand" the design
        asks for, and it is what keeps those three honest under
        `test_heximax_cannot_tell_ledger_consistent_worlds_apart`: `target`'s
        hand in the vector is always `knower`'s `expected_hand(target)`
        unless `target == knower`, never `target`'s own true composition.

        The ledger is updated from `give`/`want` directly -- `spend`/
        `receive` at the stated amounts -- never by diffing `state.hands`
        before and after. A diff would read the *true* prior hand through
        however much a clamp-at-zero had to bite (a real short holding
        certifies a smaller spend than the offer's face value), and that
        true amount is exactly what `target`'s hidden composition must not
        leak through; two ledger-identical positions that disagree only on
        a third party's hidden cards must produce the same certified diff
        regardless.

        `state.hands` is mutated for both `target` and `counterparty` --
        every party's hand SIZE has to stay right, or `Belief`'s own
        "desynced fixture" repair (built for a test that pokes `state.hands`
        behind the ledger's back) reads the trade's ledger credit as an
        overclaim and silently sheds it back off. But only `knower`'s own
        row is ever read verbatim (`Belief` treats exactly one seat's hand
        as exact: the perspective's own); everyone else's composition comes
        from the ledger, never from `state.hands`, so a seat that is not
        `knower` only needs its TOTAL to move by the right amount -- which
        entries "really" changed is exactly the kind of identity-dependent
        choice `ledger.steal` refuses to make, for the same reason -- while
        `knower`'s own hand, when it is `target` or `counterparty`, is
        mutated exactly, per resource, clamped at zero.

        `before_vector`, when given, is `_before_vector(game, knower)` --
        the "before" side never depends on `give`/`want`/`counterparty`, only
        on `(game.state, game.ledger, knower)`, so a caller making several of
        these calls against the same unmutated game (`bundle_delta`'s search
        over counterparties, `score_proposal`'s `willing`, `rank_partners`'
        `score`) computes it once and passes it to every one of them.
        """
        before = self._read_row(
            game.state, game.ledger, knower, target, rank, vector=before_vector
        )
        state = copy_state(game.state)
        self._move_hand(state, knower, target, gains=want, losses=give)
        self._move_hand(state, knower, counterparty, gains=give, losses=want)
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
    ) -> None:
        """`seat`'s hand in `state` after gaining `gains` and losing `losses`.

        Exact, per resource, when `seat == knower` -- its own hand is read
        verbatim, so it must reflect the trade precisely. Otherwise only the
        total moves (folded into one resource slot): see `_partner_delta`.
        """
        hand = state.hands[seat]
        if seat == knower:
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

        The give side is drawn from resources `seat` holds -- one resource
        at sizes 1..`max_side`, or two distinct resources one each when
        `max_side >= 2` -- since a proposer can only offer what it has. The
        want side is drawn from every resource the same way; wanting a
        resource needs no holding of `seat`'s own, only a table that might
        supply it, which `score_proposal`'s `p_holds` prices later. A
        2-for-1 of a single surplus resource is added whenever
        `economy.trade_ratios` gives `seat` a port under the bank's rate,
        independent of `max_side`: the port makes two of that resource
        worth less to `seat` than the bank would ever charge for it, so it
        is worth asking about even when the general sweep is capped at one
        card a side. Every returned pair is `well_formed` and `can_propose`
        against the current state -- no resource on both sides, never more
        than `seat` holds -- so every candidate is legal to emit as-is.

        This sweep only needs *which* resources are in play, not their
        marginal values -- `deficit()` has no filter (every resource is a
        want candidate) and `surplus()`'s only filter is `hand[r] > 0` (every
        held resource is a give candidate) -- so the two resource sets are
        built directly rather than through `deficit`/`surplus` themselves,
        which would spend a `marginal_gain`/`marginal_loss` evaluation (two
        `evaluate()` calls each) computing values this method never reads.
        `propose_actions` calls `deficit`/`surplus` itself where the values
        are actually used (its cheap pre-filter).
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

        seen: set[tuple[Bundle, Bundle]] = set()
        out: list[tuple[Bundle, Bundle]] = []
        for give in give_options:
            for want in want_options:
                key = (give, want)
                if key in seen:
                    continue
                offer = Offer(proposer=seat, give=give, want=want)
                if not well_formed(offer) or not can_propose(state, offer):
                    continue
                seen.add(key)
                out.append(key)
        return out

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

        `dEval_me` is `bundle_delta` maximised over which opponent ends up
        on the other side of the trade: partner-dependence is already in
        `bundle_delta` under `relative` (denying a leader is worth more than
        denying a trailer), so this asks who the trade would be best made
        *with*, not merely whether to make it at all. `willing(opp)` reads
        `opp`'s row of the vector on `opp`'s *expected* hand under `seat`'s
        own belief (`_partner_delta`, never `opp`'s true hand -- `seat` is
        the one deciding, and cannot see it) -- the design's "same evaluator
        applied from that seat's point of view", read the way the mover
        itself would read the table ("the table is modelled as thinking the
        way the bot does"), hence `self._rank` rather than a fixed stance.
        It is the crisp 0/1 test the design allows rather than a smoothed
        one: there is no fitted slope to smooth it with before P3, and a
        crisp gate is the simpler thing that is not obviously wrong.
        `before_vector`: see `_partner_delta` -- the same one `propose_actions`
        computes once and passes to every shortlisted candidate's call here.
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
        belief = Belief.from_game(game, seat, omniscient=self.omniscient)
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
        with `offer.proposer` as the counterparty -- the mirror image of how
        the proposer reads its own offer. Leader-denial is already in the
        number under `relative`. `margin` is a parameter rather than always
        `self.accept_margin` so the rule can be probed at any threshold.

        `knower` is whose belief the read is honestly anchored to; it
        defaults to `seat` itself, the bot's own real decision (its own
        hand exact, legitimately). `_options_in` passes the SEARCH's root
        seat instead when this gates a simulated opponent's `ACCEPT_TRADE`
        deeper in the tree, so `seat`'s hand there stays read as `knower`'s
        `expected_hand(seat)` rather than that world's sampled truth --
        `_partner_delta` is what makes the two calls the same function.
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
        in, among bundles `candidate_bundles` can already offer from my
        surplus. `None` when I have no candidate at all. This is the
        trading-design part 3 counter-offer primitive (today's engine has
        no counter action, so nothing calls it yet); it is unit-tested only.
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
        only stance that can tell opponents apart (own less the *best*
        opponent), which is exactly what ordering an ask needs. Uses the
        belief for `p_holds`, never `state.hands[opp]`, for the "could they
        even take it" question; "how much would it help them" is `opp`'s
        row read through `seat`'s own belief (`_partner_delta`, `opp`'s
        *expected* hand, never its true one) -- the same reasoning
        `score_proposal`'s `willing` uses, at the `paranoid` stance the
        design specifies for ranking partners rather than `willing`'s
        `self._rank`. `before_vector`: see `_partner_delta` -- note it is
        still the vector read under `seat`'s own belief, not `paranoid`'s;
        only the read-out at the end differs by stance.
        """
        belief = Belief.from_game(game, seat, omniscient=self.omniscient)
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

    # -- the adapter: everything above is protocol-free valuation; this
    # method is where it meets today's protocol. Two touch points, both
    # called from `search` above: `_root_options` calls `propose_actions`
    # (entry -- scores shaped as today's `Action`); `_options_in` calls
    # `accept_rule` (exit -- a plain valuation predicate reused as a gate).
    # A later protocol changes this method and that one call site; nothing
    # above needs to know.

    def propose_actions(self, game: Game, seat: int) -> list[Action]:
        """The protocol-P0 adapter: the top `propose_top_n` scored proposals.

        Replaces the engine's one-for-one `PROPOSE_TRADE` sample
        (`actions._offer_actions`) among the root options -- mechanical and
        untuned, expected to be rewritten when the protocol changes (see the
        module docstring's `trade` section and the class docstring's P2
        paragraph). Candidates and their scores are read off the true game
        directly, not per determinized world: an offer's legality depends
        only on what the proposer holds, which is always exact, and the
        valuation already reads opponents through the honest belief built
        from the real ledger, so a sampled world adds nothing here. The `k`
        worlds still decide what a proposal is *worth* once it is a root
        option -- each is searched separately by the same max^n as every
        other action. Candidates scoring at or below `propose_margin` are
        dropped: the adapter's own budget-spending rule (the exhaust-fraction
        lesson, `heximax.md` "Improvements inherited"), not the valuation's.

        `score_proposal` is partner-aware -- a `bundle_delta`/`_partner_delta`
        call per opponent, twice over (`delta_me`'s best-counterparty search
        and `willing`'s per-opponent read) -- so scoring every candidate
        `candidate_bundles` returns is the module's single largest cost
        (measured: adapter cost dominates the leaf-budgeted search itself).
        `deficit`/`surplus` are cheap (no opponent belief, one `before_vector`
        shared with every other read this call makes -- see below) and rank
        candidates almost as well on their own, so they shortlist the field
        to `PROPOSE_SHORTLIST` first; only that shortlist pays for the
        partner-aware score. This is the adapter's own cost-control, same
        standing as `propose_margin` -- untuned, and it can only ever drop a
        candidate `score_proposal` would have ranked outside the shortlist
        anyway among ones already inside it.
        """
        # Every "before" read below -- deficit, surplus, and every candidate's
        # score_proposal/rank_partners -- is against this same unmutated
        # (game.state, game.ledger, seat) triple; computed once and threaded
        # through as before_vector rather than each rebuilding the belief and
        # re-running evaluate() for a position that never changes here.
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
    are the P2 adapter's own knobs, unfitted -- see `Heximax`'s docstring.

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
