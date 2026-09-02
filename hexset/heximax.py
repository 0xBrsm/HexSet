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
* ``trade``    -- the valuation layer only: marginal values and bundle deltas
  with a named counterparty. The proposal generator and the protocol adapter
  are P2 and are not here.

Trading in P1 is the engine's one-for-one sample from `legal_actions`,
valued by the search like any other action; the tree's own responses are
searched from the responder's seat. `max_offers=0` never proposes and always
declines.

Cost: leaf evaluations per move are capped by `max_nodes`
(`DEFAULT_MAX_NODES`, 600). Measured 2026-09-01 on the dev container, three
four-seat games a side on the same boards, every seat the same bot:
`search2` 2.08 ms/move at 37.4 leaves/move (p99 253, max 647); `heximax`
at the default budget 3.15 ms/move at 41.4 leaves/move (p99 439, max 600),
and 3.83 ms/move at 50.1 leaves/move (max 1541) with the budget lifted. The
extra leaves are the hidden draws, which expand to every outcome instead of
one; the extra time per leaf is the belief built at each of them. 1.5x
`search2` per move, against the design's ceiling of 2x.
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
from .economy import COSTS, Purchase
from .evaluate import (
    PROGRESS_PURCHASES,
    ROLLS,
    WIN_SCORE,
    Evaluator,
    Weights,
)
from .game import ROLL_ODDS, Game, Phase, imagine, is_over, roll_dice, to_move
from .ledger import PublicLedger
from .mcts import draws_hidden
from .placement import best as best_opening
from .robber import DISCARD_THRESHOLD
from .trading import can_accept
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

    def progress_toward(
        self, state: GameState, seat: int, hand: Sequence[float], purchase: Purchase
    ) -> float:
        if purchase is Purchase.SETTLEMENT or purchase is Purchase.CITY:
            settlements, cities = _pieces(state, seat)
            if purchase is Purchase.SETTLEMENT and settlements >= MAX_SETTLEMENTS:
                return 0.0
            if purchase is Purchase.CITY and cities >= MAX_CITIES:
                return 0.0
        cost = COSTS[purchase]
        return sum(min(hand[r], n) for r, n in enumerate(cost) if n) / sum(cost)

    def progress(self, state: GameState, seat: int, hand: Sequence[float]) -> float:
        return max(
            self.progress_toward(state, seat, hand, purchase)
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
        walk = self.inner.survey(state, seat)
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
        return self.evaluator.omniscient

    @property
    def nodes(self) -> int:
        """Leaf evaluations the last `choose` spent."""
        return self._spent

    # -- the decision --------------------------------------------------------

    def choose(self, game: Game) -> Action:
        seat = to_move(game)
        self._spent = 0
        self._budget = self.max_nodes
        self.depth_reached = 0

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
            for action in self._options_in(world):
                seen.setdefault(action, None)
        options = within_offer_budget(game, list(seen), self.max_offers)
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
        child = imagine(game, self.rng)
        apply(child, action)
        return self._value(child, depth - 1, knower, ply + 1)

    def _over_dice(self, game: Game, depth: int, knower: int, ply: int) -> list[float]:
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

    @staticmethod
    def _options_in(world: Game) -> list[Action]:
        """`legal_actions` in a determinized world, minus an ACCEPT it cannot honour.

        The engine offers ACCEPT to whoever it is asking because it built the
        list of responders from the true hands. A world sampled without that
        knowledge may have dealt the seat being asked a hand that does not
        cover the offer; there the only response is to decline, and
        `execute_trade` is never reached with a hand that cannot pay.
        """
        options = legal_actions(world)
        if (
            world.phase is Phase.TRADE_RESPOND
            and world.offer is not None
            and not can_accept(world.state, world.offer, to_move(world))
        ):
            options = [a for a in options if a.type is not ActionType.ACCEPT_TRADE]
        return options

    def _value(self, game: Game, depth: int, knower: int, ply: int) -> list[float]:
        if depth <= 0 or is_over(game):
            return self._leaf(game, knower)
        options = self._options_in(game)
        if not options:
            return self._leaf(game, knower)
        mover = to_move(game)

        if depth == 1 or self.width is None or len(options) <= self.width:
            return self._best_of(game, options, depth, mover, knower, ply)
        ranked = sorted(
            (
                (self._rank(self._after(game, a, 1, knower, ply), mover), a)
                for a in options
            ),
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

    def _read(self, state: GameState, ledger: PublicLedger, seat: int) -> float:
        belief = Belief(state, ledger, seat, omniscient=self.omniscient)
        return self._rank(self.evaluator.evaluate(state, seat, belief), seat)

    def marginal_gain(self, game: Game, seat: int, resource: int) -> float:
        """Eval(hand + one `resource`) - Eval(hand), from `seat`'s own reading."""
        before = self._read(game.state, game.ledger, seat)
        state = copy_state(game.state)
        state.hands[seat][resource] += 1
        if state.bank[resource] > 0:
            state.bank[resource] -= 1
        return self._read(state, game.ledger, seat) - before

    def marginal_loss(self, game: Game, seat: int, resource: int) -> float:
        """Eval(hand) - Eval(hand less one `resource`); zero when none is held."""
        if game.state.hands[seat][resource] < 1:
            return 0.0
        before = self._read(game.state, game.ledger, seat)
        state = copy_state(game.state)
        state.hands[seat][resource] -= 1
        state.bank[resource] += 1
        ledger = game.ledger.copy()
        ledger.spend(seat, resource, 1)
        return before - self._read(state, ledger, seat)

    def bundle_delta(
        self,
        game: Game,
        seat: int,
        give: Sequence[int],
        want: Sequence[int],
        counterparty: int,
    ) -> float:
        """How much `seat` gains by giving `give` for `want` with `counterparty`.

        Read from `seat`'s own information under the stance, so under
        `relative` it depends on who the counterparty is: their hand changes
        too, and what they can do with it is part of the reading. The
        counterparty is assumed able to cover `want`; if the state says
        otherwise its hand is clamped at zero rather than driven negative.
        """
        before = self._read(game.state, game.ledger, seat)
        state = copy_state(game.state)
        old = [hand[:] for hand in state.hands]
        mine, theirs = state.hands[seat], state.hands[counterparty]
        for r in range(NUM_RESOURCES):
            mine[r] += want[r] - give[r]
            theirs[r] += give[r] - want[r]
            if theirs[r] < 0:
                theirs[r] = 0
            if mine[r] < 0:
                mine[r] = 0
        ledger = game.ledger.copy()
        ledger.apply_hand_diff(old, state.hands)
        return self._read(state, ledger, seat) - before


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
) -> Heximax:
    """The three shipped configurations, by `mode`.

    `honest` reads the ledger and the trading-table weights; `omniscient`
    reads every true hand with the same weights; `notrade` is honest with the
    no-trade weights. Left at `BY_MODE`, the offer budget is three for the
    first two and zero for `notrade`; any explicit value, `None` included,
    is taken as given.

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
