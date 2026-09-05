# SPDX-License-Identifier: GPL-3.0-only
"""The information set: what is certified, what is hidden, and the residual
pool the hidden cards are drawn from.

`View` is the engine's per-seat information set -- what one seat can know
about every hand and development cards. It moved here from
`hexset.bots.heximax.belief.Belief` (P0 of the trading-design registration,
`agents/reference/trading-design.md`): a seat's view of the game is engine
functionality, not something a bot should have to build for itself. Reached
through `Game.state(seat, hidden=True)` (`game.py`); `hidden=False` returns
the true `GameState` instead, and is the only sanctioned way to read it from
outside the engine.

It is also what the trade mechanic hands a seat: `Bot.gains_many(view,
received, counterparties)` (`hexset.bots.search2.Bot`) receives nothing
else, so a private gate is a function of the information set by
construction. The `ledger` a view was built from rides along for exactly
that reason -- pricing a hypothetical exchange means re-reading the position
with the transfer certified, and the certification is a ledger operation.

Every opponent quantity `HonestEvaluator` (`bots/heximax/evaluate.py`)
reads comes through a `View` -- never through `state.hands[opponent]` or
`state.dev_cards[opponent]` directly -- except in `omniscient` mode, which
keeps the old true-hand reading so the price of honesty can be measured
rather than assumed. See `View`'s own docstring for the model
(`known`/`unknown`/the shared residual `pool`).
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from .board.terrain import NUM_RESOURCES
from .cards import DECK_COMPOSITION, NUM_DEV_CARDS, DevCard
from .game import Game
from .ledger import PublicLedger
from .state import BANK_PER_RESOURCE, GameState, copy_state


class View:
    """What one seat can know about every hand, and how to draw from it.

    Per seat: `known[s]` is the certified lower bound on each resource and
    `unknown[s]` the number of cards the record cannot type. The perspective's
    own seat is exact (`known` is the hand, `unknown` is zero), and so is every
    seat when `omniscient`. Everything hidden is drawn from one shared
    **residual pool**: per resource, the cards that are neither in the bank
    nor certified in any seat's `known`, sized from the bank's initial count
    rather than the true hands, which the belief may not read. `certify`
    adds lower bounds the ledger does not carry, for a caller that knows a
    seat holds something. Robustness over purity: a test fixture that
    writes `state.hands` behind the ledger's back can leave `known` summing
    past the public hand size, or the pool short; the belief clamps `known`
    to size and pads the pool proportionally rather than raise, because a
    baseline that cannot cope with a position is not a baseline.
    """

    def __init__(
        self, state: GameState, ledger: PublicLedger, perspective: int, *,
        omniscient: bool = False, certify: Sequence[tuple[int, Sequence[int]]] = (),
    ) -> None:
        self.state = state
        self.ledger = ledger
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
        self._signature: tuple | None = None

    @classmethod
    def from_game(cls, game: Game, perspective: int, *, omniscient: bool = False) -> View:
        return cls(game._state, game.ledger, perspective, omniscient=omniscient)

    def signature(self) -> tuple:
        """`known`/`unknown`/`pool` as one hashable tuple: everything
        `expected_hand` reads, so the whole of what an evaluation keyed on a
        belief depends on. Built once -- the three are set in `__init__` and
        never mutated -- because `evaluate`'s memo rebuilt them per call."""
        if self._signature is None:
            self._signature = (
                tuple(tuple(known) for known in self.known),
                tuple(self.unknown),
                tuple(self.pool),
            )
        return self._signature

    def __eq__(self, other: object) -> bool:
        """Same information set, not the same object.

        Compares `perspective`/`omniscient`/`num_players` and `signature()`
        -- everything a caller of `expected_hand`/`table_holding`/`steal_odds`/
        `p_holds` can see -- the same fields `HonestEvaluator.belief_for`
        already treats as the whole of what a `View` is a pure function of.
        Deliberately not `self.state`: two `View`s built from independently
        replayed-but-identical games hold different `GameState` objects with
        no `__eq__` of their own, and would otherwise never compare equal --
        which is exactly what broke `gymnasium.utils.env_checker.check_env`'s
        `check_step_determinism` for `hexset.gym.HexSetEnv`, whose `info`
        carries a `View` (`docs/gym-design.md` §3).
        """
        if not isinstance(other, View):
            return NotImplemented
        return (
            self.perspective == other.perspective
            and self.omniscient == other.omniscient
            and self.num_players == other.num_players
            and self.sizes == other.sizes
            and self.signature() == other.signature()
        )

    def __hash__(self) -> int:
        return hash((self.perspective, self.omniscient, self.num_players, tuple(self.sizes), self.signature()))

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
        self, seat: int, bundle: Sequence[int], *, draws: int = 64,
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


__all__ = ["View"]


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
