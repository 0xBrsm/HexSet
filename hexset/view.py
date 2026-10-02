# SPDX-License-Identifier: GPL-3.0-only
"""The engine's per-seat information set: what is certified, what is hidden,
and the residual pool the hidden cards are drawn from.

Reached through `Game.state(seat, hidden=True)`. `hidden=False` returns the
true `GameState` and is the only sanctioned way to read it from outside the
engine; every such call site carries a `# true state:` comment saying why.
A `View` reads only sizes off the other seats, so it works the same over an
observed state whose opponents' piles are hidden counts.
"""

from __future__ import annotations

import math
import random
from typing import Protocol, Sequence

from .board.terrain import NUM_RESOURCES
from .cards import DECK_COMPOSITION, NUM_DEV_CARDS, DevCard
from .devcards import dev_count
from .economy import hand_size
from .game import Game
from .ledger import PublicLedger
from .state import BANK_PER_RESOURCE, GameState, HiddenRead, copy_state, is_hidden


class HoldReading(Protocol):
    """A caller's reading of what another seat's held development card is,
    for `View.sample`: how much more likely a card held `age` turns
    (`GameState.dev_ages`) is of kind `card`, before the unseen cards of each
    kind are counted in. The engine has none of its own."""

    def weight(self, card: int, age: int) -> float: ...


class View:
    """What one seat can know about every hand, and how to draw from it.

    `known[s]` is the certified lower bound on each resource and `unknown[s]`
    the cards the record cannot type; the perspective's own seat is exact and
    must be concrete, or `state.HiddenRead` is raised here. Hidden cards are
    drawn from one shared **residual pool**: per resource, those neither in
    the bank nor certified in any seat's `known`.

    `mover` is the seat on turn, `card_played` whether it has played its
    development card this turn and `turn` the game's turn count (`Game.turns`),
    all public; `None`, `False` and `None` where the view was built from a bare
    state rather than a game (`from_game` sets them). None is part of the
    information set's identity: equality and the hash ignore them.
    """

    def __init__(
        self, state: GameState, ledger: PublicLedger, perspective: int, *,
        certify: Sequence[tuple[int, Sequence[int]]] = (),
        mover: int | None = None, card_played: bool = False, turn: int | None = None,
    ) -> None:
        self.state = state
        self.ledger = ledger
        self.perspective = perspective
        self.mover = mover
        self.card_played = card_played
        self.turn = turn
        n = state.num_players
        self.num_players = n
        if is_hidden(state.hands[perspective]):
            raise HiddenRead(
                f"seat {perspective} cannot take a view of a state that hides "
                f"its own hand"
            )
        if is_hidden(state.dev_cards[perspective]) or is_hidden(state.new_dev_cards[perspective]):
            raise HiddenRead(f"seat {perspective} cannot take a view that hides its own development cards")
        # Sizes only: an observed state carries opponents' hands as counts.
        self.sizes = [hand_size(state, seat) for seat in range(n)]
        self.known: list[list[int]] = []
        self.unknown: list[int] = []
        for seat in range(n):
            if seat == perspective:
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
                # The record overclaims: shed certainty, largest entry
                # first, until it fits the public size.
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
    def from_game(cls, game: Game, perspective: int) -> View:
        return cls(game._state, game.ledger, perspective, mover=game.current_player,
                   card_played=game.dev_card_played, turn=game.turns)

    def signature(self) -> tuple:
        """`known`/`unknown`/`pool` as one hashable tuple: everything a
        belief over the hidden cards reads. Built once; the three are never
        mutated."""
        if self._signature is None:
            self._signature = (
                tuple(tuple(known) for known in self.known),
                tuple(self.unknown),
                tuple(self.pool),
            )
        return self._signature

    def __eq__(self, other: object) -> bool:
        """Same information set, not the same object. Deliberately does not
        compare `self.state`: `GameState` has no `__eq__` of its own, so two
        views of identically replayed games would never compare equal."""
        if not isinstance(other, View):
            return NotImplemented
        return (
            self.perspective == other.perspective
            and self.num_players == other.num_players
            and self.sizes == other.sizes
            and self.signature() == other.signature()
        )

    def __hash__(self) -> int:
        return hash((self.perspective, self.num_players, tuple(self.sizes), self.signature()))

    def unseen_dev_cards(self) -> list[int]:
        """The development cards this seat has not seen, by type. Exact: the
        deck's composition less own holdings and the two public tallies."""
        state = self.state
        counts = [DECK_COMPOSITION[card] for card in DevCard]
        me = self.perspective
        for card in range(NUM_DEV_CARDS):
            counts[card] -= state.dev_cards[me][card] + state.new_dev_cards[me][card]
            counts[card] -= state.dev_cards_played[card]
        counts[DevCard.KNIGHT] -= sum(state.knights_played)
        return [max(0, c) for c in counts]

    def p_holds(
        self, seat: int, bundle: Sequence[int], *, draws: int = 64,
        rng: random.Random | None = None,
    ) -> float:
        """Probability, on this view, that `seat` holds at least `bundle`
        (a count per resource). Exact where `known` decides it -- always, for
        the perspective's own seat -- or one card of one type is short;
        otherwise a Monte-Carlo estimate over `draws` hands dealt from the
        pool, `rng` defaulting to a fixed stream."""
        known = self.known[seat]
        need = [max(0, b - k) for b, k in zip(bundle, known)]
        short = sum(need)
        if short == 0:
            return 1.0
        hidden = self.unknown[seat]
        if short > hidden or seat == self.perspective:
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

    def sample(self, rng: random.Random, hold: HoldReading | None = None) -> GameState:
        """One determinized world consistent with everything public: a concrete
        `GameState` matching the real position's hand and card counts, hidden
        cards dealt from the pool without replacement, opponents' card types
        redrawn all matured. Every unseen development card is dealt alike,
        unless the caller passes its own `hold` reading: then, where the ages
        are known (`GameState.dev_ages`), each held card is dealt by kind in
        proportion to the unseen cards of the kind times `hold.weight(kind,
        age)`."""
        state = copy_state(self.state)
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
        others = [seat for seat in range(self.num_players) if seat != self.perspective]
        ages = [state.dev_ages[seat] for seat in others]
        if hold is not None and all(a is not None and len(a) == dev_count(state, seat)
                                    for seat, a in zip(others, ages)):
            # Each held card is dealt by the caller's reading of its age, from
            # the cards still undealt.
            left = list(composition)
            for seat, seat_ages in zip(others, ages):
                held = [0] * NUM_DEV_CARDS
                for age in seat_ages:
                    weights = [n * hold.weight(card, age) for card, n in enumerate(left)]
                    total = sum(weights)
                    if not total:
                        held[DevCard.KNIGHT] += 1
                        continue
                    pick, card = rng.random() * total, 0
                    while pick >= weights[card] and card < NUM_DEV_CARDS - 1:
                        pick -= weights[card]
                        card += 1
                    held[card] += 1
                    left[card] -= 1
                state.dev_cards[seat] = held
                state.new_dev_cards[seat] = [0] * NUM_DEV_CARDS
            unseen = [c for c in range(NUM_DEV_CARDS) for _ in range(left[c])]
            rng.shuffle(unseen)
            cursor = 0
        else:
            unseen = [c for c in range(NUM_DEV_CARDS) for _ in range(composition[c])]
            rng.shuffle(unseen)
            cursor = 0
            for seat in others:
                count = dev_count(state, seat)
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
