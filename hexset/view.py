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
    the cards the record cannot type, as the perspective knows them
    (`PublicLedger.for_seat`: the common rows, plus the steals it saw); the perspective's own seat is exact and
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
        self.room: list[list[int]] = []
        #: Per seat, whether its upper bounds bind: some resource has room
        #: for fewer than all its unknown cards.
        self.bounded: list[bool] = []
        # What this seat knows: its own rows where a steal showed it a card.
        rows = ledger.for_seat(perspective).seats
        for seat in range(n):
            if seat == perspective:
                self.known.append(state.hands[seat][:])
                self.unknown.append(0)
                self.room.append([0] * NUM_RESOURCES)
                self.bounded.append(False)
                continue
            known = rows[seat].known[:]
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
            hidden = self.sizes[seat] - sum(known)
            self.unknown.append(hidden)
            # Built for every seat of every view, so the unbound case -- every
            # resource with room for all the unknown cards -- is one pass.
            gaps = [m - k for m, k in zip(rows[seat].most, known)]
            bound = hidden > 0 and min(gaps) < hidden
            if bound:
                room = [0 if g < 0 else g if g < hidden else hidden for g in gaps]
                if sum(room) < hidden:
                    # Bounds the certified cards contradict: drop them.
                    bound = False
            if not bound:
                room = [hidden] * NUM_RESOURCES
            self.room.append(room)
            self.bounded.append(bound)

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
        """`known`/`unknown`/`pool`, and `room` where a bound binds, as one
        hashable tuple: everything a belief over the hidden cards reads.
        Built once; none of them is ever mutated."""
        if self._signature is None:
            signature: tuple = (
                tuple(tuple(known) for known in self.known),
                tuple(self.unknown),
                tuple(self.pool),
            )
            if any(self.bounded):
                signature += (tuple(tuple(room) for room in self.room),)
            self._signature = signature
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
        (a count per resource). Exact where `known` or the upper bounds
        decide it -- always, for the perspective's own seat -- or one card of
        one type is short of a seat no bound binds; otherwise a Monte-Carlo
        estimate over `draws` hands dealt from the pool within the seat's
        bounds, `rng` defaulting to a fixed stream."""
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
        bounded = self.bounded[seat]
        if bounded and any(n > r for n, r in zip(need, self.room[seat])):
            return 0.0
        if short == 1 and not bounded:
            r = need.index(1)
            return 1.0 - _hypergeometric_miss(self.pool_size, self.pool[r], hidden)
        rng = rng or random.Random(0)
        cards = self._pool_cards()
        hits = 0
        for _ in range(draws):
            drawn = (
                _capped_draw(rng, cards, hidden, self.room[seat]) if bounded
                else rng.sample(cards, hidden)
            )
            counts = known[:]
            for r in drawn:
                counts[r] += 1
            if all(c >= b for c, b in zip(counts, bundle)):
                hits += 1
        return hits / draws

    def _pool_cards(self) -> list[int]:
        return [r for r in range(NUM_RESOURCES) for _ in range(self.pool[r])]

    def _deal(self, rng: random.Random) -> list[list[int]]:
        """Each seat's hidden cards, dealt from the pool without replacement:
        one shuffle sliced seat by seat where no bound binds. Where one does,
        whole deals are drawn until one keeps every seat within its bounds
        (`DEAL_TRIES`), the exact conditional; failing that, seats are dealt
        tightest first, card by card from what their bounds allow."""
        cards = self._pool_cards()
        seats = [s for s in range(self.num_players) if s != self.perspective]
        out: list[list[int]] = [[] for _ in range(self.num_players)]
        if not any(self.bounded):
            rng.shuffle(cards)
            cursor = 0
            for seat in seats:
                hidden = self.unknown[seat]
                out[seat] = cards[cursor : cursor + hidden]
                cursor += hidden
            return out
        for _ in range(DEAL_TRIES):
            rng.shuffle(cards)
            cursor, fits = 0, True
            for seat in seats:
                hidden = self.unknown[seat]
                dealt = cards[cursor : cursor + hidden]
                cursor += hidden
                if self.bounded[seat] and not _within(dealt, self.room[seat]):
                    fits = False
                    break
                out[seat] = dealt
            if fits:
                return out
        pool = list(self.pool)
        order = sorted(seats, key=lambda s: (sum(self.room[s]) - self.unknown[s], s))
        for seat in order:
            out[seat] = _capped_draw(rng, None, self.unknown[seat], self.room[seat], pool)
        return out

    def expected_hand(self, seat: int) -> list[float]:
        """`known` plus the hidden cards spread in the pool's proportions,
        each resource held to its room: what a resource's room cannot take
        goes to the others, in their proportions. Exact in expectation where
        no bound binds."""
        known = self.known[seat]
        hidden = self.unknown[seat]
        if not hidden or not self.pool_size:
            return [float(k) for k in known]
        if not self.bounded[seat]:
            share = hidden / self.pool_size
            return [k + share * p for k, p in zip(known, self.pool)]
        room = self.room[seat]
        spread = [0.0] * NUM_RESOURCES
        open_ = [r for r in range(NUM_RESOURCES) if room[r] > 0]
        left = float(hidden)
        while open_ and left > 1e-9:
            weight = sum(self.pool[r] for r in open_)
            if weight <= 0:
                weights = {r: 1.0 / len(open_) for r in open_}
            else:
                weights = {r: self.pool[r] / weight for r in open_}
            capped = [r for r in open_ if spread[r] + left * weights[r] >= room[r]]
            if not capped:
                for r in open_:
                    spread[r] += left * weights[r]
                break
            for r in capped:
                left -= room[r] - spread[r]
                spread[r] = float(room[r])
                open_.remove(r)
        return [k + e for k, e in zip(known, spread)]

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
        for seat, hidden in enumerate(self._deal(rng)):
            if seat == self.perspective:
                continue
            hand = self.known[seat][:]
            for r in hidden:
                hand[r] += 1
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


#: Whole deals `View.sample` draws for one that keeps every seat within its
#: upper bounds, before dealing seat by seat.
DEAL_TRIES = 16


def _within(cards: Sequence[int], room: Sequence[int]) -> bool:
    counts = [0] * len(room)
    for r in cards:
        counts[r] += 1
        if counts[r] > room[r]:
            return False
    return True


def _capped_draw(
    rng: random.Random, cards: Sequence[int] | None, n: int, room: Sequence[int],
    pool: list[int] | None = None,
) -> list[int]:
    """`n` cards for one seat without replacement, no resource past its
    `room`: by rejection from `cards` while that is cheap, else card by card
    in proportion to what is left of `pool` (built from `cards` if not
    given, and drawn down in place). A resource the pool has run out of
    within the bounds is taken past them rather than not at all."""
    if cards is not None:
        for _ in range(DEAL_TRIES):
            drawn = rng.sample(cards, n)
            if _within(drawn, room):
                return drawn
    if pool is None:
        pool = [0] * len(room)
        for r in cards or ():
            pool[r] += 1
    left = list(room)
    out: list[int] = []
    for _ in range(n):
        allowed = [r for r in range(len(pool)) if pool[r] > 0 and left[r] > 0]
        if not allowed:
            allowed = [r for r in range(len(pool)) if pool[r] > 0] or list(range(len(pool)))
        total = sum(pool[r] for r in allowed)
        if total > 0:
            pick = rng.random() * total
            for r in allowed:
                pick -= pool[r]
                if pick < 0:
                    break
        else:
            r = rng.choice(allowed)
        out.append(r)
        if pool[r] > 0:
            pool[r] -= 1
        left[r] -= 1
    return out


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
