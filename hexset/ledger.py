# SPDX-License-Identifier: GPL-3.0-only
"""Public-knowledge bookkeeping: each seat's hand composition, reconstructed
incrementally from moves that are public by the rules of the game, and what
each seat knows beyond that.

Per seat, `known[5]` is a certified lower bound on how many of each resource
the seat holds, `most[5]` a certified upper bound, and `unknown` the cards the
log accounts for but cannot type. The invariant, at every step:

    sum(known) + unknown == the seat's true hand size
    known[r] <= the seat's true count of resource r <= most[r], for every r

The upper bound is what a Monopoly leaves (every seat that surrendered holds
none of the resource) and what a steal carries: the thief's card can only be
one the victim could hold. After every change a row is settled: `most` is
held to `known + unknown`, and where the unknown cards must fill a
resource's room (`most - known`) because the others have too little, they
are typed (`SeatLedger.settle`).

**`PublicLedger.seats` is the COMMON-KNOWLEDGE view.** Every way a card changes
hands is public and named except a steal, and the common rows never read the
stolen resource. They type a stolen card only where the table can: the
victim could hold one resource only.

**Each seat's own view is `for_seat(seat)`.** A steal shows its card to the
thief and the victim, so each of them keeps rows of its own from its first
such steal (`private`): the same rows, every public flow applied alike, the
card it saw typed. A seat that has seen none reads the common rows. A
seat's own view is what may reach its encoding and its belief; no other
seat's may. A copy made for one seat's search keeps that seat's rows alone
(`scrubbed`), so nothing it simulates reads what another seat saw.

The invariant holds only for a ledger that started in sync with the hand it
tracks, as every position `hexset.game` reaches on its own is. A caller that
writes `state.hands` directly desyncs it; `unknown` then clamps at zero.

**What a seat has shown it wants and will give up** is kept beside the hand:
`want[r]`, the most of resource `r` the seat has asked for, and `waste[r]`,
the most it has offered away (`show`). Both come from offers and answers put
to the table, which are as public as the cards, and both are cleared by the
cards themselves rather than by time: a want by the seat receiving that
resource, a waste by the seat parting with it. A steal moves neither, for the
same reason the table cannot name its card.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .board.terrain import NUM_RESOURCES
from .state import BANK_PER_RESOURCE, is_hidden

__all__ = [
    "SeatLedger",
    "PublicLedger",
]


@dataclass(slots=True)
class SeatLedger:
    """One seat's reconstructed hand composition: `known` and `most` bound
    each resource from below and above, `unknown` counts the cards between.

    `most` defaults to no bound, so a row written by hand -- `known` and
    `unknown` alone -- claims nothing above them; `settle` holds it to
    `known + unknown`."""

    known: list[int] = field(default_factory=lambda: [0] * NUM_RESOURCES)
    unknown: int = 0
    want: list[int] = field(default_factory=lambda: [0] * NUM_RESOURCES)
    waste: list[int] = field(default_factory=lambda: [0] * NUM_RESOURCES)
    most: list[int] = field(default_factory=lambda: [BANK_PER_RESOURCE] * NUM_RESOURCES)

    def total(self) -> int:
        return sum(self.known) + self.unknown

    def room(self) -> list[int]:
        """Per resource, how many of the unknown cards may be it: `most`
        held to `known + unknown`, less `known`."""
        u = self.unknown
        return [max(0, min(m - k, u)) for m, k in zip(self.most, self.known)]

    def settle(self) -> None:
        """Hold `most` to `known + unknown`, and type the unknown cards a
        resource's room must take: where the other resources' room together
        is short of `unknown`, the rest are this one. Rows the bounds cannot
        hold -- the unknown cards outnumber every resource's room, which only
        a hand written past the ledger makes -- lose their upper bounds
        instead."""
        known, most = self.known, self.most
        for r in range(len(known)):
            if most[r] < known[r]:
                most[r] = known[r]
            cap = known[r] + self.unknown
            if most[r] > cap:
                most[r] = cap
        if not self.unknown:
            return
        room = [m - k for m, k in zip(most, known)]
        spare = sum(room)
        if spare < self.unknown:
            for r in range(len(known)):
                most[r] = known[r] + self.unknown
            return
        for r, n in enumerate(room):
            forced = self.unknown - (spare - n)
            if forced > 0:
                known[r] += forced
                self.unknown -= forced
                spare -= forced

    def copy(self) -> "SeatLedger":
        # Past `__init__`: every search copy takes one per seat per view.
        out = _new_row(SeatLedger)
        out.known = self.known[:]
        out.unknown = self.unknown
        out.want = self.want[:]
        out.waste = self.waste[:]
        out.most = self.most[:]
        return out


_new_row = object.__new__


@dataclass(slots=True)
class PublicLedger:
    """Per-seat `SeatLedger`s, in board-seat order: the common-knowledge rows,
    and beside them each seat's own rows once a steal has shown it a card.

    `private[seat]` is `None` for a seat that knows only what the table
    knows, and the whole list `None` until the first steal anybody saw.
    Every public change is applied to the common rows and every seat's own
    alike; only `steal`'s card differs between them."""

    seats: list[SeatLedger]
    private: list[PublicLedger | None] | None = None

    @classmethod
    def new(cls, num_players: int) -> "PublicLedger":
        # Every hand starts empty, so every bound starts at zero.
        return cls(seats=[
            SeatLedger(most=[0] * NUM_RESOURCES) for _ in range(num_players)
        ])

    def copy(self) -> "PublicLedger":
        out = _new_row(PublicLedger)
        out.seats = [s.copy() for s in self.seats]
        out.private = None if self.private is None else [
            None if p is None else p.copy() for p in self.private
        ]
        return out

    def for_seat(self, seat: int) -> "PublicLedger":
        """What `seat` knows: its own rows where a steal has shown it a card,
        the common rows otherwise. The same object on every call until then,
        and after it; never a copy, so mutate a `knowledge` copy instead."""
        if self.private is not None:
            own = self.private[seat]
            if own is not None:
                return own
        return self

    def knowledge(self, seat: int) -> "PublicLedger":
        """A standalone copy of `seat`'s view (`for_seat`): its rows as the
        common ones, and no seat's own beside them."""
        rows = self.for_seat(seat).seats
        return PublicLedger(seats=[s.copy() for s in rows])

    def scrubbed(self, seat: int | None) -> "PublicLedger":
        """A copy keeping `seat`'s own rows and dropping every other seat's:
        the ledger of a world `seat` imagines, where another seat knows only
        what the table knows. `None` keeps no seat's."""
        out = PublicLedger(seats=[s.copy() for s in self.seats])
        if self.private is not None and seat is not None and self.private[seat] is not None:
            out.private = [None] * len(self.seats)
            out.private[seat] = self.private[seat].copy()
        return out

    def _views(self):
        """The common rows' ledger, then every seat's own: the ledgers a
        public change applies to."""
        yield self
        if self.private is not None:
            for own in self.private:
                if own is not None:
                    yield own

    def receive(self, seat: int, resource: int, n: int = 1) -> None:
        """`seat` publicly gained `n` of `resource`. Always safe: the identity
        is public here, so `known` cannot overstate what is true."""
        if n <= 0:
            return
        for ledger in self._views():
            seat_ledger = ledger.seats[seat]
            seat_ledger.known[resource] += n
            seat_ledger.most[resource] += n
            seat_ledger.want[resource] = max(0, seat_ledger.want[resource] - n)

    def spend(self, seat: int, resource: int, n: int = 1) -> None:
        """`seat` publicly lost `n` of `resource`. Every caller must name a
        genuinely public resource; a steal is not one and calls `steal`.
        Draws from `known[resource]` first and takes any shortfall out of
        `unknown`, since a public spend of more than was certified can only be
        explained by cards sitting there."""
        if n <= 0:
            return
        for ledger in self._views():
            ledger._spend_row(seat, resource, n)

    def _spend_row(self, seat: int, resource: int, n: int, *, shown: bool = True) -> None:
        """`n` of `resource` leave `seat`'s row; `shown` clears that much
        waste, which a steal, moving no intent, does not."""
        seat_ledger = self.seats[seat]
        if shown:
            seat_ledger.waste[resource] = max(0, seat_ledger.waste[resource] - n)
        seat_ledger.most[resource] = max(0, seat_ledger.most[resource] - n)
        from_known = min(n, seat_ledger.known[resource])
        seat_ledger.known[resource] -= from_known
        deficit = n - from_known
        if deficit:
            seat_ledger.unknown = max(0, seat_ledger.unknown - deficit)
            seat_ledger.settle()

    def show(self, seat: int, received, *, holds: bool = True) -> None:
        """`seat` put `received` to the table -- an offer, a counter, an
        acceptance, or an ask with nothing named in return -- signed towards
        `seat`: positive for what it asks for, negative for what it gives.

        Each asked resource becomes a want and stops being a waste; each given
        one the reverse. Counts keep the most shown, so hearing the same offer
        twice changes nothing. `holds` says the referee checked `seat` covers
        what it gives, as it does for every offer and answer put to a table
        here: the given cards are then certified, and `known` rises to them
        out of `unknown`, where they can only have been."""
        for ledger in self._views():
            seat_ledger = ledger.seats[seat]
            for r, n in enumerate(received):
                if n > 0:
                    seat_ledger.want[r] = max(seat_ledger.want[r], n)
                    seat_ledger.waste[r] = 0
                elif n < 0:
                    seat_ledger.waste[r] = max(seat_ledger.waste[r], -n)
                    seat_ledger.want[r] = 0
                    if holds:
                        rise = min(-n - seat_ledger.known[r], seat_ledger.unknown)
                        if rise > 0:
                            seat_ledger.known[r] += rise
                            seat_ledger.unknown -= rise
            seat_ledger.settle()

    def gain_unknown(self, seat: int, n: int = 1) -> None:
        """`seat` gained `n` cards of a resource this ledger cannot type.
        Never attributed to a specific `known[r]`."""
        if n <= 0:
            return
        for ledger in self._views():
            seat_ledger = ledger.seats[seat]
            seat_ledger.unknown += n
            seat_ledger.most = [m + n for m in seat_ledger.most]

    def steal(self, thief: int, victim: int, resource: int | None = None) -> None:
        """A robber or knight steal: one card, `victim` -> `thief`.

        `resource` is the card, where whoever applies the steal saw it (`None`
        where it did not). It reaches the thief's and the victim's own rows
        and nothing else: the common rows and every other seat's take the
        steal untyped (`_steal_unseen`)."""
        seen = resource is not None and resource >= 0
        parties: tuple[PublicLedger, ...] = ()
        if seen:
            if self.private is None:
                self.private = [None] * len(self.seats)
            for seat in (thief, victim):
                if self.private[seat] is None:
                    # From the common rows as they stand before this steal.
                    self.private[seat] = PublicLedger(seats=[s.copy() for s in self.seats])
            parties = (self.private[thief], self.private[victim])
        for ledger in self._views():
            if any(ledger is own for own in parties):
                ledger._spend_row(victim, int(resource), 1, shown=False)
                ledger.seats[thief].known[int(resource)] += 1
                ledger.seats[thief].most[int(resource)] += 1
            else:
                ledger._steal_unseen(thief, victim)

    def _steal_unseen(self, thief: int, victim: int) -> None:
        """A steal whose card this ledger was not shown. Typed where these
        rows alone decide it -- the victim can hold one resource only -- and
        otherwise identity-independent by construction: the victim's every
        `known[r]` floors by one and `unknown` is re-solved, because one entry
        visibly dropping would name the stolen type. The thief's card may be
        any resource the victim could hold, and no other."""
        victim_ledger = self.seats[victim]
        thief_ledger = self.seats[thief]
        # A row written by hand carries no bound until settled.
        victim_ledger.settle()
        could = [r for r, m in enumerate(victim_ledger.most) if m > 0]
        if len(could) == 1:
            r = could[0]
            self._spend_row(victim, r, 1, shown=False)
            thief_ledger.known[r] += 1
            thief_ledger.most[r] += 1
            return
        old_total = victim_ledger.total()
        for r in range(len(victim_ledger.known)):
            victim_ledger.known[r] = max(0, victim_ledger.known[r] - 1)
        victim_ledger.unknown = max(0, old_total - 1 - sum(victim_ledger.known))
        victim_ledger.settle()
        thief_ledger.unknown += 1
        for r in could:
            thief_ledger.most[r] += 1
        thief_ledger.settle()

    def emptied(self, seat: int, resource: int) -> None:
        """`seat` publicly holds none of `resource`: what it is left with
        after surrendering to a Monopoly. Its unknown cards are then other
        resources, which may type them."""
        for ledger in self._views():
            seat_ledger = ledger.seats[seat]
            seat_ledger.known[resource] = 0
            seat_ledger.most[resource] = 0
            seat_ledger.settle()

    def apply_hand_diff(
        self, before: list[list[int]], after: list[list[int]]
    ) -> None:
        """Record a publicly exact, simultaneous hand change across every seat
        it touches. A steal must never come through here: a diff would read
        the stolen resource straight off the hand arrays (see `steal`).

        A seat whose hand is hidden (`state.HiddenHand`, an observed state)
        is diffed by the public changes it recorded (`HiddenHand.flow`), which
        are exactly the ones a concrete hand's diff would show."""
        for seat, (old, new) in enumerate(zip(before, after)):
            if is_hidden(old) or is_hidden(new):
                if not (is_hidden(old) and is_hidden(new)):
                    raise ValueError(
                        f"seat {seat}'s hand changed between hidden and concrete mid-diff"
                    )
                pairs = zip(old.flow, new.flow)
            else:
                pairs = zip(old, new)
            for resource, (o, n) in enumerate(pairs):
                delta = n - o
                if delta > 0:
                    self.receive(seat, resource, delta)
                elif delta < 0:
                    self.spend(seat, resource, -delta)
