# SPDX-License-Identifier: GPL-3.0-only
"""Public-knowledge bookkeeping: each seat's hand composition, reconstructed
incrementally from moves that are public by the rules of the game.

Per seat, `known[5]` is a certified lower bound on how many of each resource
the seat holds and `unknown` the cards the public log accounts for but cannot
type. The invariant, at every step:

    sum(known) + unknown == the seat's true hand size
    known[r] <= the seat's true count of resource r, for every r

**This is the COMMON-KNOWLEDGE view.** Every way a card changes hands is
public and named except a steal, whose thief's and victim's private knowledge
is not modelled: no code path here reads the stolen resource, so it has no
channel to `hexset.encoding`'s output.

The invariant holds only for a ledger that started in sync with the hand it
tracks, as every position `hexset.game` reaches on its own is. A caller that
writes `state.hands` directly desyncs it; `unknown` then clamps at zero.

**What a seat has shown it wants and will give up** is kept beside the hand:
`want[r]`, the most of resource `r` the seat has asked for, and `waste[r]`,
the most it has offered away (`show`). Both come from offers and answers put
to the table, which are as public as the cards, and both are cleared by the
cards themselves rather than by time: a want by the seat receiving that
resource, a waste by the seat parting with it. A steal moves neither, for the
same reason it names no resource.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .board.terrain import NUM_RESOURCES
from .state import is_hidden

__all__ = [
    "SeatLedger",
    "PublicLedger",
]


@dataclass
class SeatLedger:
    """One seat's reconstructed hand composition."""

    known: list[int] = field(default_factory=lambda: [0] * NUM_RESOURCES)
    unknown: int = 0
    want: list[int] = field(default_factory=lambda: [0] * NUM_RESOURCES)
    waste: list[int] = field(default_factory=lambda: [0] * NUM_RESOURCES)

    def total(self) -> int:
        return sum(self.known) + self.unknown

    def copy(self) -> "SeatLedger":
        return SeatLedger(known=self.known[:], unknown=self.unknown,
                          want=self.want[:], waste=self.waste[:])


@dataclass
class PublicLedger:
    """Per-seat `SeatLedger`s, in board-seat order."""

    seats: list[SeatLedger]

    @classmethod
    def new(cls, num_players: int) -> "PublicLedger":
        return cls(seats=[SeatLedger() for _ in range(num_players)])

    def copy(self) -> "PublicLedger":
        return PublicLedger(seats=[s.copy() for s in self.seats])

    def receive(self, seat: int, resource: int, n: int = 1) -> None:
        """`seat` publicly gained `n` of `resource`. Always safe: the identity
        is public here, so `known` cannot overstate what is true."""
        if n <= 0:
            return
        seat_ledger = self.seats[seat]
        seat_ledger.known[resource] += n
        seat_ledger.want[resource] = max(0, seat_ledger.want[resource] - n)

    def spend(self, seat: int, resource: int, n: int = 1) -> None:
        """`seat` publicly lost `n` of `resource`. Every caller must name a
        genuinely public resource; a steal is not one and calls `steal`.
        Draws from `known[resource]` first and takes any shortfall out of
        `unknown`, since a public spend of more than was certified can only be
        explained by cards sitting there."""
        if n <= 0:
            return
        seat_ledger = self.seats[seat]
        seat_ledger.waste[resource] = max(0, seat_ledger.waste[resource] - n)
        from_known = min(n, seat_ledger.known[resource])
        seat_ledger.known[resource] -= from_known
        deficit = n - from_known
        if deficit:
            seat_ledger.unknown = max(0, seat_ledger.unknown - deficit)

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
        seat_ledger = self.seats[seat]
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

    def gain_unknown(self, seat: int, n: int = 1) -> None:
        """`seat` gained `n` cards of a resource only they (and a steal's
        victim) know. Never attributed to a specific `known[r]`."""
        if n <= 0:
            return
        self.seats[seat].unknown += n

    def steal(self, thief: int, victim: int) -> None:
        """A robber or knight steal: one hidden card, `victim` -> `thief`.

        Identity-independent by construction: never told which resource moved.
        The victim's every `known[r]` floors by one and `unknown` is re-solved,
        because one entry visibly dropping would name the stolen type."""
        victim_ledger = self.seats[victim]
        old_total = victim_ledger.total()
        for r in range(len(victim_ledger.known)):
            victim_ledger.known[r] = max(0, victim_ledger.known[r] - 1)
        victim_ledger.unknown = max(0, old_total - 1 - sum(victim_ledger.known))
        self.gain_unknown(thief, 1)

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
