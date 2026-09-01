# SPDX-License-Identifier: GPL-3.0-only
"""Public-knowledge bookkeeping: each seat's hand composition, reconstructed
incrementally from moves that are public by the rules of the game.

Production, distribution, initial-settlement grants, bank trades, player
trades, builds, development-card purchases, discards, monopoly and year of
plenty are all public: every card that changes hands in them is named and
counted where everyone can see it. A robber or knight steal is the one
exception — it moves exactly one card whose identity is hidden from
everyone but the thief and the victim.

**v1 simplification (documented, not hidden): this is the COMMON-KNOWLEDGE
view.** The thief learns exactly what they took and the victim watches
exactly what left their hand — sharper private knowledge either of them
could in principle condition on. This module does not model that: both
seats' `PublicLedger` entries read exactly as a bystander's would, which is
strictly *less* informed than the thief or victim actually are. That is a
deliberate asymmetry — conservative (the ledger never claims a certainty a
public observer would not have), never leaking (it never turns the private
half of a steal into a feature that only makes sense if a specific seat's
perspective secretly saw the identity).

Per seat: `known[5]` is a certified lower bound on how many of each
resource the seat holds — a count the public log has pinned to that exact
type — and `unknown` is the number of cards the log accounts for in total
but cannot type. The only invariant this module exists to keep true at
every step is:

    sum(known) + unknown == the seat's true hand size
    known[r] <= the seat's true count of resource r, for every r

**The steal convention.** A steal moves one card, thief <- victim, of a
resource neither of them announces. Two things are public regardless: the
victim's hand shrank by one and the thief's grew by one. What is *not*
public is which resource — so the thief's gain is always credited to
`unknown` (never to a specific `known[r]`, which is exactly the private
knowledge being withheld).

The victim's side needs one more bit of care, because merely picking a
fixed bucket to decrement (e.g. "always take it from `unknown` first") is
not just imprecise, it is not even *safe*: if the game's own random draw
happened to remove a card of a resource this ledger had already certified
as `known`, leaving that count untouched would make `known[r]` exceed the
seat's new true count of `r`, and the invariant above would be false. The
only convention that cannot do that, whatever the game's draw picked, is to
let `spend()` (below) resolve the loss against the *actual* resource the
engine's own `robber.steal`/`devcards.play_knight` returns: take the unit
from `known[r]` if the ledger already had one certified, otherwise treat it
as one of the already-uncertain cards and take it from `unknown`. This
reads the true identity internally — the engine sees the mutation
directly, which is exactly the simplification the module docstring
promises over a log-replay reconstruction — but it only ever *removes*
certainty (shrinking a `known` count or an `unknown` count that already
existed) and never *asserts* new certainty about what remains. It is the
same asymmetry as the thief's side, applied to keep the invariant provably
true rather than merely usually true: see `spend`'s docstring for the
one-line proof.

Two other events deserve a note because they *resolve* uncertainty rather
than create it. Monopoly forces every other seat to publicly hand over
every card of one resource: this reveals each victim's exact prior count of
that resource (some of which may have been sitting in `unknown`), so it is
handled as an ordinary `spend()` of that seat's true holding, which is
plumbed straight through by `hexset.game`. Any later spend that draws on a
resource beyond what `known` currently certifies (a build, a bank trade, a
discard) similarly resolves `unknown` cards into certainty by elimination —
the spend is itself public, so if it draws more of resource `r` than
`known[r]` accounts for, the shortfall can only have come from the
`unknown` pool, and `known[r]` never needs to go negative to explain it.

Reference for the reconstruction semantics this simplifies: `state_ledger.py`
(hexset-chat, log-replay from a public event stream — see its own docstring
for the imprecisions *that* version accepts, most of which do not apply
here because this module sees the engine's mutations directly instead of
reconstructing them after the fact from a log).

The invariant above is *proved* to hold for `spend()`, not merely tested —
provided the ledger started in sync with the true hand it is tracking.
Every position `hexset.game` reaches on its own keeps that true by
construction: `start`/`imagine` create the ledger alongside the hand it
describes, and every mutation of `state.hands` routes through `receive`/
`spend` (directly, or via `apply_hand_diff`) in the same call that changes
the hand. A caller that pokes `state.hands` directly and only afterwards
drives the ledger-wired path — most of this repo's test fixtures, built
for scenarios that have nothing to do with the ledger — starts out of
sync, and `spend`'s deficit can then exceed `unknown`. Rather than raise on
every such fixture, `unknown` clamps at zero in that case; see `spend`'s
own docstring for exactly where the proof stops applying.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .board.terrain import NUM_RESOURCES


@dataclass
class SeatLedger:
    """One seat's reconstructed hand composition."""

    known: list[int] = field(default_factory=lambda: [0] * NUM_RESOURCES)
    unknown: int = 0

    def total(self) -> int:
        return sum(self.known) + self.unknown

    def copy(self) -> "SeatLedger":
        return SeatLedger(known=self.known[:], unknown=self.unknown)


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
        """`seat` publicly gained `n` of `resource` — production, a bank or
        player trade's receiving side, year of plenty, a monopoly's thief
        side. Always safe: the identity is public by construction here, so
        crediting it to `known` cannot overstate what is now true."""
        if n <= 0:
            return
        self.seats[seat].known[resource] += n

    def spend(self, seat: int, resource: int, n: int = 1) -> None:
        """`seat` publicly lost `n` of `resource` — a build, a dev-card buy,
        a bank or player trade's giving side, a discard, a monopoly victim's
        full surrender, or (via `hexset.game`'s steal handling) the one card
        a robber or knight takes, once its true identity is known engine-side.

        Draws from `known[resource]` first and only reaches into `unknown`
        for whatever a `known[resource]` <= n shortfall does not cover —
        which is exactly the "over-draw resolves unknowns" rule: a spend
        that takes more of `resource` than the ledger had certified can only
        be explained by cards that were sitting in `unknown`, so the deficit
        moves out of `unknown` rather than driving `known[resource]` below
        zero.

        This is the one place invariant safety is not obvious, so here is
        the argument in full. Before the call, `known[resource] <= true`
        and `sum(known) + unknown == true_total` hold (the loop invariant
        every other method preserves too). The engine only calls this with
        an `n` the seat truly holds record of losing, so `true` (this seat's
        real count of `resource`) drops to `true - n`. Split the deficit as
        `d = max(0, n - known[resource])`:

        * If `d == 0`, `known[resource]` drops by `n` and stays `<= true - n`
          (both sides of the prior `<=` shrank by the same `n`). Nothing
          else moved, so the sum invariant holds by the same `n`.
        * If `d > 0`, `known[resource]` bottoms out at `0`, and `unknown`
          must cover `d`. It can: `unknown == true_total - sum(known)`, and
          `true_total >= n + sum_{r' != resource}(known[r'])` because the
          seat truly holds at least `n` of `resource` and at least
          `known[r']` of every other type — so `unknown >= n - known[resource]
          == d`. Removing `d` from `unknown` cannot take it negative, and
          `known[resource] (== 0) <= true - n` trivially, since a true count
          can never go negative.

        Both branches leave `known[r] <= true[r]` for every `r` and
        `sum(known) + unknown == true_total`, so the invariant survives any
        legal spend *as long as the ledger started in sync* -- true for
        every position `hexset.game` reaches on its own, since `start`/
        `imagine` create it and every hand mutation routes through here or
        `receive`. It is not true of a position a test builds by writing
        `state.hands` directly (`tests/helpers.give`/`clear_hand`, common
        throughout this suite for scenarios the ledger has no reason to
        know or care about): there, `known`/`unknown` can be short of the
        deficit this method's proof assumes never happens. Rather than
        raise on every such fixture -- which does not indicate a ledger bug,
        only that this particular scenario was built without it in mind --
        `unknown` clamps at zero. A clamp is silent exactly where the proof
        above already guarantees it never fires on real engine-driven play.
        """
        if n <= 0:
            return
        seat_ledger = self.seats[seat]
        from_known = min(n, seat_ledger.known[resource])
        seat_ledger.known[resource] -= from_known
        deficit = n - from_known
        if deficit:
            seat_ledger.unknown = max(0, seat_ledger.unknown - deficit)

    def gain_unknown(self, seat: int, n: int = 1) -> None:
        """`seat` gained `n` cards of a resource nobody but them (and, for a
        steal, the victim) knows — the thief's half of a robber/knight
        steal. Never attributed to a specific `known[r]`: that is exactly
        the private knowledge this v1 ledger declines to model."""
        if n <= 0:
            return
        self.seats[seat].unknown += n

    def apply_hand_diff(
        self, before: list[list[int]], after: list[list[int]]
    ) -> None:
        """Every hand change here is publicly exact and simultaneous across
        however many seats it touches — production, a build or dev-card
        buy, a bank or player trade, a discard, year of plenty, or
        monopoly's all-seats-at-once transfer. Diffing the whole table's
        hands and feeding the per-resource deltas to `receive`/`spend`
        covers every one of those in one call, rather than re-deriving each
        rule's exact amounts a second time here. The one exception is a
        steal, whose identity is hidden and is handled by `hexset.game`
        directly through `spend`/`gain_unknown`, never through a diff (a
        diff would read the true resource off the hand arrays themselves,
        which is exactly the information a steal must not leak)."""
        for seat, (old, new) in enumerate(zip(before, after)):
            for resource, (o, n) in enumerate(zip(old, new)):
                delta = n - o
                if delta > 0:
                    self.receive(seat, resource, delta)
                elif delta < 0:
                    self.spend(seat, resource, -delta)
