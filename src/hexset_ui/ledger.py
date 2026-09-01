# SPDX-License-Identifier: GPL-3.0-only
"""Public-knowledge bookkeeping: each seat's hand composition, reconstructed
incrementally from moves that are public by the rules of the game.

Ported from `0xBrsm/dev-hexset:src/hexset/ledger.py`, verbatim — that repo is
the repo of record for this engine (see `hexset_ui`'s own README). Its
`GPL-3.0-only` license is compatible with this project's `AGPL-3.0`.

Production, distribution, initial-settlement grants, bank trades, player
trades, builds, development-card purchases, discards, monopoly and year of
plenty are all public: every card that changes hands in them is named and
counted where everyone can see it. A robber or knight steal is the one
exception — it moves exactly one card whose identity is hidden from
everyone but the thief and the victim.

**v1 simplification (documented, not hidden): this is the COMMON-KNOWLEDGE
view.** The thief learns exactly what they took and the victim watches
exactly what left their hand — sharper private knowledge either of them
could in principle condition on. This module does not model that, and does
not merely approximate it: `steal()` below never reads the true stolen
resource for any purpose, so there is no channel through which that
identity could reach the encoded features — not for the thief, not for the
victim, not for a bystander. `known`/`unknown` read exactly as a
bystander's own public reasoning would.

Per seat: `known[5]` is a certified lower bound on how many of each
resource the seat holds — a count the public log has pinned to that exact
type — and `unknown` is the number of cards the log accounts for in total
but cannot type. The only invariant this module exists to keep true at
every step is:

    sum(known) + unknown == the seat's true hand size
    known[r] <= the seat's true count of resource r, for every r

**The steal convention.** A steal moves one card, thief <- victim, of a
resource neither of them announces, and that identity has to stay hidden
from `hexset.encoding`'s output too — not merely under-modelled, actually
absent from the computation. That rules out any convention that reads the
true resource to decide *where* the loss lands, even engine-side, where
the value is available: doing so and then only sometimes decrementing a
specific `known[r]` (whenever the ledger happened to already certify that
type) lets the resource leak straight back out through *which entry
visibly dropped* — every other seat's encoded ledger block would then
differ depending on the stolen identity, which is exactly the leak
`encoding`'s information-set rule exists to prevent. (An earlier version
of this module did exactly that; `test_a_steal_is_identity_independent_in_the_encoding`
in `tests/test_ledger.py` is the regression test for it.)

The safe convention is identity-independent: it floors *every* `known[r]`
by one (never below zero), then re-solves `unknown` from the victim's own
previously tracked total, `unknown = old_total - 1 - sum(known')`. This is
the honest common-knowledge state — each `known[r] - 1` is exactly the
certainty public reasoning still supports once one card of unknown type
has left, and nothing about which entry actually lost its card shows up in
the update, because every entry moves the same way regardless of the
truth. The cost is real: uncertainty can balloon by as much as
`NUM_RESOURCES - 1` in a single steal (every `known[r]` was >= 1 before it,
all five floor to zero, and `unknown` absorbs the four-card gap) — the
honest price of losing one bit of information, not a shortcut around it.
The thief's gain is, as always, credited to `unknown` only, never to a
specific `known[r]`; see `gain_unknown`.

**Two events *resolve* uncertainty rather than create it, both because
their identity really is public, not merely inferred.** Monopoly forces
every other seat to publicly hand over every card of one resource: this
reveals each victim's exact prior count of that resource (some of which
may have been sitting in `unknown`), so it is handled as an ordinary
`spend()` of that seat's true holding, plumbed straight through by
`hexset.game`. And any later spend that draws on a resource beyond what
`known` currently certifies (a build, a bank trade, a discard) similarly
resolves `unknown` cards into certainty by elimination — the spend is
itself public, so if it draws more of resource `r` than `known[r]`
accounts for, the shortfall can only have come from the `unknown` pool.
Neither of these reads a hidden resource; a steal is the only mutation
that does not get this treatment, for exactly that reason.

Reference for the reconstruction semantics this simplifies: `state_ledger.py`
(hexset-chat, log-replay from a public event stream — see its own docstring
for the imprecisions *that* version accepts, most of which do not apply
here because this module sees the engine's mutations directly instead of
reconstructing them after the fact from a log).

The invariant above is *proved* to hold for `spend()` and `steal()`, not
merely tested — provided the ledger started in sync with the true hand it
is tracking. Every position `hexset.game` reaches on its own keeps that
true by construction: `start`/`imagine` create the ledger alongside the
hand it describes, and every mutation of `state.hands` routes through
`receive`/`spend`/`steal` (directly, or via `apply_hand_diff`) in the same
call that changes the hand. A caller that pokes `state.hands` directly and
only afterwards drives the ledger-wired path — most of this repo's test
fixtures, built for scenarios that have nothing to do with the ledger —
starts out of sync, and the deficit `spend`/`steal` compute can then run
past what `unknown` has on hand. Rather than raise on every such fixture,
`unknown` clamps at zero in that case; see each method's own docstring for
exactly where its proof stops applying.
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
        a bank or player trade's giving side, a discard, or a monopoly
        victim's full surrender. Every caller of this method names a
        resource that is genuinely public at the point it is called; a
        robber/knight steal is not one of them and never calls this (see
        `steal` instead).

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

    def steal(self, thief: int, victim: int) -> None:
        """A robber or knight steal: one hidden card, `victim` -> `thief`.

        Identity-independent by construction — this method never asks or is
        told which resource actually moved, so there is nothing here a
        stolen identity could leak through. Two things are public
        regardless of the identity: the victim's hand shrank by one and the
        thief's grew by one.

        The thief's side is simple: `gain_unknown` credits one card nobody
        can type. The victim's side floors *every* `known[r]` by one
        (`max(0, known[r] - 1)`, never negative) and re-solves `unknown`
        from the seat's own previously tracked total minus one, rather than
        decrementing any single entry — seeing `known` shrink at one
        specific resource and not another is exactly the signal that would
        tell a bystander which type was taken, so no entry is allowed to
        move differently from the rest.

        Proof this keeps `known[r] <= true[r]` for every `r`, whichever
        resource `r*` the hidden draw actually took (`true[r*]` drops by
        one, every other `true[r]` is unchanged): for `r == r*`, the prior
        invariant gives `known[r*] <= true[r*]`, so `known[r*] - 1 <=
        true[r*] - 1` whenever `known[r*] >= 1`, and if `known[r*] == 0` the
        floored value stays `0 <= true[r*] - 1` (true since `true[r*] >= 1`
        pre-steal, or there would have been nothing to steal). For `r !=
        r*`, flooring only ever shrinks `known[r]`, and it was already `<=
        true[r]`, which is unchanged. Both cases hold for *every* possible
        `r*` at once, which is exactly what "identity-independent" buys:
        the safety argument never has to know which one actually happened.

        And the sum invariant holds by construction, not by luck: `unknown`
        is *defined* as `old_total - 1 - sum(known')`, so
        `sum(known') + unknown' == old_total - 1 == true_total'` follows
        immediately from `old_total == true_total` (the loop invariant
        coming in) — and `unknown' >= 0` because the per-resource proof
        above already gives `sum(known') <= sum(true[r]') == true_total'`.

        As in `spend`, `unknown'` is additionally clamped at zero for a
        ledger a test fixture desynced by writing `state.hands` directly —
        the proof above guarantees that clamp is silent on any position
        `hexset.game` reached on its own.
        """
        victim_ledger = self.seats[victim]
        old_total = victim_ledger.total()
        for r in range(len(victim_ledger.known)):
            victim_ledger.known[r] = max(0, victim_ledger.known[r] - 1)
        victim_ledger.unknown = max(0, old_total - 1 - sum(victim_ledger.known))
        self.gain_unknown(thief, 1)

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
        directly through `steal`, never through a diff or `spend` — a diff
        would read the true resource straight off the hand arrays
        themselves, which is exactly the information a steal must not
        leak (see `steal`'s docstring for the identity-independent
        convention that replaces it)."""
        for seat, (old, new) in enumerate(zip(before, after)):
            for resource, (o, n) in enumerate(zip(old, new)):
                delta = n - o
                if delta > 0:
                    self.receive(seat, resource, delta)
                elif delta < 0:
                    self.spend(seat, resource, -delta)
