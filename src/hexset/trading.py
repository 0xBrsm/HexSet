# SPDX-License-Identifier: GPL-3.0-only
"""Player-to-player trading: what an offer is, and when it is legal.

Offers are uncapped. A player may ask for any quantity of anything in exchange
for any quantity of anything they hold, because that is what the game allows and
capping it would be a modelling convenience showing through into the rules.

Two restrictions are real rules rather than conveniences. Both sides must be
non-empty, since a one-sided offer is a gift and the game has no such move. And
no resource may appear on both sides: two wood for one wood and one brick is
just one wood for one brick, so allowing it would only add duplicate ways to
express the same trade.

Whether an offer is *addressed* to particular players is deliberately not
modelled here. An offer goes to the table and the first player willing to take
it does. Counter-offers are not modelled either — a counter is simply the next
offer, made by whoever wants to make it on their own turn.
"""

from __future__ import annotations

from dataclasses import dataclass

from .board.terrain import NUM_RESOURCES
from .state import GameState

Bundle = tuple[int, ...]


@dataclass(frozen=True)
class Offer:
    """`give` leaves the proposer, `want` comes back to them."""

    proposer: int
    give: Bundle
    want: Bundle


def bundle(**amounts: int) -> Bundle:
    """A resource bundle by name, for tests and hand-written offers."""
    from .board.terrain import Resource

    counts = [0] * NUM_RESOURCES
    for name, count in amounts.items():
        counts[Resource[name.upper()]] = count
    return tuple(counts)


def holds(state: GameState, player: int, wanted: Bundle) -> bool:
    hand = state.hands[player]
    return all(hand[r] >= n for r, n in enumerate(wanted))


def well_formed(offer: Offer) -> bool:
    if len(offer.give) != NUM_RESOURCES or len(offer.want) != NUM_RESOURCES:
        return False
    if any(n < 0 for n in offer.give) or any(n < 0 for n in offer.want):
        return False
    if not any(offer.give) or not any(offer.want):
        return False
    # A resource on both sides means the offer reduces to a smaller one.
    return not any(g and w for g, w in zip(offer.give, offer.want))


def can_propose(state: GameState, offer: Offer) -> bool:
    return well_formed(offer) and holds(state, offer.proposer, offer.give)


def can_accept(state: GameState, offer: Offer, responder: int) -> bool:
    if responder == offer.proposer:
        return False
    return holds(state, responder, offer.want)


def responders(state: GameState, offer: Offer) -> tuple[int, ...]:
    """Everyone who could take the offer if they wanted to, clockwise from the proposer.

    Players who cannot cover it are left out rather than asked and forced to
    decline, so a turn does not spend actions on foregone conclusions.

    This is the *eligibility* list, in a fixed order so it can be tested and
    compared. The order the table is actually asked in is decided by
    `game.propose_trade`: the proposer's `ask` if given, otherwise a random
    permutation from the game's RNG. It used to be this clockwise order, on the
    argument that rotating with the proposer hands first refusal to nobody —
    which is true over a game and false within one duel lineup: with two copies
    of a policy seated together, the next seat is a twin half the time, and the
    2026-08-29 harness-path check measured the consequence as a +0.35 VP
    "seat geometry" effect. The trading design note has the full account.

    Ordering by anything about the players — how far ahead they are, say — would
    be a partner preference, and a preference is the proposer's to hold rather
    than the rules' to impose. It belongs on the action, where a policy can
    learn it and can learn when to break it.
    """
    n = state.num_players
    asked = ((offer.proposer + 1 + i) % n for i in range(n))
    return tuple(p for p in asked if can_accept(state, offer, p))


def execute(state: GameState, offer: Offer, responder: int) -> None:
    if not can_propose(state, offer):
        raise ValueError(f"player {offer.proposer} cannot make this offer")
    if not can_accept(state, offer, responder):
        raise ValueError(f"player {responder} cannot take this offer")

    proposer_hand = state.hands[offer.proposer]
    responder_hand = state.hands[responder]
    for r in range(NUM_RESOURCES):
        proposer_hand[r] -= offer.give[r]
        responder_hand[r] += offer.give[r]
        proposer_hand[r] += offer.want[r]
        responder_hand[r] -= offer.want[r]
