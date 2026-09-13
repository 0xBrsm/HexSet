# SPDX-License-Identifier: GPL-3.0-only
"""What a hand can reach and buy, in the closed form the network gate prices
a trade against (`hexset.clients.netbot.NetworkBot._evaluate_reachable`).

Three pure layers, in the order a hand actually moves through them:

1. `branch_hands` -- the hand itself, plus one hand per development card
   this seat may still play this turn (Year of Plenty, Monopoly). Knight
   and Road Building are never branches: neither changes what a trade is
   being priced against.
2. `bank_conversions` -- every hand reachable from a branch by zero or more
   bank trades, at this seat's own best ratio per resource.
3. `purchase_multisets` -- every multiset of `ROAD`/`SETTLEMENT`/`CITY`/
   `DEV_CARD` purchases a reachable hand pays for jointly, within the piece
   caps and the placements legal right now.

`reachable_recipes` composes the three into `R(hand)`: every purchase
multiset reachable at all, each paired with one concrete, honest way to
reach it (`Recipe`) -- not necessarily the only way, just one a position can
be built from. Nothing here touches a `Game`, a `View`, or a policy; every
input is a plain vector or count the caller (`hexset.clients.netbot`) reads
off its own seat's information set.

Two deliberate simplifications, both documented where they bite:

* Bank stock on the *receiving* side of a conversion is never checked --
  only the seat's own ratio (`hexset.economy.trade_ratios`, which already
  folds in the base rate and this seat's ports) bounds what it can give.
  The bank holds `BANK_PER_RESOURCE` of everything and a single ask moves a
  handful of cards at most, so this only matters when a resource is
  already nearly drained -- see `bank_conversions`.
* A purchase multiset's legal-placement count (`purchase_multisets`'
  `legal`) is read once, before any of the multiset is placed. A road this
  same multiset places can open an edge -- or a settlement spot -- a plain
  scan of the board beforehand would not count; this is ignored on
  purpose, so `reachable_recipes` can only ever *undercount* what building
  the multiset for real would find legal, never overcount it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Sequence

from hexset.board.terrain import NUM_RESOURCES
from hexset.cards import DevCard
from hexset.economy import COSTS, Purchase

Hand = tuple[int, ...]
# (roads, settlements, cities, dev cards) -- always in this order.
Purchases = tuple[int, int, int, int]

YEAR_OF_PLENTY_PAIRS: tuple[tuple[int, int], ...] = tuple(
    combinations_with_replacement(range(NUM_RESOURCES), 2)
)


@dataclass(frozen=True)
class Recipe:
    """One honest way to reach `purchases` from a seat's current hand.

    `branch` is `None`, `"year_of_plenty"` or `"monopoly"` -- the
    development card played first, if any (`param` is its pair or resource);
    `hand` is the seat's exact hand after that branch and every bank trade
    taken from it, immediately before paying for `purchases`. Not
    necessarily the only path to `purchases` -- `reachable_recipes` keeps
    whichever it reaches first -- just one a position can be built from.
    """

    branch: str | None
    param: object
    hand: Hand
    purchases: Purchases


def branch_hands(
    hand: Sequence[int],
    *,
    dev_cards_held: Sequence[int],
    dev_card_played: bool,
    bank: Sequence[int],
    known_others: Sequence[Sequence[int]],
) -> tuple[tuple[str | None, object, Hand], ...]:
    """Every hand this seat could hold this turn before any bank trade or
    purchase: `(branch, param, hand)` triples, starting with
    `(None, None, tuple(hand))`.

    `dev_cards_held` is this seat's own *matured* holding
    (`state.dev_cards[seat]` -- never `new_dev_cards`: a card bought this
    turn is not playable this turn, `hexset.devcards.can_play`'s own rule).
    `dev_card_played` is the turn's one-card rule
    (`hexset.game.Game.dev_card_played`); true here means no branch runs at
    all, whatever the hand holds.

    Year of Plenty branches on each of the 15 unordered resource pairs
    (`YEAR_OF_PLENTY_PAIRS`) the bank can actually supply
    (`hexset.devcards.play_year_of_plenty`'s own legality, mirrored here so
    a branch this seat could never really play is never counted). Monopoly
    branches on each resource this seat could gain something from, summing
    only `known_others` -- the asking seat's own certified lower bound on
    every other seat, never a true hand it cannot read. A branch that would
    add nothing is never generated.
    """
    branches: list[tuple[str | None, object, Hand]] = [(None, None, tuple(hand))]
    if dev_card_played:
        return tuple(branches)
    if dev_cards_held[DevCard.YEAR_OF_PLENTY] > 0:
        for pair in YEAR_OF_PLENTY_PAIRS:
            cost = [0] * NUM_RESOURCES
            for r in pair:
                cost[r] += 1
            if all(bank[r] >= cost[r] for r in range(NUM_RESOURCES)):
                branches.append(
                    ("year_of_plenty", pair, tuple(n + d for n, d in zip(hand, cost)))
                )
    if dev_cards_held[DevCard.MONOPOLY] > 0:
        for r in range(NUM_RESOURCES):
            gain = sum(other[r] for other in known_others)
            if gain > 0:
                taken = list(hand)
                taken[r] += gain
                branches.append(("monopoly", r, tuple(taken)))
    return tuple(branches)


def bank_conversions(hand: Hand, ratios: Sequence[int]) -> frozenset[Hand]:
    """Every hand reachable from `hand` by zero or more bank trades, giving
    at `ratios[give]` per resource (`hexset.economy.trade_ratios`, which
    already folds in the 4:1 base rate, a 3:1 generic port and a 2:1
    resource port this seat holds) and receiving one card of anything else.

    A closed search over resource vectors: every ratio is at least 2:1, so
    each trade strictly shrinks the hand's total size, which is what makes
    the search terminate; `seen` is both the visited set and the
    memoisation the vector search needs, so no hand is expanded twice.
    """
    seen: set[Hand] = {hand}
    frontier = [hand]
    while frontier:
        nxt: list[Hand] = []
        for h in frontier:
            for give in range(NUM_RESOURCES):
                ratio = ratios[give]
                if h[give] < ratio:
                    continue
                for receive in range(NUM_RESOURCES):
                    if receive == give:
                        continue
                    new = list(h)
                    new[give] -= ratio
                    new[receive] += 1
                    new_hand = tuple(new)
                    if new_hand not in seen:
                        seen.add(new_hand)
                        nxt.append(new_hand)
        frontier = nxt
    return frozenset(seen)


def purchase_multisets(
    hand: Hand,
    *,
    room: Sequence[int],
    legal: Sequence[int],
    deck_size: int,
) -> frozenset[Purchases]:
    """Every `(roads, settlements, cities, dev_cards)` multiset `hand` pays
    for jointly -- resource cost only (`hexset.economy.COSTS`) -- within
    the piece caps this seat still has room for (`room`: `MAX_ROADS` /
    `MAX_SETTLEMENTS` / `MAX_CITIES` less what it already holds,
    `hexset.state`) and the placements legal for it right now (`legal`:
    `road_placeable` / `settlement_placeable` / `city_upgradeable` counts,
    read once before any of this multiset is placed -- see the module
    docstring for why that only ever undercounts). `deck_size` bounds
    `DEV_CARD` the same way a room does a built piece.

    Always includes `(0, 0, 0, 0)`: buying nothing is always reachable.
    """
    max_road = min(room[0], legal[0])
    max_settlement = min(room[1], legal[1])
    max_city = min(room[2], legal[2])
    max_dev = deck_size

    road_cost = COSTS[Purchase.ROAD]
    settlement_cost = COSTS[Purchase.SETTLEMENT]
    city_cost = COSTS[Purchase.CITY]
    dev_cost = COSTS[Purchase.DEV_CARD]

    out: set[Purchases] = set()
    for nr in range(max_road + 1):
        after_r = [h - nr * c for h, c in zip(hand, road_cost)]
        if any(v < 0 for v in after_r):
            break
        for ns in range(max_settlement + 1):
            after_s = [h - ns * c for h, c in zip(after_r, settlement_cost)]
            if any(v < 0 for v in after_s):
                break
            for nc in range(max_city + 1):
                after_c = [h - nc * c for h, c in zip(after_s, city_cost)]
                if any(v < 0 for v in after_c):
                    break
                for nd in range(max_dev + 1):
                    after_d = [h - nd * c for h, c in zip(after_c, dev_cost)]
                    if any(v < 0 for v in after_d):
                        break
                    out.add((nr, ns, nc, nd))
    return frozenset(out)


def reachable_recipes(
    hand: Sequence[int],
    *,
    dev_cards_held: Sequence[int],
    dev_card_played: bool,
    bank: Sequence[int],
    known_others: Sequence[Sequence[int]],
    ratios: Sequence[int],
    room: Sequence[int],
    legal: Sequence[int],
    deck_size: int,
) -> dict[Purchases, Recipe]:
    """`R(hand)`, with one buildable `Recipe` kept for every multiset in it.

    The union, over every development-card branch this turn allows
    (`branch_hands`) and every bank-trade chain reachable from each
    (`bank_conversions`), of every jointly payable purchase
    (`purchase_multisets`). Free of any policy: this only says what is
    reachable and one honest way to reach it; `hexset.clients.netbot`
    decides what to do with that.
    """
    recipes: dict[Purchases, Recipe] = {}
    for branch, param, branch_hand in branch_hands(
        hand,
        dev_cards_held=dev_cards_held,
        dev_card_played=dev_card_played,
        bank=bank,
        known_others=known_others,
    ):
        for reached in bank_conversions(branch_hand, ratios):
            for multiset in purchase_multisets(
                reached, room=room, legal=legal, deck_size=deck_size
            ):
                if multiset not in recipes:
                    recipes[multiset] = Recipe(branch, param, reached, multiset)
    return recipes


def maximal(multisets: Sequence[Purchases]) -> list[Purchases]:
    """The members of `multisets` that are not a sub-multiset of another
    member -- the only ones worth building and valuing a position for,
    since a hand that affords the bigger one affords the smaller for free.
    """
    items = list(multisets)
    return [
        m
        for m in items
        if not any(m != other and all(m[i] <= other[i] for i in range(len(m))) for other in items)
    ]
