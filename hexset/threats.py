# SPDX-License-Identifier: GPL-3.0-only
"""What one rival build on its next turn takes from a seat: steals and blocks.

`threats(state, victim)` lists every single build a rival can make on its next
turn that costs `victim` something on the board, with what it costs and the
cards the rival still lacks for it:

- a **road** (`placeable_roads`) or a **settlement** (`settlement_placeable`,
  joined to the rival's roads), each read by placing it on a copy of the
  position and asking again: does `victim` still hold Longest Road
  (`victory.update_longest_road`), how long is its longest route, how many
  roads does it need to take the card (`roads.roads_to_take`), which free
  spots can it still settle at its road ends or one new road beyond them;
- a **knight**, where the rival holds a development card it could play by its
  turn: does it take Largest Army from `victim`, or put the card further
  out of `victim`'s reach. What the card is stays hidden; a knight threat is
  one the rival *may* hold.

A build that takes an award from `victim` is a **steal**; one that costs it
anything else (route length, roads to Longest Road, spots, knights to Largest
Army) is a **block**. Builds that cost `victim` nothing are left out.

`hands` is whatever the caller holds about each seat's resources: the true
hands by default, or a seat's certified lower bounds (`View.known`) to ask
what that seat can be sure a rival can pay for. Nothing else here reads a
hidden card. Trades, bank and port exchanges, and the rival's production
before its turn are not counted: `lacks` is what it is short now.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .devcards import dev_count
from .economy import COSTS, Purchase
from .roads import MIN_LONGEST_ROAD, longest_road, roads_to_take
from .state import (
    MAX_ROADS, MAX_SETTLEMENTS, NO_OWNER, GameState, copy_state, place_road, place_settlement, placeable_roads,
    road_count, settlement_count, settlement_placeable,
)
from .victory import MIN_LARGEST_ARMY, update_largest_army, update_longest_road

__all__ = ["Cost", "Threat", "spots", "threats"]

STEAL, BLOCK = "steal", "block"
LONGEST_ROAD, LARGEST_ARMY = "longest_road", "largest_army"


@dataclass(frozen=True)
class Cost:
    """What one build takes from `victim`, each as a loss (zero for none).

    `award` names the award it takes (`"longest_road"`, `"largest_army"`) or
    is `None`; `route` is how much shorter `victim`'s longest route gets;
    `roads` how many more roads it needs to take Longest Road (where it does
    not hold it); `knights` how many more knights it needs for Largest Army
    (where it does not hold it); `spots` the free spots it can no longer
    settle within one new road."""

    award: str | None = None
    route: int = 0
    roads: int = 0
    knights: int = 0
    spots: tuple[int, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.award or self.route or self.roads or self.knights or self.spots)


@dataclass(frozen=True)
class Threat:
    """`rival`'s build `build` (`"road"`, `"settlement"`, `"knight"`) at `at`
    (an edge, a vertex, `-1` for a knight) and what it costs `victim`.
    `kind` is `"steal"` where the cost takes an award, else `"block"`.
    `lacks` is the cards `rival` is short of the build's cost, per resource,
    by the hands the caller gave (all zero for a knight, which costs a card
    already held)."""

    kind: str
    victim: int
    rival: int
    build: str
    at: int
    cost: Cost
    lacks: tuple[int, ...]

    @property
    def short(self) -> int:
        """How many cards `rival` lacks for the build."""
        return sum(self.lacks)


def spots(state: GameState, seat: int) -> frozenset[int]:
    """The free vertices `seat` could settle at its road ends now, or at the
    far end of one new road from them: the road on a free edge out of a vertex
    its network reaches and no other seat's building stops."""
    topology = state.board.topology
    reach = {v for v, owner in enumerate(state.vertex_owner) if owner == seat}
    for e, owner in enumerate(state.edge_owner):
        if owner == seat:
            reach.update(v for v in topology.edges[e] if state.vertex_owner[v] in (NO_OWNER, seat))
    out = {v for v in reach if settlement_placeable(state, seat, v)}
    for v in reach:
        for e in topology.vertex_edges[v]:
            if state.edge_owner[e] != NO_OWNER:
                continue
            a, b = topology.edges[e]
            u = b if a == v else a
            if settlement_placeable(state, seat, u, connected=False):
                out.add(u)
    return frozenset(out)


def _lacks(hand: Sequence[int], cost: Sequence[int]) -> tuple[int, ...]:
    return tuple(max(0, c - int(h)) for h, c in zip(hand, cost))


def _knights_to_take(state: GameState, seat: int) -> int:
    """Knights `seat` still has to play to take Largest Army, outright ahead of
    every other seat and at least `MIN_LARGEST_ARMY`; 0 where it holds it."""
    played = state.knights_played
    if seat == state.largest_army_holder:
        return 0
    best = max(played[s] for s in range(state.num_players) if s != seat)
    return max(MIN_LARGEST_ARMY, best + 1) - played[seat]


class _Victim:
    """`victim`'s standing at a position, read once to compare builds against."""

    __slots__ = ("holds_road", "route", "roads", "spots")

    def __init__(self, state: GameState, victim: int) -> None:
        self.holds_road = state.longest_road_holder == victim
        self.route = longest_road(state, victim)
        self.roads = 0 if self.holds_road else roads_to_take(state, victim)
        self.spots = spots(state, victim)

    def cost(self, before: "_Victim") -> Cost:
        return Cost(
            award=LONGEST_ROAD if before.holds_road and not self.holds_road else None,
            route=max(0, before.route - self.route),
            roads=0 if before.holds_road else max(0, self.roads - before.roads),
            spots=tuple(sorted(before.spots - self.spots)),
        )


def _after(state: GameState, rival: int, build: str, at: int) -> GameState:
    s = copy_state(state)
    if build == "road":
        place_road(s, rival, at)
        update_longest_road(s, road_owner=rival)
    else:
        place_settlement(s, rival, at)
        update_longest_road(s, settlement_vertex=at, settlement_owner=rival)
    return s


def threats(state: GameState, victim: int, hands: Sequence[Sequence[int]] | None = None) -> list[Threat]:
    """Every rival's single next-turn build that steals from or blocks
    `victim`, rivals in seat order, then roads by edge, settlements by
    vertex, knights last. `hands` defaults to the state's own (concrete)
    hands; pass a seat's `View.known` for what it can certify."""
    if hands is None:
        hands = state.hands
    before = _Victim(state, victim)
    out: list[Threat] = []
    road_cost, settle_cost = COSTS[Purchase.ROAD], COSTS[Purchase.SETTLEMENT]
    for rival in range(state.num_players):
        if rival == victim:
            continue
        builds: list[tuple[str, int, tuple[int, ...]]] = []
        if road_count(state, rival) < MAX_ROADS:
            lacks = _lacks(hands[rival], road_cost)
            builds += [("road", e, lacks) for e in placeable_roads(state, rival)]
        if settlement_count(state, rival) < MAX_SETTLEMENTS:
            lacks = _lacks(hands[rival], settle_cost)
            builds += [("settlement", v, lacks) for v in range(state.board.topology.num_vertices)
                       if settlement_placeable(state, rival, v)]
        for build, at, lacks in builds:
            cost = _Victim(_after(state, rival, build, at), victim).cost(before)
            if cost:
                out.append(Threat(STEAL if cost.award else BLOCK, victim, rival, build, at, cost, lacks))
        knight = _knight(state, victim, rival)
        if knight:
            out.append(Threat(STEAL if knight.award else BLOCK, victim, rival, "knight", -1, knight,
                              (0,) * len(road_cost)))
    return out


def _knight(state: GameState, victim: int, rival: int) -> Cost:
    """What one more knight of `rival`'s takes from `victim`, where `rival`
    holds a development card to play by its turn."""
    if dev_count(state, rival) == 0:
        return Cost()
    played = list(state.knights_played)
    holder = state.largest_army_holder
    if holder == victim:
        if played[rival] + 1 > played[victim] and played[rival] + 1 >= MIN_LARGEST_ARMY:
            return Cost(award=LARGEST_ARMY)
        return Cost()
    s = copy_state(state)
    s.knights_played[rival] += 1
    update_largest_army(s)
    return Cost(knights=max(0, _knights_to_take(s, victim) - _knights_to_take(state, victim)))
