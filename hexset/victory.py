# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from .cards import DevCard
from .devcards import holdings
from .roads import MIN_LONGEST_ROAD, longest_road, road_lengths
from .state import NO_OWNER, GameState

__all__ = [
    "MIN_LARGEST_ARMY",
    "LONGEST_ROAD_VP",
    "LARGEST_ARMY_VP",
    "WINNING_POINTS",
    "StaleRoadLengths",
    "update_longest_road",
    "update_largest_army",
    "building_points",
    "card_points",
    "award_points",
    "victory_points",
    "public_victory_points",
    "relative_points",
]


MIN_LARGEST_ARMY = 3
LONGEST_ROAD_VP = 2
LARGEST_ARMY_VP = 2
# Standard-game default; the live rule is `state.rules.winning_points`.
WINNING_POINTS = 10


def _award(counts: list[int], holder: int, minimum: int) -> int:
    """Who holds a "most of X" card, given who holds it now. A challenger must
    beat the holder outright; on a tie behind a fallen holder it leaves play."""
    best = max(counts)
    if best < minimum:
        return NO_OWNER
    if holder != NO_OWNER and counts[holder] == best:
        return holder
    leaders = [p for p, n in enumerate(counts) if n == best]
    return leaders[0] if len(leaders) == 1 else NO_OWNER


class StaleRoadLengths(ValueError):
    """`state.road_lengths` cannot be the truth about `state.edge_owner`."""


def _check_cache(state: GameState, builder: int | None = None) -> None:
    """Refuse an incremental update on a cache that was never filled: a roaded
    seat cached at zero is impossible, bar `builder`'s first road."""
    cached = state.road_lengths
    if len(cached) != state.num_players:
        raise StaleRoadLengths(
            f"road_lengths has {len(cached)} entries for {state.num_players} "
            "seats: this state was built without the cache -- call "
            "update_longest_road(state) once, from scratch, before "
            "updating it incrementally"
        )
    roaded = {owner for owner in state.edge_owner if owner != NO_OWNER}
    unfilled = sorted(
        seat for seat in roaded if cached[seat] == 0 and seat != builder
    )
    if unfilled:
        raise StaleRoadLengths(
            f"road_lengths reads 0 for seat(s) {unfilled}, which own roads: "
            "this state's roads were placed around place_road and the cache "
            "never filled -- call update_longest_road(state) once, from "
            "scratch, before updating it incrementally"
        )


def update_longest_road(
    state: GameState,
    *,
    road_owner: int | None = None,
    settlement_vertex: int | None = None,
    settlement_owner: int | None = None,
) -> int:
    """Refresh `state.road_lengths` and award the card from it.

    With `road_owner`, only that seat is recomputed (roads are edge-disjoint);
    with `settlement_vertex`, only other seats owning a road touching it, a
    building being able to *cut* a route but never block its own builder; with
    neither, every seat from scratch. The incremental paths raise
    `StaleRoadLengths` rather than award off a cache that cannot be right.
    """
    if road_owner is not None or settlement_vertex is not None:
        _check_cache(state, builder=road_owner)
    if road_owner is not None:
        state.road_lengths[road_owner] = longest_road(state, road_owner)
    elif settlement_vertex is not None:
        topology = state.board.topology
        affected = {
            state.edge_owner[e]
            for e in topology.vertex_edges[settlement_vertex]
            if state.edge_owner[e] != NO_OWNER
            and state.edge_owner[e] != settlement_owner
        }
        for seat in affected:
            state.road_lengths[seat] = longest_road(state, seat)
    else:
        state.road_lengths[:] = road_lengths(state)

    state.longest_road_holder = _award(
        state.road_lengths, state.longest_road_holder, MIN_LONGEST_ROAD
    )
    return state.longest_road_holder


def update_largest_army(state: GameState) -> int:
    """Award Largest Army from `state.knights_played` and return its holder,
    `NO_OWNER` when nobody qualifies. A challenger must beat the holder
    outright."""
    state.largest_army_holder = _award(
        state.knights_played, state.largest_army_holder, MIN_LARGEST_ARMY
    )
    return state.largest_army_holder


def building_points(state: GameState, player: int) -> int:
    """Points from `player`'s buildings: one per settlement, two per city."""
    # `vertex_building` holds 1 for a settlement and 2 for a city, which is
    # also what each is worth in points.
    return sum(
        state.vertex_building[v]
        for v, owner in enumerate(state.vertex_owner)
        if owner == player
    )


def card_points(state: GameState, player: int) -> int:
    """Victory point cards score the moment they are drawn, so they can win.
    An identity read: on an observed state a hidden holding raises
    `state.HiddenRead`, and so does `victory_points` through it."""
    return holdings(state, player)[DevCard.VICTORY_POINT]


def award_points(state: GameState, player: int) -> int:
    """Points from the Longest Road and Largest Army cards `player` holds."""
    points = 0
    if state.longest_road_holder == player:
        points += LONGEST_ROAD_VP
    if state.largest_army_holder == player:
        points += LARGEST_ARMY_VP
    return points


def victory_points(state: GameState, player: int) -> int:
    """`player`'s full score: buildings, victory point cards and award cards.
    An identity read: a hidden holding raises `state.HiddenRead`;
    `public_victory_points` is what opponents see."""
    return (
        building_points(state, player)
        + card_points(state, player)
        + award_points(state, player)
    )


def public_victory_points(state: GameState, player: int) -> int:
    """What opponents can see. Victory point cards stay hidden until they win."""
    return building_points(state, player) + award_points(state, player)


def relative_points(
    points: tuple[int, ...], *, winning_points: int = WINNING_POINTS
) -> tuple[float, ...]:
    """Each seat's terminal points less the mean of the others, over the points
    the game was played to.
    Exactly zero-sum, so an action that lifts every seat equally earns nothing.

    `winning_points` is the scale, not a threshold: a 15-point game covers half
    again the range a 10-point one does, and dividing both by 10 would hand the
    longer game a reward half again as large for the same finish.

    **Do not discount this.** Roughly half of terminal values are negative, and
    γ < 1 makes a negative terminal cheaper the later it arrives, which pays a
    losing policy to stall."""
    seats = len(points)
    if seats < 2:
        raise ValueError("a relative reward needs at least two seats")
    total = sum(points)
    return tuple(
        (own - (total - own) / (seats - 1)) / winning_points for own in points
    )
