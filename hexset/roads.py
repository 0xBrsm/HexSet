# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from functools import lru_cache

from .state import MAX_ROADS, NO_OWNER, GameState, road_count

__all__ = [
    "MIN_LONGEST_ROAD",
    "ROADS_TO_TAKE_CAP",
    "longest_road",
    "longest_with_new_roads",
    "reach_all",
    "road_lengths",
    "roads_to_take",
    "roads_to_take_all",
]


MIN_LONGEST_ROAD = 5
# `roads_to_take` reads "this many or more": a seat that far from Longest Road
# is no threat to it yet, and the exact search below stops one short of it.
ROADS_TO_TAKE_CAP = 4


def longest_road(state: GameState, player: int, extra_edge: int | None = None) -> int:
    """Length of the player's longest continuous route: a longest trail, not
    a longest simple path, since a route may revisit a junction but not reuse
    a segment. An opponent's building breaks a route at that junction.
    Exhaustive depth-first search, exponential in the at most 15 roads owned.

    `extra_edge` counts one more road as the player's -- the length the route
    would have if that edge were built -- without touching the state."""
    topology = state.board.topology
    owned = [e for e in range(topology.num_edges) if state.edge_owner[e] == player]
    if extra_edge is not None and extra_edge not in owned:
        owned.append(extra_edge)
    if not owned:
        return 0

    local = {e: i for i, e in enumerate(owned)}
    junctions: dict[int, list[tuple[int, int]]] = {}
    for e in owned:
        a, b = topology.edges[e]
        junctions.setdefault(a, []).append((local[e], b))
        junctions.setdefault(b, []).append((local[e], a))

    def passable(v: int) -> bool:
        owner = state.vertex_owner[v]
        return owner == NO_OWNER or owner == player

    def extend(v: int, used: int) -> int:
        best = 0
        for idx, w in junctions[v]:
            bit = 1 << idx
            if used & bit:
                continue
            length = 1 + (extend(w, used | bit) if passable(w) else 0)
            if length > best:
                best = length
        return best

    return max(extend(v, 0) for v in junctions)


def road_lengths(state: GameState) -> list[int]:
    """Every seat's `longest_road`, in seat order."""
    return [longest_road(state, p) for p in range(state.num_players)]


def longest_with_new_roads(state: GameState, seat: int, roads: int,
                           paths: list | None = None) -> list[int]:
    """`longest[j]`: the longest route `seat` could have with at most `j`
    new roads, `j` up to `roads` -- one search over its own roads and the
    empty edges within reach, a new road costing one of the budget.
    `longest_road`'s rules: a route is a trail, an opponent's building
    ends it, and new roads must join the seat's network (a route that
    never touches it is not one it could build); `longest[0]` is
    `longest_road`. Exponential in the edges within reach, as
    `longest_road` is in the roads owned. A `paths` list is filled with the
    new edges of each `longest[j]`'s route."""
    topology = state.board.topology
    edges = topology.edges
    owner_of = state.vertex_owner
    edge_owner = state.edge_owner
    own = [e for e in range(topology.num_edges) if edge_owner[e] == seat]

    def passable(v: int) -> bool:
        return owner_of[v] == NO_OWNER or owner_of[v] == seat

    joins = {v for e in own for v in edges[e] if owner_of[v] == NO_OWNER}
    joins |= {v for v, o in enumerate(owner_of) if o == seat}
    # Empty edges a route of `roads` new ones could use: out from the
    # network through passable junctions, `roads` steps at most.
    near: set[int] = set()
    frontier = set(joins)
    for _ in range(roads):
        grown: set[int] = set()
        for v in frontier:
            for e in topology.vertex_edges[v]:
                if edge_owner[e] == NO_OWNER and e not in near:
                    near.add(e)
                    for w in edges[e]:
                        if passable(w):
                            grown.add(w)
        frontier = grown
    usable = own + sorted(near)
    cost = [0] * len(own) + [1] * len(near)
    # Per edge end: its bit, the far vertex, the road's cost, whether the walk
    # goes on past the far vertex, and whether the road joins the network.
    adjacent: dict[int, list[tuple[int, int, int, bool, bool]]] = {}
    for i, e in enumerate(usable):
        a, b = edges[e]
        adjacent.setdefault(a, []).append((1 << i, b, cost[i], passable(b), cost[i] == 0 or b in joins))
        adjacent.setdefault(b, []).append((1 << i, a, cost[i], passable(a), cost[i] == 0 or a in joins))
    best = [0] * (roads + 1)
    masks = [0] * (roads + 1)

    def walk(v: int, used: int, spent: int, length: int, touched: bool) -> None:
        if touched and length > best[spent]:
            best[spent] = length
            masks[spent] = used
        for bit, w, c, onward, joined in adjacent[v]:
            if used & bit or spent + c > roads:
                continue
            reached = touched or joined
            if onward:
                walk(w, used | bit, spent + c, length + 1, reached)
            elif reached and length + 1 > best[spent + c]:
                best[spent + c] = length + 1
                masks[spent + c] = used | bit

    for v in adjacent:
        walk(v, 0, 0, 0, v in joins)
    for j in range(1, roads + 1):
        if best[j - 1] >= best[j]:
            best[j], masks[j] = best[j - 1], masks[j - 1]
    if paths is not None:
        paths[:] = [[usable[i] for i in range(len(own), len(usable)) if mask >> i & 1]
                    for mask in masks]
    return best


def roads_to_take(state: GameState, seat: int) -> int:
    """How many new roads `seat` needs to take Longest Road: 0 where it holds
    the card; otherwise the fewest after which its longest route
    (`longest_with_new_roads`: joins count, an opponent's building breaks a
    route, new roads must join its network) is longer than every other
    seat's and at least `MIN_LONGEST_ROAD`, within the road pieces it has
    left; `ROADS_TO_TAKE_CAP` where that is more than
    `ROADS_TO_TAKE_CAP - 1`, or out of reach.

    Read off `state.road_lengths`, the engine's cache of every seat's
    `longest_road`, for the length to beat. Cached on the position's roads
    and buildings, the only things it reads: a whole game asks the same
    question again until somebody builds. Only `seat` is searched; the
    search is the one `roads_to_take_all` caches for each seat."""
    if seat == state.longest_road_holder:
        return 0
    lengths = state.road_lengths or road_lengths(state)
    beat = max(MIN_LONGEST_ROAD, 1 + max((n for s, n in enumerate(lengths) if s != seat), default=0))
    budget = min(ROADS_TO_TAKE_CAP - 1, MAX_ROADS - road_count(state, seat))
    return _roads_to_take(state.board.topology, seat, beat, budget,
                          tuple(state.edge_owner), tuple(state.vertex_owner))


def roads_to_take_all(state: GameState) -> tuple[int, ...]:
    """Every seat's `roads_to_take`, in seat order: one cache lookup for the
    position, one search per seat only where the roads or buildings changed."""
    lengths = state.road_lengths or road_lengths(state)
    counts = tuple(road_count(state, s) for s in range(state.num_players))
    return _all_roads_to_take(state.board.topology, state.longest_road_holder, tuple(lengths), counts,
                              tuple(state.edge_owner), tuple(state.vertex_owner))


@lru_cache(maxsize=1 << 14)
def _all_roads_to_take(topology, holder: int, lengths: tuple[int, ...], counts: tuple[int, ...],
                       edge_owner: tuple[int, ...], vertex_owner: tuple[int, ...]) -> tuple[int, ...]:
    out = []
    for seat in range(len(lengths)):
        if seat == holder:
            out.append(0)
            continue
        beat = max(MIN_LONGEST_ROAD, 1 + max((n for s, n in enumerate(lengths) if s != seat), default=0))
        budget = min(ROADS_TO_TAKE_CAP - 1, MAX_ROADS - counts[seat])
        out.append(_roads_to_take(topology, seat, beat, budget, edge_owner, vertex_owner))
    return tuple(out)


@lru_cache(maxsize=1 << 16)
def _roads_to_take(topology, seat: int, beat: int, budget: int,
                   edge_owner: tuple[int, ...], vertex_owner: tuple[int, ...]) -> int:
    """`roads_to_take` past the holder check, on a key of everything it reads
    (`Topology` hashes on its index tuples). A route is no longer than the
    roads in it, so `k` new roads cannot reach `beat` while the seat owns
    fewer than `beat - k`; the search starts at the first `k` that could,
    and deepens one road at a time, stopping at the first that does."""
    owned = edge_owner.count(seat)
    probe = _Roads(topology, edge_owner, vertex_owner)
    for k in range(max(1, beat - owned), budget + 1):
        if longest_with_new_roads(probe, seat, k)[k] >= beat:
            return k
    return ROADS_TO_TAKE_CAP


def reach_all(state: GameState) -> tuple[tuple[int, ...], ...]:
    """Per seat, per vertex, the fewest new roads that seat needs for its
    network to touch the vertex: 0 on its network, `-1` where it cannot.

    A seat's network is its buildings and the ends of its roads. A new road
    goes on an edge holding no road, and a route may not pass through a
    vertex holding an opponent's building, though it may end at one; nor can
    it go on from a road's end at such a vertex. Unreachable means no route,
    or one longer than the seat's road pieces left. Every vertex is read,
    legal to settle or not. A seat with no building yet reaches nothing.
    Cached on the position's roads and buildings, the only things it reads."""
    return _reach_all(state.board.topology, tuple(state.edge_owner), tuple(state.vertex_owner),
                      state.num_players)


@lru_cache(maxsize=1 << 14)
def _reach_all(topology, edge_owner: tuple[int, ...], vertex_owner: tuple[int, ...],
               players: int) -> tuple[tuple[int, ...], ...]:
    edges = topology.edges
    out = []
    for seat in range(players):
        if seat not in vertex_owner:
            out.append((-1,) * topology.num_vertices)
            continue
        left = MAX_ROADS - sum(1 for o in edge_owner if o == seat)
        dist = [-1] * topology.num_vertices
        frontier = [v for v, o in enumerate(vertex_owner) if o == seat]
        for e, o in enumerate(edge_owner):
            if o == seat:
                frontier.extend(edges[e])
        for v in frontier:
            dist[v] = 0
        steps = 0
        while frontier and steps < left:
            steps += 1
            grown = []
            for v in frontier:
                if vertex_owner[v] not in (NO_OWNER, seat):
                    continue  # an opponent's building: reached, not passed
                for e in topology.vertex_edges[v]:
                    if edge_owner[e] != NO_OWNER:
                        continue
                    a, b = edges[e]
                    w = b if a == v else a
                    if dist[w] < 0:
                        dist[w] = steps
                        grown.append(w)
            frontier = grown
        out.append(tuple(dist))
    return tuple(out)


class _Roads:
    """The three fields `longest_with_new_roads` reads, rebuilt from a cache
    key, so the cached search never holds a live state."""

    __slots__ = ("board", "edge_owner", "vertex_owner")

    def __init__(self, topology, edge_owner, vertex_owner):
        self.board = _Board(topology)
        self.edge_owner = edge_owner
        self.vertex_owner = vertex_owner


class _Board:
    __slots__ = ("topology",)

    def __init__(self, topology):
        self.topology = topology
