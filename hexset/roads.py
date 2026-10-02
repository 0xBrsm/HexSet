# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from .state import NO_OWNER, GameState

__all__ = [
    "MIN_LONGEST_ROAD",
    "longest_road",
    "road_lengths",
]


MIN_LONGEST_ROAD = 5


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
