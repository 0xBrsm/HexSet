from __future__ import annotations

from .state import NO_OWNER, GameState

MIN_LONGEST_ROAD = 5


def longest_road(state: GameState, player: int) -> int:
    """Length of the player's longest continuous route.

    A route may not reuse a road segment but may revisit a junction, so this is
    a longest trail rather than a longest simple path — loops count in full. An
    opponent's building breaks a route at that junction: roads either side of it
    still count, but a route cannot pass through.

    Exhaustive depth-first search. A player owns at most 15 roads, so the
    exponential worst case never bites, and the used-set is a bitmask to keep
    the constant factor low.
    """
    topology = state.board.topology
    owned = [e for e in range(topology.num_edges) if state.edge_owner[e] == player]
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
    return [longest_road(state, p) for p in range(state.num_players)]
