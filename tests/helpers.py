# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from hexset.board.board import Board, make_board
from hexset.board.maps import MINI_LAYOUT
from hexset.board.terrain import Terrain
from hexset.board.topology import build as build_topology
from hexset.roads import longest_road
from hexset.state import NO_OWNER, can_place_road, place_road

ROLL = 4
DESERT_HEX = 0


def mini_board(*, gold: bool = False) -> Board:
    """Desert on hex 0 so the robber starts clear of every producing hex."""
    topology = build_topology(MINI_LAYOUT)
    n = topology.num_hexes
    producer = Terrain.GOLD if gold else Terrain.FOREST
    terrain = (Terrain.DESERT,) + (producer,) * (n - 1)
    tokens = (0,) + (ROLL,) * (n - 1)
    return make_board(topology, terrain, tokens)


def a_vertex_touching(board: Board, count: int, *, exclude_hex: int = DESERT_HEX) -> int:
    for v, hexes in enumerate(board.topology.vertex_hexes):
        if len(hexes) == count and exclude_hex not in hexes:
            return v
    raise AssertionError(f"no vertex touching exactly {count} hexes")


def victim_on(game, target: int) -> int | None:
    """The seat a robber move to `target` robs: the lowest it may, or `None`
    where nobody on the hex holds a card -- the only case `None` is legal."""
    from hexset.robber import victims

    return next(iter(victims(game._state, target, game.current_player)), None)


def give(state, player: int, resource: int, count: int = 1) -> None:
    if state.bank[resource] < count:
        raise AssertionError(f"bank cannot supply {count} of resource {resource}")
    state.bank[resource] -= count
    state.hands[player][resource] += count


def deal(game, player: int, resource: int, count: int = 1) -> None:
    """`give` through the public record: the cards land in the hand and the
    ledger certifies them, as a roll's distribution would. Use this wherever
    a test seats a trading bot and means the table to know the cards: a
    counter asks only for what the ledger says the actor holds, and a gate
    prices the other seats from the record."""
    give(game._state, player, resource, count)
    game.ledger.receive(player, int(resource), count)


def clear_hand(state, player: int) -> None:
    for resource, count in enumerate(state.hands[player]):
        state.bank[resource] += count
        state.hands[player][resource] = 0


def independent_vertices(board: Board, count: int) -> list[int]:
    """No two adjacent, so all are settleable together."""
    topology = board.topology
    chosen: list[int] = []
    taken: set[int] = set()
    for v in range(topology.num_vertices):
        if v in taken:
            continue
        chosen.append(v)
        taken.add(v)
        taken.update(topology.vertex_neighbors[v])
        if len(chosen) == count:
            return chosen
    raise AssertionError(f"could not find {count} independent vertices")


def independent_producers(
    board: Board, count: int, *, exclude_hex: int = DESERT_HEX
) -> list[int]:
    """Producing vertices, far enough apart for the distance rule."""
    topology = board.topology
    chosen: list[int] = []
    taken: set[int] = set()
    for v, hexes in enumerate(topology.vertex_hexes):
        if v in taken or not any(h != exclude_hex for h in hexes):
            continue
        chosen.append(v)
        taken.add(v)
        taken.update(topology.vertex_neighbors[v])
        if len(chosen) == count:
            return chosen
    raise AssertionError(f"could not find {count} independent producing vertices")


def grow_road(state, seat: int, length: int) -> None:
    """Add roads to `seat` until its longest route is `length` and one more
    road can still lengthen it, keeping the cached lengths true and nobody
    holding Longest Road. Depth-first: a route can run into the coast."""
    edges = range(state.board.topology.num_edges)

    def grows() -> bool:
        before = longest_road(state, seat)
        for e in edges:
            if can_place_road(state, seat, e):
                place_road(state, seat, e)
                longer = longest_road(state, seat) > before
                state.edge_owner[e] = NO_OWNER
                if longer:
                    return True
        return False

    def extend() -> bool:
        before = longest_road(state, seat)
        if before >= length:
            return grows()
        for e in edges:
            if not can_place_road(state, seat, e):
                continue
            place_road(state, seat, e)
            if longest_road(state, seat) > before and extend():
                return True
            state.edge_owner[e] = NO_OWNER
        return False

    assert extend(), "no route of that length"
    state.road_lengths[:] = [longest_road(state, p) for p in range(state.num_players)]
    state.longest_road_holder = NO_OWNER


def split_route(state, seat: int, stretch: int, length: int) -> int:
    """Take one road out of `seat`'s route so no stretch left is longer than
    `stretch` and that road, rebuilt, makes a route of `length` or more; the
    cached lengths true. The road taken out, which joins the stretches."""
    for e in [e for e, o in enumerate(state.edge_owner) if o == seat]:
        state.edge_owner[e] = NO_OWNER
        if longest_road(state, seat) <= stretch and longest_road(state, seat, extra_edge=e) >= length:
            state.road_lengths[:] = [longest_road(state, p) for p in range(state.num_players)]
            return e
        state.edge_owner[e] = seat
    raise AssertionError("no road splits the route into two short stretches")
