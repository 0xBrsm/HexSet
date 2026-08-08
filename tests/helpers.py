from __future__ import annotations

from catan.board.board import Board, make_board
from catan.board.maps import MINI_LAYOUT
from catan.board.terrain import Terrain
from catan.board.topology import build as build_topology

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


def give(state, player: int, resource: int, count: int = 1) -> None:
    """Seed a hand from the bank, so tests never conjure resources from nowhere."""
    if state.bank[resource] < count:
        raise AssertionError(f"bank cannot supply {count} of resource {resource}")
    state.bank[resource] -= count
    state.hands[player][resource] += count


def clear_hand(state, player: int) -> None:
    """Return a hand to the bank, so a test can set up an exact holding."""
    for resource, count in enumerate(state.hands[player]):
        state.bank[resource] += count
        state.hands[player][resource] = 0


def independent_vertices(board: Board, count: int) -> list[int]:
    """Vertices no two of which are adjacent, so all are settleable together."""
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
    """Vertices that produce, far enough apart to satisfy the distance rule."""
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
