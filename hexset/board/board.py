# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .coords import DIRECTIONS, ORIGIN, Hex
from .maps import BASE_LAYOUT
from .ports import Port, base_port_bag, place_ports
from .terrain import BEARS_TOKEN, Terrain
from .topology import Topology
from .topology import build as build_topology

__all__ = [
    "BASE_TERRAIN",
    "BASE_TOKENS",
    "RED_TOKENS",
    "pips",
    "Board",
    "make_board",
    "random_base_board",
    "SPIRAL_TOKENS",
    "spiral_base_board",
    "BOARD_MODES",
    "base_board",
]


MIN_ROLL, MAX_ROLL = 2, 12

BASE_TERRAIN: tuple[Terrain, ...] = (
    (Terrain.FOREST,) * 4
    + (Terrain.HILLS,) * 3
    + (Terrain.PASTURE,) * 4
    + (Terrain.FIELDS,) * 4
    + (Terrain.MOUNTAINS,) * 3
    + (Terrain.DESERT,)
)

BASE_TOKENS: tuple[int, ...] = (2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12)

RED_TOKENS: frozenset[int] = frozenset({6, 8})

# The number discs in their lettered order, A to R.
SPIRAL_TOKENS: tuple[int, ...] = (5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11)


def pips(token: int) -> int:
    """Ways to roll `token` with two dice: the tile's production weight."""
    return 0 if not token else 6 - abs(7 - token)


@dataclass(frozen=True)
class Board:
    """Static setup: what is on each hex. Occupancy lives in GameState."""

    topology: Topology
    terrain: tuple[Terrain, ...]
    tokens: tuple[int, ...]
    hexes_by_roll: tuple[tuple[int, ...], ...]
    ports: tuple[Port, ...] = ()

    @property
    def num_hexes(self) -> int:
        return self.topology.num_hexes

    def desert_hexes(self) -> tuple[int, ...]:
        return tuple(
            h for h, t in enumerate(self.terrain) if t is Terrain.DESERT
        )


def make_board(
    topology: Topology,
    terrain: tuple[Terrain, ...],
    tokens: tuple[int, ...],
    ports: tuple[Port, ...] = (),
) -> Board:
    """A `Board` of per-hex `terrain` and `tokens` (0 for none) on `topology`.

    Raises `ValueError` when either tuple's length is not the hex count, a
    token sits on terrain that bears none, or a token is 7 or outside 2-12."""
    n = topology.num_hexes
    if len(terrain) != n or len(tokens) != n:
        raise ValueError(f"expected {n} terrain and token entries per hex")
    for h, (t, token) in enumerate(zip(terrain, tokens)):
        if token and t not in BEARS_TOKEN:
            raise ValueError(f"hex {h} is {t.name} and cannot bear token {token}")
        if token and not MIN_ROLL <= token <= MAX_ROLL:
            raise ValueError(f"hex {h} has out-of-range token {token}")
        if token == 7:
            raise ValueError(f"hex {h} has token 7")

    by_roll: list[list[int]] = [[] for _ in range(MAX_ROLL + 1)]
    for h, token in enumerate(tokens):
        if token:
            by_roll[token].append(h)

    return Board(
        topology=topology,
        terrain=terrain,
        tokens=tokens,
        hexes_by_roll=tuple(tuple(hs) for hs in by_roll),
        ports=ports,
    )


def _has_adjacent_reds(topology: Topology, tokens: tuple[int, ...]) -> bool:
    for h, token in enumerate(tokens):
        if token in RED_TOKENS:
            if any(tokens[n] in RED_TOKENS for n in topology.hex_neighbors[h]):
                return True
    return False


def random_base_board(
    rng: random.Random | None = None, *, separate_reds: bool = True
) -> Board:
    """A standard 19-hex board with the official terrain and token bags.
    `separate_reds` applies the variable-setup rule that 6 and 8 may not sit
    on adjacent hexes, which materially changes board value."""
    rng = rng or random.Random()
    topology = build_topology(BASE_LAYOUT)
    ports = place_ports(topology, base_port_bag(), rng)

    terrain = list(BASE_TERRAIN)
    rng.shuffle(terrain)
    slots = [h for h, t in enumerate(terrain) if t in BEARS_TOKEN]

    bag = list(BASE_TOKENS)
    for _ in range(1000):
        rng.shuffle(bag)
        tokens = [0] * len(terrain)
        for slot, token in zip(slots, bag):
            tokens[slot] = token
        if not separate_reds or not _has_adjacent_reds(topology, tuple(tokens)):
            return make_board(topology, tuple(terrain), tuple(tokens), ports)

    raise RuntimeError("could not place tokens without adjacent red numbers")


def _spiral(corner: int) -> list[Hex]:
    """The base layout's hexes in spiral order: the outer ring counterclockwise
    (as drawn) from corner `corner` (0..5), then the inner ring the same way
    from the hex inside that corner, then the centre."""
    order = []
    for radius in (2, 1):
        h = Hex(*(radius * c for c in DIRECTIONS[corner]))
        for side in range(6):
            d = DIRECTIONS[(corner - 2 - side) % 6]
            for _ in range(radius):
                order.append(h)
                h = Hex(h.q + d.q, h.r + d.r, h.s + d.s)
    order.append(ORIGIN)
    return order


def spiral_base_board(rng: random.Random | None = None) -> Board:
    """A standard 19-hex board dealt by the rulebook's variable set-up: random
    terrain, then the number discs in letter order from a random corner,
    counterclockwise toward the centre, skipping the desert. The disc order
    keeps 6 and 8 apart on its own. Ports as `random_base_board`."""
    rng = rng or random.Random()
    topology = build_topology(BASE_LAYOUT)
    ports = place_ports(topology, base_port_bag(), rng)

    terrain = list(BASE_TERRAIN)
    rng.shuffle(terrain)
    tokens = [0] * len(terrain)
    discs = iter(SPIRAL_TOKENS)
    for h in _spiral(rng.randrange(6)):
        index = topology.hex_index[h]
        if terrain[index] in BEARS_TOKEN:
            tokens[index] = next(discs)
    return make_board(topology, tuple(terrain), tuple(tokens), ports)


# How a base board is dealt: "random" places the discs in random order with
# 6 and 8 kept apart, the rulebook's fully random alternative; "spiral" lays
# them in letter order.
BOARD_MODES: dict[str, Callable[[random.Random], Board]] = {
    "random": random_base_board,
    "spiral": spiral_base_board,
}


def base_board(mode: str = "random", rng: random.Random | None = None) -> Board:
    """A standard board dealt by `BOARD_MODES[mode]`."""
    if mode not in BOARD_MODES:
        raise ValueError(f"unknown board mode {mode!r}: {sorted(BOARD_MODES)}")
    return BOARD_MODES[mode](rng or random.Random())
