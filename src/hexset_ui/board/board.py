from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from .coords import BASE_LAYOUT
from .ports import Port, base_port_bag, place_ports
from .terrain import BEARS_TOKEN, TERRAIN_RESOURCE, Resource, Terrain
from .topology import Topology
from .topology import build as build_topology

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


def scarce_resources(board: Board) -> frozenset[Resource]:
    """Resources with fewer hexes than the commonest, so brick and ore on base.

    Counted from the board rather than hardcoded, so a Seafarers layout with a
    different terrain mix gets its own answer.  Lives here rather than beside
    its callers because both the placement prior and the evaluation want it and
    it is a property of the board alone.
    """
    counts = Counter(
        resource
        for terrain in board.terrain
        if (resource := TERRAIN_RESOURCE[terrain]) is not None
    )
    if not counts:
        return frozenset()
    commonest = max(counts.values())
    return frozenset(r for r, n in counts.items() if n < commonest)


def make_board(
    topology: Topology,
    terrain: tuple[Terrain, ...],
    tokens: tuple[int, ...],
    ports: tuple[Port, ...] = (),
) -> Board:
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

    `separate_reds` applies the variable-setup rule that 6 and 8 may not sit on
    adjacent hexes, which materially changes board value and so changes the
    distribution the agent trains on.
    """
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
