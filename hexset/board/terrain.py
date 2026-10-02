# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from enum import IntEnum

__all__ = [
    "Resource",
    "NUM_RESOURCES",
    "check_resource",
    "Terrain",
    "TERRAIN_RESOURCE",
    "BEARS_TOKEN",
]


class Resource(IntEnum):
    """The five resources, in the order every hand, bank and bundle counts
    them."""

    WOOD = 0
    BRICK = 1
    SHEEP = 2
    WHEAT = 3
    ORE = 4


NUM_RESOURCES = len(Resource)


def check_resource(resource: int) -> None:
    """Refuse an index that names no resource, before a negative one wraps
    round to the far end of a hand."""
    if not 0 <= resource < NUM_RESOURCES:
        raise ValueError(f"no such resource: {resource}")


class Terrain(IntEnum):
    """What a hex is. `TERRAIN_RESOURCE` names what each produces."""

    FOREST = 0
    HILLS = 1
    PASTURE = 2
    FIELDS = 3
    MOUNTAINS = 4
    DESERT = 5
    SEA = 6
    # Seafarers' gold field: it bears a token, and a roll on it owes each
    # adjacent building a resource of its owner's choice (`state.gold_claims`)
    # rather than a fixed one.
    GOLD = 7


TERRAIN_RESOURCE: dict[Terrain, Resource | None] = {
    Terrain.FOREST: Resource.WOOD,
    Terrain.HILLS: Resource.BRICK,
    Terrain.PASTURE: Resource.SHEEP,
    Terrain.FIELDS: Resource.WHEAT,
    Terrain.MOUNTAINS: Resource.ORE,
    Terrain.DESERT: None,
    Terrain.SEA: None,
    Terrain.GOLD: None,
}

BEARS_TOKEN: frozenset[Terrain] = frozenset(
    {
        Terrain.FOREST,
        Terrain.HILLS,
        Terrain.PASTURE,
        Terrain.FIELDS,
        Terrain.MOUNTAINS,
        Terrain.GOLD,
    }
)
