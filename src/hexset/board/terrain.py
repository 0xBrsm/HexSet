# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from enum import IntEnum


class Resource(IntEnum):
    WOOD = 0
    BRICK = 1
    SHEEP = 2
    WHEAT = 3
    ORE = 4


NUM_RESOURCES = len(Resource)


class Terrain(IntEnum):
    FOREST = 0
    HILLS = 1
    PASTURE = 2
    FIELDS = 3
    MOUNTAINS = 4
    DESERT = 5
    SEA = 6
    GOLD = 7


TERRAIN_RESOURCE: dict[Terrain, Resource | None] = {
    Terrain.FOREST: Resource.WOOD,
    Terrain.HILLS: Resource.BRICK,
    Terrain.PASTURE: Resource.SHEEP,
    Terrain.FIELDS: Resource.WHEAT,
    Terrain.MOUNTAINS: Resource.ORE,
    Terrain.DESERT: None,
    Terrain.SEA: None,
    # Gold pays the holder's choice of resource, so the tile alone cannot
    # determine the yield; it is resolved by the collecting player's action.
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
