from __future__ import annotations

from .coords import Hex, hexagon, translate

BASE_LAYOUT: tuple[Hex, ...] = tuple(hexagon(2))
MINI_LAYOUT: tuple[Hex, ...] = tuple(hexagon(1))

LAYOUTS: dict[str, tuple[Hex, ...]] = {
    "base": BASE_LAYOUT,
    "mini": MINI_LAYOUT,
}


def islands(*offsets: Hex, radius: int = 1) -> tuple[Hex, ...]:
    """A layout made of several hexagonal blocks, for Seafarers-style maps."""
    out: set[Hex] = set()
    for offset in offsets:
        out.update(translate(hexagon(radius), offset))
    return tuple(sorted(out))
