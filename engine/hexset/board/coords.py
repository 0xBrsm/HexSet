# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

from typing import NamedTuple


class Hex(NamedTuple):
    """Cube coordinate for a hex. Always satisfies q + r + s == 0."""

    q: int
    r: int
    s: int


DIRECTIONS: tuple[Hex, ...] = (
    Hex(1, -1, 0),
    Hex(1, 0, -1),
    Hex(0, 1, -1),
    Hex(-1, 1, 0),
    Hex(-1, 0, 1),
    Hex(0, -1, 1),
)

ORIGIN = Hex(0, 0, 0)


def neighbor(h: Hex, direction: int) -> Hex:
    d = DIRECTIONS[direction % 6]
    return Hex(h.q + d.q, h.r + d.r, h.s + d.s)


def neighbors(h: Hex) -> tuple[Hex, ...]:
    return tuple(neighbor(h, i) for i in range(6))


def distance(a: Hex, b: Hex) -> int:
    return (abs(a.q - b.q) + abs(a.r - b.r) + abs(a.s - b.s)) // 2


def hexagon(radius: int) -> list[Hex]:
    """All hexes within `radius` steps of the origin, as a hexagonal block."""
    out = []
    for q in range(-radius, radius + 1):
        lo = max(-radius, -q - radius)
        hi = min(radius, -q + radius)
        for r in range(lo, hi + 1):
            out.append(Hex(q, r, -q - r))
    return sorted(out)


def translate(hexes: list[Hex], offset: Hex) -> list[Hex]:
    return [Hex(h.q + offset.q, h.r + offset.r, h.s + offset.s) for h in hexes]
