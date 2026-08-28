from __future__ import annotations

from .coords import Hex, hexagon

BASE_LAYOUT: tuple[Hex, ...] = tuple(hexagon(2))
MINI_LAYOUT: tuple[Hex, ...] = tuple(hexagon(1))
