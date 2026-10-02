# SPDX-License-Identifier: GPL-3.0-only
"""Resource and dev-card name tables shared between state and action translation.

Every table is a bijection used in both directions. Both engines spell these
names identically; the tables fix the index <-> name correspondence once.
"""

from __future__ import annotations

from hexset.cards import DevCard

RESOURCE_NAMES: tuple[str, ...] = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")

RESOURCE_INDEX: dict[str, int] = {name: r for r, name in enumerate(RESOURCE_NAMES)}

DEV_CARD_NAMES: dict[DevCard, str] = {
    DevCard.KNIGHT: "KNIGHT",
    DevCard.VICTORY_POINT: "VICTORY_POINT",
    DevCard.ROAD_BUILDING: "ROAD_BUILDING",
    DevCard.YEAR_OF_PLENTY: "YEAR_OF_PLENTY",
    DevCard.MONOPOLY: "MONOPOLY",
}

NAME_TO_DEV_CARD: dict[str, DevCard] = {name: card for card, name in DEV_CARD_NAMES.items()}