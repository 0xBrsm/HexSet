# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import random
from enum import IntEnum

__all__ = [
    "DevCard",
    "NUM_DEV_CARDS",
    "DECK_COMPOSITION",
    "DECK_SIZE",
    "PLAYABLE",
    "ROAD_BUILDING_ROADS",
    "YEAR_OF_PLENTY_RESOURCES",
    "make_deck",
]


class DevCard(IntEnum):
    """The five development card kinds; `DECK_COMPOSITION` says how many of
    each the deck holds."""

    KNIGHT = 0
    VICTORY_POINT = 1
    ROAD_BUILDING = 2
    YEAR_OF_PLENTY = 3
    MONOPOLY = 4


NUM_DEV_CARDS = len(DevCard)

DECK_COMPOSITION: dict[DevCard, int] = {
    DevCard.KNIGHT: 14,
    DevCard.VICTORY_POINT: 5,
    DevCard.ROAD_BUILDING: 2,
    DevCard.YEAR_OF_PLENTY: 2,
    DevCard.MONOPOLY: 2,
}

DECK_SIZE = sum(DECK_COMPOSITION.values())

# Victory point cards are revealed to win, never played as an action.
PLAYABLE: frozenset[DevCard] = frozenset(
    {
        DevCard.KNIGHT,
        DevCard.ROAD_BUILDING,
        DevCard.YEAR_OF_PLENTY,
        DevCard.MONOPOLY,
    }
)

ROAD_BUILDING_ROADS = 2
YEAR_OF_PLENTY_RESOURCES = 2


def make_deck(rng: random.Random | None = None) -> list[int]:
    """The development deck per `DECK_COMPOSITION`, shuffled by `rng` when one
    is given and in composition order otherwise."""
    deck = [card for card, count in DECK_COMPOSITION.items() for _ in range(count)]
    if rng is not None:
        rng.shuffle(deck)
    return deck
