# SPDX-License-Identifier: GPL-3.0-only
"""The two numbers that vary between Catan game types.

`winning_points`: VPs that end the game, checked on the holder's turn.
`discard_limit`: holding *more* than this when a seven is rolled discards half.

Standard Catan is 10 / 7. Colonist.io's 1v1 ladder plays 15 / 9.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    winning_points: int = 10
    discard_limit: int = 7

    def __post_init__(self) -> None:
        if self.winning_points < 1:
            raise ValueError(f"winning_points must be positive: {self.winning_points}")
        if self.discard_limit < 0:
            raise ValueError(f"discard_limit must be non-negative: {self.discard_limit}")


STANDARD: Rules = Rules()
COLONIST_1V1: Rules = Rules(winning_points=15, discard_limit=9)

GAME_TYPES: dict[str, Rules] = {
    "standard": STANDARD,
    "colonist-1v1": COLONIST_1V1,
}
