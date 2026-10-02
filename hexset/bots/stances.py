# SPDX-License-Identifier: GPL-3.0-only
"""Convert per-seat evaluation vectors to a player's search objective."""
from __future__ import annotations

from typing import Sequence

__all__ = [
    "own",
    "relative",
    "paranoid",
    "STANCES",
]


def own(vector: Sequence[float], seat: int) -> float:
    """Plain max^n: each seat wants its own score high and ignores the rest."""
    return vector[seat]


def relative(vector: Sequence[float], seat: int) -> float:
    """Own score less the average of everyone else's: a constant-sum reading."""
    others = [v for p, v in enumerate(vector) if p != seat]
    return vector[seat] - sum(others) / len(others)


def paranoid(vector: Sequence[float], seat: int) -> float:
    """Own score less the best opponent's. The leader is the only rival."""
    others = [v for p, v in enumerate(vector) if p != seat]
    return vector[seat] - max(others)


# How a seat turns the per-seat vector into the one number it maximises.
STANCES = {"own": own, "relative": relative, "paranoid": paranoid}

