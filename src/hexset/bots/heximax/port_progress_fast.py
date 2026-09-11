# SPDX-License-Identifier: GPL-3.0-only
"""Constant-size port-aware purchase progress approximation.

For each purchase, existing cards needed by the goal are reserved. Only
surplus cards can be spent on conversions; because every trade loses cards and
all received cards have equal marginal progress within a goal, the optimum is
to count available conversion credits against deficient goal cards.
"""
from __future__ import annotations

from typing import Sequence

from ..evaluate import PURCHASE_COST, PURCHASE_VALUE, Survey, affordable
from ...board.terrain import NUM_RESOURCES


def port_aware_progress_fast(
    hand: Sequence[int | float], walk: Survey, *, deck_left: int,
    bank: Sequence[int] | None = None,
) -> float:
    if len(hand) != NUM_RESOURCES or len(walk.ratios) != NUM_RESOURCES:
        raise ValueError("hand and survey must contain five resources")
    if any(x < 0 for x in hand):
        raise ValueError("hand resources must be nonnegative")
    if bank is not None and (len(bank) != NUM_RESOURCES or any(x < 0 for x in bank)):
        raise ValueError("bank must contain five nonnegative resources")
    best = 0.0
    for purchase, value in PURCHASE_VALUE.items():
        if not affordable(purchase, walk, deck_left):
            continue
        needed, total = PURCHASE_COST[purchase]
        need = [0] * NUM_RESOURCES
        for resource, count in needed:
            need[resource] = count
        held = [min(float(hand[r]), need[r]) for r in range(NUM_RESOURCES)]
        missing = sum(need) - sum(held)
        if missing:
            credits = 0
            for give, ratio in enumerate(walk.ratios):
                surplus = max(0.0, float(hand[give]) - need[give])
                credits += int(surplus // ratio)
            if bank is None:
                receive_capacity = missing
            else:
                receive_capacity = sum(
                    min(need[r] - held[r], int(bank[r]))
                    for r in range(NUM_RESOURCES) if need[r] > held[r]
                )
            filled = min(missing, credits, receive_capacity)
        else:
            filled = 0
        progress = value * (sum(held) + filled) / total
        best = max(best, progress)
    return best


__all__ = ["port_aware_progress_fast"]
