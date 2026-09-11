# SPDX-License-Identifier: GPL-3.0-only
"""Purchase progress after legal bank/port conversions.

This module is deliberately separate from the shipped evaluator.  It provides
an experimental feature for ablations: the evaluator may ask how close a hand
is to its best still-open purchase after any sequence of legal one-resource
bank/port trades.  A public bank vector can be supplied to enforce the same
availability rule as ``_trade_actions``; omitting it models an unlimited bank
and is useful for controlled unit tests only.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

from ..evaluate import PURCHASE_COST, PURCHASE_VALUE, Survey, affordable
from ...board.terrain import NUM_RESOURCES


def _raw_progress(hand: Sequence[float], walk: Survey, deck_left: int) -> float:
    """The existing purchase-progress formula, without any conversion."""
    best = 0.0
    for purchase, value in PURCHASE_VALUE.items():
        if not affordable(purchase, walk, deck_left):
            continue
        needed, total = PURCHASE_COST[purchase]
        scored = value * sum(min(hand[r], n) for r, n in needed) / total
        if scored > best:
            best = scored
    return best


def port_aware_progress(
    hand: Sequence[int | float], walk: Survey, *, deck_left: int,
    bank: Sequence[int] | None = None,
) -> float:
    """Return best purchase progress reachable by legal bank/port trades.

    Each edge trades one resource ``r`` for one resource ``s`` at
    ``walk.ratios[r]``.  A transition is legal only when the hand has enough
    cards and (when ``bank`` is given) the bank has the received resource.
    Bank holdings are updated as part of the state, so a sequence cannot spend
    the same bank card twice.  Every transition reduces the hand total by at
    least one (all legal ratios are >= 2), which makes the breadth-first search
    finite even when a caller supplies unusual ratios.
    """
    if len(hand) != NUM_RESOURCES or len(walk.ratios) != NUM_RESOURCES:
        raise ValueError("hand and survey must contain five resources")
    start_hand = tuple(int(x) if float(x).is_integer() else float(x) for x in hand)
    if any(x < 0 for x in start_hand):
        raise ValueError("hand resources must be nonnegative")
    if bank is None:
        start_bank: tuple[int, ...] | None = None
    else:
        if len(bank) != NUM_RESOURCES or any(x < 0 for x in bank):
            raise ValueError("bank must contain five nonnegative resources")
        start_bank = tuple(int(x) for x in bank)

    best = _raw_progress(start_hand, walk, deck_left)
    if best >= 1.0:
        return best
    queue = deque([(start_hand, start_bank)])
    seen = {(start_hand, start_bank)}
    while queue:
        current, current_bank = queue.popleft()
        best = max(best, _raw_progress(current, walk, deck_left))
        for give in range(NUM_RESOURCES):
            ratio = walk.ratios[give]
            if ratio < 1 or current[give] < ratio:
                continue
            for receive in range(NUM_RESOURCES):
                if receive == give:
                    continue
                if current_bank is not None and current_bank[receive] <= 0:
                    continue
                next_hand = list(current)
                next_hand[give] -= ratio
                next_hand[receive] += 1
                next_bank = current_bank
                if current_bank is not None:
                    bank_values = list(current_bank)
                    bank_values[give] += ratio
                    bank_values[receive] -= 1
                    next_bank = tuple(bank_values)
                state = (tuple(next_hand), next_bank)
                if state not in seen:
                    seen.add(state)
                    queue.append(state)
    return best


__all__ = ["port_aware_progress"]
