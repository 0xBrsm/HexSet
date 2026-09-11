# SPDX-License-Identifier: GPL-3.0-only
"""Experimental port-aware Heximax arm.

Importing this module registers ``heximax-port-aware``.  The normal
``heximax`` and ``heximax-notrade`` presets are not modified.
"""
from __future__ import annotations

import random

from hexset.arena import Entrant, register_entrant_kind, register_preset
from .evaluate import NO_TRADE_WEIGHTS, TRADING_WEIGHTS, HonestEvaluator
from .port_progress_fast import port_aware_progress_fast
from .search import BY_MODE, DEFAULT_MAX_NODES, Heximax


class PortAwareHonestEvaluator(HonestEvaluator):
    """HonestEvaluator with conversion-aware purchase progress only."""

    def _hand_terms_of(self, state, seat, hand, belief, walk=None):
        progress, spare, risk = super()._hand_terms_of(state, seat, hand, belief, walk)
        if walk is None:
            walk = self._walk(state, seat)
        converted = port_aware_progress_fast(
            hand, walk, deck_left=len(state.deck), bank=state.bank,
        )
        return converted, spare, risk


def _spawn(entrant: Entrant, board, rng: random.Random) -> Heximax:
    mode = entrant.mode
    weights = entrant.weights or (NO_TRADE_WEIGHTS if mode == "notrade" else TRADING_WEIGHTS)
    max_trades = entrant.max_trades
    if max_trades is BY_MODE:
        max_trades = 0 if mode == "notrade" else None
    evaluator = PortAwareHonestEvaluator(board, weights)
    return Heximax(
        evaluator, depth=entrant.depth, width=entrant.width,
        max_nodes=entrant.max_nodes or DEFAULT_MAX_NODES, k=entrant.k, rng=rng,
        stance=entrant.stance or "win", max_trades=max_trades,
        mode=mode, temperature=entrant.temperature,
    )


register_entrant_kind("heximax-port-aware", _spawn)

register_preset(
    "heximax-port-aware",
    Entrant("heximax-port-aware", kind="heximax-port-aware", depth=2, width=6,
            max_trades=0, mode="notrade"),
)


__all__ = ["PortAwareHonestEvaluator"]
