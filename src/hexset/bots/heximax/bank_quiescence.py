"""Experimental one-ply bank-trade quiescence for Heximax.

The shipped ``Heximax`` behavior is unchanged.  ``BankTradeQuiescence`` only
extends a path when its final ordinary action is a ``BANK_TRADE``: it evaluates
one additional action from the resulting position, which lets a conversion be
followed by an immediate purchase.  Player-to-player trade events are not
represented here.
"""
from __future__ import annotations

from hexset.actions import ActionType
from .search import Heximax


class BankTradeQuiescence(Heximax):
    """Heximax with one bounded post-bank-trade action at the horizon.

    The extension is deliberately path-local.  ``_used`` is set only while
    evaluating the synthetic child ply, so a second bank trade on that child
    returns to the ordinary leaf.  ``_spent`` and ``_budget`` are shared with
    the parent search, and ``ply`` is passed through unchanged; therefore the
    existing node cap and exact-roll horizon remain authoritative.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self._bank_quiescence_used = False

    def choose(self, game):
        self._bank_quiescence_used = False
        return super().choose(game)

    def _after(self, game, action, depth, knower, ply=0):
        if (
            action.type is ActionType.BANK_TRADE
            and depth == 1
            and not self._bank_quiescence_used
        ):
            child = self._plain_child(game, action)
            options = self._options_in(child, knower)
            if not options:
                return self._leaf(child, knower)
            self._bank_quiescence_used = True
            try:
                return self._best_of(child, options, 1, child.current_player, knower, ply + 1)
            finally:
                self._bank_quiescence_used = False
        return super()._after(game, action, depth, knower, ply)


__all__ = ["BankTradeQuiescence"]

# Opt-in registrations. Importing this module is the only activation point;
# ordinary ``heximax`` and ``heximax-notrade`` remain byte-for-byte unchanged.
import random
from hexset.arena import Entrant, register_entrant_kind, register_preset
from .evaluate import NO_TRADE_WEIGHTS, TRADING_WEIGHTS, HonestEvaluator
from .search import DEFAULT_MAX_NODES


def _spawn(entrant: Entrant, board, rng: random.Random) -> BankTradeQuiescence:
    mode = entrant.mode
    weights = entrant.weights or (NO_TRADE_WEIGHTS if mode == "notrade" else TRADING_WEIGHTS)
    max_trades = entrant.max_trades
    if max_trades is None and mode == "notrade":
        max_trades = 0
    evaluator = HonestEvaluator(board, weights, port_aware=entrant.port_aware)
    return BankTradeQuiescence(
        evaluator, depth=entrant.depth, width=entrant.width,
        max_nodes=entrant.max_nodes or DEFAULT_MAX_NODES, k=entrant.k, rng=rng,
        stance=entrant.stance or "win", max_trades=max_trades,
        mode=mode, temperature=entrant.temperature,
    )


register_entrant_kind("heximax-bankext", _spawn)
register_entrant_kind("heximax-bankext-wide", _spawn)
register_preset("heximax-bankext", Entrant(
    "heximax-bankext", kind="heximax-bankext", mode="notrade", max_trades=0,
    depth=2, width=6, max_nodes=600,
))
register_preset("heximax-bankext-wide", Entrant(
    "heximax-bankext-wide", kind="heximax-bankext-wide", mode="notrade", max_trades=0,
    depth=2, width=12, max_nodes=2400,
))
