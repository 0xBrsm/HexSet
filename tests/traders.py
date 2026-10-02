# SPDX-License-Identifier: GPL-3.0-only
"""A bot registered the way any bot outside hexset is: the engine's trade and
ledger tests seat it where they need a gate that really trades.

`test-trader` moves at random; its gate values a card more the fewer of that
kind the seat holds, so two seats short of different cards both gain from a
swap and real exchanges clear.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from hexset.actions import options_for
from hexset.arena import Entrant, register_entrant_kind, register_preset
from hexset.trading import DeclaredTrade, TradeParams

TRADER = "test-trader"
TRADE = TradeParams(max_offers=2)

#: A parameter set that sets every kind of limit a protocol installs hooks
#: for: offer budget, card cap, responder price, fragment plan, fitted offers.
FRAGMENTED = TradeParams(
    max_offers=2,
    max_give_cards=2,
    responder_card_risk=0.001875,
    fragment_trades=True,
    fragment_threshold=0.002,
    max_fragments=2,
    fragment_cards=2,
    fit_offers=True,
    trade_floor=0.0,
)


@dataclass
class ScarcityTrader(DeclaredTrade):
    rng: random.Random = field(default_factory=random.Random)
    trade: TradeParams = TRADE

    def choose(self, game):
        return self.rng.choice(options_for(game))

    def gains_many(self, view, received, counterparties):
        hand = view.state.hands[view.perspective]
        return [sum(n / (1 + hand[r]) for r, n in enumerate(bundle)) for bundle in received]


register_entrant_kind(
    TRADER, lambda entrant, board, rng: ScarcityTrader(rng, entrant.trade_params(TRADE)))
register_preset(TRADER, Entrant(TRADER, kind=TRADER))
