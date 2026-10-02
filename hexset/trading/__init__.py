# SPDX-License-Identifier: GPL-3.0-only
"""Player-to-player trading: the referee, the parameters and the protocol.

Three things live here, and the split is the point:

- `_engine` is the **referee**. It enumerates candidates, validates an
  exchange against the table's own rules, runs the clearing house and the
  trade round, and moves the cards. It takes no view on what a trade is
  worth and no view on how hard a seat should bargain.
- `_params` is what one **gate declares about itself**: `TradeParams`, the one
  object a bot written in Python and a neural checkpoint both carry. Card
  caps, the clearing floor, the responder price, the offer budget, the
  fragment plan and its cutoff.
- `_policy` is the **protocol** those parameters drive -- what a gate offers,
  answers, and signs -- built over any object that can value a batch of
  candidates. A bot supplies the valuation; the protocol is shared.

`_fragments` is the pure planning arithmetic behind a fragmented proposal.

The names in `__all__` are the package's public surface; the submodules,
underscored, are internal and may change in any release.
"""
from __future__ import annotations

from ._engine import (  # noqa: F401  (re-exported)
    RESPONSE_ACCEPT,
    RESPONSE_COUNTER,
    RESPONSE_PASS,
    TRADE_RULES,
    Bundle,
    Offer,
    Response,
    Trade,
    apply_trades,
    bundle,
    choose_and_execute,
    CounterThread,
    counter_thread,
    clears_floor,
    consent,
    counter_menu,
    default_offer,
    default_pick,
    default_respond,
    default_respond_any,
    exchange,
    execute_agreed,
    fit,
    has_room,
    holds,
    is_open,
    one_for_one,
    known_candidates,
    menu,
    offer,
    open_candidates,
    pick,
    respond,
    resolve_offer,
    show,
    trade_event,
    trade_floor_of,
    trade_round,
    valued,
    valued_many,
)
from ._fragments import (  # noqa: F401  (re-exported)
    choose_initial,
    choose_remainder,
    fragment_partitions,
    legal_fragment,
)
from ._params import (  # noqa: F401  (re-exported)
    ENUMERATION_CARDS,
    UNLIMITED,
    TradeParams,
    params_of,
)
from ._policy import (  # noqa: F401  (re-exported)
    HOOKS,
    DeclaredTrade,
    TradeProtocol,
    hooks_for,
    install,
    retune,
)

__all__ = [
    "ENUMERATION_CARDS",
    "UNLIMITED",
    "HOOKS",
    "RESPONSE_ACCEPT",
    "RESPONSE_COUNTER",
    "RESPONSE_PASS",
    "TRADE_RULES",
    "Bundle",
    "DeclaredTrade",
    "Offer",
    "Response",
    "Trade",
    "TradeParams",
    "TradeProtocol",
    "apply_trades",
    "bundle",
    "choose_and_execute",
    "CounterThread",
    "counter_thread",
    "choose_initial",
    "choose_remainder",
    "clears_floor",
    "consent",
    "counter_menu",
    "default_offer",
    "default_pick",
    "default_respond",
    "default_respond_any",
    "exchange",
    "execute_agreed",
    "fit",
    "has_room",
    "fragment_partitions",
    "holds",
    "hooks_for",
    "is_open",
    "install",
    "legal_fragment",
    "one_for_one",
    "params_of",
    "retune",
    "known_candidates",
    "menu",
    "offer",
    "open_candidates",
    "pick",
    "respond",
    "resolve_offer",
    "show",
    "trade_event",
    "trade_floor_of",
    "trade_round",
    "valued",
    "valued_many",
]
