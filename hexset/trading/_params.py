# SPDX-License-Identifier: GPL-3.0-only
"""One model-level description of how a seat bargains.

Every limit a trade gate applies to itself lives here, in one frozen object
that a bot written in Python and a neural checkpoint both carry: what it is
willing to part with and take in, the floor its own gains are held to, what a
response costs it per outgoing card, how many offers it makes in a turn,
and whether it plans one target across fragments.

A parameter here is one seat's own policy and binds nobody else: two seats at
one table may carry different ones, and a run may deliberately pair them. The
table has no card rule to check them against -- a gate refuses what it will
not move through its own consent, and a seat declaring nothing is bounded only
by the cards it holds.

**A limit not declared is a limit not imposed.** Every cap here defaults to
`None`, meaning this gate imposes none of its own. That is the difference
between a gate saying nothing and a gate saying "three": a gate read at a
hardcoded three would quietly bargain narrower than it was ever asked to.
`UNLIMITED` is that empty declaration, and it is what every gate declaring
nothing is read at.

Nothing else is shipped here. What a particular bot bargains like is that
bot's own config -- its own module for a bot written in Python, a
checkpoint's own metadata for a network -- because this module knows what a
limit is without knowing whose it should be.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

# How wide the engine *enumerates* for a gate that declares no cap of its own.
#
# This is a search bound, not a rule: nothing refuses an exchange for exceeding
# it. Candidate enumeration walks every multiset of a hand on both sides, so
# with no bound at all a fifteen-card hand is some fifteen thousand multisets a
# side and the pairing is hundreds of millions -- not a limit anyone chose, but
# one the enumeration cannot do without. A gate that wants to *propose* wider
# says so in `max_give_cards`, and is enumerated that wide.
# What a gate will *accept* is never bounded by this.
ENUMERATION_CARDS = 3

# Ceilings for metadata that arrives as text from a checkpoint file. A stale or
# absurd key is clamped rather than failing the load, the same way
# `hexset.clients._modelmeta` reads a search budget.
MAX_TRADE_FLOOR = 1.0
MAX_CARD_RISK = 1.0
MAX_OFFERS = 8
MAX_GATE_PLIES = 16

# A target is split into at most two exact fragments:
# `fragments.fragment_partitions` enumerates two-way splits only, so this is
# the shape of the decomposition rather than a budget that could be raised.
MAX_FRAGMENTS = 2

@dataclass(frozen=True)
class TradeParams:
    """How one gate bargains. Every field is that seat's own; none binds the
    table or another seat."""

    #: Offers this gate puts to the table on each of its own turns, read as
    #: `trade_offer_budget`. `None` declares no limit, so it keeps offering
    #: while it has an offer it has not already made this turn. `0` is the
    #: no-trade referent: it opens nothing and signs nothing.
    max_offers: int | None = None
    #: Cards this gate will part with in one exchange, `None` for no limit of
    #: its own -- the cards it holds are then the only bound. The proposer's
    #: own limit as well as the responder's: an answer moving more is masked
    #: out of `pick`, and its consent refuses it outright.
    max_give_cards: int | None = None
    #: The clearing floor this gate's own gains are held to, on its own value
    #: scale -- win probability for a value head, whatever units a bot's own
    #: valuation is in. `hexset.trading.trade_floor_of` reads it; there is no engine
    #: default, and a floor at `1.0` never trades.
    trade_floor: float = 0.0
    #: What a *response* costs this seat per outgoing card, charged on
    #: responses and on final consent and never to the proposer. `0.0` prices
    #: parting with a card at nothing beyond what the value function already
    #: says about it.
    responder_card_risk: float = 0.0
    #: Plan one gated target a turn and offer it as fragments, enumerating
    #: candidates in a world sampled from this seat's own belief rather than
    #: taking the engine's enumeration over the true hands. Off means the
    #: index protocol: the engine enumerates and this gate picks.
    fragment_trades: bool = False
    #: The proposer cutoff a full target is gated on, applied once to the whole
    #: parent and never reapplied to what is left of it. Read only under
    #: `fragment_trades`.
    fragment_threshold: float = 0.0
    #: Fragments one target may be offered as, `1` meaning the target is
    #: offered whole. At most `MAX_FRAGMENTS`.
    max_fragments: int = 1
    #: Cards either side of one *fragment* may move, `None` for
    #: `ENUMERATION_CARDS`. One side of a fragment is always a single card,
    #: so at `2` a fragment is `1:1`, `1:2` or `2:1`. Read only under
    #: `fragment_trades`.
    fragment_cards: int | None = None
    #: The trade gate's continuation budget: plies rolled forward over the
    #: exchanged hand before it is valued, `0` meaning one forward. Read by
    #: `hexset.clients.netbot`; a bot that searches sets its own depth.
    gate_plies: int = 0
    #: Put first the offers that fit what their counterparty has shown it
    #: wants and will give up (`hexset.trading.fit`, off the public ledger),
    #: and counter an offer the same way; among the rest, and wherever nobody
    #: has shown anything, the same choice as without it. Offers are planned
    #: only under `fragment_trades`, so there it orders both; under card caps
    #: or a responder price alone it orders this gate's counters.
    fit_offers: bool = False

    def __post_init__(self) -> None:
        if self.max_offers is not None and self.max_offers < 0:
            raise ValueError(f"max_offers cannot be negative: {self.max_offers}")
        # `None` is "no limit of my own", so only a declared limit is checked.
        for name in ("max_give_cards", "fragment_cards"):
            declared = getattr(self, name)
            if declared is not None and declared < 1:
                raise ValueError(f"{name} must be at least one card: {declared}")
        if self.trade_floor < 0.0:
            raise ValueError(f"trade_floor is negative: {self.trade_floor}")
        if self.trade_floor != self.trade_floor:
            raise ValueError("trade_floor is NaN")
        if self.responder_card_risk < 0.0:
            raise ValueError(
                f"responder_card_risk is negative: {self.responder_card_risk}"
            )
        if self.fragment_threshold < 0.0:
            raise ValueError(
                f"fragment_threshold is negative: {self.fragment_threshold}"
            )
        if not 1 <= self.max_fragments <= MAX_FRAGMENTS:
            raise ValueError(
                f"max_fragments must be 1 or {MAX_FRAGMENTS}: {self.max_fragments}"
            )
        if self.gate_plies < 0:
            raise ValueError(f"gate_plies must be non-negative: {self.gate_plies}")
        # A fragment cannot be wider than what this gate will move at all: a
        # plan whose pieces it would itself refuse never reaches the table.
        # An undeclared cap means unbounded, and nothing to contradict.
        widest = self.max_give_cards
        if self.fragment_trades and widest is not None and (self.fragment_cards or 0) > widest:
            raise ValueError(
                f"fragment_cards ({self.fragment_cards}) exceeds this gate's own "
                f"widest side ({widest})"
            )

    @property
    def trades(self) -> bool:
        """Whether this gate trades at all. `max_offers=0` is the no-trade
        referent -- it opens nothing and signs nothing -- and every pricing
        path short-circuits on it."""
        return self.max_offers != 0

    @property
    def constrains_responses(self) -> bool:
        """Whether answering under these parameters differs from answering
        under the engine's own defaults. A gate that declares no cap, charges
        nothing for a response and plans nothing is answered by
        `default_respond`/`default_pick` and installs no hooks of its own."""
        return (
            self.fragment_trades
            or self.responder_card_risk > 0.0
            or self.max_give_cards is not None
        )

    @property
    def trade_offer_budget(self) -> int:
        """Offers this gate makes on each of its own turns, `-1` for no limit
        of its own -- it keeps offering while it has an offer it has not
        already made this turn, which is what makes an undeclared limit
        terminate rather than run for ever."""
        return -1 if self.max_offers is None else self.max_offers

    @property
    def enumeration_cards(self) -> int:
        """How many cards a side the engine enumerates candidates at for this
        gate: its declared `max_give_cards` where that is wider than
        `ENUMERATION_CARDS`, else `ENUMERATION_CARDS`, since one enumeration
        feeds both sides."""
        if self.max_give_cards is None:
            return ENUMERATION_CARDS
        return max(self.max_give_cards, ENUMERATION_CARDS)

    @classmethod
    def from_meta(cls, meta: dict[str, str], *, base: "TradeParams | None" = None) -> "TradeParams":
        """The gate `meta` asks for, read off a checkpoint's own metadata.

        Every key is optional and every value arrives as text, so an absent,
        unreadable or absurd one takes `base` (`UNLIMITED` by default) rather
        than failing the load. A checkpoint that declares no cap therefore has
        none -- it is bound by the cards it holds and by nothing else.
        Unrecognised keys are ignored without complaint.
        """
        on = UNLIMITED if base is None else base
        return cls(
            max_offers=_int(meta.get("max_offers"), on.max_offers, MAX_OFFERS, floor=0),
            max_give_cards=_int(meta.get("max_give_cards"), on.max_give_cards),
            trade_floor=_float(meta.get("trade_floor"), on.trade_floor, MAX_TRADE_FLOOR),
            responder_card_risk=_float(
                meta.get("responder_card_risk"), on.responder_card_risk, MAX_CARD_RISK
            ),
            fragment_trades=_bool(meta.get("fragment_trades"), on.fragment_trades),
            fragment_threshold=_float(
                meta.get("fragment_threshold"), on.fragment_threshold, MAX_TRADE_FLOOR
            ),
            max_fragments=_int(meta.get("max_fragments"), on.max_fragments, MAX_FRAGMENTS),
            fragment_cards=_int(meta.get("fragment_cards"), on.fragment_cards),
            gate_plies=_int(meta.get("gate_plies"), on.gate_plies, MAX_GATE_PLIES, floor=0),
            fit_offers=_bool(meta.get("fit_offers"), on.fit_offers),
        )

    def as_meta(self) -> dict[str, str]:
        """These parameters as checkpoint metadata, for an exporter to embed.

        A limit this gate does not declare is left out rather than written as
        a number, so a round trip through `from_meta` gives back the same
        absence of a limit."""
        out = {
            "trade_floor": repr(self.trade_floor),
            "responder_card_risk": repr(self.responder_card_risk),
            "fragment_trades": "1" if self.fragment_trades else "0",
            "fragment_threshold": repr(self.fragment_threshold),
            "max_fragments": str(self.max_fragments),
            "gate_plies": str(self.gate_plies),
            "fit_offers": "1" if self.fit_offers else "0",
        }
        for name in ("max_offers", "max_give_cards", "fragment_cards"):
            declared = getattr(self, name)
            if declared is not None:
                out[name] = str(declared)
        return out


def params_of(gate: object, default: "TradeParams | None" = None) -> "TradeParams":
    """The parameters `gate` carries, read by name rather than by inheritance.

    A gate exposing `trade_params` answers with its own; anything else is read
    at `default` (`UNLIMITED`) after adopting whatever loose `trade_floor` and
    `gate_plies` attributes it does carry. That is how a plain gate describes
    itself.
    """
    got = getattr(gate, "trade_params", None)
    if isinstance(got, TradeParams):
        return got
    base = UNLIMITED if default is None else default
    floor = getattr(gate, "trade_floor", None)
    plies = getattr(gate, "gate_plies", None)
    return replace(
        base,
        trade_floor=base.trade_floor if floor is None else float(floor),
        gate_plies=base.gate_plies if plies is None else int(plies),
    )


def _int(
    value: object, default: "int | None", ceiling: "int | None" = None, floor: int = 1
) -> "int | None":
    """One metadata integer. An absent or unreadable one takes the default,
    which may itself be `None` for "no limit declared". A card cap has no
    ceiling -- what a model says it will move is the model's business, and the
    cards in hand bound it anyway -- so only the budgets pass one."""
    if value is None or value == "":
        return default
    try:
        wanted = int(value)
    except (TypeError, ValueError):
        return default
    wanted = max(floor, wanted)
    return wanted if ceiling is None else min(wanted, ceiling)


def _float(value: object, default: float, ceiling: float) -> float:
    try:
        wanted = float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default
    if wanted != wanted:  # NaN would compare false against every floor
        return default
    return max(0.0, min(wanted, ceiling))


def _bool(value: object, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


#: No limits of its own: no card caps, a floor at zero, nothing charged for a
#: response, no ceiling on the offers it makes in a turn, no planning. What
#: every gate that declares nothing is read at, and what a checkpoint
#: declaring no trade keys is read at. A gate carrying this installs no
#: protocol hooks, so the engine's own defaults answer for it, and nothing
#: bounds what it will sign but the cards on the table.
UNLIMITED = TradeParams()
