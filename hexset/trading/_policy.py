# SPDX-License-Identifier: GPL-3.0-only
"""The negotiation protocol a gate plays, separated from what it is worth.

A gate answers two different questions. *What is this exchange worth to me?*
is the valuation, and it is the whole of what a bot brings: one bot searches,
a value head reads a position. *What will I put to the table, what
will I answer, and what will I sign?* is the protocol, and it is the same
work whoever is answering the first question.

`TradeProtocol` is that second half, built from one `TradeParams` and any
object that can value a batch of candidates -- `gains_many(view, received,
counterparties)` for this seat's own gain and, optionally,
`estimate_many(view, candidates)` for its read of the counterparty's. It
carries the per-turn planning state and exposes exactly the hooks
`hexset.trading._engine` duck-types for -- its `candidates` is the one fragment
the plan calls for next, its `offer` takes it as it stands -- so a bot
installs it by delegating and gets the same bargaining behaviour whatever it
is underneath.

`hooks_for` says which of those hooks a parameter set actually needs. A gate
whose parameters constrain nothing -- no card cap of its own, nothing charged
for a response, no planning -- installs none of them and is answered by the
engine's own `default_offer`/`default_respond`/`default_pick`, exactly as it
was before it carried parameters at all.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from . import _engine as engine
from ._fragments import choose_initial, choose_remainder, legal_fragment
from ._params import TradeParams

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..view import View

Bundle = tuple[int, ...]

#: Every hook a `TradeProtocol` can install: the protocol names
#: `hexset.trading._engine` reads off a gate by name. A bot installing the
#: protocol delegates these and no others. The engine also reads names the
#: protocol never supplies -- the valuation (`gains_many`, `estimate_many`,
#: `accepts_many`, `accepts`), `trade_floor`, `trade_params`, the timing hook
#: `trade_now` and the activity feed `observe_trade` -- which stay the bot's
#: own.
HOOKS: tuple[str, ...] = (
    "candidates",
    "offer",
    "respond",
    "respond_any",
    "pick",
    "consent_gain",
    "allow_repeated_offer",
    "trade_round_finished",
)


def hooks_for(params: TradeParams) -> frozenset[str]:
    """Which of `HOOKS` `params` actually needs installed.

    Nothing is installed for a gate that constrains nothing, so the engine's
    defaults answer for it and a neutral parameter set is not a behaviour
    change in disguise. Planning needs every hook; card caps and a responder
    price need the answering ones (`respond`, `respond_any`, `pick`,
    `consent_gain`) but not `candidates`/`offer`, whose
    alternative -- the bot's own menu, or the engine's open offer menu -- is
    a different protocol rather than a different limit.
    """
    if not params.trades:
        return frozenset()
    if params.fragment_trades:
        return frozenset(HOOKS)
    if params.constrains_responses:
        return frozenset({"respond", "respond_any", "pick", "consent_gain"})
    return frozenset()


class TradeProtocol:
    """One seat's bargaining, over one valuation.

    `valuer` is asked for gains and estimates and is never asked to plan;
    `params` carries every limit; `seed` keys the planning draw that settles
    a target exactly at the cutoff (`choose_initial`'s `draw_key`), so the
    draw is reproducible from the gate, the turn and the seat and never from
    the true hands.
    """

    def __init__(self, valuer: object, params: TradeParams, *, seed: int = 0) -> None:
        self.valuer = valuer
        self.params = params
        self.seed = int(seed)
        self._turn: int | None = None
        self._plan: list[Bundle] = []
        self._attempt = 0
        self._repeated = False
        # Set when an exchange executed on a bundle other than the planned
        # fragment: the target is over for this turn, whatever the budget.
        self._done = False
        self._ledger_id: int | None = None
        # The seat the current fragment was planned against: the menu names
        # it so the engine's counterparty-gain estimate has someone to read.
        self._partner: int | None = None

    # -- valuation, borrowed ------------------------------------------------

    def gains_many(
        self, view: "View", received: Sequence[Bundle], counterparties: Sequence[int]
    ) -> list[float]:
        return [
            float(x) for x in self.valuer.gains_many(view, list(received), list(counterparties))
        ]

    def estimate_many(
        self, view: "View", candidates: Sequence[tuple[int, Bundle]]
    ) -> list[float]:
        """The valuer's read of each candidate's *counterparty*-side gain, or
        its own gain where it has no opponent model -- the same fallback
        `engine._estimate_many` makes, so a valuer without one offers what is
        best for itself rather than guessing."""
        fn = getattr(self.valuer, "estimate_many", None)
        if fn is not None:
            return [float(x) for x in fn(view, list(candidates))]
        return self.gains_many(view, [b for _, b in candidates], [c for c, _ in candidates])

    # -- limits -------------------------------------------------------------

    @property
    def trade_offer_budget(self) -> int:
        return self.params.trade_offer_budget

    def within_caps(self, received: Sequence[int]) -> bool:
        """Whether this exchange is inside what this gate will move at all.

        A cap it declares is a refusal rather than a preference, read on both
        sides and in both directions. A cap it does not declare is no cap at
        all: nothing else bounds what this gate will sign.
        """
        params = self.params
        if params.max_give_cards is not None:
            if sum(max(-int(n), 0) for n in received) > params.max_give_cards:
                return False
        return True

    def fragment_cap(self) -> int:
        """Cards one fragment may move a side: `TradeParams.fragment_width`,
        this gate's own `fragment_cards`, else its `max_give_cards`, else
        `ENUMERATION_CARDS`."""
        return self.params.fragment_width

    def consent_gain(
        self, view: "View", received: Bundle, other: int, *, role: str
    ) -> float:
        """This seat's gain from `received`, priced for consent.

        The card caps are hard: an exchange outside them is refused outright,
        whoever asks and however it is valued. The responder price is charged
        on responses and on final consent, never to the proposer -- a seat
        that goes looking for a trade has already decided the cards are worth
        parting with, and charging it twice would price its own offer out.
        """
        if not self.within_caps(received):
            return -1.0
        gain = self.gains_many(view, [tuple(received)], [int(other)])[0]
        if role == "responder":
            gain -= self.params.responder_card_risk * sum(max(-int(n), 0) for n in received)
        return gain

    def clears(self, gain: float) -> bool:
        """Whether `gain` clears this gate's own floor. Identical to
        `engine.clears_floor` for any nonnegative floor, which
        `TradeParams` guarantees."""
        return gain > 0.0 and gain > self.params.trade_floor

    # -- the planning state machine ----------------------------------------

    def _reset(self, turn: int, ledger_id: int) -> None:
        self._turn = int(turn)
        self._plan = []
        self._attempt = 0
        self._repeated = False
        self._done = False
        self._ledger_id = ledger_id
        self._partner = None

    def _pool(self, view: "View", locked=(), already_offered=()):
        """The parent pool: every exchange this seat's own hand can put to
        the table (`engine.open_candidates`), against every unlocked seat.

        Unprivileged by construction, and ungated: nothing another seat is
        known or sampled to hold narrows it, because an offer is broadcast
        and whoever holds the cards may take it.
        """
        if not self.params.trades:
            return []
        me = view.perspective
        locked = frozenset(int(seat) for seat in locked)
        counterparties = [
            s for s in range(view.num_players) if s != me and s not in locked
        ]
        candidates = engine.open_candidates(
            view, counterparties, self.params.enumeration_cards
        )
        offered = {tuple(map(int, b)) for b in (already_offered or ())}
        if not self._plan or self._attempt == 0:
            candidates = [c for c in candidates if tuple(c[1]) not in offered]
        if not candidates:
            return []
        raw = self.gains_many(view, [b for _, b in candidates], [p for p, _ in candidates])
        estimates = self.estimate_many(view, candidates)
        return [
            (partner, tuple(b), float(own), float(theirs))
            for (partner, b), own, theirs in zip(candidates, raw, estimates)
        ]

    def candidates(
        self, view: "View", counterparties: Sequence[int], *,
        turn: int | None = None, already_offered: Sequence[Bundle] = (),
    ) -> list[tuple[int, Bundle]]:
        """This gate's menu for `hexset.trading.menu`: the one fragment its
        plan calls for next, against the seat it was planned with, or nothing
        to stop offering this turn. A counter's menu (`turn` unknown) is
        empty -- this protocol answers through its own `respond`."""
        if turn is None:
            return []
        me = view.perspective
        locked = frozenset(
            s for s in range(view.num_players) if s != me and s not in counterparties
        )
        fragment = self._fragment(view, turn=turn, locked=locked, already_offered=already_offered)
        if fragment is None:
            return []
        partner = self._partner
        if partner is None or partner not in counterparties:
            partner = int(counterparties[0]) if counterparties else me
        return [(partner, tuple(fragment))]

    def offer(self, view: "View", candidates) -> int | None:
        """The fragment in the menu is already gated and priced by the plan;
        it goes out as it stands rather than through `default_offer`'s floors."""
        return 0 if candidates else None

    def _fragment(self, view: "View", *, turn: int, locked=(), already_offered=()):
        """This broadcast's fragment, or `None` to stop offering this turn.

        The first broadcast gates one full target on the frozen cutoff and
        plans its decomposition; each later one re-prices the planned
        remainder against the live position and offers it only if it still
        stands on its own.
        """
        params = self.params
        if not params.trades:
            return None
        ledger_id = id(view.ledger)
        if self._turn != int(turn) or self._ledger_id != ledger_id:
            self._reset(turn, ledger_id)
        # `-1` is a gate declaring no limit of its own, not a budget already
        # spent; only a declared budget bounds the attempts.
        budget = params.trade_offer_budget
        if self._done or (0 <= budget <= self._attempt):
            return None
        if self._plan and self._attempt >= len(self._plan):
            return None

        scored = self._pool(view, locked, already_offered)
        fit = self._fit(view, [(partner, bundle) for partner, bundle, _, _ in scored])
        if self._attempt == 0 and not self._plan:
            selected = choose_initial(
                scored,
                cutoff=params.fragment_threshold,
                draw_key=(self.seed, 0, int(turn), int(view.perspective)),
                max_cards=self.fragment_cap(),
                max_fragments=params.max_fragments,
                fit=fit,
            )
            if selected is None:
                return None
            self._plan = list(selected.partition)
            self._partner = int(selected.partner)
        elif self._attempt >= len(self._plan):
            return None
        elif self._attempt > 0:
            # The remainder is never carried on the parent's authority: it
            # must clear on its own value and be positive for the recipient.
            chosen = choose_remainder(
                scored, self._plan[self._attempt], max_cards=self.fragment_cap(),
                fit=fit,
            )
            if chosen is None:
                return None
            self._plan[self._attempt] = tuple(chosen[1])
            self._partner = int(chosen[0])
        return self._plan[self._attempt]

    def _fit(self, view: "View", rows):
        """`fit_offers`' reading of each `(partner, bundle)` row, as the
        `fit(partner, bundle)` the planners take, or `None` -- the choice
        without it -- when it is off or no seat has shown anything."""
        if not self.params.fit_offers:
            return None
        seats = view.ledger.seats
        if not any(any(row.want) or any(row.waste) for row in seats):
            return None
        fits = {(int(p), tuple(b)): engine.fit(view, int(p), b) for p, b in rows}
        return lambda partner, bundle: fits.get((int(partner), tuple(bundle)), 0)

    def trade_round_finished(self, view, offer, responses, trade, *, turn: int) -> None:
        """Advance the plan by attempted broadcast.

        A declined or incompatible answer is an ordinary failed attempt and
        keeps the planned remainder; only an exchange that actually executed
        on a *different* bundle ends the target.
        """
        if self._turn != int(turn) or not self._plan:
            return
        if offer is None:
            return
        offered = tuple(getattr(offer, "received", offer))
        executed = getattr(trade, "received", None) if trade is not None else None
        if executed is not None and tuple(executed) != offered:
            self._plan = []
            self._done = True
            self._repeated = False
            return
        if self._attempt < len(self._plan) and offered == self._plan[self._attempt]:
            self._attempt += 1
            self._repeated = trade is not None and self._attempt < len(self._plan)

    def allow_repeated_offer(self, view, bundle, *, turn: int) -> bool:
        """The exact-identical-remainder exception: a later fragment equal to
        an earlier one may be rebroadcast, but only after that one executed.
        An identical offer that was *refused* stays excluded."""
        return bool(
            self._repeated
            and self._turn == int(turn)
            and self._attempt < len(self._plan)
            and tuple(bundle) == self._plan[self._attempt]
        )

    # -- answering ----------------------------------------------------------

    def _counter_pool(self, view: "View", actor: int) -> list[Bundle]:
        """What this seat will counter with: the ledger-bounded candidates
        (`engine.known_candidates`), narrowed to what it will actually move.
        Under planning a counter is also held to a fragment's shape, so the
        gate never counters with something it would not itself have
        offered."""
        pool = [
            b for _, b in engine.known_candidates(view, [actor], self.params.enumeration_cards)
        ]
        if self.params.fragment_trades:
            cap = self.fragment_cap()
            return [b for b in pool if legal_fragment(b, max_cards=cap)]
        return [b for b in pool if self.within_caps(b)]

    def respond(self, view: "View", offer):
        """Accept, counter, or pass -- everything priced as a response, so the
        responder price and the card caps apply to all three."""
        if offer.received is None:
            return engine.Response(view.perspective, engine.RESPONSE_PASS, None)
        me, actor = view.perspective, int(offer.actor)
        offered = tuple(int(n) for n in offer.received)
        mine = tuple(-n for n in offered)
        covers = all(view.known[me][r] >= n for r, n in enumerate(offered) if n > 0)
        if covers and self.clears(self.consent_gain(view, mine, actor, role="responder")):
            return engine.Response(me, engine.RESPONSE_ACCEPT, offered)
        return self._counter(view, actor)

    def respond_any(self, view: "View", offer):
        """An open offer is an invitation to counter: the counter `respond`
        would send, under the same caps, responder price and fit, else a
        pass. Never an acceptance."""
        return self._counter(view, int(offer.actor))

    def _counter(self, view: "View", actor: int):
        """This seat's best counter to `actor`, signed towards `actor`, or a
        pass: the counter pool priced as a response."""
        me = view.perspective
        pool = self._counter_pool(view, actor)
        if not pool:
            return engine.Response(me, engine.RESPONSE_PASS, None)
        own = self.gains_many(view, pool, [actor] * len(pool))
        estimates = self.estimate_many(view, [(actor, b) for b in pool])
        # The counterparty test is strict positivity under planning -- a
        # fragment is offered only where the other side gains -- and this
        # gate's own floor otherwise, which is what `default_respond` applies.
        wanted = (lambda theirs: theirs > 0.0) if self.params.fragment_trades else self.clears
        eligible = []
        for i, (mine_gain, theirs) in enumerate(zip(own, estimates)):
            adjusted = mine_gain - self.params.responder_card_risk * sum(
                max(-n, 0) for n in pool[i]
            )
            if self.clears(adjusted) and wanted(theirs):
                eligible.append((i, adjusted))
        if not eligible:
            return engine.Response(me, engine.RESPONSE_PASS, None)
        fit = self._fit(view, [(actor, pool[i]) for i, _ in eligible])
        rank = (lambda i: fit(actor, pool[i])) if fit is not None else (lambda i: 0)
        # Ties break towards the smaller counter, then canonical bundle order.
        i, _ = max(eligible, key=lambda row: (
            rank(row[0]), row[1], -sum(abs(n) for n in pool[row[0]]), tuple(-n for n in pool[row[0]]),
        ))
        return engine.Response(me, engine.RESPONSE_COUNTER, tuple(-n for n in pool[i]))

    def pick(self, view: "View", responses):
        """Rank answers by this seat's own gain, after masking any answer this
        gate would not sign: the caps bind the actor as well as the
        responder, so an answer outside them is never chosen and then
        refused at execution."""
        masked = list(responses)
        for i, response in enumerate(masked):
            if response.bundle is not None and not self.within_caps(response.bundle):
                masked[i] = engine.Response(response.seat, engine.RESPONSE_PASS, None)
        return engine.default_pick(self.valuer, view, masked)


class DeclaredTrade:
    """A gate whose every bargaining limit reads through its own `trade`, so
    its settings live in one place with no second copy to drift from it.
    These are the names the engine, the arena and `retune` ask a gate for;
    each setter writes through `retune`, so flipping one on a built gate keeps
    an installed protocol in step and cannot leave two answers to the same
    question."""

    trade: TradeParams

    @property
    def trade_params(self) -> TradeParams:
        return self.trade

    @property
    def trade_offer_budget(self) -> int:
        """Broadcasts this gate asks the driver for each of its own turns."""
        return self.trade.trade_offer_budget

    @property
    def max_offers(self) -> int | None:
        """Offers this gate makes on its own turn; `0` is the no-trade arm."""
        return self.trade.max_offers

    @max_offers.setter
    def max_offers(self, value: int | None) -> None:
        retune(self, max_offers=value)

    @property
    def trade_floor(self) -> float:
        return self.trade.trade_floor

    @trade_floor.setter
    def trade_floor(self, value: float) -> None:
        retune(self, trade_floor=value)

    @property
    def gate_plies(self) -> int:
        return self.trade.gate_plies

    @gate_plies.setter
    def gate_plies(self, value: int) -> None:
        retune(self, gate_plies=value)

    @property
    def responder_card_risk(self) -> float:
        return self.trade.responder_card_risk

    @property
    def fragment_trades(self) -> bool:
        return self.trade.fragment_trades


def retune(gate: object, **changes) -> None:
    """Change fields on a gate's own `TradeParams` in place, keeping any
    installed protocol in step.

    How a seat bargains is the seat's own, so a run that wants every
    participant at (say) no trading at all says so to the participants rather
    than to the table.

    A gate carrying no `TradeParams` still answers `max_offers`, by taking the
    budget onto its `trade_offer_budget` (`-1` for `None`), which `params_of`
    reads: an override a run asked for has to reach every seat, not only the
    ones built from a parameter set. Anything else it cannot honour is
    ignored -- a gate with no valuation never trades whatever it is told.
    """
    from dataclasses import replace

    if gate is None:  # an empty seat has nothing to retune
        return
    params = getattr(gate, "trade_params", None)
    if not isinstance(params, TradeParams):
        if "max_offers" in changes:
            wanted = changes["max_offers"]
            gate.trade_offer_budget = -1 if wanted is None else int(wanted)
        return
    updated = replace(params, **changes)
    gate.trade = updated
    protocol = getattr(gate, "_protocol", None)
    if protocol is None:
        return
    protocol.params = updated
    # The hook set is a function of the parameters, so a change that adds or
    # removes a limit has to reinstall rather than leave a stale hook bound.
    install(gate, protocol)


def install(gate: object, protocol: TradeProtocol) -> TradeProtocol:
    """Bind `protocol`'s hooks onto `gate` as instance attributes, take back
    every hook its parameters do not need, and keep `protocol` on the gate
    as `_protocol`, where `retune` finds it.

    `hexset.trading._engine` reads each hook with `getattr(gate, name, None)`,
    so a hook that is not there falls back -- `menu` and `counter_menu` to
    the gate's own `candidates` where its class defines one and otherwise to
    the engine's open offer menu and known counter menu, `respond_any` to
    `default_respond_any`, `consent` to `valued`. A gate that constrains
    nothing therefore behaves exactly as an unparameterised one, and a
    reinstall after `retune` leaves no stale hook bound. Every attribute is
    set and deleted through the gate itself, so a seat that forwards them
    (`hexset.bots.TradesBy`) installs onto its trader.
    """
    wanted = hooks_for(protocol.params)
    for name in HOOKS:
        if name in wanted:
            setattr(gate, name, getattr(protocol, name))
        else:
            try:
                delattr(gate, name)
            except AttributeError:
                pass  # not bound on this instance: the class's own, or none
    gate._protocol = protocol
    return protocol
