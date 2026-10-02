# SPDX-License-Identifier: GPL-3.0-only
"""Player-to-player trading: one event a turn, no trade actions.

A candidate **bundle** is signed counts from one seat's point of view,
positive for what it receives. Each seat's **gate** prices candidates
privately -- `gains_many(view, receiveds, counterparties) -> list[float]`,
reading only that seat's own `View` -- and a deal clears only when each
side's gain exceeds that side's own `trade_floor`. The floor belongs to the
gate, not the table; there is no engine default.

`Game.trade_mode` picks the mechanism: `"round"` (the default) is
`trade_round`, one broadcast offer answered once by every other seated gate;
`"auto"` is `trade_event`, the exhaustive clearing house; `"external"` means
the caller drives rounds itself. How many offers a turn holds is the acting
gate's own `trade_offer_budget` (`TradeParams.max_offers`), not the table's:
`-1` for a gate declaring no limit, `0` for one that does not trade.

`"auto"` clears against a counterparty that never holds out and never
refuses a deal it merely dislikes, so a policy fitted or trained against it
learns an opponent no table plays.

Nothing here caps how many cards an exchange moves. A gate refuses what it
will not move through its own consent (`TradeParams.max_give_cards`), and a
seat declaring no cap is bounded only by the cards it
holds -- a manual seat's submission is its own consent. What the engine does
bound is how wide it *enumerates* candidates for a gate that declares nothing,
which is `ENUMERATION_CARDS`: a search bound, not a rule.

There are no trade actions, so nothing here reads an opponent's hand on an
actor's behalf; the engine is the referee and checks coverage itself. The
round's actor picks from a `menu` built from its own hand and the bank:
`open_candidates`, every give its exact hand covers against any cards the
other seats hold between them. An offer is broadcast, and a player asks the
table for the card it needs whether or not any one seat is known to hold it
-- whoever holds it may take the offer -- so nothing a counterparty is known
or believed to hold narrows what the actor may ask for. What the table as a
whole holds does, and it is public: every card is in the bank or a hand. A counter is different: it is addressed
to the actor alone, so a responder counters from `known_candidates`, its
exact hand against what the ledger certifies the actor holds
(`counter_menu`). A gate replaces either menu by supplying
`candidates(view, counterparties, turn=, already_offered=)` -- `turn` is
`None` for a counter -- as the fragmented protocol does with the one fragment
its plan calls for next. Coverage is the referee's: a responder accepts only
what its own hand covers, and both hands are checked when cards move.

**An agreement binds.** A responder judges an offer when it answers it, and
an actor judges an answer when it picks it -- an acceptance by who took the
offer, a counter by what it asks; nobody re-prices either at execution. An
offer goes out with some seat in mind, and a seat taking it is not a deal
with that seat: `default_pick` prices each acceptance by its taker and
declines one under the actor's floor. What execution checks is the rules:
both seats still cover their sides.
Only the clearing house (`_candidates`) enumerates from the true hands,
because there no seat is choosing: the engine clears what both gates
independently price.

The round is three stages, named for the gate verbs they run one level up
-- `offer`, `respond`, `pick` -- plus execution, and `trade_round` is those
stages run at once. A driver that must pause for a seat answering later
(the served table) runs the same stages around its pause; the round itself
is one implementation.

A gate's `offer`/`respond`/`respond_any`/`pick` are read with `getattr`,
falling back to `default_offer`/`default_respond`/`default_respond_any`/
`default_pick` -- structural, not by inheritance. A gate with an opponent
model supplies `estimate_many(view, candidates) -> list[float]`, which those
defaults read in place of guessing an unanswered counterparty.

Whatever a gate returns, the referee holds it to the rules before it counts:
an answer is always the asked seat's, an acceptance is of the offer as it
was made and only where the seat's own hand covers it, a counter is a
two-sided exchange of the table's width, and an offer gives no more than its
gate's own `max_give_cards`. Anything else is a pass.
"""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Callable, Iterable, NamedTuple, Sequence

from ..board.terrain import NUM_RESOURCES
from ..state import BANK_PER_RESOURCE, GameState, is_hidden
from ._params import ENUMERATION_CARDS, params_of

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..game import Game
    from ..view import View

Bundle = tuple[int, ...]

# `Game.trade_rule`'s legal values (`_best_clearing`'s selection key).
# `"egalitarian"` is the shipped default; the other two are lab-only.
TRADE_RULES: tuple[str, ...] = ("egalitarian", "nash", "actor")

# The enumeration helpers below take their width as a required argument, from
# the gate whose candidates they are building (`TradeParams.enumeration_cards`):
# a call site that forgot it would quietly enumerate at someone else's width,
# which is a wrong answer rather than an error.


def _tie(received: Sequence[int]) -> tuple:
    """The tie-break part of a ranking key, for `max`: canonical bundle
    order, the counts negated and compared resource by resource. For
    determinism only; it is not a preference for the smaller exchange."""
    return (tuple(-int(n) for n in received),)


def trade_floor_of(gate: object) -> float:
    """The clearing floor `gate`'s own gains are held to: its declared
    `trade_floor`. There is no engine default, so a gate declaring none
    raises `TypeError`; a negative floor raises `ValueError`."""
    floor = getattr(gate, "trade_floor", None)
    if floor is None:
        raise TypeError(
            f"{type(gate).__name__} declares no trade_floor: every seat's gate carries "
            "its own measured clearing floor, and the table has no default"
        )
    if floor < 0.0:
        raise ValueError(f"{type(gate).__name__}.trade_floor is negative")
    return float(floor)


def clears_floor(gain: float, gate: object) -> bool:
    """Whether `gain` clears `gate`'s own floor -- the one predicate every
    admission point reads. A gain at or below zero short-circuits, so a gate
    that only ever declines is never asked for a floor."""
    if gain <= 0.0:
        return False
    return gain > trade_floor_of(gate)


class Trade(NamedTuple):
    """One executed exchange; `received` is signed positive towards `a`.

    `gain_a`/`gain_b` are each side's own private gain as the clearing gate
    computed it, `0.0` where a side's gain was never evaluated.
    """

    a: int
    b: int
    received: Bundle
    gain_a: float = 0.0
    gain_b: float = 0.0


def valued(trader: object, view: "View", received: Bundle, counterparty: int) -> float:
    """`trader`'s own private gain from one candidate exchange.

    Dispatches structurally, not by inheritance: `gains_many`, then
    `accepts_many`, then `accepts`. A trader with none of them never trades.
    """
    return valued_many(trader, view, [received], [counterparty])[0]


def valued_many(
    trader: object,
    view: "View",
    received: Sequence[Bundle],
    counterparties: Sequence[int],
) -> list[float]:
    """Batched `valued`: `trader`'s gain on each of `received`, in the same
    order, so a seat's gate is asked once an event rather than once a
    candidate."""
    gains_many = getattr(trader, "gains_many", None)
    if gains_many is not None:
        return [float(x) for x in gains_many(view, list(received), list(counterparties))]
    accepts_many = getattr(trader, "accepts_many", None)
    if accepts_many is not None:
        verdicts = accepts_many(view, list(received), list(counterparties))
        return [1.0 if ok else -1.0 for ok in verdicts]
    accepts = getattr(trader, "accepts", None)
    if accepts is not None:
        return [1.0 if accepts(view, r, c) else -1.0 for r, c in zip(received, counterparties)]
    return [-1.0] * len(received)


def consent(
    gate: object, view: "View", received: Bundle, counterparty: int, *, role: str
) -> float:
    """`gate`'s gain from `received` at the moment cards move, as the gate
    itself prices consent.

    Dispatches structurally, like `valued`: a gate exposing `consent_gain`
    prices its own consent and is told which side of the exchange it is on,
    so a policy that charges a seat for the cards it parts with can charge
    the responder without also charging the proposer. Every other gate falls
    through to `valued`, so this is `valued` for all of them.
    """
    fn = getattr(gate, "consent_gain", None)
    if fn is None:
        return valued(gate, view, received, counterparty)
    return float(fn(view, received, counterparty, role=role))


def bundle(**amounts: int) -> Bundle:
    """A resource bundle by name, for tests and hand-written trades."""
    from ..board.terrain import Resource

    counts = [0] * NUM_RESOURCES
    for name, count in amounts.items():
        counts[Resource[name.upper()]] = count
    return tuple(counts)


def holds(state: GameState, player: int, wanted: Sequence[int]) -> bool:
    """Whether `player`'s hand covers `wanted`, counted per resource. An
    identity read: a hidden hand raises `HiddenRead`."""
    hand = state.hands[player]
    return all(hand[r] >= n for r, n in enumerate(wanted))


def exchange(state: GameState, a: int, b: int, received: Sequence[int]) -> None:
    """Move `received` between two seats, positive counts towards `a`.
    Checks no legality; the caller must have established coverage. A trade
    is public, so a hidden hand (an observed state) takes it by name."""
    hand_a = state.hands[a]
    hand_b = state.hands[b]
    if is_hidden(hand_a) or is_hidden(hand_b):
        for r, n in enumerate(received):
            if not n:
                continue
            if is_hidden(hand_a):
                hand_a.move(r, n)
            else:
                hand_a[r] += n
            if is_hidden(hand_b):
                hand_b.move(r, -n)
            else:
                hand_b[r] -= n
        return
    for r, n in enumerate(received):
        hand_a[r] += n
        hand_b[r] -= n


def one_for_one(given: int, wanted: int) -> Bundle:
    """The signed bundle for "I give one `given`, I receive one `wanted`"."""
    out = [0] * NUM_RESOURCES
    out[wanted] += 1
    out[given] -= 1
    return tuple(out)


def _hand_multisets(hand: Sequence[int], max_cards: int):
    """Every distinct nonempty multiset of cards this hand can cover, up to
    `max_cards` cards total, as nonnegative counts by resource index.
    Yielded lazily; branches over the cap are pruned as they are walked."""
    resources = [r for r in range(NUM_RESOURCES) if hand[r] > 0]
    counts = [0] * NUM_RESOURCES

    def walk(idx: int, remaining: int):
        if idx == len(resources):
            if any(counts):
                yield tuple(counts)
            return
        r = resources[idx]
        for n in range(min(hand[r], remaining) + 1):
            counts[r] = n
            yield from walk(idx + 1, remaining - n)
        counts[r] = 0

    yield from walk(0, max_cards)


def _candidates(state: GameState, me: int, locked: frozenset[int], max_cards: int):
    """`(counterparty, bundle)` for every coverable exchange `me` could
    propose, bundles signed towards `me`: both sides nonempty, each at most
    `max_cards` cards (the enumerating gate's own width), the two sides on disjoint
    resource sets, both coverable from the true hands. Seats in `locked` are
    never counterparties. Nothing filters ahead of the gates."""
    give_options = list(_hand_multisets(state.hands[me], max_cards))
    if not give_options:
        return
    for them in range(state.num_players):
        if them == me or them in locked:
            continue
        receive_options = list(_hand_multisets(state.hands[them], max_cards))
        for given in give_options:
            for received in receive_options:
                if any(g and r for g, r in zip(given, received)):
                    continue  # the two sides must not share a resource
                yield them, tuple(r - g for r, g in zip(received, given))


def _position_key(state: GameState, ledger) -> tuple:
    """Everything a gate can read that a trade moves: every hand and every
    seat's ledger row. The trade event's revisit check hashes this."""
    return (
        tuple(tuple(hand) for hand in state.hands),
        tuple((tuple(row.known), row.unknown) for row in ledger.seats),
    )


def trade_event(
    game: "Game", gate: "Callable[[int, View, Bundle, int], float] | None" = None,
) -> list[Trade]:
    """Clear every deal the current player and one other seat both gain
    from (the `"auto"` mechanism).

    The seated gates (`game.gates`) price every candidate. `gate`, a bare
    `(seat, view, received, counterparty) -> gain` callable carrying its own
    `trade_floor`, answers for every seat instead only where `game.gates` is
    `None`; with neither, nothing clears.

    Returns the trades executed by *this* call, in clearing order, and
    appends them to `game.trades`, which accumulates across the turn. Stops
    at the acting gate's own `trade_offer_budget` exchanges (`-1` uncapped),
    when nothing clears, or when a position (every hand plus the public
    ledger) repeats -- the last of which terminates a sampling gate, which
    can price a reverse exchange positive once a trade has moved the ledger.
    """
    # `pending` describes *this* event only; what a `PendingGate` recorded
    # last event no longer describes hands that may have since moved.
    game.pending = []
    state = game._state
    me = game.current_player
    actor_gate = game.gates[me] if game.gates is not None else None
    budget = params_of(actor_gate).trade_offer_budget
    if budget == 0 or (game.gates is None and gate is None):
        return []

    views: dict[int, "View"] = {}

    def view(seat: int) -> "View":
        got = views.get(seat)
        if got is None:
            got = game.state(seat)
            views[seat] = got
        return got

    executed: list[Trade] = []
    seen: set[tuple] = set()
    while budget < 0 or len(executed) < budget:
        position = _position_key(state, game.ledger)
        if position in seen:
            break  # a sampling gate came back round; the event is over
        seen.add(position)
        views.clear()
        best = _best_clearing(game, me, gate, view)
        if best is None:
            break
        them, received, gain_me, gain_them = best
        before = [hand[:] for hand in state.hands]
        exchange(state, me, them, received)
        game.ledger.apply_hand_diff(before, state.hands)
        executed.append(Trade(me, them, received, gain_a=gain_me, gain_b=gain_them))

    game.trades.extend(executed)
    game.trades_made += len(executed)
    return executed


def apply_trades(game: "Game", trades: Sequence[Trade]) -> None:
    """Execute an already-decided list of exchanges. For replay only: a
    replayed game has nobody seated to answer a gate. Does everything the
    live path does to the ledger, so a replayed position is the same
    position."""
    state = game._state
    for trade in trades:
        before = [hand[:] for hand in state.hands]
        exchange(state, trade.a, trade.b, trade.received)
        game.ledger.apply_hand_diff(before, state.hands)
        game.trades.append(trade)
        game.trades_made += 1


def _two_sided(received: object) -> bool:
    """Whether `received` is an exchange at all: one integer count per
    resource, something moving each way. A signed bundle's two sides are
    disjoint by construction."""
    if not isinstance(received, (tuple, list)) or len(received) != NUM_RESOURCES:
        return False
    if any(isinstance(n, bool) or not isinstance(n, Integral) for n in received):
        return False
    return any(n > 0 for n in received) and any(n < 0 for n in received)


def _validate_exchange(game: "Game", actor: int, counterparty: int, received: Bundle) -> None:
    """The referee's checks on an exchange between `actor` and `counterparty`
    (`received` signed towards `actor`), before any gate is asked: both are
    seats at this table and neither is locked, the two differ, the bundle is
    two-sided (`_two_sided`), the phase is `Phase.MAIN`, one of the two is
    the current player (a seat trades on its own turn with anyone, or on
    another seat's turn with that seat only), and both sides cover their half
    from the true hands. Raises `ValueError` naming the first check that
    fails.

    No card cap is checked here. How many cards a seat will move is that
    seat's own, refused at its own consent, so a referee cap would either
    duplicate it or overrule it.
    """
    from ..game import Phase  # local: avoids a game/trading import cycle

    seats = game._state.num_players
    for seat in (actor, counterparty):
        if isinstance(seat, bool) or not isinstance(seat, int) or not 0 <= seat < seats:
            raise ValueError(f"seat {seat!r} is not a seat at this table")
        if seat in game.locked:
            raise ValueError(f"seat {seat} is locked and cannot trade")
    if actor == counterparty:
        raise ValueError("a seat cannot trade with itself")
    if not _two_sided(received):
        raise ValueError(f"{received!r} is not an exchange: one count per resource, cards both ways")
    if game.phase is not Phase.MAIN:
        raise ValueError(f"trading is only open in {Phase.MAIN.name}, not {game.phase.name}")
    if game.current_player not in (actor, counterparty):
        raise ValueError(f"neither seat {actor} nor {counterparty} is the current player")
    state = game._state
    give = [max(0, -n) for n in received]
    take = [max(0, n) for n in received]
    if not _covers(state, actor, give):
        raise ValueError(f"seat {actor} cannot cover its side of this trade")
    if not _covers(state, counterparty, take):
        raise ValueError(f"seat {counterparty} cannot cover its side of this trade")


def _covers(state: GameState, seat: int, wanted: Sequence[int]) -> bool:
    """`holds`, or for a hand this state hides (an observed state) the only
    part of it that can be checked: enough cards. The host is the referee of
    the rest."""
    hand = state.hands[seat]
    if is_hidden(hand):
        return len(hand) >= sum(wanted)
    return holds(state, seat, wanted)


def execute_agreed(
    game: "Game",
    actor: int,
    counterparty: int,
    received: Bundle,
    *,
    ask_actor: bool,
    ask_counterparty: bool,
    price_actor: bool = False,
    price_counterparty: bool = False,
) -> Trade:
    """Execute one exchange between `actor` and `counterparty` (`received`
    signed towards `actor`) agreed outside the clearing house.

    `_validate_exchange` runs first; each side whose `ask_` flag is set then
    has its own gate asked, and the trade is refused with `ValueError`
    unless that gain clears that gate's own floor. That is for an exchange
    a seat has *not* yet agreed to, such as a bundle composed by hand and put
    to a bot. A clear flag means that seat's consent is already given: its
    submission, its accept, or its pick.

    `price_actor`/`price_counterparty` record that side's gain on the
    `Trade` without letting it refuse -- what the round's drivers pass for a
    gate that agreed in the round, so a census still reads what each side
    made of the exchange. A side neither asked nor priced records `0.0`.
    """
    _validate_exchange(game, actor, counterparty, received)
    gates = game.gates
    gain_a = 0.0
    gain_b = 0.0
    if ask_actor or price_actor:
        trader = gates[actor] if gates is not None else None
        gain_a = consent(trader, game.state(actor), received, counterparty, role="actor")
        if ask_actor and not clears_floor(gain_a, trader):
            raise ValueError(f"seat {actor} does not want this exchange")
    if ask_counterparty or price_counterparty:
        trader = gates[counterparty] if gates is not None else None
        mirror = tuple(-n for n in received)
        gain_b = consent(trader, game.state(counterparty), mirror, actor, role="responder")
        if ask_counterparty and not clears_floor(gain_b, trader):
            raise ValueError(f"seat {counterparty} does not want this exchange")

    state = game._state
    before = [hand[:] for hand in state.hands]
    exchange(state, actor, counterparty, received)
    game.ledger.apply_hand_diff(before, state.hands)
    trade = Trade(actor, counterparty, received, gain_a=gain_a, gain_b=gain_b)
    game.trades.append(trade)
    game.trades_made += 1
    return trade


def _best_clearing(
    game: "Game",
    me: int,
    gate: "Callable[[int, View, Bundle, int], float] | None",
    view,
) -> tuple[int, Bundle, float, float] | None:
    """The clearing deal `game.trade_rule` ranks highest, as
    `(counterparty, bundle, actor gain, counterparty gain)`, or `None`.

    Gates are asked in two batches: `me`'s once over every candidate, then
    each distinct counterparty's once over the subset that cleared `me`'s
    floor. Among candidates clearing both floors, `"egalitarian"` maximises
    the smaller gain, `"nash"` their product, `"actor"` the acting seat's
    own; ties break on the acting seat's gain, then canonical bundle order
    (`_tie`), then the lower counterparty seat, for determinism only. A
    seated gate that does not trade (`max_offers=0`) is never a
    counterparty. Batching needs `game.gates`; the `gate` callable is
    the unseated fallback.
    """
    state = game._state
    traders = game.gates
    actor_gate = traders[me] if traders is not None else gate
    skipped = frozenset(game.locked)
    if traders is not None:
        skipped |= {s for s, g in enumerate(traders) if s != me and not params_of(g).trades}
    candidates = list(_candidates(
        state, me, skipped, params_of(actor_gate).enumeration_cards
    ))
    if not candidates:
        return None

    def ask(seat: int, seat_view, receiveds: list[Bundle], counterparties: list[int]) -> list[float]:
        if traders is not None:
            return valued_many(traders[seat], seat_view, receiveds, counterparties)
        return [gate(seat, seat_view, r, c) for r, c in zip(receiveds, counterparties)]

    def gate_of(seat: int) -> object:
        # Whose floor a gain is held to: the seated trader's, or the bare
        # callable's own `trade_floor` when there are no seated gates.
        return traders[seat] if traders is not None else gate

    receiveds = [received for _them, received in candidates]
    thems = [them for them, _received in candidates]
    mine = ask(me, view(me), receiveds, thems)

    # Group the candidates clearing `me`'s floor by counterparty, order
    # preserved within each group: one call per counterparty.
    by_counterparty: dict[int, list[int]] = {}
    for i, gain in enumerate(mine):
        if clears_floor(gain, gate_of(me)):
            by_counterparty.setdefault(thems[i], []).append(i)
    if not by_counterparty:
        return None

    theirs: dict[int, float] = {}
    for them, indices in by_counterparty.items():
        mirrors = [tuple(-n for n in receiveds[i]) for i in indices]
        answers = ask(them, view(them), mirrors, [me] * len(indices))
        for i, gain in zip(indices, answers):
            theirs[i] = gain

    rule = game.trade_rule
    if rule not in TRADE_RULES:
        raise ValueError(f"unknown trade rule: {rule!r}")

    def key(i: int) -> tuple:
        gain_me = mine[i]
        gain_them = theirs[i]
        if rule == "egalitarian":
            primary = min(gain_me, gain_them)
        elif rule == "nash":
            primary = gain_me * gain_them
        else:  # "actor"
            primary = gain_me
        # `max` breaks ties on canonical bundle order, then the lower
        # counterparty seat.
        return (primary, gain_me, *_tie(receiveds[i]), -thems[i])

    cleared = [i for i, gain in theirs.items() if clears_floor(gain, gate_of(thems[i]))]
    if not cleared:
        return None
    winner = max(cleared, key=key)
    return thems[winner], receiveds[winner], mine[winner], theirs[winner]


# ---------------------------------------------------------------------------
# The trade round. Everything above this line is the automatic clearing house.


class Offer(NamedTuple):
    """One seat's broadcast offer: `actor` proposes `received` -- signed
    positive towards `actor` -- to every other seat at once.

    `any` is an offer's "any" cards, signed towards `actor` like `received`:
    `any > 0`, the actor takes that many cards of the responder's choosing
    ("my sheep for any card"); `any < 0`, it gives that many, of the
    responder's naming ("any card for your ore"). An offer with any cards is
    *open* (`is_open`): an invitation to counter. It cannot be accepted as it
    stands, only countered, and a counter to it is an ordinary one, which
    the actor picks or refuses like any other."""

    actor: int
    received: Bundle
    any: int = 0


# `Response.kind`'s legal values.
RESPONSE_ACCEPT = "accept"
RESPONSE_COUNTER = "counter"
RESPONSE_PASS = "pass"


class Response(NamedTuple):
    """One seat's answer to an `Offer`. `bundle` is `None` only for
    `"pass"` and is **always signed towards the offer's actor**, never
    towards `seat` -- so `pick` prices every non-`"pass"` response alike."""

    seat: int
    kind: str
    bundle: Bundle | None = None


def is_open(offer: Offer) -> bool:
    """Whether `offer` holds any cards: answered only by a counter naming them."""
    return bool(offer.any)


def _belief_candidates(
    view: "View", me: int, counterparty: int, max_cards: int
) -> list[Bundle]:
    """Every candidate `me` could counter `counterparty` with, bundles
    signed towards `me`. Shaped like `_candidates` but enumerated over
    `view.known` rather than the true hands, so a counter is bounded by the
    ledger's certified lower bound and never by a hand `me` may not read."""
    give_options = list(_hand_multisets(view.known[me], max_cards))
    if not give_options:
        return []
    receive_options = list(_hand_multisets(view.known[counterparty], max_cards))
    out: list[Bundle] = []
    for given in give_options:
        for received in receive_options:
            if any(g and r for g, r in zip(given, received)):
                continue  # the two sides must not share a resource
            out.append(tuple(r - g for r, g in zip(received, given)))
    return out


def _estimate_many(
    gate: object, view: "View", candidates: Sequence[tuple[int, Bundle]],
    own_gains: list[float],
) -> list[float]:
    """`gate`'s best estimate of each candidate's *counterparty*-side gain:
    `gate.estimate_many(view, candidates)` when the gate has an opponent
    model, else `own_gains`, so a plain gate offers what is best for itself.
    Neither path reaches a hidden hand."""
    fn = getattr(gate, "estimate_many", None)
    if fn is not None:
        return [float(x) for x in fn(view, list(candidates))]
    return own_gains


def default_offer(
    gate: object, view: "View", candidates: Sequence[tuple[int, Bundle]]
) -> int | None:
    """The default `offer(view, candidates) -> index | None` for a gate with
    only `gains_many`: the candidate maximising the actor's own gain among
    those clearing the actor's floor on *both* readings -- its own gain and
    the estimated counterparty gain, which is in the actor's own units -- or
    `None`, this seat passing. Ties break as `_best_clearing`'s do.

    The own-gain floor is what makes an offer a promise: an offer that is
    accepted executes, so the actor offers only what it would itself sign.
    """
    if not candidates:
        return None
    receiveds = [b for _, b in candidates]
    thems = [c for c, _ in candidates]
    own_gains = valued_many(gate, view, receiveds, thems)
    if not any(clears_floor(gain, gate) for gain in own_gains):
        return None
    estimates = _estimate_many(gate, view, candidates, own_gains)
    eligible = [
        i for i in range(len(candidates))
        if clears_floor(estimates[i], gate) and clears_floor(own_gains[i], gate)
    ]
    if not eligible:
        return None

    return max(eligible, key=lambda i: (own_gains[i], *_tie(receiveds[i]), -thems[i]))


def default_respond(gate: object, view: "View", offer: Offer) -> Response:
    """The default `respond(view, offer) -> Response` for a gate with only
    `gains_many`: accept when this seat's own gain on the offered exchange
    clears its own floor and its own hand covers it; else counter with the
    `counter_menu` bundle maximising this seat's own gain among those whose
    estimated actor gain also clears the floor; else pass.

    A counter must clear this seat's own floor too, because it binds this
    seat the moment the actor picks it.
    """
    me = view.perspective
    actor = offer.actor
    received_for_me = tuple(-n for n in offer.received)
    # `view.known[me]` is exact for the perspective's own hand, so checking
    # coverage here costs nothing in information and saves a later failure.
    covers = all(view.known[me][r] >= n for r, n in enumerate(offer.received) if n > 0)
    gain = valued(gate, view, received_for_me, actor)
    if covers and clears_floor(gain, gate):
        return Response(me, RESPONSE_ACCEPT, offer.received)
    return _best_counter(gate, view, actor)


def _best_counter(gate: object, view: "View", actor: int) -> Response:
    """`default_respond`'s counter: the `counter_menu` bundle maximising this
    seat's own gain among those whose estimated actor gain also clears the
    floor, signed towards `actor`; else a pass."""
    me = view.perspective
    candidates = [received for _them, received in counter_menu(gate, view, actor)]
    if candidates:
        counterparties = [actor] * len(candidates)
        own_gains = valued_many(gate, view, candidates, counterparties)
        if not any(clears_floor(gain, gate) for gain in own_gains):
            return Response(me, RESPONSE_PASS)
        estimates = _estimate_many(gate, view, list(zip(counterparties, candidates)), own_gains)
        eligible = [
            i
            for i in range(len(candidates))
            if clears_floor(estimates[i], gate) and clears_floor(own_gains[i], gate)
        ]
        if eligible:
            best = max(eligible, key=lambda i: (own_gains[i], *_tie(candidates[i])))
            return Response(me, RESPONSE_COUNTER, tuple(-n for n in candidates[best]))

    return Response(me, RESPONSE_PASS)


def _answer_concrete(gate: object, view: "View", offer: Offer) -> Response:
    """`gate`'s own answer to a concrete `offer`: its `respond`, else
    `default_respond`."""
    respond_fn = getattr(gate, "respond", None)
    return respond_fn(view, offer) if respond_fn is not None else default_respond(gate, view, offer)


def default_respond_any(gate: object, view: "View", offer: Offer) -> Response:
    """The default answer to an open `offer`, an invitation to counter: the
    counter this seat would send any offer (`default_respond`'s), else a
    pass. Never an acceptance."""
    return _best_counter(gate, view, offer.actor)


def default_pick(
    gate: object, view: "View", responses: Sequence[Response]
) -> int | None:
    """The default `pick(view, responses) -> index | None` for a gate with
    only `gains_many`: the answer with the highest own gain above this
    gate's own floor, acceptance or counter, else `None`. Ties break as
    `_best_clearing`'s do.

    An acceptance is priced by who took the offer: the offer went out with
    some seat in mind, and the same cards from another seat -- one closer to
    winning, say -- can be a trade this gate would never have proposed. A
    counter is a new proposal, priced the same way."""
    eligible = [
        (i, r) for i, r in enumerate(responses)
        if r.kind != RESPONSE_PASS and r.bundle is not None
    ]
    if not eligible:
        return None
    receiveds = [r.bundle for _, r in eligible]
    counterparties = [r.seat for _, r in eligible]
    gains = valued_many(gate, view, receiveds, counterparties)

    best_idx: int | None = None
    best_key: tuple | None = None
    for (i, r), gain in zip(eligible, gains):
        if not clears_floor(gain, gate):
            continue
        key = (gain, *_tie(r.bundle), -r.seat)
        if best_key is None or key > best_key:
            best_key, best_idx = key, i
    return best_idx


def has_room(view: "View", seat: int, received: Sequence[int]) -> bool:
    """Whether `seat`'s public hand has room for its side of `received`
    (signed towards the perspective, so `seat` gives the positive counts):
    every card it is not certified to hold fits among its untyped cards.
    False means the record proves `seat` cannot make the exchange. A gate's
    opponent model reads this, never the menu: an offer may still ask for
    anything, since some other seat may hold it."""
    known = view.known[seat]
    return sum(max(0, n - k) for n, k in zip(received, known)) <= view.unknown[seat]


def fit(view: "View", seat: int, received: Sequence[int]) -> int:
    """How far `received` (signed towards the perspective, so `seat` gives
    the positive counts) fits what `seat` has shown the table
    (`PublicLedger.show`): one for giving it a resource it wants, one for
    asking only for cards it has offered away, within the counts it offered.
    Zero where it has shown nothing."""
    row = view.ledger.seats[seat]
    gives_want = any(n < 0 and row.want[r] > 0 for r, n in enumerate(received))
    asks = [(r, n) for r, n in enumerate(received) if n > 0]
    asks_waste = bool(asks) and all(n <= row.waste[r] for r, n in asks)
    return int(gives_want) + int(asks_waste)


def known_candidates(
    view: "View", counterparties: Sequence[int], max_cards: int = ENUMERATION_CARDS
) -> list[tuple[int, Bundle]]:
    """Every exchange the perspective could propose to `counterparties` from
    what it *knows*: its exact hand against each seat's ledger-certified
    holdings (`_belief_candidates`), at most `max_cards` a side, bundles
    signed towards the perspective. The default counter menu
    (`counter_menu`): a counter is put to the actor alone, and this is every
    exchange the record proves that one seat can cover."""
    me = view.perspective
    out: list[tuple[int, Bundle]] = []
    for them in counterparties:
        if them == me:
            continue
        out.extend(
            (them, received) for received in _belief_candidates(view, me, them, max_cards)
        )
    return out


def open_candidates(
    view: "View", counterparties: Sequence[int], max_cards: int = ENUMERATION_CARDS
) -> list[tuple[int, Bundle]]:
    """Every exchange the perspective could put to the table: each give its
    exact hand covers against any cards the other seats hold between them,
    at most `max_cards` a side, the two sides on disjoint resources, bundles
    signed towards the perspective -- once per seat in `counterparties`, so
    a gate can price who takes it. The default offer menu (`menu`).

    Nothing a counterparty holds, is certified to hold or could be sampled
    to hold narrows it. An offer is broadcast: the actor asks for the card
    it needs, and whoever holds it may take it. What bounds the ask is the
    table's total, which is exact and public -- every card is in the bank or
    a hand, so the other seats hold what the bank and the actor do not -- and
    an ask past it is one no seat can take. A responder's coverage is still
    the responder's, and the engine checks both hands when cards move."""
    me = view.perspective
    give_options = list(_hand_multisets(view.known[me], max_cards))
    if not give_options:
        return []
    held = [max(0, min(max_cards, BANK_PER_RESOURCE - view.state.bank[r] - view.known[me][r]))
            for r in range(NUM_RESOURCES)]
    receive_options = list(_hand_multisets(held, max_cards))
    bundles = [
        tuple(r - g for r, g in zip(received, given))
        for given in give_options
        for received in receive_options
        if not any(g and r for g, r in zip(given, received))
    ]
    return [(them, b) for them in counterparties if them != me for b in bundles]


def menu(
    gate: object, view: "View", counterparties: Sequence[int], *,
    turn: int | None = None, already_offered: Sequence[Bundle] = (),
) -> list[tuple[int, Bundle]]:
    """The offers `gate` may put to `counterparties`: its own
    `candidates(view, counterparties, turn=, already_offered=)` when it has
    one -- aimed at what the hand wants, or the one fragment a plan calls
    for next -- else `open_candidates` at the gate's own proposal width.
    `turn` and `already_offered` are for a planning gate, whose menu depends
    on where in the turn it is. Whatever the menu, it is built from the gate's
    own view, and it holds nothing giving more than the gate's own
    `max_give_cards`: a gate never offers what it would itself refuse. The
    engine re-checks coverage when cards move, never before."""
    params = params_of(gate)
    candidates_fn = getattr(gate, "candidates", None)
    if candidates_fn is not None:
        out = list(candidates_fn(
            view, list(counterparties), turn=turn, already_offered=already_offered
        ))
    else:
        out = open_candidates(view, counterparties, params.enumeration_cards)
    cap = params.max_give_cards
    if cap is None:
        return out
    return [(them, b) for them, b in out if not _gives_more(b, cap)]


def _gives_more(received: object, cap: int) -> bool:
    """Whether `received` (signed towards its proposer) gives more than `cap`
    cards. Something that is not a bundle at all gives nothing here; the
    offer stage's own shape check refuses it."""
    try:
        return sum(max(-int(n), 0) for n in received) > cap
    except (TypeError, ValueError):
        return False


def counter_menu(gate: object, view: "View", actor: int) -> list[tuple[int, Bundle]]:
    """The counters `gate` may answer `actor`'s offer with: its own
    `candidates(view, [actor])` when it has one -- called with `turn=None`,
    which is how a gate tells a counter from an offer -- else
    `known_candidates` at the gate's own proposal width. Unlike an offer,
    a counter goes to the actor alone, so it asks only for what the record
    proves the actor holds."""
    candidates_fn = getattr(gate, "candidates", None)
    if candidates_fn is not None:
        return list(candidates_fn(view, [actor], turn=None, already_offered=()))
    return known_candidates(view, [actor], params_of(gate).enumeration_cards)


def _well_formed(view: "View", received: object) -> bool:
    """Whether a gate-supplied bundle is one the actor can put to the table:
    right width, integer counts, both sides nonempty and disjoint, and the
    giving side covered by the actor's own hand. A menu is the gate's word;
    this is the shape check the engine's own enumeration gives for free."""
    if not _two_sided(received):
        return False
    hand = view.known[view.perspective]
    return all(max(-n, 0) <= int(h) for n, h in zip(received, hand))


def offer(
    game: "Game", gates: Sequence[object], already_offered: set[Bundle] | None = None
) -> Offer | None:
    """The round's first stage: the current player's gate picks one offer
    from its `menu`, through its `offer(view, candidates)` or `default_offer`.
    `None` when there is nothing to broadcast: not `Phase.MAIN`, a locked or
    unseated actor, no candidate left, or the gate passing.

    `already_offered`, when given, excludes bundles already put to the table
    this turn and gains the one returned -- except where the gate's
    `allow_repeated_offer(view, bundle, turn=)` claims a repeat, which is
    what lets a planned second fragment equal to a first that executed go
    out again while a refused offer stays excluded.
    """
    from ..game import Phase  # local: avoids a game/trading import cycle

    if game.phase is not Phase.MAIN:
        return None
    me = game.current_player
    if me in game.locked:
        return None
    gate = gates[me]
    if gate is None:
        return None
    view = game.state(me)
    others = [s for s in range(game._state.num_players) if s != me and s not in game.locked]
    offered = already_offered if already_offered is not None else frozenset()
    repeat_fn = getattr(gate, "allow_repeated_offer", None)

    def admitted(received: Bundle) -> bool:
        if received not in offered:
            return True
        return repeat_fn is not None and bool(repeat_fn(view, received, turn=game.turns))

    candidates = [
        (them, tuple(received))
        for them, received in menu(gate, view, others, turn=game.turns, already_offered=offered)
        if _well_formed(view, received) and admitted(tuple(received))
    ]
    if not candidates:
        return None
    offer_fn = getattr(gate, "offer", None)
    index = (
        offer_fn(view, candidates)
        if offer_fn is not None
        else default_offer(gate, view, candidates)
    )
    if index is None or not (0 <= index < len(candidates)):
        return None
    _them, received = candidates[index]
    if already_offered is not None:
        already_offered.add(received)
    return Offer(me, received)


def respond(game: "Game", gate: object, seat: int, offer: Offer) -> Response:
    """The round's second stage, once per other seat: its gate's
    `respond(view, offer)`, else `default_respond`. The seat reads only its
    own view. An open offer (`is_open`) is its gate's `respond_any(view,
    offer)`, else `default_respond_any`. A gate that does not trade
    (`max_offers=0`) is not asked: it signs nothing, so it passes.

    The answer is the referee's to admit (`_admit`): it is always `seat`'s;
    an acceptance is of `offer` as it was made, and a pass where `seat`'s
    own hand does not cover its side or the offer is open; a counter must be
    a two-sided exchange of the table's width. Anything else is a pass."""
    if not params_of(gate).trades:
        return Response(seat, RESPONSE_PASS)
    view = game.state(seat)
    if is_open(offer):
        any_fn = getattr(gate, "respond_any", None)
        response = any_fn(view, offer) if any_fn is not None else default_respond_any(gate, view, offer)
    else:
        response = _answer_concrete(gate, view, offer)
    return _admit(view, seat, offer, response)


def _admit(view: "View", seat: int, offer: Offer, response: object) -> Response:
    """`response`, a gate's answer from `seat` to `offer`, as the referee
    admits it (`respond`). Coverage is read off `seat`'s own view, whose own
    hand is exact, so nothing hidden is consulted."""
    passed = Response(seat, RESPONSE_PASS)
    kind = getattr(response, "kind", None)
    if kind == RESPONSE_ACCEPT:
        if is_open(offer):
            return passed
        received = tuple(int(n) for n in offer.received)
        own = view.known[seat]
        if not all(own[r] >= n for r, n in enumerate(received) if n > 0):
            return passed
        return Response(seat, RESPONSE_ACCEPT, received)
    if kind == RESPONSE_COUNTER:
        bundle = getattr(response, "bundle", None)
        if not _two_sided(bundle):
            return passed
        return Response(seat, RESPONSE_COUNTER, tuple(int(n) for n in bundle))
    return passed


def pick(
    game: "Game", gate: object, actor: int, responses: Sequence[Response]
) -> Response | None:
    """The round's third stage: the actor's choice among `responses`, its
    gate's `pick(view, responses)` else `default_pick`. `None` when it
    declines them all, or picks a pass -- nothing to execute either way."""
    if not responses:
        return None
    view = game.state(actor)
    pick_fn = getattr(gate, "pick", None)
    chosen = (
        pick_fn(view, responses) if pick_fn is not None else default_pick(gate, view, responses)
    )
    if chosen is None or not (0 <= chosen < len(responses)):
        return None
    response = responses[chosen]
    if response.kind == RESPONSE_PASS or response.bundle is None:
        return None
    return response


def choose_and_execute(
    game: "Game", gates: Sequence[object], actor: int, responses: Sequence[Response]
) -> Trade | None:
    """The choice-and-execution half of a round (`pick`, then execution
    under the referee's own checks), separate from `trade_round` so a
    caller that assembled `responses` across several calls can resolve one
    too. Returns `None`, executing nothing, wherever `trade_round` would
    return `[]`.

    Safe to call after every new response whoever the actor is: a manual
    seat's `PendingGate.pick` always declines, so a human never resolves
    here and picks through its own explicit call instead.
    """
    actor_gate = gates[actor]
    if actor_gate is None:
        return None
    response = pick(game, actor_gate, actor, responses)
    if response is None:
        return None
    return _execute_round(game, gates, actor, response.seat, response.bundle)


def _execute_round(
    game: "Game", gates: Sequence[object], actor: int, counterparty: int, received: Bundle
) -> Trade | None:
    """Execute one round's chosen exchange between two bot gates. Both have
    already agreed -- the responder by its answer, the actor by its offer
    or its pick -- so neither is asked again; each side's gain is priced for
    the `Trade`'s record only. What can still stop it is the referee: a
    side that no longer covers its cards. Returns `None`, executing nothing,
    on the first check that fails. `gates` is installed on the game for the
    call's duration."""
    had = game.gates
    game.gates = tuple(gates)
    try:
        return execute_agreed(
            game, actor, counterparty, received,
            ask_actor=False, ask_counterparty=False,
            price_actor=True, price_counterparty=True,
        )
    except ValueError:
        return None
    finally:
        game.gates = had


def show(game: "Game", seat: int, received: Sequence[int]) -> None:
    """`seat` put `received` (signed towards `seat`) to the table: its offer,
    or an answer to one. Every seat sees it, so it goes on the public ledger
    (`PublicLedger.show`) as what `seat` wants and will give up. The referee
    checks here that `seat` covers what it gives; only then are those cards
    certified as held. `game.shown` keeps it for a record, placed among the
    turn's exchanges."""
    received = tuple(int(n) for n in received)
    give = [max(0, -n) for n in received]
    game.ledger.show(seat, received, holds=_covers(game._state, seat, give))
    game.shown.append((len(game.trades), int(seat), received))


def show_offer(game: "Game", offer: Offer) -> None:
    """`show` an offer, from its actor."""
    show(game, offer.actor, offer.received)


def show_response(game: "Game", actor: int, response: Response) -> None:
    """`show` an acceptance or a counter to `actor`'s offer, from the seat
    that gave it: its bundle is signed towards `actor`, so the seat's own
    side is its mirror. A pass shows nothing."""
    if response.kind == RESPONSE_PASS or response.bundle is None:
        return
    show(game, response.seat, tuple(-int(n) for n in response.bundle))


class CounterThread:
    """One thread of counters, between the mover and one `seat`, from `seat`'s
    counter to the mover's offer, `received` (signed towards the mover).

    The counter standing is put to the other side as an offer to it alone
    (`offer`), whose answer -- accept, counter, or pass -- `take` folds in:
    a counter becomes the one standing, from the side that gave it, so the
    mover and `seat` answer in turn. Every counter is between the mover and
    one seat: the mover executes every trade, so whose counter another
    counters changes nothing. The thread holds at most `counter_steps`
    counters, `seat`'s own the first, so the last can only be taken or left;
    a counter past that, or one repeating a bundle already put in it (`seen`,
    signed towards the mover), is a pass. The engine's own round runs one to
    the end (`counter_thread`); a hosted seat, one answer at a time
    (`hexset.seat.Seat.counter`)."""

    def __init__(self, mover: int, seat: int, received: Bundle, counter_steps: int,
                 seen: Iterable[Bundle] = ()) -> None:
        self.mover, self.seat, self.counter_steps = mover, seat, counter_steps
        self.received = tuple(received)     # the counter standing, signed towards the mover
        self.proposer, self.answerer, self.count = seat, mover, 1
        self.seen = set(seen) | {self.received}

    def offer(self) -> Offer:
        """The counter standing, as an offer from its proposer to the answerer."""
        return Offer(self.proposer, self._towards(self.proposer, self.received))

    def take(self, answer: Response) -> Response:
        """The answerer's `answer` to `offer`, as the thread allows it: an
        acceptance ends the thread, to execute `received`; a counter the
        thread has room for becomes the one standing; anything else is a
        pass."""
        if answer.kind == RESPONSE_ACCEPT:
            return answer
        if answer.kind == RESPONSE_COUNTER and answer.bundle is not None and self.count < self.counter_steps:
            received = self._towards(self.mover, self._towards(self.proposer, answer.bundle))
            if received not in self.seen:
                self.seen.add(received)
                self.received = received
                self.proposer, self.answerer, self.count = self.answerer, self.proposer, self.count + 1
                return answer
        return Response(answer.seat, RESPONSE_PASS)

    def _towards(self, side: int, received: Sequence[int]) -> Bundle:
        """`received`, signed towards the mover, as `side` holds it: the same
        for the mover, reversed for the seat -- and back again."""
        return tuple(int(n) for n in received) if side == self.mover else tuple(-int(n) for n in received)


def counter_thread(
    game: "Game", gates: Sequence[object], mover: int, seat: int, received: Bundle,
    counter_steps: int, seen: Iterable[Bundle] = (),
) -> Trade | None:
    """Run `seat`'s counter to the mover, `received` (signed towards the
    mover), as a `CounterThread` to its end: each side answers the counter
    standing with its own `respond`, as if it were an offer to it alone. An
    acceptance executes (`_execute_round`, the referee's checks); a pass
    ends it with nothing. Every answer the thread allows goes on the public
    ledger."""
    thread = CounterThread(mover, seat, received, counter_steps, seen)
    while True:
        proposer, answerer = thread.proposer, thread.answerer
        answer = thread.take(respond(game, gates[answerer], answerer, thread.offer()))
        if answer.kind == RESPONSE_PASS:
            return None
        show_response(game, proposer, answer)
        if answer.kind == RESPONSE_ACCEPT:
            return _execute_round(game, gates, mover, seat, thread.received)


def resolve_offer(
    game: "Game", gates: Sequence[object], offer: Offer, *, counter_steps: int = 1,
) -> list[Trade]:
    """The round after its broadcast: the offer goes on the public ledger
    (`show_offer`), every other seated, unlocked gate `respond`s once and
    each answer goes on it too (`show_response`), `choose_and_execute` lets the actor pick and executes
    it under the referee's checks, and the actor's gate is told how
    the round ended through `trade_round_finished` -- offer, answers and the
    executed `Trade` or `None` -- which is how a gate carrying a
    multi-broadcast plan learns whether its fragment actually cleared.

    Where the actor picks nothing and `counter_steps` allows more than the
    one counter a seat answers with, each counter in turn, in seat order,
    goes on as a `counter_thread` until one executes; a round still clears
    at most one exchange."""
    me = offer.actor
    show_offer(game, offer)
    # Each answer is public as it is given, as at the served table, so a
    # later seat answers having seen an earlier one's.
    responses = []
    for seat in range(game._state.num_players):
        if seat != me and seat not in game.locked and gates[seat] is not None:
            response = respond(game, gates[seat], seat, offer)
            show_response(game, me, response)
            responses.append(response)
    trade = choose_and_execute(game, gates, me, responses) if responses else None
    if trade is None and counter_steps > 1 and gates[me] is not None:
        for response in responses:
            if response.kind == RESPONSE_COUNTER and response.bundle is not None:
                trade = counter_thread(game, gates, me, response.seat, response.bundle,
                                       counter_steps, {tuple(offer.received)})
                if trade is not None:
                    break
    finished_fn = getattr(gates[me], "trade_round_finished", None)
    if finished_fn is not None:
        finished_fn(game.state(me), offer, responses, trade, turn=game.turns)
    return [] if trade is None else [trade]


def trade_round(
    game: "Game", gates: Sequence[object], already_offered: set[Bundle] | None = None,
    offers_made: list[Bundle] | None = None, counter_steps: int = 1,
) -> list[Trade]:
    """One round, driven synchronously: `offer`, then `resolve_offer` --
    every other seated gate `respond`s at once and `choose_and_execute`
    resolves it. The served table drives the same stages with a pause for
    seats that answer later; nothing about the round itself differs between
    the two.

    `gates` is a parameter rather than `game.gates`, so nothing here assumes
    the `Game` is seated. Nothing counts or caps rounds. `already_offered`,
    when given, excludes bundles already put to the table this turn and
    gains the one broadcast here -- except where the actor's gate claims a
    repeat through `allow_repeated_offer`. `offers_made`, when given, gains
    the bundle actually broadcast; unlike `already_offered` it is a list, so
    a permitted repeat still registers and a caller looping over rounds can
    tell "offered again" from "had nothing to offer". A seat with no gate,
    and a locked seat, are never asked.

    Returns the one executed `Trade` as a one-element list, or `[]` if: the
    game is not in `Phase.MAIN`; the current player has no gate, no
    candidate, or passed on every one; nobody else answered; the actor
    declined every answer; or a side no longer covered the chosen exchange
    at execution.

    `counter_steps` is how many counters one exchange may carry, a seat's
    answer the first (`resolve_offer`, `counter_thread`); the default, one,
    is a counter the actor takes or leaves.
    """
    offered = offer(game, gates, already_offered)
    if offered is None:
        return []
    # `already_offered` is a set, so a permitted repeat does not grow it;
    # `offers_made` counts what actually went to the table, which is what a
    # caller looping over rounds has to stop on.
    if offers_made is not None:
        offers_made.append(offered.received)
    return resolve_offer(game, gates, offered, counter_steps=counter_steps)
