# SPDX-License-Identifier: GPL-3.0-only
"""Player-to-player trading: one event a turn, no trade actions at all.

Registered `agents/reference/trading-final.md`, superseding the shipped
"one-event" mechanic's public layer (`agents/reference/trading-design.md`
§8). Derivation and the reasoning behind every choice below:
`agents/reference/trading-theory.md`.

* **No public layer.** There is no published valuation vector any more.
  Nothing here is advertised, and nothing filters a candidate before a gate
  is asked about it.
* A candidate **bundle** `b` is signed counts from one seat's point of view,
  positive for what it receives.
* Each seat's **gate** is `gains_many(view, receiveds, counterparties) ->
  list[float]`: that seat's own private gain, in whatever unit its value is,
  for each candidate at once, read through that seat's own information-set
  `View`. A deal *clears* only when both sides' gain exceeds `TRADE_FLOOR`
  (τ) -- no seat's own gate can be forced into a trade it prices at or below
  the floor.
* Once a turn, after the roll and the robber, the engine enumerates every
  coverable candidate exchange, asks the current player's gate once over all
  of them, keeps the subset that clears the floor, asks each counterparty's
  gate once over its own accepted subset, keeps the subset of *that* which
  clears the floor, and clears the one candidate `Game.trade_rule` ranks
  highest -- `"egalitarian"` (the default): the smaller of the two private
  gains; `"nash"`: their product; `"actor"`: the current player's own gain.
  Ties break on the actor's own gain, then a canonical bundle order, then
  the lower counterparty seat, for determinism only. Then it loops: the
  private gates are re-evaluated on the position the last trade left, and
  clearing continues until nothing clears. There is no budget: the acting
  seat's own gain exceeds the floor at every step -- strictly positive,
  since `TRADE_FLOOR >= 0` -- the state space is finite, so no cycle is
  possible. `Game.max_trades` is an off switch (`0`), not a budget; `None`
  is the unbounded default.

There are no trade actions -- no propose, respond, accept or decline -- so
nothing here reads an opponent's hand on an actor's behalf and the action
space carries no trade slot to mask. The engine is the referee: it checks
coverage (`holds`) itself, which is a rules check, not information handed to
a player.

Candidate bundles are any signed bundle -- both sides nonempty, disjoint
resource sets, each side bounded by what that hand holds and by
`MAX_TRADE_CARDS` -- coverable from the true hands. `_candidates` enumerates
bundles, not one-for-one swaps: a 2-for-1 does not arise as a sequence of
1-for-1 steps that each have to clear both gates on their own.

A trade moves at most `MAX_TRADE_CARDS` cards on either side, a table rule
applied to every seat alike (99.1% of the human corpus is at or under it):
`_hand_multisets` is the one place the automatic event, the
`trade/acceptable` preview and pending offers all draw from, and
`execute_trade` enforces the same limit on a manually composed bundle.

Every enumerated candidate is now put to the acting seat's gate -- there is
no cheap public-surplus pre-filter left to skip a seat that would refuse
everything (`agents/reference/trading-theory.md` §5-6): the one
approximation this mechanic makes is the gate itself, and everything
downstream of it is exact.

## The trade round (the served table's protocol)

Everything above -- `trade_event`/`_best_clearing`, fired once a turn from
`hexset.game.enter_main`/`move_robber_to` -- is unchanged and stays the
engine's own automatic clearing: what the arena, the bench, `record_game`
and the gym all still play under, and what a self-play run's acceptance
labels are drawn from (`agents/reference/trading-final.md`, "Training").

`trade_round` (below) is a second, separate protocol for a *served* game
(`hexset.server`) with a human, an LLM or a foreign bot at the table:
propose-and-respond rather than exhaustive clearing, because a served
table's seats answer through a wire, not through a synchronous function
call. A session that wants it seats `game.max_trades = 0` -- the existing
off switch, not a new flag -- so the automatic event no-ops itself on
every `enter_main`/`move_robber_to`, and calls `trade_round(game, gates)`
itself instead, as many times a turn as the acting seat wants: nothing
here counts rounds or caps them, since the floor and the card cap already
bound what one round can move. A game nobody serves never calls this at
all and never notices it exists.

One round: the current player's gate broadcasts one offer (`Bot.offer`,
new); every other seated gate answers once (`Bot.respond`, new) --
accept, counter, or pass; the actor's gate picks one answer to execute
(`Bot.choose`, new) or declines them all. A gate that only has
`gains_many` gets a sensible default for all three (`default_offer`/
`default_respond`/`default_choose`), the same "structural, not by
inheritance" convention `valued`/`valued_many` already give `accepts`/
`accepts_many`; a gate with a real opponent model instead implements
`estimate_many(view, candidates) -> list[float]` (heximax, search2), read
by the defaults wherever they would otherwise have to guess a counterparty
that has not answered yet with "this seat's own gain stands in for
theirs."
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, NamedTuple, Sequence

from .board.terrain import NUM_RESOURCES
from .state import GameState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .game import Game
    from .view import View

Bundle = tuple[int, ...]

# How a seat judges a concrete exchange, in one candidate's worth: its own
# private gain, a float, positive means it wants the trade. `view` is always
# that seat's own `Game.state(seat)` information set, never the true state
# and never another seat's.
Gate = Callable[[int, "View", Bundle, int], float]

# `Game.trade_rule`'s legal values (`_best_clearing`'s selection key).
# `"egalitarian"` is the shipped default; `"nash"` and `"actor"` are lab-only
# alternatives (`agents/reference/trading-final.md`, item 4).
TRADE_RULES: tuple[str, ...] = ("egalitarian", "nash", "actor")

# The clearing floor τ: a private gain must exceed this, on both sides, for
# a candidate to be admitted at all -- `clears_floor` below is the one
# predicate every admission point in the engine reads, so the floor is
# applied in exactly one place. Not a chosen number: it is the gate's
# measured resolution under paired chance (the trade lab's phase 3,
# `agents/reference/trading-final.md` item 4). 300 bank positions x 8 paired
# chance streams, each played traded and untraded to a 600-action cap by
# four heximax seats under identical dice, steals and draws; no bin of
# claimed gain showed a realised gain distinguishable from zero, so by the
# registered rule the floor is the instrument's own resolution -- the
# half-width of the pooled 95% interval on the acting seat's realised
# win-rate gain, 0.0197. A gate that claims less than this is claiming
# something no outcome can verify, and the table does not honour it.
TRADE_FLOOR: float = 0.0197

# A trade moves at most this many cards on either side -- a table rule, not
# a knob: the human corpus puts 99.1% of recorded trades at or under three
# cards a side (`agents/reference/trading-final.md`). Read in one place,
# `_hand_multisets`, which is what `_candidates` -- and so `trade_event`,
# the server's `trade/acceptable` preview and pending offers -- all draw
# from; `execute_trade` enforces the same limit again on a bundle composed
# outside that enumeration.
MAX_TRADE_CARDS = 3


def clears_floor(gain: float) -> bool:
    """Whether a private gain clears `TRADE_FLOOR` -- the one predicate
    `trade_event` (both the acting seat's and each counterparty's subset),
    `execute_trade` (the counterparty's gain) and the server's
    `trade/acceptable` preview all read, so every admission point applies
    the same rule."""
    return gain > TRADE_FLOOR


# How many of a trade event's candidates a network gate will score, at most,
# in its one batched forward -- a network gate's own bound on its evaluation,
# not a bound the engine itself places on the candidate set (see
# `hexset.clients.onnxbot.NetworkBot.accepts_many`, which reads this).
# Registered fallback (`agents/reference/trading-design.md`'s post-data note,
# "gate re-run with batched gates: 3.0-3.3x, still failing"): batching cut
# calls, not rows -- ~85% of events clear nothing, so a batched ask over
# every clearing candidate still scores everything a sequential ask would
# have stopped short of.
NETWORK_GATE_ROWS = 32


class Trade(NamedTuple):
    """One executed exchange. `received` is signed, positive towards `a`.

    `gain_a`/`gain_b` are each side's own private gain from this exchange, as
    computed by the gate that cleared it -- `0.0` when a side's gain was
    never evaluated (the manual `execute_trade` path never asks the
    proposer's own gate, so `gain_a` stays at its default there).
    """

    a: int
    b: int
    received: Bundle
    gain_a: float = 0.0
    gain_b: float = 0.0


def valued(trader: object, view: "View", received: Bundle, counterparty: int) -> float:
    """`trader`'s own private gain from one candidate exchange, or the
    default for a bot with no trading surface at all: never trade.

    Implemented once here, rather than by inheritance, so it holds for a bot
    that satisfies `hexset.bots.search2.Bot` only structurally: a bot from
    another package that defines `gains_many` (the primary surface),
    `accepts_many`, or plain `accepts` all work, in that preference order,
    and one that defines none of them never trades.
    """
    return valued_many(trader, view, [received], [counterparty])[0]


def valued_many(
    trader: object,
    view: "View",
    received: Sequence[Bundle],
    counterparties: Sequence[int],
) -> list[float]:
    """Batched `valued`: `trader`'s gain on every one of `received` at once,
    in the same order, via `gains_many` when `trader` defines it -- what lets
    `_best_clearing` ask a seat's gate once per event instead of once per
    candidate bundle.
    """
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


def bundle(**amounts: int) -> Bundle:
    """A resource bundle by name, for tests and hand-written trades."""
    from .board.terrain import Resource

    counts = [0] * NUM_RESOURCES
    for name, count in amounts.items():
        counts[Resource[name.upper()]] = count
    return tuple(counts)


def holds(state: GameState, player: int, wanted: Sequence[int]) -> bool:
    hand = state.hands[player]
    return all(hand[r] >= n for r, n in enumerate(wanted))


def exchange(state: GameState, a: int, b: int, received: Sequence[int]) -> None:
    """Move `received` between two seats, positive counts towards `a`.

    The one primitive that moves cards between players. It does not check
    legality; `trade_event` has already established that both sides can
    cover their halves.
    """
    hand_a = state.hands[a]
    hand_b = state.hands[b]
    for r, n in enumerate(received):
        hand_a[r] += n
        hand_b[r] -= n


def one_for_one(given: int, wanted: int) -> Bundle:
    """The signed bundle for "I give one `given`, I receive one `wanted`"."""
    out = [0] * NUM_RESOURCES
    out[wanted] += 1
    out[given] -= 1
    return tuple(out)


def _hand_multisets(hand: Sequence[int]):
    """Every distinct nonempty multiset of cards this hand can cover, up to
    `MAX_TRADE_CARDS` cards total, as nonnegative counts by resource index.

    Bounded both by what the hand holds and by `MAX_TRADE_CARDS` -- the
    walk below prunes a branch the moment its running total would exceed
    the cap, rather than generating every hand-coverable multiset and
    filtering after, so a hand rich in several resources at quantity still
    costs no more than the cap allows. A generator, not a list: `_candidates`
    below only ever needs to walk it.

    Only resources actually held are walked, so a hand with one or two
    resource types never touches the combinations the other three could
    have contributed.
    """
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

    yield from walk(0, MAX_TRADE_CARDS)


def _candidates(state: GameState, me: int, locked: frozenset[int]):
    """`(counterparty, bundle)` for every coverable exchange `me` could
    propose: any nonempty multiset given and any nonempty multiset received,
    each side at most `MAX_TRADE_CARDS` cards, the two sides on disjoint
    resource sets, both coverable from the true hands -- the engine is the
    referee, so no gate ever has to check this itself.

    A generator, deliberately: only resources actually held are walked. No
    filter runs before this -- there is no public vector left to check a
    seat's or a counterparty's zero against, so every locked-out seat aside,
    every coverable candidate reaches the gates in `_best_clearing`.
    """
    give_options = list(_hand_multisets(state.hands[me]))
    if not give_options:
        return
    for them in range(state.num_players):
        if them == me or them in locked:
            continue
        receive_options = list(_hand_multisets(state.hands[them]))
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


def trade_event(game: "Game", gate: Gate) -> list[Trade]:
    """Clear every deal the current player and one other seat both gain
    from.

    Called by `hexset.game.run_trade_event`, once a turn, on the transition
    into `Phase.MAIN` -- so every driver (the arena's loop, the gym, the
    server, a search stepping its own copy) gets it without having to
    remember to. Returns the trades executed by *this* call, in the order
    they cleared, and appends them to `game.trades`, which accumulates
    across every call within the turn.

    Private gates are asked in two batches per counterparty considered: once
    for the acting seat over every coverable candidate, then once per
    distinct counterparty over the acting seat's subset that clears the
    floor with it. Among the candidates both sides clear `TRADE_FLOOR` on,
    `game.trade_rule` picks the winner (`_best_clearing`). No budget: the
    loop runs until nothing clears.

    The single engine limit is the assertion below: an event never revisits
    a position (every seat's hand plus the public ledger). The acting seat's
    own gain exceeds the floor at every clearing -- strictly positive, since
    `TRADE_FLOOR >= 0` -- so a position that comes back means a gate is
    broken (not strictly increasing in the acting seat's own value), and
    that is a bug to surface rather than a knob to tune. It is deliberately
    not a count of trades: a legitimate event of one- and two-card exchanges
    can run longer than there are cards on the table without ever repeating
    a position -- self-play against a network gate does so in about one
    event in two hundred.
    """
    # A snapshot of *this* event only: whatever a manual seat's `PendingGate`
    # recorded last event no longer describes hands that may have since
    # moved, so it is dropped before this event records its own.
    game.pending = []
    if game.max_trades == 0:
        return []

    state = game._state
    me = game.current_player
    views: dict[int, "View"] = {}

    def view(seat: int) -> "View":
        got = views.get(seat)
        if got is None:
            got = game.state(seat)
            views[seat] = got
        return got

    executed: list[Trade] = []
    seen: set[tuple] = set()
    while game.max_trades is None or len(executed) < game.max_trades:
        position = _position_key(state, game.ledger)
        assert position not in seen, (
            "a trade event revisited a position: a gate is not strictly "
            "increasing in the acting seat's own value"
        )
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
    """Execute an already-decided list of exchanges. For replay only.

    A replayed game has nobody seated to answer a gate, so its trade event
    clears nothing and the trades that *did* happen are re-executed from the
    record instead (`hexset.record.advance`,
    `hexset.server.webplay.GameSession.restore`). Everything the live path
    does to the ledger is done here too, so a replayed position is the same
    position, ledger included.
    """
    state = game._state
    for trade in trades:
        before = [hand[:] for hand in state.hands]
        exchange(state, trade.a, trade.b, trade.received)
        game.ledger.apply_hand_diff(before, state.hands)
        game.trades.append(trade)
        game.trades_made += 1


def execute_trade(game: "Game", proposer: int, counterparty: int, received: Bundle) -> Trade:
    """A manually composed exchange between `proposer` and `counterparty`,
    signed positive towards `proposer` -- the negotiation interface's one
    engine entry point for a bundle that was never enumerated by
    `_candidates`.

    Re-validates, in order, and raises `ValueError` naming the first check
    that fails:

    1. **The card cap.** Neither side moves more than `MAX_TRADE_CARDS`
       cards -- the same limit `_hand_multisets`/`_candidates` enumerate
       under, refused here the same way an uncoverable bundle is.
    2. **Coverage.** Both sides can actually pay their half, from the true
       hands (`holds`) -- the engine is the referee, exactly as it is for
       the automatic event.
    3. **The counterparty's own gain.** Must exceed `TRADE_FLOOR`
       (`clears_floor`), read through the counterparty's own gate (`valued`)
       on its own view. The proposer's own gate is never asked -- submitting
       a trade *is* the proposer's consent, so a seat may compose a bundle
       its own gate would refuse.

    Also enforces the turn-timing rule: one of `proposer`/`counterparty`
    must be `game.current_player`, and the phase must be `Phase.MAIN` -- a
    seat proposes on its own turn to anyone, or during another seat's turn
    naming only that seat.

    On success, moves the cards (`exchange`), certifies the diff on the
    ledger, appends to `game.trades` -- the same two calls `trade_event`
    makes for an automatic clearing -- and returns the `Trade`.
    """
    from .game import Phase  # local: avoids a game/trading import cycle

    if proposer == counterparty:
        raise ValueError("a seat cannot trade with itself")
    if game.phase is not Phase.MAIN:
        raise ValueError(f"trading is only open in {Phase.MAIN.name}, not {game.phase.name}")
    if game.current_player not in (proposer, counterparty):
        raise ValueError(
            f"neither seat {proposer} nor {counterparty} is the current player"
        )

    state = game._state
    # `received` is signed towards `proposer`: its negative entries are what
    # `proposer` must give, and its positive entries -- what `proposer`
    # receives -- are exactly what `counterparty` must give, the same cards
    # seen from the other side.
    give = [max(0, -n) for n in received]
    take = [max(0, n) for n in received]
    if sum(give) > MAX_TRADE_CARDS or sum(take) > MAX_TRADE_CARDS:
        raise ValueError(f"a trade moves at most {MAX_TRADE_CARDS} cards a side")
    if not holds(state, proposer, give):
        raise ValueError(f"seat {proposer} cannot cover its side of this trade")
    if not holds(state, counterparty, take):
        raise ValueError(f"seat {counterparty} cannot cover its side of this trade")
    counterparty_received = tuple(-n for n in received)

    trader = game.gates[counterparty] if game.gates is not None else None
    view = game.state(counterparty)
    gain = valued(trader, view, counterparty_received, proposer)
    if not clears_floor(gain):
        raise ValueError(f"seat {counterparty} does not want this exchange")

    before = [hand[:] for hand in state.hands]
    exchange(state, proposer, counterparty, received)
    game.ledger.apply_hand_diff(before, state.hands)
    trade = Trade(proposer, counterparty, received, gain_b=gain)
    game.trades.append(trade)
    game.trades_made += 1
    return trade


def _best_clearing(
    game: "Game",
    me: int,
    gate: Gate,
    view,
) -> tuple[int, Bundle, float, float] | None:
    """The clearing deal `game.trade_rule` ranks highest, or `None`.

    Every coverable candidate is asked about, in two batches: `me`'s gate is
    asked once, over every candidate, via `valued_many`; then, only for the
    candidates that clear `TRADE_FLOOR` for `me` (`clears_floor`), each
    distinct counterparty's gate is asked once, over its own accepted
    subset. There is no cheap pre-filter left to rank candidates before a
    gate is asked -- the mechanic's one approximation is the gate itself
    (`agents/reference/trading-theory.md` §5) -- so every enumerated
    candidate costs one row in the acting seat's one batched call.

    Among the candidates that clear the floor on both sides, the winner is
    chosen by `game.trade_rule`: `"egalitarian"` maximises the smaller of the two
    gains; `"nash"` maximises their product; `"actor"` maximises the acting
    seat's own gain. Ties break on the acting seat's own gain, then a
    canonical bundle order, then the lower counterparty seat -- purely for
    determinism, since real-valued gains essentially never tie in practice.

    `game.gates` (the real trader objects, when the event has any --
    `run_trade_event` never reaches here otherwise) is what makes the
    batching possible; the single-candidate `gate` callable is the fallback
    for a direct caller with no seated `game.gates` (this module's own
    tests), where nothing can be batched because there is no trader object
    to call `gains_many` on.
    """
    state = game._state
    candidates = list(_candidates(state, me, game.locked))
    if not candidates:
        return None

    traders = game.gates

    def ask(seat: int, seat_view, receiveds: list[Bundle], counterparties: list[int]) -> list[float]:
        if traders is not None:
            return valued_many(traders[seat], seat_view, receiveds, counterparties)
        return [gate(seat, seat_view, r, c) for r, c in zip(receiveds, counterparties)]

    receiveds = [received for _them, received in candidates]
    thems = [them for them, _received in candidates]
    mine = ask(me, view(me), receiveds, thems)

    # Group the candidates that clear the floor for `me` by counterparty,
    # preserving order within each group -- exactly the candidates a
    # sequential loop would have asked that counterparty about, just
    # gathered into one call per counterparty instead of one call per
    # candidate.
    by_counterparty: dict[int, list[int]] = {}
    for i, gain in enumerate(mine):
        if clears_floor(gain):
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
        # Key 2 is always the actor's own gain, whatever the primary rule;
        # key 3 -- the negated bundle -- sorts so `max` picks the smallest
        # original bundle on a tie; key 4 -- the negated counterparty seat --
        # so `max` picks the lower one. Determinism only.
        canonical = tuple(-n for n in receiveds[i])
        return (primary, gain_me, canonical, -thems[i])

    cleared = [i for i, gain in theirs.items() if clears_floor(gain)]
    if not cleared:
        return None
    winner = max(cleared, key=key)
    return thems[winner], receiveds[winner], mine[winner], theirs[winner]


# ---------------------------------------------------------------------------
# The trade round: the served table's protocol (see the module docstring,
# "The trade round"). Everything above this line is the automatic clearing
# house and is untouched by any of it.


class Offer(NamedTuple):
    """One seat's broadcast offer: `actor` proposes `received` (signed
    positive towards `actor`, the same convention `_candidates` yields) to
    every other seat at once."""

    actor: int
    received: Bundle


# `Response.kind`'s legal values.
RESPONSE_ACCEPT = "accept"
RESPONSE_COUNTER = "counter"
RESPONSE_PASS = "pass"


class Response(NamedTuple):
    """One seat's answer to an `Offer`.

    `seat` is who answered. `bundle` is the exchange this response
    proposes, `None` only for `"pass"` -- and, whichever kind it is,
    **always signed towards the offer's own actor**, never towards `seat`:
    for `"accept"` this is exactly `offer.received` echoed back; for
    `"counter"` it is `seat`'s own preferred bundle, translated into the
    actor's terms. One convention for both kinds is what lets `choose`/
    `default_choose` price every non-`"pass"` response the same way,
    without first working out which kind it is looking at.
    """

    seat: int
    kind: str
    bundle: Bundle | None = None


def _belief_candidates(view: "View", me: int, counterparty: int) -> list[Bundle]:
    """Every candidate `me` could counter `counterparty` with, using only
    what `me` can actually know: `me`'s own hand exactly (`view.known[me]`,
    exact since a `View` is always exact about its own perspective) and
    `counterparty`'s hand from the ledger's certified lower bound
    (`view.known[counterparty]`) -- never the true hand, which `me` may not
    read. Mirrors `_candidates`' own enumeration shape (same disjoint-sides
    rule, same `MAX_TRADE_CARDS` cap via `_hand_multisets`) over a per-seat
    *known* count instead of `state.hands`, since a responding seat's
    counter-offer is bounded by what it knows the actor holds, not by the
    actor's true hand (`hexset.trading`'s "the trade round", item 2).
    Bundles are signed towards `me`, the same convention `_candidates`
    uses.
    """
    give_options = list(_hand_multisets(view.known[me]))
    if not give_options:
        return []
    receive_options = list(_hand_multisets(view.known[counterparty]))
    out: list[Bundle] = []
    for given in give_options:
        for received in receive_options:
            if any(g and r for g, r in zip(given, received)):
                continue  # the two sides must not share a resource
            out.append(tuple(r - g for r, g in zip(received, given)))
    return out


def _estimate_many(
    gate: object, view: "View", candidates: Sequence[tuple[int, Bundle]]
) -> list[float]:
    """`gate`'s best estimate of each `(counterparty, bundle)` candidate's
    *counterparty*-side gain -- `gate.estimate_many(view, candidates)` when
    the gate provides one (heximax, search2: read through this seat's own
    belief or, for search2, the true state it is sanctioned to read -- see
    each bot's own `estimate_many`), else this gate's own gain on every
    candidate (`valued_many`), "so a plain gate offers what is best for
    itself" (`hexset.trading`'s "the trade round", item 1). Never reaches a
    hidden hand either way: the fallback reads only the gate's own
    information, the same as `gains_many` always has.
    """
    fn = getattr(gate, "estimate_many", None)
    if fn is not None:
        return [float(x) for x in fn(view, list(candidates))]
    receiveds = [b for _, b in candidates]
    counterparties = [c for c, _ in candidates]
    return valued_many(gate, view, receiveds, counterparties)


def default_offer(
    gate: object, view: "View", candidates: Sequence[tuple[int, Bundle]]
) -> int | None:
    """The default `offer(view, candidates) -> index | None` for a gate that
    only has `gains_many`: the candidate maximising the actor's own gain
    (`valued_many`) among those whose *estimated* counterparty gain clears
    `TRADE_FLOOR` (`_estimate_many`, `clears_floor`) -- so a gate with a
    real opponent model offers what it believes the table will actually
    take, and a plain gate falls back to "what is best for itself" (its own
    gain stands in for the estimate too). `None` when nothing clears the
    estimate -- this seat passes rather than broadcasts. Ties break on the
    actor's own gain again, then canonical bundle order, then the lower
    counterparty seat -- the same purely-deterministic tie-break
    `_best_clearing` uses, for the same reason (real-valued gains
    essentially never tie in practice).
    """
    if not candidates:
        return None
    receiveds = [b for _, b in candidates]
    thems = [c for c, _ in candidates]
    own_gains = valued_many(gate, view, receiveds, thems)
    estimates = _estimate_many(gate, view, candidates)
    eligible = [i for i in range(len(candidates)) if clears_floor(estimates[i])]
    if not eligible:
        return None

    def key(i: int) -> tuple:
        canonical = tuple(-n for n in receiveds[i])
        return (own_gains[i], canonical, -thems[i])

    return max(eligible, key=key)


def default_respond(gate: object, view: "View", offer: Offer) -> Response:
    """The default `respond(view, offer) -> Response` for a gate that only
    has `gains_many`: accept outright when this seat's own gain on the
    offered exchange clears `TRADE_FLOOR`; else counter with the bundle
    this seat can actually propose (`_belief_candidates`: its own hand
    exactly, the actor's hand from the ledger's known lower bound)
    maximising this seat's own gain among those whose *estimated* actor
    gain clears the floor (`_estimate_many`); else pass. Ties among
    counters break the same way `default_offer`'s do.
    """
    me = view.perspective
    actor = offer.actor
    received_for_me = tuple(-n for n in offer.received)
    gain = valued(gate, view, received_for_me, actor)
    if clears_floor(gain):
        return Response(me, RESPONSE_ACCEPT, offer.received)

    candidates = _belief_candidates(view, me, actor)
    if candidates:
        counterparties = [actor] * len(candidates)
        own_gains = valued_many(gate, view, candidates, counterparties)
        estimates = _estimate_many(gate, view, list(zip(counterparties, candidates)))
        eligible = [i for i in range(len(candidates)) if clears_floor(estimates[i])]
        if eligible:

            def key(i: int) -> tuple:
                canonical = tuple(-n for n in candidates[i])
                return (own_gains[i], canonical)

            best = max(eligible, key=key)
            return Response(me, RESPONSE_COUNTER, tuple(-n for n in candidates[best]))

    return Response(me, RESPONSE_PASS)


def default_choose(
    gate: object, view: "View", responses: Sequence[Response]
) -> int | None:
    """The default `choose(view, responses) -> index | None` for a gate
    that only has `gains_many`: the acceptance or counter with the highest
    own gain above `TRADE_FLOOR`, else `None`. Every non-`"pass"`
    response's bundle is already signed towards this seat (`Response`'s own
    convention), so this is one batched `valued_many` over them.
    """
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
        if not clears_floor(gain):
            continue
        canonical = tuple(-n for n in r.bundle)
        key = (gain, canonical, -r.seat)
        if best_key is None or key > best_key:
            best_key, best_idx = key, i
    return best_idx


def _execute_round(
    game: "Game", gates: Sequence[object], actor: int, counterparty: int, received: Bundle
) -> Trade | None:
    """Execute one round's chosen exchange, re-validating everything at the
    moment cards actually move -- coverage both sides, the card cap, and
    both sides' own gain read *fresh* through their own gate, never the
    reported estimate an offer, a response or a choice was built from.

    Unlike `execute_trade` (whose proposer is a human or LLM submitting a
    bundle it composed itself -- consent, so its own gate is never asked),
    both ends of a round are gates the engine itself asked to trade, and
    either can misreport (a stale estimate, a gate that changed its mind
    between answering and being chosen, an adversarial implementation) --
    so neither report is trusted without a fresh check here, the same
    "engine is the referee" property the clearing house has. Returns
    `None`, executing nothing, on the first check that fails, rather than
    raising: a round's chosen response failing here is an ordinary
    "nothing happened", not a caller error the way a hand-composed
    `execute_trade` bundle's failure is.
    """
    state = game._state
    give = [max(0, -n) for n in received]
    take = [max(0, n) for n in received]
    if sum(give) > MAX_TRADE_CARDS or sum(take) > MAX_TRADE_CARDS:
        return None
    if not holds(state, actor, give) or not holds(state, counterparty, take):
        return None

    gain_a = valued(gates[actor], game.state(actor), received, counterparty)
    if not clears_floor(gain_a):
        return None
    mirror = tuple(-n for n in received)
    gain_b = valued(gates[counterparty], game.state(counterparty), mirror, actor)
    if not clears_floor(gain_b):
        return None

    before = [hand[:] for hand in state.hands]
    exchange(state, actor, counterparty, received)
    game.ledger.apply_hand_diff(before, state.hands)
    trade = Trade(actor, counterparty, received, gain_a=gain_a, gain_b=gain_b)
    game.trades.append(trade)
    game.trades_made += 1
    return trade


def trade_round(game: "Game", gates: Sequence[object]) -> list[Trade]:
    """One live-table round: the current player's gate broadcasts one offer,
    every other seated gate answers once, and the actor's gate picks one
    answer to execute. See the module docstring, "The trade round", for how
    this differs from -- and coexists with -- the automatic clearing house
    (`trade_event`).

    `gates` is a parameter, not read off `game.gates`: a served table's own
    session already keeps its seat -> gate mapping (bots, `PendingGate`s)
    and drives this explicitly, rather than through the engine's own
    trade-event dispatch from `enter_main`/`move_robber_to`, so nothing
    here assumes a `Game` is even seated with `gates` at all. A caller runs
    this as many times a turn as the acting seat wants -- once per
    broadcast, nothing here counts rounds or caps them.

    Every method asked of a gate -- `offer`, `respond`, `choose` -- is read
    with `getattr`, falling back to this module's own `default_offer`/
    `default_respond`/`default_choose` for a gate that only has
    `gains_many` (or less), the same "structural, not by inheritance"
    convention `valued`/`valued_many` already give `accepts`/`accepts_many`.

    A responding seat with no gate at all (`gates[seat] is None`) is
    silently skipped, exactly as an unseated seat is invisible to the
    clearing house; a locked seat is never asked, exactly as
    `_candidates`/`trade_event` already never counterparty one.

    Returns the one `Trade` executed, as a one-element list, or `[]` if:
    the game is not in `Phase.MAIN`; the current player has no gate, no
    coverable candidate, or its own gate passed on every one of them;
    nobody else answered; the actor's own gate declined every answer; or
    the chosen exchange failed the engine's fresh re-check at execution
    (`_execute_round`) -- the same "engine is the referee" property the
    clearing house has: neither side's own report is trusted at the moment
    cards actually move.
    """
    from .game import Phase  # local: avoids a game/trading import cycle, as `execute_trade` does

    if game.phase is not Phase.MAIN:
        return []

    state = game._state
    me = game.current_player
    if me in game.locked:
        return []
    actor_gate = gates[me]
    if actor_gate is None:
        return []

    candidates = list(_candidates(state, me, game.locked))
    if not candidates:
        return []

    actor_view = game.state(me)
    offer_fn = getattr(actor_gate, "offer", None)
    index = (
        offer_fn(actor_view, candidates)
        if offer_fn is not None
        else default_offer(actor_gate, actor_view, candidates)
    )
    if index is None or not (0 <= index < len(candidates)):
        return []
    _them0, received0 = candidates[index]
    offer = Offer(me, received0)

    responses: list[Response] = []
    for seat in range(state.num_players):
        if seat == me or seat in game.locked:
            continue
        gate = gates[seat]
        if gate is None:
            continue
        seat_view = game.state(seat)
        respond_fn = getattr(gate, "respond", None)
        response = (
            respond_fn(seat_view, offer)
            if respond_fn is not None
            else default_respond(gate, seat_view, offer)
        )
        responses.append(response)
    if not responses:
        return []

    choose_fn = getattr(actor_gate, "choose", None)
    chosen = (
        choose_fn(actor_view, responses)
        if choose_fn is not None
        else default_choose(actor_gate, actor_view, responses)
    )
    if chosen is None or not (0 <= chosen < len(responses)):
        return []
    response = responses[chosen]
    if response.kind == RESPONSE_PASS or response.bundle is None:
        return []

    trade = _execute_round(game, gates, me, response.seat, response.bundle)
    return [] if trade is None else [trade]
