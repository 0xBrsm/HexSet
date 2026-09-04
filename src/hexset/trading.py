# SPDX-License-Identifier: GPL-3.0-only
"""Player-to-player trading: one event a turn, no trade actions at all.

The mechanic (the trading design's §8, "trading is one event", ratified
2026-09-02 and amended 2026-09-03):

* Every seat holds a **public valuation vector** `v in [-1, 1]^NUM_RESOURCES`
  -- positive means "I want more of this", negative "I would give this up".
  It is published by the seat's own bot or policy, never by the engine, and
  is all-zero until something publishes: a seat that has not spoken never
  trades.
* A candidate **bundle** `b` is signed counts from one seat's point of view,
  positive for what it receives. Its **public surplus** is `v_me . b` for me
  and `v_them . (-b)` for the counterparty. A deal is advertised when both
  are strictly positive.
* Each seat's **private gate** is its own judgement of the post-trade
  position (`accepts`), read through that seat's information-set view. A
  deal *clears* only when both public surpluses and both private gates say
  yes. No vector anyone publishes can force a trade a seat's own gate
  rejects; that veto is what makes clearing automatically safe.
* Once a turn, after the roll and the robber, and again after every MAIN
  action the current player takes -- build, buy, a bank/port trade, a
  development card (owner review against the rulebook, 2026-09-03: the
  event and the turn interleave, rather than one event before the first
  build) -- the engine clears deals for the current player -- best deal
  first (maximise the smaller public surplus; ties broken by the acting
  seat's own surplus, then the total, then a canonical order, purely for
  determinism) -- until nothing clears. There is no budget: the gate is
  re-evaluated after every executed trade and must be *strictly* positive,
  so the acting seat's own valuation strictly increases at each step, the
  state space is finite, and no cycle is possible. `Game.max_trades` is an
  off switch (`0`), not a budget; `None` is the unbounded default.

There are no trade actions -- no propose, respond, accept or decline -- so
nothing here reads an opponent's hand on an actor's behalf and the action
space carries no trade slot to mask. The engine is the referee: it checks
coverage (`holds`) itself, which is a rules check, not information handed to
a player.

Candidate bundles are the trading design's actual language: any signed
bundle -- both sides nonempty, disjoint resource sets, each side bounded
only by what that hand holds, no fixed cap (owner review, 2026-09-03) --
coverable from the true hands (owner review and PI correction, 2026-09-03,
`agents/reference/trading-design.md`'s post-data note). A 2-for-1 does not
arise as two 1-for-1 steps that each have to clear both gates on their own
-- that claim was checked and was false -- so `_candidates` enumerates
bundles, not swaps. Candidates are ranked by public surplus (cheap: a dot
product per side) and the private gates -- the expensive half, a position
evaluation each -- are asked in that rank order until one clears both or
candidates run out (the registered gate-budget ablation measured a budget
of 8/16/32 against unbounded and found unbounded both the strongest arm and
within cost: `agents/reference/trading-design.md`'s post-data note "the
gate budget goes away" -- there is no budget).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, NamedTuple, Sequence

import numpy as np

from .board.terrain import NUM_RESOURCES
from .state import GameState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .game import Game
    from .view import View

Bundle = tuple[int, ...]

# How a seat judges a concrete exchange. `view` is always that seat's own
# `Game.state(seat)` information set, never the true state and never another
# seat's. There is no `Valuation` callable any more: a seat's vector is
# published onto `Game.valuations` at its own decisions (`Game.publish`,
# driven by whichever driver is stepping the game), and `trade_event` only
# ever reads it -- see the module docstring's "one-event" account and
# `agents/reference/trading-design.md`'s post-data note on why the callback
# form was wrong (one forward per seat per lane per turn, uncollectable).
Gate = Callable[[int, "View", Bundle, int], bool]


class Trade(NamedTuple):
    """One executed exchange. `received` is signed, positive towards `a`."""

    a: int
    b: int
    received: Bundle


# What a seat that has published nothing advertises: no wants, no gives, so
# no bundle it is party to can ever have positive public surplus.
NO_VALUATION: tuple[float, ...] = (0.0,) * NUM_RESOURCES

# The unit a derived valuation is published in: `v_r = tanh(delta_V_r /
# VALUE_SCALE)`, where `delta_V_r` is a value head's own-row marginal for one
# more card of resource `r`. Lives here rather than beside any one bot's
# implementation so that every derived trader in this repo -- and dev-HexNet's
# `hexnet.policy.DerivedTrader`, which should import it from here rather than
# carry its own copy -- cites the same number. Fixed across checkpoints: the
# mean absolute one-card value-head marginal on the mover's own row over five
# trade-free games (`agents/reference/trading-design.md`'s post-data note,
# "HexNet lands contract 5"; dev-HexNet's `agents/scripts/value_scale.py`
# recomputes it).
VALUE_SCALE = 0.022126919066234662

# How many of a trade event's public-rank-ordered candidates a network gate
# will score, at most, in its one batched forward. `_best_clearing` ranks
# every coverable candidate by public surplus before asking any private gate
# (see its `ranked` list, built once per event) and asks `judged_many` over
# that list in rank order -- so a gate that only reads the first
# `NETWORK_GATE_ROWS` of `received` is reading the highest-ranked candidates
# this event considered, not an arbitrary prefix. Registered fallback
# (`agents/reference/trading-design.md`'s post-data note, "gate re-run with
# batched gates: 3.0-3.3x, still failing"): batching cut calls, not rows --
# ~85% of events clear nothing, so a batched ask over every clearing
# candidate still scores everything the sequential ask would have stopped
# short of. Bounding a network gate to its top-ranked rows and declining the
# rest costs at most ~5% of trades/turn (budget 32 read 0.250 against
# unbounded's 0.254 in the registered gate-budget ablation) because clearing
# deals sit near the top of the public ranking. The engine itself asks about
# every candidate and is unchanged; only a network gate's own evaluation is
# bounded. Lives here, beside `VALUE_SCALE`, so `hexnet.policy.
# NETWORK_GATE_ROWS` can import the same number rather than carry its own
# copy -- exactly the reason `VALUE_SCALE` lives here instead of beside one
# bot's implementation.
NETWORK_GATE_ROWS = 32


def published(trader: object, view: "View") -> Sequence[float]:
    """What `trader` advertises, or nothing at all.

    A seat whose bot defines no `valuation` never advertises and therefore
    never trades. That is `hexset.bots.Bot.valuation`'s documented default,
    implemented once here so it holds for *structural* implementers too --
    a bot from another package satisfies `Bot` by having `choose`, and must
    not have to inherit anything to sit at a table that trades.
    """
    method = getattr(trader, "valuation", None)
    return NO_VALUATION if method is None else method(view)


def judged(trader: object, view: "View", received: Bundle, counterparty: int) -> bool:
    """Whether `trader` will take this exchange. No `accepts`, no trade."""
    method = getattr(trader, "accepts", None)
    return False if method is None else bool(method(view, received, counterparty))


def judged_many(
    trader: object,
    view: "View",
    received: Sequence[Bundle],
    counterparties: Sequence[int],
) -> list[bool]:
    """Batched `judged`: `trader`'s verdict on every one of `received` at
    once, in the same order, via `accepts_many` when `trader` defines it.

    `hexset.bots.Bot.accepts_many`'s documented default -- loop over
    `accepts` -- is implemented here rather than by inheritance, exactly as
    `judged` already implements `accepts`'s own default, so it holds for a
    bot that satisfies `Bot` only structurally. This is what lets
    `_best_clearing` ask a seat's gate once per event instead of once per
    candidate bundle (`agents/reference/trading-design.md`'s post-data
    note, "the collector cost gate fails at 2.9-3.6x").
    """
    method = getattr(trader, "accepts_many", None)
    if method is not None:
        return [bool(x) for x in method(view, list(received), list(counterparties))]
    return [judged(trader, view, r, c) for r, c in zip(received, counterparties)]


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
    """Every distinct nonempty multiset of cards this hand can cover, as
    nonnegative counts by resource index.

    No cap: bounded only by what the hand holds (owner review, 2026-09-03,
    "no bundle-size cap" -- replacing a fixed 1..3-card limit). A generator,
    not a list: `_candidates` below only ever needs to walk it, and a hand
    with several resources in quantity can cover a real number of these, so
    building the full list before anything is filtered would hold all of it
    in memory for no reason.

    Only resources actually held are walked, so a hand with one or two
    resource types never touches the combinations the other three could
    have contributed.
    """
    resources = [r for r in range(NUM_RESOURCES) if hand[r] > 0]
    counts = [0] * NUM_RESOURCES

    def walk(idx: int):
        if idx == len(resources):
            if any(counts):
                yield tuple(counts)
            return
        r = resources[idx]
        for n in range(hand[r] + 1):
            counts[r] = n
            yield from walk(idx + 1)
        counts[r] = 0

    yield from walk(0)


def _candidates(
    state: GameState,
    me: int,
    locked: frozenset[int],
    vectors: Sequence[Sequence[float]] | None = None,
):
    """`(counterparty, bundle)` for every coverable exchange `me` could
    propose: any nonempty multiset given and any nonempty multiset received,
    the two sides on disjoint resource sets, both coverable from the true
    hands -- the engine is the referee, so neither a vector nor a gate ever
    has to check this itself (trading-design.md's owner review and PI
    correction, 2026-09-03; the owner's follow-up review the same day drops
    the 1..3-card cap the original correction had: cost is bounded by the
    public-surplus filter downstream, in `_best_clearing`, not by a limit
    here).

    A generator, deliberately: only resources actually held are walked, and
    `_best_clearing` decides how to rank the result (a plain loop or a
    `numpy` batch, by size) without this function needing to order or
    pre-materialise anything for a hand fertile enough to advertise many
    bundles.

    `vectors`, when given, is `game.valuations`: a seat published nothing
    (`NO_VALUATION`, all zero) can never clear a trade in either role, since
    `mine = dot(v_me, b)` and `theirs = dot(v_them, -b)` are both exactly
    zero for every candidate `_rank_candidates_loop`/`_rank_candidates_
    vectorized` would go on to filter out anyway (`mine <= 0.0` / `theirs <=
    0.0`) -- so a zero vector on `me` or on a given `them` is checked here,
    before either side's hand is even walked, purely to skip the wasted
    enumeration; the set of candidates this yields when it does run is
    unchanged (a caller that omits `vectors` gets the old, unfiltered
    enumeration, for tests that exercise bundle enumeration on its own
    terms).
    """
    if vectors is not None and not any(vectors[me]):
        return
    give_options = list(_hand_multisets(state.hands[me]))
    if not give_options:
        return
    for them in range(state.num_players):
        if them == me or them in locked:
            continue
        if vectors is not None and not any(vectors[them]):
            continue
        receive_options = list(_hand_multisets(state.hands[them]))
        for given in give_options:
            for received in receive_options:
                if any(g and r for g, r in zip(given, received)):
                    continue  # the two sides must not share a resource
                yield them, tuple(r - g for r, g in zip(received, given))


def trade_event(game: "Game", gate: Gate) -> list[Trade]:
    """Clear every deal the current player and one other seat both want.

    Called by `hexset.game.run_trade_event`, once on the transition into
    `Phase.MAIN` and again after every MAIN action the current player takes
    (owner review against the rulebook, 2026-09-03: trade and build
    interleave) -- so every driver (the arena's loop, the gym, the server, a
    search stepping its own copy) gets it without having to remember to.
    Returns the trades executed by *this* call, in the order they cleared,
    and appends them to `game.trades`, which accumulates across every call
    within the turn.

    **Reads `game.valuations`; publishes nothing.** Every seat's vector was
    already written there by `Game.publish`, at that seat's own last
    decision -- a driver's job, not this function's (a callback here that
    asked each seat fresh was tried and measured: one forward per seat per
    lane per turn, the one place a batched collector stopped batching). The
    vectors are therefore already fixed for the whole event by the time this
    runs; only the private gates move as cards change hands, which is what
    makes the second ore worth less than the first. `gate(seat, view,
    bundle, counterparty)` is that seat's private judgement of one concrete
    exchange; both sides must return True.

    Private gates are asked in rank order (`_best_clearing`) until one
    candidate clears both or candidates run out -- no budget. The registered
    ablation (`agents/reference/trading-design.md`'s post-data note "the
    gate budget goes away") measured a cap of 8/16/32 candidates against
    unbounded and found unbounded both the strongest arm and within the
    cost ceiling, so the cap was deleted rather than tuned.

    The single engine limit is the assertion below: an event cannot execute
    more trades than there are cards on the table. It can only fire if a
    gate is broken (a gate that is not strictly increasing in the acting
    seat's own valuation), and that is a bug to surface rather than a knob
    to tune.
    """
    # A snapshot of *this* event only (PI ratification,
    # `docs/negotiation-interface.md` decision 2): whatever a manual seat's
    # `PendingGate` recorded last event no longer describes hands that may
    # have since moved, so it is dropped before this event records its own.
    game.pending = []
    if game.max_trades == 0:
        return []

    state = game._state
    me = game.current_player
    vectors = game.valuations
    views: dict[int, "View"] = {}

    def view(seat: int) -> "View":
        got = views.get(seat)
        if got is None:
            got = game.state(seat)
            views[seat] = got
        return got

    # The assertion's ceiling, measured before anything moves: any exchange
    # only moves cards between two hands, never creates or destroys one, so
    # the total is fixed for the whole event regardless of bundle size.
    cards = sum(sum(hand) for hand in state.hands)

    executed: list[Trade] = []
    while game.max_trades is None or len(executed) < game.max_trades:
        views.clear()
        best = _best_clearing(game, me, vectors, gate, view)
        if best is None:
            break
        them, received = best
        before = [hand[:] for hand in state.hands]
        exchange(state, me, them, received)
        game.ledger.apply_hand_diff(before, state.hands)
        executed.append(Trade(me, them, received))
        assert len(executed) <= cards, "a trade event outran the cards on the table"

    game.trades.extend(executed)
    game.trades_made += len(executed)
    return executed


def apply_trades(game: "Game", trades: Sequence[Trade]) -> None:
    """Execute an already-decided list of exchanges. For replay only.

    A replayed game has nobody seated to publish or judge, so its trade
    event clears nothing and the trades that *did* happen are re-executed
    from the record instead (`hexset.record.advance`,
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
    signed positive towards `proposer` (`docs/negotiation-interface.md` §1) --
    the negotiation interface's one engine entry point for a bundle that was
    never enumerated by `_candidates`.

    Re-validates, in order, and raises `ValueError` naming the first check
    that fails:

    1. **Coverage.** Both sides can actually pay their half, from the true
       hands (`holds`) -- the engine is the referee, exactly as it is for the
       automatic event.
    2. **The counterparty's public surplus, a hard rule.** `dot(v[counterparty],
       -received)` must be strictly positive: the counterparty's own
       advertised vector must already say this exchange helps it. The
       proposer's own surplus is never checked -- submitting a trade *is* the
       proposer's consent (PI ratification, decision 4), so a seat may
       compose a bundle its own vector calls bad for itself.
    3. **The counterparty's private gate.** `judged` against `game.gates
       [counterparty]` and that seat's own view (`game.state(counterparty)`)
       -- the same call the automatic event makes for `them`. **The
       proposer's own gate is never asked**, for the same reason as (2).

    Also enforces the turn-timing rule (`docs/negotiation-interface.md` §2):
    one of `proposer`/`counterparty` must be `game.current_player`, and the
    phase must be `Phase.MAIN` -- a seat proposes on its own turn to anyone,
    or during another seat's turn naming only that seat.

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
    if not holds(state, proposer, give):
        raise ValueError(f"seat {proposer} cannot cover its side of this trade")
    take = [max(0, n) for n in received]
    if not holds(state, counterparty, take):
        raise ValueError(f"seat {counterparty} cannot cover its side of this trade")
    counterparty_received = tuple(-n for n in received)

    vectors = game.valuations
    theirs = sum(vectors[counterparty][r] * n for r, n in enumerate(counterparty_received))
    if theirs <= 0.0:
        raise ValueError(f"seat {counterparty} has not advertised wanting this exchange")

    trader = game.gates[counterparty] if game.gates is not None else None
    view = game.state(counterparty)
    if not judged(trader, view, counterparty_received, proposer):
        raise ValueError(f"seat {counterparty} declined this trade")

    before = [hand[:] for hand in state.hands]
    exchange(state, proposer, counterparty, received)
    game.ledger.apply_hand_diff(before, state.hands)
    trade = Trade(proposer, counterparty, received)
    game.trades.append(trade)
    game.trades_made += 1
    return trade


# Encodes a signed bundle as one integer that sorts exactly the way the
# bundle's own tuple does lexicographically (resource 0 most significant),
# so key 4's "canonical bundle order" is one more vectorisable arithmetic
# column instead of a per-candidate Python tuple comparison. 1000 is well
# above any hand size this engine (or Catan) can produce, so `component +
# _BUNDLE_OFFSET` is always nonnegative and the mixed-radix encoding below
# is monotonic in the original component.
_BUNDLE_OFFSET = 1000
_BUNDLE_BASE = 2 * _BUNDLE_OFFSET + 1
# A measured threshold (see the module's history): below it, per-candidate
# Python loops beat the fixed cost of building `numpy` arrays; at and above
# it, `numpy`'s vectorised arithmetic wins by a growing margin, and a
# fertile late-game hand can advertise tens of thousands of bundles at once
# now that nothing caps a candidate's size.
_VECTORIZE_ABOVE = 32


def _best_clearing(
    game: "Game",
    me: int,
    vectors: Sequence[Sequence[float]],
    gate: Gate,
    view,
) -> tuple[int, Bundle] | None:
    """The clearing deal ranked highest, or None.

    Rank keys (owner review, 2026-09-03, "the tie-break" -- replacing fewer
    cards/canonical/lower-seat as the *whole* rule): (1) the smaller of the
    two public surpluses, highest first -- maximin, so the party who gains
    less from a deal still gains the most one is available; (2) the current
    player's own surplus, highest first -- the rulebook gives the acting
    seat the choice among the deals on offer, so among equally fair ones it
    takes the better one for itself; (3) the total surplus, highest first;
    (4) a canonical order over the bundle, then the lower counterparty seat
    -- determinism only, since real-valued surpluses essentially never tie
    on keys 1-3 in practice.

    Candidates are ranked by public surplus alone (cheap: a dot product per
    side) and the private gates -- the expensive half -- are asked in that
    rank order, with no budget: the first candidate both gates accept *is*
    the maximiser over the whole clearing set (the registered ablation,
    `agents/reference/trading-design.md`'s post-data note "the gate budget
    goes away", found a cap only ever suppressed clearing trades that
    unbounded found within cost).

    **The gates themselves are asked in two batches, not one per candidate**
    (`agents/reference/trading-design.md`'s post-data note, "the collector
    cost gate fails at 2.9-3.6x"): `me`'s gate is asked once, over every
    ranked candidate, via `judged_many`; then, only for the candidates `me`
    accepted, each distinct counterparty's gate is asked once, over its own
    accepted subset. The winner is still the first candidate in rank order
    where both sides said yes -- identical to the old one-candidate-at-a-
    time loop's result, since a batched `accepts_many` is required to agree
    with `accepts` row for row -- but the whole clearing attempt costs at
    most `1 + (distinct counterparties)` gate calls instead of up to `2 *
    len(ranked)`. `game.gates` (the real trader objects, when the event has
    any -- `run_trade_event` never reaches here otherwise) is what makes the
    batching possible; the single-candidate `gate` callable is the fallback
    for a direct caller with no seated `game.gates` (this module's own
    tests), where nothing can be batched because there is no trader object
    to call `accepts_many` on.
    """
    state = game._state
    candidates = list(_candidates(state, me, game.locked, vectors))
    if not candidates:
        return None

    if len(candidates) < _VECTORIZE_ABOVE:
        ranked = _rank_candidates_loop(me, vectors, candidates)
    else:
        ranked = _rank_candidates_vectorized(me, vectors, candidates)
    if not ranked:
        return None

    traders = game.gates

    def ask(seat: int, seat_view, receiveds: list[Bundle], counterparties: list[int]) -> list[bool]:
        if traders is not None:
            return judged_many(traders[seat], seat_view, receiveds, counterparties)
        return [gate(seat, seat_view, r, c) for r, c in zip(receiveds, counterparties)]

    receiveds = [received for received, _them in ranked]
    thems = [them for _received, them in ranked]
    mine_ok = ask(me, view(me), receiveds, thems)

    # Group the candidates `me` accepted by counterparty, preserving rank
    # order within each group -- exactly the candidates the old sequential
    # loop would have asked that counterparty about, just gathered into one
    # call per counterparty instead of one call per candidate.
    by_counterparty: dict[int, list[int]] = {}
    for i, ok in enumerate(mine_ok):
        if ok:
            by_counterparty.setdefault(thems[i], []).append(i)

    theirs_ok = [False] * len(ranked)
    for them, indices in by_counterparty.items():
        mirrors = [tuple(-n for n in receiveds[i]) for i in indices]
        answers = ask(them, view(them), mirrors, [me] * len(indices))
        for i, ok in zip(indices, answers):
            theirs_ok[i] = ok

    for i, (received, them) in enumerate(ranked):
        if mine_ok[i] and theirs_ok[i]:
            return them, received
    return None


def _rank_candidates_loop(
    me: int,
    vectors: Sequence[Sequence[float]],
    candidates: list[tuple[int, Bundle]],
) -> list[tuple[Bundle, int]]:
    """Plain Python: cheaper than building `numpy` arrays for a short list
    of candidates (see `_VECTORIZE_ABOVE`)."""
    v_me = vectors[me]
    rows: list[tuple[float, float, float, Bundle, int, int]] = []
    for them, received in candidates:
        mine = sum(v_me[r] * n for r, n in enumerate(received))
        if mine <= 0.0:
            continue
        v_them = vectors[them]
        theirs = sum(-v_them[r] * n for r, n in enumerate(received))
        if theirs <= 0.0:
            continue
        rows.append((min(mine, theirs), mine, mine + theirs, received, them, -them))

    def rank_key(row):
        min_surplus, mine, total, received, them, neg_them = row
        canonical = tuple(-n for n in received)  # descending sort -> smallest bundle first
        # Keys 1-3 are "highest first"; key 4 is "lowest first"
        # (determinism only) -- negating the bundle tuple element-wise
        # reverses its lexicographic order, so sorting descending on the
        # negated tuple picks the smallest original bundle on a tie.
        return (min_surplus, mine, total, canonical, neg_them)

    rows.sort(key=rank_key, reverse=True)
    return [(row[3], row[4]) for row in rows]


def _rank_candidates_vectorized(
    me: int,
    vectors: Sequence[Sequence[float]],
    candidates: list[tuple[int, Bundle]],
) -> list[tuple[Bundle, int]]:
    """`numpy`: one dot product per side over every candidate at once, and
    one `lexsort` over the rank keys (bundle and counterparty encoded as
    sortable integer columns, `_BUNDLE_OFFSET`/`_BUNDLE_BASE` above) --
    exact, not an approximation of the loop version, just vectorised."""
    bundles = np.array([b for _, b in candidates], dtype=np.int64)
    thems = np.array([t for t, _ in candidates], dtype=np.intp)
    v_me_arr = np.asarray(vectors[me], dtype=np.float64)
    vectors_arr = np.asarray(vectors, dtype=np.float64)

    bundles_f = bundles.astype(np.float64)
    mine = bundles_f @ v_me_arr
    theirs = -(bundles_f * vectors_arr[thems]).sum(axis=1)
    mask = (mine > 0.0) & (theirs > 0.0)
    if not np.any(mask):
        return []

    idx = np.flatnonzero(mask)
    min_surplus = np.minimum(mine[idx], theirs[idx])
    total = mine[idx] + theirs[idx]
    weights = _BUNDLE_BASE ** np.arange(NUM_RESOURCES - 1, -1, -1, dtype=np.int64)
    codes = (bundles[idx] + _BUNDLE_OFFSET) @ weights
    counterparties = thems[idx]

    # `lexsort`'s last key is primary; keys wanting "highest first" are
    # negated (`lexsort` is ascending); the canonical-order/lower-seat
    # determinism keys want "lowest first" and are used as-is.
    sort_idx = np.lexsort((counterparties, codes, -total, -mine[idx], -min_surplus))
    ranked_idx = idx[sort_idx]
    return [(candidates[i][1], int(thems[i])) for i in ranked_idx]


def checked_valuation(vector: Sequence[float], seat: int) -> tuple[float, ...]:
    """A published vector, validated: `NUM_RESOURCES` numbers in `[-1, 1]`.

    Rejected rather than clamped or padded -- a caller that sent something
    else meant something, and silently repairing it would hide the bug.
    `Game.publish` is the one place this runs live; it is public here so
    nothing needs its own copy (`hexset.server.webplay` used to keep one).
    """
    try:
        out = tuple(float(x) for x in vector)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"seat {seat} published a malformed valuation: {vector!r}") from exc
    if len(out) != NUM_RESOURCES:
        raise ValueError(
            f"seat {seat} published {len(out)} valuations, expected {NUM_RESOURCES}"
        )
    if any(x < -1.0 or x > 1.0 or x != x for x in out):
        raise ValueError(f"seat {seat} published a valuation outside [-1, 1]: {out}")
    return out


def publish_valuation(game: "Game", seat: int, trader: object) -> None:
    """The driver's fixed point: right after `seat`'s own decision, ask what
    `trader` now brings to the table and record it as `seat`'s current
    public vector (`Game.publish`), so it is there whenever anyone's trade
    event next reads it -- this seat's own turn included, since only the
    private gates move within one event, not the vectors.

    Every driver that steps a seated game calls this once per decision, for
    whichever seat just acted (`hexset.arena.play`, `hexset.bench.aivat`,
    `hexset.record.record_game`, the gym's auto-played opponents, the
    server's embedded bots). A trader with no `valuation` method (`Bot`'s
    documented default) is skipped outright rather than published as an
    explicit zero, since that is already every seat's resting value until it
    first speaks.
    """
    method = getattr(trader, "valuation", None)
    if method is not None:
        game.publish(seat, published(trader, game.state(seat)))
