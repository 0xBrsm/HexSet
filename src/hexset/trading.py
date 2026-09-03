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
evaluation each -- are asked in that order, capped at `GATE_BUDGET`
candidate pairs per clearing attempt so a hand that can advertise many
bundles at once cannot make one iteration price all of them; the cost bound
is this gate budget, not the enumeration, which is why the bundle-size cap
was dropped rather than kept as a second cost control. `Game.budget_binds`
counts how often the gate budget is the reason nothing cleared.
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

# A hand with several resources in quantity can advertise many bundles at
# once (no size cap, owner review 2026-09-03); pricing every one of them
# against two private gates -- a position evaluation each -- would make one
# clearing attempt as expensive as the hand is fertile. The budget is the
# cost bound, not a fixed enumeration limit: `Game.budget_binds` counts
# every time it is the reason nothing cleared. `trade_event`'s default
# (registered ablation, `agents/reference/trading-design.md`'s post-data
# note "bundles land": gate budget 8/16/32/unbounded, plus a ranking
# variant, measured before any of them replaces this default). `None` means
# unbounded: every candidate that clears the public filter is asked, in
# rank order, until one clears both private gates or none are left.
GATE_BUDGET = 8

# The two `_best_clearing` rank orders `trade_event`'s `order` keyword
# selects between. Both keep the maximin key first (the smaller of the two
# public surpluses, highest first) -- neither is a different clearing rule,
# only a different tie-break among candidates equally fair by that key:
# "maximin" (the default, owner review 2026-09-03, "the tie-break") breaks
# ties by the acting seat's own surplus, then the total surplus, then a
# canonical bundle order and the lower counterparty seat for determinism.
# "minimal_bundle" (the registered ablation's fifth arm) breaks ties by
# fewer total cards moved first instead -- a cost-motivated ordering that
# may raise the clearing rate within a fixed gate budget, distinct from the
# fairness tie-break it replaces -- falling back to the same canonical
# order and lower counterparty seat for determinism beyond that.
BUNDLE_ORDERS = ("maximin", "minimal_bundle")


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


def _candidates(state: GameState, me: int, locked: frozenset[int]):
    """`(counterparty, bundle)` for every coverable exchange `me` could
    propose: any nonempty multiset given and any nonempty multiset received,
    the two sides on disjoint resource sets, both coverable from the true
    hands -- the engine is the referee, so neither a vector nor a gate ever
    has to check this itself (trading-design.md's owner review and PI
    correction, 2026-09-03; the owner's follow-up review the same day drops
    the 1..3-card cap the original correction had: cost is bounded by the
    public-surplus filter and the gate budget downstream, in
    `_best_clearing`, not by a limit here).

    A generator, deliberately: only resources actually held are walked, and
    `_best_clearing` decides how to rank the result (a plain loop or a
    `numpy` batch, by size) without this function needing to order or
    pre-materialise anything for a hand fertile enough to advertise many
    bundles.
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


def trade_event(
    game: "Game",
    gate: Gate,
    *,
    gate_budget: int | None = GATE_BUDGET,
    order: str = "maximin",
) -> list[Trade]:
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

    `gate_budget` and `order` are keyword parameters, not module constants,
    so a caller can run the registered ablation
    (`agents/reference/trading-design.md`'s post-data note "bundles land")
    as one code path with different arguments rather than editing
    `GATE_BUDGET` per run. `gate_budget` bounds how many ranked candidates
    per clearing attempt are put to the two private gates (`None` is
    unbounded: every candidate the public filter passes is asked, in rank
    order, until one clears or none are left); `order` selects
    `_best_clearing`'s tie-break among candidates equal on the maximin key
    (`BUNDLE_ORDERS`). Defaults (`GATE_BUDGET`, `"maximin"`) reproduce
    today's behaviour exactly -- every existing call site is unaffected.

    The single engine limit is the assertion below: an event cannot execute
    more trades than there are cards on the table. It can only fire if a
    gate is broken (a gate that is not strictly increasing in the acting
    seat's own valuation), and that is a bug to surface rather than a knob
    to tune.
    """
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
        best = _best_clearing(
            game, me, vectors, gate, view, gate_budget=gate_budget, order=order
        )
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
    *,
    gate_budget: int | None,
    order: str,
) -> tuple[int, Bundle] | None:
    """The clearing deal ranked highest, or None.

    Rank keys under `order="maximin"` (the default; owner review,
    2026-09-03, "the tie-break" -- replacing fewer cards/canonical/lower-seat
    as the *whole* rule): (1) the smaller of the two public surpluses,
    highest first -- maximin, so the party who gains less from a deal still
    gains the most one is available; (2) the current player's own surplus,
    highest first -- the rulebook gives the acting seat the choice among the
    deals on offer, so among equally fair ones it takes the better one for
    itself; (3) the total surplus, highest first; (4) a canonical order over
    the bundle, then the lower counterparty seat -- determinism only, since
    real-valued surpluses essentially never tie on keys 1-3 in practice.
    Under `order="minimal_bundle"` (the registered ablation's fifth arm,
    `BUNDLE_ORDERS`), key (1) is unchanged and keys (2)-(3) are replaced by
    fewer total cards moved, highest first; key (4) still breaks any
    remaining tie.

    Candidates are ranked by public surplus alone (cheap: a dot product per
    side) and the gates -- the expensive half -- are asked in that order,
    capped at `gate_budget` candidates (`None` is unbounded: every ranked
    candidate is asked), so the first one within the budget both gates
    accept *is* the maximiser over the clearing set that budget could reach.
    """
    state = game._state
    candidates = list(_candidates(state, me, game.locked))
    if not candidates:
        return None

    if len(candidates) < _VECTORIZE_ABOVE:
        ranked, seen = _rank_candidates_loop(me, vectors, candidates, order)
    else:
        ranked, seen = _rank_candidates_vectorized(me, vectors, candidates, order)

    asking = ranked if gate_budget is None else ranked[:gate_budget]
    for received, them in asking:
        mirror = tuple(-n for n in received)
        if gate(me, view(me), received, them) and gate(them, view(them), mirror, me):
            return them, received
    if gate_budget is not None and seen > gate_budget:
        game.budget_binds += 1
    return None


def _rank_candidates_loop(
    me: int,
    vectors: Sequence[Sequence[float]],
    candidates: list[tuple[int, Bundle]],
    order: str,
) -> tuple[list[tuple[Bundle, int]], int]:
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
        if order == "minimal_bundle":
            # (1) maximin, unchanged; (2) fewer total cards first, in place
            # of the dropped own-surplus/total-surplus tie-break; (3) the
            # same determinism fallback as "maximin".
            total_cards = sum(abs(n) for n in received)
            return (min_surplus, -total_cards, canonical, neg_them)
        # Keys 1-3 are "highest first"; key 4 is "lowest first"
        # (determinism only) -- negating the bundle tuple element-wise
        # reverses its lexicographic order, so sorting descending on the
        # negated tuple picks the smallest original bundle on a tie.
        return (min_surplus, mine, total, canonical, neg_them)

    rows.sort(key=rank_key, reverse=True)
    return [(row[3], row[4]) for row in rows], len(rows)


def _rank_candidates_vectorized(
    me: int,
    vectors: Sequence[Sequence[float]],
    candidates: list[tuple[int, Bundle]],
    order: str,
) -> tuple[list[tuple[Bundle, int]], int]:
    """`numpy`: one dot product per side over every candidate at once, and
    one `lexsort` over the rank keys (bundle and counterparty encoded as
    sortable integer columns, `_BUNDLE_OFFSET`/`_BUNDLE_BASE` above) --
    exact, not an approximation of the loop version, just vectorised. Sorts
    the whole viable set (not only the top `gate_budget`) so the caller can
    slice at any budget, including unbounded."""
    bundles = np.array([b for _, b in candidates], dtype=np.int64)
    thems = np.array([t for t, _ in candidates], dtype=np.intp)
    v_me_arr = np.asarray(vectors[me], dtype=np.float64)
    vectors_arr = np.asarray(vectors, dtype=np.float64)

    bundles_f = bundles.astype(np.float64)
    mine = bundles_f @ v_me_arr
    theirs = -(bundles_f * vectors_arr[thems]).sum(axis=1)
    mask = (mine > 0.0) & (theirs > 0.0)
    seen = int(np.count_nonzero(mask))
    if seen == 0:
        return [], 0

    idx = np.flatnonzero(mask)
    min_surplus = np.minimum(mine[idx], theirs[idx])
    total = mine[idx] + theirs[idx]
    weights = _BUNDLE_BASE ** np.arange(NUM_RESOURCES - 1, -1, -1, dtype=np.int64)
    codes = (bundles[idx] + _BUNDLE_OFFSET) @ weights
    counterparties = thems[idx]

    # `lexsort`'s last key is primary; keys wanting "highest first" are
    # negated (`lexsort` is ascending); the canonical-order/lower-seat
    # determinism keys want "lowest first" and are used as-is.
    if order == "minimal_bundle":
        total_cards = np.abs(bundles[idx]).sum(axis=1)
        sort_idx = np.lexsort((counterparties, codes, total_cards, -min_surplus))
    else:
        sort_idx = np.lexsort((counterparties, codes, -total, -mine[idx], -min_surplus))
    ranked_idx = idx[sort_idx]
    ranked = [(candidates[i][1], int(thems[i])) for i in ranked_idx]
    return ranked, seen


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
