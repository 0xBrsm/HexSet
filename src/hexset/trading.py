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
* Once a turn, after the roll and the robber and before any build is served,
  the engine clears deals for the current player -- best deal first
  (maximise the smaller public surplus; the canonical trade index breaks
  ties) -- until nothing clears. There is no budget: the gate is
  re-evaluated after every executed trade and must be *strictly* positive,
  so the acting seat's own valuation strictly increases at each step, the
  state space is finite, and no cycle is possible. `Game.max_trades` is an
  off switch (`0`), not a budget; `None` is the unbounded default.

There are no trade actions -- no propose, respond, accept or decline -- so
nothing here reads an opponent's hand on an actor's behalf and the action
space carries no trade slot to mask. The engine is the referee: it checks
coverage (`holds`) itself, which is a rules check, not information handed to
a player.

Candidate bundles are the one-for-one exchanges: one card out, one card
back. That is deliberately the whole language. Because the gate is
re-evaluated after every trade and the loop does not stop until nothing
clears, any multi-card exchange both sides still want is reached as a
sequence of single-card steps, and the canonical trade index --
`given * NUM_RESOURCES + wanted`, the strict ordering the trading design
names -- is already a total order over exactly these, which is what the
tie-break is defined on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, NamedTuple, Sequence

from .board.terrain import NUM_RESOURCES
from .state import GameState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .game import Game
    from .view import View

Bundle = tuple[int, ...]

# What a seat publishes, and how it judges a concrete exchange. `view` is
# always that seat's own `Game.state(seat)` information set, never the true
# state and never another seat's.
Valuation = Callable[[int, "View"], Sequence[float]]
Gate = Callable[[int, "View", Bundle, int], bool]


class Trade(NamedTuple):
    """One executed exchange. `received` is signed, positive towards `a`."""

    a: int
    b: int
    received: Bundle


# What a seat that has published nothing advertises: no wants, no gives, so
# no bundle it is party to can ever have positive public surplus.
NO_VALUATION: tuple[float, ...] = (0.0,) * NUM_RESOURCES


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


def _candidates(state: GameState, me: int, locked: frozenset[int]) -> list[tuple[int, int, int]]:
    """`(counterparty, given, wanted)` for every coverable one-for-one swap.

    In canonical trade index order (`given * NUM_RESOURCES + wanted`),
    counterparty ascending within a pair, so the tie-break below is a stable
    property of the position rather than of iteration order.
    """
    hand = state.hands[me]
    out = []
    for given in range(NUM_RESOURCES):
        if not hand[given]:
            continue
        for wanted in range(NUM_RESOURCES):
            if wanted == given:
                continue
            for them in range(state.num_players):
                if them == me or them in locked:
                    continue
                if state.hands[them][wanted]:
                    out.append((them, given, wanted))
    return out


def trade_event(game: "Game", valuation_of: Valuation, gate: Gate) -> list[Trade]:
    """Clear every deal the current player and one other seat both want.

    Called once a turn by the engine, on the transition into
    `Phase.MAIN` -- so every driver (the arena's loop, the gym, the server,
    a search stepping its own copy) gets it without having to remember to.
    Returns the trades executed, in the order they cleared, and appends them
    to `game.trades`.

    `valuation_of(seat, view)` publishes each seat's public vector, read once
    at the start of the event and fixed for its duration: the vectors were
    published at the last decision, and only the private gates move as cards
    change hands, which is what makes the second ore worth less than the
    first. `gate(seat, view, bundle, counterparty)` is that seat's private
    judgement of one concrete exchange; both sides must return True.

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
    players = state.num_players
    views: dict[int, "View"] = {}

    def view(seat: int) -> "View":
        got = views.get(seat)
        if got is None:
            got = game.state(seat)
            views[seat] = got
        return got

    game.valuations = [
        _checked(valuation_of(seat, view(seat)), seat) for seat in range(players)
    ]
    vectors = game.valuations

    # The assertion's ceiling, measured before anything moves: a one-for-one
    # exchange conserves the number of cards held, so this is fixed for the
    # whole event.
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


def _best_clearing(
    game: "Game", me: int, vectors: Sequence[Sequence[float]], gate: Gate, view
) -> tuple[int, Bundle] | None:
    """The clearing deal with the largest smaller-public-surplus, or None.

    Candidates are ranked by public surplus alone (cheap) and the gates --
    the expensive half, a position evaluation each -- are asked in that
    order, so the first candidate both gates accept *is* the maximiser over
    the clearing set. Ties fall to the canonical trade index, then to the
    lower counterparty seat, both of which `_candidates` already orders by.
    """
    state = game._state
    v_me = vectors[me]
    ranked: list[tuple[float, int, int, Bundle]] = []
    for rank, (them, given, wanted) in enumerate(
        _candidates(state, me, game.locked)
    ):
        mine = v_me[wanted] - v_me[given]
        if mine <= 0.0:
            continue
        v_them = vectors[them]
        theirs = v_them[given] - v_them[wanted]
        if theirs <= 0.0:
            continue
        ranked.append((min(mine, theirs), rank, them, one_for_one(given, wanted)))

    # `-surplus` first, then the canonical order `rank` already carries, so
    # the sort is total and deterministic without comparing bundles.
    ranked.sort(key=lambda row: (-row[0], row[1]))
    for _, _, them, received in ranked:
        mirror = tuple(-n for n in received)
        if gate(me, view(me), received, them) and gate(them, view(them), mirror, me):
            return them, received
    return None


def _checked(vector: Sequence[float], seat: int) -> tuple[float, ...]:
    """A published vector, validated: `NUM_RESOURCES` floats in `[-1, 1]`."""
    out = tuple(float(x) for x in vector)
    if len(out) != NUM_RESOURCES:
        raise ValueError(
            f"seat {seat} published {len(out)} valuations, expected {NUM_RESOURCES}"
        )
    if any(x < -1.0 or x > 1.0 for x in out):
        raise ValueError(f"seat {seat} published a valuation outside [-1, 1]: {out}")
    return out
