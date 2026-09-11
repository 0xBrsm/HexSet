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
  `View`. A deal *clears* only when each side's gain exceeds that side's own
  `trade_floor` (τ) -- no seat's own gate can be forced into a trade it
  prices at or below its floor. The floor is the gate's, not the table's:
  every gate declares its own measured resolution (`trade_floor_of`), and
  there is no engine default.
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
  clearing continues until nothing clears or a position comes back. There
  is no budget: a gate that is a strict function of the position cannot
  revisit one (the acting seat's own gain exceeds its floor at every step,
  and every floor is non-negative), and a gate that is not -- one that
  scores candidates in a world sampled from its belief, as the network gate
  and heximax do -- ends the event at the first revisit rather than cycling.
  `Game.max_trades` is an off switch (`0`), not a budget; `None` is the
  unbounded default.

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

## The trade round (the default protocol)

Everything above -- `trade_event`/`_best_clearing`, fired once a turn from
`hexset.game.enter_main`/`move_robber_to` -- is unchanged, and is the
engine's own automatic clearing. It is **no longer what anything plays under
by default**: `Game.trade_mechanism` chooses, and defaults to `"round"`.
`"clearing"` opts back in.

Clearing is not a model of bargaining; it is a strong approximation standing
in for one. It enumerates every candidate deal and keeps clearing until
nothing clears, against a counterparty that never holds out, never asks for
more and never refuses a deal it merely dislikes. No table plays that way. A
policy fitted or trained against it learns to exploit an exhaustive,
perfectly agreeable opponent, and that edge does not survive contact with a
real one -- which is why the arena, the bench, `record_game` and the gym now
run rounds, and why self-play acceptance labels are drawn from them
(`agents/reference/trading-final.md`, "Training").

Results recorded before this switched -- the fitted presets, the adaptive
slider, the trading-condition screens -- are clearing-mechanism results.
They are not wrong, they are about the other mechanism, and `docs/research.md`
already requires a historical study to run from its recorded source
revision. `"clearing"` stays reachable so they stay reproducible, not
because it is the better default.

`trade_round` (below) is propose-and-respond: the actor broadcasts one
offer, every other seated gate answers once, and the actor picks. Nothing
here counts rounds or caps them -- the floor and the card cap already bound
what one round can move -- so the *caller* decides how many a turn is worth,
and there are two callers.

An unserved game is driven by `hexset.game.run_trade_event`, once a turn
from `enter_main`/`move_robber_to`, under `Game.trade_rounds`: `1` by
default, `0` for no trading at all, `-1` for as many as keep clearing.

A *served* game (`hexset.server`) drives its own instead, because its seats
answer through a wire rather than a synchronous call -- a round there spans
many requests while a person or an LLM thinks. Such a session seats both
switches off, `game.trade_rounds = 0` and `game.max_trades = 0`
(`api.build_session`), so the engine drives neither mechanism underneath the
one the session is already running, and calls `trade_round(game, gates)`
itself as many times a turn as the acting seat wants.

One round: the current player's gate broadcasts one offer (`Bot.offer`,
new); every other seated gate answers once (`Bot.respond`, new) --
accept, counter, or pass; the actor's gate picks one answer to execute
(`Bot.pick`, new) or declines them all. A gate that only has
`gains_many` gets a sensible default for all three (`default_offer`/
`default_respond`/`default_pick`), the same "structural, not by
inheritance" convention `valued`/`valued_many` already give `accepts`/
`accepts_many`; a gate with a real opponent model instead implements
`estimate_many(view, candidates) -> list[float]` (heximax), read
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

# A trade moves at most this many cards on either side -- a table rule, not
# a knob: the human corpus puts 99.1% of recorded trades at or under three
# cards a side (`agents/reference/trading-final.md`). Read in one place,
# `_hand_multisets`, which is what `_candidates` -- and so `trade_event`,
# the server's `trade/acceptable` preview and pending offers -- all draw
# from; `execute_trade` enforces the same limit again on a bundle composed
# outside that enumeration.
MAX_TRADE_CARDS = 3


def trade_floor_of(gate: object) -> float:
    """The clearing floor τ this gate's own gains are held to: its
    `trade_floor`, the gate's measured resolution under paired chance
    (the trade lab's phase 3, `agents/reference/trading-final.md` item 4 and
    its 2026-09-05 amendment). A gain below a gate's own resolution is a
    claim no outcome can verify, and the table does not honour it.

    The floor is a property of the gate, not of the table: every seat's
    gate declares its own -- heximax from its own measurement, a checkpoint
    from its file's metadata, `0.0` for a gate whose resolution has not been
    measured -- and there is no engine default. A gate that prices a
    candidate positive without declaring one is refused loudly rather than
    judged by a number that was measured on some other gate.
    """
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
    """Whether `gate`'s private gain clears `gate`'s own floor -- the one
    predicate `trade_event` (both the acting seat's and each counterparty's
    subset), `execute_agreed`, and the round's defaults all read, so every
    admission point applies the same rule. A gain at or below zero never
    clears -- every floor is non-negative -- so a gate that only ever
    declines (a manual seat's `PendingGate`, a bot with no trading surface)
    is never asked for a floor it has no use for."""
    if gain <= 0.0:
        return False
    return gain > trade_floor_of(gate)


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
    that satisfies `hexset.bots.base.Bot` only structurally: a bot from
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
    floor with it. Among the candidates both sides clear their own floor on,
    `game.trade_rule` picks the winner (`_best_clearing`). No budget: the
    loop runs until nothing clears.

    The single engine limit is the revisit check below: an event ends the
    moment a position (every seat's hand plus the public ledger) comes
    back. For a gate that is a strict function of the position that never
    happens -- the acting seat's own gain exceeds its floor at every
    clearing, strictly positive since every floor is non-negative -- and it
    used to be an assertion for that reason. It is a termination now,
    because the gates that matter are not strict functions of the position:
    the network gate scores each candidate in a world drawn from its belief
    (`hexset.clients.netbot`), heximax samples worlds too, and a trade that
    changes the ledger changes the next draw, so a reverse exchange can
    price positive at the new position without anything being broken. The
    check is deliberately not a count of trades: a legitimate event of one-
    and two-card exchanges can run longer than there are cards on the table
    without ever repeating a position.
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


def _validate_exchange(game: "Game", actor: int, counterparty: int, received: Bundle) -> None:
    """The referee's checks on an exchange between `actor` and `counterparty`
    (`received` signed towards `actor`), before any gate is asked: the two
    seats differ, the phase is `Phase.MAIN`, one of the two is the current
    player (a seat trades on its own turn with anyone, or on another seat's
    turn with that seat only), neither side moves more than
    `MAX_TRADE_CARDS`, and both sides can cover their half from the true
    hands (`holds`). Raises `ValueError` naming the first check that fails.
    """
    from .game import Phase  # local: avoids a game/trading import cycle

    if actor == counterparty:
        raise ValueError("a seat cannot trade with itself")
    if game.phase is not Phase.MAIN:
        raise ValueError(f"trading is only open in {Phase.MAIN.name}, not {game.phase.name}")
    if game.current_player not in (actor, counterparty):
        raise ValueError(f"neither seat {actor} nor {counterparty} is the current player")
    state = game._state
    give = [max(0, -n) for n in received]
    take = [max(0, n) for n in received]
    if sum(give) > MAX_TRADE_CARDS or sum(take) > MAX_TRADE_CARDS:
        raise ValueError(f"a trade moves at most {MAX_TRADE_CARDS} cards a side")
    if not holds(state, actor, give):
        raise ValueError(f"seat {actor} cannot cover its side of this trade")
    if not holds(state, counterparty, take):
        raise ValueError(f"seat {counterparty} cannot cover its side of this trade")


def execute_agreed(
    game: "Game",
    actor: int,
    counterparty: int,
    received: Bundle,
    *,
    ask_actor: bool,
    ask_counterparty: bool,
) -> Trade:
    """Execute one exchange between `actor` and `counterparty` (`received`
    signed towards `actor`) that was agreed outside the clearing house --
    a trade round's chosen response, or a bundle a seat composed by hand.

    `_validate_exchange` runs first. Then each side whose flag is set has
    its own gate (`game.gates`) asked fresh, on its own view, and the trade
    is refused (`ValueError`) unless that gain clears that gate's own floor. A
    side whose flag is clear is a seat whose consent is the submission
    itself -- a person or an LLM that composed, accepted or countered
    through the server -- and its gate (a `PendingGate`, which never
    clears) is not asked. On success the cards move (`exchange`), the
    ledger certifies the diff, and the `Trade` is appended to `game.trades`
    -- the same two calls `trade_event` makes for an automatic clearing.
    """
    _validate_exchange(game, actor, counterparty, received)
    gates = game.gates
    gain_a = 0.0
    gain_b = 0.0
    if ask_actor:
        trader = gates[actor] if gates is not None else None
        gain_a = valued(trader, game.state(actor), received, counterparty)
        if not clears_floor(gain_a, trader):
            raise ValueError(f"seat {actor} does not want this exchange")
    if ask_counterparty:
        trader = gates[counterparty] if gates is not None else None
        mirror = tuple(-n for n in received)
        gain_b = valued(trader, game.state(counterparty), mirror, actor)
        if not clears_floor(gain_b, trader):
            raise ValueError(f"seat {counterparty} does not want this exchange")

    state = game._state
    before = [hand[:] for hand in state.hands]
    exchange(state, actor, counterparty, received)
    game.ledger.apply_hand_diff(before, state.hands)
    trade = Trade(actor, counterparty, received, gain_a=gain_a, gain_b=gain_b)
    game.trades.append(trade)
    game.trades_made += 1
    return trade


def execute_trade(game: "Game", proposer: int, counterparty: int, received: Bundle) -> Trade:
    """A bundle `proposer` composed by hand and put to a bot `counterparty`:
    `execute_agreed` with only the counterparty's gate asked -- submitting
    the bundle is the proposer's own consent, so a seat may propose an
    exchange its own gate would refuse. Raises `ValueError` naming the
    first check that fails (`_validate_exchange`, then the counterparty's
    gain against its own floor).
    """
    return execute_agreed(
        game, proposer, counterparty, received, ask_actor=False, ask_counterparty=True
    )


def _best_clearing(
    game: "Game",
    me: int,
    gate: Gate,
    view,
) -> tuple[int, Bundle, float, float] | None:
    """The clearing deal `game.trade_rule` ranks highest, or `None`.

    Every coverable candidate is asked about, in two batches: `me`'s gate is
    asked once, over every candidate, via `valued_many`; then, only for the
    candidates that clear `me`'s own floor (`clears_floor`), each
    distinct counterparty's gate is asked once, over its own accepted
    subset. There is no cheap pre-filter left to rank candidates before a
    gate is asked -- the mechanic's one approximation is the gate itself
    (`agents/reference/trading-theory.md` §5) -- so every enumerated
    candidate costs one row in the acting seat's one batched call.

    Among the candidates that clear each side's own floor, the winner is
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

    def gate_of(seat: int) -> object:
        # Whose floor a gain is held to: the seated trader's, or -- for a
        # direct caller with a bare `gate` callable and no `game.gates` --
        # the callable's own `trade_floor`.
        return traders[seat] if traders is not None else gate

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
        # Key 2 is always the actor's own gain, whatever the primary rule;
        # key 3 -- the negated bundle -- sorts so `max` picks the smallest
        # original bundle on a tie; key 4 -- the negated counterparty seat --
        # so `max` picks the lower one. Determinism only.
        canonical = tuple(-n for n in receiveds[i])
        return (primary, gain_me, canonical, -thems[i])

    cleared = [i for i, gain in theirs.items() if clears_floor(gain, gate_of(thems[i]))]
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
    actor's terms. One convention for both kinds is what lets `pick`/
    `default_pick` price every non-`"pass"` response the same way,
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
    the gate provides one (Heximax uses this seat's belief), else this gate's own gain on every
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
    (`valued_many`) among those that clear the actor's own floor on *both*
    readings -- its own gain and the *estimated* counterparty gain
    (`_estimate_many`, `clears_floor`; the estimate is in the actor's own
    units, so the actor's floor is the one that applies) -- so a gate with a real opponent
    model offers what it believes the table will actually take, and a plain
    gate falls back to "what is best for itself" (its own gain stands in
    for the estimate too). `None` when nothing clears both -- this seat
    passes rather than broadcasts. Ties break on the actor's own gain
    again, then canonical bundle order, then the lower counterparty seat --
    the same purely-deterministic tie-break `_best_clearing` uses, for the
    same reason (real-valued gains essentially never tie in practice).

    The own-gain floor is what makes an offer a promise. Without it the
    actor broadcast whatever the *table* liked best among what it could
    cover -- routinely a deal it would then refuse itself, since `pick`
    (`default_pick`) and the engine's re-ask at execution
    (`execute_agreed`) both apply the actor's own floor. Measured on
    heximax, 2026-09-08: every one of its broadcasts priced negative for
    itself, so an accepted offer never completed. The same rule already
    governed a counter (`default_respond`: "never counter with a deal it
    would then refuse"); an offer is held to it too.
    """
    if not candidates:
        return None
    receiveds = [b for _, b in candidates]
    thems = [c for c, _ in candidates]
    own_gains = valued_many(gate, view, receiveds, thems)
    estimates = _estimate_many(gate, view, candidates)
    eligible = [
        i for i in range(len(candidates))
        if clears_floor(estimates[i], gate) and clears_floor(own_gains[i], gate)
    ]
    if not eligible:
        return None

    def key(i: int) -> tuple:
        canonical = tuple(-n for n in receiveds[i])
        return (own_gains[i], canonical, -thems[i])

    return max(eligible, key=key)


def default_respond(gate: object, view: "View", offer: Offer) -> Response:
    """The default `respond(view, offer) -> Response` for a gate that only
    has `gains_many`: accept outright when this seat's own gain on the
    offered exchange clears its own floor; else counter with the bundle
    this seat can actually propose (`_belief_candidates`: its own hand
    exactly, the actor's hand from the ledger's known lower bound)
    maximising this seat's own gain among those whose *estimated* actor
    gain clears the floor (`_estimate_many`); else pass. Ties among
    counters break the same way `default_offer`'s do.

    A counter has to clear the floor on *this* seat's own gain as well as
    on the actor's estimated one. Own gain used to rank the candidates
    without admitting them, so a gate could counter with an exchange it
    would then refuse: `execute_agreed` asks the responder's gate again
    when the actor takes the deal, and that fresh ask applies the same
    floor. The seat that answered "counter" was the one saying no, in an
    error the actor could do nothing about.
    """
    me = view.perspective
    actor = offer.actor
    received_for_me = tuple(-n for n in offer.received)
    # An accept this seat cannot cover would only fail later, at the
    # actor's `execute_round_choice`; `view.known[me]` is exact for the
    # perspective's own hand, so the check costs nothing in information.
    covers = all(view.known[me][r] >= n for r, n in enumerate(offer.received) if n > 0)
    gain = valued(gate, view, received_for_me, actor)
    if covers and clears_floor(gain, gate):
        return Response(me, RESPONSE_ACCEPT, offer.received)

    candidates = _belief_candidates(view, me, actor)
    if candidates:
        counterparties = [actor] * len(candidates)
        own_gains = valued_many(gate, view, candidates, counterparties)
        estimates = _estimate_many(gate, view, list(zip(counterparties, candidates)))
        eligible = [
            i
            for i in range(len(candidates))
            if clears_floor(estimates[i], gate) and clears_floor(own_gains[i], gate)
        ]
        if eligible:

            def key(i: int) -> tuple:
                canonical = tuple(-n for n in candidates[i])
                return (own_gains[i], canonical)

            best = max(eligible, key=key)
            return Response(me, RESPONSE_COUNTER, tuple(-n for n in candidates[best]))

    return Response(me, RESPONSE_PASS)


def default_pick(
    gate: object, view: "View", responses: Sequence[Response]
) -> int | None:
    """The default `pick(view, responses) -> index | None` for a gate
    that only has `gains_many`: the acceptance or counter with the highest
    own gain above this gate's own floor, else `None`. Every non-`"pass"`
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
        if not clears_floor(gain, gate):
            continue
        canonical = tuple(-n for n in r.bundle)
        key = (gain, canonical, -r.seat)
        if best_key is None or key > best_key:
            best_key, best_idx = key, i
    return best_idx


def choose_and_execute(
    game: "Game", gates: Sequence[object], actor: int, responses: Sequence[Response]
) -> Trade | None:
    """The choice-and-execution half of a round (`trade_round`'s own tail,
    factored out here so a caller that assembled its own `responses` --
    across more than one call, not just one synchronous broadcast -- can
    resolve a round the same way: a served table's session, which keeps a
    round open across a manual seat's late answer (`hexset.server.webplay.
    GameSession`, `agents/reference/trading-final.md`'s "the trade round").

    `actor`'s own gate picks one response (`pick`/`default_pick`, the
    same `getattr`-first, default-second dispatch every round method uses),
    then the engine re-checks it fresh at the moment cards would actually
    move (`_execute_round`), trusting neither side's report. `None` --
    executing nothing -- wherever `trade_round` itself would return `[]`:
    no responses, nothing clears `actor`'s own gate, or the chosen exchange
    fails `_execute_round`'s re-check.

    A manual (human/LLM) `actor`'s own gate is a `PendingGate`, whose
    `pick` always declines (see its own docstring) -- so calling this
    after every new response a round collects is always safe, whoever the
    actor is: a bot may resolve here, a human never does, and picks
    instead through its own explicit call (`GameSession.execute_round_choice`,
    which goes through `execute_trade`'s consent-based checks instead of
    this function's fresh-both-sides re-check).
    """
    if not responses:
        return None
    actor_gate = gates[actor]
    if actor_gate is None:
        return None
    actor_view = game.state(actor)
    choose_fn = getattr(actor_gate, "pick", None)
    chosen = (
        choose_fn(actor_view, responses)
        if choose_fn is not None
        else default_pick(actor_gate, actor_view, responses)
    )
    if chosen is None or not (0 <= chosen < len(responses)):
        return None
    response = responses[chosen]
    if response.kind == RESPONSE_PASS or response.bundle is None:
        return None
    return _execute_round(game, gates, actor, response.seat, response.bundle)


def _execute_round(
    game: "Game", gates: Sequence[object], actor: int, counterparty: int, received: Bundle
) -> Trade | None:
    """Execute one round's chosen exchange between two bot gates: both
    sides re-asked fresh at the moment cards actually move
    (`execute_agreed` with both flags set), trusting neither the offer nor
    the response it was built from -- a stale estimate, a gate that changed
    its mind, an adversarial implementation. `None`, executing nothing, on
    the first check that fails: a round coming to nothing is an ordinary
    outcome, not a caller error the way a hand-composed bundle's failure
    is. `gates` is the caller's seat -> gate mapping, installed on the game
    for the duration of the call.
    """
    had = game.gates
    game.gates = tuple(gates)
    try:
        return execute_agreed(
            game, actor, counterparty, received, ask_actor=True, ask_counterparty=True
        )
    except ValueError:
        return None
    finally:
        game.gates = had


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

    Every method asked of a gate -- `offer`, `respond`, `pick` -- is read
    with `getattr`, falling back to this module's own `default_offer`/
    `default_respond`/`default_pick` for a gate that only has
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

    trade = choose_and_execute(game, gates, me, responses)
    return [] if trade is None else [trade]
